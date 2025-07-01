import datasets
import torch
import json
from torch.utils.data import DataLoader, Dataset
from utils import get_local_dir, TemporarilySeededRandom
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
import tqdm
import random
from bs4 import BeautifulSoup, NavigableString
import numpy as np
from typing import Dict, List, Optional, Iterator, Callable, Union, Tuple
import json


def extract_anthropic_prompt(prompt_and_response):
    """Extract the anthropic prompt from a prompt and response pair."""
    search_term = '\n\nAssistant:'
    search_term_idx = prompt_and_response.rfind(search_term)
    assert search_term_idx != -1, f"Prompt and response does not contain '{search_term}'"
    return prompt_and_response[:search_term_idx + len(search_term)]


def strip_html_tags(html_string):
    """Strip HTML tags from a string, except for <code> tags (which contain real code in the StackExchange answers)."""
    # Create a BeautifulSoup object
    soup = BeautifulSoup(html_string, 'html.parser')

    # Initialize an empty list to store the text
    text = []
    for element in soup.children:
        if isinstance(element, NavigableString):
            continue
        if element.name == 'p':
            text.append(''.join(child.string for child in element.children if isinstance(child, NavigableString)))
        elif element.name == 'pre':
            for code in element.find_all('code'):
                text.append("<code>" + code.get_text() + "</code>")
        elif element.name == 'code':
            text.append("<code>" + element.get_text() + "</code>")

    # Join the text together with newlines in between
    text = "\n\n".join(text)

    return text


def get_se(split, silent=False, cache_dir: str = None) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    """Load the StackExchange dataset from Huggingface, and return a dict of prompts and responses. See get_hh for the format.
    
       We strip the HTML tags from the responses (except for <code> tags), and we add necessary newlines.
    """
    print(f'Loading SE dataset ({split} split) from Huggingface...')
    dataset = datasets.load_dataset('HuggingFaceH4/stack-exchange-preferences', cache_dir=cache_dir)['train']
    print('done')

    # shuffle the dataset and select 1% for test
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.select(range(int(len(dataset) * 0.01))) if split == 'test' else dataset.select(
        range(int(len(dataset) * 0.01), len(dataset)))

    def strip_html(x):
        x['question'] = strip_html_tags(x['question'])
        for a in x['answers']:
            a['text'] = strip_html_tags(a['text'])
        return x

    dataset = dataset.map(strip_html, num_proc=64)

    data = defaultdict(dict)
    for row in tqdm.tqdm(dataset, desc='Processing SE', disable=silent):
        prompt = '\n\nHuman: ' + row['question'] + '\n\nAssistant:'
        responses = [' ' + a['text'] for a in row['answers']]
        scores = [a['pm_score'] for a in row['answers']]

        pairs = []
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                pairs.append((i, j) if scores[i] > scores[j] else (j, i))

        data[prompt]['responses'] = responses
        data[prompt]['pairs'] = pairs
        data[prompt]['sft_target'] = max(responses, key=lambda x: scores[responses.index(x)])

    return data

def get_shp(split: str, silent: bool = False, cache_dir: str = None) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    """Load the Stanford Human Preferences dataset from Huggingface and convert it to the necessary format. See hh for the format.

       We filter preference pairs to only keep pairs where the score ratio is at least 2.
       For this dataset, the sft_target is the response with the highest score.
    """
    print(f'Loading SHP dataset ({split} split) from Huggingface...')
    dataset = datasets.load_dataset('stanfordnlp/SHP', split=split, cache_dir=cache_dir)
    print('done')

    data = defaultdict(lambda: defaultdict(list))
    for row in tqdm.tqdm(dataset, desc='Processing SHP', disable=silent):
        prompt = '\n\nHuman: ' + row['history'] + '\n\nAssistant:'
        responses = [' ' + row['human_ref_A'], ' ' + row['human_ref_B']]
        scores = [row['score_A'], row['score_B']]
        if prompt in data:
            n_responses = len(data[prompt]['responses'])
        else:
            n_responses = 0
        score_ratio = max(scores[0] / scores[1], scores[1] / scores[0])
        if score_ratio < 2:
            continue

        # according to https://huggingface.co/datasets/stanfordnlp/SHP
        data[prompt]['pairs'].append((n_responses, n_responses + 1) if row['labels'] == 1 else (n_responses + 1, n_responses))
        data[prompt]['responses'].extend(responses)
        data[prompt]['scores'].extend(scores)

    for prompt in data:
        data[prompt]['sft_target'] = max(data[prompt]['responses'], key=lambda x: data[prompt]['scores'][data[prompt]['responses'].index(x)])
        del data[prompt]['scores']

    return data


def get_hh(split: str, silent: bool = False, cache_dir: str = None) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    """Load the Anthropic Helpful-Harmless dataset from Huggingface and convert it to the necessary format.
    
       The dataset is converted to a dictionary with the following structure:
       {
           'prompt1': {
               'responses': List[str],
               'pairs': List[Tuple[int, int]],
               'sft_target': str
           },
           'prompt2': {
               ...
           },
       }

       Prompts should be structured as follows:
         \n\nHuman: <prompt>\n\nAssistant:
       Multiple turns are allowed, but the prompt should always start with \n\nHuman: and end with \n\nAssistant:.
       
       For this dataset, the sft_target is just the chosen response.
    """
    print(f'Loading HH dataset ({split} split) from Huggingface...')
    dataset = datasets.load_dataset('Anthropic/hh-rlhf', split=split, cache_dir=cache_dir)
    print('done')

    def split_prompt_and_responses(ex):
        prompt = extract_anthropic_prompt(ex['chosen'])
        chosen_response = ex['chosen'][len(prompt):]
        rejected_response = ex['rejected'][len(prompt):]
        return prompt, chosen_response, rejected_response

    data = defaultdict(lambda: defaultdict(list))
    for row in tqdm.tqdm(dataset, desc='Processing HH', disable=silent):
        prompt, chosen, rejected = split_prompt_and_responses(row)
        responses = [chosen, rejected]
        n_responses = len(data[prompt]['responses'])
        data[prompt]['pairs'].append((n_responses, n_responses + 1))
        data[prompt]['responses'].extend(responses)
        data[prompt]['sft_target'] = chosen

    return data

def get_custom_hh(split: str, silent: bool = False, cache_dir: str = None, file_path: str = None) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    """Load the Anthropic Helpful-Harmless dataset from a JSONL file and convert it to the necessary format."""
    
    print(f'Loading dataset ({split} split) from {file_path}')
    
    data = defaultdict(lambda: defaultdict(list))

    def split_prompt_and_responses(ex):
        prompt = extract_anthropic_prompt(ex['chosen'])
        chosen_response = ex['chosen'][len(prompt):]
        rejected_response = ex['rejected'][len(prompt):]
        return prompt, chosen_response, rejected_response
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            example = json.loads(line.strip())
            prompt, chosen, rejected = split_prompt_and_responses(example)
            
            responses = [chosen, rejected]
            n_responses = len(data[prompt]['responses'])
            data[prompt]['pairs'].append((n_responses, n_responses + 1))
            data[prompt]['responses'].extend(responses)
            data[prompt]['sft_target'] = chosen
    
    print(f'Dataset {file_path} loading done.')
    return data

def get_helpsteer(split: str, silent: bool = False, cache_dir: str = None, file_path: str = None)-> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    """Load the HelpSteer dataset (processed format) from a JSONL file and convert it to the necessary format."""
    print(f'Loading HelpSteer dataset ({split} split) from {file_path}')

    data = defaultdict(lambda: defaultdict(list))

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            example = json.loads(line.strip())
            prompt = example["prompt"]
            chosen = example["chosen"]
            rejected = example["rejected"]
            
            # HelpSteer format: chosen/rejected already contain full prompt+response
            # Extract only the response part by removing the prompt
            chosen_response = chosen[len(prompt):].strip() if chosen.startswith(prompt) else chosen
            rejected_response = rejected[len(prompt):].strip() if rejected.startswith(prompt) else rejected
            
            responses = [chosen_response, rejected_response]
            n_responses = len(data[prompt]['responses'])
            data[prompt]['pairs'].append((n_responses, n_responses + 1))
            data[prompt]['responses'].extend(responses)
            data[prompt]['sft_target'] = chosen_response
    
    print(f'HelpSteer dataset {file_path} loading done.')
    return data

def get_ultrafeedback_binarized(split: str, silent: bool = False, cache_dir: str = None, file_path: str = None)-> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    print(f'Loading dataset ({split} split) from {file_path}')

    data = defaultdict(lambda: defaultdict(list))

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            example = json.loads(line.strip())
            prompt = example["prompt"]
            chosen = example["chosen"]
            rejected = example["rejected"]

            responses = [chosen, rejected]
            n_responses = len(data[prompt]['responses'])
            data[prompt]['pairs'].append((n_responses, n_responses + 1))
            data[prompt]['responses'].extend(responses)
            data[prompt]['sft_target'] = chosen
    return data

def get_dataset(name: str, split: str, silent: bool = False, cache_dir: str = None, file_path:str = None, ):
    """Load the given dataset by name. Supported by default are 'shp', 'hh', 'se', 'ufb', 'helpsteer'."""
    if name == 'shp':
        data = get_shp(split, silent=silent, cache_dir=cache_dir)
    elif name == 'hh':
        data = get_hh(split, silent=silent, cache_dir=cache_dir)
    elif name == 'se':
        data = get_se(split, silent=silent, cache_dir=cache_dir)
    elif name =='custom':
        data = get_custom_hh(split, silent=silent, cache_dir=cache_dir, file_path=file_path)
    elif name =='ufb':
        data = get_ultrafeedback_binarized(split, silent=silent, cache_dir=cache_dir, file_path=file_path)
    elif name == 'helpsteer':
        data = get_helpsteer(split, silent=silent, cache_dir=cache_dir, file_path=file_path)
    else:
        raise ValueError(f"Unknown dataset '{name}'")

    assert set(list(data.values())[0].keys()) == {'responses', 'pairs', 'sft_target'}, \
        f"Unexpected keys in dataset: {list(list(data.values())[0].keys())}"

    return data

def get_dataset_with_preference(name: str, split: str, preference_text: str = None, 
                               prompt_embeddings: torch.Tensor = None, silent: bool = False, 
                               cache_dir: str = None, file_path: str = None):
    """Load the given dataset by name and add preference instruction to each prompt.
    Similar to get_dataset but adds preference instruction p to each prompt or uses embeddings.
    
    Args:
        preference_text: Text-based preference instruction (mutually exclusive with prompt_embeddings)
        prompt_embeddings: Pre-trained prompt embeddings tensor (mutually exclusive with preference_text)
    """
    # Validate that exactly one of preference_text or prompt_embeddings is provided
    if preference_text is None and prompt_embeddings is None:
        raise ValueError("Either preference_text or prompt_embeddings must be provided")
    if preference_text is not None and prompt_embeddings is not None:
        raise ValueError("Only one of preference_text or prompt_embeddings should be provided")
    
    use_prompt_embeddings = prompt_embeddings is not None
    
    # Get the original dataset first
    original_data = get_dataset(name, split, silent, cache_dir, file_path)
    
    # Create a new dataset with both original and preference-augmented prompts
    preference_data = {}
    
    for prompt, data in original_data.items():
        if use_prompt_embeddings:
            # For embedding mode, we need to create a special marker for preference prompt
            # The original prompt stays the same for π_base calculation
            # We use a special marker "<PROMPT_EMBEDDING>" to indicate where embeddings should be inserted
            preference_prompt = prompt + " <PROMPT_EMBEDDING>"  # Special marker for embedding insertion
            preference_data[prompt] = {
                'responses': data['responses'],
                'pairs': data['pairs'],
                'sft_target': data['sft_target'],
                'preference_prompt': preference_prompt,  # Contains special marker
                'prompt_embeddings': prompt_embeddings,  # Store embeddings for later use
                'use_embeddings': True  # Flag to indicate embedding mode
            }
        else:
            # Create preference-augmented prompt (original_prompt + preference_text)
            preference_prompt = prompt + " " + preference_text
            preference_data[prompt] = {
                'responses': data['responses'],
                'pairs': data['pairs'],
                'sft_target': data['sft_target'],
                'preference_prompt': preference_prompt,  # Store the preference-augmented prompt
                'use_embeddings': False  # Flag to indicate text mode
            }
    
    return preference_data

def get_collate_fn(tokenizer) -> Callable[[List[Dict]], Dict[str, Union[List, torch.Tensor]]]:
    """Returns a collate function for the given tokenizer.
    
       The collate function takes a list of examples (dicts, where values are lists of
         ints [tokens] or strings [the original texts]) and returns a batch of examples,
         PyTorch tensors padded to the maximum length. Strings are passed through."""
    def collate_fn(batch):
        # first, pad everything to the same length
        padded_batch = {}
        for k in batch[0].keys():
            if k.endswith('_input_ids') or k.endswith('_attention_mask') or k.endswith('_labels'):
                if 'prompt' in k:  # adapted from https://stackoverflow.com/questions/73256206
                    to_pad = [torch.LongTensor(ex[k][::-1]) for ex in batch]
                else:
                    to_pad = [torch.LongTensor(ex[k]) for ex in batch]
                if k.endswith('_input_ids'):
                    padding_value = tokenizer.pad_token_id
                elif k.endswith('_labels'):
                    padding_value = -100
                elif k.endswith('_attention_mask'):
                    padding_value = 0
                else:
                    raise ValueError(f"Unexpected key in batch '{k}'")

                padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
                if 'prompt' in k:  # for the prompt, flip back so padding is on left side
                    padded_batch[k] = padded_batch[k].flip(dims=[1])
            else:
                padded_batch[k] = [ex[k] for ex in batch]

        return padded_batch
    return collate_fn


def tokenize_batch_element(prompt: str, chosen: str, rejected: str, truncation_mode: str, tokenizer, max_length: int, max_prompt_length: int) -> Dict:
    """Tokenize a single batch element.
    
       At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
         in case the prompt + chosen or prompt + rejected responses is/are too long. First
         we truncate the prompt; if we're still too long, we truncate the chosen/rejected.
       
       We also create the labels for the chosen/rejected responses, which are of length equal to
         the sum of the length of the prompt and the chosen/rejected response, with -100 for the
         prompt tokens.
    """
    chosen_tokens = tokenizer(chosen, add_special_tokens=False)
    rejected_tokens = tokenizer(rejected, add_special_tokens=False)
    prompt_tokens = tokenizer(prompt, add_special_tokens=False)

    assert tokenizer.eos_token_id not in prompt_tokens['input_ids'], f"Prompt contains EOS token: {prompt}"
    assert tokenizer.eos_token_id not in chosen_tokens['input_ids'], f"Chosen response contains EOS token: {chosen}"
    assert tokenizer.eos_token_id not in rejected_tokens['input_ids'], f"Rejected response contains EOS token: {rejected}"

    chosen_tokens['input_ids'].append(tokenizer.eos_token_id)
    chosen_tokens['attention_mask'].append(1)

    rejected_tokens['input_ids'].append(tokenizer.eos_token_id)
    rejected_tokens['attention_mask'].append(1)

    longer_response_length = max(len(chosen_tokens['input_ids']), len(rejected_tokens['input_ids']))

    # if combined sequence is too long, truncate the prompt
    if len(prompt_tokens['input_ids']) + longer_response_length > max_length:
        if truncation_mode == 'keep_start':
            prompt_tokens = {k: v[:max_prompt_length] for k, v in prompt_tokens.items()}
        elif truncation_mode == 'keep_end':
            prompt_tokens = {k: v[-max_prompt_length:] for k, v in prompt_tokens.items()}
        else:
            raise ValueError(f'Unknown truncation mode: {truncation_mode}')

    # if that's still too long, truncate the response
    if len(prompt_tokens['input_ids']) + longer_response_length > max_length:
        chosen_tokens = {k: v[:max_length - max_prompt_length] for k, v in chosen_tokens.items()}
        rejected_tokens = {k: v[:max_length - max_prompt_length] for k, v in rejected_tokens.items()}

    # Create labels
    chosen_sequence_tokens = {k: prompt_tokens[k] + chosen_tokens[k] for k in chosen_tokens}
    rejected_sequence_tokens = {k: prompt_tokens[k] + rejected_tokens[k] for k in rejected_tokens}
    chosen_sequence_tokens['labels'] = chosen_sequence_tokens['input_ids'][:]
    chosen_sequence_tokens['labels'][:len(prompt_tokens['input_ids'])] = [-100] * len(prompt_tokens['input_ids'])
    rejected_sequence_tokens['labels'] = rejected_sequence_tokens['input_ids'][:]
    rejected_sequence_tokens['labels'][:len(prompt_tokens['input_ids'])] = [-100] * len(prompt_tokens['input_ids'])

    batch = {}

    batch['prompt'] = prompt
    batch['chosen'] = prompt + chosen
    batch['rejected'] = prompt + rejected
    batch['chosen_response_only'] = chosen
    batch['rejected_response_only'] = rejected

    for k, toks in {'chosen': chosen_sequence_tokens, 'rejected': rejected_sequence_tokens, 'prompt': prompt_tokens}.items():
        for type_key, tokens in toks.items():
            if type_key == 'token_type_ids':
                continue
            batch[f'{k}_{type_key}'] = tokens

    return batch

def tokenize_batch_element_with_preference(prompt: str, preference_prompt: str, chosen: str, rejected: str, 
                                          truncation_mode: str, tokenizer, max_length: int, max_prompt_length: int,
                                          prompt_embeddings: torch.Tensor = None) -> Dict:
    """Tokenize a batch element for preference-augmented training.
    
    Similar to tokenize_batch_element but handles both the original prompt and the preference-augmented prompt.
    For embedding mode, preference_prompt contains "<PROMPT_EMBEDDING>" marker that will be handled specially.
    """
    # Tokenize original inputs (same as the existing function)
    original_batch = tokenize_batch_element(prompt, chosen, rejected, truncation_mode, tokenizer, max_length, max_prompt_length)
    
    # Handle preference prompt tokenization
    if prompt_embeddings is not None and "<PROMPT_EMBEDDING>" in preference_prompt:
        # For embedding mode: remove the marker and prepare for embedding insertion
        # We'll tokenize the base prompt and mark where embeddings should be inserted
        base_prompt = preference_prompt.replace(" <PROMPT_EMBEDDING>", "")
        preference_prompt_tokens = tokenizer(base_prompt, add_special_tokens=False)
        
        # Add a special flag to indicate this batch element uses embeddings
        has_prompt_embeddings = True
        embedding_insert_position = len(preference_prompt_tokens['input_ids'])  # Insert at the end of prompt
    else:
        # For text mode: normal tokenization
        preference_prompt_tokens = tokenizer(preference_prompt, add_special_tokens=False)
        has_prompt_embeddings = False
        embedding_insert_position = None
    
    # Tokenize response parts
    chosen_tokens = tokenizer(chosen, add_special_tokens=False)
    rejected_tokens = tokenizer(rejected, add_special_tokens=False)
    
    assert tokenizer.eos_token_id not in preference_prompt_tokens['input_ids'], f"Preference prompt contains EOS token: {preference_prompt}"
    
    chosen_tokens['input_ids'].append(tokenizer.eos_token_id)
    chosen_tokens['attention_mask'].append(1)
    
    rejected_tokens['input_ids'].append(tokenizer.eos_token_id)
    rejected_tokens['attention_mask'].append(1)
    
    longer_response_length = max(len(chosen_tokens['input_ids']), len(rejected_tokens['input_ids']))
    
    # Truncate if needed
    if len(preference_prompt_tokens['input_ids']) + longer_response_length > max_length:
        if truncation_mode == 'keep_start':
            preference_prompt_tokens = {k: v[:max_prompt_length] for k, v in preference_prompt_tokens.items()}
        elif truncation_mode == 'keep_end':
            preference_prompt_tokens = {k: v[-max_prompt_length:] for k, v in preference_prompt_tokens.items()}
        else:
            raise ValueError(f'Unknown truncation mode: {truncation_mode}')
        
        # Update embedding insert position if it was truncated
        if has_prompt_embeddings:
            embedding_insert_position = min(embedding_insert_position, len(preference_prompt_tokens['input_ids']))
    
    # Further truncate if still too long
    if len(preference_prompt_tokens['input_ids']) + longer_response_length > max_length:
        chosen_tokens = {k: v[:max_length - max_prompt_length] for k, v in chosen_tokens.items()}
        rejected_tokens = {k: v[:max_length - max_prompt_length] for k, v in rejected_tokens.items()}
    
    # Create labels for preference-augmented inputs
    pref_chosen_sequence_tokens = {k: preference_prompt_tokens[k] + chosen_tokens[k] for k in chosen_tokens}
    pref_rejected_sequence_tokens = {k: preference_prompt_tokens[k] + rejected_tokens[k] for k in rejected_tokens}
    pref_chosen_sequence_tokens['labels'] = pref_chosen_sequence_tokens['input_ids'][:]
    pref_chosen_sequence_tokens['labels'][:len(preference_prompt_tokens['input_ids'])] = [-100] * len(preference_prompt_tokens['input_ids'])
    pref_rejected_sequence_tokens['labels'] = pref_rejected_sequence_tokens['input_ids'][:]
    pref_rejected_sequence_tokens['labels'][:len(preference_prompt_tokens['input_ids'])] = [-100] * len(preference_prompt_tokens['input_ids'])
    
    # Add preference-augmented inputs to batch
    preference_batch = {}
    preference_batch['preference_prompt'] = preference_prompt
    preference_batch['preference_chosen'] = preference_prompt.replace(" <PROMPT_EMBEDDING>", "") + chosen if has_prompt_embeddings else preference_prompt + chosen
    preference_batch['preference_rejected'] = preference_prompt.replace(" <PROMPT_EMBEDDING>", "") + rejected if has_prompt_embeddings else preference_prompt + rejected
    
    for k, toks in {'preference_chosen': pref_chosen_sequence_tokens, 'preference_rejected': pref_rejected_sequence_tokens, 'preference_prompt': preference_prompt_tokens}.items():
        for type_key, tokens in toks.items():
            if type_key == 'token_type_ids':
                continue
            preference_batch[f'{k}_{type_key}'] = tokens
    
    # Add embedding metadata if using embeddings
    if has_prompt_embeddings:
        preference_batch['has_prompt_embeddings'] = True
        preference_batch['prompt_embeddings'] = prompt_embeddings
        preference_batch['embedding_insert_position'] = embedding_insert_position
    else:
        preference_batch['has_prompt_embeddings'] = False
    
    # Merge original batch with preference-augmented batch
    merged_batch = {**original_batch, **preference_batch}
    return merged_batch

def get_batch_iterator(names: List[str],
                       tokenizer,
                       split: str = 'train',
                       batch_size: int = 1,
                       shuffle: bool = True,
                       max_length: int = 512,
                       max_prompt_length: int = 128,
                       sft_mode: bool = False,
                       n_epochs: Optional[int] = None,
                       n_examples: Optional[int] = None,
                       seed:int = 0,
                       silent: bool = False,
                       cache_dir: Optional[str] = None,
                       file_path: str = None) -> Iterator[Dict]:
    """Get an iterator over batches of data. Stops after n_epochs or n_examples, whichever comes first.

    Args:
        names: Names of datasets to use.
        tokenizer: Tokenizer to use.
        split: Which split to use.
        batch_size: Batch size.
        shuffle: Whether to shuffle the data after each epoch.
        max_length: Maximum length of the combined prompt + response.
        max_prompt_length: Maximum length of the prompt.
        sft_mode: Whether to use SFT mode (i.e., return sft_target instead of chosen/rejected). In sft mode, we just return chosen_input_ids, but they contain the sft_target.
        n_epochs: Number of epochs to run for. This or n_examples must be specified.
        n_examples: Number of examples to run for. This or n_epochs must be specified.
        seed: Random seed.
        silent: Whether to silence the progress bar(s).
        cache_dir: Directory to cache the datasets in.
    """
    assert n_epochs is not None or n_examples is not None, "Must specify either n_epochs or n_examples"
    if silent:
        datasets.logging.disable_progress_bar()
        datasets.logging.set_verbosity_error()

    with TemporarilySeededRandom(seed):
        permutation_seeds = iter(np.random.randint(0, 2**32, size=1000000))
        flat_data = []
        for name in names:
            truncation_mode = 'keep_end' if name == 'hh' else 'keep_start'
            for prompt, data in get_dataset(name, split, silent=silent, cache_dir=cache_dir, file_path=file_path).items():
                flat_data.append((prompt, data['responses'], data['pairs'], data['sft_target'], truncation_mode))

    collate_fn = get_collate_fn(tokenizer)

    epoch_idx = 0
    example_idx = 0
    done = False
    while True:
        if n_epochs is not None and epoch_idx >= n_epochs:
            if not silent:
                print(f'Finished generating {n_epochs} epochs on {split} split')
            break
        if shuffle:
            with TemporarilySeededRandom(int(next(permutation_seeds))):
                random.shuffle(flat_data)

        batch = []
        for prompt, responses, pairs, sft_target, truncation_mode in flat_data:
            if done:
                break
            if sft_mode:
                batch_element = tokenize_batch_element(prompt, sft_target, sft_target, truncation_mode, tokenizer, max_length, max_prompt_length)
                batch_element = {k: v for k, v in batch_element.items() if 'rejected' not in k}
                batch.append(batch_element)
                example_idx += 1
                if len(batch) == batch_size:
                    yield collate_fn(batch)
                    if n_examples is not None and example_idx >= n_examples:
                        if not silent:
                            print(f'Finished generating {n_examples} examples on {split} split')
                        done = True

                    batch = []
            else:
                for p in pairs:
                    if done:
                        break
                    batch_element = tokenize_batch_element(prompt, responses[p[0]], responses[p[1]], truncation_mode, tokenizer, max_length, max_prompt_length)
                    batch.append(batch_element)
                    example_idx += 1
                    if len(batch) == batch_size:
                        yield collate_fn(batch)
                        if n_examples is not None and example_idx >= n_examples:
                            if not silent:
                                print(f'FINISHED {n_examples} EXAMPLES on {split} split')
                            done = True
                        batch = []
        if done:
            break

        epoch_idx += 1

def get_batch_iterator_with_preference(names: List[str],
                                      tokenizer,
                                      preference_text: str = None,
                                      prompt_embeddings: torch.Tensor = None,
                                      split: str = 'train',
                                      batch_size: int = 1,
                                      shuffle: bool = True,
                                      max_length: int = 512,
                                      max_prompt_length: int = 128,
                                      sft_mode: bool = False,
                                      n_epochs: Optional[int] = None,
                                      n_examples: Optional[int] = None,
                                      seed: int = 0,
                                      silent: bool = False,
                                      cache_dir: Optional[str] = None,
                                      file_path: str = None) -> Iterator[Dict]:
    """Get an iterator over batches of data with both original and preference-augmented inputs.
    
    Similar to get_batch_iterator but loads data with preference instructions or embeddings.
    
    Args:
        preference_text: Text-based preference instruction (mutually exclusive with prompt_embeddings)
        prompt_embeddings: Pre-trained prompt embeddings tensor (mutually exclusive with preference_text)
    """
    # Validate that exactly one of preference_text or prompt_embeddings is provided
    if preference_text is None and prompt_embeddings is None:
        raise ValueError("Either preference_text or prompt_embeddings must be provided")
    if preference_text is not None and prompt_embeddings is not None:
        raise ValueError("Only one of preference_text or prompt_embeddings should be provided")
    
    use_prompt_embeddings = prompt_embeddings is not None
    
    assert n_epochs is not None or n_examples is not None, "Must specify either n_epochs or n_examples"
    if silent:
        datasets.logging.disable_progress_bar()
        datasets.logging.set_verbosity_error()

    with TemporarilySeededRandom(seed):
        permutation_seeds = iter(np.random.randint(0, 2**32, size=1000000))
        flat_data = []
        for name in names:
            truncation_mode = 'keep_end' if name == 'hh' else 'keep_start'
            if use_prompt_embeddings:
                dataset_data = get_dataset_with_preference(name, split, prompt_embeddings=prompt_embeddings, silent=silent, cache_dir=cache_dir, file_path=file_path)
            else:
                dataset_data = get_dataset_with_preference(name, split, preference_text=preference_text, silent=silent, cache_dir=cache_dir, file_path=file_path)
            
            for prompt, data in dataset_data.items():
                flat_data.append((prompt, data['preference_prompt'], data['responses'], data['pairs'], data['sft_target'], truncation_mode, data.get('prompt_embeddings')))

    collate_fn = get_collate_fn(tokenizer)

    epoch_idx = 0
    example_idx = 0
    done = False
    while True:
        if n_epochs is not None and epoch_idx >= n_epochs:
            if not silent:
                print(f'Finished generating {n_epochs} epochs on {split} split')
            break
        if shuffle:
            with TemporarilySeededRandom(int(next(permutation_seeds))):
                random.shuffle(flat_data)

        batch = []
        for prompt, preference_prompt, responses, pairs, sft_target, truncation_mode, batch_prompt_embeddings in flat_data:
            if done:
                break
            if sft_mode:
                batch_element = tokenize_batch_element_with_preference(prompt, preference_prompt, sft_target, sft_target,
                                                                     truncation_mode, tokenizer, max_length, max_prompt_length,
                                                                     prompt_embeddings=batch_prompt_embeddings)
                batch_element = {k: v for k, v in batch_element.items() if 'rejected' not in k}
                batch.append(batch_element)
                example_idx += 1
                if len(batch) == batch_size:
                    yield collate_fn(batch)
                    if n_examples is not None and example_idx >= n_examples:
                        if not silent:
                            print(f'Finished generating {n_examples} examples on {split} split')
                        done = True

                    batch = []
            else:
                for p in pairs:
                    if done:
                        break
                    batch_element = tokenize_batch_element_with_preference(prompt, preference_prompt, 
                                                                         responses[p[0]], responses[p[1]], 
                                                                         truncation_mode, tokenizer, max_length, max_prompt_length,
                                                                         prompt_embeddings=batch_prompt_embeddings)
                    batch.append(batch_element)
                    example_idx += 1
                    if len(batch) == batch_size:
                        yield collate_fn(batch)
                        if n_examples is not None and example_idx >= n_examples:
                            if not silent:
                                print(f'FINISHED {n_examples} EXAMPLES on {split} split')
                            done = True
                        batch = []
        if done:
            break

        epoch_idx += 1


def strings_match_up_to_spaces(str_a: str, str_b: str) -> bool:
    """Returns True if str_a and str_b match up to spaces, False otherwise."""
    for idx in range(min(len(str_a), len(str_b)) - 2):
        if str_a[idx] != str_b[idx]:
            if str_a[idx] != ' ' and str_b[idx] != ' ':
                return False
            else:
                if str_a[idx] == ' ':
                    str_a = str_a[:idx] + str_a[idx + 1:]
                else:
                    str_b = str_b[:idx] + str_b[idx + 1:]

    return True