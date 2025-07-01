#!/usr/bin/env python3
"""
Memory-Optimized Preference Dataset Processing

This module provides memory-efficient alternatives to the standard preference dataset processing
by avoiding prompt duplication and implementing lazy concatenation during training.

Key optimizations:
1. Store prompt and responses separately to avoid duplication
2. Lazy concatenation during tokenization 
3. Reduced memory footprint for large datasets
4. Compatible with existing training pipeline
"""

import json
import torch
from typing import Dict, List, Union, Iterator, Optional, Callable
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
import numpy as np
from tqdm import tqdm


def tokenize_batch_element_memory_optimized(
    prompt: str, 
    chosen_response: str, 
    rejected_response: str, 
    truncation_mode: str, 
    tokenizer, 
    max_length: int, 
    max_prompt_length: int
) -> Dict:
    """
    Memory-optimized tokenization that stores prompt and responses separately.
    
    This avoids storing the concatenated sequences until they're actually needed,
    reducing memory usage especially for datasets with long prompts.
    """
    # Tokenize each component separately
    prompt_tokens = tokenizer(prompt, add_special_tokens=False)
    chosen_tokens = tokenizer(chosen_response, add_special_tokens=False)
    rejected_tokens = tokenizer(rejected_response, add_special_tokens=False)
    
    # Validate tokens
    assert tokenizer.eos_token_id not in prompt_tokens['input_ids'], f"Prompt contains EOS token: {prompt}"
    assert tokenizer.eos_token_id not in chosen_tokens['input_ids'], f"Chosen response contains EOS token: {chosen_response}"
    assert tokenizer.eos_token_id not in rejected_tokens['input_ids'], f"Rejected response contains EOS token: {rejected_response}"
    
    # Add EOS tokens to responses
    chosen_tokens['input_ids'].append(tokenizer.eos_token_id)
    chosen_tokens['attention_mask'].append(1)
    rejected_tokens['input_ids'].append(tokenizer.eos_token_id)
    rejected_tokens['attention_mask'].append(1)
    
    # Calculate lengths for truncation
    longer_response_length = max(len(chosen_tokens['input_ids']), len(rejected_tokens['input_ids']))
    
    # Apply truncation if needed
    if len(prompt_tokens['input_ids']) + longer_response_length > max_length:
        if truncation_mode == 'keep_start':
            prompt_tokens = {k: v[:max_prompt_length] for k, v in prompt_tokens.items()}
        elif truncation_mode == 'keep_end':
            prompt_tokens = {k: v[-max_prompt_length:] for k, v in prompt_tokens.items()}
        else:
            raise ValueError(f'Unknown truncation mode: {truncation_mode}')
    
    # Further truncate responses if still too long
    if len(prompt_tokens['input_ids']) + longer_response_length > max_length:
        max_response_length = max_length - len(prompt_tokens['input_ids'])
        chosen_tokens = {k: v[:max_response_length] for k, v in chosen_tokens.items()}
        rejected_tokens = {k: v[:max_response_length] for k, v in rejected_tokens.items()}
    
    # Store components separately (memory optimization)
    batch = {
        'prompt': prompt,
        'chosen_response_only': chosen_response,
        'rejected_response_only': rejected_response,
        
        # Store tokenized components separately
        'prompt_input_ids': prompt_tokens['input_ids'],
        'prompt_attention_mask': prompt_tokens['attention_mask'],
        
        'chosen_response_input_ids': chosen_tokens['input_ids'],
        'chosen_response_attention_mask': chosen_tokens['attention_mask'],
        
        'rejected_response_input_ids': rejected_tokens['input_ids'],
        'rejected_response_attention_mask': rejected_tokens['attention_mask'],
        
        # Metadata for lazy concatenation
        'prompt_length': len(prompt_tokens['input_ids']),
        'chosen_response_length': len(chosen_tokens['input_ids']),
        'rejected_response_length': len(rejected_tokens['input_ids']),
    }
    
    return batch


def lazy_concatenate_sequences(batch_element: Dict, sequence_type: str) -> Dict:
    """
    Lazy concatenation of prompt + response only when needed.
    
    Args:
        batch_element: Single batch element with separate components
        sequence_type: 'chosen' or 'rejected'
    
    Returns:
        Dict with concatenated sequences for the specified type
    """
    prompt_ids = batch_element['prompt_input_ids']
    prompt_mask = batch_element['prompt_attention_mask']
    
    if sequence_type == 'chosen':
        response_ids = batch_element['chosen_response_input_ids']
        response_mask = batch_element['chosen_response_attention_mask']
    elif sequence_type == 'rejected':
        response_ids = batch_element['rejected_response_input_ids']
        response_mask = batch_element['rejected_response_attention_mask']
    else:
        raise ValueError(f"Invalid sequence_type: {sequence_type}")
    
    # Concatenate sequences
    concatenated_ids = prompt_ids + response_ids
    concatenated_mask = prompt_mask + response_mask
    
    # Create labels (mask prompt tokens)
    labels = concatenated_ids[:]
    labels[:len(prompt_ids)] = [-100] * len(prompt_ids)
    
    return {
        f'{sequence_type}_input_ids': concatenated_ids,
        f'{sequence_type}_attention_mask': concatenated_mask,
        f'{sequence_type}_labels': labels,
    }


def get_memory_optimized_collate_fn(tokenizer) -> Callable:
    """
    Memory-optimized collate function that performs lazy concatenation.
    
    This function concatenates prompt + response only during collation,
    reducing memory usage during data loading.
    """
    
    def collate_fn(batch: List[Dict]) -> Dict[str, Union[List, torch.Tensor]]:
        # Perform lazy concatenation for each element
        processed_batch = []
        
        for element in batch:
            # Create concatenated sequences only when needed
            chosen_seq = lazy_concatenate_sequences(element, 'chosen')
            rejected_seq = lazy_concatenate_sequences(element, 'rejected')
            
            # Combine with original data
            processed_element = {
                'prompt': element['prompt'],
                'chosen': element['prompt'] + element['chosen_response_only'],
                'rejected': element['prompt'] + element['rejected_response_only'],
                'chosen_response_only': element['chosen_response_only'],
                'rejected_response_only': element['rejected_response_only'],
                **chosen_seq,
                **rejected_seq,
                # Add prompt tokens for compatibility
                'prompt_input_ids': element['prompt_input_ids'],
                'prompt_attention_mask': element['prompt_attention_mask'],
            }
            processed_batch.append(processed_element)
        
        # Standard padding logic
        padded_batch = {}
        for k in processed_batch[0].keys():
            if k.endswith('_input_ids') or k.endswith('_attention_mask') or k.endswith('_labels'):
                if 'prompt' in k and not k.startswith('prompt_'):
                    # For prompt sequences, pad on left
                    to_pad = [torch.LongTensor(ex[k][::-1]) for ex in processed_batch]
                else:
                    to_pad = [torch.LongTensor(ex[k]) for ex in processed_batch]
                
                # Determine padding value
                if k.endswith('_input_ids'):
                    padding_value = tokenizer.pad_token_id
                elif k.endswith('_labels'):
                    padding_value = -100
                elif k.endswith('_attention_mask'):
                    padding_value = 0
                else:
                    raise ValueError(f"Unexpected key in batch '{k}'")
                
                padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
                
                # Flip back prompt sequences
                if 'prompt' in k and not k.startswith('prompt_'):
                    padded_batch[k] = padded_batch[k].flip(dims=[1])
            else:
                padded_batch[k] = [ex[k] for ex in processed_batch]
        
        return padded_batch
    
    return collate_fn


def convert_to_memory_optimized_format(input_file: str, output_file: str):
    """
    Convert standard preference dataset to memory-optimized format.
    
    This removes prompt duplication from chosen/rejected fields and stores
    them separately to reduce file size and memory usage.
    """
    print(f"Converting {input_file} to memory-optimized format...")
    
    converted_count = 0
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line in tqdm(infile, desc="Converting"):
            data = json.loads(line.strip())
            
            prompt = data['prompt']
            chosen = data['chosen']
            rejected = data['rejected']
            
            # Extract response parts by removing prompt
            if chosen.startswith(prompt):
                chosen_response = chosen[len(prompt):]
            else:
                # Fallback: assume the prompt is included somewhere
                chosen_response = chosen
                
            if rejected.startswith(prompt):
                rejected_response = rejected[len(prompt):]
            else:
                rejected_response = rejected
            
            # Create memory-optimized format
            optimized_data = {
                'prompt': prompt,
                'chosen_response': chosen_response,
                'rejected_response': rejected_response
            }
            
            # Preserve any additional metadata
            for key, value in data.items():
                if key not in ['prompt', 'chosen', 'rejected']:
                    optimized_data[key] = value
            
            outfile.write(json.dumps(optimized_data) + '\n')
            converted_count += 1
    
    print(f"Converted {converted_count} examples to memory-optimized format")
    print(f"Output saved to: {output_file}")


def load_memory_optimized_dataset(file_path: str) -> List[Dict]:
    """Load memory-optimized dataset format."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            example = json.loads(line.strip())
            data.append(example)
    return data


def get_memory_optimized_batch_iterator(
    file_path: str,
    tokenizer,
    batch_size: int = 1,
    max_length: int = 512,
    max_prompt_length: int = 128,
    shuffle: bool = True,
    n_examples: Optional[int] = None
) -> Iterator[Dict]:
    """
    Memory-optimized batch iterator that loads and processes data on-demand.
    """
    # Load dataset
    dataset = load_memory_optimized_dataset(file_path)
    
    if shuffle:
        np.random.shuffle(dataset)
    
    if n_examples:
        dataset = dataset[:n_examples]
    
    # Get collate function
    collate_fn = get_memory_optimized_collate_fn(tokenizer)
    
    # Process in batches
    for i in range(0, len(dataset), batch_size):
        batch_data = dataset[i:i + batch_size]
        
        # Tokenize each element
        tokenized_batch = []
        for example in batch_data:
            tokenized_element = tokenize_batch_element_memory_optimized(
                example['prompt'],
                example['chosen_response'],
                example['rejected_response'],
                'keep_start',  # Default truncation mode
                tokenizer,
                max_length,
                max_prompt_length
            )
            tokenized_batch.append(tokenized_element)
        
        # Collate and yield
        yield collate_fn(tokenized_batch)


def estimate_memory_savings(standard_file: str, optimized_file: str) -> Dict[str, float]:
    """
    Estimate memory savings from using optimized format.
    """
    import os
    
    standard_size = os.path.getsize(standard_file)
    optimized_size = os.path.getsize(optimized_file)
    
    savings_bytes = standard_size - optimized_size
    savings_percentage = (savings_bytes / standard_size) * 100
    
    return {
        'standard_size_mb': standard_size / (1024 * 1024),
        'optimized_size_mb': optimized_size / (1024 * 1024),
        'savings_mb': savings_bytes / (1024 * 1024),
        'savings_percentage': savings_percentage
    }


def main():
    """Example usage and testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Memory-optimized dataset processing")
    parser.add_argument("--input-file", required=True, help="Input preference dataset file")
    parser.add_argument("--output-file", required=True, help="Output optimized file")
    parser.add_argument("--estimate-savings", action="store_true", help="Estimate memory savings")
    
    args = parser.parse_args()
    
    # Convert to optimized format
    convert_to_memory_optimized_format(args.input_file, args.output_file)
    
    # Estimate savings if requested
    if args.estimate_savings:
        savings = estimate_memory_savings(args.input_file, args.output_file)
        print(f"\n📊 Memory Savings Analysis:")
        print(f"Standard format: {savings['standard_size_mb']:.2f} MB")
        print(f"Optimized format: {savings['optimized_size_mb']:.2f} MB")
        print(f"Savings: {savings['savings_mb']:.2f} MB ({savings['savings_percentage']:.1f}%)")


if __name__ == "__main__":
    main()
