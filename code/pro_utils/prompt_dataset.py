from torch.utils.data import ConcatDataset, Dataset
from transformers import AutoTokenizer
from sklearn.preprocessing import LabelEncoder
import pyarrow.parquet as pq
import json

class TextDataset(Dataset):
    def __init__(self, filename, tokenizer, mode='train') -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.lines = []
        self.mode = mode
        with open(filename, 'r') as file:
            if mode =='train':
                for line in file.readlines():
                    data = json.loads(line)
                    instruction = data['instruction']
                    rewards = data['rewards']
                    for reward in rewards:
                        response = reward['response']
                        scores = reward['scores']
                        score = dict_to_stringline(scores)
                        input_text = "instruction: " + instruction + "response: " + response + "Evaluate the quality of the response in the following ways: instruction_following, honesty, truthfulness, helpfulness, overall_score" + score
                        self.lines.append(input_text)
            elif mode=='test':
                for line in file.readlines():
                    data = json.loads(line)
                    instruction = data['instruction']
                    rewards = data['rewards']
                    for reward in rewards:
                        response = reward['response']
                        input_text = "instruction: " + instruction + "response: " + response + "Evaluate the quality of the response in the following five ways: "
                        self.lines.append(input_text)

    def __len__(self):
        return len(self.lines)
    
    def __getitem__(self, index):
        text = self.lines[index]
        if self.mode=='train':
            return text
        elif self.mode=='test':
            encoded_text = self.tokenizer.encode(text, add_special_tokens=True, return_tensors='pt')
            return encoded_text 
        #encoded_text = self.tokenizer(text, return_tensors='pt', padding=True, max_length=4096,truncation=True)


def dict_to_stringline(scores:dict):
        
        joint_string = ','.join(f"{key}:{value}" for key, value in scores.items())
        return joint_string

class TextDataset_NoRobots(Dataset):
    
    def __init__(self, filename, tokenizer, mode='train') -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.lines = []
        self.mode = mode
        parquet_file = pq.ParquetFile(filename)
        batch_size = 1
        
        if mode == 'train':
            for batch in parquet_file.iter_batches(batch_size=batch_size):
                tabel = batch.to_pandas()
                for index, row in tabel.iterrows():
                    category = row['category']
                    messages = row['messages']
                    prompt = row['prompt']
                    dialogue_list = preprocess_conversation(messages)
                    dialogue = ' '.join(dialogue_list)
                    input_text = prompt + dialogue
                    self.lines.append(input_text)
        elif mode =='test':
            for batch in parquet_file.iter_batches(batch_size=batch_size):
                tabel = batch.to_pandas()
                for index, row in tabel.iterrows():
                    messages = row['messages']
                    prompt = row['prompt']
                    for conver in messages:
                        if conver['role'] == 'user':
                            dialogue = conver['content']
                            input_text = prompt + dialogue
                            self.lines.append(input_text)
                    # dialogue_list = preprocess_conversation(messages)
                    # dialogue = ' '.join(dialogue_list)
                    # input_text = "prompt: " + prompt + "dialogue: " + dialogue
                    # self.lines.append(input_text)

    def __len__(self):
        return len(self.lines)
    
    def __getitem__(self, index):
        text = self.lines[index]
        if self.mode=='train':
            return text
        elif self.mode=='test':
            encoded_text = self.tokenizer.encode(text, add_special_tokens=True, return_tensors='pt')
            return encoded_text


def preprocess_conversation(data:list):
    dialogue_list = []
    for conver in data:
        dialogue = conver['content']
        role = conver['role']
        dialogue_list.append(dialogue)
    return dialogue_list


if __name__=="__main__":
    dataset_path = '../../datasets/NoRobots/test_sft-00000-of-00001-fe658ed8e3578d4a.parquet'
    tokenier = AutoTokenizer.from_pretrained('/mnt/disk2/wbr/LLM_models/llama-2-7b-hf')
    NoRobots = TextDataset_NoRobots(dataset_path, tokenizer=tokenier)
    print(NoRobots[0])
