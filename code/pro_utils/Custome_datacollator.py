from typing import Any, List, Dict
from transformers import PreTrainedTokenizerBase
import torch 

class DataCollatorForQA_T5:
    def __init__(self, tokenizer:PreTrainedTokenizerBase, padding:bool =True) -> None:
        self.tokenizer = tokenizer
        self.padding = padding

    def __call__(self, batch: List[Dict[str, torch.Tensor]], **kwargs) -> Dict[str, torch.tensor]:

        input_ids = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        # print("input_ids:",input_ids,
        #     "labels:",labels
        #)
        if self.padding:
            inputs_encoding = self.tokenizer(list(input_ids), return_tensors='pt',padding= True, truncation=True, max_length=1024)
            tokenized_input_ids = inputs_encoding["input_ids"]
            attention_mask =  inputs_encoding["attention_mask"]
            label_encoding = self.tokenizer(list(labels), return_tensors='pt',padding= False, truncation=True, max_length=1024)
            tokenized_labels = label_encoding["input_ids"]
            print("label_encoding:",label_encoding)
            print("tokenized_labels:",tokenized_labels[:,:-1])
        # print("tokenized_input_ids:",tokenized_input_ids,
        #     "attention_mask:",attention_mask,
        #     "labels:",labels
        #)
        return {
            "input_ids":tokenized_input_ids,
            "attention_mask":attention_mask,
            #"labels": tokenized_labels[:,:-1]
            "labels": tokenized_labels
        }
class DataCollatorFor_CasualLM:
    def __init__(self, tokenizer:PreTrainedTokenizerBase, padding:bool=True) -> None:
        self.tokenizer = tokenizer
        self.padding = padding
        pass
    def __call__(self, batch: List[Dict[str, torch.Tensor]],*args: Any, **kwds: Any) -> Any:
        input_text_for_casualLM = [item for item in batch]
        if self.padding:
            input_encoding = self.tokenizer(input_text_for_casualLM, return_tensors='pt', padding=True, max_length=4096,truncation=True)
            input_ids = input_encoding["input_ids"]
            attention_mask = input_encoding["attention_mask"]
            labels = input_encoding["input_ids"].clone()
            shift_labels = torch.cat((torch.full_like(labels[:,:1], -100), labels[:,:-1]), dim=1)
        else: 
            input_ids = torch.tensor([self.tokenizer.encode(text, add_special_tokens=True) for text in input_text_for_casualLM])
            attention_mask = None
        # print(input_text_for_casualLM)
        # print("input_ids:",input_ids,input_ids.shape)
        # print("labels:",labels,labels.shape)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": shift_labels
        }