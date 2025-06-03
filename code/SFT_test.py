from typing import Optional
from utils import prompt_dataset, Custome_datacollator
from transformers import Trainer, TrainingArguments, TextDataset, DataCollatorForSeq2Seq, AutoTokenizer, AutoModelForCausalLM,DataCollatorForLanguageModeling,\
AutoModelForSeq2SeqLM, T5Tokenizer,GenerationConfig
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model, PeftModel
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import json
import os

#base_model_path = '/mnt/disk2/wbr/streamingLM/streaming-llm-main/models/ori_vicuna'
base_model_path = '/mnt/disk2/wbr/LLM_models/llama-2-7b-hf'
base_model_name = '/mnt/disk2/wbr/streamingLM/streaming-llm-main/models/vicuna-7b-v1.5'
adapter_path = './model_output/checkpoint-500/adapter_model.bin'
adapter_config = './model_output/checkpoint-500/adapter_config.json'
test_path = '../datasets/NoRobots/test_sft-00000-of-00001-fe658ed8e3578d4a.parquet'


# Import necessary libraries
tokenizer = AutoTokenizer.from_pretrained(
    base_model_path,                    # Load tokenizer from the base model
    torch_dtype=torch.float16,          # Set the torch data type to float16
    trust_remote_code=True,             # Trust remote code for loading tokenizer
)

# Configure LORA model
lora_config = LoraConfig(
    peft_type="LORA",                   # Set the PEFT type to LORA
    task_type="CASUAL_LM",              # Set the task type to CASUAL_LM
    r=8,                               # Set the value of r to 16
    lora_alpha=16,                      # Set the value of lora_alpha to 32
    target_modules=["q_proj", "v_proj"],  # Set the target modules for LORA
    lora_dropout=0.1,                   # Set the dropout rate for LORA
)

# Configure generation settings
generation_config = GenerationConfig(
    do_sample=True,                     # Enable sampling during generation
    min_length=-1,                      # Set minimum length of generated sequence to -1 (no minimum)
    top_p=1,                            # Set top-p sampling probability to 1 (include all possibilities)
    max_new_tokens=410,                   # Set maximum number of new tokens to add to the input
    temperature=0.1,                    # Set temperature for sampling to 0.1
    eos_token_id=tokenizer.eos_token_id,  # Set the EOS token ID
)

if __name__ == "__main__":
    # Create test dataloader using the TextDataset class
    test_dataloader = prompt_dataset.TextDataset_NoRobots(test_path, tokenizer=tokenizer, mode='test')

    # Load the base model for causal language modeling
    model = AutoModelForCausalLM.from_pretrained(base_model_path)

    # Load the PEFT model
    peft_model = PeftModel.from_pretrained(
        model,
        './model_output/sft-llama2-r8-2/checkpoint-47500',
        torch_dtype=torch.float16,
        device_map="auto",
    )

    # Initialize counters
    total_count = len(test_dataloader)
    correct_count = 0
    count = 0

    # Set models to evaluation mode
    model.eval()
    peft_model.eval()

    # Perform inference on test data
    with torch.no_grad():
        for encoded_input in test_dataloader:
            #print("encoded_input:", encoded_input)
            count += 1
            input_text = tokenizer.decode(encoded_input[0],skip_special_tokens=True)

            # Generate output using the PEFT model
            output_result = peft_model.generate(encoded_input, generation_config=generation_config)
            output_text = tokenizer.decode(output_result[0], skip_special_tokens=True)

            # Compare last word of output with last word of label
            # Print information for debugging
            print("input_text:", input_text)
            print("output_text:", output_text)
            print("count:", count)
