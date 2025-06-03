from typing import Optional
from transformers import Trainer, TrainingArguments, TextDataset, DataCollatorForSeq2Seq, AutoTokenizer, AutoModelForCausalLM,DataCollatorForLanguageModeling,\
AutoModelForSeq2SeqLM, T5Tokenizer
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model, PeftModel
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import json
import os
import deepspeed
from pro_utils import preference_datasets, trainers
deepspeed.ops.op_builder.CPUAdamBuilder().load()


class MyTrainingArguments(TrainingArguments):
    def __init__(self, *args, **kwargs):
        # 从 kwargs 中提取自定义参数并删除，避免传递给父类构造函数
        self.datasets = kwargs.pop("datasets", None)
        self.max_length = kwargs.pop("max_length", None)
        self.max_prompt_length = kwargs.pop("max_prompt_length", None)
        self.loss_name = kwargs.pop("loss_name", None)
        self.beta = kwargs.pop("beta", None)
        self.label_smoothing = kwargs.pop("label_smoothing", None)
        self.run_dir = kwargs.pop("run_dir", None)
        self.eval_every = kwargs.pop("eval_every", None)
        self.do_eval_at_start = kwargs.pop("do_eval_at_start", None)
        self.num_examples = kwargs.pop("num_examples", None)
        self.cache_dirs = kwargs.pop("cache_dirs", './cache/huggingface/transformers')
        self.reference_free = kwargs.pop("reference_free", False)
        self.logging_steps = kwargs.pop("logging_steps", None)
        self.wandb_enabled = kwargs.pop("wandb_enabled", False)
        self.sample_during_eval = kwargs.pop("sample_during_eval", False)
        self.n_eval_model_samples = kwargs.pop("n_eval_model_samples", 10)
        self.n_eval_batches = kwargs.pop("n_eval_batches", 10)

        # 调用父类构造函数，传递剩余的标准参数
        super().__init__(*args, **kwargs)

deepspeed_cfg = json.load(open("./deepspeed_config/ds_config.json", encoding="utf8"))
reference_cfg = json.load(open("./deepspeed_config/ref_config.json", encoding="utf8"))

training_args = MyTrainingArguments(
    output_dir = './model_output/dpo-llama2-r8-2',
    save_strategy="steps",
    evaluation_strategy="steps",
    eval_steps = 1000,
    eval_every = 1000,
    n_eval_model_samples = 10,
    n_eval_batches = 100,
    save_steps = 2000,
    do_eval_at_start = False,
    num_train_epochs= 1,
    gradient_checkpointing=True,
    per_device_eval_batch_size= 1,
    per_device_train_batch_size= 1,
    gradient_accumulation_steps= 512,
    learning_rate= 5e-4,
    logging_dir= './logs/dpo-llama2-r8-2',
    logging_strategy="epoch",
    save_total_limit = 2,
    load_best_model_at_end=True,
    deepspeed=deepspeed_cfg,
    bf16=True,
    report_to="tensorboard",
    datasets = "hh",
    max_length = 512,
    max_prompt_length = 128,
    loss_name = 'sft',
    beta = 0.1,
    label_smoothing = 0,
    run_dir = './model_output/dpo-r8-base',
    cache_dirs = './cache/huggingface/transformers',
    num_examples = None,
    reference_free = True,
    logging_steps = 100,
    wandb_enabled = True,
    sample_during_eval = False,
    #resume_from_checkpoint="./model_output/rm-llama2-r16/checkpoint-1150",
)
#lora config
lora_config = LoraConfig(
    peft_type= "LORA",
    task_type="CASUAL_LM",
    r = 8,
    lora_alpha = 1,
    target_modules =["q_proj","v_proj"],
    #target_modules =["query_key_value"],
    lora_dropout= 0.1
)

#model_path = '/mnt/disk2/wbr/streamingLM/streaming-llm-main/models/vicuna-7b-v1.5'
policy_model_path = "/home/wangbinrui/research_projects/LLM_models/llama2-7b-hf"
reference_model_path = "/home/wangbinrui/research_projects/LLM_models/llama2-7b-hf"
adapter_config = ''
policy_adapter_model_path = './model_output/sft-llama2-r8-2/checkpoint-47500'

tokenizer = AutoTokenizer.from_pretrained(
    policy_model_path,
    torch_dtype = torch.float16,
    trust_remote_code=True,
)
tokenizer.pad_token = tokenizer.eos_token
policy_model = AutoModelForCausalLM.from_pretrained(policy_model_path)

policy_lora_model = get_peft_model(policy_model, lora_config)

policy_lora_model.enable_input_require_grads()
policy_lora_model.print_trainable_parameters()
for name, param in policy_lora_model.named_parameters():
    print(name, param.requires_grad, param.grad)

train_data_name = training_args.datasets
test_data_name = training_args.datasets

train_dataset = preference_datasets.get_dataset(train_data_name, 'train')
test_dataset = preference_datasets.get_dataset(test_data_name, 'test')

data_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=preference_datasets.get_collate_fn(tokenizer=tokenizer))


sft_trainer = trainers.DPO_trainer(
    policy_model=policy_lora_model, args= training_args, reference_model= None,
    policy_deepspeed_config_path= deepspeed_cfg, reference_deepspeed_config_path= reference_cfg,tokenizer= tokenizer, train_dataset= train_dataset, test_dataset= test_dataset,
    data_loader = data_loader)

if __name__ == "__main__":
    sft_trainer.train()
    #dpo_trainer.test()
    #dpo_trainer.generate()
    #dpo

