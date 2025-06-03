import os
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

import deepspeed
from pro_utils import preference_datasets, trainers
deepspeed.ops.op_builder.CPUAdamBuilder().load()

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ['HF_HUB_OFFLINE'] = '1'

class MyTrainingArguments(TrainingArguments):
    def __init__(self, *args, **kwargs):
        # 从 kwargs 中提取自定义参数并删除，避免传递给父类构造函数
        self.datasets = kwargs.pop("datasets", None)
        self.custom_dataset_path = kwargs.pop("custom_dataset_path", None)
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
        self.wandb_name = kwargs.pop("wandb_name", None)
        self.wandb_project = kwargs.pop("wandb_project", None)

        # 调用父类构造函数，传递剩余的标准参数
        super().__init__(*args, **kwargs)

deepspeed_cfg = json.load(open("./deepspeed_config/ds_config.json", encoding="utf8"))
reference_cfg = json.load(open("./deepspeed_config/ref_config.json", encoding="utf8"))


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

#policy_model_path = 'mistralai/Mistral-7B-v0.3'  /home/wangbinrui/research_projects/LLM_models/llama2-7b-hf 'BAAI/bge-large-en-v1.5' Qwen/Qwen2-7B-Instruct 
# Qwen/Qwen2.5-1.5B-Instruct "HuggingFaceTB/SmolLM2-1.7B-Instruct" 
policy_model_path = "Qwen/Qwen2.5-1.5B-Instruct"
reference_model_path = "Qwen/Qwen3-4B"

#llama3_8b_instruct_path ="/var/models/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/e1945c40cd546c78e41f1151f4db032b271faeaa/" 
#adapter_config = ''
#policy_adapter_model_path = './model_output_SIPA-UFB/SIPA-Iter2-Qwen2.5-1.5B-Ins-SG_aligned_datas_3-6k-beta0.05/step-10000/checkpoint-10000'
policy_adapter_model_path = '/home/wangbinrui/research_projects/llama_rlhf/code/model_output_SIPA-HH-Helpful/SIPA-Iter1-Qwen2.5-1.5B-Ins-ori_datas_0-1k-beta0.05/step-2000/checkpoint-2000'

tokenizer = AutoTokenizer.from_pretrained(
    policy_model_path,
    torch_dtype = torch.float16,
    trust_remote_code=True,
)
tokenizer.pad_token = tokenizer.eos_token
policy_model = AutoModelForCausalLM.from_pretrained(policy_model_path)
reference_model = AutoModelForCausalLM.from_pretrained(reference_model_path, torch_dtype="auto")
#reference_model = None
# for name, param in model.named_parameters():
#     print(name, param.requires_grad, param.grad)
# policy_lora_model = PeftModel.from_pretrained(policy_model, policy_adapter_model_path, 
#                                              torch_dtype=torch.float16, device_map="auto", 
#                                                peft_config= lora_config,is_trainable=True)
policy_lora_model = get_peft_model(policy_model, lora_config)

policy_lora_model.enable_input_require_grads()
#policy_lora_model.print_trainable_parameters()
# for name, param in policy_lora_model.named_parameters():
#     print(name, param.requires_grad, param.grad)



wandb_project = 'DPO_UltraFeedback'
run_name = "SCPD-Qwen2.5-1.5B-Ins-preference-Qwen3-4B-lr1e5"

beta = 0.05
wandb_name = f'{wandb_project}-{run_name}-beta{beta}'
#wandb_name = f'{run_name}-beta{beta}'

hh_mistakes_datasets_path = "/home/wangbinrui/research_projects/llama_rlhf/FastChat/fastchat/llm_judge/data/hh_bench/splited_dataset/"
hh_dataset_file = "hh-helpful-train_0_10000.jsonl"
hh_helpful_SG_dataset_file = "/home/wangbinrui/research_projects/llama_rlhf/FastChat/fastchat/llm_judge/data/SIPA_self_generated_datas/HH-Helpful_self_generated_training_HH-Helpful-SIPA-Iter1-Qwen2.5-1.5B-Ins-ori_1k-step2000-1-5k.jsonl"

hh_test_file = "/home/wangbinrui/research_projects/llama_rlhf/FastChat/fastchat/llm_judge/data/hh_bench/hh-helpful-test.jsonl"

ultrafeedback_datasets_path = "/home/wangbinrui/research_projects/llama_rlhf/datasets/ultrafeedback_binarized"

#SM trainig set
SM_ultrafeedback_datasets_path = "/home/wangbinrui/research_projects/llama_rlhf/FastChat/fastchat/llm_judge/data/ufb/ufb_self_mistake"
Qwen_SM_ultrafeedback_data_file= "ufb_self_mistake_training_qwen2.5-1.5B-instruct-tp0-1w.jsonl"
Llama2_7b_ultrafeedback_data_file = "ufb_self_mistake_training_llama2-7b-base-tp0-1w.jsonl"
#ori training set
ori_ultrafeedback_data_file = "train_prefs_ultrafeedback_binarized_6-9k.jsonl"
ultrafeedback_test_data_file = "test_prefs_ultrafeedback_binarized.jsonl"


#dataset_path = f"{ultrafeedback_datasets_path}/{ori_ultrafeedback_data_file}"
#dataset_path = "/home/wangbinrui/research_projects/llama_rlhf/FastChat/fastchat/llm_judge/data/SIPA_self_generated_datas/ufb_self_generated_training_SIPA_Iter1_SG_aligned_3-6k-Qwen2.5-1.5B-ins-tp1-top_p5-top_k10-lr1e5-beta0.05-step10000-6-9000.jsonl"
dataset_path = f"{SM_ultrafeedback_datasets_path}/{Qwen_SM_ultrafeedback_data_file}"
test_dataset_path = f"{ultrafeedback_datasets_path}/{ultrafeedback_test_data_file}"

training_args = MyTrainingArguments(
    output_dir = f'./model_output/{wandb_project}',
    save_strategy="steps",
    eval_strategy="steps",
    eval_steps = 1000,
    eval_every = 1000,
    n_eval_model_samples = 10,
    n_eval_batches = 100,
    save_steps = 2000,
    do_eval_at_start = False,
    num_train_epochs= 5,
    gradient_checkpointing=True,
    per_device_eval_batch_size= 1,
    per_device_train_batch_size= 1,
    gradient_accumulation_steps= 512,
    learning_rate= 5e-4,
    logging_dir= f'./logs/{wandb_project}',
    logging_strategy="epoch",
    save_total_limit = 2,
    load_best_model_at_end=True,
    deepspeed=deepspeed_cfg,
    bf16=True,
    report_to="tensorboard",
    datasets = "ufb",
    #datasets = "custom",
    custom_dataset_path = dataset_path,
    max_length = 300,
    max_prompt_length = 128,
    loss_name = 'scpd',
    beta = beta,
    label_smoothing = 0,
    run_dir = f'./model_output_{wandb_project}/{run_name}-beta{beta}',
    cache_dirs = './cache/huggingface/transformers',
    num_examples = None,
    reference_free = False,
    logging_steps = 100,
    wandb_enabled = True,
    sample_during_eval = False,
    wandb_name = wandb_name,
    wandb_project = wandb_project,
    #resume_from_checkpoint="./model_output_self_mistakes/dpo-r8-llama2-7b-base-preference-chat-lr1e5-beta0.05/step-9000",
)

train_data_name = training_args.datasets
test_data_name = 'ufb'

#train_dataset = preference_datasets.get_dataset(train_data_name, 'train')
train_dataset = preference_datasets.get_dataset(train_data_name, 'train', file_path=dataset_path)
test_dataset = preference_datasets.get_dataset(test_data_name, 'test', file_path=test_dataset_path)

data_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=preference_datasets.get_collate_fn(tokenizer=tokenizer))

dpo_trainer = trainers.DPO_trainer(
    policy_model=policy_lora_model, args= training_args, reference_model= reference_model,
    policy_deepspeed_config_path= deepspeed_cfg, reference_deepspeed_config_path= reference_cfg,tokenizer= tokenizer, train_dataset= train_dataset, test_dataset= test_dataset,
    data_loader = data_loader)

if __name__ == "__main__":
    dpo_trainer.train()
    #dpo_trainer.test()
    #dpo_trainer.generate()
    #dpo
