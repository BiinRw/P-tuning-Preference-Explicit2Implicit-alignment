import os
from typing import Optional
from transformers import TrainingArguments, AutoTokenizer, AutoModelForCausalLM
import torch
import json

import deepspeed
from pro_utils.trainers import PreferenceDPO_trainer
from pro_utils.preference_datasets import get_dataset_with_preference
deepspeed.ops.op_builder.CPUAdamBuilder().load()

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ['HF_HUB_OFFLINE'] = '1'

class PreferenceTrainingArguments(TrainingArguments):
    def __init__(self, *args, **kwargs):
        # Extract custom parameters
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
        self.preference_text = kwargs.pop("preference_text", "Please provide a helpful, harmless, and honest response.")

        # Call parent constructor
        super().__init__(*args, **kwargs)

# Load configuration
deepspeed_cfg = json.load(open("./deepspeed_config/ds_config.json", encoding="utf8"))
reference_cfg = json.load(open("./deepspeed_config/ref_config.json", encoding="utf8"))

# Model paths
policy_model_path = "Qwen/Qwen2.5-1.5B-Instruct"
reference_model_path = "Qwen/Qwen3-4B"

# Prepare tokenizer and models
tokenizer = AutoTokenizer.from_pretrained(
    policy_model_path,
    torch_dtype=torch.float16,
    trust_remote_code=True,
)
tokenizer.pad_token = tokenizer.eos_token

policy_model = AutoModelForCausalLM.from_pretrained(policy_model_path)
reference_model = AutoModelForCausalLM.from_pretrained(reference_model_path, torch_dtype="auto")

# Dataset path
dataset_path = "/home/wangbinrui/research_projects/llama_rlhf/FastChat/fastchat/llm_judge/data/ufb/ufb_self_mistake/ufb_self_mistake_training_qwen2.5-1.5B-instruct-tp0-1w.jsonl"
test_dataset_path = "/home/wangbinrui/research_projects/llama_rlhf/datasets/ultrafeedback_binarized/test_prefs_ultrafeedback_binarized.jsonl"

# Define preference text
preference_instruction = "Please provide a helpful, honest, harmless, and concise response."

# Training arguments
wandb_project = 'Preference_DPO'
run_name = "SCPD-Pref-Qwen2.5-1.5B"
beta = 0.05
wandb_name = f'{wandb_project}-{run_name}-beta{beta}'

training_args = PreferenceTrainingArguments(
    output_dir=f'./model_output/{wandb_project}',
    save_strategy="steps",
    eval_strategy="steps",
    eval_steps=1000,
    eval_every=1000,
    n_eval_model_samples=10,
    n_eval_batches=100,
    save_steps=2000,
    do_eval_at_start=False,
    num_train_epochs=5,
    gradient_checkpointing=True,
    per_device_eval_batch_size=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=512,
    learning_rate=5e-4,
    logging_dir=f'./logs/{wandb_project}',
    logging_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    deepspeed=deepspeed_cfg,
    bf16=True,
    report_to="tensorboard",
    datasets="ufb",
    custom_dataset_path=dataset_path,
    max_length=300,
    max_prompt_length=128,
    loss_name='scpd',
    beta=beta,
    label_smoothing=0,
    run_dir=f'./model_output_{wandb_project}/{run_name}-beta{beta}',
    cache_dirs='./cache/huggingface/transformers',
    num_examples=None,
    reference_free=False,
    logging_steps=100,
    wandb_enabled=True,
    sample_during_eval=False,
    wandb_name=wandb_name,
    wandb_project=wandb_project,
    preference_text=preference_instruction,
)

# Initialize trainer
dpo_trainer = PreferenceDPO_trainer(
    policy_model=policy_model,
    args=training_args,
    reference_model=reference_model,
    policy_deepspeed_config_path=deepspeed_cfg,
    reference_deepspeed_config_path=reference_cfg,
    tokenizer=tokenizer,
    preference_text=preference_instruction
)

if __name__ == "__main__":
    dpo_trainer.train()
