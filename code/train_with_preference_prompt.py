import os
import argparse
from typing import Optional
from transformers import Trainer, TrainingArguments, TextDataset, DataCollatorForSeq2Seq, AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model, PeftModel
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import json

import deepspeed
from pro_utils import preference_datasets
from pro_utils.trainers import PreferenceDPO_trainer
deepspeed.ops.op_builder.CPUAdamBuilder().load()

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ['HF_HUB_OFFLINE'] = '1'

def load_prompt_embeddings(prompt_embedding_path: str, device: str = 'cuda'):
    """Load P-tuning prompt embeddings from file"""
    if prompt_embedding_path and os.path.exists(prompt_embedding_path):
        print(f"Loading prompt embeddings from {prompt_embedding_path}")
        prompt_embeddings = torch.load(prompt_embedding_path, map_location=device)
        
        # Handle different file formats
        if isinstance(prompt_embeddings, dict):
            if 'prompt_embeddings' in prompt_embeddings:
                embeddings = prompt_embeddings['prompt_embeddings']
            elif 'embeddings' in prompt_embeddings:
                embeddings = prompt_embeddings['embeddings']
            else:
                # Take the first tensor value in the dict
                embeddings = list(prompt_embeddings.values())[0]
                if not isinstance(embeddings, torch.Tensor):
                    raise ValueError("Prompt embedding file does not contain valid embeddings")
        else:
            embeddings = prompt_embeddings
            
        if not isinstance(embeddings, torch.Tensor):
            raise ValueError("Prompt embedding must be a torch.Tensor")
            
        print(f"Loaded prompt embeddings with shape: {embeddings.shape}")
        return embeddings
    return None

class PreferenceTrainingArguments(TrainingArguments):
    def __init__(self, *args, **kwargs):
        # Extract custom parameters including preference_text and prompt_embedding_path
        self.datasets = kwargs.pop("datasets", None)
        self.custom_dataset_path = kwargs.pop("custom_dataset_path", None)
        self.max_length = kwargs.pop("max_length", None)
        self.max_prompt_length = kwargs.pop("max_prompt_length", None)
        self.loss_name = kwargs.pop("loss_name", None)
        self.beta = kwargs.pop("beta", None)
        self.alpha = kwargs.pop("alpha", 0.1)  # Parameter for preference loss weighting
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
        self.prompt_embedding_path = kwargs.pop("prompt_embedding_path", None)
        self.use_prompt_embedding = kwargs.pop("use_prompt_embedding", False)
        
        # Call parent constructor
        super().__init__(*args, **kwargs)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Preference-guided DPO training with optional prompt embeddings")
    
    # Training mode selection
    parser.add_argument(
        "--use-prompt-embedding",
        action="store_true",
        help="Use prompt embeddings instead of hard text instructions"
    )
    parser.add_argument(
        "--prompt-embedding-path",
        type=str,
        default=None,
        help="Path to the trained prompt embedding file (.pt or .pth). Required if --use-prompt-embedding is set."
    )
    
    # Model and dataset paths
    parser.add_argument(
        "--policy-model-path",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Path to the policy model"
    )
    parser.add_argument(
        "--reference-model-path", 
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Path to the reference model"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="/home/wangbinrui/research_projects/llama_rlhf/datasets/ultrafeedback_binarized/train_prefs_ultrafeedback_binarized.jsonl",
        help="Path to the training dataset"
    )
    parser.add_argument(
        "--test-dataset-path",
        type=str,
        default="/home/wangbinrui/research_projects/llama_rlhf/datasets/ultrafeedback_binarized/test_prefs_ultrafeedback_binarized.jsonl",
        help="Path to the test dataset"
    )
    
    # Training parameters
    parser.add_argument(
        "--preference-text",
        type=str,
        default="Please provide a helpful, honest, harmless, and concise response that respects user autonomy.",
        help="Preference instruction text (ignored if using prompt embeddings)"
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.05,
        help="Beta parameter for DPO loss"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="Alpha parameter for preference loss weighting"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--num-train-epochs",
        type=int,
        default=1,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=512,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=300,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--max-prompt-length",
        type=int,
        default=128,
        help="Maximum prompt length"
    )
    
    # LoRA parameters
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank"
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha"
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.1,
        help="LoRA dropout"
    )
    
    # Output and logging
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./model_output/Preference_Guided_DPO",
        help="Output directory"
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Run name for wandb logging"
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="Preference_Guided_DPO",
        help="Wandb project name"
    )
    
    # DeepSpeed parameters (automatically added by DeepSpeed launcher)
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training (automatically set by DeepSpeed)"
    )
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Validate arguments
    if args.use_prompt_embedding and not args.prompt_embedding_path:
        raise ValueError("--prompt-embedding-path is required when --use-prompt-embedding is set")
    
    if args.use_prompt_embedding and not os.path.exists(args.prompt_embedding_path):
        raise ValueError(f"Prompt embedding file not found: {args.prompt_embedding_path}")
    
    # Load prompt embeddings if specified
    prompt_embeddings = None
    if args.use_prompt_embedding:
        prompt_embeddings = load_prompt_embeddings(args.prompt_embedding_path)
        if prompt_embeddings is None:
            raise ValueError("Failed to load prompt embeddings")
        print(f"Using prompt embeddings for training (shape: {prompt_embeddings.shape})")
    else:
        print(f"Using hard text instruction: \"{args.preference_text}\"")

    # Load DeepSpeed configuration
    deepspeed_cfg = json.load(open("./deepspeed_config/ds_config.json", encoding="utf8"))
    reference_cfg = json.load(open("./deepspeed_config/ref_config.json", encoding="utf8"))

    # LoRA configuration
    lora_config = LoraConfig(
        peft_type="LORA",
        task_type="CASUAL_LM",
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=args.lora_dropout
    )

    # Tokenizer and models setup
    tokenizer = AutoTokenizer.from_pretrained(
        args.policy_model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    policy_model = AutoModelForCausalLM.from_pretrained(args.policy_model_path)
    reference_model = AutoModelForCausalLM.from_pretrained(args.reference_model_path, torch_dtype="auto")

    # Apply LoRA to policy model
    policy_lora_model = get_peft_model(policy_model, lora_config)
    policy_lora_model.enable_input_require_grads()

    # Generate run name
    if args.run_name is None:
        if args.use_prompt_embedding:
            embedding_name = os.path.basename(args.prompt_embedding_path).replace('.pt', '').replace('.pth', '')
            args.run_name = f"PrefEmb-{embedding_name}-Qwen2.5-1.5B-Ins-beta{args.beta}-alpha{args.alpha}"
        else:
            args.run_name = f"PrefText-Qwen2.5-1.5B-Ins-beta{args.beta}-alpha{args.alpha}"
    
    wandb_name = f'{args.wandb_project}-{args.run_name}'

    # Training arguments
    training_args = PreferenceTrainingArguments(
        output_dir=args.output_dir,
        save_strategy="steps",
        eval_strategy="steps",
        eval_steps=1000,
        eval_every=1000,
        n_eval_model_samples=10,
        n_eval_batches=100,
        save_steps=2000,
        do_eval_at_start=False,
        num_train_epochs=args.num_train_epochs,
        gradient_checkpointing=True,
        per_device_eval_batch_size=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_dir=f'./logs/{args.wandb_project}',
        logging_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        deepspeed=deepspeed_cfg,
        bf16=True,
        report_to="tensorboard",
        datasets="ufb",
        custom_dataset_path=args.dataset_path,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        loss_name='new_pref_po',  # Using custom preference loss
        beta=args.beta,
        alpha=args.alpha,  # Weight for preference consistency loss
        label_smoothing=0,
        run_dir=f'./model_output_{args.wandb_project}/{wandb_name}',
        cache_dirs='./cache/huggingface/transformers',
        num_examples=None,
        reference_free=False,
        logging_steps=100,
        wandb_enabled=True,
        sample_during_eval=False,
        wandb_name=wandb_name,
        wandb_project=args.wandb_project,
        preference_text=args.preference_text,
        prompt_embedding_path=args.prompt_embedding_path,
        use_prompt_embedding=args.use_prompt_embedding,
    )

    # Load datasets
    train_data_name = training_args.datasets
    test_data_name = 'ufb'

    # Load the datasets based on the training mode
    if args.use_prompt_embedding:
        # For prompt embedding mode, we need to modify the dataset loading
        # to use embeddings instead of text instructions
        print("Loading datasets for prompt embedding mode...")
        train_dataset = preference_datasets.get_dataset(train_data_name, 'train', file_path=args.dataset_path)
        test_dataset = preference_datasets.get_dataset(test_data_name, 'test', file_path=args.test_dataset_path)
        
        # Create data loaders that can handle prompt embeddings
        data_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, 
                                collate_fn=preference_datasets.get_collate_fn(tokenizer=tokenizer))
    else:
        # Original text-based preference mode
        print("Loading datasets for text instruction mode...")
        train_dataset = preference_datasets.get_dataset(train_data_name, 'train', file_path=args.dataset_path)
        test_dataset = preference_datasets.get_dataset(test_data_name, 'test', file_path=args.test_dataset_path)
        
        # Create data loaders
        data_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, 
                                collate_fn=preference_datasets.get_collate_fn(tokenizer=tokenizer))

    # Initialize the preference DPO trainer with the datasets
    # Pass prompt embeddings if available
    trainer_kwargs = {
        'policy_model': policy_lora_model, 
        'args': training_args, 
        'reference_model': reference_model,
        'policy_deepspeed_config_path': deepspeed_cfg, 
        'reference_deepspeed_config_path': reference_cfg, 
        'tokenizer': tokenizer,
        'preference_text': args.preference_text,
        'train_dataset': train_dataset,
        'test_dataset': test_dataset,
        'data_loader': data_loader
    }
    
    # Add prompt embeddings if using embedding mode
    if args.use_prompt_embedding:
        trainer_kwargs['prompt_embeddings'] = prompt_embeddings
    
    preference_dpo_trainer = PreferenceDPO_trainer(**trainer_kwargs)

    # Print training configuration
    print("\n" + "="*60)
    print("🚀 Training Configuration")
    print("="*60)
    print(f"Training Mode: {'Prompt Embedding' if args.use_prompt_embedding else 'Text Instruction'}")
    if args.use_prompt_embedding:
        print(f"Prompt Embedding Path: {args.prompt_embedding_path}")
        print(f"Embedding Shape: {prompt_embeddings.shape}")
    else:
        print(f"Preference Text: \"{args.preference_text}\"")
    print(f"Policy Model: {args.policy_model_path}")
    print(f"Reference Model: {args.reference_model_path}")
    print(f"Beta: {args.beta}")
    print(f"Alpha: {args.alpha}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Epochs: {args.num_train_epochs}")
    print(f"LoRA Config: r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
    print(f"Output Dir: {args.output_dir}")
    print("="*60)

    # Start training
    print("\n🚀 Starting training...")
    preference_dpo_trainer.train()
    
    print("✅ Training completed successfully!")
    print(f"Model saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
