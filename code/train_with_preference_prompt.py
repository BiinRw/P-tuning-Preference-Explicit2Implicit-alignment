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
from pro_utils.trainers import PreferenceDPO_trainer, DPO_trainer
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
        self.lambda_kl = kwargs.pop("lambda_kl", 0.1)  # Parameter for KL divergence constraint
        self.normalize_strategy = kwargs.pop("normalize_strategy", "scale_to_base")  # Normalization strategy
        self.pre_normalize_strategy = kwargs.pop("pre_normalize_strategy", "distribution_aware")  # Pre-normalization strategy
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
        "--lambda-kl",
        type=float,
        default=0.1,
        help="Lambda parameter for KL divergence constraint"
    )
    parser.add_argument(
        "--normalize-strategy",
        type=str,
        default="magnitude_preserve",
        choices=["none", "min_max", "z_score", "scale_to_base", "adaptive_scaling", "soft_clamp", 
                "robust_scaling", "magnitude_preserve", "percentile_scaling", "dynamic_range"],
        help="Normalization strategy for log ratios: none, min_max, z_score, scale_to_base, adaptive_scaling, soft_clamp, robust_scaling, magnitude_preserve, percentile_scaling, dynamic_range"
    )
    parser.add_argument(
        "--pre-normalize-strategy",
        type=str,
        default="distribution_aware",
        choices=["none", "distribution_aware", "robust_standardize", "percentile_clamp"],
        help="Pre-normalization strategy for log probabilities before computing ratios: none, distribution_aware (for embedding vs hard prompt mismatch), robust_standardize, percentile_clamp"
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
    parser.add_argument(
        "--loss-name",
        type=str,
        default="new_pref_po",
        help="Loss function name: dpo, ipo, new_pref_po, sipa, etc."
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

    # Extract model short name from path for naming
    model_short_name = args.policy_model_path.split('/')[-1]
    if model_short_name == "Qwen2.5-1.5B-Instruct":
        model_short_name = "Qwen2.5-1.5B"
    elif "DeepSeek-R1-Distill-Qwen-1.5B" in model_short_name:
        model_short_name = "DeepSeek-R1-Qwen1.5B"
    # Add more model name mappings as needed

    # Generate run name based on configuration
    if args.run_name is None:
        # Determine dataset prefix
        if 'helpsteer' in args.dataset_path.lower():
            dataset_prefix = "HelpSteer"
        elif 'ultrafeedback' in args.dataset_path.lower() or 'ufb' in args.dataset_path.lower():
            dataset_prefix = "UFB"
        else:
            dataset_prefix = "Custom"
        
        # Determine mode prefix
        if args.use_prompt_embedding:
            mode_prefix = "Emb"
        else:
            mode_prefix = "Text"
        
        # Generate comprehensive run name
        args.run_name = f"{dataset_prefix}-{mode_prefix}-{model_short_name}-{args.loss_name}-beta{args.beta}-alpha{args.alpha}-norm{args.normalize_strategy}"
    
    wandb_name = f'{args.wandb_project}-{args.run_name}'

    # Determine dataset type from path before creating training_args
    if 'helpsteer' in args.dataset_path.lower():
        train_data_name = 'helpsteer'
        test_data_name = 'helpsteer'
    elif 'ultrafeedback' in args.dataset_path.lower() or 'ufb' in args.dataset_path.lower():
        train_data_name = 'ufb'
        test_data_name = 'ufb'
    else:
        train_data_name = 'ufb'  # Default to ultrafeedback
        test_data_name = 'ufb'

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
        datasets=train_data_name,  # Use automatically determined dataset name
        custom_dataset_path=args.dataset_path,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        loss_name=args.loss_name,  # Use configurable loss function
        beta=args.beta,
        alpha=args.alpha,  # Weight for preference consistency loss
        lambda_kl=args.lambda_kl,  # Weight for KL divergence constraint
        normalize_strategy=args.normalize_strategy,  # Normalization strategy for log ratios
        pre_normalize_strategy=args.pre_normalize_strategy,  # Pre-normalization strategy for log probabilities
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
    
    preference_dpo_trainer = DPO_trainer(**trainer_kwargs)

    # Print training configuration
    print("\n" + "="*60)
    print("🚀 Training Configuration")
    print("="*60)
    print(f"Dataset: {'HelpSteer' if 'helpsteer' in args.dataset_path.lower() else 'UltraFeedback' if 'ultrafeedback' in args.dataset_path.lower() else 'Custom'}")
    print(f"Training Mode: {'Prompt Embedding' if args.use_prompt_embedding else 'Text Instruction'}")
    print(f"Loss Function: {args.loss_name}")
    if args.use_prompt_embedding:
        print(f"Prompt Embedding Path: {args.prompt_embedding_path}")
        print(f"Embedding Shape: {prompt_embeddings.shape}")
    else:
        print(f"Preference Text: \"{args.preference_text}\"")
    print(f"Model: {model_short_name} ({args.policy_model_path})")
    print(f"Beta: {args.beta}")
    print(f"Alpha: {args.alpha}")
    print(f"Lambda KL: {args.lambda_kl}")
    print(f"Normalize Strategy: {args.normalize_strategy}")
    print(f"Pre-Normalize Strategy: {args.pre_normalize_strategy}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Epochs: {args.num_train_epochs}")
    print(f"LoRA Config: r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
    print(f"Run Name: {args.run_name}")
    print(f"Output Dir: {args.output_dir}")
    print("="*60)

    # Start training
    print("\n🚀 Starting training...")
    preference_dpo_trainer.train()
    
    print("✅ Training completed successfully!")
    print(f"Model saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
