# Preference alignment Project

This repository contains the implementation of DPO () training for LLaMA and Qwen models, focusing on preference-guided optimization.

## Project Structure

```
code/
├── pro_utils/           # Core utilities for preference-guided training
│   ├── trainers.py      # Main DPO and Preference-DPO trainers
│   ├── datasets_preprocess.py
│   ├── preference_datasets.py
│   └── ...
├── deepspeed_config/    # DeepSpeed configuration files
├── DPO_train.py        # Direct Preference Optimization training
├── preference_train.py  # Preference-guided training
├── SFT_train.py        # Supervised Fine-tuning
└── ...
```

## Key Features

- **Preference-Augmented DPO**: Extended DPO trainer supporting both text-based and embedding-based preferences
- **Multiple Training Methods**: SFT, DPO, IPO, SCPD, SIPA
- **DeepSpeed Integration**: Efficient distributed training
- **Flexible Data Processing**: Support for various dataset formats

## Training Methods

### 1. Supervised Fine-tuning (SFT)
```bash
python code/SFT_train.py --config your_config
```

### 2. Direct Preference Optimization (DPO)
```bash
python code/DPO_train.py --config your_config
```

### 3. Preference-Guided Training
```bash
python code/preference_train.py --config your_config
```

## Main Components

### PreferenceDPO_trainer
Located in `code/pro_utils/trainers.py`, this is the core trainer that extends standard DPO to support preference-augmented inputs:

- Supports both text-based and embedding-based preferences
- Implements preference-augmented loss function
- Caches embedding information for efficiency
- Handles both original and preference-augmented forward passes

### Key Methods
- `preference_augmented_loss()`: Computes loss using both original and preference-augmented inputs
- `concatenated_forward_with_preference()`: Runs forward pass on both input types
- `_forward_with_embeddings()`: Incorporates prompt embeddings into input sequences

## Usage

1. Configure your training parameters in the appropriate config files
2. Prepare your dataset in the required format
3. Run the training script for your chosen method
4. Monitor training progress through wandb integration

## Requirements

- PyTorch
- DeepSpeed
- Transformers
- wandb (optional, for logging)

## Note

Large files (models, datasets, outputs) are excluded from this repository via `.gitignore`. Only source code and configuration files are tracked.
