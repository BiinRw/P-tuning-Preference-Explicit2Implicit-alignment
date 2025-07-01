import os
import sys
import torch
import json
import wandb
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    AutoTokenizer,
    AutoModelForCausalLM,
)
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# 确保能够导入本地模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ptuning_model import PTuningModel
from ptuning_trainer import PTuningTrainer

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

@dataclass
class PTuningArguments:
    """
    P-tuning训练参数配置类
    
    定义了P-tuning训练过程中需要的所有参数，包括模型配置、数据集配置、
    训练超参数等。使用dataclass装饰器简化参数管理。
    """
    model_name_or_path: str = field(
        default="Qwen/Qwen2.5-1.5B",
        metadata={"help": "预训练模型路径或模型标识符"}
    )
    dataset_name: str = field(
        default="Anthropic/hh-rlhf",
        metadata={"help": "偏好学习数据集名称"}
    )
    num_virtual_tokens: int = field(
        default=50,
        metadata={"help": "P-tuning虚拟token数量，控制软提示的长度"}
    )
    prompt_init_method: str = field(
        default="random",
        metadata={"help": "提示嵌入初始化方法：random或vocab"}
    )
    preference_loss_weight: float = field(
        default=1.0,
        metadata={"help": "偏好损失权重，用于平衡不同损失项"}
    )
    margin: float = field(
        default=0.1,
        metadata={"help": "偏好排序损失的边距参数，控制偏好强度"}
    )
    # 🆕 新增KL散度约束参数
    kl_loss_weight: float = field(
        default=0.1,
        metadata={"help": "KL散度损失权重，用于约束prompt对原模型分布的影响"}
    )
    max_length: int = field(
        default=512,
        metadata={"help": "最大序列长度，超过此长度的序列将被截断"}
    )
    # 添加新的验证和监控参数
    eval_split_ratio: float = field(
        default=0.1,
        metadata={"help": "验证集划分比例"}
    )
    early_stopping_patience: int = field(
        default=5,
        metadata={"help": "早停耐心值，连续多少次验证没有改善就停止"}
    )
    early_stopping_threshold: float = field(
        default=0.01,
        metadata={"help": "早停阈值，验证指标改善小于此值认为没有改善"}
    )
    target_accuracy: float = field(
        default=0.9,
        metadata={"help": "目标准确率，达到此值后可以考虑停止训练"}
    )
    wandb_project: str = field(
        default="ptuning-preference",
        metadata={"help": "Wandb项目名称"}
    )
    use_wandb: bool = field(
        default=True,
        metadata={"help": "是否使用wandb监控"}
    )
    # 注意：save_steps 和 eval_steps 参数已在 TrainingArguments 中定义，不需要重复定义
    # 移除 output_dir 参数，因为 TrainingArguments 已经包含了这个参数


def prepare_preference_data(examples, tokenizer, max_length):
    """
    准备偏好学习数据
    
    将原始的偏好数据（chosen/rejected对）转换为模型训练所需的格式，
    包括分词、填充、截断等预处理步骤。
    
    Args:
        examples: 包含chosen和rejected字段的数据样本
        tokenizer: 分词器
        max_length: 最大序列长度
        
    Returns:
        处理后的数据字典，包含偏好和非偏好回复的token ID和注意力掩码
    """
    preferred_texts = examples["chosen"]    # 偏好回复文本
    rejected_texts = examples["rejected"]   # 非偏好回复文本
    
    # 对偏好回复进行分词处理（添加进度条）
    print("Processing preferred texts...")
    preferred_encodings = tokenizer(
        preferred_texts,
        truncation=True,           # 启用截断
        padding="max_length",      # 填充到最大长度
        max_length=max_length,     # 最大序列长度
        return_tensors="pt"        # 返回PyTorch张量
    )
    
    # 对非偏好回复进行分词处理
    print("Processing rejected texts...")
    rejected_encodings = tokenizer(
        rejected_texts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt"
    )
    
    # 返回格式化的数据
    return {
        "preferred_input_ids": preferred_encodings["input_ids"],
        "preferred_attention_mask": preferred_encodings["attention_mask"],
        "rejected_input_ids": rejected_encodings["input_ids"],
        "rejected_attention_mask": rejected_encodings["attention_mask"],
    }


def load_preference_dataset(file_path: str) -> List[Dict[str, str]]:
    """
    加载偏好数据集，支持JSONL格式
    
    Args:
        file_path: 数据文件路径，支持.jsonl格式
        
    Returns:
        包含字典的列表，每个字典包含prompt、chosen、rejected字段
    """
    print(f"📚 Loading dataset from: {file_path}")
    
    dataset = []
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension == '.jsonl':
        # 处理JSONL格式
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    
                    # 验证必需字段
                    if not isinstance(data, dict):
                        print(f"Warning: Line {line_num} is not a dict, skipping")
                        continue
                    
                    # 检查必需字段
                    required_fields = ['chosen', 'rejected']
                    missing_fields = [field for field in required_fields if field not in data]
                    
                    if missing_fields:
                        print(f"Warning: Line {line_num} missing fields {missing_fields}, skipping")
                        continue
                    
                    # 标准化数据格式
                    sample = {
                        'prompt': str(data.get('prompt', '')),
                        'chosen': str(data['chosen']),
                        'rejected': str(data['rejected'])
                    }
                    
                    # 验证数据不为空
                    if not sample['chosen'].strip() or not sample['rejected'].strip():
                        print(f"Warning: Line {line_num} has empty chosen/rejected, skipping")
                        continue
                    
                    dataset.append(sample)
                    
                except json.JSONDecodeError as e:
                    print(f"Warning: Line {line_num} is not valid JSON: {e}")
                    continue
                except Exception as e:
                    print(f"Warning: Error processing line {line_num}: {e}")
                    continue
    
    elif file_extension == '.json':
        # 处理JSON格式
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            # 如果是列表格式
            for i, item in enumerate(data):
                if isinstance(item, dict) and 'chosen' in item and 'rejected' in item:
                    sample = {
                        'prompt': str(item.get('prompt', '')),
                        'chosen': str(item['chosen']),
                        'rejected': str(item['rejected'])
                    }
                    dataset.append(sample)
                else:
                    print(f"Warning: Item {i} missing required fields, skipping")
        
        elif isinstance(data, dict):
            # 如果是字典格式（类似defaultdict）
            for key, item in data.items():
                if isinstance(item, dict) and 'chosen' in item and 'rejected' in item:
                    sample = {
                        'prompt': str(item.get('prompt', '')),
                        'chosen': str(item['chosen']),
                        'rejected': str(item['rejected'])
                    }
                    dataset.append(sample)
                else:
                    print(f"Warning: Item {key} missing required fields, skipping")
    
    else:
        raise ValueError(f"Unsupported file format: {file_extension}. Supported: .json, .jsonl")
    
    print(f"✅ Successfully loaded {len(dataset)} samples")
    
    # 显示样本预览
    if dataset:
        sample = dataset[0]
        print(f"📝 Sample preview:")
        print(f"  Prompt: {sample['prompt'][:100]}..." if sample['prompt'] else "  Prompt: (empty)")
        print(f"  Chosen: {sample['chosen'][:100]}...")
        print(f"  Rejected: {sample['rejected'][:100]}...")
    
    return dataset


class PreferenceDataset:
    """
    偏好数据集类，用于P-tuning训练
    """
    
    def __init__(self, data: List[Dict[str, str]]):
        """
        初始化数据集
        
        Args:
            data: 包含prompt、chosen、rejected的字典列表
        """
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.data[idx]
        elif isinstance(idx, slice):
            return [self.data[i] for i in range(*idx.indices(len(self.data)))]
        else:
            raise TypeError(f"Invalid index type: {type(idx)}")
    
    def __iter__(self):
        return iter(self.data)


def create_data_collator():
    """
    创建数据收集器，处理批次数据
    
    Returns:
        数据收集器函数
    """
    def collate_fn(batch: List[Dict[str, str]]) -> Dict[str, List[str]]:
        """
        将一个批次的样本收集成批次字典
        
        Args:
            batch: 包含prompt、chosen、rejected的字典列表
            
        Returns:
            批次数据，包含chosen和rejected的文本列表
        """
        if not batch:
            return {}
        
        # 提取批次数据
        chosen_texts = []
        rejected_texts = []
        prompts = []
        
        for sample in batch:
            # 验证样本格式
            if not isinstance(sample, dict):
                raise ValueError(f"Expected dict sample, got {type(sample)}")
            
            if 'chosen' not in sample or 'rejected' not in sample:
                raise ValueError(f"Sample missing required fields. Available: {list(sample.keys())}")
            
            # 提取文本
            chosen_texts.append(str(sample['chosen']))
            rejected_texts.append(str(sample['rejected']))
            prompts.append(str(sample.get('prompt', '')))
        
        return {
            'chosen': chosen_texts,
            'rejected': rejected_texts,
            'prompt': prompts
        }
    
    return collate_fn


def split_dataset(dataset: List[Dict[str, str]], eval_ratio: float = 0.1, seed: int = 42) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """
    划分训练集和验证集
    
    Args:
        dataset: 完整数据集
        eval_ratio: 验证集比例
        seed: 随机种子
        
    Returns:
        (train_dataset, eval_dataset)
    """
    if eval_ratio <= 0 or eval_ratio >= 1:
        raise ValueError(f"eval_ratio must be between 0 and 1, got {eval_ratio}")
    
    train_data, eval_data = train_test_split(
        dataset, 
        test_size=eval_ratio, 
        random_state=seed,
        shuffle=True
    )
    
    print(f"📊 Dataset split: {len(train_data)} train, {len(eval_data)} eval")
    return train_data, eval_data


def main():
    """
    主训练函数
    """
    # 解析命令行参数
    parser = HfArgumentParser((PTuningArguments, TrainingArguments))
    ptuning_args, training_args = parser.parse_args_into_dataclasses()
    
    # 在解析后立即设置load_best_model_at_end和相关参数
    # 这样可以避免初始化时的策略验证错误
    training_args.load_best_model_at_end = True
    training_args.evaluation_strategy = "steps"
    training_args.save_strategy = "steps"
    
    # 如果用户没有指定 save_steps 和 eval_steps，则使用默认值
    if training_args.save_steps is None or training_args.save_steps == 500:  # 500是HF默认值
        training_args.save_steps = 200  # 我们的默认值
    if training_args.eval_steps is None or training_args.eval_steps == 500:  # 500是HF默认值
        training_args.eval_steps = 200  # 我们的默认值
    
    training_args.metric_for_best_model = "eval_ranking_accuracy"
    training_args.greater_is_better = True
    
    print(f"✅ Set training strategies: eval={training_args.evaluation_strategy}, save={training_args.save_strategy}")
    print(f"✅ Set steps: eval_steps={training_args.eval_steps}, save_steps={training_args.save_steps}")
    print(f"✅ Set load_best_model_at_end: {training_args.load_best_model_at_end}")

    # 初始化wandb
    if (ptuning_args.use_wandb):
        wandb.init(
            project=ptuning_args.wandb_project,
            name=f"ptuning-{ptuning_args.model_name_or_path.split('/')[-1]}-{ptuning_args.num_virtual_tokens}tokens",
            config={
                **vars(ptuning_args),
                **vars(training_args)
            }
        )
    
    # 加载分词器和基础模型
    print(f"🔧 Loading model and tokenizer: {ptuning_args.model_name_or_path}")
    with tqdm(total=2, desc="Loading model components") as pbar:
        tokenizer = AutoTokenizer.from_pretrained(ptuning_args.model_name_or_path)
        pbar.update(1)
        
        base_model = AutoModelForCausalLM.from_pretrained(
            ptuning_args.model_name_or_path,
            torch_dtype=torch.float16,  # 使用半精度以节省显存
            device_map="auto"           # 自动设备映射
        )
        pbar.update(1)
    
    # 添加填充token（如果不存在）
    if (tokenizer.pad_token is None):
        tokenizer.pad_token = tokenizer.eos_token
    
    # 创建P-tuning模型包装器
    print(f"🎯 Creating P-tuning model with {ptuning_args.num_virtual_tokens} virtual tokens")
    ptuning_model = PTuningModel(
        base_model=base_model,
        num_virtual_tokens=ptuning_args.num_virtual_tokens,
        init_method=ptuning_args.prompt_init_method,
        margin=ptuning_args.margin  # 添加margin参数
    )
    
    # 输出可训练参数信息
    total_params = sum(p.numel() for p in ptuning_model.parameters())
    trainable_params = sum(p.numel() for p in ptuning_model.parameters() if p.requires_grad)
    print(f"📊 Total parameters: {total_params:,}")
    print(f"🔥 Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    
    # 加载数据集 - 使用新的数据加载方法
    print(f"📚 Loading dataset: {ptuning_args.dataset_name}")
    
    try:
        # 直接加载数据
        raw_data = load_preference_dataset(ptuning_args.dataset_name)
        
        # 划分训练集和验证集
        train_data, eval_data = split_dataset(raw_data, ptuning_args.eval_split_ratio)
        
        # 创建数据集对象
        train_dataset = PreferenceDataset(train_data)
        eval_dataset = PreferenceDataset(eval_data)
        
        print(f"📊 Train dataset size: {len(train_dataset)} samples")
        print(f"📊 Eval dataset size: {len(eval_dataset)} samples")
        
        # 验证数据集
        if len(train_dataset) == 0:
            raise ValueError("Train dataset is empty!")
        if len(eval_dataset) == 0:
            raise ValueError("Eval dataset is empty!")
        
        # 检查第一个样本
        sample = train_dataset[0]
        print(f"✅ Dataset loaded successfully")
        print(f"Sample keys: {list(sample.keys())}")
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # 创建数据收集器
    data_collator = create_data_collator()
    
    # 创建P-tuning训练器（添加KL约束参数）
    trainer = PTuningTrainer(
        model=ptuning_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,  # 添加验证集
        preference_loss_weight=ptuning_args.preference_loss_weight,
        margin=ptuning_args.margin,
        kl_loss_weight=ptuning_args.kl_loss_weight,  # 🆕 传入KL损失权重
        data_collator=data_collator,
        processing_class=tokenizer,
        # 早停参数
        early_stopping_patience=ptuning_args.early_stopping_patience,
        early_stopping_threshold=ptuning_args.early_stopping_threshold,
        target_accuracy=ptuning_args.target_accuracy,
        use_wandb=ptuning_args.use_wandb,
    )
    
    # 开始训练
    print("🚀 Starting P-tuning training...")
    trainer.train()
    
    # 只保存prompt embeddings（因为基础模型参数被冻结）
    # P-tuning的核心优势：只需要保存少量的prompt embedding参数
    print("💾 Saving trained prompt embeddings...")
    
    # 确保输出目录存在
    os.makedirs(training_args.output_dir, exist_ok=True)
    
    # 保存prompt embeddings
    prompt_embeddings_path = os.path.join(training_args.output_dir, "prompt_embeddings.pt")
    ptuning_model.save_prompt_embeddings(prompt_embeddings_path)
    
    # 保存训练配置以便后续加载
    config_path = os.path.join(training_args.output_dir, "ptuning_config.json")
    config = {
        "model_name_or_path": ptuning_args.model_name_or_path,
        "num_virtual_tokens": ptuning_args.num_virtual_tokens,
        "prompt_init_method": ptuning_args.prompt_init_method,
        "prompt_embedding_dim": ptuning_model.prompt_embedding_dim,
        "trainable_params": trainable_params,
        "total_params": total_params
    }
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # 计算保存的文件大小
    embedding_size_mb = os.path.getsize(prompt_embeddings_path) / (1024 * 1024)
    
    print(f"✅ Training completed!")
    print(f"📁 Prompt embeddings saved to: {prompt_embeddings_path}")
    print(f"📁 Config saved to: {config_path}")
    print(f"💾 Embedding file size: {embedding_size_mb:.2f} MB")
    print(f"🎯 Efficiency: Only {trainable_params:,} parameters need to be saved!")
    print(f"   (vs {total_params:,} parameters for full model fine-tuning)")


if __name__ == "__main__":
    main()
