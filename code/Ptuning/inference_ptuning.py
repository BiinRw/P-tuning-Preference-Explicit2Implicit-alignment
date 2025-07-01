import os
import sys
import torch
import json
import re
from typing import List, Dict, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import argparse
import numpy as np
from datetime import datetime

# 确保能够导入本地模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ptuning_model import PTuningModel

def load_trained_ptuning_model(
    base_model_path: str,
    prompt_embeddings_path: str,
    config_path: Optional[str] = None,
    device: str = "auto"
) -> tuple:
    """
    加载训练好的P-tuning模型
    """
    print(f"🔧 Loading base model and tokenizer from: {base_model_path}")
    
    # 加载基础模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map=device
    )
    
    # 🚨 关键修复：设置不同的pad_token
    if tokenizer.pad_token is None:
        # 不要使用eos_token作为pad_token，这会导致attention mask问题
        if tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            # 添加一个新的pad token
            tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
            base_model.resize_token_embeddings(len(tokenizer))
    
    print(f"🔧 Tokenizer config:")
    print(f"   pad_token: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})")
    print(f"   eos_token: {tokenizer.eos_token} (id: {tokenizer.eos_token_id})")
    
    print(f"📁 Loading P-tuning configuration and embeddings...")
    
    # 加载P-tuning配置 - 修复语法错误
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"✅ Loaded config: {config}")
    else:
        # 使用默认配置
        config = {
            "num_virtual_tokens": 50,
            "prompt_embedding_dim": base_model.config.hidden_size,
            "margin": 0.1
        }
        print(f"⚠️ Using default config: {config}")
    
    # 创建P-tuning模型
    ptuning_model = PTuningModel(
        base_model=base_model,
        num_virtual_tokens=config["num_virtual_tokens"],
        prompt_embedding_dim=config.get("prompt_embedding_dim"),
        margin=config.get("margin", 0.1)
    )
    
    # 获取模型所在的设备和数据类型
    model_device = next(ptuning_model.parameters()).device
    model_dtype = next(ptuning_model.parameters()).dtype
    print(f"🎯 Model device: {model_device}")
    print(f"🎯 Model dtype: {model_dtype}")
    
    # 加载训练好的prompt embeddings并确保在正确设备和数据类型上
    print(f"📥 Loading prompt embeddings from: {prompt_embeddings_path}")
    state_dict = torch.load(prompt_embeddings_path, map_location=model_device)
    ptuning_model.prompt_embeddings.load_state_dict(state_dict)
    
    # 确保prompt embeddings在正确设备和数据类型上
    ptuning_model.prompt_embeddings = ptuning_model.prompt_embeddings.to(device=model_device, dtype=model_dtype)
    
    # 验证设备和数据类型一致性
    prompt_device = ptuning_model.prompt_embeddings.weight.device
    prompt_dtype = ptuning_model.prompt_embeddings.weight.dtype
    print(f"✅ Prompt embeddings loaded to device: {prompt_device}, dtype: {prompt_dtype}")
    
    if model_device != prompt_device or model_dtype != prompt_dtype:
        print(f"⚠️ Device/dtype mismatch detected! Moving prompt embeddings to {model_device}, {model_dtype}")
        ptuning_model.prompt_embeddings = ptuning_model.prompt_embeddings.to(device=model_device, dtype=model_dtype)
        print(f"✅ Fixed device/dtype mismatch")
    
    # 设置为评估模式
    ptuning_model.eval()
    
    print(f"✅ P-tuning model loaded successfully!")
    print(f"📊 Virtual tokens: {config['num_virtual_tokens']}")
    print(f"📊 Embedding dimension: {config.get('prompt_embedding_dim', 'auto')}")
    
    return ptuning_model, tokenizer

def generate_with_ptuning(
    model: PTuningModel,
    tokenizer,
    input_text: str,
    max_length: int = 2048,  # 🚨 增大最大长度限制，但不强制使用
    temperature: float = 0.7,
    do_sample: bool = True,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.1,
    pad_token_id: Optional[int] = None,
    # 🆕 移除max_new_tokens等限制参数
    **kwargs  # 接收其他参数但不使用
) -> str:
    """
    使用P-tuning模型生成文本 - 让模型自然生成到EOS token
    
    Args:
        max_length: 序列总长度上限（包括输入），防止无限生成
        temperature: 采样温度
        do_sample: 是否使用采样
        top_p: nucleus采样参数
        top_k: top-k采样参数
        repetition_penalty: 重复惩罚
        pad_token_id: padding token ID
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    
    # 确保prompt embeddings在正确设备和数据类型上
    if (model.prompt_embeddings.weight.device != device or 
        model.prompt_embeddings.weight.dtype != dtype):
        print(f"⚠️ Moving prompt embeddings from {model.prompt_embeddings.weight.device}:{model.prompt_embeddings.weight.dtype} to {device}:{dtype}")
        model.prompt_embeddings = model.prompt_embeddings.to(device=device, dtype=dtype)
    
    # 编码输入文本
    encoding = tokenizer(
        input_text,
        return_tensors="pt",
        add_special_tokens=True,
        padding=False,
        truncation=False  # 🚨 不截断输入
    )
    
    input_ids = encoding["input_ids"].to(device)
    input_attention_mask = encoding["attention_mask"].to(device)
    
    print(f"📝 P-tuning input: {input_text[:100]}{'...' if len(input_text) > 100 else ''}")
    print(f"🔢 P-tuning input length: {input_ids.size(1)} tokens")
    print(f"🔢 Prompt tokens: {model.num_virtual_tokens} tokens")
    
    batch_size = input_ids.size(0)
    # 获取input embeddings
    input_embeddings = model.base_model.get_input_embeddings()(input_ids)
    
    # 获取prompt embeddings
    prompt_embeddings = model.get_prompt_embeddings(batch_size)
    
    # 拼接 embeddings: [prompt_embeddings | input_embeddings]
    inputs_embeds = torch.cat([prompt_embeddings, input_embeddings], dim=1)
    
    # 创建完整的attention mask [prompt_mask | input_mask]
    prompt_attention_mask = torch.ones(
        batch_size, model.num_virtual_tokens,
        dtype=input_attention_mask.dtype,
        device=device
    )
    
    # 拼接attention mask
    full_attention_mask = torch.cat([prompt_attention_mask, input_attention_mask], dim=1)
    
    print(f"📊 Combined embeddings shape: {inputs_embeds.shape}")
    print(f"📊 Full attention mask shape: {full_attention_mask.shape}")
    
    # 设置生成参数
    if pad_token_id is None:
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    
    # 🚨 关键修改：移除所有限制性参数，让模型自然生成
    generate_kwargs = {
        "inputs_embeds": inputs_embeds,
        "attention_mask": full_attention_mask,
        "do_sample": do_sample,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "repetition_penalty": repetition_penalty,
        "pad_token_id": pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
        # 🚨 关键：只设置总长度上限，不设置新生成的token数量限制
        "max_length": max_length,  # 总序列长度上限
        # 🚨 移除这些限制性参数：
        # "max_new_tokens": 不设置，让模型自由生成
        # "min_new_tokens": 不设置最小生成数
        # "early_stopping": 使用默认行为
    }
    
    print(f"🔧 P-tuning generation mode: Natural ending (no token limits)")
    print(f"🔧 P-tuning generation parameters:")
    for key, value in generate_kwargs.items():
        if key not in ["inputs_embeds", "attention_mask"]:
            print(f"   {key}: {value}")
    
    # 生成
    with torch.no_grad():
        output_ids = model.base_model.generate(**generate_kwargs)
    
    print(f"🔢 P-tuning generated output shape: {output_ids.shape}")
    
    # 🚨 关键修复：model.generate()返回的是完整的生成序列
    # 当使用inputs_embeds时，输出结构是：[新生成的tokens]
    generated_ids = output_ids[0]  # 直接使用所有生成的token
    
    print(f"🔢 P-tuning analysis:")
    print(f"   Generated total length: {len(generated_ids)} tokens")
    
    # 检查生成结束原因
    if len(generated_ids) > 0:
        last_token = generated_ids[-1].item()
        if last_token == tokenizer.eos_token_id:
            print(f"✅ P-tuning generation ended naturally with EOS token")
        elif len(generated_ids) >= max_length:
            print(f"⚠️ P-tuning generation reached max_length limit ({max_length})")
        else:
            print(f"ℹ️ P-tuning generation ended (reason: unknown)")
    
    # 解码生成的文本
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    print(f"🎯 P-tuning final generated text length: {len(generated_text)} chars")
    
    return generated_text

def debug_model_state(model: PTuningModel, tokenizer, test_input: str = "Hello, how are you?"):
    """
    调试模型状态 - 增强版本，包括prompt embeddings解码
    """
    print("🔍 Debugging model state...")
    
    device = next(model.parameters()).device
    
    # 检查prompt embeddings
    prompt_emb = model.prompt_embeddings.weight
    print(f"📊 Prompt embeddings shape: {prompt_emb.shape}")
    print(f"📊 Prompt embeddings device: {prompt_emb.device}")
    print(f"📊 Prompt embeddings dtype: {prompt_emb.dtype}")
    print(f"📊 Prompt embeddings mean: {prompt_emb.mean().item():.6f}")
    print(f"📊 Prompt embeddings std: {prompt_emb.std().item():.6f}")
    print(f"📊 Prompt embeddings min: {prompt_emb.min().item():.6f}")
    print(f"📊 Prompt embeddings max: {prompt_emb.max().item():.6f}")
    
    # 🆕 解码prompt embeddings - 找到最相似的词汇
    print(f"\n🔤 Decoding prompt embeddings to nearest tokens:")
    decode_prompt_embeddings(model, tokenizer)
    
    # 🚨 关键检查：prompt embeddings的数值范围
    mean_val = abs(prompt_emb.mean().item())
    std_val = prompt_emb.std().item()
    
    if mean_val > 1.0:
        print(f"⚠️ Warning: Prompt embeddings mean ({mean_val:.4f}) is large, might be overtrained!")
    
    if std_val > 2.0:
        print(f"⚠️ Warning: Prompt embeddings std ({std_val:.4f}) is large, might be overtrained!")
    
    if std_val < 0.01:
        print(f"⚠️ Warning: Prompt embeddings std ({std_val:.4f}) is too small, might be undertrained!")
    
    # 检查是否有异常值
    if torch.isnan(prompt_emb).any():
        print("❌ Found NaN in prompt embeddings!")
    if torch.isinf(prompt_emb).any():
        print("❌ Found Inf in prompt embeddings!")
    
    # 检查数值分布
    emb_flat = prompt_emb.flatten().float()
    percentiles = torch.quantile(emb_flat, torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9]).to(device))
    print(f"📊 Prompt embeddings percentiles:")
    print(f"   10%: {percentiles[0].item():.4f}")
    print(f"   25%: {percentiles[1].item():.4f}")
    print(f"   50%: {percentiles[2].item():.4f}")
    print(f"   75%: {percentiles[3].item():.4f}")
    print(f"   90%: {percentiles[4].item():.4f}")
    
    # 🚨 新增：检查tokenizer配置
    print(f"\n🔧 Tokenizer configuration:")
    print(f"   Vocab size: {len(tokenizer)}")
    print(f"   BOS token: {tokenizer.bos_token} (id: {tokenizer.bos_token_id})")
    print(f"   EOS token: {tokenizer.eos_token} (id: {tokenizer.eos_token_id})")
    print(f"   PAD token: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})")
    print(f"   UNK token: {tokenizer.unk_token} (id: {tokenizer.unk_token_id})")
    
    # 测试编码和注意力掩码创建
    encoding = tokenizer(test_input, return_tensors="pt", add_special_tokens=True)
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    
    print(f"\n🔢 Test encoding:")
    print(f"   Input: '{test_input}'")
    print(f"   Input IDs: {input_ids.tolist()[0]}")
    print(f"   Attention mask: {attention_mask.tolist()[0]}")
    print(f"   Decoded: '{tokenizer.decode(input_ids[0])}'")
    
    with torch.no_grad():
        # 获取输入embedding
        input_emb = model.base_model.get_input_embeddings()(input_ids)
        print(f"📊 Input embeddings shape: {input_emb.shape}")
        print(f"📊 Input embeddings mean: {input_emb.mean().item():.6f}")
        print(f"📊 Input embeddings std: {input_emb.std().item():.6f}")
        
        # 获取prompt embedding
        prompt_emb_batch = model.get_prompt_embeddings(1)
        print(f"📊 Prompt embeddings batch shape: {prompt_emb_batch.shape}")
        print(f"📊 Prompt embeddings batch mean: {prompt_emb_batch.mean().item():.6f}")
        print(f"📊 Prompt embeddings batch std: {prompt_emb_batch.std().item():.6f}")
        
        # 检查拼接后的embedding
        combined_emb = torch.cat([prompt_emb_batch, input_emb], dim=1)
        print(f"📊 Combined embeddings shape: {combined_emb.shape}")
        print(f"📊 Combined embeddings mean: {combined_emb.mean().item():.6f}")
        print(f"📊 Combined embeddings std: {combined_emb.std().item():.6f}")
        
        # 🚨 关键检查：比较prompt和input embedding的数值范围
        prompt_range = prompt_emb_batch.max() - prompt_emb_batch.min()
        input_range = input_emb.max() - input_emb.min()
        print(f"📊 Prompt embeddings range: {prompt_range.item():.4f}")
        print(f"📊 Input embeddings range: {input_range.item():.4f}")
        
        if input_range.item() > 0:
            ratio = (prompt_range/input_range).item()
            print(f"📊 Range ratio (prompt/input): {ratio:.4f}")
            
            if ratio > 10:
                print("⚠️ Warning: Prompt embeddings have much larger range than input embeddings!")
                print("   This might cause the model to ignore input and only focus on prompts.")
            elif ratio < 0.1:
                print("⚠️ Warning: Prompt embeddings have much smaller range than input embeddings!")
                print("   Prompt might not have enough influence.")
        
        # 🚨 新增：测试attention mask创建
        print(f"\n🔍 Testing attention mask creation:")
        prompt_mask = torch.ones(1, model.num_virtual_tokens, dtype=attention_mask.dtype, device=device)
        full_mask = torch.cat([prompt_mask, attention_mask], dim=1)
        
        print(f"   Prompt mask shape: {prompt_mask.shape}")
        print(f"   Input mask shape: {attention_mask.shape}")
        print(f"   Full mask shape: {full_mask.shape}")
        print(f"   Full mask: {full_mask.tolist()[0]}")
        
        expected_length = model.num_virtual_tokens + input_ids.size(1)
        if full_mask.size(1) != expected_length:
            print(f"❌ Attention mask length mismatch! Expected: {expected_length}, Got: {full_mask.size(1)}")
        else:
            print(f"✅ Attention mask length correct: {full_mask.size(1)}")

def decode_prompt_embeddings(model: PTuningModel, tokenizer, top_k: int = 5):
    """
    将prompt embeddings解码为最相似的词汇tokens
    
    Args:
        model: P-tuning模型
        tokenizer: 分词器
        top_k: 显示每个prompt token的top-k最相似词汇
    """
    print(f"🔍 Finding nearest vocabulary tokens for each prompt embedding...")
    
    device = next(model.parameters()).device
    
    # 获取词汇表embeddings
    vocab_embeddings = model.base_model.get_input_embeddings().weight  # [vocab_size, hidden_dim]
    prompt_embeddings = model.prompt_embeddings.weight  # [num_virtual_tokens, hidden_dim]
    
    print(f"📊 Vocabulary embeddings shape: {vocab_embeddings.shape}")
    print(f"📊 Prompt embeddings shape: {prompt_embeddings.shape}")
    
    # 计算余弦相似度
    # 归一化embeddings
    vocab_embeddings_norm = torch.nn.functional.normalize(vocab_embeddings, dim=1)
    prompt_embeddings_norm = torch.nn.functional.normalize(prompt_embeddings, dim=1)
    
    # 计算相似度矩阵 [num_virtual_tokens, vocab_size]
    similarity_matrix = torch.mm(prompt_embeddings_norm, vocab_embeddings_norm.t())
    
    print(f"📊 Similarity matrix shape: {similarity_matrix.shape}")
    
    # 为每个prompt token找到最相似的词汇
    print(f"\n🔤 Top-{top_k} nearest tokens for each prompt position:")
    print("=" * 80)
    
    for i in range(prompt_embeddings.size(0)):
        # 获取第i个prompt token的相似度
        similarities = similarity_matrix[i]  # [vocab_size]
        
        # 找到top-k最相似的token
        top_similarities, top_indices = torch.topk(similarities, top_k)
        
        print(f"Prompt Token {i:2d}:")
        
        # 解码并显示
        for j, (sim_score, token_id) in enumerate(zip(top_similarities, top_indices)):
            # 解码token
            try:
                token_text = tokenizer.decode([token_id.item()])
                # 处理特殊字符显示
                if token_text.strip() == '':
                    token_text = '<SPACE>'
                elif token_text == '\n':
                    token_text = '<NEWLINE>'
                elif token_text == '\t':
                    token_text = '<TAB>'
                elif len(token_text.strip()) == 0:
                    token_text = '<WHITESPACE>'
                
                print(f"  {j+1}. Token {token_id.item():5d}: '{token_text:15s}' (sim: {sim_score.item():.4f})")
            except Exception as e:
                print(f"  {j+1}. Token {token_id.item():5d}: <DECODE_ERROR> (sim: {sim_score.item():.4f})")
        
        print()
        
        # 每10个prompt token分组显示，避免输出过长
        if (i + 1) % 10 == 0 and i < prompt_embeddings.size(0) - 1:
            print("-" * 40)
    
    print("=" * 80)
    
    # 🆕 额外分析：prompt embeddings的聚类特征
    analyze_prompt_clustering(model, tokenizer, similarity_matrix)

def analyze_prompt_clustering(model: PTuningModel, tokenizer, similarity_matrix: torch.Tensor):
    """
    分析prompt embeddings的聚类特征
    
    Args:
        model: P-tuning模型
        tokenizer: 分词器
        similarity_matrix: 相似度矩阵 [num_virtual_tokens, vocab_size]
    """
    print(f"\n📊 Analyzing prompt embedding clustering patterns...")
    
    # 1. 分析每个prompt token最相似的词汇类型
    print(f"\n🏷️ Most similar token types for each prompt position:")
    
    # 获取每个prompt token最相似的token
    _, top_indices = torch.topk(similarity_matrix, 1, dim=1)  # [num_virtual_tokens, 1]
    
    # 按类型分组
    token_types = {
        'punctuation': [],
        'letters': [],
        'numbers': [],
        'special': [],
        'chinese': [],
        'other': []
    }
    
    for i, token_id in enumerate(top_indices.flatten()):
        try:
            token_text = tokenizer.decode([token_id.item()]).strip()
            
            if not token_text:
                token_types['special'].append(i)
            elif token_text.isalpha():
                token_types['letters'].append(i)
            elif token_text.isdigit():
                token_types['numbers'].append(i)
            elif token_text in '.,!?;:()[]{}"\'-':
                token_types['punctuation'].append(i)
            elif any('\u4e00' <= char <= '\u9fff' for char in token_text):
                token_types['chinese'].append(i)
            else:
                token_types['other'].append(i)
        except:
            token_types['special'].append(i)
    
    for token_type, positions in token_types.items():
        if positions:
            print(f"  {token_type:12s}: {len(positions):2d} tokens at positions {positions[:10]}{'...' if len(positions) > 10 else ''}")
    
    # 2. 分析prompt内部相似度
    print(f"\n🔗 Internal prompt similarity analysis:")
    prompt_embeddings = model.prompt_embeddings.weight
    prompt_embeddings_norm = torch.nn.functional.normalize(prompt_embeddings, dim=1)
    
    # 计算prompt tokens之间的相似度
    internal_similarity = torch.mm(prompt_embeddings_norm, prompt_embeddings_norm.t())
    
    # 排除对角线
    mask = ~torch.eye(internal_similarity.size(0), dtype=torch.bool, device=internal_similarity.device)
    off_diagonal = internal_similarity[mask]
    
    print(f"  Mean internal similarity: {off_diagonal.mean().item():.4f}")
    print(f"  Std internal similarity:  {off_diagonal.std().item():.4f}")
    print(f"  Min internal similarity:  {off_diagonal.min().item():.4f}")
    print(f"  Max internal similarity:  {off_diagonal.max().item():.4f}")
    
    # 3. 找到最相似和最不相似的prompt token对
    internal_similarity_flat = internal_similarity[mask]
    max_sim_idx = torch.argmax(internal_similarity_flat)
    min_sim_idx = torch.argmin(internal_similarity_flat)
    
    # 将平坦索引转换为2D索引
    num_tokens = internal_similarity.size(0)
    max_i, max_j = divmod(max_sim_idx.item(), num_tokens - 1)
    if max_j >= max_i:
        max_j += 1
    
    min_i, min_j = divmod(min_sim_idx.item(), num_tokens - 1)
    if min_j >= min_i:
        min_j += 1
    
    print(f"  Most similar pair:     Prompt[{max_i:2d}] ↔ Prompt[{max_j:2d}] (sim: {internal_similarity[max_i, max_j].item():.4f})")
    print(f"  Least similar pair:    Prompt[{min_i:2d}] ↔ Prompt[{min_j:2d}] (sim: {internal_similarity[min_i, min_j].item():.4f})")

def generate_prompt_summary(model: PTuningModel, tokenizer):
    """
    生成prompt embeddings的简洁摘要
    
    Args:
        model: P-tuning模型
        tokenizer: 分词器
    """
    print(f"\n📋 Prompt Embeddings Summary:")
    print("=" * 50)
    
    device = next(model.parameters()).device
    
    # 获取词汇表embeddings
    vocab_embeddings = model.base_model.get_input_embeddings().weight
    prompt_embeddings = model.prompt_embeddings.weight
    
    # 计算相似度
    vocab_embeddings_norm = torch.nn.functional.normalize(vocab_embeddings, dim=1)
    prompt_embeddings_norm = torch.nn.functional.normalize(prompt_embeddings, dim=1)
    similarity_matrix = torch.mm(prompt_embeddings_norm, vocab_embeddings_norm.t())
    
    # 获取每个prompt token最相似的词
    _, top_indices = torch.topk(similarity_matrix, 1, dim=1)
    
    # 创建简洁的表示
    prompt_tokens = []
    for i, token_id in enumerate(top_indices.flatten()):
        try:
            token_text = tokenizer.decode([token_id.item()]).strip()
            if not token_text or len(token_text) > 10:
                token_text = f"[{token_id.item()}]"
            prompt_tokens.append(token_text)
        except:
            prompt_tokens.append(f"[{token_id.item()}]")
    
    # 分行显示，每行10个token
    print("Learned prompt sequence (nearest vocabulary tokens):")
    for i in range(0, len(prompt_tokens), 10):
        line_tokens = prompt_tokens[i:i+10]
        positions = f"[{i:2d}-{min(i+9, len(prompt_tokens)-1):2d}]"
        tokens_str = " ".join(f"{token:>8s}" for token in line_tokens)
        print(f"  {positions}: {tokens_str}")
    
    print("=" * 50)

def generate_with_detailed_logging(
    model: PTuningModel,
    tokenizer,
    input_text: str,
    max_new_tokens: int = 50,
    **kwargs
) -> str:
    """
    带详细日志的生成函数 - 使用与正常生成完全相同的逻辑
    """
    print(f"🔍 Debug mode: Using same logic as generate_with_ptuning")
    
    # 🚨 关键修复：使用完全相同的生成逻辑
    return generate_with_ptuning(
        model=model,
        tokenizer=tokenizer,
        input_text=input_text,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        do_sample=True,
        top_p=0.9,
        top_k=50,
        **kwargs
    )

def check_prompt_embedding_quality(model: PTuningModel, tokenizer):
    """
    检查prompt embedding的质量
    """
    print("\n🔍 Checking prompt embedding quality...")
    
    # 测试不同输入的生成一致性
    test_inputs = [
        "Hello",
        "Hi there",
        "Good morning",
        "How are you?"
    ]
    
    device = next(model.parameters()).device
    
    # 检查对于不同输入，prompt是否产生类似的影响
    prompt_influences = []
    
    with torch.no_grad():
        for test_input in test_inputs:
            encoding = tokenizer(test_input, return_tensors="pt", add_special_tokens=True)
            input_ids = encoding["input_ids"].to(device)
            
            # 获取有prompt和无prompt的输出logits
            input_emb = model.base_model.get_input_embeddings()(input_ids)
            prompt_emb = model.get_prompt_embeddings(1)
            
            # 有prompt的情况
            combined_emb = torch.cat([prompt_emb, input_emb], dim=1)
            with_prompt_output = model.base_model(inputs_embeds=combined_emb)
            
            # 记录prompt的影响
            logits_mean = with_prompt_output.logits.mean().item()
            prompt_influences.append(logits_mean)
            
    print(f"📊 Prompt influence consistency:")
    print(f"   Mean logits: {prompt_influences}")
    print(f"   Std deviation: {np.std(prompt_influences):.6f}")
    
    if np.std(prompt_influences) > 1.0:
        print("⚠️ Warning: Prompt influence varies significantly across inputs!")
        print("   This suggests prompt embeddings might be overtrained or unstable.")
    
    # 🚨 添加更详细的prompt embedding分析
    prompt_emb = model.prompt_embeddings.weight
    
    # 检查prompt embedding的相似性
    print(f"\n📊 Prompt embedding similarity analysis:")
    
    # 计算prompt tokens之间的相似性
    prompt_emb_norm = torch.nn.functional.normalize(prompt_emb, dim=1)
    similarity_matrix = torch.mm(prompt_emb_norm, prompt_emb_norm.t())
    
    # 排除对角线（自己与自己的相似性）
    mask = ~torch.eye(similarity_matrix.size(0), dtype=torch.bool, device=device)
    off_diagonal_similarities = similarity_matrix[mask]
    
    mean_similarity = off_diagonal_similarities.mean().item()
    std_similarity = off_diagonal_similarities.std().item()
    
    print(f"   Mean similarity between prompt tokens: {mean_similarity:.4f}")
    print(f"   Std similarity: {std_similarity:.4f}")
    
    if mean_similarity > 0.9:
        print("⚠️ Warning: Prompt tokens are too similar! This reduces diversity.")
    elif mean_similarity < 0.1:
        print("⚠️ Warning: Prompt tokens are too different! This might cause instability.")
    else:
        print("✅ Prompt token similarity looks good.")

def batch_generate_with_ptuning(
    model: PTuningModel,
    tokenizer,
    input_texts: List[str],
    **generate_kwargs
) -> List[str]:
    """
    批量生成文本
    
    Args:
        model: P-tuning模型
        tokenizer: 分词器
        input_texts: 输入文本列表
        **generate_kwargs: 生成参数
        
    Returns:
        生成文本列表
    """
    generated_texts = []
    
    print(f"🔄 Batch generating for {len(input_texts)} inputs...")
    
    for i, input_text in enumerate(tqdm(input_texts, desc="Generating")):
        try:
            generated_text = generate_with_ptuning(
                model, tokenizer, input_text, **generate_kwargs
            )
            generated_texts.append(generated_text)
            
            print(f"\n📋 Sample {i+1}/{len(input_texts)}:")
            print(f"Input: {input_text[:100]}{'...' if len(input_text) > 100 else ''}")
            print(f"Output: {generated_text[:200]}{'...' if len(generated_text) > 200 else ''}")
            print("-" * 80)
            
        except Exception as e:
            print(f"❌ Error generating for input {i+1}: {e}")
            generated_texts.append("")
    
    return generated_texts

def load_test_dataset(file_path: str, num_samples: Optional[int] = None) -> List[str]:
    """
    加载测试数据集
    
    Args:
        file_path: 数据文件路径
        num_samples: 采样数量，None表示加载全部
        
    Returns:
        输入文本列表
    """
    print(f"📚 Loading test dataset from: {file_path}")
    
    inputs = []
    
    if (file_path.endswith('.jsonl')):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if num_samples and len(inputs) >= num_samples:
                    break
                    
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    
                    # 提取输入文本
                    if 'prompt' in data and data['prompt'].strip():
                        inputs.append(data['prompt'])
                    elif 'input' in data:
                        inputs.append(data['input'])
                    elif 'chosen' in data:
                        # 如果没有prompt，使用chosen作为示例
                        inputs.append(data['chosen'][:100] + "...")  # 截取前100字符作为输入
                    
                except json.JSONDecodeError as e:
                    print(f"Warning: Line {line_num} is not valid JSON: {e}")
                    continue
    
    elif file_path.endswith('.json'):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if isinstance(data, list):
            for item in data:
                if num_samples and len(inputs) >= num_samples:
                    break
                if isinstance(item, dict):
                    if 'prompt' in item and item['prompt'].strip():
                        inputs.append(item['prompt'])
                    elif 'input' in item:
                        inputs.append(item['input'])
    
    print(f"✅ Loaded {len(inputs)} test inputs")
    return inputs

def compare_outputs(
    base_model_path: str,
    ptuning_model_path: str,
    config_path: str,
    test_inputs: List[str],
    output_file: str = None,
    **generate_kwargs
):
    """
    比较基础模型和P-tuning模型的输出，移除生成限制让模型自然结束
    """
    print("🔄 Loading models for comparison...")
    
    # 加载基础模型
    print("Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载P-tuning模型
    print("Loading P-tuning model...")
    ptuning_model, _ = load_trained_ptuning_model(
        base_model_path, ptuning_model_path, config_path
    )
    
    print("🔍 Comparing outputs...")
    
    comparison_results = []
    
    # 🚨 关键修改：移除所有生成限制参数
    temperature = generate_kwargs.get('temperature', 0.7)
    max_length = generate_kwargs.get('max_length', 2048)  # 只保留总长度上限
    
    print(f"🔧 统一生成参数 (自然结束模式):")
    print(f"   max_length: {max_length} (总序列长度上限)")
    print(f"   temperature: {temperature}")
    print(f"   do_sample: True")
    print(f"   top_p: 0.9")
    print(f"   top_k: 50")
    print(f"   repetition_penalty: 1.1")
    print(f"   无max_new_tokens限制 - 让模型自然生成到EOS")
    
    for i, input_text in enumerate(test_inputs):
        print(f"\n{'='*80}")
        print(f"Sample {i+1}/{len(test_inputs)}: {input_text[:100]}{'...' if len(input_text) > 100 else ''}")
        print('='*80)
        
        # 🚨 基础模型生成 - 移除所有限制让其自然生成
        print("🤖 Base Model Output:")
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(base_model.device)
        
        print(f"🔧 Base model generation parameters:")
        print(f"   input_ids shape: {input_ids.shape}")
        print(f"   max_length: {max_length}")
        print(f"   temperature: {temperature}")
        print(f"   do_sample: True")
        print(f"   无max_new_tokens限制")
        
        with torch.no_grad():
            base_output = base_model.generate(
                input_ids,
                max_length=max_length,  # 🚨 只设置总长度上限
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True
                # 🚨 移除这些限制：
                # max_new_tokens=xxx
                # min_new_tokens=xxx
                # early_stopping=xxx
            )
        
        print(f"🔢 Base model output shape: {base_output.shape}")
        
        # 🚨 修复：正确解码基础模型输出
        # base_model.generate()返回完整序列 [input_tokens + generated_tokens]
        base_generated_ids = base_output[0][input_ids.size(1):]  # 只取新生成的部分
        base_text = tokenizer.decode(base_generated_ids, skip_special_tokens=True)
        
        print(f"🔢 Base model generated {len(base_generated_ids)} tokens")
        print(f"Base model output: {base_text}")
        
        print("\n🎯 P-tuning Model Output:")
        
        # 🚨 P-tuning模型生成 - 同样移除限制
        ptuning_text = generate_with_ptuning(
            ptuning_model, tokenizer, input_text,
            max_length=max_length,  # 🚨 只设置总长度上限
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1
            # 🚨 移除max_new_tokens等限制参数
        )
        print(ptuning_text)
        print()
        
        # 比较生成长度
        print(f"📊 Generation length comparison:")
        print(f"   Input length: {len(input_text)} chars")
        print(f"   Base model output: {len(base_text)} chars, {len(base_generated_ids)} tokens")
        print(f"   P-tuning output: {len(ptuning_text)} chars")
        
        # 检查结束原因
        base_ended_naturally = (len(base_generated_ids) > 0 and 
                               base_generated_ids[-1].item() == tokenizer.eos_token_id)
        
        print(f"📊 Generation ending:")
        print(f"   Base model ended naturally: {base_ended_naturally}")
        
        # 保存比较结果
        comparison_results.append({
            "sample_id": i + 1,
            "input": input_text,
            "base_model_output": base_text,
            "ptuning_model_output": ptuning_text,
            "input_length": len(input_text),
            "base_output_length": len(base_text),
            "ptuning_output_length": len(ptuning_text),
            "base_tokens_generated": len(base_generated_ids),
            "base_ended_naturally": base_ended_naturally
        })
    
    # 保存比较结果到文件
    if output_file:
        comparison_data = {
            "metadata": {
                "base_model": base_model_path,
                "ptuning_model": ptuning_model_path,
                "config": config_path,
                "comparison_timestamp": datetime.now().isoformat(),
                "total_samples": len(test_inputs),
                "generation_config": {
                    "mode": "natural_ending",
                    "max_length": max_length,
                    "temperature": temperature,
                    "note": "No max_new_tokens limit - models generate until EOS"
                }
            },
            "results": comparison_results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(comparison_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Comparison results saved to: {output_file}")
    
    return comparison_results

def extract_experiment_info(prompt_embeddings_path: str) -> Dict[str, str]:
    """
    从prompt embeddings路径中提取实验信息
    
    Args:
        prompt_embeddings_path: prompt embeddings文件路径
        
    Returns:
        包含实验信息的字典
    """
    # 获取绝对路径并标准化
    abs_path = os.path.abspath(prompt_embeddings_path)
    path_parts = abs_path.split(os.sep)
    
    experiment_info = {
        'model': 'unknown',
        'vtokens': 'unknown', 
        'init_method': 'unknown',
        'kl_weight': 'unknown',
        'margin': 'unknown',
        'learning_rate': 'unknown',
        'epochs': 'unknown',
        'batch_size': 'unknown',
        'timestamp': 'unknown',
        'checkpoint': 'unknown'
    }
    
    # 寻找包含实验参数的目录名
    experiment_dir = None
    checkpoint_dir = None
    
    for i, part in enumerate(path_parts):
        # 匹配实验参数目录格式：model_vtokensX_initY_klZ_marginW_lrV_epU_bsT_timestamp
        if re.match(r'.*_vtokens\d+_init.*_kl[\d.]+_margin[\d.]+_lr[\de-]+_ep\d+_bs\d+_\d{8}_\d{6}$', part):
            experiment_dir = part
        
        # 匹配checkpoint目录格式：checkpoint-数字
        if re.match(r'checkpoint-\d+$', part):
            checkpoint_dir = part
    
    # 解析实验目录名称
    if experiment_dir:
        # 使用正则表达式提取各个参数
        patterns = {
            'model': r'^([^_]+)',
            'vtokens': r'vtokens(\d+)',
            'init_method': r'init([^_]+)',
            'kl_weight': r'kl([\d.]+)',
            'margin': r'margin([\d.]+)',
            'learning_rate': r'lr([\de.-]+)',
            'epochs': r'ep(\d+)',
            'batch_size': r'bs(\d+)',
            'timestamp': r'(\d{8}_\d{6})$'
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, experiment_dir)
            if match:
                experiment_info[key] = match.group(1)
    
    # 提取checkpoint信息
    if checkpoint_dir:
        checkpoint_match = re.search(r'checkpoint-(\d+)', checkpoint_dir)
        if checkpoint_match:
            experiment_info['checkpoint'] = checkpoint_match.group(1)
    
    # 如果没有找到实验目录，尝试从文件名本身提取信息
    if experiment_dir is None:
        filename = os.path.basename(prompt_embeddings_path)
        # 尝试从父目录获取信息
        parent_dirs = path_parts[-3:]  # 取最后3级目录
        for parent_dir in parent_dirs:
            if '_vtokens' in parent_dir or '_kl' in parent_dir:
                experiment_dir = parent_dir
                break
    
    return experiment_info

def generate_output_filename(
    prompt_embeddings_path: str,
    test_mode: str = "inference",
    timestamp: bool = True
) -> str:
    """
    生成输出文件名
    
    Args:
        prompt_embeddings_path: prompt embeddings文件路径
        test_mode: 测试模式 (inference, compare, dataset等)
        timestamp: 是否添加时间戳
        
    Returns:
        生成的输出文件名
    """
    # 提取实验信息
    exp_info = extract_experiment_info(prompt_embeddings_path)
    
    # 构建文件名组件
    filename_parts = []
    
    # 1. 测试模式
    filename_parts.append(f"ptuning_{test_mode}")
    
    # 2. 模型名称（简化）
    model_name = exp_info['model'].replace('.', '-').lower()
    filename_parts.append(model_name)
    
    # 3. 关键超参数
    key_params = []
    if exp_info['vtokens'] != 'unknown':
        key_params.append(f"v{exp_info['vtokens']}")
    
    if exp_info['init_method'] != 'unknown':
        init_short = exp_info['init_method']
        # 缩短初始化方法名称
        init_mapping = {
            'natural_language': 'nat',
            'natural': 'nat',
            'random': 'rand',
            'vocab': 'vocab',
            'cluster_center': 'cluster'
        }
        init_short = init_mapping.get(init_short, init_short[:4])
        key_params.append(f"init{init_short}")
    
    if exp_info['kl_weight'] != 'unknown':
        key_params.append(f"kl{exp_info['kl_weight']}")
    
    if exp_info['margin'] != 'unknown':
        key_params.append(f"m{exp_info['margin']}")
    
    if key_params:
        filename_parts.append("_".join(key_params))
    
    # 4. 训练参数（可选）
    train_params = []
    if exp_info['learning_rate'] != 'unknown':
        # 简化学习率表示，如1e-5 -> 1e5
        lr = exp_info['learning_rate'].replace('-', '').replace('e', 'e')
        train_params.append(f"lr{lr}")
    
    if exp_info['epochs'] != 'unknown':
        train_params.append(f"ep{exp_info['epochs']}")
    
    if train_params:
        filename_parts.append("_".join(train_params))
    
    # 5. Checkpoint信息
    if exp_info['checkpoint'] != 'unknown':
        filename_parts.append(f"ckpt{exp_info['checkpoint']}")
    
    # 6. 原始时间戳（实验时间）
    if exp_info['timestamp'] != 'unknown':
        # 转换时间戳格式：20250528_133602 -> 0528-1336
        ts = exp_info['timestamp']
        if len(ts) == 15:  # YYYYMMDD_HHMMSS
            short_ts = ts[4:8] + "-" + ts[9:13]  # MMDD-HHMM
            filename_parts.append(short_ts)
    
    # 7. 当前推理时间戳（可选）
    if timestamp:
        current_ts = datetime.now().strftime("%m%d_%H%M")
        filename_parts.append(f"inf{current_ts}")
    
    # 组合文件名
    base_filename = "_".join(filename_parts)
    
    # 确保文件名不会过长（限制在200字符内）
    if len(base_filename) > 200:
        # 如果太长，保留最重要的部分
        essential_parts = [
            filename_parts[0],  # ptuning_mode
            filename_parts[1],  # model
            filename_parts[2] if len(filename_parts) > 2 else "",  # key_params
        ]
        if exp_info['checkpoint'] != 'unknown':
            essential_parts.append(f"ckpt{exp_info['checkpoint']}")
        if timestamp:
            essential_parts.append(f"inf{datetime.now().strftime('%m%d_%H%M')}")
        
        base_filename = "_".join(filter(None, essential_parts))
    
    return f"{base_filename}.json"

def create_output_directory(prompt_embeddings_path: str) -> str:
    """
    创建输出目录
    
    Args:
        prompt_embeddings_path: prompt embeddings文件路径
        
    Returns:
        输出目录路径
    """
    # 提取实验信息用于目录命名
    exp_info = extract_experiment_info(prompt_embeddings_path)
    
    # 创建基础输出目录
    base_output_dir = "./inference_outputs"
    
    # 创建实验特定的子目录
    exp_subdir_parts = []
    
    if exp_info['model'] != 'unknown':
        exp_subdir_parts.append(exp_info['model'])
    
    if exp_info['vtokens'] != 'unknown':
        exp_subdir_parts.append(f"vtokens{exp_info['vtokens']}")
    
    if exp_info['init_method'] != 'unknown':
        exp_subdir_parts.append(f"init{exp_info['init_method']}")
    
    if exp_info['timestamp'] != 'unknown':
        # 使用原始训练时间戳
        exp_subdir_parts.append(exp_info['timestamp'])
    
    if exp_subdir_parts:
        exp_subdir = "_".join(exp_subdir_parts)
        output_dir = os.path.join(base_output_dir, exp_subdir)
    else:
        output_dir = os.path.join(base_output_dir, "default")
    
    # 创建目录
    os.makedirs(output_dir, exist_ok=True)
    
    return output_dir

def debug_generation_process(
    model: PTuningModel,
    tokenizer,
    input_text: str = "Explain the importance of data privacy in the digital age."
):
    """
    调试生成过程，特别关注token截断问题
    """
    print("🔍 Debug: Step-by-step generation process")
    print("=" * 80)
    
    device = next(model.parameters()).device
    
    # 1. 输入编码
    encoding = tokenizer(input_text, return_tensors="pt", add_special_tokens=True)
    input_ids = encoding["input_ids"].to(device)
    input_attention_mask = encoding["attention_mask"].to(device)
    
    print(f"1️⃣ Input encoding:")
    print(f"   Text: '{input_text}'")
    print(f"   Input IDs: {input_ids.tolist()[0]}")
    print(f"   Length: {input_ids.size(1)}")
    print(f"   Decoded back: '{tokenizer.decode(input_ids[0])}'")
    print()
    
    # 2. 创建完整的inputs_embeds
    batch_size = input_ids.size(0)
    input_embeddings = model.base_model.get_input_embeddings()(input_ids)
    prompt_embeddings = model.get_prompt_embeddings(batch_size)
    inputs_embeds = torch.cat([prompt_embeddings, input_embeddings], dim=1)
    
    prompt_attention_mask = torch.ones(
        batch_size, model.num_virtual_tokens,
        dtype=input_attention_mask.dtype,
        device=device
    )
    full_attention_mask = torch.cat([prompt_attention_mask, input_attention_mask], dim=1)
    
    print(f"2️⃣ Combined embeddings:")
    print(f"   Prompt embeddings: {prompt_embeddings.shape}")
    print(f"   Input embeddings: {input_embeddings.shape}")
    print(f"   Combined: {inputs_embeds.shape}")
    print(f"   Attention mask: {full_attention_mask.shape}")
    print()
    
    # 3. 生成
    with torch.no_grad():
        output_ids = model.base_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    print(f"3️⃣ Generation output:")
    print(f"   Output shape: {output_ids.shape}")
    print(f"   Output IDs: {output_ids.tolist()[0]}")
    print()
    
    # 4. 分析输出结构
    print(f"4️⃣ Output analysis:")
    
    # 🚨 关键：理解输出结构
    # 当使用inputs_embeds时，生成的output_ids不会包含prompt embeddings的对应token
    # 因为prompt embeddings是连续向量，没有对应的token ID
    # 所以output_ids = [original_input_tokens] + [newly_generated_tokens]
    
    original_length = input_ids.size(1)
    total_length = output_ids.size(1)
    generated_length = total_length - original_length
    
    print(f"   Original input length: {original_length}")
    print(f"   Total output length: {total_length}")
    print(f"   Generated length: {generated_length}")
    print()
    
    # 5. 验证输入部分
    if total_length >= original_length:
        input_part = output_ids[0][:original_length]
        print(f"5️⃣ Input part verification:")
        print(f"   Original: {input_ids.tolist()[0]}")
        print(f"   In output: {input_part.tolist()}")
        print(f"   Match: {torch.equal(input_ids[0], input_part)}")
        print()
    
    # 6. 提取和解码生成部分
    if generated_length > 0:
        generated_part = output_ids[0]
        print(f"6️⃣ Generated part:")
        print(f"   Generated IDs: {generated_part.tolist()}")
        print()
        
        # 逐token解码
        print(f"   Token-by-token decode:")
        for i, token_id in enumerate(generated_part[:10]):  # 前10个
            token_text = tokenizer.decode([token_id])
            print(f"     {i}: {token_id} -> '{token_text}'")
        print()
        
        # 完整解码
        generated_text = tokenizer.decode(generated_part, skip_special_tokens=True)
        print(f"   Complete generated text:")
        print(f"     '{generated_text}'")
        print()
        
        # 检查开头是否完整
        if generated_text and not generated_text[0].isspace():
            first_char = generated_text[0]
            if first_char.islower():
                print(f"⚠️ Warning: Generated text starts with lowercase: '{first_char}'")
                print("   This might indicate incomplete word segmentation")
            else:
                print(f"✅ Generated text starts properly: '{first_char}'")
        
    print("=" * 80)

def main():
    """
    主函数：演示P-tuning模型的推理过程
    """
    parser = argparse.ArgumentParser(description="P-tuning Model Inference")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
                        help="Base model path")
    parser.add_argument("--prompt_embeddings", type=str, required=True,
                        help="Path to trained prompt embeddings")
    parser.add_argument("--config", type=str, 
                        help="Path to P-tuning config file")
    parser.add_argument("--test_data", type=str,
                        help="Path to test dataset")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Number of samples to test")
    # 🚨 修改参数：移除max_new_tokens，改为max_length
    parser.add_argument("--max_length", type=int, default=2048,
                        help="Maximum total sequence length (including input)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--compare", action="store_true",
                        help="Compare with base model")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode with detailed logging")
    parser.add_argument("--debug_generation", action="store_true",
                        help="Debug generation process step by step")
    
    args = parser.parse_args()
    
    print("🚀 P-tuning Model Inference Demo")
    print("=" * 50)
    
    print(f"🔧 Generation strategy: Natural ending (no token limits)")
    print(f"🔧 Max sequence length: {args.max_length}")
    
    # 显示提取的实验信息
    exp_info = extract_experiment_info(args.prompt_embeddings)
    print(f"📊 实验信息提取结果:")
    for key, value in exp_info.items():
        if value != 'unknown':
            print(f"   {key}: {value}")
    print()
    
    # 加载训练好的P-tuning模型
    ptuning_model, tokenizer = load_trained_ptuning_model(
        args.base_model,
        args.prompt_embeddings,
        args.config
    )
    
    # 添加生成过程调试
    if args.debug_generation:
        debug_generation_process(ptuning_model, tokenizer)
        return
    
    # 添加调试信息
    if args.debug:
        debug_model_state(ptuning_model, tokenizer)
        check_prompt_embedding_quality(ptuning_model, tokenizer)
        generate_prompt_summary(ptuning_model, tokenizer)
    
    # 准备测试输入
    if args.test_data:
        test_inputs = load_test_dataset(args.test_data, args.num_samples)
    else:
        test_inputs = [
            "Please help me understand the concept of machine learning.",
            "What are the benefits of renewable energy?",
            "Explain the importance of data privacy in the digital age.",
            "How can I improve my communication skills?",
            "What are some healthy eating habits I should adopt?"
        ][:args.num_samples]
    
    if args.compare:
        # 比较模式
        if not test_inputs:
            print("❌ Compare mode requires test inputs!")
            return
        
        test_mode = "compare"
        output_dir = create_output_directory(args.prompt_embeddings)
        output_filename = generate_output_filename(
            args.prompt_embeddings, 
            test_mode=test_mode,
            timestamp=True
        )
        output_file = os.path.join(output_dir, output_filename)
        
        # 运行比较测试
        compare_outputs(
            args.base_model,
            args.prompt_embeddings,
            args.config,
            test_inputs,
            output_file=output_file,
            max_length=args.max_length,  # 🚨 传入max_length而不是max_new_tokens
            temperature=args.temperature
        )
    else:
        # 单独测试P-tuning模型
        print(f"\n🎯 Testing P-tuning model with {len(test_inputs)} samples:")
        
        # 如果启用调试模式，先用详细日志测试第一个样本
        if args.debug and test_inputs:
            print("\n" + "="*80)
            print("🔍 DETAILED DEBUG FOR FIRST SAMPLE")
            print("="*80)
            debug_result = generate_with_detailed_logging(
                ptuning_model, tokenizer, test_inputs[0], max_length=args.max_length
            )
            print(f"🎯 Debug result: '{debug_result}'")
            print("="*80)
        
        results = []
        for i, input_text in enumerate(test_inputs):
            print(f"\n📋 Sample {i+1}/{len(test_inputs)}:")
            print(f"Input: {input_text}")
            
            # 🚨 使用新的生成参数（移除限制）
            generated_text = generate_with_ptuning(
                ptuning_model,
                tokenizer,
                input_text,
                max_length=args.max_length,  # 🚨 只设置总长度上限
                temperature=args.temperature,
                do_sample=True,
                top_p=0.9
                # 🚨 移除max_new_tokens等限制参数
            )
            
            print(f"Output: {generated_text}")
            print("-" * 80)
            
            results.append({
                "input": input_text,
                "output": generated_text,
                "generation_config": {
                    "mode": "natural_ending",
                    "max_length": args.max_length,
                    "no_token_limits": True
                }
            })
        
        # 确定测试模式
        if args.test_data:
            test_mode = "dataset"
        else:
            test_mode = "basic"
        
        # 创建输出目录和文件
        output_dir = create_output_directory(args.prompt_embeddings)
        output_filename = generate_output_filename(
            args.prompt_embeddings, 
            test_mode=test_mode,
            timestamp=True
        )
        output_file = os.path.join(output_dir, output_filename)
        
        # 保存结果
        results_with_metadata = {
            "metadata": {
                "experiment_info": exp_info,
                "inference_config": {
                    "base_model": args.base_model,
                    "prompt_embeddings_path": args.prompt_embeddings,
                    "config_path": args.config,
                    "test_mode": test_mode,
                    "num_samples": args.num_samples,
                    "max_length": args.max_length,
                    "temperature": args.temperature,
                    "generation_mode": "natural_ending",
                    "inference_timestamp": datetime.now().isoformat()
                }
            },
            "results": results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results_with_metadata, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Results saved to: {output_file}")
        print(f"📁 Output directory: {output_dir}")
        
        # 创建最新结果的符号链接
        latest_link = os.path.join(output_dir, f"latest_{test_mode}.json")
        if os.path.lexists(latest_link):
            os.unlink(latest_link)
        
        try:
            os.symlink(output_filename, latest_link)
            print(f"🔗 Latest result link: {latest_link}")
        except OSError:
            import shutil
            shutil.copy2(output_file, latest_link)
            print(f"📋 Latest result copy: {latest_link}")
    
    print("\n✅ Inference completed!")

if __name__ == "__main__":
    main()