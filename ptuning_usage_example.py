#!/usr/bin/env python3
"""
示例：如何使用新增的P-tuning功能

这个脚本展示了如何创建和使用prompt embeddings进行P-tuning推理
"""

import torch
import os

def create_sample_prompt_embedding():
    """创建一个示例的prompt embedding文件"""
    
    # 创建示例prompt embeddings
    # 假设我们有5个prompt tokens，每个token的embedding维度是4096 (LLaMA的hidden size)
    prompt_length = 5
    hidden_size = 4096
    
    # 随机初始化prompt embeddings (在实际使用中，这些应该是训练好的)
    prompt_embeddings = torch.randn(prompt_length, hidden_size)
    
    # 保存为.pt文件
    save_path = "/tmp/sample_prompt_embeddings.pt"
    torch.save(prompt_embeddings, save_path)
    
    print(f"✓ 创建了示例prompt embedding文件: {save_path}")
    print(f"  - Prompt length: {prompt_length}")
    print(f"  - Hidden size: {hidden_size}")
    print(f"  - Shape: {prompt_embeddings.shape}")
    
    return save_path

def create_dict_format_prompt_embedding():
    """创建字典格式的prompt embedding文件 (更常见的保存格式)"""
    
    prompt_length = 10
    hidden_size = 4096
    
    # 创建prompt embeddings
    prompt_embeddings = torch.randn(prompt_length, hidden_size)
    
    # 保存为字典格式 (这是更常见的保存方式)
    embedding_dict = {
        'prompt_embeddings': prompt_embeddings,
        'prompt_length': prompt_length,
        'hidden_size': hidden_size,
        'model_name': 'sample_model',
        'training_steps': 1000,
    }
    
    save_path = "/tmp/sample_prompt_embeddings_dict.pt"
    torch.save(embedding_dict, save_path)
    
    print(f"✓ 创建了字典格式的prompt embedding文件: {save_path}")
    print(f"  - 包含元数据的完整保存格式")
    
    return save_path

def show_usage_examples():
    """显示使用示例"""
    
    print("\n" + "="*60)
    print("🚀 P-tuning功能使用示例")
    print("="*60)
    
    # 创建示例文件
    simple_path = create_sample_prompt_embedding()
    dict_path = create_dict_format_prompt_embedding()
    
    print(f"\n📝 使用方法:")
    print(f"原来的命令:")
    print(f"python3 gen_hh_answers.py \\")
    print(f"    --model-path /path/to/your/model \\")
    print(f"    --model-id your-model-name")
    
    print(f"\n🆕 新增P-tuning支持后的命令:")
    print(f"python3 gen_hh_answers.py \\")
    print(f"    --model-path /path/to/your/model \\")
    print(f"    --model-id your-model-name \\")
    print(f"    --prompt-embedding-path {simple_path}")
    
    print(f"\n或者使用字典格式的embedding:")
    print(f"python3 gen_hh_answers.py \\")
    print(f"    --model-path /path/to/your/model \\")
    print(f"    --model-id your-model-name \\")
    print(f"    --prompt-embedding-path {dict_path}")
    
    print(f"\n💡 完整示例命令 (使用其他参数):")
    print(f"python3 gen_hh_answers.py \\")
    print(f"    --model-path /path/to/llama2-7b \\")
    print(f"    --model-id llama2-7b-ptuned \\")
    print(f"    --bench-name hh_bench \\")
    print(f"    --prompt-embedding-path {dict_path} \\")
    print(f"    --max-new-token 512 \\")
    print(f"    --num-choices 1")
    
    print(f"\n📋 P-tuning特性:")
    print(f"✓ 自动检测prompt embedding文件格式 (tensor或dict)")
    print(f"✓ 支持不同的embedding维度")
    print(f"✓ 兼容原有的所有参数")
    print(f"✓ 可选功能 - 不影响原有用法")
    print(f"✓ 支持多GPU推理")
    
    print(f"\n📁 Prompt Embedding文件格式:")
    print(f"1. 简单tensor格式: torch.save(embeddings, 'file.pt')")
    print(f"   - embeddings shape: [prompt_length, hidden_size]")
    print(f"2. 字典格式: torch.save({{'prompt_embeddings': embeddings, ...}}, 'file.pt')")
    print(f"   - 可以包含额外的元数据")
    
    print(f"\n🔧 如何训练P-tuning prompt embeddings:")
    print(f"1. 使用训练脚本训练prompt embeddings")
    print(f"2. 保存训练好的embeddings到.pt/.pth文件")
    print(f"3. 使用--prompt-embedding-path参数加载进行推理")

if __name__ == "__main__":
    show_usage_examples()
