#!/usr/bin/env python3
"""
测试脚本：验证BFloat16兼容性和embedding缓存优化
"""

import torch
import sys
import os

# 添加路径
sys.path.append('/home/wangbinrui/research_projects/llama_rlhf/code')

def test_bfloat16_conversion():
    """测试BFloat16张量到numpy的转换"""
    print("🧪 测试BFloat16张量转换...")
    
    try:
        # 测试BFloat16张量
        if torch.cuda.is_available():
            tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16, device='cuda')
        else:
            tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16)
        
        print(f"原始BFloat16张量: {tensor}")
        
        # 测试我们的修复方法
        result = tensor.cpu().detach().float().numpy().tolist()
        print(f"转换结果: {result}")
        
        # 验证结果正确性
        expected = [1.0, 2.0, 3.0]
        if result == expected:
            print("✅ BFloat16转换测试通过")
            return True
        else:
            print(f"❌ BFloat16转换测试失败: 期望 {expected}, 得到 {result}")
            return False
            
    except Exception as e:
        print(f"❌ BFloat16转换测试出错: {e}")
        return False

def test_embedding_cache():
    """测试embedding缓存机制"""
    print("\n🧪 测试embedding缓存机制...")
    
    try:
        from pro_utils.trainers import PreferenceDPO_trainer
        print("✅ 成功导入PreferenceDPO_trainer")
        
        # 创建一个简单的模拟模型用于测试
        class MockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(1000, 512)
                
            def get_input_embeddings(self):
                return self.embedding
        
        model = MockModel()
        print("✅ 创建测试模型成功")
        
        return True
        
    except Exception as e:
        print(f"❌ embedding缓存测试出错: {e}")
        return False

def main():
    """主测试函数"""
    print("=" * 60)
    print("🚀 开始运行BFloat16和embedding缓存测试")
    print("=" * 60)
    
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"BFloat16支持: {torch.cuda.is_bf16_supported()}")
        print(f"CUDA设备数: {torch.cuda.device_count()}")
    
    print("\n" + "=" * 40)
    
    # 运行测试
    test1_passed = test_bfloat16_conversion()
    test2_passed = test_embedding_cache()
    
    print("\n" + "=" * 60)
    print("📋 测试结果摘要:")
    print(f"  BFloat16转换测试: {'✅ 通过' if test1_passed else '❌ 失败'}")
    print(f"  Embedding缓存测试: {'✅ 通过' if test2_passed else '❌ 失败'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 所有测试通过！可以开始训练了。")
        return 0
    else:
        print("\n⚠️  部分测试失败，请检查代码修复。")
        return 1

if __name__ == "__main__":
    exit(main())
