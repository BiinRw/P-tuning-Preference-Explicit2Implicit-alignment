#!/usr/bin/env python3
"""
测试预归一化修复的脚本
主要测试数据类型兼容性问题
"""
import torch
import torch.nn.functional as F


def test_quantile_dtype_compatibility():
    """测试torch.quantile对不同数据类型的支持"""
    print("🧪 测试torch.quantile数据类型兼容性...")
    
    # 模拟不同的数据类型
    dtypes_to_test = [torch.float32, torch.float16, torch.bfloat16, torch.float64]
    
    for dtype in dtypes_to_test:
        print(f"\n  测试数据类型: {dtype}")
        try:
            # 创建测试数据
            test_tensor = torch.randn(100, dtype=dtype)
            
            # 测试直接使用quantile
            try:
                q75_direct = torch.quantile(test_tensor, 0.75)
                print(f"    ✅ 直接quantile成功: {q75_direct:.4f}")
            except Exception as e:
                print(f"    ❌ 直接quantile失败: {e}")
            
            # 测试转换为float后使用quantile
            try:
                test_tensor_float = test_tensor.float()
                q75_float = torch.quantile(test_tensor_float, 0.75)
                print(f"    ✅ 转换为float后quantile成功: {q75_float:.4f}")
            except Exception as e:
                print(f"    ❌ 转换为float后quantile失败: {e}")
                
        except Exception as e:
            print(f"    ❌ 创建{dtype}类型tensor失败: {e}")


def simulate_pre_normalize_logps():
    """模拟预归一化函数的核心逻辑"""
    print("\n🧪 测试预归一化函数逻辑...")
    
    # 模拟embedding vs hard prompt的分布差异
    # 模拟hard prompt的log概率 (较小范围)
    policy_chosen_logps = torch.randn(10, dtype=torch.float16) * 0.5  # [-1, 1]范围
    policy_rejected_logps = torch.randn(10, dtype=torch.float16) * 0.5
    
    # 模拟embedding prompt的log概率 (较大范围)
    policy_pref_chosen_logps = torch.randn(10, dtype=torch.float16) * 20 - 50  # [-70, -30]范围
    policy_pref_rejected_logps = torch.randn(10, dtype=torch.float16) * 20 - 50
    
    print(f"原始数据范围:")
    print(f"  Base chosen: [{policy_chosen_logps.min():.4f}, {policy_chosen_logps.max():.4f}]")
    print(f"  Pref chosen: [{policy_pref_chosen_logps.min():.4f}, {policy_pref_chosen_logps.max():.4f}]")
    
    # 测试分布感知的归一化策略
    try:
        # 存储原始数据类型
        original_dtype = policy_chosen_logps.dtype
        original_device = policy_chosen_logps.device
        
        # 转换为float进行计算
        base_logps = torch.cat([policy_chosen_logps, policy_rejected_logps]).float()
        pref_logps = torch.cat([policy_pref_chosen_logps, policy_pref_rejected_logps]).float()
        
        base_mean, base_std = base_logps.mean(), base_logps.std() + 1e-8
        pref_mean, pref_std = pref_logps.mean(), pref_logps.std() + 1e-8
        
        base_range = base_logps.max() - base_logps.min()
        pref_range = pref_logps.max() - pref_logps.min()
        
        print(f"\n分布统计:")
        print(f"  Base: mean={base_mean:.4f}, std={base_std:.4f}, range={base_range:.4f}")
        print(f"  Pref: mean={pref_mean:.4f}, std={pref_std:.4f}, range={pref_range:.4f}")
        
        # 应用归一化策略
        if pref_range > base_range * 2:  # Pref has much larger range
            print(f"\n检测到Pref分布范围更大，应用归一化...")
            
            target_mean, target_std = base_mean, base_std
            
            # 使用robust scaling
            pref_median = pref_logps.median()
            pref_q75 = torch.quantile(pref_logps, 0.75)
            pref_q25 = torch.quantile(pref_logps, 0.25)
            pref_iqr = pref_q75 - pref_q25 + 1e-8
            
            scale_factor = target_std / (pref_iqr / 1.349)
            
            policy_pref_chosen_logps_norm = (policy_pref_chosen_logps.float() - pref_median) * scale_factor + target_mean
            policy_pref_rejected_logps_norm = (policy_pref_rejected_logps.float() - pref_median) * scale_factor + target_mean
            
            policy_chosen_logps_norm = policy_chosen_logps.float()
            policy_rejected_logps_norm = policy_rejected_logps.float()
            
            # 转换回原始数据类型
            policy_chosen_logps_norm = policy_chosen_logps_norm.to(dtype=original_dtype)
            policy_rejected_logps_norm = policy_rejected_logps_norm.to(dtype=original_dtype)
            policy_pref_chosen_logps_norm = policy_pref_chosen_logps_norm.to(dtype=original_dtype)
            policy_pref_rejected_logps_norm = policy_pref_rejected_logps_norm.to(dtype=original_dtype)
            
            print(f"归一化后范围:")
            print(f"  Base chosen: [{policy_chosen_logps_norm.min():.4f}, {policy_chosen_logps_norm.max():.4f}]")
            print(f"  Pref chosen: [{policy_pref_chosen_logps_norm.min():.4f}, {policy_pref_chosen_logps_norm.max():.4f}]")
            
            # 测试log ratio计算
            pi_logratios_raw = policy_chosen_logps_norm - policy_rejected_logps_norm
            pi_pref_logratios_raw = policy_pref_chosen_logps_norm - policy_pref_rejected_logps_norm
            
            print(f"\nLog ratios范围:")
            print(f"  Base ratios: [{pi_logratios_raw.min():.4f}, {pi_logratios_raw.max():.4f}]")
            print(f"  Pref ratios: [{pi_pref_logratios_raw.min():.4f}, {pi_pref_logratios_raw.max():.4f}]")
            
            print("✅ 预归一化测试成功！")
        else:
            print("ℹ️  分布范围相似，不需要特殊归一化")
            
    except Exception as e:
        print(f"❌ 预归一化测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("🚀 开始测试预归一化修复...")
    print("="*50)
    
    test_quantile_dtype_compatibility()
    simulate_pre_normalize_logps()
    
    print("\n" + "="*50)
    print("🎉 测试完成！")
