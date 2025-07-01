#!/usr/bin/env python3
"""
测试归一化策略的效果
"""

import torch
import torch.nn.functional as F

def test_normalization_strategies():
    """测试不同归一化策略的效果"""
    
    print("开始测试归一化策略...")
    
    # 模拟数据：preference logratios 比 base logratios 数值范围更大
    torch.manual_seed(42)
    
    # 基础logratios (隐式偏好) - 较小的数值范围
    pi_logratios_raw = torch.randn(8) * 0.5  # [-1.5, 1.5] 范围
    
    # 偏好logratios (显式偏好) - 较大的数值范围
    pi_pref_logratios_raw = torch.randn(8) * 2.0 + 3.0  # [1, 5] 范围左右
    
    print("原始数据统计:")
    print(f"Base logratios - Mean: {pi_logratios_raw.mean():.4f}, Std: {pi_logratios_raw.std():.4f}")
    print(f"Pref logratios - Mean: {pi_pref_logratios_raw.mean():.4f}, Std: {pi_pref_logratios_raw.std():.4f}")
    print(f"Base range: [{pi_logratios_raw.min():.4f}, {pi_logratios_raw.max():.4f}]")
    print(f"Pref range: [{pi_pref_logratios_raw.min():.4f}, {pi_pref_logratios_raw.max():.4f}]")
    print()
    
    strategies = ["min_max", "z_score", "scale_to_base", "adaptive_scaling", "soft_clamp", "none"]
    
    for strategy in strategies:
        print(f"=== 归一化策略: {strategy} ===")
        
        if strategy == "min_max":
            # Min-Max归一化
            pi_logratios_min, pi_logratios_max = pi_logratios_raw.min(), pi_logratios_raw.max()
            pi_pref_logratios_min, pi_pref_logratios_max = pi_pref_logratios_raw.min(), pi_pref_logratios_raw.max()
            
            pi_range = pi_logratios_max - pi_logratios_min + 1e-8
            pi_pref_range = pi_pref_logratios_max - pi_pref_logratios_min + 1e-8
            
            pi_logratios = (pi_logratios_raw - pi_logratios_min) / pi_range
            pi_pref_logratios = (pi_pref_logratios_raw - pi_pref_logratios_min) / pi_pref_range
            
        elif strategy == "z_score":
            # Z-score标准化
            pi_logratios = (pi_logratios_raw - pi_logratios_raw.mean()) / (pi_logratios_raw.std() + 1e-8)
            pi_pref_logratios = (pi_pref_logratios_raw - pi_pref_logratios_raw.mean()) / (pi_pref_logratios_raw.std() + 1e-8)
            
        elif strategy == "scale_to_base":
            # 缩放到base量级
            base_scale = pi_logratios_raw.abs().mean() + 1e-8
            pref_scale = pi_pref_logratios_raw.abs().mean() + 1e-8
            scale_factor = base_scale / pref_scale
            
            pi_logratios = pi_logratios_raw
            pi_pref_logratios = pi_pref_logratios_raw * scale_factor
            print(f"Scale factor: {scale_factor:.4f}")
            
        elif strategy == "adaptive_scaling":
            # 自适应缩放
            base_var = pi_logratios_raw.var() + 1e-8
            pref_var = pi_pref_logratios_raw.var() + 1e-8
            scale_factor = torch.sqrt(base_var / pref_var)
            
            pi_logratios = pi_logratios_raw
            pi_pref_logratios = pi_pref_logratios_raw * scale_factor
            print(f"Adaptive scale factor: {scale_factor:.4f}")
            
        elif strategy == "soft_clamp":
            # 软钳位
            pi_logratios = torch.tanh(pi_logratios_raw / 2.0) * 2.0
            pi_pref_logratios = torch.tanh(pi_pref_logratios_raw / 2.0) * 2.0
            
        else:  # "none"
            # 简单clamp
            pi_logratios = torch.clamp(pi_logratios_raw, min=-10, max=10)
            pi_pref_logratios = torch.clamp(pi_pref_logratios_raw, min=-10, max=10)
        
        # 计算对齐损失
        explicit_to_implicit_logp = pi_pref_logratios.detach() - pi_logratios
        aligned_loss = F.relu(explicit_to_implicit_logp).mean()
        
        print(f"归一化后 Base - Mean: {pi_logratios.mean():.4f}, Std: {pi_logratios.std():.4f}")
        print(f"归一化后 Pref - Mean: {pi_pref_logratios.mean():.4f}, Std: {pi_pref_logratios.std():.4f}")
        print(f"归一化后 Base range: [{pi_logratios.min():.4f}, {pi_logratios.max():.4f}]")
        print(f"归一化后 Pref range: [{pi_pref_logratios.min():.4f}, {pi_pref_logratios.max():.4f}]")
        print(f"对齐损失: {aligned_loss.item():.4f}")
        print(f"差值统计 - Mean: {explicit_to_implicit_logp.mean():.4f}, Std: {explicit_to_implicit_logp.std():.4f}")
        print()

if __name__ == "__main__":
    try:
        test_normalization_strategies()
        print("测试完成！")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
