#!/usr/bin/env python3
"""
数值稳定性诊断脚本 - 用于分析归一化策略中的NaN问题
"""

import torch
import numpy as np
import sys
import os

# 添加项目路径
sys.path.append('/home/wangbinrui/research_projects/llama_rlhf/code')

def test_normalization_stability():
    """测试各种归一化策略的数值稳定性"""
    
    print("🔍 数值稳定性诊断开始...")
    print("=" * 60)
    
    # 创建各种可能的问题数据
    test_cases = [
        {
            "name": "正常数据",
            "base": torch.randn(8) * 0.5,
            "pref": torch.randn(8) * 2.0
        },
        {
            "name": "极小数值",
            "base": torch.randn(8) * 1e-6,
            "pref": torch.randn(8) * 1e-5
        },
        {
            "name": "极大数值",
            "base": torch.randn(8) * 100,
            "pref": torch.randn(8) * 200
        },
        {
            "name": "数值差异极大",
            "base": torch.randn(8) * 1e-6,
            "pref": torch.randn(8) * 100
        },
        {
            "name": "包含零值",
            "base": torch.tensor([0.0, 0.0, 1e-8, 1e-7, 0.1, 0.2, 0.0, 0.0]),
            "pref": torch.randn(8) * 2.0
        },
        {
            "name": "标准差接近零",
            "base": torch.tensor([0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001]),
            "pref": torch.randn(8) * 2.0
        }
    ]
    
    strategies = [
        "none", "scale_to_base", "magnitude_preserve", 
        "robust_scaling", "percentile_scaling", "dynamic_range"
    ]
    
    for case in test_cases:
        print(f"\n📊 测试用例: {case['name']}")
        print("-" * 40)
        
        base_data = case["base"]
        pref_data = case["pref"]
        
        print(f"原始base数据: {base_data}")
        print(f"原始pref数据: {pref_data}")
        print(f"base范围: [{base_data.min():.6f}, {base_data.max():.6f}], std: {base_data.std():.6f}")
        print(f"pref范围: [{pref_data.min():.6f}, {pref_data.max():.6f}], std: {pref_data.std():.6f}")
        
        for strategy in strategies:
            try:
                normalized_base, normalized_pref = apply_normalization(
                    base_data.clone(), pref_data.clone(), strategy
                )
                
                # 检查NaN和Inf
                has_nan = torch.isnan(normalized_base).any() or torch.isnan(normalized_pref).any()
                has_inf = torch.isinf(normalized_base).any() or torch.isinf(normalized_pref).any()
                
                status = "✅" if not (has_nan or has_inf) else "❌"
                
                print(f"  {status} {strategy:20s}: base=[{normalized_base.min():.3f}, {normalized_base.max():.3f}], pref=[{normalized_pref.min():.3f}, {normalized_pref.max():.3f}]", end="")
                
                if has_nan:
                    print(" [NaN检测]", end="")
                if has_inf:
                    print(" [Inf检测]", end="")
                print()
                
            except Exception as e:
                print(f"  ❌ {strategy:20s}: 错误 - {str(e)}")

def apply_normalization(pi_logratios_raw, pi_pref_logratios_raw, normalize_strategy):
    """应用归一化策略"""
    
    if normalize_strategy == "magnitude_preserve":
        # 保持数值大小的归一化：只对齐方向和相对大小，保持绝对数值范围
        # 添加数值稳定性检查
        base_std = pi_logratios_raw.std() + 1e-8
        pref_std = pi_pref_logratios_raw.std() + 1e-8
        
        # 保持较大的标准差作为目标范围，但限制最大放大倍数
        target_std = torch.max(base_std, pref_std)
        
        # 限制放大倍数，避免数值爆炸
        max_scale_factor = 10.0  # 限制最大放大10倍
        base_scale_factor = torch.clamp(target_std / base_std, min=0.1, max=max_scale_factor)
        pref_scale_factor = torch.clamp(target_std / pref_std, min=0.1, max=max_scale_factor)
        
        # 缩放到相同的标准差，但保持均值
        pi_logratios = pi_logratios_raw * base_scale_factor
        pi_pref_logratios = pi_pref_logratios_raw * pref_scale_factor
        
        # 额外的数值稳定性检查
        pi_logratios = torch.clamp(pi_logratios, min=-100, max=100)
        pi_pref_logratios = torch.clamp(pi_pref_logratios, min=-100, max=100)
        
    elif normalize_strategy == "scale_to_base":
        # 将preference logratios缩放到与base logratios相同的量级
        base_scale = pi_logratios_raw.abs().mean() + 1e-8
        pref_scale = pi_pref_logratios_raw.abs().mean() + 1e-8
        scale_factor = base_scale / pref_scale
        
        pi_logratios = pi_logratios_raw
        pi_pref_logratios = pi_pref_logratios_raw * scale_factor
        
    elif normalize_strategy == "robust_scaling":
        # 鲁棒缩放：保持数值范围的同时对齐分布
        base_median = torch.median(pi_logratios_raw)
        pref_median = torch.median(pi_pref_logratios_raw)
        
        base_q75 = torch.quantile(pi_logratios_raw, 0.75)
        base_q25 = torch.quantile(pi_logratios_raw, 0.25)
        base_iqr = base_q75 - base_q25 + 1e-8
        
        pref_q75 = torch.quantile(pi_pref_logratios_raw, 0.75)
        pref_q25 = torch.quantile(pi_pref_logratios_raw, 0.25)
        pref_iqr = pref_q75 - pref_q25 + 1e-8
        
        # 缩放到相同的四分位距
        scale_factor = base_iqr / pref_iqr
        
        pi_logratios = pi_logratios_raw
        pi_pref_logratios = (pi_pref_logratios_raw - pref_median) * scale_factor + base_median
        
    elif normalize_strategy == "percentile_scaling":
        # 百分位缩放：基于90%分位数进行缩放，避免极值影响
        base_p90 = torch.quantile(torch.abs(pi_logratios_raw), 0.9) + 1e-8
        pref_p90 = torch.quantile(torch.abs(pi_pref_logratios_raw), 0.9) + 1e-8
        
        scale_factor = base_p90 / pref_p90
        
        pi_logratios = pi_logratios_raw
        pi_pref_logratios = pi_pref_logratios_raw * scale_factor
        
    elif normalize_strategy == "dynamic_range":
        # 动态范围保持：保持原始数值的动态范围
        base_range = pi_logratios_raw.max() - pi_logratios_raw.min() + 1e-8
        pref_range = pi_pref_logratios_raw.max() - pi_pref_logratios_raw.min() + 1e-8
        
        # 选择较大的范围作为目标
        target_range = torch.max(base_range, pref_range)
        
        # 缩放到目标范围
        base_scale = target_range / base_range
        pref_scale = target_range / pref_range
        
        pi_logratios = pi_logratios_raw * base_scale
        pi_pref_logratios = pi_pref_logratios_raw * pref_scale
        
    else:  # "none" or default
        # 不进行归一化，只是简单clamp
        pi_logratios = torch.clamp(pi_logratios_raw, min=-10, max=10)
        pi_pref_logratios = torch.clamp(pi_pref_logratios_raw, min=-10, max=10)
    
    return pi_logratios, pi_pref_logratios

def recommend_strategy():
    """推荐稳定的归一化策略"""
    print("\n🎯 推荐策略")
    print("=" * 60)
    print("基于数值稳定性分析，推荐使用以下策略：")
    print()
    print("1. 🥇 robust_scaling - 最稳定，对异常值鲁棒")
    print("2. 🥈 percentile_scaling - 基于分位数，避免极值影响") 
    print("3. 🥉 none - 简单clamp，最保守")
    print()
    print("❌ 避免使用：")
    print("- magnitude_preserve (原版) - 可能导致数值爆炸")
    print("- scale_to_base - 在数值差异大时不稳定")
    print()
    print("✅ magnitude_preserve (修复版) - 已添加数值稳定性保护")

if __name__ == "__main__":
    test_normalization_stability()
    recommend_strategy()
