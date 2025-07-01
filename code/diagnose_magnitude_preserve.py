#!/usr/bin/env python3
"""
诊断magnitude_preserve归一化策略导致损失爆炸的问题
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def test_magnitude_preserve_stability():
    """
    测试magnitude_preserve策略在不同输入条件下的数值稳定性
    """
    print("=== 测试magnitude_preserve归一化策略的数值稳定性 ===\n")
    
    # 创建不同类型的测试数据
    test_cases = {
        "normal_case": {
            "pi_logratios": torch.randn(100) * 2.0,
            "pi_pref_logratios": torch.randn(100) * 0.5,
            "description": "正常情况：标准差相差4倍"
        },
        "extreme_variance": {
            "pi_logratios": torch.randn(100) * 10.0,
            "pi_pref_logratios": torch.randn(100) * 0.01,
            "description": "极端方差：标准差相差1000倍"
        },
        "very_small_values": {
            "pi_logratios": torch.randn(100) * 1e-6,
            "pi_pref_logratios": torch.randn(100) * 1e-8,
            "description": "极小值：非常接近0的数据"
        },
        "large_values": {
            "pi_logratios": torch.randn(100) * 50.0,
            "pi_pref_logratios": torch.randn(100) * 100.0,
            "description": "大值：较大的logratios数值"
        },
        "zero_variance": {
            "pi_logratios": torch.ones(100) * 2.0,
            "pi_pref_logratios": torch.randn(100) * 0.5,
            "description": "零方差：一个tensor为常数"
        }
    }
    
    # 测试原始的magnitude_preserve实现
    def original_magnitude_preserve(pi_logratios_raw, pi_pref_logratios_raw):
        """原始的可能有问题的实现"""
        base_std = pi_logratios_raw.std()
        pref_std = pi_pref_logratios_raw.std()
        
        # 保持较大的标准差作为目标范围
        target_std = torch.max(base_std, pref_std)
        
        # 缩放到相同的标准差
        pi_logratios = pi_logratios_raw * (target_std / (base_std + 1e-8))
        pi_pref_logratios = pi_pref_logratios_raw * (target_std / (pref_std + 1e-8))
        
        return pi_logratios, pi_pref_logratios
    
    # 测试改进的magnitude_preserve实现
    def improved_magnitude_preserve(pi_logratios_raw, pi_pref_logratios_raw):
        """改进的数值稳定实现"""
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
        
        return pi_logratios, pi_pref_logratios
    
    # 模拟DPO损失计算
    def calculate_dpo_loss(pi_logratios, pi_pref_logratios, beta=0.1):
        """计算DPO损失"""
        try:
            # 计算偏好对数比差异
            logits = beta * (pi_logratios - pi_pref_logratios)
            
            # DPO损失：-log(sigmoid(logits))
            loss = -torch.nn.functional.logsigmoid(logits).mean()
            
            return loss.item(), False  # 返回损失值和是否有NaN的标志
        except:
            return float('inf'), True
    
    print("测试结果：")
    print("-" * 80)
    
    for case_name, data in test_cases.items():
        print(f"\n【{case_name}】{data['description']}")
        
        pi_logratios_raw = data["pi_logratios"]
        pi_pref_logratios_raw = data["pi_pref_logratios"]
        
        print(f"  原始数据统计:")
        print(f"    pi_logratios: mean={pi_logratios_raw.mean():.6f}, std={pi_logratios_raw.std():.6f}")
        print(f"    pi_pref_logratios: mean={pi_pref_logratios_raw.mean():.6f}, std={pi_pref_logratios_raw.std():.6f}")
        print(f"    标准差比例: {(pi_logratios_raw.std() / (pi_pref_logratios_raw.std() + 1e-8)):.2f}")
        
        # 测试原始实现
        try:
            pi_norm_orig, pi_pref_norm_orig = original_magnitude_preserve(pi_logratios_raw, pi_pref_logratios_raw)
            loss_orig, has_nan_orig = calculate_dpo_loss(pi_norm_orig, pi_pref_norm_orig)
            
            print(f"  原始magnitude_preserve:")
            print(f"    归一化后: pi_std={pi_norm_orig.std():.6f}, pi_pref_std={pi_pref_norm_orig.std():.6f}")
            print(f"    损失值: {loss_orig:.6f}, 有NaN: {has_nan_orig}")
            if torch.isnan(pi_norm_orig).any() or torch.isnan(pi_pref_norm_orig).any():
                print(f"    ❌ 归一化后出现NaN值!")
        except Exception as e:
            print(f"  原始magnitude_preserve: ❌ 出现异常: {e}")
        
        # 测试改进实现
        try:
            pi_norm_imp, pi_pref_norm_imp = improved_magnitude_preserve(pi_logratios_raw, pi_pref_logratios_raw)
            loss_imp, has_nan_imp = calculate_dpo_loss(pi_norm_imp, pi_pref_norm_imp)
            
            print(f"  改进magnitude_preserve:")
            print(f"    归一化后: pi_std={pi_norm_imp.std():.6f}, pi_pref_std={pi_pref_norm_imp.std():.6f}")
            print(f"    损失值: {loss_imp:.6f}, 有NaN: {has_nan_imp}")
            if torch.isnan(pi_norm_imp).any() or torch.isnan(pi_pref_norm_imp).any():
                print(f"    ❌ 归一化后出现NaN值!")
            else:
                print(f"    ✅ 数值稳定")
        except Exception as e:
            print(f"  改进magnitude_preserve: ❌ 出现异常: {e}")
        
        print("-" * 80)

def analyze_scaling_factors():
    """
    分析不同情况下的缩放因子
    """
    print("\n=== 分析缩放因子的影响 ===\n")
    
    # 创建一系列标准差比例
    std_ratios = [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 10.0, 100.0, 1000.0]
    
    print("标准差比例 | 无限制缩放因子 | 限制后缩放因子 | 放大倍数")
    print("-" * 60)
    
    for ratio in std_ratios:
        base_std = 1.0
        pref_std = ratio
        target_std = max(base_std, pref_std)
        
        # 原始无限制的缩放因子
        unlimited_scale = target_std / pref_std
        
        # 限制后的缩放因子
        limited_scale = torch.clamp(torch.tensor(unlimited_scale), min=0.1, max=10.0).item()
        
        amplification = limited_scale / ratio if ratio != 0 else float('inf')
        
        print(f"{ratio:>8.3f}   | {unlimited_scale:>12.2f}   | {limited_scale:>12.2f}   | {amplification:>8.2f}x")

def test_loss_evolution():
    """
    测试损失在训练过程中的演化
    """
    print("\n=== 测试损失演化过程 ===\n")
    
    # 模拟训练过程中logratios的变化
    steps = 100
    losses_original = []
    losses_improved = []
    
    for step in range(steps):
        # 模拟logratios在训练过程中的变化
        # 通常在训练开始时logratios较小，随着训练进行会增大
        scale = 0.1 + step * 0.05
        
        pi_logratios = torch.randn(64) * scale
        pi_pref_logratios = torch.randn(64) * (scale * 0.1)  # preference logratios通常较小
        
        # 原始实现
        try:
            base_std = pi_logratios.std()
            pref_std = pi_pref_logratios.std()
            target_std = torch.max(base_std, pref_std)
            
            pi_norm = pi_logratios * (target_std / (base_std + 1e-8))
            pi_pref_norm = pi_pref_logratios * (target_std / (pref_std + 1e-8))
            
            logits = 0.1 * (pi_norm - pi_pref_norm)
            loss = -torch.nn.functional.logsigmoid(logits).mean()
            losses_original.append(loss.item() if not torch.isnan(loss) else float('inf'))
        except:
            losses_original.append(float('inf'))
        
        # 改进实现
        try:
            base_std = pi_logratios.std() + 1e-8
            pref_std = pi_pref_logratios.std() + 1e-8
            target_std = torch.max(base_std, pref_std)
            
            base_scale_factor = torch.clamp(target_std / base_std, min=0.1, max=10.0)
            pref_scale_factor = torch.clamp(target_std / pref_std, min=0.1, max=10.0)
            
            pi_norm = torch.clamp(pi_logratios * base_scale_factor, min=-100, max=100)
            pi_pref_norm = torch.clamp(pi_pref_logratios * pref_scale_factor, min=-100, max=100)
            
            logits = 0.1 * (pi_norm - pi_pref_norm)
            loss = -torch.nn.functional.logsigmoid(logits).mean()
            losses_improved.append(loss.item())
        except:
            losses_improved.append(float('inf'))
    
    # 统计结果
    original_finite = [l for l in losses_original if np.isfinite(l)]
    improved_finite = [l for l in losses_improved if np.isfinite(l)]
    
    print(f"原始实现:")
    print(f"  有效损失值: {len(original_finite)}/{len(losses_original)}")
    if original_finite:
        print(f"  损失范围: {min(original_finite):.4f} ~ {max(original_finite):.4f}")
        print(f"  平均损失: {np.mean(original_finite):.4f}")
    
    print(f"\n改进实现:")
    print(f"  有效损失值: {len(improved_finite)}/{len(losses_improved)}")
    if improved_finite:
        print(f"  损失范围: {min(improved_finite):.4f} ~ {max(improved_finite):.4f}")
        print(f"  平均损失: {np.mean(improved_finite):.4f}")

if __name__ == "__main__":
    # 设置随机种子以确保可重现性
    torch.manual_seed(42)
    np.random.seed(42)
    
    test_magnitude_preserve_stability()
    analyze_scaling_factors()
    test_loss_evolution()
    
    print("\n=== 总结 ===")
    print("magnitude_preserve策略导致数值不稳定的主要原因:")
    print("1. 当两个tensor的标准差相差很大时，缩放因子会变得极大")
    print("2. 极大的缩放因子会导致数值溢出，产生inf或NaN")
    print("3. 在DPO损失计算中，大的logratios会导致sigmoid函数饱和")
    print("\n改进措施:")
    print("1. 限制最大缩放因子（如10倍）")
    print("2. 限制最小缩放因子（如0.1倍）")
    print("3. 对归一化后的值进行范围限制（如[-100, 100]）")
    print("4. 添加更强的数值稳定性检查")
