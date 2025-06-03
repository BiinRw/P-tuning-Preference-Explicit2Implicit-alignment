#!/bin/bash

# 🔍 训练配置验证脚本
echo "🔍 验证训练配置..."

cd /home/wangbinrui/research_projects/llama_rlhf/code

echo "📁 检查必要文件..."
echo "✓ 主训练脚本: $([ -f train_with_preference_prompt.py ] && echo "存在" || echo "❌缺失")"
echo "✓ 快速训练脚本: $([ -f fast_train.sh ] && echo "存在" || echo "❌缺失")"
echo "✓ DeepSpeed配置: $([ -f deepspeed_config/ds_config.json ] && echo "存在" || echo "❌缺失")"

echo ""
echo "📊 检查数据集..."
TRAIN_DATA="/home/wangbinrui/research_projects/llama_rlhf/datasets/ultrafeedback_binarized/train_prefs_ultrafeedback_binarized.jsonl"
TEST_DATA="/home/wangbinrui/research_projects/llama_rlhf/datasets/ultrafeedback_binarized/test_prefs_ultrafeedback_binarized.jsonl"

if [ -f "$TRAIN_DATA" ]; then
    TRAIN_COUNT=$(wc -l < "$TRAIN_DATA")
    echo "✓ 训练数据集: 存在 ($TRAIN_COUNT 样本)"
else
    echo "❌ 训练数据集: 缺失 ($TRAIN_DATA)"
fi

if [ -f "$TEST_DATA" ]; then
    TEST_COUNT=$(wc -l < "$TEST_DATA")
    echo "✓ 测试数据集: 存在 ($TEST_COUNT 样本)"
else
    echo "❌ 测试数据集: 缺失 ($TEST_DATA)"
fi

echo ""
echo "🖥️ 检查GPU可用性..."
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA驱动: 已安装"
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    echo "✓ 可用GPU数量: $GPU_COUNT"
    echo "📊 GPU状态:"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | while read line; do
        echo "  GPU $line"
    done
else
    echo "❌ NVIDIA驱动: 未安装或不可用"
fi

echo ""
echo "🐍 检查Python环境..."
echo "✓ Python版本: $(python --version)"

echo ""
echo "📦 检查关键依赖..."
python -c "
import sys
packages = ['torch', 'transformers', 'peft', 'deepspeed', 'datasets']
for pkg in packages:
    try:
        __import__(pkg)
        print(f'✓ {pkg}: 已安装')
    except ImportError:
        print(f'❌ {pkg}: 未安装')
"

echo ""
echo "🎯 快速训练命令示例:"
echo "文本模式: ./fast_train.sh"
echo "嵌入模式: 先修改fast_train.sh中的TRAINING_MODE=\"embedding\"和PROMPT_EMBEDDING_PATH"

echo ""
echo "📖 详细使用说明请查看: 训练使用指南.md"
