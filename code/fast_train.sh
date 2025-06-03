#!/bin/bash

# 🎯 偏好引导的DPO训练脚本
# 支持文本指令和嵌入向量两种模式
# 作者: 王斌睿
# 更新日期: 2025-05-29

echo "=================================================="
echo "🚀 开始偏好引导DPO训练 (Preference-Guided DPO Training)"
echo "=================================================="

# 🔧 设置环境变量
echo "🔧 配置训练环境..."
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_OFFLINE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false

# 🎯 优化PyTorch性能和避免trace cache失效
export TORCH_COMPILE_DEBUG=0
export TORCH_LOGS=""
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
# 禁用不必要的CUDA优化以减少trace cache失效
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDNN_V8_API_ENABLED=1

echo "✅ 环境变量设置完成"

# 📁 训练配置参数
echo ""
echo "📁 设置训练配置参数..."

# 🖥️ GPU节点配置
GPU_NODES="localhost:1,2"  # 可根据实际GPU配置修改，如: "localhost:0,1,2,3" 或 "node1:0,1,node2:0,1"

# 🎛️ 训练模式选择 (二选一)
# MODE 1: 使用文本指令模式
TRAINING_MODE="embedding"  # 可选: "text" 或 "embedding"

# MODE 2: 使用嵌入向量模式 (如果有预训练的prompt embedding文件)
# TRAINING_MODE="embedding"
PROMPT_EMBEDDING_PATH="/home/wangbinrui/research_projects/llama_rlhf/code/Ptuning/ptuning_outputs/qwen2.5-1.5b_vtokens10_initnatural_language_kl0.1_margin0.05_lr1e-5_ep10_bs2_20250528_133602/checkpoint-32600/prompt_embeddings.pt"

# 🏗️ 模型路径配置
POLICY_MODEL_PATH="Qwen/Qwen2.5-1.5B-Instruct"
REFERENCE_MODEL_PATH="Qwen/Qwen2.5-1.5B-Instruct"

# 📊 数据集路径配置
TRAIN_DATASET_PATH="/home/wangbinrui/research_projects/llama_rlhf/datasets/ultrafeedback_binarized/train_prefs_ultrafeedback_binarized.jsonl"
TEST_DATASET_PATH="/home/wangbinrui/research_projects/llama_rlhf/datasets/ultrafeedback_binarized/test_prefs_ultrafeedback_binarized.jsonl"

# 💬 偏好文本指令 (仅在文本模式下使用)
PREFERENCE_TEXT="Please provide a helpful, honest, harmless, and concise response."

# ⚙️ 训练超参数
BETA=0.05              # DPO损失的温度参数
ALPHA=0.1              # 偏好一致性损失权重
LEARNING_RATE=5e-4     # 学习率
NUM_EPOCHS=1           # 训练轮数
GRADIENT_ACCUM_STEPS=512  # 梯度累积步数
MAX_LENGTH=300         # 最大序列长度
MAX_PROMPT_LENGTH=128  # 最大提示长度

# 🔗 LoRA配置
LORA_R=16              # LoRA秩
LORA_ALPHA=32          # LoRA缩放参数
LORA_DROPOUT=0.1       # LoRA dropout率

# 📝 输出和日志配置
OUTPUT_DIR="./model_output/Preference_Guided_Ptuning"
WANDB_PROJECT="Preference_Guided_Ptuning"

echo "✅ 参数配置完成"

# 🔍 数据集验证
echo ""
echo "🔍 验证数据集文件..."
if [ ! -f "$TRAIN_DATASET_PATH" ]; then
    echo "❌ 错误: 训练数据集文件不存在: $TRAIN_DATASET_PATH"
    exit 1
fi

if [ ! -f "$TEST_DATASET_PATH" ]; then
    echo "❌ 错误: 测试数据集文件不存在: $TEST_DATASET_PATH"
    exit 1
fi

# 统计数据集样本数量
TRAIN_SAMPLES=$(wc -l < "$TRAIN_DATASET_PATH")
TEST_SAMPLES=$(wc -l < "$TEST_DATASET_PATH")
echo "📊 训练样本数量: $TRAIN_SAMPLES"
echo "📊 测试样本数量: $TEST_SAMPLES"
echo "✅ 数据集验证通过"

# 🗂️ 创建输出目录
echo ""
echo "🗂️ 创建输出目录..."
mkdir -p "$OUTPUT_DIR"
mkdir -p "./logs/$WANDB_PROJECT"
echo "✅ 输出目录创建完成"

# 🧠 构建训练命令
echo ""
echo "🧠 构建训练命令..."

# 🚀 DeepSpeed配置
DEEPSPEED_CMD="CUDA_ALLOC_CONF=expandable_segments deepspeed --include=$GPU_NODES"
TRAIN_SCRIPT="train_with_preference_prompt.py"

# 📝 添加基础参数
ARGS="--policy-model-path $POLICY_MODEL_PATH"
ARGS="$ARGS --reference-model-path $REFERENCE_MODEL_PATH"
ARGS="$ARGS --dataset-path $TRAIN_DATASET_PATH"
ARGS="$ARGS --test-dataset-path $TEST_DATASET_PATH"
ARGS="$ARGS --beta $BETA"
ARGS="$ARGS --alpha $ALPHA"
ARGS="$ARGS --learning-rate $LEARNING_RATE"
ARGS="$ARGS --num-train-epochs $NUM_EPOCHS"
ARGS="$ARGS --gradient-accumulation-steps $GRADIENT_ACCUM_STEPS"
ARGS="$ARGS --max-length $MAX_LENGTH"
ARGS="$ARGS --max-prompt-length $MAX_PROMPT_LENGTH"
ARGS="$ARGS --lora-r $LORA_R"
ARGS="$ARGS --lora-alpha $LORA_ALPHA"
ARGS="$ARGS --lora-dropout $LORA_DROPOUT"
ARGS="$ARGS --output-dir $OUTPUT_DIR"
ARGS="$ARGS --wandb-project $WANDB_PROJECT"

# 🎯 根据训练模式添加特定参数
if [ "$TRAINING_MODE" = "embedding" ]; then
    echo "🎯 训练模式: 嵌入向量模式"
    if [ -z "$PROMPT_EMBEDDING_PATH" ]; then
        echo "❌ 错误: 嵌入向量模式需要设置 PROMPT_EMBEDDING_PATH"
        exit 1
    fi
    if [ ! -f "$PROMPT_EMBEDDING_PATH" ]; then
        echo "❌ 错误: 嵌入向量文件不存在: $PROMPT_EMBEDDING_PATH"
        exit 1
    fi
    ARGS="$ARGS --use-prompt-embedding --prompt-embedding-path $PROMPT_EMBEDDING_PATH"
    echo "📁 嵌入向量文件: $PROMPT_EMBEDDING_PATH"
    echo "📏 嵌入向量文件大小: $(du -h "$PROMPT_EMBEDDING_PATH" | cut -f1)"
elif [ "$TRAINING_MODE" = "text" ]; then
    echo "🎯 训练模式: 文本指令模式"
    ARGS="$ARGS --preference-text \"$PREFERENCE_TEXT\""
    echo "💬 偏好指令: $PREFERENCE_TEXT"
else
    echo "❌ 错误: 无效的训练模式: $TRAINING_MODE (应为 'text' 或 'embedding')"
    exit 1
fi

# 🔥 构建完整的DeepSpeed训练命令
FULL_CMD="$DEEPSPEED_CMD $TRAIN_SCRIPT $ARGS"

echo "✅ 训练命令构建完成"

# 📋 打印完整配置信息
echo ""
echo "=================================================="
echo "📋 完整训练配置"
echo "=================================================="
echo "🎯 训练模式: $TRAINING_MODE"
echo "🏗️ 策略模型: $POLICY_MODEL_PATH"
echo "🏗️ 参考模型: $REFERENCE_MODEL_PATH"
echo "📊 训练数据: $TRAIN_DATASET_PATH ($TRAIN_SAMPLES 样本)"
echo "📊 测试数据: $TEST_DATASET_PATH ($TEST_SAMPLES 样本)"
if [ "$TRAINING_MODE" = "text" ]; then
    echo "💬 偏好指令: $PREFERENCE_TEXT"
else
    echo "📁 嵌入向量: $PROMPT_EMBEDDING_PATH"
fi
echo "⚙️ Beta: $BETA, Alpha: $ALPHA"
echo "⚙️ 学习率: $LEARNING_RATE, 训练轮数: $NUM_EPOCHS"
echo "⚙️ LoRA配置: r=$LORA_R, alpha=$LORA_ALPHA, dropout=$LORA_DROPOUT"
echo "🔥 DeepSpeed节点: $GPU_NODES"
echo "📝 输出目录: $OUTPUT_DIR"
echo "=================================================="

# 🚦 最终确认
echo ""
echo "🚦 即将开始训练，请确认配置无误..."
echo "🚀 DeepSpeed训练命令:"
echo "   $FULL_CMD"
echo ""
read -p "是否继续训练？(y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ 训练已取消"
    exit 1
fi

# 🚀 开始训练
echo ""
echo "🚀 开始训练..."
echo "=================================================="

# 记录开始时间
START_TIME=$(date +%s)
echo "⏰ 训练开始时间: $(date)"

# 🔥 执行DeepSpeed训练
echo "🔥 启动DeepSpeed分布式训练进程..."
echo "🖥️  使用GPU节点: $GPU_NODES"
echo "⚡ 扩展内存段配置已启用"
echo ""

# 执行训练命令
eval $FULL_CMD

# 检查训练结果
TRAIN_EXIT_CODE=$?
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo ""
echo "=================================================="
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "✅ 训练成功完成！"
    echo "🏆 模型已保存到: $OUTPUT_DIR"
    echo "📊 日志文件位置: ./logs/$WANDB_PROJECT"
    echo "⏰ 总训练时间: ${HOURS}小时 ${MINUTES}分钟 ${SECONDS}秒"
    
    # 检查输出文件
    echo ""
    echo "📁 检查输出文件..."
    if [ -d "$OUTPUT_DIR" ]; then
        echo "📂 输出目录内容:"
        ls -la "$OUTPUT_DIR"
    fi
    
    echo ""
    echo "🎉 训练流程全部完成！"
else
    echo "❌ 训练失败，退出码: $TRAIN_EXIT_CODE"
    echo "⏰ 训练持续时间: ${HOURS}小时 ${MINUTES}分钟 ${SECONDS}秒"
    echo "🔍 请检查上方的错误信息"
fi
echo "=================================================="
