#!/bin/bash

# 切换到脚本所在目录
cd "$(dirname "$0")"

# P-tuning训练脚本：适用于Qwen2.5-1.5B模型的偏好学习
# 该脚本配置了P-tuning训练的所有重要参数

# 🆕 关键训练参数配置
NUM_VIRTUAL_TOKENS=10
PROMPT_INIT_METHOD="natural_language"
PREFERENCE_LOSS_WEIGHT=1.0
MARGIN=0.05
KL_LOSS_WEIGHT=0.1
LEARNING_RATE=1e-6
NUM_EPOCHS=10
BATCH_SIZE=2

# 🆕 模型保存和评估频率配置
SAVE_STEPS=2000           # 每多少步保存一次模型检查点（默认500步，原为200步）
EVAL_STEPS=2000           # 每多少步进行一次评估（默认500步，原为200步）

# 🆕 自动生成带参数信息的输出目录
# 格式：ptuning_[模型名称]_vtokens[数量]_init[方法]_kl[权重]_margin[值]_lr[学习率]_ep[轮次]_[时间戳]
MODEL_NAME="DeepSeek-R1-Distill-Qwen-1.5B"  # 基础模型名称
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 构建输出目录名称
OUTPUT_DIR="./ptuning_outputs/${MODEL_NAME}_vtokens${NUM_VIRTUAL_TOKENS}_init${PROMPT_INIT_METHOD}_kl${KL_LOSS_WEIGHT}_margin${MARGIN}_lr${LEARNING_RATE}_ep${NUM_EPOCHS}_bs${BATCH_SIZE}_${TIMESTAMP}"

# 🆕 显示配置信息
echo "🚀 P-tuning训练配置"
echo "================================"
echo "模型: $MODEL_NAME"
echo "虚拟token数量: $NUM_VIRTUAL_TOKENS"
echo "初始化方法: $PROMPT_INIT_METHOD"
echo "偏好损失权重: $PREFERENCE_LOSS_WEIGHT"
echo "边距参数: $MARGIN"
echo "KL散度权重: $KL_LOSS_WEIGHT"
echo "学习率: $LEARNING_RATE"
echo "训练轮次: $NUM_EPOCHS"
echo "批次大小: $BATCH_SIZE"
echo "保存频率: 每${SAVE_STEPS}步"
echo "评估频率: 每${EVAL_STEPS}步"
echo "输出目录: $OUTPUT_DIR"
echo "================================"
echo ""

# 🆕 确保输出目录存在
mkdir -p "$(dirname "$OUTPUT_DIR")"

# 🆕 保存训练配置到文件
CONFIG_FILE="${OUTPUT_DIR}_config.txt"
cat > "$CONFIG_FILE" << EOF
P-tuning训练配置
==========================================
训练时间: $(date '+%Y-%m-%d %H:%M:%S')
基础模型: $MODEL_NAME
数据集: /home/wangbinrui/research_projects/llama_rlhf/datasets/ultrafeedback_binarized/train_prefs_ultrafeedback_binarized.jsonl

核心参数:
- 虚拟token数量: $NUM_VIRTUAL_TOKENS
- 初始化方法: $PROMPT_INIT_METHOD
- 偏好损失权重: $PREFERENCE_LOSS_WEIGHT
- 边距参数: $MARGIN
- KL散度权重: $KL_LOSS_WEIGHT

训练参数:
- 学习率: $LEARNING_RATE
- 训练轮次: $NUM_EPOCHS
- 批次大小: $BATCH_SIZE
- 梯度累积步数: 8
- 最大序列长度: 512
- 保存频率: 每${SAVE_STEPS}步
- 评估频率: 每${EVAL_STEPS}步

早停参数:
- 耐心值: 5
- 阈值: 0.01
- 目标准确率: 0.9

输出目录: $OUTPUT_DIR
==========================================
EOF

echo "📝 配置信息已保存到: $CONFIG_FILE"
echo ""

# 🆕 询问用户确认
read -p "是否开始训练？(y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ 用户取消训练"
    exit 0
fi

echo "🚀 开始P-tuning训练..."
echo ""

# 执行训练
CUDA_VISIBLE_DEVICES=2, python train_ptuning.py \
    --model_name_or_path "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" \
    --dataset_name "/home/wangbinrui/research_projects/llama_rlhf/datasets/ultrafeedback_binarized/train_prefs_ultrafeedback_binarized.jsonl" \
    --num_virtual_tokens $NUM_VIRTUAL_TOKENS \
    --prompt_init_method $PROMPT_INIT_METHOD \
    --preference_loss_weight $PREFERENCE_LOSS_WEIGHT \
    --margin $MARGIN \
    --kl_loss_weight $KL_LOSS_WEIGHT \
    --save_steps $SAVE_STEPS \
    --eval_steps $EVAL_STEPS \
    --max_length 512 \
    --eval_split_ratio 0.1 \
    --early_stopping_patience 5 \
    --early_stopping_threshold 0.01 \
    --target_accuracy 0.9 \
    --wandb_project "ptuning-preference-learning" \
    --use_wandb True \
    --output_dir "$OUTPUT_DIR" \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs $NUM_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --warmup_steps 100 \
    --logging_steps 50 \
    --save_total_limit 10 \
    --remove_unused_columns False \
    --dataloader_pin_memory False \
    --fp16 \
    --dataloader_num_workers 0

# 🆕 检查训练结果
TRAINING_EXIT_CODE=$?

echo ""
echo "================================"
if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "✅ 训练完成！"
    echo "📁 输出目录: $OUTPUT_DIR"
    echo "📁 配置文件: $CONFIG_FILE"
    
    # 🆕 显示输出文件
    if [ -d "$OUTPUT_DIR" ]; then
        echo ""
        echo "📋 输出文件列表:"
        ls -la "$OUTPUT_DIR"
        
        # 🆕 显示prompt embeddings文件信息
        PROMPT_EMBEDDINGS_FILE="$OUTPUT_DIR/prompt_embeddings.pt"
        if [ -f "$PROMPT_EMBEDDINGS_FILE" ]; then
            FILE_SIZE=$(du -h "$PROMPT_EMBEDDINGS_FILE" | cut -f1)
            echo ""
            echo "🎯 Prompt Embeddings信息:"
            echo "   文件: $PROMPT_EMBEDDINGS_FILE"
            echo "   大小: $FILE_SIZE"
            echo "   参数: ${NUM_VIRTUAL_TOKENS}个虚拟token，初始化方法=${PROMPT_INIT_METHOD}"
        fi
        
        # 🆕 创建符号链接到最新训练结果
        LATEST_LINK="./ptuning_outputs/latest_${MODEL_NAME}"
        if [ -L "$LATEST_LINK" ]; then
            rm "$LATEST_LINK"
        fi
        ln -s "$(basename "$OUTPUT_DIR")" "$LATEST_LINK"
        echo "🔗 最新结果链接: $LATEST_LINK -> $(basename "$OUTPUT_DIR")"
    fi
    
    # 🆕 提供推理测试建议
    echo ""
    echo "💡 下一步建议:"
    echo "   测试推理: ./test_ptuning_inference.sh basic --prompt_embeddings \"$OUTPUT_DIR/prompt_embeddings.pt\" --config \"$OUTPUT_DIR/ptuning_config.json\""
    echo "   对比测试: ./test_ptuning_inference.sh compare --prompt_embeddings \"$OUTPUT_DIR/prompt_embeddings.pt\""
else
    echo "❌ 训练失败！退出码: $TRAINING_EXIT_CODE"
    echo "📁 检查日志: $OUTPUT_DIR"
    echo "📁 配置文件: $CONFIG_FILE"
    exit $TRAINING_EXIT_CODE
fi

echo "================================"
