#!/bin/bash

# 🧪 DeepSpeed命令测试脚本
echo "🧪 测试DeepSpeed训练命令构建..."

cd /home/wangbinrui/research_projects/llama_rlhf/code

# 模拟fast_train.sh中的关键变量
TRAINING_MODE="embedding"
PROMPT_EMBEDDING_PATH="/home/wangbinrui/research_projects/llama_rlhf/code/Ptuning/ptuning_outputs/qwen2.5-1.5b_vtokens10_initnatural_language_kl0.1_margin0.05_lr1e-5_ep10_bs2_20250528_133602/checkpoint-32600/prompt_embeddings.pt"

POLICY_MODEL_PATH="Qwen/Qwen2.5-1.5B-Instruct"
REFERENCE_MODEL_PATH="Qwen/Qwen2.5-1.5B-Instruct"
TRAIN_DATASET_PATH="/home/wangbinrui/research_projects/llama_rlhf/datasets/ultrafeedback_binarized/train_prefs_ultrafeedback_binarized.jsonl"
TEST_DATASET_PATH="/home/wangbinrui/research_projects/llama_rlhf/datasets/ultrafeedback_binarized/test_prefs_ultrafeedback_binarized.jsonl"

BETA=0.05
ALPHA=0.1
LEARNING_RATE=5e-4
NUM_EPOCHS=1
GRADIENT_ACCUM_STEPS=512
MAX_LENGTH=300
MAX_PROMPT_LENGTH=128
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.1
OUTPUT_DIR="./model_output/Preference_Guided_Ptuning"
WANDB_PROJECT="Preference_Guided_Ptuning"

# 🖥️ GPU节点配置 (可根据实际情况修改)
GPU_NODES="localhost:1,2"  # 可修改为: "localhost:0,1", "localhost:2,3", "localhost:0,1,2,3" 等

# 构建命令 (与fast_train.sh相同的逻辑)
DEEPSPEED_CMD="CUDA_ALLOC_CONF=expandable_segments deepspeed --include=$GPU_NODES"
TRAIN_SCRIPT="train_with_preference_prompt.py"

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

# 根据训练模式添加参数
if [ "$TRAINING_MODE" = "embedding" ]; then
    ARGS="$ARGS --use-prompt-embedding --prompt-embedding-path $PROMPT_EMBEDDING_PATH"
fi

FULL_CMD="$DEEPSPEED_CMD $TRAIN_SCRIPT $ARGS"

echo ""
echo "🔍 生成的DeepSpeed训练命令:"
echo "=================================================="
echo "$FULL_CMD"
echo "=================================================="

echo ""
echo "📋 命令分解:"
echo "🔥 DeepSpeed部分: $DEEPSPEED_CMD"
echo "📄 训练脚本: $TRAIN_SCRIPT"
echo "⚙️  训练参数: $ARGS"

echo ""
echo "🧪 测试--help输出 (仅验证参数解析)..."
echo "python $TRAIN_SCRIPT --help"

echo ""
echo "✅ DeepSpeed命令构建测试完成!"
echo "📝 您可以复制上述命令手动执行，或直接运行 ./fast_train.sh"
