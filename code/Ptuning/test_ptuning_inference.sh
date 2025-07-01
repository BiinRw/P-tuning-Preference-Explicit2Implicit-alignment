#!/bin/bash

# P-tuning推理测试脚本
# 支持三种测试模式：基础测试、数据集测试、模型对比测试

# 切换到脚本所在目录
cd "$(dirname "$0")"

# 默认配置
BASE_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
PROMPT_EMBEDDINGS="./ptuning_output/prompt_embeddings.pt"
CONFIG_FILE="./ptuning_output/ptuning_config.json"
TEST_DATA="/home/wangbinrui/research_projects/llama_rlhf/datasets/ultrafeedback_binarized/test_prefs_ultrafeedback_binarized.jsonl"
GPU_ID=3

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# 检查文件是否存在
check_file() {
    if [ ! -f "$1" ]; then
        print_error "File not found: $1"
        return 1
    fi
    return 0
}

# 显示帮助信息
show_help() {
    echo "P-tuning推理测试脚本"
    echo ""
    echo "用法: $0 [测试模式] [选项]"
    echo ""
    echo "测试模式:"
    echo "  basic        基础测试 - 使用预定义的测试样本"
    echo "  dataset      数据集测试 - 使用指定数据集进行测试"
    echo "  compare      对比测试 - 比较基础模型和P-tuning模型的输出"
    echo "  all          运行所有测试模式"
    echo ""
    echo "选项:"
    echo "  --base_model PATH           基础模型路径 (默认: $BASE_MODEL)"
    echo "  --prompt_embeddings PATH    训练好的prompt embeddings路径 (默认: $PROMPT_EMBEDDINGS)"
    echo "  --config PATH               P-tuning配置文件路径 (默认: $CONFIG_FILE)"
    echo "  --test_data PATH            测试数据集路径 (默认: $TEST_DATA)"
    echo "  --num_samples N             测试样本数量 (默认: 5)"
    echo "  --max_length N              最大序列总长度 (默认: 2048，让模型自然结束)"  # 🚨 更新帮助文本
    echo "  --temperature F             采样温度 (默认: 0.7)"
    echo "  --gpu_id N                  GPU设备ID (默认: $GPU_ID)"
    echo "  -h, --help                  显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 basic                                    # 基础测试"
    echo "  $0 dataset --num_samples 10                 # 数据集测试，10个样本"
    echo "  $0 compare --num_samples 3                  # 对比测试，3个样本"
    echo "  $0 all                                      # 运行所有测试"
}

# 解析命令行参数
parse_args() {
    TEST_MODE=""
    NUM_SAMPLES=5
    MAX_LENGTH=2048  # 🚨 改为max_length
    TEMPERATURE=0.7
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            basic|dataset|compare|all)
                TEST_MODE="$1"
                shift
                ;;
            --base_model)
                BASE_MODEL="$2"
                shift 2
                ;;
            --prompt_embeddings)
                PROMPT_EMBEDDINGS="$2"
                shift 2
                ;;
            --config)
                CONFIG_FILE="$2"
                shift 2
                ;;
            --test_data)
                TEST_DATA="$2"
                shift 2
                ;;
            --num_samples)
                NUM_SAMPLES="$2"
                shift 2
                ;;
            --max_length)  # 🚨 改为max_length
                MAX_LENGTH="$2"
                shift 2
                ;;
            --temperature)
                TEMPERATURE="$2"
                shift 2
                ;;
            --gpu_id)
                GPU_ID="$2"
                shift 2
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                print_error "未知参数: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    if [ -z "$TEST_MODE" ]; then
        print_error "请指定测试模式"
        show_help
        exit 1
    fi
}

# 检查必要文件
check_requirements() {
    print_info "检查必要文件..."
    
    # 检查Python脚本
    if ! check_file "inference_ptuning.py"; then
        exit 1
    fi
    
    # 检查prompt embeddings
    if ! check_file "$PROMPT_EMBEDDINGS"; then
        print_error "Prompt embeddings文件不存在: $PROMPT_EMBEDDINGS"
        print_info "请先运行训练脚本生成prompt embeddings"
        exit 1
    fi
    
    # 检查配置文件（可选）
    if [ -n "$CONFIG_FILE" ] && [ ! -f "$CONFIG_FILE" ]; then
        print_warning "配置文件不存在: $CONFIG_FILE，将使用默认配置"
        CONFIG_FILE=""
    fi
    
    # 🚨 修复：对于compare模式，测试数据是可选的
    # 如果没有测试数据，会使用预定义的测试样本
    if [[ "$TEST_MODE" == "dataset" ]]; then
        # 只有dataset模式才强制要求测试数据
        if ! check_file "$TEST_DATA"; then
            print_error "测试数据文件不存在: $TEST_DATA"
            exit 1
        fi
    elif [[ "$TEST_MODE" == "compare" || "$TEST_MODE" == "all" ]]; then
        # compare模式检查测试数据，如果不存在给出警告但不退出
        if [ ! -f "$TEST_DATA" ]; then
            print_warning "测试数据文件不存在: $TEST_DATA"
            print_info "将使用预定义的测试样本进行对比"
            TEST_DATA=""  # 清空以避免传入无效路径
        fi
    fi
    
    print_success "文件检查完成"
}

# 基础测试 - 使用预定义样本
run_basic_test() {
    print_info "🧪 运行基础测试..."
    print_info "使用预定义的测试样本进行P-tuning模型推理（自然结束模式）"
    
    local cmd="CUDA_VISIBLE_DEVICES=$GPU_ID python inference_ptuning.py"
    cmd="$cmd --base_model \"$BASE_MODEL\""
    cmd="$cmd --prompt_embeddings \"$PROMPT_EMBEDDINGS\""
    
    if [ -n "$CONFIG_FILE" ]; then
        cmd="$cmd --config \"$CONFIG_FILE\""
    fi
    
    cmd="$cmd --num_samples $NUM_SAMPLES"
    cmd="$cmd --max_length $MAX_LENGTH"  # 🚨 改为max_length
    cmd="$cmd --temperature $TEMPERATURE"
    
    print_info "执行命令: $cmd"
    eval $cmd
    
    if [ $? -eq 0 ]; then
        print_success "基础测试完成"
        print_info "结果已保存到: ptuning_inference_results.json"
    else
        print_error "基础测试失败"
        return 1
    fi
}

# 数据集测试 - 使用真实数据集
run_dataset_test() {
    print_info "📊 运行数据集测试..."
    print_info "使用真实数据集进行P-tuning模型推理（自然结束模式）"
    
    local cmd="CUDA_VISIBLE_DEVICES=$GPU_ID python inference_ptuning.py"
    cmd="$cmd --base_model \"$BASE_MODEL\""
    cmd="$cmd --prompt_embeddings \"$PROMPT_EMBEDDINGS\""
    
    if [ -n "$CONFIG_FILE" ]; then
        cmd="$cmd --config \"$CONFIG_FILE\""
    fi
    
    cmd="$cmd --test_data \"$TEST_DATA\""
    cmd="$cmd --num_samples $NUM_SAMPLES"
    cmd="$cmd --max_length $MAX_LENGTH"  # 🚨 改为max_length
    cmd="$cmd --temperature $TEMPERATURE"
    
    print_info "执行命令: $cmd"
    eval $cmd
    
    if [ $? -eq 0 ]; then
        print_success "数据集测试完成"
        print_info "结果已保存到: ptuning_inference_results.json"
    else
        print_error "数据集测试失败"
        return 1
    fi
}

# 对比测试 - 比较基础模型和P-tuning模型
run_compare_test() {
    print_info "🔍 运行对比测试..."
    print_info "比较基础模型和P-tuning模型的输出差异（自然结束模式）"
    
    local cmd="CUDA_VISIBLE_DEVICES=$GPU_ID python inference_ptuning.py"
    cmd="$cmd --base_model \"$BASE_MODEL\""
    cmd="$cmd --prompt_embeddings \"$PROMPT_EMBEDDINGS\""
    
    if [ -n "$CONFIG_FILE" ]; then
        cmd="$cmd --config \"$CONFIG_FILE\""
    fi
    
    if [ -n "$TEST_DATA" ] && [ -f "$TEST_DATA" ]; then
        cmd="$cmd --test_data \"$TEST_DATA\""
    fi
    
    cmd="$cmd --num_samples $NUM_SAMPLES"
    cmd="$cmd --max_length $MAX_LENGTH"  # 🚨 改为max_length
    cmd="$cmd --temperature $TEMPERATURE"
    cmd="$cmd --compare"
    
    print_info "执行命令: $cmd"
    eval $cmd
    
    if [ $? -eq 0 ]; then
        print_success "对比测试完成"
        print_info "对比结果已显示在控制台并保存到文件"
    else
        print_error "对比测试失败"
        return 1
    fi
}

# 运行所有测试
run_all_tests() {
    print_info "🚀 运行所有测试模式..."
    
    echo ""
    echo "========================================"
    echo "1/3 基础测试"
    echo "========================================"
    run_basic_test
    if [ $? -ne 0 ]; then
        print_error "基础测试失败，停止后续测试"
        return 1
    fi
    
    echo ""
    echo "========================================"
    echo "2/3 数据集测试"
    echo "========================================"
    run_dataset_test
    if [ $? -ne 0 ]; then
        print_error "数据集测试失败，停止后续测试"
        return 1
    fi
    
    echo ""
    echo "========================================"
    echo "3/3 对比测试"
    echo "========================================"
    run_compare_test
    if [ $? -ne 0 ]; then
        print_error "对比测试失败"
        return 1
    fi
    
    print_success "所有测试完成！"
}

# 显示配置信息
show_config() {
    print_info "当前配置:"
    echo "  基础模型: $BASE_MODEL"
    echo "  Prompt embeddings: $PROMPT_EMBEDDINGS"
    echo "  配置文件: ${CONFIG_FILE:-"使用默认配置"}"
    echo "  测试数据: $TEST_DATA"
    echo "  测试样本数: $NUM_SAMPLES"
    echo "  最大序列长度: $MAX_LENGTH (自然结束模式)"  # 🚨 更新描述
    echo "  采样温度: $TEMPERATURE"
    echo "  GPU设备: $GPU_ID"
    echo ""
}

# 主函数
main() {
    print_info "🚀 P-tuning推理测试脚本"
    echo "========================================"
    
    # 解析参数
    parse_args "$@"
    
    # 显示配置
    show_config
    
    # 检查必要文件
    check_requirements
    
    echo ""
    print_info "开始执行测试模式: $TEST_MODE"
    echo "========================================"
    
    # 根据测试模式执行相应测试
    case $TEST_MODE in
        basic)
            run_basic_test
            ;;
        dataset)
            run_dataset_test
            ;;
        compare)
            run_compare_test
            ;;
        all)
            run_all_tests
            ;;
        *)
            print_error "未知测试模式: $TEST_MODE"
            exit 1
            ;;
    esac
    
    if [ $? -eq 0 ]; then
        echo ""
        print_success "测试执行完成！"
        print_info "检查输出结果和日志信息"
    else
        echo ""
        print_error "测试执行失败！"
        exit 1
    fi
}

# 运行主函数
main "$@"
