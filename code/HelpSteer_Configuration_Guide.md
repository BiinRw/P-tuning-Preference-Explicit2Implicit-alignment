# 🎯 HelpSteer训练配置适配说明

## 📋 配置适配完成项目

### ✅ 已完成的适配

1. **模型名称自动提取**: 从 `POLICY_MODEL_PATH` 自动提取简短模型名称
2. **训练模式识别**: 自动识别 text/embedding 模式
3. **Loss函数配置**: 支持可配置的loss函数
4. **运行名称生成**: 自动生成结构化的运行名称
5. **输出目录管理**: 层次化的输出目录结构

### 🔧 配置参数

#### 在 `train_helpsteer.sh` 中可配置的参数：

```bash
# 🎛️ 训练模式
TRAINING_MODE="text"  # 或 "embedding"

# 🏗️ 模型配置
POLICY_MODEL_PATH="Qwen/Qwen2.5-1.5B-Instruct"
MODEL_SHORT_NAME="Qwen2.5-1.5B"  # 用于命名

# 🎯 Loss函数配置
LOSS_NAME="new_pref_po"  # 可选: "dpo", "ipo", "new_pref_po", "sipa"

# ⚙️ 训练超参数
BETA=0.05
ALPHA=0.1
```

### 📁 自动生成的名称结构

#### 运行名称格式：
```
{Dataset}-{Mode}-{Model}-{Loss}-beta{Beta}-alpha{Alpha}
```

**示例**：
- HelpSteer数据集 + 文本模式 + Qwen2.5-1.5B + new_pref_po：
  `HelpSteer-Text-Qwen2.5-1.5B-new_pref_po-beta0.05-alpha0.1`

#### Wandb项目名称格式：
```
HelpSteer_{Loss}_{Mode}
```

**示例**：
- `HelpSteer_new_pref_po_Text`
- `HelpSteer_dpo_Emb`

#### 输出目录结构：
```
./model_output/{WandbProject}/{RunName}/
```

**示例**：
```
./model_output/HelpSteer_new_pref_po_Text/HelpSteer-Text-Qwen2.5-1.5B-new_pref_po-beta0.05-alpha0.1/
```

### 🎮 使用方法

#### 1. 文本指令模式
```bash
# 编辑 train_helpsteer.sh
TRAINING_MODE="text"
LOSS_NAME="new_pref_po"
MODEL_SHORT_NAME="Qwen2.5-1.5B"

# 运行训练
./train_helpsteer.sh
```

#### 2. 嵌入向量模式  
```bash
# 编辑 train_helpsteer.sh
TRAINING_MODE="embedding"
PROMPT_EMBEDDING_PATH="/path/to/embeddings.pt"
LOSS_NAME="dpo"
MODEL_SHORT_NAME="DeepSeek-R1-Qwen1.5B"

# 运行训练
./train_helpsteer.sh
```

#### 3. 不同Loss函数
```bash
# DPO训练
LOSS_NAME="dpo"

# IPO训练  
LOSS_NAME="ipo"

# SIPA训练
LOSS_NAME="sipa"

# 偏好引导DPO (默认)
LOSS_NAME="new_pref_po"
```

### 📊 生成的文件结构示例

```
model_output/
├── HelpSteer_new_pref_po_Text/
│   └── HelpSteer-Text-Qwen2.5-1.5B-new_pref_po-beta0.05-alpha0.1/
│       ├── config.json
│       ├── pytorch_model.bin
│       ├── adapter_config.json
│       └── adapter_model.bin
├── HelpSteer_dpo_Emb/
│   └── HelpSteer-Emb-DeepSeek-R1-Qwen1.5B-dpo-beta0.1-alpha0.2/
│       └── ...
└── ...
```

### 🔄 与原fast_train.sh的兼容性

所有适配都保持与原有training pipeline的完整兼容性：
- ✅ DeepSpeed配置不变
- ✅ 数据加载逻辑不变  
- ✅ 训练参数传递不变
- ✅ 支持所有原有功能

### 🎯 关键改进

1. **智能命名**: 文件名包含完整的训练配置信息
2. **参数化配置**: 所有关键参数都可在.sh文件中配置
3. **层次化存储**: 清晰的目录结构便于管理
4. **自动检测**: 自动识别数据集类型和模型名称
5. **完整适配**: .sh配置与Python训练脚本完全同步

现在您可以通过修改 `train_helpsteer.sh` 中的参数来控制所有的训练配置和输出命名！
