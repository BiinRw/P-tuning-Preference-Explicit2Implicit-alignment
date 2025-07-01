import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AdamW
import torch.nn.functional as F

# P-tuning训练配置参数
MODEL_NAME = "gpt2-xl"  # 基础模型名称
PROMPT_LEN  = 50        # 软提示的长度（虚拟token数量）
LR          = 5e-4      # 学习率
EPOCHS      = 3         # 训练轮数
BETA        = 1.0       # DPO损失中的温度参数
BATCH_SIZE  = 16        # 批次大小

# 1. 加载模型和分词器，并冻结基础模型参数
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# 冻结基础模型的所有参数，只训练软提示嵌入
for p in model.parameters():
    p.requires_grad = False
model.eval()  # 设置为评估模式

# 2. 创建可学习的软提示嵌入
# 获取模型的嵌入维度
embed_size = model.transformer.wte.weight.shape[1]
# 初始化软提示参数：这些是可训练的连续向量，代替离散的token
prompt_emb = nn.Parameter(torch.randn(PROMPT_LEN, embed_size) * 0.02)

# 创建优化器，只优化软提示参数
optimizer = AdamW([prompt_emb], lr=LR)

# 数据格式说明：
# train_dataloader 每个 batch 返回字典:
# {
#   "input_ids":      [2*B, L],     # 输入token ID
#   "attention_mask": [2*B, L],     # 注意力掩码
#   "labels":         [2*B, L]      # 标签
# }
# 前 B 条是 (prompt + x + y_w)，后 B 条是 (prompt + x + y_l)
# 其中 y_w 是偏好回复，y_l 是非偏好回复

def compute_dpo_loss(logits, labels):
    """
    计算DPO（Direct Preference Optimization）损失
    
    Args:
        logits: 模型输出的logits [2B, seq_len, vocab_size]
        labels: 目标标签 [2B, seq_len]
    
    Returns:
        DPO损失值
    """
    # 计算每个token的对数概率
    logps = F.log_softmax(logits, dim=-1)
    
    # 收集目标token的对数概率并求和，得到每个序列的总对数概率
    seq_logps = torch.gather(logps, 2, labels.unsqueeze(-1)).squeeze(-1).sum(dim=1)
    
    # 分离偏好和非偏好回复的对数概率
    B = seq_logps.size(0) // 2
    logp_w = seq_logps[:B]    # 偏好回复的对数概率
    logp_l = seq_logps[B:]    # 非偏好回复的对数概率
    
    # 计算DPO损失：鼓励偏好回复有更高的概率
    diff  = BETA * (logp_w - logp_l)
    return -torch.log(torch.sigmoid(diff)).mean()

# 3. 训练循环
for epoch in range(EPOCHS):
    for batch in train_dataloader:
        # 获取输入的原始嵌入表示
        emb = model.transformer.wte(batch["input_ids"])  # [2B, L, d]
        B, L, _ = emb.shape
        
        # 4. 拼接软提示嵌入到输入序列前面
        # 扩展软提示嵌入到批次大小
        prompt = prompt_emb.unsqueeze(0).expand(B, -1, -1)  # [2B, m, d]
        # 将软提示和原始输入嵌入拼接
        inp_emb = torch.cat([prompt, emb], dim=1)           # [2B, m+L, d]
        
        # 调整注意力掩码，为软提示部分添加注意力
        attn_mask = torch.cat([
            torch.ones(B, PROMPT_LEN, device=emb.device),  # 软提示部分全为1
            batch["attention_mask"]                         # 原始注意力掩码
        ], dim=1)
        
        # 调整标签，软提示部分的标签设为-100（不计算损失）
        lbls = torch.cat([
            torch.full((B, PROMPT_LEN), -100, dtype=torch.long, device=emb.device),
            batch["labels"]
        ], dim=1)
        
        # 5. 前向传播并计算损失
        outputs = model(inputs_embeds=inp_emb, attention_mask=attn_mask)
        logits  = outputs.logits  # [2B, m+L, V]
        
        # 只对非软提示部分计算DPO损失
        loss = compute_dpo_loss(logits[:, PROMPT_LEN:], lbls[:, PROMPT_LEN:])
        
        # 6. 反向传播和参数更新
        optimizer.zero_grad()  # 清零梯度
        loss.backward()        # 反向传播
        optimizer.step()       # 更新参数
    
    print(f"Epoch {epoch+1} loss {loss.item():.4f}")

# 7. 保存训练好的软提示向量
# 这些向量包含了学习到的偏好信息，可以在推理时使用
torch.save(prompt_emb.detach().cpu(), "soft_prompt.pt")
