import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from typing import Optional, Dict, Any
import os


class PTuningModel(nn.Module):
    """
    P-tuning模型包装器，为基础模型添加可学习的提示嵌入
    
    P-tuning是一种参数高效的微调方法，通过在输入序列前添加可训练的连续提示向量，
    而不是微调整个模型，来适应特定任务。这种方法特别适用于偏好学习任务。
    """
    
    def __init__(
        self,
        base_model: PreTrainedModel,
        num_virtual_tokens: int = 50,
        prompt_embedding_dim: Optional[int] = None,
        init_method: str = "random",
        margin: float = 0.1,
        natural_prompts: Optional[list] = None,
        cluster_centers: Optional[torch.Tensor] = None
    ):
        """
        初始化P-tuning模型
        
        Args:
            base_model: 基础预训练模型
            num_virtual_tokens: 虚拟token数量（软提示长度）
            prompt_embedding_dim: 提示嵌入维度，如果为None则使用模型的隐藏维度
            init_method: 初始化方法，"random"、"vocab"、"natural_language"或"cluster_center"
            margin: 偏好损失的边距参数
            natural_prompts: 自然语言提示列表（用于natural_language初始化）
            cluster_centers: 聚类中心张量（用于cluster_center初始化）
        """
        super().__init__()
        self.base_model = base_model
        self.num_virtual_tokens = num_virtual_tokens
        self.margin = margin
        
        # 从基础模型获取嵌入维度
        if prompt_embedding_dim is None:
            prompt_embedding_dim = base_model.config.hidden_size
        
        self.prompt_embedding_dim = prompt_embedding_dim
        
        # 初始化提示嵌入层：这是P-tuning的核心组件
        self.prompt_embeddings = nn.Embedding(num_virtual_tokens, prompt_embedding_dim)
        self._init_prompt_embeddings(init_method, natural_prompts, cluster_centers)
        
        # 冻结基础模型参数，只训练提示嵌入
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # 确保prompt embeddings可以训练
        for param in self.prompt_embeddings.parameters():
            param.requires_grad = True
        
        # 验证参数冻结状态
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"🔒 Frozen parameters: {total_params - trainable_params:,}")
        print(f"🔥 Trainable parameters (prompt embeddings): {trainable_params:,}")
        print(f"📊 Trainable ratio: {trainable_params/total_params*100:.4f}%")
    
    def _init_prompt_embeddings(
        self, 
        init_method: str, 
        natural_prompts: Optional[list] = None,
        cluster_centers: Optional[torch.Tensor] = None
    ):
        """
        初始化提示嵌入权重
        
        Args:
            init_method: 初始化方法
                - "random": 随机正态分布初始化
                - "vocab": 使用随机词汇嵌入初始化
                - "natural_language": 使用自然语言提示的平均嵌入初始化
                - "cluster_center": 使用聚类中心初始化
            natural_prompts: 自然语言提示列表，用于natural_language方法
            cluster_centers: 聚类中心张量，用于cluster_center方法
        """
        if init_method == "random":
            # 使用正态分布随机初始化
            nn.init.normal_(self.prompt_embeddings.weight, std=0.02)
            
        elif init_method == "vocab":
            # 使用随机词汇表嵌入初始化，这样可以让软提示有更好的起始点
            vocab_size = self.base_model.config.vocab_size
            random_indices = torch.randint(0, vocab_size, (self.num_virtual_tokens,))
            with torch.no_grad():
                self.prompt_embeddings.weight.copy_(
                    self.base_model.get_input_embeddings().weight[random_indices]
                )
                
        elif init_method == "natural_language":
            # 使用自然语言提示的平均嵌入初始化
            # 这种方法利用人类撰写的高质量提示作为初值，既简单又带语义
            self._init_with_natural_prompts(natural_prompts)
            
        elif init_method == "cluster_center":
            # 使用聚类中心初始化
            # 如果对输入进行了聚类，为每个簇初始化一个更语义贴近该簇的提示
            self._init_with_cluster_centers(cluster_centers)
            
        else:
            raise ValueError(f"Unknown init_method: {init_method}")
    
    def _init_with_natural_prompts(self, natural_prompts: Optional[list]):
        """
        使用自然语言提示的平均嵌入初始化
        
        Args:
            natural_prompts: 自然语言提示列表，例如：
                ["Please help me with this task:", "You are a helpful assistant."]
        """
        if natural_prompts is None:
            # 如果没有提供自然语言提示，使用默认的高质量提示
            natural_prompts = [
                "You are a helpful, harmless, and honest assistant.",
                "Please provide a thoughtful and accurate response.",
                "Consider the following carefully and respond appropriately:"
            ]
            print("⚠️  No natural prompts provided, using default prompts.")
        
        # 获取分词器（假设base_model有tokenizer属性，或从全局获取）
        try:
            # 尝试从模型获取分词器
            if hasattr(self.base_model, 'tokenizer'):
                tokenizer = self.base_model.tokenizer
            else:
                # 如果模型没有分词器属性，需要手动传入或从其他地方获取
                print("⚠️  Cannot find tokenizer, falling back to random initialization")
                nn.init.normal_(self.prompt_embeddings.weight, std=0.02)
                return
        except:
            print("⚠️  Error accessing tokenizer, falling back to random initialization")
            nn.init.normal_(self.prompt_embeddings.weight, std=0.02)
            return
        
        # 对自然语言提示进行编码
        all_prompt_embeddings = []
        
        for prompt_text in natural_prompts:
            # 分词并获取嵌入
            tokens = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt")
            tokens = tokens.to(self.base_model.device)
            
            # 获取token的嵌入向量
            token_embeddings = self.base_model.get_input_embeddings()(tokens)  # [1, seq_len, hidden_size]
            
            # 计算平均嵌入（对序列长度维度求平均）
            avg_embedding = token_embeddings.mean(dim=1)  # [1, hidden_size]
            all_prompt_embeddings.append(avg_embedding)
        
        # 将所有提示的嵌入堆叠并计算总平均
        stacked_embeddings = torch.cat(all_prompt_embeddings, dim=0)  # [num_prompts, hidden_size]
        mean_embedding = stacked_embeddings.mean(dim=0)  # [hidden_size]
        
        # 用平均嵌入初始化所有虚拟token，并添加小的随机扰动以增加多样性
        with torch.no_grad():
            for i in range(self.num_virtual_tokens):
                # 基础嵌入 + 小的随机扰动
                noise = torch.randn_like(mean_embedding) * 0.01
                self.prompt_embeddings.weight[i] = mean_embedding + noise
        
        print(f"✅ Initialized {self.num_virtual_tokens} prompt embeddings using {len(natural_prompts)} natural language prompts")
    
    def _init_with_cluster_centers(self, cluster_centers: Optional[torch.Tensor]):
        """
        使用聚类中心初始化提示嵌入
        
        Args:
            cluster_centers: 聚类中心张量 [num_clusters, embedding_dim]
                           每个聚类中心代表一类输入的语义特征
        """
        if cluster_centers is None:
            print("⚠️  No cluster centers provided, falling back to random initialization")
            nn.init.normal_(self.prompt_embeddings.weight, std=0.02)
            return
        
        # 确保cluster_centers在正确的设备上
        cluster_centers = cluster_centers.to(self.prompt_embeddings.weight.device)
        
        num_clusters = cluster_centers.size(0)
        embedding_dim = cluster_centers.size(1)
        
        # 检查维度匹配
        if embedding_dim != self.prompt_embedding_dim:
            print(f"⚠️  Cluster center dimension ({embedding_dim}) doesn't match prompt embedding dimension ({self.prompt_embedding_dim})")
            print("   Falling back to random initialization")
            nn.init.normal_(self.prompt_embeddings.weight, std=0.02)
            return
        
        with torch.no_grad():
            if num_clusters >= self.num_virtual_tokens:
                # 如果聚类中心数量 >= 虚拟token数量，直接使用前num_virtual_tokens个中心
                self.prompt_embeddings.weight.copy_(cluster_centers[:self.num_virtual_tokens])
            else:
                # 如果聚类中心数量 < 虚拟token数量，需要重复或插值
                for i in range(self.num_virtual_tokens):
                    cluster_idx = i % num_clusters
                    # 使用对应的聚类中心，并添加小的随机扰动以增加多样性
                    noise = torch.randn_like(cluster_centers[cluster_idx]) * 0.01
                    self.prompt_embeddings.weight[i] = cluster_centers[cluster_idx] + noise
        
        print(f"✅ Initialized {self.num_virtual_tokens} prompt embeddings using {num_clusters} cluster centers")

    def get_prompt_embeddings(self, batch_size: int) -> torch.Tensor:
        """
        获取批次的提示嵌入
        
        Args:
            batch_size: 批次大小
            
        Returns:
            提示嵌入张量 [batch_size, num_virtual_tokens, embedding_dim]
        """
        # 创建提示token的索引
        prompt_indices = torch.arange(self.num_virtual_tokens).unsqueeze(0).expand(
            batch_size, -1
        ).to(self.prompt_embeddings.weight.device)
        
        # 返回对应的嵌入向量
        return self.prompt_embeddings(prompt_indices)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """
        前向传播：将提示嵌入与输入嵌入拼接后传入基础模型
        
        Args:
            input_ids: 输入token ID [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            labels: 标签 [batch_size, seq_len]
            
        Returns:
            模型输出
        """
        batch_size = input_ids.size(0)
        
        # 获取输入的嵌入表示
        input_embeddings = self.base_model.get_input_embeddings()(input_ids)
        
        # 获取提示嵌入
        prompt_embeddings = self.get_prompt_embeddings(batch_size)
        
        # 将提示嵌入与输入嵌入拼接：[prompt_embeddings | input_embeddings]
        inputs_embeds = torch.cat([prompt_embeddings, input_embeddings], dim=1)
        
        # 调整注意力掩码以包含提示部分
        if attention_mask is not None:
            # 注意力掩码的值含义：
            # - 1: 允许注意力计算，token参与注意力机制
            # - 0: 屏蔽注意力计算，token不参与注意力机制（如padding token）
            #
            # 为提示部分创建全1的注意力掩码的原因：
            # 1. 提示嵌入是可学习的参数，需要参与注意力计算才能发挥作用
            # 2. 提示token不是padding，而是有意义的软提示向量
            # 3. 模型需要能够"看到"这些提示信息来理解任务指令
            # 4. 如果设为0，提示嵌入将被忽略，失去P-tuning的效果
            prompt_attention_mask = torch.ones(
                batch_size, self.num_virtual_tokens,
                dtype=attention_mask.dtype,
                device=attention_mask.device
            )
            # 拼接提示和输入的注意力掩码
            # 最终形状: [batch_size, num_virtual_tokens + original_seq_len]
            # 前num_virtual_tokens位为1（提示部分），后续为原始输入的掩码
            attention_mask = torch.cat([prompt_attention_mask, attention_mask], dim=1)
        
        # 调整标签以包含提示部分（提示部分标签设为-100，不参与损失计算）
        if labels is not None:
            prompt_labels = torch.full(
                (batch_size, self.num_virtual_tokens),
                -100,  # -100表示在损失计算中忽略这些位置
                dtype=labels.dtype,
                device=labels.device
            )
            labels = torch.cat([prompt_labels, labels], dim=1)
        
        # 通过基础模型进行前向传播
        return self.base_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )
    
    def generate(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **kwargs):
        """
        生成文本：在生成过程中包含提示嵌入 - 完全修复版本
        """
        batch_size = input_ids.size(0)
        
        # 获取输入嵌入
        input_embeddings = self.base_model.get_input_embeddings()(input_ids)
        
        # 获取提示嵌入
        prompt_embeddings = self.get_prompt_embeddings(batch_size)
        
        # 拼接提示嵌入和输入嵌入
        inputs_embeds = torch.cat([prompt_embeddings, input_embeddings], dim=1)
        
        # 🚨 关键修复：正确处理attention_mask
        if attention_mask is not None:
            # 为prompt部分创建attention mask (全部为1，因为prompt需要参与attention)
            prompt_attention_mask = torch.ones(
                batch_size, self.num_virtual_tokens,
                dtype=attention_mask.dtype,
                device=attention_mask.device
            )
            # 拼接prompt和输入的attention mask
            combined_attention_mask = torch.cat([prompt_attention_mask, attention_mask], dim=1)
        else:
            # 如果没有提供attention_mask，为整个序列创建全1的mask
            total_length = self.num_virtual_tokens + input_ids.size(1)
            combined_attention_mask = torch.ones(
                batch_size, total_length,
                dtype=torch.long,
                device=input_ids.device
            )
        
        # 🚨 修复：更新kwargs，移除冲突参数
        generation_kwargs = kwargs.copy()
        
        # 移除input_ids，因为我们使用inputs_embeds
        generation_kwargs.pop('input_ids', None)
        
        # 设置正确的参数
        generation_kwargs['inputs_embeds'] = inputs_embeds
        generation_kwargs['attention_mask'] = combined_attention_mask
        
        # 🚨 重要修复：如果同时设置了max_length和max_new_tokens，优先使用max_new_tokens
        if 'max_length' in generation_kwargs and 'max_new_tokens' in generation_kwargs:
            generation_kwargs.pop('max_length', None)
        
        # 🚨 新增：设置默认的停止条件
        if 'eos_token_id' not in generation_kwargs:
            generation_kwargs['eos_token_id'] = self.base_model.config.eos_token_id
        
        if 'pad_token_id' not in generation_kwargs:
            generation_kwargs['pad_token_id'] = self.base_model.config.pad_token_id
        
        # 🚨 新增：确保生成会停止
        if 'max_new_tokens' not in generation_kwargs and 'max_length' not in generation_kwargs:
            generation_kwargs['max_new_tokens'] = 100  # 默认最多生成100个token
        
        # 使用基础模型生成
        return self.base_model.generate(**generation_kwargs)
    
    def save_prompt_embeddings(self, path: str):
        """
        保存训练好的提示嵌入
        
        Args:
            path: 保存路径
        """
        torch.save(self.prompt_embeddings.state_dict(), path)
    
    def load_prompt_embeddings(self, path: str):
        """
        加载预训练的提示嵌入
        
        Args:
            path: 加载路径
        """
        state_dict = torch.load(path, map_location='cpu')
        self.prompt_embeddings.load_state_dict(state_dict)
        print(f"✅ Loaded prompt embeddings from {path}")
    
    @classmethod
    def from_pretrained_prompts(
        cls, 
        base_model: PreTrainedModel, 
        prompt_embeddings_path: str,
        config_path: Optional[str] = None
    ):
        """
        从保存的prompt embeddings创建P-tuning模型
        
        Args:
            base_model: 基础预训练模型
            prompt_embeddings_path: prompt embeddings文件路径
            config_path: 配置文件路径（可选）
            
        Returns:
            加载了训练好的prompt embeddings的P-tuning模型
        """
        # 如果提供了配置文件，使用配置中的参数
        if config_path and os.path.exists(config_path):
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            model = cls(
                base_model=base_model,
                num_virtual_tokens=config['num_virtual_tokens'],
                prompt_embedding_dim=config.get('prompt_embedding_dim'),
                init_method=config.get('prompt_init_method', 'random')
            )
        else:
            # 使用默认参数
            model = cls(base_model=base_model)
        
        # 加载训练好的prompt embeddings
        model.load_prompt_embeddings(prompt_embeddings_path)
        return model

    # 🚨 移除冗余的_compute_preference_loss和_compute_log_probs方法
    # 这些方法已经在PTuningTrainer中实现，保留这里会造成混淆
