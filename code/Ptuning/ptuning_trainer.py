import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Any
from transformers import Trainer, TrainerCallback
from ptuning_model import PTuningModel
import logging
import wandb
import numpy as np
from tqdm import tqdm
import os

logger = logging.getLogger(__name__)


class PTuningEarlyStoppingCallback(TrainerCallback):
    """
    P-tuning早停回调函数
    """
    
    def __init__(self, patience: int = 5, threshold: float = 0.01, target_accuracy: float = 0.9):
        self.patience = patience
        self.threshold = threshold
        self.target_accuracy = target_accuracy
        self.best_accuracy = 0.0
        self.best_margin = 0.0
        self.patience_counter = 0
        self.should_stop = False
    
    def on_evaluate(self, args, state, control, logs=None, **kwargs):
        """
        验证时调用
        """
        current_accuracy = logs.get("eval_ranking_accuracy", 0.0)
        current_margin = logs.get("eval_mean_margin", 0.0)
        
        # 检查是否达到目标准确率
        if current_accuracy >= self.target_accuracy:
            print(f"🎯 Target accuracy {self.target_accuracy:.1%} reached! Current: {current_accuracy:.1%}")
            control.should_training_stop = True
            return control
        
        # 早停逻辑：基于准确率和边距的综合改善
        improvement = (current_accuracy - self.best_accuracy) + 0.1 * (current_margin - self.best_margin)
        
        if improvement > self.threshold:
            self.best_accuracy = max(self.best_accuracy, current_accuracy)
            self.best_margin = max(self.best_margin, current_margin)
            self.patience_counter = 0
            print(f"✅ Validation improved! Accuracy: {current_accuracy:.1%}, Margin: {current_margin:.4f}")
        else:
            self.patience_counter += 1
            print(f"⚠️ No improvement for {self.patience_counter}/{self.patience} evaluations")
            
            if self.patience_counter >= self.patience:
                print(f"🛑 Early stopping triggered! Best accuracy: {self.best_accuracy:.1%}")
                control.should_training_stop = True
        
        return control


class PTuningTrainer(Trainer):
    """
    P-tuning专用训练器，带验证评估和wandb监控
    """
    
    def __init__(
        self,
        model: PTuningModel,
        preference_loss_weight: float = 1.0,
        margin: float = 0.1,
        kl_loss_weight: float = 0.1,  # 🆕 新增KL散度损失权重
        early_stopping_patience: int = 5,
        early_stopping_threshold: float = 0.01,
        target_accuracy: float = 0.9,
        use_wandb: bool = True,
        **kwargs
    ):
        """
        初始化P-tuning训练器
        
        Args:
            kl_loss_weight: KL散度损失权重，用于约束prompt对原模型输出分布的影响
        """
        super().__init__(model=model, **kwargs)
        self.preference_loss_weight = preference_loss_weight
        self.margin = margin
        self.kl_loss_weight = kl_loss_weight  # 🆕 KL散度约束权重
        self.use_wandb = use_wandb
        
        # 添加早停回调
        early_stopping_callback = PTuningEarlyStoppingCallback(
            patience=early_stopping_patience,
            threshold=early_stopping_threshold,
            target_accuracy=target_accuracy
        )
        self.add_callback(early_stopping_callback)
        
        # 训练状态记录
        self.step_count = 0
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        计算总损失 = 偏好损失 + KL约束损失
        """
        # 直接处理你的数据格式
        device = next(model.parameters()).device
        
        # 从数据集中获取chosen和rejected文本
        if 'chosen' in inputs and 'rejected' in inputs:
            chosen_texts = inputs['chosen']
            rejected_texts = inputs['rejected']
            prompts = inputs.get('prompt', [''] * len(chosen_texts))
            
            # 获取tokenizer进行编码
            if hasattr(self, 'processing_class'):
                tokenizer_processor = self.processing_class
            elif hasattr(self, 'tokenizer'):
                tokenizer_processor = self.tokenizer
            
            # 🚨 关键修复：只需要编码原始chosen/rejected文本
            chosen_encoding = tokenizer_processor(
                chosen_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            
            rejected_encoding = tokenizer_processor(
                rejected_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            
            # 移动到设备
            chosen_input_ids = chosen_encoding["input_ids"].to(device)
            chosen_attention_mask = chosen_encoding["attention_mask"].to(device)
            rejected_input_ids = rejected_encoding["input_ids"].to(device)
            rejected_attention_mask = rejected_encoding["attention_mask"].to(device)
            
        else:
            raise ValueError(f"Expected 'chosen' and 'rejected' keys in inputs, got: {list(inputs.keys())}")
        
        # 🚨 修复：计算偏好损失和KL损失，传入正确的参数
        preference_loss, kl_loss = self._compute_preference_loss(
            model=model,
            preferred_input_ids=chosen_input_ids,
            rejected_input_ids=rejected_input_ids,
            preferred_attention_mask=chosen_attention_mask,
            rejected_attention_mask=rejected_attention_mask
        )
        
        # 组合总损失
        total_loss = (
            self.preference_loss_weight * preference_loss + 
            self.kl_loss_weight * kl_loss
        )
        
        # 记录训练指标到wandb
        if self.use_wandb and wandb.run is not None:
            self.step_count += 1
            wandb.log({
                "train/total_loss": total_loss.item(),
                "train/preference_loss": preference_loss.item(),
                "train/kl_loss": kl_loss.item(),
                "train/step": self.step_count
            })
        
        return (total_loss, None) if return_outputs else total_loss
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """
        执行验证评估，计算ranking accuracy和mean margin
        """
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        
        if (eval_dataset is None):
            print("⚠️ No evaluation dataset provided")
            return {}
        
        print(f"🔍 Starting evaluation on {len(eval_dataset)} samples...")
        
        # 设置模型为评估模式
        self.model.eval()
        
        correct_predictions = 0
        total_predictions = 0
        log_prob_differences = []
        
        device = next(self.model.parameters()).device
        
        # 获取tokenizer
        if hasattr(self, 'processing_class'):
            tokenizer_processor = self.processing_class
        elif hasattr(self, 'tokenizer'):
            tokenizer_processor = self.tokenizer
        
        with torch.no_grad():
            for i in tqdm(range(len(eval_dataset)), desc="Evaluating"):
                sample = eval_dataset[i]
                
                # 准备数据
                chosen_text = sample['chosen']
                rejected_text = sample['rejected']
                
                # 编码
                chosen_encoding = tokenizer_processor(
                    [chosen_text],
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                )
                
                rejected_encoding = tokenizer_processor(
                    [rejected_text],
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                )
                
                # 移动到设备
                chosen_input_ids = chosen_encoding["input_ids"].to(device)
                chosen_attention_mask = chosen_encoding["attention_mask"].to(device)
                rejected_input_ids = rejected_encoding["input_ids"].to(device)
                rejected_attention_mask = rejected_encoding["attention_mask"].to(device)
                
                # 前向传播
                chosen_outputs = self.model(
                    input_ids=chosen_input_ids,
                    attention_mask=chosen_attention_mask
                )
                rejected_outputs = self.model(
                    input_ids=rejected_input_ids,
                    attention_mask=rejected_attention_mask
                )
                
                # 计算对数概率
                chosen_log_prob = self._compute_sequence_log_probs(
                    chosen_outputs.logits, chosen_input_ids, chosen_attention_mask
                )
                rejected_log_prob = self._compute_sequence_log_probs(
                    rejected_outputs.logits, rejected_input_ids, rejected_attention_mask
                )
                
                # 计算概率差 δpi = log P(y_wi|xi,p) - log P(y_li|xi,p)
                log_prob_diff = chosen_log_prob.item() - rejected_log_prob.item()
                log_prob_differences.append(log_prob_diff)
                
                # 判断是否正确预测（δpi > 0 表示chosen概率更高）
                if log_prob_diff > 0:
                    correct_predictions += 1
                
                total_predictions += 1
        
        # 计算指标
        ranking_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        mean_margin = np.mean(log_prob_differences) if log_prob_differences else 0.0
        std_margin = np.std(log_prob_differences) if log_prob_differences else 0.0
        
        # 创建评估结果
        eval_results = {
            f"{metric_key_prefix}_ranking_accuracy": ranking_accuracy,
            f"{metric_key_prefix}_mean_margin": mean_margin,
            f"{metric_key_prefix}_std_margin": std_margin,
            f"{metric_key_prefix}_correct_predictions": correct_predictions,
            f"{metric_key_prefix}_total_predictions": total_predictions,
        }
        
        # 记录到wandb
        if self.use_wandb and wandb.run is not None:
            wandb_logs = {
                "eval/ranking_accuracy": ranking_accuracy,
                "eval/mean_margin": mean_margin,
                "eval/std_margin": std_margin,
                "eval/correct_predictions": correct_predictions,
                "eval/total_predictions": total_predictions,
            }
            
            # 添加概率差分布直方图
            if log_prob_differences:
                wandb_logs["eval/margin_distribution"] = wandb.Histogram(log_prob_differences)
            
            wandb.log(wandb_logs)
        
        # 打印结果
        print(f"📊 Evaluation Results:")
        print(f"   Ranking Accuracy: {ranking_accuracy:.1%}")
        print(f"   Mean Margin: {mean_margin:.4f}")
        print(f"   Std Margin: {std_margin:.4f}")
        print(f"   Correct Predictions: {correct_predictions}/{total_predictions}")
        
        # 恢复训练模式
        self.model.train()
        
        return eval_results

    def _compute_preference_loss(
        self,
        model: PTuningModel,
        preferred_input_ids: torch.Tensor,
        rejected_input_ids: torch.Tensor,
        preferred_attention_mask: Optional[torch.Tensor] = None,
        rejected_attention_mask: Optional[torch.Tensor] = None
    ) -> tuple:
        """
        计算偏好排序损失 + KL散度约束损失
        
        🎯 正确的KL散度计算逻辑：
        
        1. P-tuning模型：P(y|x, prompt_embeddings)
           - 对原始文本x使用model()，内部会自动添加训练的prompt embeddings
           
        2. 原始模型：P(y|x)  
           - 对相同原始文本x使用model.base_model()，不添加任何prompt
           
        3. KL约束：确保添加prompt后的分布不要偏离原始分布太远
           - KL(P(y|x, prompt_embeddings) || P(y|x)) < threshold
        
        Args:
            model: P-tuning模型
            preferred_input_ids: 偏好回复的token ID
            rejected_input_ids: 非偏好回复的token ID
            preferred_attention_mask: 偏好回复的注意力掩码
            rejected_attention_mask: 非偏好回复的注意力掩码
            
        Returns:
            (preference_loss, kl_loss): 偏好损失和KL散度损失的元组
        """
        # 1. 计算P-tuning模型的输出（包含prompt embeddings）
        # model() 内部会自动添加prompt embeddings到输入前面
        preferred_outputs_ptuning = model(
            input_ids=preferred_input_ids,
            attention_mask=preferred_attention_mask
        )
        rejected_outputs_ptuning = model(
            input_ids=rejected_input_ids,
            attention_mask=rejected_attention_mask
        )
        
        # 2. 计算P-tuning模型的序列对数概率
        preferred_log_probs_ptuning = self._compute_sequence_log_probs(
            preferred_outputs_ptuning.logits, preferred_input_ids, preferred_attention_mask
        )
        rejected_log_probs_ptuning = self._compute_sequence_log_probs(
            rejected_outputs_ptuning.logits, rejected_input_ids, rejected_attention_mask
        )
        
        # 3. 计算偏好损失
        preference_loss = F.relu(
            rejected_log_probs_ptuning - preferred_log_probs_ptuning + self.margin
        )
        
        # 4. 🚨 正确的KL散度约束计算
        kl_loss = torch.tensor(0.0, device=preferred_input_ids.device)
        
        if self.kl_loss_weight > 0:  # 只有当KL权重>0时才计算
            # 计算原始模型在相同文本上的输出（不使用prompt）
            with torch.no_grad():  # 不需要对原始模型计算梯度
                preferred_outputs_original = model.base_model(
                    input_ids=preferred_input_ids,
                    attention_mask=preferred_attention_mask
                )
                rejected_outputs_original = model.base_model(
                    input_ids=rejected_input_ids,
                    attention_mask=rejected_attention_mask
                )
            
            # 计算原始模型的序列对数概率
            preferred_log_probs_original = self._compute_sequence_log_probs_original(
                preferred_outputs_original.logits, 
                preferred_input_ids, 
                preferred_attention_mask
            )
            rejected_log_probs_original = self._compute_sequence_log_probs_original(
                rejected_outputs_original.logits, 
                rejected_input_ids, 
                rejected_attention_mask
            )
            
            # 🚨 关键修复：正确计算KL散度
            # 目标：P-tuning模型的分布不要偏离原始模型太远
            # 使用MSE损失近似KL散度（在对数空间中）
            kl_loss_preferred = F.mse_loss(
                preferred_log_probs_ptuning, 
                preferred_log_probs_original.detach()  # 确保不对原始模型计算梯度
            )
            kl_loss_rejected = F.mse_loss(
                rejected_log_probs_ptuning, 
                rejected_log_probs_original.detach()
            )
            kl_loss = (kl_loss_preferred + kl_loss_rejected) / 2
        
        return preference_loss.mean(), kl_loss

    def _compute_sequence_log_probs_original(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        专门用于原始模型的序列对数概率计算
        
        与_compute_sequence_log_probs的区别：
        - 这个方法用于原始模型（不含prompt embeddings）
        - 直接处理完整的input_ids，不需要跳过prompt部分
        - logits长度 = input_ids长度
        
        Args:
            logits: 原始模型输出的logits [batch_size, seq_len, vocab_size]
            input_ids: 输入token ID [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            
        Returns:
            序列的平均对数概率 [batch_size]
        """
        # 将logits转换为对数概率
        log_probs = F.log_softmax(logits, dim=-1)
        
        # 🚨 关键：原始模型的logits长度应该等于input_ids长度
        # 如果不等，需要截断到较短的长度
        seq_len = min(log_probs.size(1), input_ids.size(1))
        log_probs = log_probs[:, :seq_len, :]
        input_ids = input_ids[:, :seq_len]
        
        if attention_mask is not None:
            attention_mask = attention_mask[:, :seq_len]
        
        # 收集目标token的对数概率
        gathered_log_probs = log_probs.gather(
            dim=-1, 
            index=input_ids.unsqueeze(-1)
        ).squeeze(-1)  # [batch_size, seq_len]
        
        # 计算平均对数概率
        if attention_mask is not None:
            masked_log_probs = gathered_log_probs * attention_mask.float()
            valid_lengths = attention_mask.sum(dim=1).clamp(min=1)
            return masked_log_probs.sum(dim=1) / valid_lengths
        else:
            return gathered_log_probs.mean(dim=1)
    
    def _compute_sequence_log_probs(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算整个序列的对数概率
        
        关键修改：
        1. 计算完整序列的概率（包括prompt影响）
        2. 不跳过prompt部分，因为prompt的作用就是影响整个序列的概率分布
        
        Args:
            logits: 模型输出的logits [batch_size, seq_len_with_prompt, vocab_size]
            input_ids: 输入token ID [batch_size, seq_len_without_prompt]
            attention_mask: 注意力掩码 [batch_size, seq_len_without_prompt]
            
        Returns:
            序列的平均对数概率
        """
        # 将logits转换为对数概率
        log_probs = F.log_softmax(logits, dim=-1)
        
        # 由于prompt被添加到序列前面，我们需要从logits中提取对应原始输入的部分
        num_prompt_tokens = self.model.num_virtual_tokens
        
        # 提取对应原始输入序列的logits（跳过prompt部分）
        # logits形状: [batch_size, num_prompt_tokens + seq_len, vocab_size]
        # 我们需要: [batch_size, seq_len, vocab_size]
        relevant_log_probs = log_probs[:, num_prompt_tokens:num_prompt_tokens + input_ids.size(1), :]
        
        # 收集目标token的对数概率
        gathered_log_probs = relevant_log_probs.gather(
            dim=-1, 
            index=input_ids.unsqueeze(-1)
        ).squeeze(-1)  # [batch_size, seq_len]
        
        # 如果有attention_mask，只计算非padding部分的平均概率
        if attention_mask is not None:
            # 将padding位置的概率设为0
            masked_log_probs = gathered_log_probs * attention_mask.float()
            # 计算有效token的平均对数概率
            valid_lengths = attention_mask.sum(dim=1).clamp(min=1)
            return masked_log_probs.sum(dim=1) / valid_lengths
        else:
            # 如果没有attention_mask，计算所有位置的平均概率
            return gathered_log_probs.mean(dim=1)
    
    def save_prompt_embeddings(self, output_dir: str):
        """
        保存训练好的提示嵌入
        
        Args:
            output_dir: 输出目录
        """
        import os
        prompt_path = os.path.join(output_dir, "prompt_embeddings.pt")
        self.model.save_prompt_embeddings(prompt_path)
        print(f"Prompt embeddings saved to {prompt_path}")
    
    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        重写训练步骤以避免optimizer.train()调用
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        del inputs
        
        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)

        return loss.detach() / self.args.gradient_accumulation_steps

    def _prepare_inputs(self, inputs):
        """
        重写_prepare_inputs以处理自定义数据格式
        """
        # 直接返回inputs，因为我们在compute_loss中处理数据转换
        return inputs

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        """
        重写保存方法，只保存prompt embeddings而不是整个模型
        
        由于我们只训练prompt embeddings，基础模型参数被冻结，
        因此只需要保存prompt embeddings即可，避免保存共享tensor的问题。
        """
        # 如果没有指定输出目录，使用默认的
        if (output_dir is None):
            output_dir = self.args.output_dir

        os.makedirs(output_dir, exist_ok=True)
        
        # 只保存prompt embeddings
        prompt_embeddings_path = os.path.join(output_dir, "prompt_embeddings.pt")
        self.model.save_prompt_embeddings(prompt_embeddings_path)
        
        # 保存训练配置
        config_path = os.path.join(output_dir, "ptuning_config.json")
        config = {
            "model_name_or_path": self.model.base_model.config.name_or_path if hasattr(self.model.base_model.config, 'name_or_path') else 'unknown',
            "num_virtual_tokens": self.model.num_virtual_tokens,
            "prompt_embedding_dim": self.model.prompt_embedding_dim,
            "margin": self.model.margin,
        }
        
        with open(config_path, 'w') as f:
            import json
            json.dump(config, f, indent=2)
        
        print(f"💾 P-tuning model saved to {output_dir}")
        print(f"   - Prompt embeddings: {prompt_embeddings_path}")
        print(f"   - Config: {config_path}")
    
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        重写保存模型方法，只保存prompt embeddings
        """
        self._save(output_dir)
    
    def _save_checkpoint(self, model, trial, metrics=None):
        """
        重写保存检查点方法
        """
        # 生成检查点目录名
        checkpoint_folder = f"checkpoint-{self.state.global_step}"
        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)
        
        # 保存prompt embeddings
        self._save(output_dir)
        
        # 保存训练状态
        self.state.save_to_json(os.path.join(output_dir, "trainer_state.json"))
        
        # 保存训练参数
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
        
        # 如果是最佳模型，创建最佳模型的标记
        if getattr(self, '_load_best_model_at_end', False) and metrics is not None:
            metric_key = self.args.metric_for_best_model
            if metric_key in metrics:
                best_metric_path = os.path.join(run_dir, "best_metric.txt")
                with open(best_metric_path, "w") as f:
                    f.write(f"{metrics[metric_key]}")
        
        # 清理旧的检查点
        self._rotate_checkpoints(use_mtime=False, output_dir=run_dir)
