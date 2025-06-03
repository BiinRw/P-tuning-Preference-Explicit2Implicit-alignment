import torch
torch.backends.cuda.matmul.allow_tf32 = True
import torch.nn.functional as F
import torch.nn as nn
from transformers import Trainer, TrainingArguments

import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    StateDictType,
    BackwardPrefetch,
    ShardingStrategy,
    CPUOffload,
)
from torch.distributed.fsdp.api import FullStateDictConfig, FullOptimStateDictConfig
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import contextlib

from pro_utils.preference_datasets import get_batch_iterator
from pro_utils.DPO_utils import (
    slice_and_move_batch_for_device,
    formatted_dict,
    all_gather_if_needed,
    pad_to_length,
    get_block_class_from_model,
    rank0_print,
    get_local_dir,
)
import numpy as np
import wandb
from tqdm.auto import tqdm

import random
import os
from collections import defaultdict
import time
import json
import functools
from typing import Optional, Dict, List, Union, Tuple
import deepspeed


def preference_loss(policy_chosen_logps: torch.FloatTensor,
                    policy_rejected_logps: torch.FloatTensor,
                    reference_chosen_logps: torch.FloatTensor,
                    reference_rejected_logps: torch.FloatTensor,
                    beta: float,
                    tau : float = 0.1,
                    label_smoothing: float = 0.0,
                    loss_name: str = "DPO",
                    reference_free: bool = False) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Compute the DPO loss for a batch of policy and reference model log probabilities.

    Args:
        policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
        policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
        reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
        reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
        beta: Temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.
        label_smoothing: conservativeness for DPO loss, which assumes that preferences are noisy (flipped with probability label_smoothing)
        ipo: If True, use the IPO loss instead of the DPO loss.
        reference_free: If True, we ignore the _provided_ reference model and implicitly use a reference model that assigns equal probability to all responses.

    Returns:
        A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
        The losses tensor contains the DPO loss for each example in the batch.
        The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    if reference_free:
        ref_logratios = 0

    logits = pi_logratios - ref_logratios  # also known as h_{\pi_\theta}^{y_w,y_l}
    logits = torch.clamp(logits, min=-10, max=10)
    if loss_name == "ipo":
        losses = (logits - 1/(2 * beta)) ** 2  # Eq. 17 of https://arxiv.org/pdf/2310.12036v2.pdf
    elif loss_name == "dpo":
        # Eq. 3 https://ericmitchell.ai/cdpo.pdf; label_smoothing=0 gives original DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)
        losses = -torch.log(1 + torch.exp(-beta * logits) + 1e-8) * (1 - label_smoothing) - torch.log(1 + torch.exp(beta * logits)+ 1e-8) * label_smoothing
        #losses = -F.logsigmoid(beta * logits) * (1 - label_smoothing) - F.logsigmoid(-beta * logits) * label_smoothing 
    elif loss_name == "scpd":
        # Eq. 3 https://ericmitchell.ai/cdpo.pdf; label_smoothing=0 gives original DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)
        #dpo_losses = -F.logsigmoid(beta * logits) * (1 - label_smoothing) - F.logsigmoid(-beta * logits) * label_smoothing
        cl_logits = -F.log_softmax((reference_chosen_logps - policy_rejected_logps)/tau, dim = -1)
        dpo_losses = -F.logsigmoid(beta * logits) * (1 - label_smoothing) - F.logsigmoid(-beta * logits) * label_smoothing
        losses = dpo_losses + cl_logits
        #losses = dpo_losses        
    elif loss_name == "sipa":
        losses = -F.logsigmoid(beta * logits) * (1 - label_smoothing) - F.logsigmoid(-beta * logits) * label_smoothing

    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()

    return losses, chosen_rewards, rejected_rewards

def scale_logps(policy_chosen_logps: torch.Tensor, policy_rejected_logps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    
    if not isinstance(policy_chosen_logps, torch.Tensor) or not isinstance(policy_rejected_logps, torch.Tensor):
        raise TypeError("Inputs for scale_logps() must be torch.FloatTensor")
    max_abs_value = max(policy_chosen_logps.abs().max(),policy_rejected_logps.abs().max())
    if max_abs_value == 0:
        return policy_chosen_logps, policy_rejected_logps
    
    policy_chosen_logps_scaled = policy_chosen_logps / max_abs_value
    policy_rejected_logps_scaled = policy_rejected_logps / max_abs_value

    return policy_chosen_logps_scaled, policy_rejected_logps_scaled


def odd_ratio_loss(policy_chosen_logps: torch.FloatTensor, policy_rejected_logps: torch.FloatTensor, beta: float) ->Tuple[torch.FloatTensor, torch.FloatTensor,torch.FloatTensor, torch.FloatTensor,torch.FloatTensor]:
    """Compute the ORPO loss from paper https://arxiv.org/abs/2403.07691; 《ORPO: Monolithic Preference Optimization without Reference Model》
    Args:
        policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
        policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
        beta: Temperature parameter for the ORPO loss.
    returns:
        A tuple of three tensors:(losses, chosen_rewards, rejected_rewards).
        The losses tensor contains the ORPO loss for each example in the batch.
        The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
        The log odds ratio of the chosen reponses over the rejected responses ratio for logging purposes.
        The 'log(sigmoid(log_odd_chosen))' for logging purposes.
    """
    # Derived from Eqs. (4) and (7) from https://arxiv.org/abs/2403.07691 by using log identities and exp(logP(y|x)) = P(y|x)

    scaled_policy_chosen_logps, scaled_policy_rejected_logps = scale_logps(policy_chosen_logps, policy_rejected_logps)

    log_odds = (scaled_policy_chosen_logps - scaled_policy_rejected_logps) - (
        torch.log1p(-torch.exp(policy_chosen_logps)) - torch.log1p(-torch.exp(policy_rejected_logps))
    )

    if torch.isnan(log_odds).any() or torch.isinf(log_odds).any():
        print(f"log_odds contains NaN or Inf: {log_odds}")
    sig_ratio  = F.sigmoid(log_odds)
    ratio = torch.log(sig_ratio)
    losses = beta * ratio

    chosen_rewards = beta * (scaled_policy_chosen_logps).detach()
    rejected_rewards = beta * (scaled_policy_rejected_logps).detach()

    tqdm.write(f"policy_chosen_logps: {scaled_policy_chosen_logps}")
    tqdm.write(f"policy_rejected_logps: {scaled_policy_rejected_logps}")
    tqdm.write(f"log_odds: {log_odds}")
    tqdm.write(f"sig_ratio: {sig_ratio}")
    tqdm.write(f"ratio: {ratio}")
    tqdm.write(f"orpo_losses: {losses}")
    
    return losses, chosen_rewards, rejected_rewards, torch.mean(ratio).item(), torch.mean(log_odds).item()

def _get_batch_logps(logits: torch.FloatTensor, labels: torch.LongTensor, average_log_prob: bool = False) -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
        average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

    Returns:
        A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
    """
    assert logits.shape[:-1] == labels.shape

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = (labels != -100)

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == -100] = 0

    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

    if average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (per_token_logps * loss_mask).sum(-1)


def concatenated_inputs(batch: Dict[str, Union[List, torch.LongTensor]]) -> Dict[str, torch.LongTensor]:
    """Concatenate the chosen and rejected inputs into a single tensor.
    
    Args:
        batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).
        
    Returns:
        A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
    """
    max_length = max(batch['chosen_input_ids'].shape[1], batch['rejected_input_ids'].shape[1])
    concatenated_batch = {}
    for k in batch:
        if k.startswith('chosen') and isinstance(batch[k], torch.Tensor):
            pad_value = -100 if 'labels' in k else 0
            concatenated_key = k.replace('chosen', 'concatenated')
            concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
    for k in batch:
        if k.startswith('rejected') and isinstance(batch[k], torch.Tensor):
            pad_value = -100 if 'labels' in k else 0
            concatenated_key = k.replace('rejected', 'concatenated')
            concatenated_batch[concatenated_key] = torch.cat((
                concatenated_batch[concatenated_key],
                pad_to_length(batch[k], max_length, pad_value=pad_value),
            ), dim=0)
    return concatenated_batch

class DPO_trainer(Trainer):
    def __init__(self, policy_model:nn.Module, args:TrainingArguments, reference_model : Optional[nn.Module], 
                 policy_deepspeed_config_path, reference_deepspeed_config_path, tokenizer, **kwargs):
        #self.policy_model = policy_model
        
        self.tokenizer = tokenizer
        self.args = args

        self.policy_engine, self.optimizer, _, _ = deepspeed.initialize(
            model= policy_model,
            config=policy_deepspeed_config_path,
            model_parameters= policy_model.parameters(),
        )
        if reference_model:
            #self.reference_model = reference_model
            self.reference_engine, _, _, _ = deepspeed.initialize(
                model= reference_model,
                config=reference_deepspeed_config_path,
                model_parameters= reference_model.parameters(),
            )
        data_iterator_kwargs =  dict(
            names = [self.args.datasets],
            tokenizer = self.tokenizer,
            shuffle = True,
            max_length = self.args.max_length,
            max_prompt_length = self.args.max_prompt_length,
            sft_mode = self.args.loss_name == 'sft',
        )
        self.train_dataset_iterator = get_batch_iterator(**data_iterator_kwargs, split='train', 
                                                         n_epochs=self.args.num_train_epochs, n_examples=self.args.num_examples,
                                                         batch_size=self.args.train_batch_size, cache_dir=get_local_dir(self.args.cache_dirs),file_path=self.args.custom_dataset_path)
        self.eval_dataset_iterator = get_batch_iterator(**data_iterator_kwargs, split='test',
                                                        n_epochs=self.args.num_train_epochs, n_examples=self.args.num_examples,
                                                        batch_size=self.args.eval_batch_size, cache_dir=get_local_dir(self.args.cache_dirs),file_path=self.args.custom_dataset_path)
        
        self.train_batches = list(self.train_dataset_iterator)
        self.eval_batches = list(self.eval_dataset_iterator)
        self.eval_batches = self.eval_batches[:self.args.n_eval_batches]
        n_train_examples = len(self.train_batches)

        # for name, param in self.policy_engine.named_parameters():
        #     print(name, param.requires_grad, param.grad)

    def get_batch_samples(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Generate samples from the policy (and reference model, if doing DPO training) for the given batch of inputs."""
        # print("self.policy_model.device:", self.policy_model.device)
        # print("batch.device:", batch["prompt_input_ids"].device)
        #print("batch:", batch)
        rank = dist.get_rank()
        policy_output_decoded = []
        reference_output_decoded = []
        if rank==0:
            batch = {key: value.to(rank) if torch.is_tensor(value) else value for key, value in batch.items()}
            policy_output = self.policy_engine.generate(
                input_ids = batch["prompt_input_ids"], 
                attention_mask = batch["prompt_attention_mask"],
                max_length = self.args.max_length,
                do_sample = True, 
                pad_token_id = self.tokenizer.pad_token_id, 
                num_return_sequences = self.args.n_eval_model_samples,
            )
            if self.args.loss_name in {'dpo', 'ipo', 'scpd', 'sipa'}:
                reference_output = self.reference_engine.generate(
                    input_ids = batch["prompt_input_ids"], attention_mask = batch["prompt_attention_mask"], max_length = self.args.max_length,
                    do_sample = True, pad_token_id = self.tokenizer.pad_token_id, num_return_sequences = self.args.n_eval_model_samples,
                )
                reference_output_decoded = self.tokenizer.batch_decode(reference_output, skip_special_tokens=True)
            else: 
                reference_output_decoded = []
            policy_output_decoded = self.tokenizer.batch_decode(policy_output, skip_special_tokens=True)

        return policy_output_decoded, reference_output_decoded

    def concatenated_forward(self, model:nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:

        #print("batch:", batch)
        concatenated_batch = concatenated_inputs(batch)
    
        device = next(model.parameters()).device
        for key, value in concatenated_batch.items():
            if isinstance(value, torch.Tensor):
                 #print(f"Device of {key}: {value.device}")
                 #print("concatenated_batch:", concatenated_batch[key], concatenated_batch[key].shape)
                 concatenated_batch[key] = concatenated_batch[key].to(device)
            else:
                print(f"{key} is not a tensor.")
        # for param in model.parameters():
        #     print(param.device)
        
        all_logits = model(concatenated_batch['concatenated_input_ids'], attention_mask=concatenated_batch
        ['concatenated_attention_mask']).logits.to(torch.float32)
        all_logps = _get_batch_logps(all_logits, concatenated_batch['concatenated_labels'], average_log_prob=False)
        avg_logps = _get_batch_logps(all_logits, concatenated_batch['concatenated_labels'], average_log_prob=True)

        def cross_entropy_loss(logits, labels):
            loss_fct = nn.CrossEntropyLoss()
            return loss_fct(logits.view(-1,logits.size(-1)), labels.view(-1))
        
        chosen_logps = avg_logps[:batch['chosen_input_ids'].shape[0]]
        rejected_logps = avg_logps[batch['chosen_input_ids'].shape[0]:]
        chosen_logits = all_logits[:batch['chosen_input_ids'].shape[0]]
        rejected_logits = all_logits[batch['chosen_input_ids'].shape[0]:]

        #chosen_logps, rejected_logps = scale_logps(chosen_logps,rejected_logps)

        #ce_loss = cross_entropy_loss(chosen_logits, concatenated_batch['concatenated_labels'][:batch['chosen_input_ids'].shape[0]])
        sft_loss = self.sft_loss(all_logits, concatenated_batch['concatenated_labels'])
        #sft_loss =0 
        return chosen_logps, rejected_logps, sft_loss, chosen_logits, rejected_logits
    
    def sft_loss(self, all_logits, labels):
        all_logps = _get_batch_logps(all_logits, labels, average_log_prob=True)
        return -all_logps.mean()

    def compute_loss(self, batch: Dict[str, Union[List, torch.LongTensor]], training_args: TrainingArguments, train=True):

        metrics = {}
        train_test = 'train' if train else 'eval'

        if training_args.loss_name =='orpo': 
            policy_chosen_logps, policy_rejected_logps, sft_loss, chosen_logits, rejected_logits = self.concatenated_forward(self.policy_engine, batch)
            losses, chosen_rewards, rejected_rewards, log_odd_ratio, log_odds_chosen = odd_ratio_loss(policy_chosen_logps, policy_rejected_logps, training_args.beta)
            orpo_loss = losses.mean()
            reward_accuracies = (chosen_rewards > rejected_rewards).float()

            metrics[f'rewards_{train_test}/chosen'] = chosen_rewards.cpu().float().numpy().tolist()
            metrics[f'rewards_{train_test}/rejected'] = rejected_rewards.cpu().float().numpy().tolist()
            metrics[f'rewards_{train_test}/accuracy'] = reward_accuracies.cpu().float().numpy().tolist()
            metrics[f'rewards_{train_test}/margins'] = (chosen_rewards - rejected_rewards).cpu().float().numpy().tolist()
            metrics[f'log_odds_ratio_{train_test}'] = [log_odd_ratio]
            metrics[f'log_odds_chosen_{train_test}'] = [log_odds_chosen]

        if training_args.loss_name in {'dpo', 'ipo','scpd','sipa'}:
            policy_chosen_logps, policy_rejected_logps, sft_loss, _, _ = self.concatenated_forward(self.policy_engine, batch)
            with torch.no_grad():
                reference_chosen_logps, reference_rejected_logps, _, _, _ = self.concatenated_forward(self.reference_engine, batch)
            
            if training_args.loss_name == 'dpo':
                loss_kwargs = {'beta': training_args.beta, 'label_smoothing': training_args.label_smoothing, 'loss_name':'dpo', 'reference_free': training_args.reference_free}
            elif training_args.loss_name == 'ipo':
                loss_kwargs = {'beta':training_args.beta, 'loss_name':'ipo'}
            elif training_args.loss_name == 'scpd':
                loss_kwargs = {'beta':training_args.beta, 'tau': 100, 'loss_name':'scpd','reference_free': training_args.reference_free}
            elif training_args.loss_name == 'sipa':
                loss_kwargs = {'beta':training_args.beta, 'loss_name':'sipa','reference_free': training_args.reference_free}
            else:
                raise ValueError(f'unknown loss {training_args.loss_name}')
        
            losses, chosen_rewards, rejected_rewards = preference_loss(
                policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, **loss_kwargs
            )
            reward_accuracies = (chosen_rewards > rejected_rewards).float()
            
            metrics[f'rewards_{train_test}/chosen'] = chosen_rewards.cpu().float().numpy().tolist()
            metrics[f'rewards_{train_test}/rejected'] = rejected_rewards.cpu().float().numpy().tolist()
            metrics[f'rewards_{train_test}/accuracy'] = reward_accuracies.cpu().float().numpy().tolist()
            metrics[f'rewards_{train_test}/margins'] = (chosen_rewards - rejected_rewards).cpu().float().numpy().tolist()

        elif training_args.loss_name == 'sft':
            device = next(self.policy_engine.parameters()).device
            policy_chosen_logits = self.policy_engine(batch['chosen_input_ids'].to(device), attention_mask=batch['chosen_attention_mask']).logits.to(torch.float32)
            policy_chosen_logps = _get_batch_logps(policy_chosen_logits, batch['chosen_labels'].to(device), average_log_prob=False)

            losses = -policy_chosen_logps
        

        metrics[f'logps_{train_test}/chosen'] = policy_chosen_logps.cpu().detach().float().numpy().tolist()
        if training_args.loss_name in {'dpo', 'ipo','scpd','sipa'}:
            metrics[f'logps_{train_test}/rejected'] = policy_rejected_logps.cpu().detach().float().numpy().tolist()
        metrics[f'loss/{train_test}'] = losses.cpu().detach().float().numpy().tolist()
        #metrics[f'sft_loss/{train_test}'] = sft_loss.cpu().detach().float().numpy().tolist()
        if training_args.loss_name == 'orpo':
            tqdm.write(f"sft_loss: {sft_loss}")
            loss = sft_loss - orpo_loss 
            return loss.mean(), metrics
        
        elif training_args.loss_name == 'scpd':
            #SCPD loss
            return losses.mean() + 0.05 *sft_loss, metrics
            #return losses.mean(), metrics
            #SFT loss
        #DPO loss
        elif training_args.loss_name == 'sipa':
            #SIPA loss
            return losses.mean() + 0.05 *sft_loss, metrics
        else: 
            return losses.mean(), metrics
        
    def train(self):

        torch.manual_seed(self.args.seed)
        np.random.seed(self.args.seed)
        random.seed(self.args.seed)

        if self.args.loss_name in {'dpo', 'ipo','scpd','sipa'}:
            self.reference_engine.eval()
        
        self.example_counter = 0 
        self.batch_counter = 0
        last_log = None
        rank = dist.get_rank()
        ## Start Evaluation before training
        if self.args.wandb_enabled and rank==0:
            wandb.init(project=self.args.wandb_project, name= self.args.wandb_name)
        for batch in tqdm(self.train_batches, desc = 'Training'):
            
            if self.example_counter % self.args.eval_every == 0 and (self.example_counter > 0 or self.args.do_eval_at_start):
                self.policy_engine.eval()

                all_eval_metrics = defaultdict(list)
                if self.args.sample_during_eval:
                    all_policy_samples, all_reference_samples = [], []
                    policy_text_table = wandb.Table(columns=["step", "prompt", "sample"])
                    if self.args.loss_name in {'dpo', 'ipo','scpd', 'sipa'}:
                        reference_text_table = wandb.Table(columns=["step", "prompt", "sample"])
                
                for eval_batch in tqdm(self.eval_batches, desc = 'Evaluating'):
                    with torch.no_grad():
                        _, eval_metrics = self.compute_loss(eval_batch, self.args, train=False)

                    for k, v in eval_metrics.items():
                        if isinstance(v, (list, tuple)):
                            all_eval_metrics[k].extend(v)
                        else:
                            # Handle scalar values (like floats)
                            all_eval_metrics[k].append(v)

                if self.args.sample_during_eval:
                    if self.args.n_eval_model_samples < self.args.eval_batch_size:
                        rank0_print(f'Warning: n_eval_model_samples ({self.args.n_eval_model_samples}) < eval_batch_size ({self.args.eval_batch_size}).Sampling from the first complete eval batch of prompts.'
                        )
                        sample_batches = self.eval_batches[:1]
                    else:
                        n_sample_batches = self.args.n_eval_model_samples // self.args.eval_batch_size
                        sample_batches = self.eval_batches[:n_sample_batches]
                    for eval_batch in sample_batches:
                        policy_samples, reference_samples = self.get_batch_samples(eval_batch)
                        all_policy_samples.extend(policy_samples)
                        if self.args.loss_name in {'dpo', 'ipo','scpd','sipa'}:
                            all_reference_samples.extend(reference_samples)
                        
                        for prompt, sample in zip(eval_batch['prompt'], policy_samples):
                            policy_text_table.add_data(self.examples_counter, prompt, sample)
                        if self.args.loss_name in {'dpo', 'ipo','scpd','sipa'}:
                            for prompt, sample in zip(eval_batch['prompt'], reference_samples):
                                reference_text_table.add_data(self.examples_counter, prompt, sample)
                mean_eval_metrics = {k: sum(v) / len(v) for k, v in all_eval_metrics.items()}
                rank0_print(f'eval after {self.example_counter}: {formatted_dict(mean_eval_metrics)}')
                if self.args.sample_during_eval:
                    rank0_print(json.dumps(all_policy_samples[:10], indent=2))
                    if self.args.loss_name in {'dpo', 'ipo','scpd','sipa'}:
                        rank0_print(json.dumps(all_reference_samples[:10], indent=2))

                if self.args.wandb_enabled and rank==0:
                    
                    wandb.log(mean_eval_metrics, step=self.example_counter)

                    if self.args.sample_during_eval:
                        wandb.log({"policy_samples": policy_text_table}, step=self.example_counter)
                        if self.args.loss_name in {'dpo', 'ipo','scpd','sipa'}:
                            wandb.log({"reference_samples": reference_text_table}, step=self.example_counter)
                if self.example_counter > 0:
                    if self.args.debug:
                        rank0_print('skipping save in debug mode')
                    else:
                        output_dir = os.path.join(self.args.run_dir, f'step-{self.example_counter}')
                        rank0_print(f'creating checkpoint to write to {output_dir}...')
                        self.save(output_dir, mean_eval_metrics)
                        print("="*20+"save_done"+"="*20)
        ## End Evaluation before training
        ## Begin Training
            #print("="*20+"checkpoinit-1"+"="*20)
            self.policy_engine.train()

            start_time = time.time()
            batch_metrics = defaultdict(list)

            loss, metrics = self.compute_loss(batch, self.args, train=True)
            print(metrics)
            print(type(metrics))
            #loss.backward()
            #print("="*20+"checkpoinit-2"+"="*20)
            #self.optimizer.step()
            self.policy_engine.backward(loss)
            self.policy_engine.step()
            for name, param in self.policy_engine.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        print(f"Gradient issue detected in {name}")
            self.batch_counter += 1
            self.example_counter += self.args.train_batch_size
            
            for k, v in metrics.items():
                if isinstance(v, (list, tuple)):
                    batch_metrics[k].extend(v)
                else:
                    # Handle scalar values (like floats)
                    batch_metrics[k].append(v)
                    
            step_time = time.time() - start_time
            expamples_per_second = self.args.train_batch_size / step_time
            batch_metrics['examples_per_second'].append(expamples_per_second)

            #print("="*20+"checkpoinit-4"+"="*20)
            
            if last_log is None or time.time() - last_log > self.args.logging_steps:
                mean_train_metrics = {k: sum(v) / len(v) for k, v in batch_metrics.items()}
                mean_train_metrics['counters/examples'] = self.example_counter
                mean_train_metrics['counters/updates'] = self.batch_counter
                rank0_print(f'train stats after {self.example_counter} examples: {formatted_dict(mean_train_metrics)}')
                if rank == 0 and self.args.wandb_enabled:
                    wandb.log(mean_train_metrics, step=self.example_counter)

                last_log = time.time()
            else:
                rank0_print(f'skipping logging after {self.example_counter} examples to avoid logging too frequently')


    def write_state_dict(self, step: int, state: Dict[str, torch.Tensor], metrics: Dict, filename: str, dir_name: Optional[str] = None):
        """Write a checkpoint to disk."""
        if dir_name is None:
            dir_name = os.path.join(self.run_dir, f'LATEST')

        os.makedirs(dir_name, exist_ok=True)
        output_path = os.path.join(dir_name, filename)
        rank0_print(f'writing checkpoint to {output_path}...')
        torch.save({
            'step_idx': step,
            'state': state,
            'metrics': metrics if metrics is not None else {},
        }, output_path)
    
    def save(self, output_dir: Optional[str] = None, metrics: Optional[Dict] = None):
        """Save policy, optimizer, and scheduler state to disk using DeepSpeed."""
        
        if output_dir is not None:
            checkpoint_path = os.path.join(output_dir, f'checkpoint-{self.example_counter}')
            os.makedirs(checkpoint_path, exist_ok=True)
        else:
            checkpoint_path = None

        # 保存 DeepSpeed 模型检查点
        policy_state_dict = deepspeed.checkpoint.utils.clone_tensors_for_torch_save(self.policy_engine.module.state_dict())
        self.policy_engine.module.save_pretrained(checkpoint_path, save_state_dict=policy_state_dict)

        #self.policy_engine.save_checkpoint(output_dir, client_state=metrics)

        # 注意：DeepSpeed 的 save_checkpoint 方法会处理模型和优化器的状态
        # 您不需要单独保存优化器和调度器状态

class PreferenceDPO_trainer(DPO_trainer):
    """Extended DPO trainer that supports both original and preference-augmented inputs."""
    
    def __init__(self, policy_model:nn.Module, args:TrainingArguments, reference_model : Optional[nn.Module], 
                 policy_deepspeed_config_path, reference_deepspeed_config_path, tokenizer, 
                 preference_text: str = None, prompt_embeddings: torch.Tensor = None, **kwargs):
        # Call parent constructor
        super().__init__(policy_model, args, reference_model, 
                        policy_deepspeed_config_path, reference_deepspeed_config_path, 
                        tokenizer, **kwargs)
        
        # Support both text-based and embedding-based preferences
        self.preference_text = preference_text
        self.prompt_embeddings = prompt_embeddings
        self.use_prompt_embeddings = prompt_embeddings is not None
        
        # 缓存 embedding layer 信息以避免重复调用
        self._cached_embedding_info = None
        self._cache_embedding_info()
        
        # Override dataset iterators to use preference-augmented versions
        data_iterator_kwargs = dict(
            names = [self.args.datasets],
            tokenizer = self.tokenizer,
            shuffle = True,
            max_length = self.args.max_length,
            max_prompt_length = self.args.max_prompt_length,
            sft_mode = self.args.loss_name == 'sft',
        )
        
        # Add preference configuration based on mode
        if self.use_prompt_embeddings:
            data_iterator_kwargs['prompt_embeddings'] = self.prompt_embeddings
        else:
            data_iterator_kwargs['preference_text'] = self.preference_text
        
        from pro_utils.preference_datasets import get_batch_iterator_with_preference
        
        self.train_dataset_iterator = get_batch_iterator_with_preference(
            **data_iterator_kwargs, split='train', 
            n_epochs=self.args.num_train_epochs, n_examples=self.args.num_examples,
            batch_size=self.args.train_batch_size, 
            cache_dir=get_local_dir(self.args.cache_dirs),
            file_path=self.args.custom_dataset_path
        )
        
        self.eval_dataset_iterator = get_batch_iterator_with_preference(
            **data_iterator_kwargs, split='test',
            n_epochs=self.args.num_train_epochs, n_examples=self.args.num_examples,
            batch_size=self.args.eval_batch_size, 
            cache_dir=get_local_dir(self.args.cache_dirs),
            file_path=self.args.custom_dataset_path
        )
        
        self.train_batches = list(self.train_dataset_iterator)
        self.eval_batches = list(self.eval_dataset_iterator)
        self.eval_batches = self.eval_batches[:self.args.n_eval_batches]
    
    def concatenated_forward_with_preference(self, model:nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]):
        """Run forward pass on both original and preference-augmented inputs."""
        # Process original inputs (π_base)
        chosen_logps, rejected_logps, sft_loss, chosen_logits, rejected_logits = self.concatenated_forward(model, batch)
        
        # Process preference-augmented inputs (π_pref)
        # Check if this batch uses prompt embeddings
        has_prompt_embeddings = batch.get('has_prompt_embeddings', [False])[0]
        
        if has_prompt_embeddings and self.use_prompt_embeddings:
            # For embedding mode: we need to modify the input to include prompt embeddings
            preference_chosen_logps, preference_rejected_logps = self._forward_with_embeddings(model, batch)
        else:
            # For text mode: use regular forward pass with preference-augmented text
            preference_batch = {
                'chosen_input_ids': batch['preference_chosen_input_ids'],
                'chosen_attention_mask': batch['preference_chosen_attention_mask'],
                'chosen_labels': batch['preference_chosen_labels'],
                'rejected_input_ids': batch['preference_rejected_input_ids'],
                'rejected_attention_mask': batch['preference_rejected_attention_mask'],
                'rejected_labels': batch['preference_rejected_labels']
            }
            
            preference_chosen_logps, preference_rejected_logps, _, preference_chosen_logits, preference_rejected_logits = self.concatenated_forward(model, preference_batch)
        
        return chosen_logps, rejected_logps, preference_chosen_logps, preference_rejected_logps, sft_loss
    
    def _forward_with_embeddings(self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]):
        """Forward pass that incorporates prompt embeddings into the input sequence."""
        # Get prompt embeddings and insertion positions
        prompt_embeddings = batch['prompt_embeddings'][0]  # Shape: [embed_dim] or [seq_len, embed_dim]
        embedding_insert_position = batch['embedding_insert_position'][0]
        
        # 使用缓存的embedding信息而不是重复调用get_input_embeddings
        self._refresh_embedding_cache_if_needed()
        
        model_dtype = self._cached_embedding_info['model_dtype']
        model_device = self._cached_embedding_info['model_device']
        
        # Ensure prompt embeddings are on the right device and dtype
        if isinstance(prompt_embeddings, torch.Tensor):
            prompt_embeddings = prompt_embeddings.to(device=model_device, dtype=model_dtype)
        else:
            prompt_embeddings = torch.tensor(prompt_embeddings, device=model_device, dtype=model_dtype)
        
        # Process chosen and rejected separately
        chosen_logps = self._forward_single_with_embeddings(
            model, 
            batch['preference_chosen_input_ids'], 
            batch['preference_chosen_attention_mask'],
            batch['preference_chosen_labels'],
            prompt_embeddings,
            embedding_insert_position
        )
        
        rejected_logps = self._forward_single_with_embeddings(
            model,
            batch['preference_rejected_input_ids'],
            batch['preference_rejected_attention_mask'], 
            batch['preference_rejected_labels'],
            prompt_embeddings,
            embedding_insert_position
        )
        
        return chosen_logps, rejected_logps
    
    def _forward_single_with_embeddings(self, model: nn.Module, input_ids: torch.Tensor, 
                                       attention_mask: torch.Tensor, labels: torch.Tensor,
                                       prompt_embeddings: torch.Tensor, insert_position: int):
        """Forward pass for a single sequence with prompt embeddings."""
        batch_size = input_ids.shape[0]
        
        # 使用缓存的embedding信息
        self._refresh_embedding_cache_if_needed()
            
        embedding_layer = self._cached_embedding_info['embedding_layer']
        model_dtype = self._cached_embedding_info['model_dtype']
        model_device = self._cached_embedding_info['model_device']
        embed_dim = self._cached_embedding_info['embed_dim']
        
        # Handle DeepSpeed wrapper - access underlying model
        if hasattr(model, 'module'):
            actual_model = model.module
        else:
            actual_model = model
        
        # Ensure input tensors are on correct device and dtype
        input_ids = input_ids.to(device=model_device)
        attention_mask = attention_mask.to(device=model_device)
        labels = labels.to(device=model_device)
        
        # Get token embeddings using cached embedding layer
        try:
            if embedding_layer is not None:
                with torch.no_grad():
                    token_embeddings = embedding_layer(input_ids)  # [batch_size, seq_len, embed_dim]
            else:
                # Fallback: 使用模型的embedding层
                token_embeddings = actual_model.get_input_embeddings()(input_ids)
                
        except Exception as e:
            print(f"获取token embeddings时出错: {e}")
            # 最终fallback
            token_embeddings = actual_model.get_input_embeddings()(input_ids)
            model_dtype = token_embeddings.dtype
            model_device = token_embeddings.device
        
        # 更新embed_dim以防缓存失效
        if token_embeddings.shape[-1] != embed_dim:
            embed_dim = token_embeddings.shape[-1]
            self._cached_embedding_info['embed_dim'] = embed_dim
        
        # Ensure prompt embeddings match token embeddings' device and dtype
        prompt_embeddings = prompt_embeddings.to(device=model_device, dtype=model_dtype)
        
        # Prepare prompt embeddings for insertion
        if prompt_embeddings.dim() == 1:
            # Single embedding vector, expand to [1, embed_dim]
            if prompt_embeddings.shape[0] != embed_dim:
                raise ValueError(f"Prompt embedding dimension {prompt_embeddings.shape[0]} doesn't match model embedding dimension {embed_dim}")
            prompt_embeddings = prompt_embeddings.unsqueeze(0)
        elif prompt_embeddings.dim() == 2:
            # Multiple embeddings [seq_len, embed_dim]
            if prompt_embeddings.shape[-1] != embed_dim:
                raise ValueError(f"Prompt embedding dimension {prompt_embeddings.shape[-1]} doesn't match model embedding dimension {embed_dim}")
        else:
            raise ValueError(f"Unexpected prompt embeddings shape: {prompt_embeddings.shape}")
        
        # Expand prompt embeddings to match batch size
        prompt_embeddings = prompt_embeddings.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, prompt_len, embed_dim]
        
        # Insert prompt embeddings at the specified position
        if insert_position == 0:
            # Insert at the beginning
            modified_embeddings = torch.cat([prompt_embeddings, token_embeddings], dim=1)
            modified_attention_mask = torch.cat([
                torch.ones(batch_size, prompt_embeddings.shape[1], device=model_device, dtype=attention_mask.dtype),
                attention_mask
            ], dim=1)
        elif insert_position >= token_embeddings.shape[1]:
            # Insert at the end
            modified_embeddings = torch.cat([token_embeddings, prompt_embeddings], dim=1)
            modified_attention_mask = torch.cat([
                attention_mask,
                torch.ones(batch_size, prompt_embeddings.shape[1], device=model_device, dtype=attention_mask.dtype)
            ], dim=1)
        else:
            # Insert in the middle
            before_embeddings = token_embeddings[:, :insert_position, :]
            after_embeddings = token_embeddings[:, insert_position:, :]
            modified_embeddings = torch.cat([before_embeddings, prompt_embeddings, after_embeddings], dim=1)
            
            before_mask = attention_mask[:, :insert_position]
            after_mask = attention_mask[:, insert_position:]
            modified_attention_mask = torch.cat([
                before_mask,
                torch.ones(batch_size, prompt_embeddings.shape[1], device=model_device, dtype=attention_mask.dtype),
                after_mask
            ], dim=1)
        
        # Modify labels accordingly (add -100 for prompt embedding positions)
        if insert_position == 0:
            modified_labels = torch.cat([
                torch.full((batch_size, prompt_embeddings.shape[1]), -100, device=model_device, dtype=labels.dtype),
                labels
            ], dim=1)
        elif insert_position >= labels.shape[1]:
            modified_labels = torch.cat([
                labels,
                torch.full((batch_size, prompt_embeddings.shape[1]), -100, device=model_device, dtype=labels.dtype)
            ], dim=1)
        else:
            before_labels = labels[:, :insert_position]
            after_labels = labels[:, insert_position:]
            modified_labels = torch.cat([
                before_labels,
                torch.full((batch_size, prompt_embeddings.shape[1]), -100, device=model_device, dtype=labels.dtype),
                after_labels
            ], dim=1)
        
        # Forward pass with modified embeddings
        outputs = actual_model(
            inputs_embeds=modified_embeddings,
            attention_mask=modified_attention_mask,
            labels=modified_labels,
            return_dict=True
        )
        
        # Calculate log probabilities
        logits = outputs.logits
        log_probs = _get_batch_logps(logits, modified_labels)
        
        return log_probs
    
    def preference_augmented_loss(self, policy_chosen_logps, policy_rejected_logps,
                                 policy_pref_chosen_logps, policy_pref_rejected_logps,
                                 reference_chosen_logps, reference_rejected_logps,
                                 reference_pref_chosen_logps, reference_pref_rejected_logps,
                                 beta: float, alpha: float, lambda_kl: float = 0.1, label_smoothing: float = 0.0):
        """Compute loss using both original and preference-augmented inputs."""
        # Calculate P_theta(y_w|x) - P_theta(y_l|x) and P_theta(y_w|x,p) - P_theta(y_l|x,p)
        pi_logratios = torch.clamp(policy_chosen_logps - policy_rejected_logps, min=-10, max=10)
        pi_pref_logratios = torch.clamp(policy_pref_chosen_logps - policy_pref_rejected_logps, min=-10, max=10)
        
        # Calculate P_ref(y_w|x) - P_ref(y_l|x) and P_ref(y_w|x,p) - P_ref(y_l|x,p)
        ref_logratios = reference_chosen_logps - reference_rejected_logps
        ref_pref_logratios = reference_pref_chosen_logps - reference_pref_rejected_logps
        
        dpo_loss = -F.logsigmoid(beta * pi_logratios).mean()

        # Calculate logits for original and preference-augmented inputs
        explicit_to_implicit_logp = pi_pref_logratios.detach() - pi_logratios
        # Clamp to avoid extreme values
        aligned_loss = F.relu(explicit_to_implicit_logp).mean()

        # Reference KL 约束
        logits_pol_base = torch.clamp(torch.stack([policy_chosen_logps, policy_rejected_logps], dim=1), min=-10, max=10)
        logits_ref_base = torch.clamp(torch.stack([reference_chosen_logps, reference_rejected_logps], dim=1), min=-10, max=10)
        logp_pol_base = F.log_softmax(logits_pol_base, dim=1)
        # 注意：PyTorch的kl_div要求第二个参数是概率而不是对数概率
        p_ref_base = F.softmax(logits_ref_base, dim=1).detach() + 1e-8  # 添加小常数避免零概率
        kl_base = F.kl_div(logp_pol_base, p_ref_base, reduction='batchmean', log_target=False)
        
        # Preference KL 约束
        logits_pol_pref = torch.clamp(torch.stack([policy_pref_chosen_logps, policy_pref_rejected_logps], dim=1), min=-10, max=10)
        logits_ref_pref = torch.clamp(torch.stack([reference_pref_chosen_logps, reference_pref_rejected_logps], dim=1), min=-10, max=10)
        logp_pol_pref = F.log_softmax(logits_pol_pref, dim=1)
        p_ref_pref = F.softmax(logits_ref_pref, dim=1).detach() + 1e-8  # 添加小常数避免零概率
        kl_pref = F.kl_div(logp_pol_pref, p_ref_pref, reduction='batchmean', log_target=False)
        
        # 检测并处理NaN值
        if torch.isnan(kl_base):
            print(f"Warning: kl_base is NaN! logits_pol_base: {logits_pol_base}, logits_ref_base: {logits_ref_base}")
            kl_base = torch.tensor(0.0, device=kl_pref.device)
        if torch.isnan(kl_pref):
            print(f"Warning: kl_pref is NaN! logits_pol_pref: {logits_pol_pref}, logits_ref_pref: {logits_ref_pref}")
            kl_pref = torch.tensor(0.0, device=kl_base.device)
            
        kl_div = (kl_base + kl_pref) / 2.0

        # Combine losses from original and preference-augmented inputs
        # You can adjust the weighting as needed
        losses = dpo_loss + alpha * aligned_loss + lambda_kl * kl_div
                
        # Calculate rewards for logging
        chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()
        pref_chosen_rewards = beta * (policy_pref_chosen_logps - reference_pref_chosen_logps).detach()
        pref_rejected_rewards = beta * (policy_pref_rejected_logps - reference_pref_rejected_logps).detach()
        
# 返回各个损失组件，用于监控训练过程
        loss_components = {
            'dpo_loss': dpo_loss.item(),
            'aligned_loss': aligned_loss.item(),
            'kl_div': kl_div.item(),
            'weighted_aligned_loss': (alpha * aligned_loss).item(),
            'weighted_kl_div': (lambda_kl * kl_div).item()
        }
        
        return losses, chosen_rewards, rejected_rewards, pref_chosen_rewards, pref_rejected_rewards, loss_components
    
    def compute_loss(self, batch: Dict[str, Union[List, torch.LongTensor]], training_args: TrainingArguments, train=True):
        """Override compute_loss to handle preference-augmented inputs."""
        metrics = {}
        train_test = 'train' if train else 'eval'

        if training_args.loss_name in {'new_pref_po'}:
            # Get logps for both original and preference-augmented inputs
            policy_chosen_logps, policy_rejected_logps, policy_pref_chosen_logps, policy_pref_rejected_logps, sft_loss = self.concatenated_forward_with_preference(self.policy_engine, batch)
            
            with torch.no_grad():
                reference_chosen_logps, reference_rejected_logps, reference_pref_chosen_logps, reference_pref_rejected_logps, _ = self.concatenated_forward_with_preference(self.reference_engine, batch)
            
            if training_args.loss_name == 'new_pref_po':
                loss_kwargs = {'beta': training_args.beta, 'alpha': training_args.alpha, 'lambda_kl': getattr(training_args, 'lambda_kl', 0.1)}
            
# 修改这里以接收返回的损失组件
            losses, chosen_rewards, rejected_rewards, pref_chosen_rewards, pref_rejected_rewards, loss_components = self.preference_augmented_loss(
                policy_chosen_logps, policy_rejected_logps,
                policy_pref_chosen_logps, policy_pref_rejected_logps,
                reference_chosen_logps, reference_rejected_logps,
                reference_pref_chosen_logps, reference_pref_rejected_logps,
                **loss_kwargs
            )
            
# 将损失组件添加到 metrics 中
            for loss_name, loss_value in loss_components.items():
                metrics[f'loss_components/{loss_name}_{train_test}'] = loss_value
            
            # ... 现有代码 ...
            # 计算指标添加到metrics
            pref_advantage_ratio = torch.exp((policy_pref_chosen_logps - policy_chosen_logps) - (policy_pref_rejected_logps - policy_rejected_logps))
            pref_delta_w = policy_pref_chosen_logps - policy_chosen_logps
            pref_delta_l = policy_pref_rejected_logps - policy_rejected_logps
            
            delta_w_ref_logps = policy_chosen_logps - reference_chosen_logps
            delta_l_ref_logps = policy_rejected_logps - reference_rejected_logps
            ref_advantage_ratio = torch.exp(delta_w_ref_logps - delta_l_ref_logps)

            # 添加到metrics字典，保存每个样本的值而不是均值
            metrics[f'preference_metrics/pref_advantage_ratio_{train_test}'] = pref_advantage_ratio.cpu().detach().float().numpy().tolist()
            metrics[f'preference_metrics/pref_delta_w_{train_test}'] = pref_delta_w.cpu().detach().float().numpy().tolist()
            metrics[f'preference_metrics/pref_delta_l_{train_test}'] = pref_delta_l.cpu().detach().float().numpy().tolist()
            # 对差值也使用逐元素操作保存列表
            metrics[f'preference_metrics/pref_delta_diff_{train_test}'] = (pref_delta_w - pref_delta_l).cpu().detach().float().numpy().tolist()
            
            metrics[f'reference_metrics/ref_advantage_ratio_{train_test}'] = ref_advantage_ratio.cpu().detach().float().numpy().tolist()
            metrics[f'reference_metrics/delta_w_ref_logps_{train_test}'] = delta_w_ref_logps.cpu().detach().float().numpy().tolist()
            metrics[f'reference_metrics/delta_l_ref_logps_{train_test}'] = delta_l_ref_logps.cpu().detach().float().numpy().tolist()
            metrics[f'reference_metrics/ref_delta_diff_{train_test}'] = (delta_w_ref_logps - delta_l_ref_logps).cpu().detach().float().numpy().tolist()


            reward_accuracies = (chosen_rewards > rejected_rewards).float()
            pref_reward_accuracies = (pref_chosen_rewards > pref_rejected_rewards).float()
            
            # Log metrics for both original and preference-augmented inputs
            metrics[f'rewards_{train_test}/chosen'] = chosen_rewards.cpu().float().numpy().tolist()
            metrics[f'rewards_{train_test}/rejected'] = rejected_rewards.cpu().float().numpy().tolist()
            metrics[f'rewards_{train_test}/pref_chosen'] = pref_chosen_rewards.cpu().float().numpy().tolist()
            metrics[f'rewards_{train_test}/pref_rejected'] = pref_rejected_rewards.cpu().float().numpy().tolist()
            metrics[f'rewards_{train_test}/accuracy'] = reward_accuracies.cpu().float().numpy().tolist()
            metrics[f'rewards_{train_test}/pref_accuracy'] = pref_reward_accuracies.cpu().float().numpy().tolist()

        elif training_args.loss_name == 'sft':
            # For SFT mode, just use the parent implementation
            return super().compute_loss(batch, training_args, train)
        
        # Convert to float32 before numpy conversion to handle BFloat16 compatibility
        metrics[f'logps_{train_test}/chosen'] = policy_chosen_logps.cpu().detach().float().numpy().tolist()
        metrics[f'logps_{train_test}/rejected'] = policy_rejected_logps.cpu().detach().float().numpy().tolist()
        metrics[f'logps_{train_test}/pref_chosen'] = policy_pref_chosen_logps.cpu().detach().float().numpy().tolist()
        metrics[f'logps_{train_test}/pref_rejected'] = policy_pref_rejected_logps.cpu().detach().float().numpy().tolist()
        metrics[f'loss/{train_test}'] = losses.cpu().detach().float().numpy().tolist()
        
        if training_args.loss_name in {'new_pref_po'}:
            return losses.mean(),  metrics
        else:
            return losses.mean(), metrics

    def _cache_embedding_info(self):
        """缓存模型的embedding层信息，避免训练时重复调用get_input_embeddings"""
        try:
            # 从policy_engine获取embedding信息
            if hasattr(self.policy_engine, 'module'):
                actual_model = self.policy_engine.module
            else:
                actual_model = self.policy_engine
            
            embedding_layer = actual_model.get_input_embeddings()
            
            self._cached_embedding_info = {
                'embedding_layer': embedding_layer,
                'model_dtype': embedding_layer.weight.dtype,
                'model_device': embedding_layer.weight.device,
                'embed_dim': embedding_layer.weight.shape[1]
            }
            print(f"缓存embedding信息成功: dtype={self._cached_embedding_info['model_dtype']}, device={self._cached_embedding_info['model_device']}, embed_dim={self._cached_embedding_info['embed_dim']}")
            
        except Exception as e:
            print(f"缓存embedding信息失败: {e}")
            # 使用默认值作为fallback
            self._cached_embedding_info = {
                'embedding_layer': None,
                'model_dtype': torch.float32,
                'model_device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                'embed_dim': 4096  # 默认的embedding维度
            }

    def _refresh_embedding_cache_if_needed(self):
        """如果embedding缓存失效则重新缓存"""
        try:
            if self._cached_embedding_info is None:
                self._cache_embedding_info()
                return
                
            # 检查缓存的embedding_layer是否仍然有效
            cached_layer = self._cached_embedding_info.get('embedding_layer')
            if cached_layer is not None:
                # 简单检查：确保权重仍然可访问
                _ = cached_layer.weight.shape
        except:
            # 如果访问失败，重新缓存
            print("检测到embedding缓存失效，正在重新缓存...")
            self._cache_embedding_info()
