from typing import Optional
from utils import prompt_dataset, Custome_datacollator
from transformers import Trainer, TrainingArguments, TextDataset, DataCollatorForSeq2Seq, AutoTokenizer, AutoModelForCausalLM,DataCollatorForLanguageModeling,\
AutoModelForSeq2SeqLM, T5Tokenizer
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import json
import os
import deepspeed
deepspeed.ops.op_builder.CPUAdamBuilder().load() 


class QA_trainer(Trainer):

    def __init__(self, model, args, train_dataset, eval_dataset, tokenizer, **kwargs):
        super(QA_trainer, self).__init__(
            model=model,    
            args=args,     
            train_dataset=train_dataset,   
            eval_dataset= eval_dataset,     
            tokenizer=tokenizer,        
            **kwargs        
        )


    def compute_loss(self, model, inputs, return_outputs=False):
        # 从输入中获取input_ids、labels和attention_mask
        input_ids = inputs["input_ids"]
        labels  = inputs["labels"]
        attention_mask = inputs["attention_mask"]
        # 将这些输入传递给模型，获取输出
        output_logits = model(input_ids, attention_mask=attention_mask, labels= input_ids)
        # print("input_ids:",input_ids)
        # print("output_logits:",output_logits)
        # print("labels:",labels)
        # 从输出中获取损失
        loss = output_logits.loss
        # 如果return_outputs为True，返回损失和输出，否则只返回损失
        return (loss, output_logits) if return_outputs else loss
    
    def compute_metrics(self,pred):
        predictions, label_ids = pred.predictions, pred.label_ids
        accuracy = accuracy_score(label_ids,predictions)
        predictions = np.argmax(predictions, axis=1)
        print("label_ids,predictions:",label_ids,predictions)
        return {"costumed_acc": accuracy}
    
    def save_model(self, output_dir: str | None = None, _internal_call: bool = False):
        if output_dir is None:
            output_dir = self.args.output_dir
        self.model.save_pretrained(output_dir)

    def _save_optimizer_and_scheduler(self, output_dir):
        if self.deepspeed:
            self.deepspeed.save_checkpoint(output_dir)
        else:
            self.optimizer.save_pretrained(output_dir)
            self.lr_scheduler.save_pretrained(output_dir)

    def _save_checkpoint(self, model, trial=None, metrics=None):
        checkpoint_folder = f"checkpoint-{self.state.global_step}"
        output_dir = os.path.join(self.args.output_dir, checkpoint_folder)
        self.save_model(output_dir)
        #self.model.save_pretrained(output_dir)
        self._save_optimizer_and_scheduler(output_dir)

traing_args = TrainingArguments(
    output_dir = './model_output/sft-llama2-r8-2',
    save_strategy="epoch",
    evaluation_strategy="epoch",
    eval_steps= 1000,
    save_steps = 1000,
    num_train_epochs= 10,
    gradient_checkpointing=True,
    per_device_eval_batch_size= 1,
    per_device_train_batch_size= 1,
    gradient_accumulation_steps= 1,
    learning_rate= 1e-3,
    logging_dir= './logs/sft-llama2-r8-2',
    logging_strategy="epoch",
    save_total_limit = 2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    deepspeed=json.load(open("./deepspeed_config/ds_config.json", encoding="utf8")),
    bf16=True,
    report_to="tensorboard",
    #resume_from_checkpoint="./model_output/rm-llama2-r16/checkpoint-1150",
)
#lora config
lora_config = LoraConfig(
    peft_type= "LORA",
    task_type="CASUAL_LM",
    r = 8,
    lora_alpha = 16,
    target_modules =["q_proj","v_proj"],
    #target_modules =["query_key_value"],
    lora_dropout= 0.1
)

#model_path = '/mnt/disk2/wbr/streamingLM/streaming-llm-main/models/vicuna-7b-v1.5'
model_path = "/mnt/disk2/wbr/LLM_models/llama-2-7b-hf"
adapter_config = ''
adapter_model_path = ''

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    torch_dtype = torch.float16,
    trust_remote_code=True,
)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_path)

# for name, param in model.named_parameters():
#     print(name, param.requires_grad, param.grad)



lora_model = get_peft_model(model, peft_config= lora_config)

lora_model.enable_input_require_grads()
lora_model.print_trainable_parameters()
# for name, param in model.named_parameters():
#     print(name, param.requires_grad, param.grad)

train_path = '../datasets/NoRobots/train_sft-00000-of-00001-8aba5401a3b757f5.parquet'
valid_path = '../datasets/NoRobots/test_sft-00000-of-00001-fe658ed8e3578d4a.parquet'
#test_path = ''
#model_checkpoint_path = './model_output/checkpoint-500'

train_dataset = prompt_dataset.TextDataset_NoRobots(train_path, tokenizer=tokenizer)
valid_dataset = prompt_dataset.TextDataset_NoRobots(valid_path, tokenizer=tokenizer)
#test_dataset = prompt_dataset.TextDataset(test_path, tokenizer=tokenizer)
#print("train_dataset:",train_dataset)
#print("train_dataset[0]:",train_dataset[0])
#sample_text, sample_label = train_dataset[0]
#print(sample_text)
#print(sample_label)

data_collator = Custome_datacollator.DataCollatorFor_CasualLM(tokenizer=tokenizer,padding=True)


trainer = QA_trainer(
    model= model,
    args= traing_args,
    train_dataset= train_dataset,
    eval_dataset= valid_dataset,
    data_collator= data_collator,
    tokenizer= tokenizer,
)

if __name__ == "__main__":
    trainer.train()
    trainer.evaluate()
    # results = trainer.evaluate()
    # print(results)