import os
import argparse
import torch
from datasets import Dataset
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)
from datasets import load_dataset
from peft import LoraConfig
from optimizers.adamw_variants.c_adamw import AdamW as CAdamW
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="Qwen/Qwen2-0.5B-Instruct")
parser.add_argument("--max_length", type=int, default = 16384)
parser.add_argument("--output_dir", type=str, default="gkd-model")
parser.add_argument("--per_device_train_batch_size", type=int, default=1)
parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
parser.add_argument("--resume_from_checkpoint", action="store_true", default=False)
parser.add_argument("--lora", action="store_true")
parser.add_argument("--cautious", action="store_true")
parser.add_argument("--num_gpus", default = 8, type=int)
args = parser.parse_args()

# qwq_dataset = load_dataset("amphora/QwQ-LongCoT-130K-2", split = "train")
qwq_dataset = load_dataset("PowerInfer/QWQ-LONGCOT-500K", split = "train")
# qwq_dataset = load_dataset("PowerInfer/LONGCOT-Refine-500K", split = "train")
messages = []
for each in qwq_dataset:
    msg = [
        {"role": "system", "content": "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step."},
        {"role": "user", "content": each["prompt"]},
        {"role": "assistant", "content": each["response"]},
    ]
    messages.append(msg)

TRAIN_SPLIT_RATIO = 0.99
train_size = int(TRAIN_SPLIT_RATIO * len(messages))
eval_size = len(messages) - train_size

tokenizer = AutoTokenizer.from_pretrained(args.model)

# The model to optimise
model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="auto")



### Real Dataset
train_dataset = Dataset.from_dict({"messages":messages[:train_size]})
eval_dataset = Dataset.from_dict({"messages":messages[train_size:]})
training_args = SFTConfig(
    output_dir=args.output_dir,
    max_seq_length=args.max_length,
    per_device_train_batch_size=args.per_device_train_batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    gradient_checkpointing = args.gradient_checkpointing,
    save_steps = 100,
    save_total_limit = 5,
    report_to="wandb"
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

response_template = "<|im_start|>assistant\n"

collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

num_training_steps = train_size // args.per_device_train_batch_size // args.num_gpus // args.gradient_accumulation_steps
num_warmup_steps = int(0.1 * num_training_steps)
if args.cautious:
    optim = CAdamW(model.parameters(), lr=5e-5, weight_decay = 0.01, betas = (0.9, 0.95))
else:
    optim = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay = 0.01, betas = (0.9, 0.95))
scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps, num_training_steps)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    processing_class=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=None, # doesn't support lora yet
    data_collator=collator,
    optimizers = (optim, scheduler)
)
trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
