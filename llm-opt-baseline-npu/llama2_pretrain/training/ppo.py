# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import shutil

import torch
import transformers
from accelerate import PartialState
from datasets import load_dataset
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
)

from trl import (
    ModelConfig,
    ScriptArguments,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE
from training.ppo_config import PPOConfig
from training.ppo_trainer import PPOTrainer
from optimizers.adamw_variants.c_adamw import AdamW as C_AdamW
"""
python -i examples/scripts/ppo/ppo.py \
    --dataset_name trl-internal-testing/descriptiveness-sentiment-trl-style \
    --dataset_train_split descriptiveness \
    --learning_rate 3e-6 \
    --output_dir models/minimal/ppo \
    --per_device_train_batch_size 64 \
    --gradient_accumulation_steps 1 \
    --total_episodes 10000 \
    --model_name_or_path EleutherAI/pythia-1b-deduped \
    --missing_eos_penalty 1.0

accelerate launch --config_file examples/accelerate_configs/deepspeed_zero3.yaml \
    examples/scripts/ppo/ppo.py \
    --dataset_name trl-internal-testing/descriptiveness-sentiment-trl-style \
    --dataset_train_split descriptiveness \
    --output_dir models/minimal/ppo \
    --num_ppo_epochs 1 \
    --num_mini_batches 1 \
    --learning_rate 3e-6 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --total_episodes 10000 \
    --model_name_or_path EleutherAI/pythia-1b-deduped \
    --sft_model_path EleutherAI/pythia-1b-deduped \
    --reward_model_path EleutherAI/pythia-1b-deduped \
    --local_rollout_forward_batch_size 1 \
    --missing_eos_penalty 1.0
"""

if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, PPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_into_dataclasses()
    # remove output_dir if exists
    shutil.rmtree(training_args.output_dir, ignore_errors=True)

    ################
    # Model & Tokenizer
    ################
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, padding_side="left", trust_remote_code=model_args.trust_remote_code
    )
    rm_tokenizer = AutoTokenizer.from_pretrained(
        training_args.reward_model_path, padding_side="left", trust_remote_code=model_args.trust_remote_code
    )

    # tokenizer.add_special_tokens({"pad_token": "[PAD]"}) # confirm what padding token works for qwen
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE # also figure out what chat template is that
    value_model = AutoModelForSequenceClassification.from_pretrained(
        training_args.sft_model_path, trust_remote_code=model_args.trust_remote_code, num_labels=1
    )
    if training_args.prm:
        reward_model = AutoModel.from_pretrained(
            training_args.reward_model_path, trust_remote_code=True
        )
    else:
        reward_model = AutoModelForSequenceClassification.from_pretrained(
            training_args.reward_model_path, trust_remote_code=model_args.trust_remote_code, num_labels=1
        )
    policy = AutoModelForCausalLM.from_pretrained(
        training_args.sft_model_path, trust_remote_code=model_args.trust_remote_code
    )
    
    peft_config = get_peft_config(model_args)
    if peft_config is None:
        ref_policy = AutoModelForCausalLM.from_pretrained(
            training_args.sft_model_path, trust_remote_code=model_args.trust_remote_code
        )
    else:
        ref_policy = None
    
    ################
    # Dataset
    ################
    dataset = load_dataset(
        script_args.dataset_name, name=script_args.dataset_config, split=script_args.dataset_train_split
    )
    eval_samples = 100
    train_dataset = dataset.select(range(len(dataset) - eval_samples))
    eval_dataset = dataset.select(range(len(dataset) - eval_samples, len(dataset)))
    dataset_text_field = training_args.dataset_text_field

    def prepare_dataset(dataset, tokenizer):
        """pre-tokenize the dataset before training; only collate during training"""

        def tokenize(element):

            messages = [[{"role": "system", "content": "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step."},{"role": "user", "content": prompt}] for prompt in element[dataset_text_field]]
            # Tokenize the input
            texts = [tokenizer.apply_chat_template(each, tokenize=False, add_generation_prompt=True) for each in messages]
            outputs = tokenizer(texts, padding=False)
            return {"input_ids": outputs["input_ids"]}

        return dataset.map(
            tokenize,
            batched=True,
            remove_columns=dataset.column_names,
            num_proc=training_args.dataset_num_proc,
         )

    # Compute that only on the main process for faster data processing.
    # see: https://github.com/huggingface/trl/pull/1255
    with PartialState().local_main_process_first():
        train_dataset = prepare_dataset(train_dataset, tokenizer)
        eval_dataset = prepare_dataset(eval_dataset, tokenizer)


    num_training_steps = len(train_dataset) // training_args.per_device_train_batch_size // training_args.num_gpus // training_args.gradient_accumulation_steps * training_args.num_train_epochs
    
    trainable_parameters = [each for each in policy.parameters()] + [each for each in value_model.parameters()]
    if training_args.custom_optim == "c_adamw":
        optim = C_AdamW(trainable_parameters, betas = (0.9, 0.999), weight_decay= training_args.weight_decay, lr = training_args.learning_rate)
    else:
        optim = transformers.AdamW(trainable_parameters, betas = (0.9, 0.999), weight_decay = training_args.weight_decay, lr = training_args.learning_rate)
    scheduler = transformers.get_linear_schedule_with_warmup(optim, int(num_training_steps * training_args.warmup_ratio) , num_training_steps)
    ################
    # Training
    ################
    trainer = PPOTrainer(
        args=training_args,
        processing_class=tokenizer,
        rm_processing_class = rm_tokenizer,
        model=policy,
        ref_model=ref_policy,
        reward_model=reward_model,
        value_model=value_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        optimizers =  [optim, scheduler]
    )
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)

    trainer.generate_completions()
