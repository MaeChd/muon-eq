import argparse
import json
import os
import random

import numpy as np
import torch
import torch.distributed as dist
import torch_npu
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from models.llama import LlamaForCausalLM
from scripts.pretrain_c4_dist import (
    _log_model_dtype_summary,
    _register_dtype_debug_hooks,
    evaluate_model,
    logger,
    materialize_val_batches,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a final checkpoint on the validation split once."
    )
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--model_config", type=str, default=None)
    parser.add_argument("--use_hf_model", action="store_true", default=False)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True,
                        help="Per-rank training micro batch size used as the default eval batch size.")
    parser.add_argument("--eval_batch_size", type=int, default=None,
                        help="Per-rank eval batch size. Defaults to --batch_size.")
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--tokenizer_batch_size", type=int, default=64)
    parser.add_argument("--text_buffer_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--cache_eval_data", action="store_true", default=False)
    parser.add_argument("--single_gpu", action="store_true", default=False)
    parser.add_argument("--result_path", type=str, default=None)
    return parser.parse_args()


def resolve_model_config(args):
    if args.model_config:
        return args.model_config

    checkpoint_config = os.path.join(args.checkpoint_dir, "config.json")
    if os.path.isfile(checkpoint_config):
        return checkpoint_config

    raise FileNotFoundError(
        "Could not find config.json in the checkpoint directory. "
        "Pass --model_config explicitly."
    )


def default_run_name(checkpoint_dir):
    checkpoint_dir = os.path.normpath(checkpoint_dir)
    base = os.path.basename(checkpoint_dir)
    if base == "final":
        return f"eval-final-{os.path.basename(os.path.dirname(checkpoint_dir))}"
    return f"eval-final-{base}"


def load_checkpoint_state(checkpoint_dir):
    checkpoint_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint weights not found: {checkpoint_path}")
    return torch.load(checkpoint_path, map_location="cpu"), checkpoint_path


def load_training_state(checkpoint_dir):
    training_state_path = os.path.join(checkpoint_dir, "training_state.json")
    if not os.path.isfile(training_state_path):
        return {}
    with open(training_state_path) as f:
        return json.load(f)


def write_result(path, result):
    result_dir = os.path.dirname(path)
    if result_dir:
        os.makedirs(result_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)


def main():
    args = parse_args()
    args.model_config = resolve_model_config(args)
    args.name = args.name or default_run_name(args.checkpoint_dir)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.single_gpu:
        global_rank = 0
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = 1
        torch.npu.set_device(local_rank)
        device = f"npu:{local_rank}"
    else:
        if "LOCAL_RANK" not in os.environ:
            raise RuntimeError("Distributed evaluation expects torchrun to set LOCAL_RANK.")
        global_rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.npu.set_device(local_rank)
        dist.init_process_group(backend="hccl", rank=global_rank, world_size=world_size)
        device = f"npu:{local_rank}"

    if global_rank != 0:
        logger.remove()

    training_state = load_training_state(args.checkpoint_dir)
    eval_batch_size = args.eval_batch_size or args.batch_size

    if global_rank == 0:
        logger.info("*" * 40)
        logger.info("Starting final-checkpoint evaluation")
        for key, value in vars(args).items():
            logger.info(f"{key:30} {value}")
        if training_state:
            logger.info(f"checkpoint_update_step       {training_state.get('update_step')}")
            logger.info(f"checkpoint_tokens_seen      {training_state.get('tokens_seen')}")
        logger.info("*" * 40)

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path,
        model_max_length=args.max_length,
        use_fast=True,
    )
    pad_idx = tokenizer.pad_token_id

    model_config = AutoConfig.from_pretrained(args.model_config)
    if args.use_hf_model:
        model = AutoModelForCausalLM.from_config(model_config)
    else:
        model = LlamaForCausalLM(model_config)

    state_dict, checkpoint_path = load_checkpoint_state(args.checkpoint_dir)
    model.load_state_dict(state_dict, strict=True)

    if args.dtype in ["bf16", "bfloat16"]:
        model = model.to(device=device, dtype=torch.bfloat16)
    else:
        model = model.to(device=device)

    _log_model_dtype_summary(model)
    dtype_debug_hooks = _register_dtype_debug_hooks(model, global_rank)

    cached_eval = None
    if args.cache_eval_data:
        logger.info("Materializing validation batches into memory...")
        cached_eval = materialize_val_batches(
            tokenizer, pad_idx, global_rank, world_size, args, eval_batch_size
        )
        logger.info(f"Cached {len(cached_eval[0])} eval batches.")

    eval_loss, eval_tokens = evaluate_model(
        model,
        tokenizer,
        pad_idx,
        global_rank,
        world_size,
        device,
        eval_batch_size,
        args,
        cached_eval=cached_eval,
    )

    result = {
        "checkpoint_dir": args.checkpoint_dir,
        "checkpoint_path": checkpoint_path,
        "eval_batch_size": eval_batch_size,
        "eval_loss": eval_loss,
        "eval_tokens": eval_tokens,
        "model_config": args.model_config,
        "name": args.name,
        "seed": args.seed,
        "tokens_seen": training_state.get("tokens_seen"),
        "update_step": training_state.get("update_step"),
        "world_size": world_size,
    }

    if global_rank == 0:
        logger.info(f"Eval loss: {eval_loss}")
        logger.info(f"Eval tokens: {eval_tokens}")
        logger.info("FINAL_EVAL_RESULT " + json.dumps(result, sort_keys=True))
        if args.result_path:
            write_result(args.result_path, result)
            logger.info(f"Saved eval result to {args.result_path}")

    for hook in dtype_debug_hooks:
        hook.remove()

    if not args.single_gpu:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
