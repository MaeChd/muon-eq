import os
import time
import json
import random
import argparse
import shutil
import sys
import math
import numpy as np
import hashlib
import socket

import torch
import torch_npu
torch.npu.config.allow_internal_format = True
import torch.nn as nn
import torch.utils.data
import torch.distributed as dist

import transformers
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaForCausalLM as HF_LlamaForCausalLM

import datasets
import datasets.distributed

try:
    import wandb
except ImportError:
    wandb = None

try:
    import swanlab
except ImportError:
    swanlab = None

from tqdm import tqdm
from loguru import logger
from optimizers.adamw_variants.mars import MARS
from training import utils as training_utils, args as args_utils
from data.dataloader import PreprocessedIterableDataset
from models.llama import LlamaForCausalLM
from optimizers.muon_variants.muon import Muon
from optimizers.misc.torch_optimizer import AdamW
import glob
from optimizers.muon_variants.muon_mvr import MuonMVR
from optimizers.memory_efficient.foam import FOAM
from optimizers.muon_variants.muoneq import MuonEq

import copy


transformers.logging.set_verbosity_error()

import contextlib

DATA_STATE_FILENAME = "data_state.pt"
RESUME_MANIFEST_FILENAME = "resume_manifest.json"
TRAIN_DATA_GLOB = "c4-train.*.json.gz"
VAL_DATA_GLOB = "c4-validation.*.json.gz"
TRAIN_SHUFFLE_SEED = 42


class ExperimentTracker:
    def __init__(self, backend=None):
        self.backend = backend
        self.enabled = False

    def init(self, project, run_name, tags=None):
        if self.backend is None:
            return

        if self.backend == "wandb":
            if wandb is None:
                raise ImportError("wandb is not installed. Install wandb or use --report_to swanlab.")
            init_kwargs = {
                "project": project,
                "name": run_name,
            }
            parsed_tags = _parse_tags(tags)
            if parsed_tags:
                init_kwargs["tags"] = parsed_tags
            wandb.init(**init_kwargs)
        elif self.backend == "swanlab":
            if swanlab is None:
                raise ImportError("swanlab is not installed. Install swanlab or use --report_to wandb.")
            swanlab.init(project=project, experiment_name=run_name)
        else:
            raise ValueError(f"Unsupported tracker backend: {self.backend}")

        self.enabled = True

    def update_config(self, config):
        if not self.enabled:
            return
        if self.backend == "wandb":
            wandb.config.update(config, allow_val_change=True)
        elif self.backend == "swanlab":
            swanlab.config.update(config, allow_val_change=True)

    def log(self, metrics, step=None):
        if not self.enabled:
            return
        if self.backend == "wandb":
            wandb.log(metrics, step=step)
        elif self.backend == "swanlab":
            swanlab.log(metrics, step=step)

    def finish(self):
        if not self.enabled:
            return
        if self.backend == "wandb":
            wandb.finish()
            return
        finish_fn = getattr(swanlab, "finish", None)
        if callable(finish_fn):
            finish_fn()


def _parse_tags(tags):
    if tags is None:
        return None
    parsed_tags = [tag.strip() for tag in tags.split(",") if tag.strip()]
    return parsed_tags or None


def _default_project_name(model_config_path):
    model_name = os.path.splitext(os.path.basename(model_config_path))[0]
    return f"adabeta-c4-{model_name}"


def _resolve_tracking_project(args):
    default_project = _default_project_name(args.model_config)
    if args.report_to == "wandb":
        return args.wandb_proj or args.swanlab_proj or default_project
    if args.report_to == "swanlab":
        return args.swanlab_proj or args.wandb_proj or default_project
    return default_project

def _unwrap_model(m):
    return m.module if isinstance(m, torch.nn.parallel.DistributedDataParallel) else m

def _clone_state_dict_to_cpu(m):
    return {k: v.detach().clone() for k, v in _unwrap_model(m).state_dict().items()}

def _load_state_dict_strict(m, state):
    _unwrap_model(m).load_state_dict(state, strict=True)


def _log_model_dtype_summary(model):
    dtype_totals = {}
    total_params = 0
    for param in _unwrap_model(model).parameters():
        count = param.numel()
        total_params += count
        key = str(param.dtype)
        dtype_totals[key] = dtype_totals.get(key, 0) + count

    summary = ", ".join(
        f"{dtype}={count / 1_000_000:.2f}M"
        for dtype, count in sorted(dtype_totals.items())
    )
    logger.info(f"Parameter dtype summary: {summary} (total={total_params / 1_000_000:.2f}M)")


def _register_dtype_debug_hooks(model, global_rank):
    if global_rank != 0:
        return []

    core_model = _unwrap_model(model)
    hooks = []
    seen = set()

    def _first_tensor_dtype(values):
        for value in values:
            if torch.is_tensor(value):
                return value.dtype
            if isinstance(value, (tuple, list)):
                nested_dtype = _first_tensor_dtype(value)
                if nested_dtype is not None:
                    return nested_dtype
        return None

    def _make_hook(name):
        def _hook(_module, inputs, output):
            if name in seen:
                return
            seen.add(name)

            input_dtype = _first_tensor_dtype(inputs)
            if isinstance(output, (tuple, list)):
                output_dtype = _first_tensor_dtype(output)
            elif torch.is_tensor(output):
                output_dtype = output.dtype
            else:
                output_dtype = None

            logger.info(f"{name} dtype: input={input_dtype}, output={output_dtype}")
        return _hook

    if hasattr(core_model, "model") and hasattr(core_model.model, "embed_tokens"):
        hooks.append(core_model.model.embed_tokens.register_forward_hook(_make_hook("embed_tokens")))
    if hasattr(core_model, "lm_head"):
        hooks.append(core_model.lm_head.register_forward_hook(_make_hook("lm_head")))

    return hooks


def _call_npu_api(api_name, device=None, default=0):
    api = getattr(torch.npu, api_name, None)
    if not callable(api):
        return default

    try:
        return api()
    except TypeError:
        if device is not None:
            try:
                return api(device)
            except Exception:
                return default
        return default
    except Exception:
        return default


def _npu_synchronize(device=None):
    _call_npu_api("synchronize", device=device, default=None)


def _reset_npu_peak_memory_stats(device=None):
    _call_npu_api("reset_peak_memory_stats", device=device, default=None)


def _get_npu_memory_stats_mb(device=None):
    mb = 1024 * 1024
    allocated = _call_npu_api("memory_allocated", device=device, default=0)
    reserved = _call_npu_api("memory_reserved", device=device, default=0)
    peak_allocated = _call_npu_api("max_memory_allocated", device=device, default=allocated)
    peak_reserved = _call_npu_api("max_memory_reserved", device=device, default=reserved)
    return {
        "memory_allocated_mb": float(allocated) / mb,
        "memory_reserved_mb": float(reserved) / mb,
        "peak_memory_allocated_mb": float(peak_allocated) / mb,
        "peak_memory_reserved_mb": float(peak_reserved) / mb,
    }


def _distributed_max_scalar(value, device, use_dist):
    if not use_dist:
        return float(value)

    t = torch.tensor(float(value), dtype=torch.float32).to(device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.item()


def _get_optimizer_lr_summary(optimizer):
    lr_values = [float(param_group["lr"]) for param_group in optimizer.param_groups]
    if not lr_values:
        return 0.0, "n/a"
    if len(lr_values) == 1:
        return lr_values[0], f"{lr_values[0]:.8e}"
    lr_summary = ", ".join(f"group{i}={lr:.8e}" for i, lr in enumerate(lr_values))
    return lr_values[0], lr_summary


def _validate_batch(input_ids, vocab_size):
    """Check that token IDs are within [0, vocab_size). All ops stay on device."""
    return (input_ids.min() >= 0) & (input_ids.max() < vocab_size)


def _compute_grad_norm(trainable_params):
    grad_norm_sq = None
    for param in trainable_params:
        if param.grad is None:
            continue
        grad_sq = param.grad.detach().float().square().sum()
        if grad_norm_sq is None:
            grad_norm_sq = grad_sq
        else:
            grad_norm_sq = grad_norm_sq + grad_sq
    if grad_norm_sq is None:
        return 0.0
    return float(grad_norm_sq.sqrt().item())


def _summarize_nonfinite_gradients(named_trainable_params, limit=8):
    offenders = []
    for name, param in named_trainable_params:
        grad = param.grad
        if grad is None:
            continue
        grad_fp32 = grad.detach().float()
        finite_mask = torch.isfinite(grad_fp32)
        if bool(finite_mask.all().item()):
            continue
        invalid_count = int((~finite_mask).sum().item())
        max_abs = float(grad_fp32.abs().max().item()) if grad_fp32.numel() > 0 else 0.0
        offenders.append(
            f"{name} shape={tuple(grad.shape)} invalid={invalid_count}/{grad.numel()} max_abs={max_abs:.6e}"
        )
        if len(offenders) >= limit:
            break
    return offenders


def _validate_grad_norm_or_raise(grad_norm, named_trainable_params=None, context="", limit=8):
    if math.isfinite(grad_norm):
        return

    message = [f"Non-finite grad norm detected: grad_norm={grad_norm}"]
    if context:
        message.append(context)
    if named_trainable_params is not None:
        offenders = _summarize_nonfinite_gradients(named_trainable_params, limit=limit)
        if offenders:
            message.append("Offending gradients: " + " | ".join(offenders))
        else:
            message.append("No parameter-level non-finite gradients were found during post-check.")
    raise RuntimeError(". ".join(message))


def _manual_clip_grad_norm_(trainable_params, max_norm, eps=1e-6):
    grad_norm_sq = None
    for param in trainable_params:
        grad = param.grad
        if grad is None:
            continue
        grad_sq = grad.detach().float().square().sum()
        if grad_norm_sq is None:
            grad_norm_sq = grad_sq
        else:
            grad_norm_sq = grad_norm_sq + grad_sq

    if grad_norm_sq is None:
        return 0.0

    total_norm = grad_norm_sq.sqrt()
    total_norm_value = float(total_norm.item())
    if not math.isfinite(total_norm_value):
        return total_norm_value

    clip_coef = torch.tensor(float(max_norm), device=total_norm.device, dtype=total_norm.dtype)
    clip_coef = clip_coef / (total_norm + eps)
    clip_coef = torch.clamp(clip_coef, max=1.0)

    if bool((clip_coef < 1.0).item()):
        for param in trainable_params:
            grad = param.grad
            if grad is None:
                continue
            grad.mul_(clip_coef.to(device=grad.device, dtype=grad.dtype))

    return total_norm_value


def _clip_grad_norm(
    trainable_params,
    max_norm,
    *,
    named_trainable_params=None,
    backend="torch",
    context="",
    grad_debug_param_limit=8,
):
    if backend == "manual":
        grad_norm = _manual_clip_grad_norm_(trainable_params, max_norm)
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm)
        grad_norm = grad_norm.item() if torch.is_tensor(grad_norm) else float(grad_norm)

    _validate_grad_norm_or_raise(
        grad_norm,
        named_trainable_params=named_trainable_params,
        context=context,
        limit=grad_debug_param_limit,
    )
    return grad_norm


def _parse_debug_ranks(value):
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    if value.lower() == "all":
        return "all"

    ranks = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        ranks.add(int(item))
    return ranks or None


def _debug_rank_enabled(args, global_rank):
    if args.debug_update_step_start is None:
        return False
    if args.debug_ranks in (None, "all"):
        return True
    return global_rank in args.debug_ranks


def _debug_step_enabled(args, global_rank, target_update_step):
    if not _debug_rank_enabled(args, global_rank):
        return False
    return args.debug_update_step_start <= target_update_step <= args.debug_update_step_end


def _dataset_num_shards(dataset):
    candidates = [dataset, getattr(dataset, "_ex_iterable", None)]
    for candidate in candidates:
        if candidate is None:
            continue
        for attr_name in ("n_shards", "num_shards"):
            attr = getattr(candidate, attr_name, None)
            if attr is None:
                continue
            try:
                value = attr() if callable(attr) else attr
            except Exception:
                continue
            if value is None:
                continue
            try:
                return int(value)
            except Exception:
                return value
    return None


def _summarize_cpu_batch(batch, pad_idx):
    input_ids = batch["input_ids"].detach().reshape(-1).contiguous()
    attention_mask = batch["attention_mask"].detach().reshape(-1).contiguous()
    digest = hashlib.sha1(input_ids.numpy().tobytes()).hexdigest()[:16]
    first_tokens = input_ids[: min(8, input_ids.numel())].tolist()
    if pad_idx is not None and pad_idx >= 0:
        non_pad_tokens = int((input_ids != pad_idx).sum().item())
    else:
        non_pad_tokens = int(input_ids.numel())
    return {
        "sha1": digest,
        "shape": tuple(batch["input_ids"].shape),
        "token_min": int(input_ids.min().item()),
        "token_max": int(input_ids.max().item()),
        "token_sum": int(input_ids.sum().item()),
        "attention_sum": int(attention_mask.sum().item()),
        "non_pad_tokens": non_pad_tokens,
        "first_tokens": first_tokens,
    }


def _dump_debug_batch(
    dump_dir,
    batch,
    batch_summary,
    *,
    host_name,
    global_rank,
    local_rank,
    target_update_step,
    global_step,
    batch_idx,
):
    if not dump_dir:
        return None

    os.makedirs(dump_dir, exist_ok=True)
    payload = {
        "host_name": host_name,
        "global_rank": int(global_rank),
        "local_rank": int(local_rank),
        "target_update_step": int(target_update_step),
        "global_step": int(global_step),
        "batch_idx": int(batch_idx),
        "batch_summary": dict(batch_summary),
        "batch": {
            key: value.detach().cpu().clone()
            for key, value in batch.items()
        },
    }
    file_name = (
        f"rank{global_rank:04d}_update{target_update_step:06d}_global{global_step:06d}_"
        f"batch{batch_idx:06d}_{batch_summary['sha1']}.pt"
    )
    output_path = os.path.join(dump_dir, file_name)
    torch.save(payload, output_path)
    return output_path


def _emit_step_debug(host_name, global_rank, local_rank, target_update_step, global_step, batch_idx, stage, message):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(
        f"{timestamp} [step-debug] host={host_name} rank={global_rank} local_rank={local_rank} "
        f"target_update_step={target_update_step} global_step={global_step} batch_idx={batch_idx} "
        f"stage={stage} {message}",
        flush=True,
    )


def _step_debug_sync(enabled, device, host_name, global_rank, local_rank, target_update_step, global_step, batch_idx, stage):
    if not enabled:
        return
    _emit_step_debug(
        host_name,
        global_rank,
        local_rank,
        target_update_step,
        global_step,
        batch_idx,
        stage,
        "before_npu_sync",
    )
    _npu_synchronize(device)
    _emit_step_debug(
        host_name,
        global_rank,
        local_rank,
        target_update_step,
        global_step,
        batch_idx,
        stage,
        "after_npu_sync",
    )


def _data_state_path(save_dir, global_rank=None):
    if global_rank is None:
        return os.path.join(save_dir, DATA_STATE_FILENAME)
    return os.path.join(save_dir, f"data_state_rank{global_rank}.pt")


def _resume_manifest_path(save_dir):
    return os.path.join(save_dir, RESUME_MANIFEST_FILENAME)


def _normalize_path(path):
    return os.path.realpath(os.path.abspath(path))


def _sorted_data_files(data_dir, pattern):
    search_root = _normalize_path(data_dir)
    data_files = sorted(
        _normalize_path(path)
        for path in glob.glob(os.path.join(search_root, pattern))
    )
    if not data_files:
        raise FileNotFoundError(f"No data files matched {pattern} under {search_root}")
    return data_files


def _build_resume_manifest(args, world_size, data_train_files, shuffle_seed):
    return {
        "world_size": int(world_size),
        "single_gpu": bool(args.single_gpu),
        "expected_rank_count": 1 if args.single_gpu else int(world_size),
        "data_files": list(data_train_files),
        "shuffle_seed": int(shuffle_seed),
        "batch_size": int(args.batch_size),
        "total_batch_size": int(args.total_batch_size),
        "gradient_accumulation": int(args.gradient_accumulation),
        "max_length": int(args.max_length),
        "tokenizer_path": _normalize_path(args.tokenizer_path),
    }


def _save_resume_manifest(save_dir, manifest):
    with open(_resume_manifest_path(save_dir), "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def _load_resume_manifest(save_dir):
    manifest_path = _resume_manifest_path(save_dir)
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"Resume checkpoint at {save_dir} is missing {RESUME_MANIFEST_FILENAME}. "
            "Recreate the checkpoint with the strict resume format."
        )
    with open(manifest_path) as f:
        return json.load(f)


def _format_file_list_diff(expected_files, actual_files):
    if len(expected_files) != len(actual_files):
        return (
            f"expected {len(expected_files)} files but found {len(actual_files)} "
            f"(first expected={expected_files[:1]}, first found={actual_files[:1]})"
        )

    for idx, (expected_path, actual_path) in enumerate(zip(expected_files, actual_files)):
        if expected_path != actual_path:
            return (
                f"first mismatch at index {idx}: "
                f"expected {expected_path}, found {actual_path}"
            )

    return "file lists differ"


def _validate_resume_manifest(manifest, expected_manifest, continue_from):
    required_keys = (
        "world_size",
        "single_gpu",
        "expected_rank_count",
        "data_files",
        "shuffle_seed",
        "batch_size",
        "total_batch_size",
        "gradient_accumulation",
        "max_length",
        "tokenizer_path",
    )
    missing_keys = [key for key in required_keys if key not in manifest]
    if missing_keys:
        raise ValueError(
            f"Resume checkpoint at {continue_from} has an incomplete manifest. "
            f"Missing keys: {missing_keys}"
        )

    for key in required_keys:
        if key == "data_files":
            continue
        expected_value = expected_manifest[key]
        actual_value = manifest[key]
        if actual_value != expected_value:
            raise ValueError(
                f"Resume checkpoint at {continue_from} has manifest mismatch for {key}: "
                f"expected {expected_value!r}, found {actual_value!r}"
            )

    actual_files = list(manifest["data_files"])
    expected_files = list(expected_manifest["data_files"])
    if actual_files != expected_files:
        raise ValueError(
            f"Resume checkpoint at {continue_from} was created with a different training file order: "
            f"{_format_file_list_diff(expected_files, actual_files)}"
        )


def _validate_resume_artifacts(continue_from, manifest, global_rank):
    required_files = (
        "pytorch_model.bin",
        "optimizer.pt",
        "scheduler.pt",
        "training_state.json",
        RESUME_MANIFEST_FILENAME,
    )
    missing_files = [
        filename for filename in required_files
        if not os.path.exists(os.path.join(continue_from, filename))
    ]
    if missing_files:
        raise FileNotFoundError(
            f"Resume checkpoint at {continue_from} is incomplete. Missing files: {missing_files}"
        )

    expected_rank_count = int(manifest["expected_rank_count"])
    if manifest["single_gpu"]:
        data_state_path = _data_state_path(continue_from)
        if not os.path.exists(data_state_path):
            raise FileNotFoundError(
                f"Resume checkpoint at {continue_from} is missing dataset state {data_state_path}"
            )
        actual_rank_count = 1
    else:
        rank_state_files = sorted(glob.glob(os.path.join(continue_from, "data_state_rank*.pt")))
        actual_rank_count = len(rank_state_files)
        if actual_rank_count != expected_rank_count:
            raise ValueError(
                f"Resume checkpoint at {continue_from} expected {expected_rank_count} rank data states "
                f"but found {actual_rank_count}"
            )
        rank_data_path = _data_state_path(continue_from, global_rank)
        if not os.path.exists(rank_data_path):
            raise FileNotFoundError(
                f"Resume checkpoint at {continue_from} is missing dataset state for rank {global_rank}: "
                f"{rank_data_path}"
            )

    if actual_rank_count != expected_rank_count:
        raise ValueError(
            f"Resume checkpoint at {continue_from} expected rank count {expected_rank_count} "
            f"but found {actual_rank_count}"
        )


def _load_rank_data_state(save_dir, global_rank):
    rank_path = _data_state_path(save_dir, global_rank)
    if os.path.exists(rank_path):
        return torch.load(rank_path, map_location="cpu")

    legacy_path = _data_state_path(save_dir)
    if os.path.exists(legacy_path):
        data_states = torch.load(legacy_path, map_location="cpu")
        if isinstance(data_states, list):
            if global_rank >= len(data_states):
                raise ValueError(
                    f"Checkpoint data state at {legacy_path} only has {len(data_states)} rank entries, "
                    f"but requested rank {global_rank}"
                )
            return data_states[global_rank]
        return data_states

    return None


def save_resume_checkpoint(model, optimizer, scheduler, args, update_step, global_step, tokens_seen, save_dir, global_rank, data_state=None, resume_manifest=None):
    """Save full training state for resumption (model + optimizer + scheduler + RNG + training state)."""
    os.makedirs(save_dir, exist_ok=True)
    if data_state is not None:
        data_state_path = _data_state_path(save_dir, None if args.single_gpu else global_rank)
        torch.save(data_state, data_state_path)
    if not args.single_gpu:
        dist.barrier()

    if global_rank != 0:
        dist.barrier()
        return
    torch.save(_unwrap_model(model).state_dict(), os.path.join(save_dir, "pytorch_model.bin"))
    torch.save(optimizer.state_dict(), os.path.join(save_dir, "optimizer.pt"))
    if scheduler is not None:
        torch.save(scheduler.state_dict(), os.path.join(save_dir, "scheduler.pt"))
    # Save RNG states
    rng_state = {
        'torch_rng': torch.get_rng_state(),
        'npu_rng': torch.npu.get_rng_state(),
        'numpy_rng': np.random.get_state(),
        'python_rng': random.getstate(),
    }
    torch.save(rng_state, os.path.join(save_dir, "rng_state.pt"))
    with open(os.path.join(save_dir, "training_state.json"), "w") as f:
        json.dump({
            "global_step": global_step,
            "update_step": update_step,
            "tokens_seen": tokens_seen,
            "tokens_seen_before": tokens_seen,
        }, f, indent=2)
    if resume_manifest is not None:
        _save_resume_manifest(save_dir, resume_manifest)
    logger.info(f"Resume checkpoint saved to {save_dir} at update_step={update_step}")
    if not args.single_gpu:
        dist.barrier()


def save_final_checkpoint(model, args, update_step, tokens_seen, save_dir, global_rank):
    """Save model-only checkpoint for evaluation (model weights + config + training state)."""
    if global_rank != 0:
        if not args.single_gpu:
            dist.barrier()
        return
    os.makedirs(save_dir, exist_ok=True)
    torch.save(_unwrap_model(model).state_dict(), os.path.join(save_dir, "pytorch_model.bin"))
    if os.path.exists(args.model_config):
        shutil.copy2(args.model_config, os.path.join(save_dir, "config.json"))
    with open(os.path.join(save_dir, "training_state.json"), "w") as f:
        json.dump({
            "update_step": update_step,
            "tokens_seen": tokens_seen,
        }, f, indent=2)
    logger.info(f"Final checkpoint saved to {save_dir} at update_step={update_step}")
    if not args.single_gpu:
        dist.barrier()


def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--use_hf_model", default=False, action="store_true")
    parser.add_argument("--continue_from", type=str, default=None)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--gradient_accumulation", type=int, default=None)
    parser.add_argument("--total_batch_size", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--optimizer", default="Adam")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--scheduler", type=str, default="cosine",
                        choices=training_utils.SCHEDULER_CHOICES)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--activation_checkpointing", action="store_true")

    # Weight decay and gradient clipping
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clipping", type=float, default=1.0)
    parser.add_argument("--grad_clip_backend", type=str, default="auto",
                        choices=["auto", "torch", "manual"],
                        help="Gradient clipping backend. 'auto' uses manual clipping on NPU and torch elsewhere.")
    parser.add_argument("--grad_debug_param_limit", type=int, default=8,
                        help="Maximum number of parameter names to report when non-finite gradients are detected.")

    # Training steps
    parser.add_argument("--warmup_steps", type=int, default=1_000)
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--eval_batch_size", type=int, default=None,
                        help="Batch size for evaluation. Defaults to 2x training batch_size.")
    parser.add_argument("--cache_eval_data", action="store_true", default=False,
                        help="Cache validation dataset in memory to avoid reloading each eval.")
    parser.add_argument("--num_training_steps", type=int, default=10_000,
                        help="Number of **update steps** to train for.")
    parser.add_argument("--max_train_tokens", type=training_utils.max_train_tokens_to_number, default=None,
                        help="Number of tokens to train on. Overwrites num_training_steps. "
                             "You can use M and B suffixes, e.g. 100M or 1B.")

    # Save
    parser.add_argument("--save_every", type=int, default=100000000)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--save_steps", type=str, default=None,
                        help="Comma-separated update steps to save resume checkpoints, e.g. '4500,9000,18000'")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Base directory for WSD checkpoints (overrides save_dir for resume checkpoints)")

    # Betas for adamw-type optimizers
    parser.add_argument('--betas', nargs='+', default=[0.9, 0.95], help="betas for adamw", type=float)
    parser.add_argument('--mg_params', nargs='+', default=[0.5, 2.0, 0.1], help="mgup params", type=float)

    # Momentum for muon-type optimizers
    parser.add_argument("--beta1", type=float, default=0.95,
                        help="momentum for muon-type optimizers")
    parser.add_argument("--beta2", type=float, default=0.0)
    parser.add_argument("--nesterov", action="store_true", default=False,
                        help="Use Nesterov momentum for Muon optimizer")
    parser.add_argument("--rowcol_scale_exponent", type=float, default=-0.5,
                        help="Exponent for MuonEQ row/col scaling. Default -0.5, can set -0.25.")
    parser.add_argument("--muoneq_zeropower_mode", type=str, default="native",
                        choices=["native", "spc"],
                        help="Zeropower backend for MuonEq: native or spc.")
    parser.add_argument("--phase", type=int, default=None,
                        help="Phase switch step for MuonEq row/col mode. When optimizer=muoneq-rowcol, before phase use row/col normalization; at and after phase switch to row normalization.")

    # MARS gamma
    parser.add_argument('--mars_gamma', default=0.025, help="gamma param for mars/muon-mvr", type=float)

    # FOAM fold level
    parser.add_argument("--fold_level", type=int, default=1,
                        help="Fold level for FOAM optimizer (block_size = 2^fold_level)")

    # WSD schedule
    parser.add_argument("--wsd_decay_steps", type=int, default=None,
                        help="Decay steps for WSD schedule. Default: 10%% of total steps. "
                             "Set to 0 for trunk-only training (no decay).")

    # Data paths
    parser.add_argument("--data_dir", type=str,
                        default=os.environ.get("C4_DATA_DIR", "/path/to/c4"),
                        help="Directory containing c4 data files. Can also be set via C4_DATA_DIR.")
    parser.add_argument("--tokenizer_path", type=str,
                        default=os.environ.get("TOKENIZER_PATH", "t5-base"),
                        help="Path to tokenizer. Can also be set via TOKENIZER_PATH.")

    # Other
    parser.add_argument("--tags", type=str, default=None,
                        help="Comma-separated tags for wandb tracking.")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--tokenizer_batch_size", type=int, default=64,
                        help="Number of raw texts to tokenize together inside each data worker.")
    parser.add_argument("--text_buffer_size", type=int, default=256,
                        help="Number of raw texts buffered before batch tokenization and packing.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug_update_step_start", type=int, default=None,
                        help="Start of the update-step window for targeted debug logging.")
    parser.add_argument("--debug_update_step_end", type=int, default=None,
                        help="End of the update-step window for targeted debug logging.")
    parser.add_argument("--debug_ranks", type=str, default=None,
                        help="Comma-separated global ranks to debug, or 'all'. Default: all ranks in the debug window.")
    parser.add_argument("--debug_force_sync", action="store_true", default=False,
                        help="Synchronize the NPU after each debug stage inside the debug window.")
    parser.add_argument("--debug_dump_batch_dir", type=str, default=None,
                        help="Optional directory to dump exact CPU batches for the selected debug window.")
    parser.add_argument("--name", type=str, default=None,
                        help="Experiment name for wandb/swanlab. Auto-generated if not provided.")
    parser.add_argument("--report_to", type=str, default="wandb", choices=["wandb", "swanlab", "none"],
                        help="Tracking backend to use. Defaults to wandb.")
    parser.add_argument("--wandb_proj", default=None, type=str,
                        help="wandb project name override.")
    parser.add_argument("--single_gpu", default=False, action="store_true")
    parser.add_argument("--swanlab_proj", default=None, type=str,
                        help="swanlab project name override.")
    parser.add_argument("--block", type=int, default=8)

    # Legacy args
    parser.add_argument('--reps_rel', type=float, default=1e-6)
    parser.add_argument('--init_eta', type=float, default=0)
    parser.add_argument('--avg_gamma', type=float, default=8)
    parser.add_argument("--dog", action="store_true", default=False)

    raw_args = sys.argv[1:] if args is None else list(args)
    report_to_explicitly_set = any(
        arg == "--report_to" or arg.startswith("--report_to=")
        for arg in raw_args
    )
    args = parser.parse_args(raw_args)
    args.scheduler = training_utils.normalize_scheduler_type(args.scheduler)

    args = args_utils.check_args_torchrun_main(args)

    if not report_to_explicitly_set and args.swanlab_proj is not None and args.wandb_proj is None:
        args.report_to = "swanlab"

    # Parse save_steps
    if args.save_steps is not None:
        args.save_steps = sorted([int(s.strip()) for s in args.save_steps.split(",")])
    else:
        args.save_steps = []

    # Default WSD decay steps
    if training_utils.is_wsd_scheduler(args.scheduler) and args.wsd_decay_steps is None:
        args.wsd_decay_steps = max(1, int(0.1 * args.num_training_steps))

    if args.debug_update_step_start is None and args.debug_update_step_end is not None:
        args.debug_update_step_start = args.debug_update_step_end
    if args.debug_update_step_end is None and args.debug_update_step_start is not None:
        args.debug_update_step_end = args.debug_update_step_start
    if args.debug_update_step_start is not None:
        if args.debug_update_step_start < 1:
            raise ValueError("--debug_update_step_start must be >= 1")
        if args.debug_update_step_end < args.debug_update_step_start:
            raise ValueError("--debug_update_step_end must be >= --debug_update_step_start")
    args.debug_ranks = _parse_debug_ranks(args.debug_ranks)

    # Auto-generate experiment name
    if args.name is None:
        lr_str = f"lr{args.lr}".replace(".", "").replace("-", "")
        gamma_str = (
            f"-g{args.mars_gamma}".replace(".", "")
            if args.optimizer.lower() in ["mars", "muon-mvr1"]
            else ""
        )
        fold_str = f"-f{args.fold_level}" if args.optimizer.lower() == "foam" else ""
        args.name = f"{args.optimizer}{lr_str}{gamma_str}{fold_str}-s{args.seed}"

    return args


def materialize_val_batches(tokenizer, pad_idx, global_rank, world_size, args, batch_size):
    """Load, tokenize, and store all eval batches in CPU memory once.

    Uses the same efficient PreprocessedIterableDataset pipeline as training
    (batch tokenization + token packing, zero padding waste).

    Returns a list of pre-built batch dicts (CPU tensors) and the total
    number of non-padding tokens (already multiplied by world_size).
    """
    _time = time.time()

    data_files = _sorted_data_files(args.data_dir, VAL_DATA_GLOB)
    val_data = datasets.load_dataset(
        "json",
        data_files=data_files,
        split="train",
        streaming=True
    )
    val_data = val_data.shuffle(seed=TRAIN_SHUFFLE_SEED)

    if not args.single_gpu:
        val_data = datasets.distributed.split_dataset_by_node(val_data, rank=global_rank, world_size=world_size)

    val_dataset = PreprocessedIterableDataset(
        val_data,
        tokenizer,
        batch_size=batch_size,
        max_length=args.max_length,
        tokenizer_batch_size=args.tokenizer_batch_size,
        text_buffer_size=args.text_buffer_size,
        drop_last=False,
    )

    target_eval_tokens = 10_000_000
    evaluated_on_tokens = 0
    cached_batches = []

    for batch in val_dataset:
        if evaluated_on_tokens > target_eval_tokens:
            break
        cached_batches.append(batch)
        evaluated_on_tokens += batch["input_ids"].numel() * world_size

    logger.info(
        f"Materialized {len(cached_batches)} eval batches ({evaluated_on_tokens} tokens) "
        f"in {time.time() - _time:.2f} seconds"
    )
    return cached_batches, evaluated_on_tokens


@torch.no_grad()
def evaluate_model(model, tokenizer, pad_idx, global_rank, world_size, device, batch_size, args,
                   cached_eval=None):
    _time = time.time()

    if cached_eval is not None:
        batches, evaluated_on_tokens = cached_eval
    else:
        batches, evaluated_on_tokens = materialize_val_batches(
            tokenizer, pad_idx, global_rank, world_size, args, batch_size
        )

    model.eval()

    total_loss = torch.tensor(0.0).to(device)
    total_batches = 0

    for batch in batches:
        total_batches += 1
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["input_ids"].clone()
        labels[labels == pad_idx] = -100
        loss = model(**batch, labels=labels).loss
        total_loss += loss.detach()

    if total_batches > 0:
        total_loss = total_loss / total_batches

    model.train()

    # Gather losses across all GPUs
    if not args.single_gpu:
        gathered_losses = [torch.zeros_like(total_loss) for _ in range(world_size)]
        dist.all_gather(gathered_losses, total_loss)
        total_loss = sum([t.item() for t in gathered_losses]) / world_size
    else:
        total_loss = total_loss.item()

    logger.info(f"Evaluation completed in {time.time() - _time:.2f} seconds")
    return total_loss, evaluated_on_tokens


def _split_params_muon_adamw(model, exact_2d=False):
    """Split model parameters into muon_params and adamw_params."""
    def _use_muon(name, p):
        if not p.requires_grad:
            return False
        if "embed_tokens" in name or "lm_head" in name:
            return False
        return p.ndim == 2 if exact_2d else p.ndim >= 2

    muon_params = [
        p
        for name, p in model.named_parameters()
        if _use_muon(name, p)
    ]
    adamw_params = [
        p
        for name, p in model.named_parameters()
        if not _use_muon(name, p)
    ]
    return muon_params, adamw_params


def create_optimizer(model, args):
    """Create optimizer based on args.optimizer."""
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    named_trainable_params = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    opt_name = args.optimizer.lower()

    if opt_name == "adamw":
        optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay, betas=args.betas)

    elif opt_name == "mars":
        optimizer = MARS(trainable_params, lr=args.lr, weight_decay=args.weight_decay, betas=args.betas,
                         is_approx=True, gamma=args.mars_gamma, mgup=False)

    elif opt_name == "muon":
        muon_params, adamw_params = _split_params_muon_adamw(model)
        optimizer = Muon(
            lr=args.lr,
            wd=args.weight_decay,
            muon_params=muon_params,
            adamw_params=adamw_params,
            momentum=args.beta1,
            adamw_betas=tuple(args.betas),
            nesterov=False,
        )

    elif opt_name == "muon-nesterov":
        muon_params, adamw_params = _split_params_muon_adamw(model)
        optimizer = Muon(
            lr=args.lr,
            wd=args.weight_decay,
            muon_params=muon_params,
            adamw_params=adamw_params,
            momentum=args.beta1,
            adamw_betas=tuple(args.betas),
            nesterov=True,
        )

    elif opt_name == "muon-mvr1":
        muon_params, adamw_params = _split_params_muon_adamw(model)
        optimizer = MuonMVR(
            lr=args.lr,
            weight_decay=args.weight_decay,
            muon_params=muon_params,
            adamw_params=adamw_params,
            momentum=args.beta1,
            adamw_betas=tuple(args.betas),
            gamma=args.mars_gamma,
            is_approx=True,
        )

    elif opt_name == "muon-mvr2":
        muon_params, adamw_params = _split_params_muon_adamw(model)
        optimizer = MuonMVR(
            lr=args.lr,
            weight_decay=args.weight_decay,
            muon_params=muon_params,
            adamw_params=adamw_params,
            momentum=args.beta1,
            adamw_betas=tuple(args.betas),
            gamma=args.mars_gamma,
            is_approx=False,
        )

    elif opt_name == "foam":
        foam_params, adamw_params = _split_params_muon_adamw(model)
        optimizer = FOAM(
            lr=args.lr,
            wd=args.weight_decay,
            foam_params=foam_params,
            adamw_params=adamw_params,
            betas=tuple(args.betas),
            fold_level=args.fold_level,
            adamw_betas=tuple(args.betas),
        )
    elif opt_name in [
        "muoneq-rowcol",
        "muoneq-row",
        "muoneq-col",
    ]:
        muon_params, adamw_params = _split_params_muon_adamw(model)
        muoneq_mode_aliases = {
            "muoneq-rowcol": "rowcol",
            "muoneq-row": "row",
            "muoneq-col": "col",
        }
        optimizer = MuonEq(
            lr=args.lr,
            wd=args.weight_decay,
            muon_params=muon_params,
            adamw_params=adamw_params,
            momentum=args.beta1,
            adamw_betas=tuple(args.betas),
            rowcol_scale_exponent=args.rowcol_scale_exponent,
            normalize_mode=muoneq_mode_aliases[opt_name],
            phase=args.phase,
            zeropower_mode=args.muoneq_zeropower_mode,
        )
    else:
        raise ValueError(f"Optimizer {args.optimizer} not supported. "
                         f"Supported: adamw, mars, muon, muon-nesterov, muon-mvr1, muon-mvr2, foam, muoneq-rowcol, muoneq-row, muoneq-col")

    return optimizer


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    host_name = socket.gethostname()

    if args.single_gpu:
        global_rank = 0
        local_rank = 0
        world_size = 1
        if "LOCAL_RANK" in os.environ:
            local_rank = int(os.environ["LOCAL_RANK"])
        torch.npu.set_device(local_rank)
        device = f"npu:{local_rank}"
        logger.info(f"Single GPU mode, device: {device}")
    else:
        assert "LOCAL_RANK" in os.environ, "torchrun should set LOCAL_RANK"
        global_rank = int(os.environ['RANK'])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.npu.set_device(local_rank)
        logger.info(f"Global rank {global_rank}, local rank {local_rank}, device: {torch.npu.current_device()}")
        dist.init_process_group(backend="hccl", rank=global_rank, world_size=world_size)
        logger.info("Process group initialized")
        device = f"npu:{local_rank}"

    if args.total_batch_size is not None:
        if args.gradient_accumulation is None:
            assert args.total_batch_size % world_size == 0, "total_batch_size must be divisible by world_size"
            args.gradient_accumulation = args.total_batch_size // (args.batch_size * world_size)
            assert args.gradient_accumulation > 0, "gradient_accumulation must be greater than 0"

    assert args.gradient_accumulation * args.batch_size * world_size == args.total_batch_size, \
        "gradient_accumulation * batch_size * world_size must be equal to total_batch_size"

    # turn off logger
    if global_rank != 0:
        logger.remove()

    tracker_backend = None if args.report_to == "none" else args.report_to
    tracker = ExperimentTracker(tracker_backend if global_rank == 0 else None)
    if global_rank == 0 and tracker_backend is not None:
        tracker_project = _resolve_tracking_project(args)
        tracker.init(project=tracker_project, run_name=args.name, tags=args.tags)
        logger.info(f"Initialized {args.report_to} tracking in project {tracker_project}")

    logger.info(f"Using dist with rank {global_rank} (only rank 0 will log)")
    logger.info("*" * 40)
    logger.info(f"Starting training with the arguments")
    for k, v in vars(args).items():
        logger.info(f"{k:30} {v}")
    logger.info("*" * 40)

    if (args.save_steps or args.continue_from is not None) and args.workers != 0:
        if global_rank == 0:
            logger.warning(
                f"Overriding workers from {args.workers} to 0. "
                "Exact resume for streaming data requires num_workers=0."
            )
        args.workers = 0
    # Fix the global data order.
    data_train_files = _sorted_data_files(args.data_dir, TRAIN_DATA_GLOB)
    seed_for_shuffle = TRAIN_SHUFFLE_SEED
    resume_manifest = _build_resume_manifest(args, world_size, data_train_files, seed_for_shuffle)
    if global_rank == 0:
        logger.info(f"Matched {len(data_train_files)} training files under {args.data_dir}")

    if args.continue_from is not None:
        checkpoint_manifest = _load_resume_manifest(args.continue_from)
        _validate_resume_manifest(checkpoint_manifest, resume_manifest, args.continue_from)
        _validate_resume_artifacts(args.continue_from, checkpoint_manifest, global_rank)

    data = datasets.load_dataset(
        "json",
        data_files=data_train_files,
        split="train",
        streaming=True
    )
    if global_rank == 0:
        logger.info(f"Dataset shard count before shuffle: {_dataset_num_shards(data)}")

    logger.info(f"Shuffling data with seed {seed_for_shuffle}")
    data: datasets.Dataset = data.shuffle(seed=seed_for_shuffle)
    if global_rank == 0:
        logger.info(f"Dataset shard count after shuffle: {_dataset_num_shards(data)}")
    if not args.single_gpu:
        data = datasets.distributed.split_dataset_by_node(
            data, rank=global_rank, world_size=world_size,
        )
    if global_rank == 0:
        logger.info(f"Dataset shard count after rank split: {_dataset_num_shards(data)}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path,
        model_max_length=args.max_length,
        use_fast=True,
    )

    def preprocess_batched(batch):
        batch = tokenizer(
            batch["text"],
            max_length=args.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return batch

    dataset = PreprocessedIterableDataset(
        data,
        tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        tokenizer_batch_size=args.tokenizer_batch_size,
        text_buffer_size=args.text_buffer_size,
    )
    dataloader_kwargs = {
        "batch_size": None,
        "num_workers": args.workers,
    }
    if args.workers > 0:
        dataloader_kwargs["persistent_workers"] = True
        dataloader_kwargs["prefetch_factor"] = 2
    dataloader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)
    logger.info(
        f"Training data pipeline: packed batches, workers={args.workers}, "
        f"tokenizer_batch_size={args.tokenizer_batch_size}, text_buffer_size={args.text_buffer_size}"
    )

    # Evaluation settings
    eval_batch_size = args.eval_batch_size if args.eval_batch_size else args.batch_size
    logger.info(f"Eval batch size: {eval_batch_size} (train batch size: {args.batch_size})")

    model_config = AutoConfig.from_pretrained(args.model_config)
    if args.use_hf_model:
        model: HF_LlamaForCausalLM = AutoModelForCausalLM.from_config(model_config)
    else:
        model = LlamaForCausalLM(model_config)

    if args.activation_checkpointing:
        model.gradient_checkpointing_enable()

    global_step = 0
    update_step = 0
    beginning_step = 0
    tokens_seen = 0
    tokens_seen_before = 0

    if args.continue_from is not None:
        logger.info("*" * 40)
        logger.info(f"Loading model from {args.continue_from}")
        checkpoint_path = os.path.join(args.continue_from, "pytorch_model.bin")
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"), strict=True)
        logger.info(f"Model successfully loaded (strict=True policy)")

        if os.path.exists(os.path.join(args.continue_from, "training_state.json")):
            logger.info(f"Loading training state from {args.continue_from}")
            with open(os.path.join(args.continue_from, "training_state.json")) as f:
                _old_state = json.load(f)
            global_step = _old_state["global_step"]
            update_step = _old_state["update_step"]
            tokens_seen = _old_state["tokens_seen"]
            tokens_seen_before = _old_state.get("tokens_seen_before", tokens_seen)
            beginning_step = update_step
            if args.num_training_steps < update_step:
                raise ValueError(
                    f"--num_training_steps={args.num_training_steps} is smaller than checkpoint "
                    f"update_step={update_step}. When resuming continuous WSD decay, pass the target "
                    "total update steps instead of the remaining decay steps."
                )
            logger.info(f"global_step       : {global_step}")
            logger.info(f"update_step       : {update_step}")
            logger.info(f"tokens_seen       : {tokens_seen}")
            logger.info(f"tokens_seen_before: {tokens_seen_before}")
            logger.info(f"Will train for {args.num_training_steps - update_step} update steps")
        else:
            logger.warning(f"Did not find training state in {args.continue_from}, global step will start from zero")

        data_state = _load_rank_data_state(args.continue_from, global_rank)
        if data_state is not None:
            dataset.load_state_dict(data_state)
            logger.info(f"Loaded dataset state from {args.continue_from}")
        else:
            raise FileNotFoundError(
                f"Checkpoint at {args.continue_from} is missing dataset state for rank {global_rank}"
            )
        logger.info("*" * 40)

    if args.dtype in ["bf16", "bfloat16"]:
        model = model.to(device=device, dtype=torch.bfloat16)
    else:
        model = model.to(device=device)

    n_total_params = sum(p.numel() for p in model.parameters())
    named_trainable_params = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    trainable_params = [p for _, p in named_trainable_params]
    grad_clip_backend = args.grad_clip_backend
    if grad_clip_backend == "auto":
        grad_clip_backend = "manual" if str(device).startswith("npu") else "torch"

    if global_rank == 0:
        pbar = tqdm(total=args.num_training_steps - update_step, desc="Update steps", ncols=80)

    logger.info(f"\n{model}\n")
    logger.info(f"Total params: {n_total_params / 1_000_000:.2f}M")
    logger.info(f"Trainable params: {sum(p.numel() for p in trainable_params) / 1_000_000:.2f}M")
    logger.info(f"Training dtype: {args.dtype}")
    logger.info(f"Grad clip backend: {grad_clip_backend}")
    logger.info(f"Saving model to {args.save_dir} every {args.save_every} update steps")
    if args.save_steps:
        logger.info(f"Will save resume checkpoints at steps: {args.save_steps}")

    run_config = dict(vars(args))
    run_config.update({
        "world_size": world_size,
        "global_rank": global_rank,
        "hostname": host_name,
        "total_params": n_total_params,
        "trainable_params": sum(p.numel() for p in trainable_params),
    })
    if global_rank == 0:
        tracker.update_config(run_config)

    # Create optimizer
    optimizer = create_optimizer(model, args)

    # Create scheduler
    scheduler_kwargs = {}
    if training_utils.is_wsd_scheduler(args.scheduler):
        scheduler_kwargs['wsd_decay_steps'] = args.wsd_decay_steps
        decay_kind = "cosine" if args.scheduler == "wsd_cosine" else "linear"
        logger.info(f"WSD schedule ({decay_kind} decay): warmup={args.warmup_steps}, decay={args.wsd_decay_steps}, "
                    f"total={args.num_training_steps}, stable={args.num_training_steps - args.warmup_steps - args.wsd_decay_steps}")

    scheduler = training_utils.get_scheduler(
        optimizer=optimizer,
        scheduler_type=args.scheduler,
        num_training_steps=args.num_training_steps,
        warmup_steps=args.warmup_steps,
        min_lr_ratio=args.min_lr_ratio,
        **scheduler_kwargs,
    )

    # Load optimizer/scheduler state if resuming
    if args.continue_from is not None:
        opt_state_path = os.path.join(args.continue_from, "optimizer.pt")
        if os.path.exists(opt_state_path):
            logger.info(f"Loading optimizer state from {opt_state_path}")
            optimizer.load_state_dict(torch.load(opt_state_path, map_location=device))
            logger.info("Optimizer state loaded successfully")

        sched_state_path = os.path.join(args.continue_from, "scheduler.pt")
        if os.path.exists(sched_state_path):
            logger.info(f"Loading scheduler state from {sched_state_path}")
            scheduler.load_state_dict(torch.load(sched_state_path, map_location="cpu"))
            logger.info("Scheduler state loaded successfully")

        rng_state_path = os.path.join(args.continue_from, "rng_state.pt")
        if os.path.exists(rng_state_path):
            logger.info(f"Loading RNG state from {rng_state_path}")
            rng_state = torch.load(rng_state_path, map_location="cpu")
            torch.set_rng_state(rng_state['torch_rng'])
            torch.npu.set_rng_state(rng_state['npu_rng'])
            np.random.set_state(rng_state['numpy_rng'])
            random.setstate(rng_state['python_rng'])
            logger.info("RNG state loaded successfully")

    if not args.single_gpu:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
        )

    _log_model_dtype_summary(model)
    dtype_debug_hooks = _register_dtype_debug_hooks(model, global_rank)

    prev_state_dict = None

    # global steps and others are defined above
    pad_idx = tokenizer.pad_token_id
    vocab_size = model_config.vocab_size

    # Materialize eval batches into memory if caching is enabled
    cached_eval = None
    if args.cache_eval_data:
        logger.info("Materializing validation batches into memory...")
        cached_eval = materialize_val_batches(
            tokenizer, pad_idx, global_rank, world_size, args, eval_batch_size
        )
        logger.info(f"Cached {len(cached_eval[0])} eval batches.")

    update_timer_start = time.time()
    local_step = 0
    running_peak_memory_allocated_mb = 0.0
    running_peak_memory_reserved_mb = 0.0
    _reset_npu_peak_memory_stats(device)

    # ##############################
    # TRAINING LOOP
    # ##############################
    previous_X, previous_Y = None, None
    dataloader_iter = iter(dataloader)
    batch_idx = 0
    while update_step < args.num_training_steps:
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            logger.info("Dataloader exhausted before reaching the requested number of update steps.")
            break

        batch_cpu = batch
        batch_idx += 1
        global_step += 1
        local_step += 1
        is_update_step = (global_step % args.gradient_accumulation == 0)
        debug_update_target = update_step + 1
        debug_this_batch = _debug_step_enabled(args, global_rank, debug_update_target)
        batch_cpu_summary = None
        if debug_this_batch:
            batch_cpu_summary = _summarize_cpu_batch(batch_cpu, pad_idx)
            dumped_batch_path = _dump_debug_batch(
                args.debug_dump_batch_dir,
                batch_cpu,
                batch_cpu_summary,
                host_name=host_name,
                global_rank=global_rank,
                local_rank=local_rank,
                target_update_step=debug_update_target,
                global_step=global_step,
                batch_idx=batch_idx,
            )
            _emit_step_debug(
                host_name,
                global_rank,
                local_rank,
                debug_update_target,
                global_step,
                batch_idx,
                "batch_loaded_cpu",
                (
                    f"is_update_step={is_update_step} shape={batch_cpu_summary['shape']} "
                    f"sha1={batch_cpu_summary['sha1']} token_range=[{batch_cpu_summary['token_min']}, {batch_cpu_summary['token_max']}] "
                    f"non_pad={batch_cpu_summary['non_pad_tokens']} attention_sum={batch_cpu_summary['attention_sum']} "
                    f"token_sum={batch_cpu_summary['token_sum']} first_tokens={batch_cpu_summary['first_tokens']} "
                    f"dump_path={dumped_batch_path or 'n/a'}"
                ),
            )

        batch = {k: v.to(device) for k, v in batch_cpu.items()}
        if debug_this_batch:
            _step_debug_sync(
                args.debug_force_sync,
                device,
                host_name,
                global_rank,
                local_rank,
                debug_update_target,
                global_step,
                batch_idx,
                "after_batch_to_device",
            )

        # Validate batch: check token IDs in [0, vocab_size)
        valid_flag = _validate_batch(batch["input_ids"], vocab_size).float().reshape(1)
        if not args.single_gpu:
            dist.all_reduce(valid_flag, op=dist.ReduceOp.MIN)
        valid_flag_value = float(valid_flag.item())
        if debug_this_batch:
            _emit_step_debug(
                host_name,
                global_rank,
                local_rank,
                debug_update_target,
                global_step,
                batch_idx,
                "after_valid_check",
                f"valid_flag={valid_flag_value:.1f}",
            )
        if valid_flag_value < 0.5:
            logger.warning(f"Skipping invalid batch at global_step {global_step} (rank {global_rank})")
            global_step -= 1
            local_step -= 1
            batch_idx -= 1
            continue

        labels = batch["input_ids"].clone()
        labels[labels == pad_idx] = -100
        tokens_seen += (batch["input_ids"] != pad_idx).sum().item() * world_size

        def _current_batch_context():
            summary = batch_cpu_summary
            if summary is None:
                summary = _summarize_cpu_batch(batch_cpu, pad_idx)
            return (
                f"optimizer={args.optimizer} rank={global_rank} local_rank={local_rank} "
                f"update_step_target={debug_update_target} global_step={global_step} batch_idx={batch_idx} "
                f"tokens_seen={tokens_seen} batch_sha1={summary['sha1']} "
                f"token_range=[{summary['token_min']}, {summary['token_max']}] "
                f"token_sum={summary['token_sum']} attention_sum={summary['attention_sum']} "
                f"first_tokens={summary['first_tokens']}"
            )

        # ===== Exact variance reduction for muon-mvr2 =====
        if args.optimizer.lower() == "muon-mvr3" and previous_X is not None and is_update_step:
            loss_prev = model(**previous_X, labels=previous_Y).loss
            (loss_prev / args.gradient_accumulation).backward()
            if args.grad_clipping != 0.0:
                try:
                    _clip_grad_norm(
                        trainable_params,
                        args.grad_clipping,
                        named_trainable_params=named_trainable_params,
                        backend=grad_clip_backend,
                        context=f"{_current_batch_context()} stage=muon_mvr3_prev_grad_clip",
                        grad_debug_param_limit=args.grad_debug_param_limit,
                    )
                except Exception as exc:
                    raise RuntimeError(f"Gradient clipping failed in muon-mvr3 previous-step replay. {_current_batch_context()}") from exc
            optimizer.update_last_grad()
            optimizer.zero_grad(set_to_none=True)

        if args.optimizer.lower() == "muon-mvr2" and prev_state_dict is not None and is_update_step:
            cur_state_dict = _clone_state_dict_to_cpu(model)
            _load_state_dict_strict(model, prev_state_dict)
            loss_prev = model(**batch, labels=labels).loss
            (loss_prev / args.gradient_accumulation).backward()
            if args.grad_clipping != 0.0:
                try:
                    _clip_grad_norm(
                        trainable_params,
                        args.grad_clipping,
                        named_trainable_params=named_trainable_params,
                        backend=grad_clip_backend,
                        context=f"{_current_batch_context()} stage=muon_mvr2_prev_grad_clip",
                        grad_debug_param_limit=args.grad_debug_param_limit,
                    )
                except Exception as exc:
                    raise RuntimeError(f"Gradient clipping failed in muon-mvr2 previous-step replay. {_current_batch_context()}") from exc
            optimizer.update_last_grad()
            optimizer.zero_grad(set_to_none=True)
            _load_state_dict_strict(model, cur_state_dict)

        # Non-update step: accumulate gradients without sync
        if global_step % args.gradient_accumulation != 0:
            if args.single_gpu:
                if debug_this_batch:
                    _emit_step_debug(
                        host_name,
                        global_rank,
                        local_rank,
                        debug_update_target,
                        global_step,
                        batch_idx,
                        "before_forward_accum",
                        "single_gpu=True",
                    )
                loss = model(**batch, labels=labels).loss
                scaled_loss = loss / args.gradient_accumulation
                if debug_this_batch:
                    loss_value = float(loss.detach().float().item())
                    scaled_loss_value = float(scaled_loss.detach().float().item())
                    _emit_step_debug(
                        host_name,
                        global_rank,
                        local_rank,
                        debug_update_target,
                        global_step,
                        batch_idx,
                        "after_forward_accum",
                        f"loss={loss_value:.6f} scaled_loss={scaled_loss_value:.6f}",
                    )
                scaled_loss.backward()
            else:
                with model.no_sync():
                    if debug_this_batch:
                        _emit_step_debug(
                            host_name,
                            global_rank,
                            local_rank,
                            debug_update_target,
                            global_step,
                            batch_idx,
                            "before_forward_accum",
                            "single_gpu=False no_sync=True",
                        )
                    loss = model(**batch, labels=labels).loss
                    scaled_loss = loss / args.gradient_accumulation
                    if debug_this_batch:
                        loss_value = float(loss.detach().float().item())
                        scaled_loss_value = float(scaled_loss.detach().float().item())
                        _emit_step_debug(
                            host_name,
                            global_rank,
                            local_rank,
                            debug_update_target,
                            global_step,
                            batch_idx,
                            "after_forward_accum",
                            f"loss={loss_value:.6f} scaled_loss={scaled_loss_value:.6f}",
                        )
                    scaled_loss.backward()
            if debug_this_batch:
                _step_debug_sync(
                    args.debug_force_sync,
                    device,
                    host_name,
                    global_rank,
                    local_rank,
                    debug_update_target,
                    global_step,
                    batch_idx,
                    "after_backward_accum",
                )
            continue

        # Update step
        if debug_this_batch:
            _emit_step_debug(
                host_name,
                global_rank,
                local_rank,
                debug_update_target,
                global_step,
                batch_idx,
                "before_forward_update",
                f"optimizer={args.optimizer}",
            )
        loss = model(**batch, labels=labels).loss
        scaled_loss = loss / args.gradient_accumulation
        if debug_this_batch:
            loss_value = float(loss.detach().float().item())
            scaled_loss_value = float(scaled_loss.detach().float().item())
            _emit_step_debug(
                host_name,
                global_rank,
                local_rank,
                debug_update_target,
                global_step,
                batch_idx,
                "after_forward_update",
                f"loss={loss_value:.6f} scaled_loss={scaled_loss_value:.6f}",
            )
        scaled_loss.backward()
        if debug_this_batch:
            _step_debug_sync(
                args.debug_force_sync,
                device,
                host_name,
                global_rank,
                local_rank,
                debug_update_target,
                global_step,
                batch_idx,
                "after_backward_update",
            )

        if args.grad_clipping != 0.0:
            if debug_this_batch:
                _emit_step_debug(
                    host_name,
                    global_rank,
                    local_rank,
                    debug_update_target,
                    global_step,
                    batch_idx,
                    "before_grad_clip",
                    f"grad_clipping={args.grad_clipping}",
                )
            try:
                grad_norm = _clip_grad_norm(
                    trainable_params,
                    args.grad_clipping,
                    named_trainable_params=named_trainable_params,
                    backend=grad_clip_backend,
                    context=f"{_current_batch_context()} stage=main_grad_clip",
                    grad_debug_param_limit=args.grad_debug_param_limit,
                )
            except Exception as exc:
                raise RuntimeError(f"Gradient clipping failed. {_current_batch_context()}") from exc
        else:
            try:
                grad_norm = _compute_grad_norm(trainable_params)
                _validate_grad_norm_or_raise(
                    grad_norm,
                    named_trainable_params=named_trainable_params,
                    context=f"{_current_batch_context()} stage=main_grad_norm_only",
                    limit=args.grad_debug_param_limit,
                )
            except Exception as exc:
                raise RuntimeError(f"Gradient norm computation failed. {_current_batch_context()}") from exc
        if debug_this_batch:
            _emit_step_debug(
                host_name,
                global_rank,
                local_rank,
                debug_update_target,
                global_step,
                batch_idx,
                "after_grad_norm",
                f"grad_norm={grad_norm:.6f} tokens_seen={tokens_seen}",
            )

        if global_rank == 0:
            pbar.update(1)

        if args.optimizer.lower() == "muon-mvr3":
            previous_X, previous_Y = copy.deepcopy(batch), labels.clone()

        if args.optimizer.lower() == "muon-mvr2":
            prev_state_dict = _clone_state_dict_to_cpu(model)

        if debug_this_batch:
            _emit_step_debug(
                host_name,
                global_rank,
                local_rank,
                debug_update_target,
                global_step,
                batch_idx,
                "before_optimizer_step",
                "",
            )
        optimizer.step()
        if debug_this_batch:
            _step_debug_sync(
                args.debug_force_sync,
                device,
                host_name,
                global_rank,
                local_rank,
                debug_update_target,
                global_step,
                batch_idx,
                "after_optimizer_step",
            )
        scheduler.step()
        if debug_this_batch:
            _emit_step_debug(
                host_name,
                global_rank,
                local_rank,
                debug_update_target,
                global_step,
                batch_idx,
                "after_scheduler_step",
                "",
            )
        optimizer.zero_grad()
        if debug_this_batch:
            _emit_step_debug(
                host_name,
                global_rank,
                local_rank,
                debug_update_target,
                global_step,
                batch_idx,
                "after_zero_grad",
                "",
            )

        update_step += 1
        _npu_synchronize(device)
        if debug_this_batch:
            _emit_step_debug(
                host_name,
                global_rank,
                local_rank,
                debug_update_target,
                global_step,
                batch_idx,
                "after_update_sync",
                f"completed_update_step={update_step}",
            )
        update_time = time.time() - update_timer_start

        iter_time_avg = update_time / args.gradient_accumulation
        memory_stats = _get_npu_memory_stats_mb(device)
        peak_memory_allocated_mb = _distributed_max_scalar(
            memory_stats["peak_memory_allocated_mb"], device, use_dist=not args.single_gpu
        )
        peak_memory_reserved_mb = _distributed_max_scalar(
            memory_stats["peak_memory_reserved_mb"], device, use_dist=not args.single_gpu
        )
        current_memory_allocated_mb = _distributed_max_scalar(
            memory_stats["memory_allocated_mb"], device, use_dist=not args.single_gpu
        )
        current_memory_reserved_mb = _distributed_max_scalar(
            memory_stats["memory_reserved_mb"], device, use_dist=not args.single_gpu
        )
        running_peak_memory_allocated_mb = max(running_peak_memory_allocated_mb, peak_memory_allocated_mb)
        running_peak_memory_reserved_mb = max(running_peak_memory_reserved_mb, peak_memory_reserved_mb)

        # Evaluation
        if update_step % args.eval_every == 0:
            logger.info(f"Performing evaluation at step {update_step}")
            total_loss, evaluated_on_tokens = evaluate_model(
                model, tokenizer, pad_idx, global_rank, world_size, device, eval_batch_size, args,
                cached_eval=cached_eval
            )
            logger.info(f"Eval loss at step {update_step}: {total_loss}")
            if global_rank == 0:
                tracker.log({
                    "eval_loss": total_loss,
                    "eval_tokens": evaluated_on_tokens,
                }, step=update_step)

        lr, lr_summary = _get_optimizer_lr_summary(optimizer)
        tokens_in_update = tokens_seen - tokens_seen_before
        tokens_seen_before = tokens_seen
        batches_in_update = args.gradient_accumulation * world_size

        if global_rank == 0:
            logger.info(f"Train loss at step {update_step}: {loss}")
            logger.info(f"Learning rate at step {update_step}: {lr_summary}")
            logger.info(
                f"Optimization at step {update_step}: grad_norm={grad_norm:.4f}, "
                f"tokens_seen={tokens_seen}"
            )
            logger.info(
                f"Timing at step {update_step}: update_time={update_time:.3f}s, "
                f"iter_time_avg={iter_time_avg:.3f}s"
            )
            logger.info(
                f"Memory at step {update_step}: allocated={current_memory_allocated_mb:.1f}MB, "
                f"reserved={current_memory_reserved_mb:.1f}MB, "
                f"peak_allocated={peak_memory_allocated_mb:.1f}MB, "
                f"peak_reserved={peak_memory_reserved_mb:.1f}MB, "
                f"running_peak_allocated={running_peak_memory_allocated_mb:.1f}MB, "
                f"running_peak_reserved={running_peak_memory_reserved_mb:.1f}MB"
            )
            tracker.log({
                "loss": loss.item(),
                "lr": lr,
                "update_step": update_step,
                "tokens_seen": tokens_seen,
                "update_time": update_time,
                "iter_time_avg": iter_time_avg,
                "throughput_tokens": tokens_in_update / update_time,
                "throughput_examples": args.total_batch_size / update_time,
                "throughput_batches": batches_in_update / update_time,
                "grad_norm": grad_norm,
                "memory_allocated_mb": current_memory_allocated_mb,
                "memory_reserved_mb": current_memory_reserved_mb,
                "peak_memory_allocated_mb": peak_memory_allocated_mb,
                "peak_memory_reserved_mb": peak_memory_reserved_mb,
                "running_peak_memory_allocated_mb": running_peak_memory_allocated_mb,
                "running_peak_memory_reserved_mb": running_peak_memory_reserved_mb,
            }, step=update_step)

        # Save resume checkpoint at specified steps
        if update_step in args.save_steps:
            ckpt_dir = args.checkpoint_dir or args.save_dir
            save_path = os.path.join(ckpt_dir, f"resume_step{update_step}")
            logger.info(f"Saving resume checkpoint at step {update_step}")
            local_data_state = dataset.state_dict()
            if args.single_gpu:
                # For single GPU, save directly
                if global_rank == 0:
                    os.makedirs(save_path, exist_ok=True)
                    torch.save(_unwrap_model(model).state_dict(), os.path.join(save_path, "pytorch_model.bin"))
                    torch.save(optimizer.state_dict(), os.path.join(save_path, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(save_path, "scheduler.pt"))
                    rng_state = {
                        'torch_rng': torch.get_rng_state(),
                        'npu_rng': torch.npu.get_rng_state(),
                        'numpy_rng': np.random.get_state(),
                        'python_rng': random.getstate(),
                    }
                    torch.save(rng_state, os.path.join(save_path, "rng_state.pt"))
                    with open(os.path.join(save_path, "training_state.json"), "w") as f:
                        json.dump({
                            "global_step": global_step,
                            "update_step": update_step,
                            "tokens_seen": tokens_seen,
                            "tokens_seen_before": tokens_seen,
                        }, f, indent=2)
                    torch.save(local_data_state, _data_state_path(save_path))
                    _save_resume_manifest(save_path, resume_manifest)
                    logger.info(f"Resume checkpoint saved to {save_path}")
            else:
                save_resume_checkpoint(model, optimizer, scheduler, args,
                                       update_step, global_step, tokens_seen,
                                       save_path, global_rank, data_state=local_data_state,
                                       resume_manifest=resume_manifest)

        # Save at regular intervals
        if args.save_every and update_step % args.save_every == 0:
            save_path = os.path.join(args.save_dir, f"step_{update_step}")
            logger.info(f"Saving checkpoint at step {update_step}")
            save_final_checkpoint(model, args, update_step, tokens_seen, save_path, global_rank)

        _reset_npu_peak_memory_stats(device)
        update_timer_start = time.time()

    # ##############################
    # END of training loop
    # ##############################
    logger.info("Training finished")

    # Save final checkpoint
    if args.save_dir:
        save_path = os.path.join(args.save_dir, "final")
        save_final_checkpoint(model, args, update_step, tokens_seen, save_path, global_rank)

    if global_rank == 0:
        tracker.finish()
        pbar.close()

    for handle in dtype_debug_hooks:
        handle.remove()


if __name__ == "__main__":
    print("Starting script")
    args = parse_args(None)
    main(args)
