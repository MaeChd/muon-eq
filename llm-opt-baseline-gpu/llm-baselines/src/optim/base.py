import copy
import math
import time
from contextlib import nullcontext
from pathlib import Path
from tqdm import tqdm
import sys
import torch
import wandb
import yaml

from logger.logger import DynamicsLogger
from optim.weight_averaging import (ExponentialWeightAverager, WeightAverager,
                                    eval_ewa, eval_wa)

from .utils import (eval, get_batch, get_parameter_norms, load_checkpoint,
                    load_worker_state, log_prodigy_lr, save_checkpoint,
                    save_worker_state, visualize_routing)

def _fmt_hms(seconds: float) -> str:
    if seconds is None or math.isnan(seconds) or seconds < 0:
        return "?"
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _fmt_gb(num_bytes: float) -> str:
    if num_bytes is None:
        return "?"
    return f"{num_bytes / 1024**3:.1f}GB"


def _get_cuda_memory_stats():
    if not torch.cuda.is_available():
        return {}

    device = torch.cuda.current_device()
    return {
        "memory_allocated_bytes": torch.cuda.memory_allocated(device),
        "memory_reserved_bytes": torch.cuda.memory_reserved(device),
        "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }


def _distributed_max_scalar(value: float, device: str) -> float:
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return value

    dist_device = device if "cuda" in device and torch.cuda.is_available() else "cpu"
    tensor = torch.tensor(float(value), device=dist_device, dtype=torch.float64)
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MAX)
    return tensor.item()

def train(
    model,
    opt,
    datareaders,
    scheduler,
    exp_dir,
    distributed_backend,
    cfg,
):
    not_compiled_model = model
    if cfg.compile:
        print(f"Compiling model ...")
        model = torch.compile(model)

    if "cuda" in cfg.device:
        type_ctx = torch.amp.autocast(
            device_type="cuda",
            dtype={
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }[cfg.dtype],
        )
    else:
        type_ctx = nullcontext()

    if cfg.resume_from:
        # This is a full resume including the model weights, optimizer, state
        # dataloader state, random seed, etc. Not indended for fine tuning or
        # other scenarios where some of these should change.
        print(f"\nResuming Training From {cfg.resume_from}")
        ckpt_dir = Path(cfg.resume_from)
        curr_iter = load_checkpoint(
            model,
            opt,
            scheduler,
            ckpt_dir / "main.pt",
            cfg.device,
        )
        load_worker_state(ckpt_dir)
    else:
        curr_iter = 0

    if cfg.weight_average:
        # This does generally not support resuming training, but will work if
        # cfg.wa_interval perfectly divides the iteration number of the chkpt.
        # Otherwise, the first avg will not be correctly computed, with a bias
        # towards the first sample and missing values for earlier iterations.
        weight_averager = WeightAverager(
            not_compiled_model,
            horizon=cfg.wa_horizon,
            interval=cfg.wa_interval,
            save_dir=None if cfg.wa_use_temp_dir else exp_dir / "avgs",
            dtype={
                "float32": torch.float32,
                "float64": torch.float64,
            }[cfg.wa_dtype],
            count=curr_iter,
        )
    if cfg.exponential_weight_average:
        ewa = ExponentialWeightAverager(
            not_compiled_model,
            interval=cfg.ewa_interval,
            decay=cfg.ewa_decay,
            warmup=cfg.warmup_steps if cfg.ewa_after_warmup else 0,
            dtype={
                "float32": torch.float32,
                "float64": torch.float64,
            }[cfg.wa_dtype],
        )

    if distributed_backend.is_master_process() and cfg.log_dynamics:
        with open(cfg.dynamics_logger_cfg, "r") as f:
            dlcfg = yaml.safe_load(f)

        # Hooks into optimizer
        dlogger = DynamicsLogger(
            model, opt, dlcfg, cfg.results_base_folder, wandb=cfg.wandb
        )
        dlogger.iteration = curr_iter

    if "cuda" in cfg.device and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(torch.cuda.current_device())

    wall_start_time = time.perf_counter()
    substep = curr_iter * cfg.acc_steps
    start_iter = curr_iter
    start_substep = substep
    train_reader, val_reader = datareaders["train"], datareaders["val"]
    train_reader.set_step(substep)
    stats = {"train_loss": [], "val_loss": [], "val_pp": [], "val_acc": []}
    grad_norms = []
    model.train()
    pbar = None
    iter_dt_ema = None  # Smooth iteration time with EMA for a steadier ETA.
    iter_dt_sum = 0.0
    rank = getattr(distributed_backend, "rank", 0)
    if distributed_backend.is_master_process():
        pbar = tqdm(
            total=cfg.iterations,
            initial=curr_iter,
            desc=f"train[r{rank}]",
            dynamic_ncols=True,
            file=sys.stdout,     # Use stdout instead of the default stderr.
            leave=True,
            mininterval=1.0,     # Avoid overly frequent refreshes that can destabilize output systems.
        )

    try:
        while curr_iter <= cfg.iterations:
            # Save permanent checkpoint
            if cfg.permanent_ckpt_interval > 0:
                if curr_iter % cfg.permanent_ckpt_interval == 0:
                    ckpt_dir = exp_dir / "ckpts" / str(curr_iter)
                    if distributed_backend.is_master_process():
                        save_checkpoint(model, opt, scheduler, curr_iter, ckpt_dir)
                    save_worker_state(ckpt_dir)

            # Save temporary checkpoint for resuming training
            if cfg.latest_ckpt_interval > 0:
                if curr_iter % cfg.latest_ckpt_interval == 0 or curr_iter == cfg.iterations:
                    ckpt_dir = exp_dir / "ckpts" / "latest"
                    if distributed_backend.is_master_process():
                        save_checkpoint(model, opt, scheduler, curr_iter, ckpt_dir)
                    save_worker_state(ckpt_dir)

            ws = distributed_backend.get_world_size()
            tokens = ws * substep * cfg.sequence_length * cfg.batch_size
            epoch = tokens / train_reader.num_tokens
            if (
                curr_iter % cfg.eval_interval == 0
                or curr_iter == cfg.iterations
                or (curr_iter in cfg.full_eval_at)
            ):
                eval_and_log(
                    tokens,
                    curr_iter,
                    epoch,
                    model,
                    val_reader,
                    type_ctx,
                    distributed_backend,
                    cfg,
                    opt,
                    full_eval=(curr_iter in cfg.full_eval_at),
                )

                if curr_iter > cfg.wa_interval and cfg.weight_average:
                    eval_wa(
                        curr_iter,
                        not_compiled_model,
                        weight_averager,
                        val_reader,
                        type_ctx,
                        distributed_backend,
                        cfg,
                        full_eval=(curr_iter in cfg.full_eval_at),
                    )

                if cfg.exponential_weight_average:
                    eval_ewa(
                        curr_iter,
                        not_compiled_model,
                        ewa,
                        val_reader,
                        type_ctx,
                        distributed_backend,
                        cfg,
                        full_eval=(curr_iter in cfg.full_eval_at),
                    )

            if curr_iter == cfg.iterations:
                # Save checkpoints and evaluate at final iteration, but no need to train further
                break

            # Train model
            t_start = time.perf_counter_ns()
            for microstep_idx in range(cfg.acc_steps):  # gradient accumulation
                x, y = get_batch(
                    train_reader,
                    device=cfg.device,
                    vocab_size=cfg.vocab_size,
                    batch_name=f"train(iter={curr_iter}, microstep={microstep_idx})",
                )
                with type_ctx:
                    with distributed_backend.get_context_for_microstep_forward(
                        model=model,
                        microstep_idx=microstep_idx,
                        gradient_accumulation_steps=cfg.acc_steps,
                    ):
                        outputs = model(x, targets=y, moe=cfg.moe)

                loss = outputs["loss"] / cfg.acc_steps
                loss.backward()
                substep += 1

            if cfg.grad_clip != 0.0:
                if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.module.parameters(), cfg.grad_clip
                    )
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), cfg.grad_clip
                    )
                grad_norms.append(grad_norm)

            if cfg.opt == "sf-sgd" or cfg.opt == "sf-adamw":
                opt.train()
            (
                opt.step()
                if cfg.opt != "sophiag"
                else opt.step(bs=cfg.sophia_bs * cfg.sequence_length)
            )
            if cfg.scheduler != "none":
                scheduler.step()
            if cfg.opt == "sophiag":
                opt.zero_grad(set_to_none=True)
                if curr_iter % cfg.precondition_frequency == cfg.precondition_frequency - 1:
                    sample_again = model(x, targets=y, get_logits=True)
                    samp_dist = torch.distributions.Categorical(
                        logits=sample_again["logits"]
                    )
                    y_sample = samp_dist.sample()
                    loss_sampled = torch.nn.functional.cross_entropy(
                        sample_again["logits"].view(-1, sample_again["logits"].size(-1)),
                        y_sample.view(-1),
                        ignore_index=-1,
                    )
                    (loss_sampled / cfg.acc_steps).backward()
                    opt.update_hessian()
                    opt.zero_grad(set_to_none=True)
                    model.zero_grad()
            elif cfg.opt == "mars":
                opt.zero_grad(set_to_none=True)
                opt.update_last_grad()
            else:
                opt.zero_grad(set_to_none=True)

            if cfg.weight_average:
                weight_averager.step(
                    not_compiled_model, distributed_backend.is_master_process()
                )
            if cfg.exponential_weight_average:
                ewa.step(not_compiled_model, distributed_backend.is_master_process())

            dt = (time.perf_counter_ns() - t_start) / 1e9
            iter_dt_sum += dt

            curr_iter += 1
            # ---- tqdm progress (master only) ----
            if pbar is not None:
                # Smooth iteration time for a steadier ETA.
                iter_dt_ema = dt if iter_dt_ema is None else (0.9 * iter_dt_ema + 0.1 * dt)

                ws = distributed_backend.get_world_size()
                tokens_per_iter = ws * cfg.acc_steps * cfg.sequence_length * cfg.batch_size
                tokens_done = ws * substep * cfg.sequence_length * cfg.batch_size
                epoch_f = tokens_done / train_reader.num_tokens
                elapsed = time.perf_counter() - wall_start_time
                tok_per_sec = tokens_done / elapsed if elapsed > 0 else float("nan")

                # ETA to the next epoch boundary.
                frac = epoch_f - math.floor(epoch_f)
                remaining_frac = 1.0 - frac if frac > 1e-12 else 1.0
                remaining_tokens = remaining_frac * train_reader.num_tokens
                iters_to_epoch = remaining_tokens / tokens_per_iter
                eta_to_epoch = iters_to_epoch * iter_dt_ema

                # Display the current-iteration loss after multiplying /acc_steps back in.
                train_loss = loss.detach().float().cpu().item() * cfg.acc_steps
                lr0 = opt.param_groups[0]["lr"]

                peak_mem = ""
                if "cuda" in cfg.device:
                    cuda_stats = _get_cuda_memory_stats()
                    peak_mem = _fmt_gb(cuda_stats.get("peak_memory_allocated_bytes"))

                desc = f"iter {curr_iter}/{cfg.iterations} | epoch {epoch_f:.3f}"
                postfix = dict(
                    loss=f"{train_loss:.3f}",
                    dt=f"{iter_dt_ema:.2f}s",
                    toks=f"{tok_per_sec:,.0f}/s",
                    lr=f"{lr0:.2e}",
                    elapsed=_fmt_hms(elapsed),
                    ep_eta=_fmt_hms(eta_to_epoch),
                    peak_mem=peak_mem,
                )

                # Keep refresh=False so each set_* call does not trigger printing.
                pbar.set_description(desc, refresh=False)
                pbar.set_postfix(postfix, refresh=False)

                # Refresh only once on update.
                pbar.update(1)

            do_interval_log = cfg.log_interval and curr_iter % cfg.log_interval == 0
            if do_interval_log:
                train_loss = loss.detach().cpu().item() * cfg.acc_steps
                train_aux_losses = {
                    f"train/{k}": v for k, v in outputs["aux_losses"].items()
                }

                current_lrs = [param_group["lr"] for param_group in opt.param_groups]
                ws = distributed_backend.get_world_size()
                tokens_done = ws * substep * cfg.sequence_length * cfg.batch_size
                epoch_done = tokens_done / train_reader.num_tokens
                elapsed = time.perf_counter() - wall_start_time
                tok_per_sec = tokens_done / elapsed if elapsed > 0 else float("nan")
                mean_iter_dt = iter_dt_sum / max(curr_iter - start_iter, 1)
                cuda_stats = _get_cuda_memory_stats() if "cuda" in cfg.device else {}
                peak_alloc_gb = (
                    _distributed_max_scalar(
                        cuda_stats.get("peak_memory_allocated_bytes", 0) / 1024**3,
                        cfg.device,
                    )
                    if cuda_stats
                    else 0.0
                )
                peak_reserved_gb = (
                    _distributed_max_scalar(
                        cuda_stats.get("peak_memory_reserved_bytes", 0) / 1024**3,
                        cfg.device,
                    )
                    if cuda_stats
                    else 0.0
                )

                if distributed_backend.is_master_process():  # Only log on master rank
                    if cfg.opt == "prodigy":
                        prodigy_efective_lrs = log_prodigy_lr(opt)

                    tqdm.write(
                        f"Train: Iter={curr_iter} ({epoch_done:0.3f} epochs) "
                        f"train_loss={train_loss:.3f} iter_dt={dt:.2e}s "
                        f"mean_iter_dt={mean_iter_dt:.2e}s "
                        f"elapsed={_fmt_hms(elapsed)} "
                        f"tok/s={tok_per_sec:,.0f} "
                        f"peak_mem={peak_alloc_gb:.2f}GB "
                        f"lr={current_lrs[0]:.2e}"
                    )
                    if cfg.opt == "prodigy":
                        tqdm.write(f"effective_lr={prodigy_efective_lrs[0]:.2e}")

                    if cfg.wandb:
                        wandb_logs = {
                            "tokens": tokens_done,
                            "iter": curr_iter,
                            "train/loss": train_loss,
                            "train/perplexity": 2.71828**train_loss,
                            "lr": current_lrs[0],
                            "iter_dt": dt,
                            "iter_dt_mean": mean_iter_dt,
                            "elapsed_time_s": elapsed,
                            "tokens_per_sec": tok_per_sec,
                            "max_grad_norm": max(grad_norms).item() if grad_norms else 0,
                            "mean_grad_norm": (
                                torch.tensor(grad_norms).mean().item() if grad_norms else 0
                            ),
                            "peak_memory_allocated_gb": peak_alloc_gb,
                            "peak_memory_reserved_gb": peak_reserved_gb,
                            **train_aux_losses,
                        }

                        if cfg.opt == "prodigy":
                            wandb_logs["effective_lr"] = prodigy_efective_lrs[0]

                        if cfg.log_parameter_norms:
                            raw_model = distributed_backend.get_raw_model(model)
                            model_norm = get_parameter_norms(raw_model, order=cfg.norm_order)
                            wandb_logs["model_norm"] = model_norm

                        wandb.log(wandb_logs)

                grad_norms = []
    finally:
        if pbar is not None:
            pbar.close()

    ws = distributed_backend.get_world_size()
    total_runtime_s = time.perf_counter() - wall_start_time
    run_iters = max(curr_iter - start_iter, 0)
    total_tokens_done = ws * substep * cfg.sequence_length * cfg.batch_size
    run_tokens_done = ws * max(substep - start_substep, 0) * cfg.sequence_length * cfg.batch_size
    mean_iter_dt = iter_dt_sum / max(run_iters, 1)
    tok_per_sec = run_tokens_done / total_runtime_s if total_runtime_s > 0 else float("nan")
    cuda_stats = _get_cuda_memory_stats() if "cuda" in cfg.device else {}
    peak_memory_allocated_bytes = (
        int(
            _distributed_max_scalar(
                cuda_stats.get("peak_memory_allocated_bytes", 0), cfg.device
            )
        )
        if cuda_stats
        else 0
    )
    peak_memory_reserved_bytes = (
        int(
            _distributed_max_scalar(
                cuda_stats.get("peak_memory_reserved_bytes", 0), cfg.device
            )
        )
        if cuda_stats
        else 0
    )

    stats.update(
        {
            "train_runtime_s": total_runtime_s,
            "train_runtime_hms": _fmt_hms(total_runtime_s),
            "train_iterations_completed": run_iters,
            "train_substeps_completed": max(substep - start_substep, 0),
            "train_tokens_processed": run_tokens_done,
            "total_tokens_seen": total_tokens_done,
            "mean_iter_dt_s": mean_iter_dt,
            "tokens_per_sec": tok_per_sec,
            "peak_memory_allocated_bytes": peak_memory_allocated_bytes,
            "peak_memory_reserved_bytes": peak_memory_reserved_bytes,
            "peak_memory_allocated_gb": peak_memory_allocated_bytes / 1024**3,
            "peak_memory_reserved_gb": peak_memory_reserved_bytes / 1024**3,
        }
    )

    if distributed_backend.is_master_process():
        peak_alloc_gb = stats["peak_memory_allocated_gb"]
        peak_reserved_gb = stats["peak_memory_reserved_gb"]
        tqdm.write(
            "Training finished: "
            f"runtime={stats['train_runtime_hms']} "
            f"iters={run_iters} "
            f"tokens={run_tokens_done:,} "
            f"mean_iter_dt={mean_iter_dt:.2e}s "
            f"tok/s={tok_per_sec:,.0f} "
            f"peak_alloc={peak_alloc_gb:.2f}GB "
            f"peak_reserved={peak_reserved_gb:.2f}GB"
        )

        if cfg.wandb and wandb.run is not None:
            wandb.run.summary["train_runtime_s"] = total_runtime_s
            wandb.run.summary["train_runtime_hms"] = stats["train_runtime_hms"]
            wandb.run.summary["train_iterations_completed"] = run_iters
            wandb.run.summary["train_tokens_processed"] = run_tokens_done
            wandb.run.summary["mean_iter_dt_s"] = mean_iter_dt
            wandb.run.summary["tokens_per_sec"] = tok_per_sec
            wandb.run.summary["peak_memory_allocated_gb"] = peak_alloc_gb
            wandb.run.summary["peak_memory_reserved_gb"] = peak_reserved_gb

    return stats


def eval_and_log(
    tokens,
    curr_iter,
    epoch,
    model,
    val_reader,
    type_ctx,
    distributed_backend,
    cfg,
    opt,
    full_eval=False,
):
    if not distributed_backend.is_master_process():
        # Only evaluate and log on master rank
        return

    model.eval()
    if cfg.opt == "sf-sgd" or cfg.opt == "sf-adamw":
        opt.eval()

    if curr_iter == cfg.iterations or full_eval:
        max_num_batches = val_reader.num_batches()
    else:
        max_num_batches = cfg.eval_batches

    # to make sure we start from the beginning of the validation set,
    # i.e. repeat the same batches
    val_reader.set_step(0)
    val_acc, val_loss, val_perplexity, val_aux_losses, router_logits = eval(
        model,
        val_reader,
        cfg.device,
        max_num_batches=max_num_batches,
        ctx=type_ctx,
        moe=cfg.moe,
        get_router_logits=cfg.moe and cfg.plot_router_logits,
        cfg=cfg,
    )

    tqdm.write(
        f">Eval: Iter={curr_iter} ({epoch:0.3f} epochs) "
        f"val_loss={val_loss:.3f} "
        f"val_pp={val_perplexity:.3f} "
        f"val_acc={val_acc:3f}"
    )

    if cfg.wandb:
        if curr_iter == cfg.iterations or full_eval:
            logs = {
                "tokens": tokens,
                "iter": curr_iter,
                "final-val/loss": val_loss,
                "final-val/perplexity": val_perplexity,
                "final-val/acc": val_acc,
                **val_aux_losses,
            }
        else:
            logs = {
                "tokens": tokens,
                "iter": curr_iter,
                "val/loss": val_loss,
                "val/perplexity": val_perplexity,
                "val/acc": val_acc,
                **val_aux_losses,
            }
        if cfg.moe and cfg.plot_router_logits:
            routing_logs = visualize_routing(router_logits, cfg)
            logs = {**logs, **routing_logs}

        wandb.log(logs)
        if cfg.eval_seq_prefix != "none" and (
            curr_iter % (cfg.eval_interval * 5) == 0 or curr_iter == cfg.iterations
        ):
            text_table = wandb.Table(columns=["itr", "val-pp", "text"])

            out_str = distributed_backend.get_raw_model(model).generate_from_string(
                cfg.eval_seq_prefix,
                max_new_tokens=40,
                temperature=0.9,
                top_k=None,
            )
            text_table.add_data(curr_iter, val_perplexity, out_str)
            # why a copy? see github.com/wandb/wandb/issues/2981
            wandb.log({f"generated-text-{wandb.run.name}": copy.copy(text_table)})
    model.train()
