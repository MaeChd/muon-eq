import argparse
import copy
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from compare_resnet_optimizers import (  # noqa: E402
    DEFAULT_LR_SEARCH_SPACE,
    DEFAULT_OPTIMIZER_CONFIGS,
    OPTIMIZER_ORDER,
    build_log_path,
    close_logger,
    create_model,
    create_optimizer,
    evaluate_model,
    finish_swanlab_run,
    finish_wandb_run,
    json_ready,
    log_info,
    maybe_start_swanlab_run,
    maybe_start_wandb_run,
    none_or_int,
    print_summary,
    progress_bar,
    resolve_optimizer_order,
    safe_log_swanlab_metrics,
    safe_log_wandb_metrics,
    save_initial_model_state,
    set_seed,
    setup_logger,
    summarize_results,
)


def ensure_cifar10_downloaded(data_dir: str, rank: int) -> None:
    if rank == 0:
        torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True)
        torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True)
    dist.barrier()


def is_rank0() -> bool:
    return dist.get_rank() == 0


def rank0_print(message: str) -> None:
    if is_rank0():
        print(message)


def broadcast_object(value: Any, src: int = 0) -> Any:
    objects = [value]
    dist.broadcast_object_list(objects, src=src)
    return objects[0]


def resolve_distributed_device(requested: str, local_rank: int) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda", local_rank)
        return torch.device("cpu")

    requested_device = torch.device(requested)
    if requested_device.type == "cuda":
        return torch.device("cuda", local_rank)
    return requested_device


def init_distributed(requested_device: str, backend: Optional[str]) -> tuple[torch.device, int, int, int, str]:
    if "RANK" not in os.environ or "LOCAL_RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise RuntimeError("This script must be launched with torchrun so RANK/LOCAL_RANK/WORLD_SIZE are available.")

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device = resolve_distributed_device(requested_device, local_rank)
    dist_backend = backend or ("nccl" if device.type == "cuda" else "gloo")

    if device.type == "cuda":
        torch.cuda.set_device(local_rank)

    dist.init_process_group(backend=dist_backend)
    return device, rank, local_rank, world_size, dist_backend


def destroy_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def get_cifar10_ddp_loaders(
    data_dir: str,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    rank: int,
    world_size: int,
) -> tuple[DataLoader, DistributedSampler, Optional[DataLoader]]:
    ensure_cifar10_downloaded(data_dir, rank)

    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=False, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=False, transform=transform_test)

    train_sampler = DistributedSampler(
        trainset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=False,
    )
    trainloader = DataLoader(
        trainset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    testloader = None
    if is_rank0():
        testloader = DataLoader(
            testset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    return trainloader, train_sampler, testloader


def reduce_train_metrics(loss_sum: torch.Tensor, correct_sum: torch.Tensor, sample_sum: torch.Tensor) -> Dict[str, float]:
    stats = torch.stack([loss_sum, correct_sum, sample_sum]).to(torch.float64)
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    total_loss, total_correct, total_samples = stats.tolist()
    return {
        "loss": total_loss / total_samples,
        "acc": 100.0 * total_correct / total_samples,
    }


def train_one_epoch_ddp(
    model: DDP,
    dataloader: DataLoader,
    train_sampler: DistributedSampler,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    progress_desc: str,
    show_progress: bool,
) -> Dict[str, float]:
    train_sampler.set_epoch(epoch)
    model.train()
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    correct_sum = torch.zeros((), device=device, dtype=torch.float64)
    sample_sum = torch.zeros((), device=device, dtype=torch.float64)

    iterator = progress_bar(
        dataloader,
        desc=progress_desc,
        disable=(not show_progress) or (not is_rank0()),
        leave=False,
    )
    for batch_idx, (inputs, targets) in enumerate(iterator, start=1):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        if hasattr(optimizer, "update_last_grad"):
            optimizer.update_last_grad()

        batch_size = targets.size(0)
        loss_sum += loss.detach().to(torch.float64) * batch_size
        correct_sum += outputs.argmax(dim=1).eq(targets).sum().to(torch.float64)
        sample_sum += torch.tensor(batch_size, device=device, dtype=torch.float64)

        if hasattr(iterator, "set_postfix") and is_rank0():
            iterator.set_postfix(
                loss=f"{(loss_sum / sample_sum).item():.4f}",
                acc=f"{(100.0 * correct_sum / sample_sum).item():.2f}",
            )

    if hasattr(iterator, "close"):
        iterator.close()

    return reduce_train_metrics(loss_sum, correct_sum, sample_sum)


def evaluate_model_ddp(
    model: DDP,
    dataloader: Optional[DataLoader],
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    if is_rank0():
        metrics = evaluate_model(model.module, dataloader, criterion, device)
    else:
        metrics = None
    return broadcast_object(metrics)


def ensure_initial_model_state(seed: int, device: torch.device, output_dir: Path) -> Path:
    state_path = output_dir / f"resnet18_initial_state_seed{seed}.pt"
    if is_rank0():
        save_initial_model_state(seed=seed, device=device, output_dir=output_dir)
    dist.barrier()
    return state_path


def wrap_model_ddp(model: nn.Module, device: torch.device, local_rank: int) -> DDP:
    if device.type == "cuda":
        return DDP(model, device_ids=[local_rank], output_device=local_rank)
    return DDP(model)


def run_single_experiment_ddp(
    *,
    optimizer_name: str,
    optimizer_config: Dict[str, Any],
    epochs: int,
    run_seed: int,
    run_id: int,
    phase: str,
    initial_state_path: Path,
    data_dir: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    rank: int,
    local_rank: int,
    world_size: int,
    use_swanlab: bool,
    use_wandb: bool,
    wandb_project: str,
    wandb_entity: Optional[str],
    logger: Optional[logging.Logger],
    show_progress: bool,
) -> Optional[Dict[str, Any]]:
    set_seed(run_seed + rank)
    pin_memory = device.type == "cuda"

    model = create_model(device=device, initial_state_path=initial_state_path)
    model = wrap_model_ddp(model, device, local_rank)
    trainloader, train_sampler, testloader = get_cifar10_ddp_loaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        rank=rank,
        world_size=world_size,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(optimizer_name, model, copy.deepcopy(optimizer_config))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

    run_name = f"ddp_{phase}_{optimizer_name}_run{run_id}_seed{run_seed}_ws{world_size}"
    swanlab_active = False
    wandb_run = None
    if is_rank0():
        swanlab_active = maybe_start_swanlab_run(
            enabled=use_swanlab,
            run_name=run_name,
            config={
                "phase": phase,
                "optimizer": optimizer_name,
                "run_id": run_id,
                "seed": run_seed,
                "epochs": epochs,
                "batch_size_per_rank": batch_size,
                "world_size": world_size,
                "global_batch_size": batch_size * world_size,
                "optimizer_config": optimizer_config,
                "ddp": True,
            },
            logger=logger,
        )
        wandb_run = maybe_start_wandb_run(
            enabled=use_wandb,
            run_name=run_name,
            config={
                "phase": phase,
                "optimizer": optimizer_name,
                "run_id": run_id,
                "seed": run_seed,
                "epochs": epochs,
                "batch_size_per_rank": batch_size,
                "world_size": world_size,
                "global_batch_size": batch_size * world_size,
                "optimizer_config": optimizer_config,
                "ddp": True,
            },
            project=wandb_project,
            entity=wandb_entity,
            logger=logger,
        )

    best_acc = float("-inf")
    best_epoch = -1
    epoch_metrics: List[Dict[str, Any]] = []
    start_time = time.time()

    if is_rank0():
        log_info(logger, f"\n[{phase}] optimizer={optimizer_name} run={run_id} seed={run_seed} world_size={world_size}")
        log_info(logger, f"config={json.dumps(json_ready(optimizer_config), ensure_ascii=False)}")

    for epoch in range(epochs):
        epoch_start = time.time()
        train_metrics = train_one_epoch_ddp(
            model,
            trainloader,
            train_sampler,
            optimizer,
            criterion,
            device,
            epoch,
            progress_desc=f"{phase}:{optimizer_name}:run{run_id} epoch {epoch + 1}/{epochs}",
            show_progress=show_progress,
        )
        dist.barrier()
        eval_metrics = evaluate_model_ddp(model, testloader, criterion, device)
        current_lr = optimizer.param_groups[0]["lr"]

        if eval_metrics["acc"] > best_acc:
            best_acc = eval_metrics["acc"]
            best_epoch = epoch

        epoch_result = {
            "epoch": epoch + 1,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "test_loss": eval_metrics["loss"],
            "test_acc": eval_metrics["acc"],
            "best_acc": best_acc,
            "lr": current_lr,
            "epoch_time_sec": time.time() - epoch_start,
        }
        epoch_metrics.append(epoch_result)

        if is_rank0():
            swanlab_active = safe_log_swanlab_metrics(swanlab_active, epoch_result, logger=logger)
            wandb_run = safe_log_wandb_metrics(wandb_run, epoch_result, logger=logger)
            log_info(
                logger,
                f"epoch {epoch + 1:03d}/{epochs:03d} "
                f"train_acc={train_metrics['acc']:.2f} "
                f"test_acc={eval_metrics['acc']:.2f} "
                f"best_acc={best_acc:.2f} "
                f"lr={current_lr:.6f} "
                f"time={epoch_result['epoch_time_sec']:.2f}s"
            )

        scheduler.step()

    dist.barrier()
    if is_rank0():
        finish_swanlab_run(swanlab_active, logger=logger)
        finish_wandb_run(wandb_run, logger=logger)

    total_time = time.time() - start_time
    result = None
    if is_rank0():
        result = {
            "phase": phase,
            "optimizer": optimizer_name,
            "run_id": run_id,
            "seed": run_seed,
            "epochs": epochs,
            "optimizer_config": copy.deepcopy(optimizer_config),
            "best_acc": best_acc,
            "best_epoch": best_epoch + 1,
            "final_acc": epoch_metrics[-1]["test_acc"],
            "final_loss": epoch_metrics[-1]["test_loss"],
            "wall_time_sec": total_time,
            "epoch_metrics": epoch_metrics,
            "world_size": world_size,
            "batch_size_per_rank": batch_size,
            "global_batch_size": batch_size * world_size,
        }

    dist.barrier()
    return result


def run_lr_grid_search_ddp(
    *,
    optimizer_order: List[str],
    base_optimizer_configs: Dict[str, Dict[str, Any]],
    epochs: int,
    base_seed: int,
    initial_state_path: Path,
    data_dir: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    rank: int,
    local_rank: int,
    world_size: int,
    use_swanlab: bool,
    use_wandb: bool,
    wandb_project: str,
    wandb_entity: Optional[str],
    logger: Optional[logging.Logger],
    show_progress: bool,
) -> Dict[str, Any]:
    if is_rank0():
        log_info(logger, "\n=== DDP LR grid search ===")
    grid_results: Optional[Dict[str, Any]] = None
    if is_rank0():
        grid_results = {
            "enabled": True,
            "epochs": epochs,
            "search_space": {name: copy.deepcopy(DEFAULT_LR_SEARCH_SPACE[name]) for name in optimizer_order},
            "best_lrs": {},
            "trials": [],
            "ddp": True,
            "world_size": world_size,
            "batch_size_per_rank": batch_size,
            "global_batch_size": batch_size * world_size,
        }

    for optimizer_name in optimizer_order:
        best_lr = None
        best_acc = float("-inf")
        base_config = copy.deepcopy(base_optimizer_configs[optimizer_name])

        for trial_id, lr in enumerate(DEFAULT_LR_SEARCH_SPACE[optimizer_name]):
            trial_config = copy.deepcopy(base_config)
            trial_config["lr"] = lr
            result = run_single_experiment_ddp(
                optimizer_name=optimizer_name,
                optimizer_config=trial_config,
                epochs=epochs,
                run_seed=base_seed,
                run_id=trial_id,
                phase="grid_search",
                initial_state_path=initial_state_path,
                data_dir=data_dir,
                batch_size=batch_size,
                num_workers=num_workers,
                device=device,
                rank=rank,
                local_rank=local_rank,
                world_size=world_size,
                use_swanlab=use_swanlab,
                use_wandb=use_wandb,
                wandb_project=wandb_project,
                wandb_entity=wandb_entity,
                logger=logger,
                show_progress=show_progress,
            )

            if is_rank0():
                trial_summary = {
                    "optimizer": optimizer_name,
                    "lr": lr,
                    "best_acc": result["best_acc"],
                    "final_acc": result["final_acc"],
                    "epochs": epochs,
                    "seed": base_seed,
                }
                grid_results["trials"].append(trial_summary)
                if result["best_acc"] > best_acc or (
                    abs(result["best_acc"] - best_acc) < 1e-12 and (best_lr is None or lr < best_lr)
                ):
                    best_acc = result["best_acc"]
                    best_lr = lr

        if is_rank0():
            grid_results["best_lrs"][optimizer_name] = best_lr
            log_info(logger, f"best lr for {optimizer_name}: {best_lr}")

    return broadcast_object(grid_results)


def build_output_path_ddp(requested_path: Optional[str]) -> Path:
    if requested_path:
        return Path(requested_path).expanduser().resolve()
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return (THIS_DIR / "results" / f"compare_resnet_optimizers_ddp_{timestamp}.json").resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DDP comparison of SGD, AdamW, and Muon-family optimizers on CIFAR-10 ResNet18.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128, help="Per-rank batch size.")
    parser.add_argument("--num-runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=str, default=str(THIS_DIR / "data"))
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--backend", type=str, default=None, help="DDP backend. Defaults to nccl on CUDA, else gloo.")
    parser.add_argument("--use-swanlab", action="store_true")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="cifar10_resnet_optimizer_compare_ddp")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--enable-lr-grid-search", action="store_true")
    parser.add_argument("--grid-search-epochs", type=int, default=20)
    parser.add_argument("--disable-tqdm", action="store_true")
    parser.add_argument("--muoneq-rc-phase", type=none_or_int, default=None, help="MuonEq-rc phase switch step. Use None to keep row/col normalization for all steps.")
    parser.add_argument(
        "--row-norm",
        type=str,
        default="l2",
        choices=("l2", "inf"),
        help="Norm used by MuonEq-rc / MuonEq-r / MuonEq-c row scaling.",
    )
    parser.add_argument(
        "--col-norm",
        type=str,
        default="l2",
        choices=("l2", "inf"),
        help="Norm used by MuonEq-rc / MuonEq-r / MuonEq-c column scaling.",
    )
    parser.add_argument(
        "--use-muonplus-update",
        action="store_true",
        help="Use Muon+ post-polar normalization for MuonEq-rc / MuonEq-r / MuonEq-c.",
    )
    parser.add_argument("--optimizers", nargs="+", default=None, help=f"Subset to run. Available: {', '.join(OPTIMIZER_ORDER)}")
    parser.add_argument("--exclude-optimizers", nargs="*", default=[], help=f"Optimizers to skip. Available: {', '.join(OPTIMIZER_ORDER)}")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device, rank, local_rank, world_size, backend = init_distributed(args.device, args.backend)
    logger: Optional[logging.Logger] = None

    try:
        selected_optimizer_order = resolve_optimizer_order(args.optimizers, args.exclude_optimizers)

        output_path_value = build_output_path_ddp(args.output_json) if is_rank0() else None
        output_path = Path(broadcast_object(str(output_path_value)))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        log_path = build_log_path(output_path)
        if is_rank0():
            logger = setup_logger(log_path, f"cifar10.compare.ddp.{output_path.stem}")

        if is_rank0():
            log_info(logger, "CIFAR-10 ResNet18 optimizer comparison (DDP)")
            log_info(logger, f"device={device}")
            log_info(logger, f"backend={backend}")
            log_info(logger, f"world_size={world_size}")
            log_info(logger, f"optimizers={selected_optimizer_order}")
            log_info(logger, f"output_json={output_path}")
            log_info(logger, f"log_path={log_path}")
        if device.type == "cuda":
            if is_rank0():
                log_info(logger, f"gpu={torch.cuda.get_device_name(local_rank)}")

        initial_state_path = ensure_initial_model_state(
            seed=args.seed,
            device=device,
            output_dir=output_path.parent / "initial_states",
        )

        lr_grid_search_results: Dict[str, Any] = {"enabled": False}
        optimizer_configs = {name: copy.deepcopy(DEFAULT_OPTIMIZER_CONFIGS[name]) for name in selected_optimizer_order}
        if "muoneq-rc" in optimizer_configs:
            optimizer_configs["muoneq-rc"]["phase"] = args.muoneq_rc_phase
        for optimizer_name in ("muoneq-rc", "muoneq-r", "muoneq-c"):
            if optimizer_name in optimizer_configs:
                optimizer_configs[optimizer_name]["row_norm"] = args.row_norm
                optimizer_configs[optimizer_name]["col_norm"] = args.col_norm
                optimizer_configs[optimizer_name]["use_muonplus"] = args.use_muonplus_update
        if args.enable_lr_grid_search:
            lr_grid_search_results = run_lr_grid_search_ddp(
                optimizer_order=selected_optimizer_order,
                base_optimizer_configs=optimizer_configs,
                epochs=args.grid_search_epochs,
                base_seed=args.seed,
                initial_state_path=initial_state_path,
                data_dir=args.data_dir,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                device=device,
                rank=rank,
                local_rank=local_rank,
                world_size=world_size,
                use_swanlab=args.use_swanlab,
                use_wandb=args.use_wandb,
                wandb_project=args.wandb_project,
                wandb_entity=args.wandb_entity,
                logger=logger,
                show_progress=not args.disable_tqdm,
            )
            for optimizer_name, best_lr in lr_grid_search_results["best_lrs"].items():
                optimizer_configs[optimizer_name]["lr"] = best_lr

        per_run_results: List[Dict[str, Any]] = []
        for run_id in range(args.num_runs):
            run_seed = args.seed + run_id
            initial_state_path = ensure_initial_model_state(
                seed=run_seed,
                device=device,
                output_dir=output_path.parent / "initial_states",
            )
            for optimizer_name in selected_optimizer_order:
                result = run_single_experiment_ddp(
                    optimizer_name=optimizer_name,
                    optimizer_config=optimizer_configs[optimizer_name],
                    epochs=args.epochs,
                    run_seed=run_seed,
                    run_id=run_id,
                    phase="main",
                    initial_state_path=initial_state_path,
                    data_dir=args.data_dir,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    device=device,
                    rank=rank,
                    local_rank=local_rank,
                    world_size=world_size,
                    use_swanlab=args.use_swanlab,
                    use_wandb=args.use_wandb,
                    wandb_project=args.wandb_project,
                    wandb_entity=args.wandb_entity,
                    logger=logger,
                    show_progress=not args.disable_tqdm,
                )
                if is_rank0():
                    per_run_results.append(result)

        if is_rank0():
            summary = summarize_results(per_run_results, selected_optimizer_order)
            print_summary(summary, selected_optimizer_order, logger=logger)

            payload = {
                "config": {
                    "epochs": args.epochs,
                    "batch_size_per_rank": args.batch_size,
                    "global_batch_size": args.batch_size * world_size,
                    "num_runs": args.num_runs,
                    "seed": args.seed,
                    "data_dir": args.data_dir,
                    "num_workers": args.num_workers,
                    "device": device,
                    "backend": backend,
                    "world_size": world_size,
                    "use_swanlab": args.use_swanlab,
                    "use_wandb": args.use_wandb,
                    "wandb_project": args.wandb_project,
                    "wandb_entity": args.wandb_entity,
                    "disable_tqdm": args.disable_tqdm,
                    "log_path": log_path,
                    "available_optimizers": OPTIMIZER_ORDER,
                    "optimizer_order": selected_optimizer_order,
                    "default_optimizer_configs": DEFAULT_OPTIMIZER_CONFIGS,
                    "selected_optimizer_configs": optimizer_configs,
                    "initial_state_dir": output_path.parent / "initial_states",
                    "ddp": True,
                },
                "lr_grid_search": lr_grid_search_results,
                "optimizer_summaries": summary,
                "per_run_metrics": per_run_results,
            }

            output_path.write_text(json.dumps(json_ready(payload), indent=2))
            log_info(logger, f"\nSaved results to {output_path}")
            log_info(logger, f"Saved logs to {log_path}")
    finally:
        destroy_distributed()
        close_logger(logger)


if __name__ == "__main__":
    main()
