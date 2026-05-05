import argparse
import copy
import json
import logging
import math
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


THIS_DIR = Path(__file__).resolve().parent

from optimizers import CifarMuon, CifarMuonEq, CifarMuonMVR  # noqa: E402
from models import resnet  # noqa: E402

try:
    import swanlab
except ImportError:
    swanlab = None

try:
    import wandb
except ImportError:
    wandb = None

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

try:
    import numpy as np
except ImportError:
    np = None


OPTIMIZER_ORDER = ["sgd", "adamw", "muon", "muon_mvr", "muoneq-rc", "muoneq-r", "muoneq-c"]
DEFAULT_OPTIMIZER_CONFIGS: Dict[str, Dict[str, Any]] = {
    "sgd": {
        "lr": 0.05,
        "momentum": 0.9,
        "weight_decay": 1e-4,
    },
    "adamw": {
        "lr": 1e-3,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": 1e-4,
    },
    "muon": {
        "lr": 5e-2,
        "momentum": 0.9,
        "nesterov": True,
        "ns_steps": 5,
        "wd": 0.1,
        "adamw_betas": (0.95, 0.95),
        "adamw_eps": 1e-8,
    },
    "muon_mvr": {
        "lr": 5e-2,
        "momentum": 0.9,
        "gamma": 0.1,
        "ns_steps": 5,
        "weight_decay": 0.0,
        "adamw_betas": (0.95, 0.99),
        "eps": 1e-8,
    },
    "muoneq-rc": {
        "lr": 5e-2,
        "momentum": 0.9,
        "nesterov": True,
        "ns_steps": 5,
        "wd": 0.1,
        "adamw_betas": (0.95, 0.95),
        "adamw_eps": 1e-8,
        "rowcol_scale_exponent": -0.5,
        "rowcol_eps": 1e-8,
        "rowcol_clip": None,
        "row_norm": "l2",
        "col_norm": "l2",
        "normalize_mode": "rowcol",
        "phase": None,
        "use_muonplus": False,
    },
    "muoneq-r": {
        "lr": 5e-2,
        "momentum": 0.9,
        "nesterov": True,
        "ns_steps": 5,
        "wd": 0.1,
        "adamw_betas": (0.95, 0.95),
        "adamw_eps": 1e-8,
        "rowcol_scale_exponent": -0.5,
        "rowcol_eps": 1e-8,
        "rowcol_clip": None,
        "row_norm": "l2",
        "col_norm": "l2",
        "normalize_mode": "row",
        "use_muonplus": False,
    },
    "muoneq-c": {
        "lr": 5e-2,
        "momentum": 0.9,
        "nesterov": True,
        "ns_steps": 5,
        "wd": 0.1,
        "adamw_betas": (0.95, 0.95),
        "adamw_eps": 1e-8,
        "rowcol_scale_exponent": -0.5,
        "rowcol_eps": 1e-8,
        "rowcol_clip": None,
        "row_norm": "l2",
        "col_norm": "l2",
        "normalize_mode": "col",
        "use_muonplus": False,
    },
}
DEFAULT_LR_SEARCH_SPACE: Dict[str, List[float]] = {
    "sgd": [1e-1, 5e-2, 1e-2, 5e-3],
    "adamw": [5e-3, 1e-3, 5e-4, 1e-4],
    "muon": [1e-1, 5e-2, 1e-2, 5e-3, 1e-3],
    "muon_mvr": [1e-1, 5e-2, 1e-2, 5e-3, 1e-3],
    "muoneq-rc": [1e-1, 5e-2, 1e-2, 5e-3, 1e-3],
    "muoneq-r": [1e-1, 5e-2, 1e-2, 5e-3, 1e-3],
    "muoneq-c": [1e-1, 5e-2, 1e-2, 5e-3, 1e-3],
}


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if np is not None:
        np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, tuple):
        return [json_ready(v) for v in value]
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    return value


def none_or_int(value: str) -> Optional[int]:
    if value == "None":
        return None
    return int(value)


def build_log_path(output_path: Path) -> Path:
    return output_path.with_suffix(".log")


def setup_logger(log_path: Path, logger_name: str) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def close_logger(logger: Optional[logging.Logger]) -> None:
    if logger is None:
        return
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()


def log_info(logger: Optional[logging.Logger], message: str) -> None:
    if logger is not None:
        logger.info(message)
    else:
        print(message)


def progress_bar(iterable: Any, *, desc: str, disable: bool, leave: bool = False) -> Any:
    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, dynamic_ncols=True, leave=leave, disable=disable)


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_optimizer_order(
    requested_optimizers: Optional[List[str]],
    excluded_optimizers: List[str],
) -> List[str]:
    available = set(OPTIMIZER_ORDER)

    if requested_optimizers:
        selected: List[str] = []
        seen = set()
        for name in requested_optimizers:
            if name not in available:
                raise ValueError(f"Unsupported optimizer '{name}'. Available: {OPTIMIZER_ORDER}")
            if name not in seen:
                selected.append(name)
                seen.add(name)
    else:
        selected = list(OPTIMIZER_ORDER)

    for name in excluded_optimizers:
        if name not in available:
            raise ValueError(f"Unsupported optimizer '{name}'. Available: {OPTIMIZER_ORDER}")

    selected = [name for name in selected if name not in set(excluded_optimizers)]
    if not selected:
        raise ValueError("No optimizers selected after applying --optimizers / --exclude-optimizers.")
    return selected


def get_cifar10_loaders(
    data_dir: str,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> tuple[DataLoader, DataLoader]:
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

    trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    trainloader = DataLoader(trainset, shuffle=True, **loader_kwargs)
    testloader = DataLoader(testset, shuffle=False, **loader_kwargs)
    return trainloader, testloader


def save_initial_model_state(seed: int, device: torch.device, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / f"resnet18_initial_state_seed{seed}.pt"
    if state_path.exists():
        return state_path

    set_seed(seed)
    model = resnet.ResNet18(num_classes=10).to(device)
    cpu_state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    torch.save(cpu_state, state_path)
    return state_path


def create_model(device: torch.device, initial_state_path: Path) -> nn.Module:
    model = resnet.ResNet18(num_classes=10).to(device)
    initial_state = torch.load(initial_state_path, map_location="cpu")
    model.load_state_dict(initial_state)
    return model


def create_optimizer(name: str, model: nn.Module, config: Dict[str, Any]) -> torch.optim.Optimizer:
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), **config)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), **config)
    if name == "muon":
        return CifarMuon(model.parameters(), **config)
    if name == "muon_mvr":
        return CifarMuonMVR(model.parameters(), **config)
    if name in {"muoneq-rc", "muoneq-r", "muoneq-c"}:
        return CifarMuonEq(model.parameters(), **config)
    raise ValueError(f"Unsupported optimizer: {name}")


def evaluate_model(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            total_correct += outputs.argmax(dim=1).eq(targets).sum().item()
            total_samples += targets.size(0)

    return {
        "loss": total_loss / len(dataloader),
        "acc": 100.0 * total_correct / total_samples,
    }


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    progress_desc: str,
    show_progress: bool,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    iterator = progress_bar(dataloader, desc=progress_desc, disable=not show_progress, leave=False)
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

        total_loss += loss.item()
        total_correct += outputs.argmax(dim=1).eq(targets).sum().item()
        total_samples += targets.size(0)

        if hasattr(iterator, "set_postfix"):
            iterator.set_postfix(
                loss=f"{total_loss / batch_idx:.4f}",
                acc=f"{100.0 * total_correct / total_samples:.2f}",
            )

    if hasattr(iterator, "close"):
        iterator.close()

    return {
        "loss": total_loss / len(dataloader),
        "acc": 100.0 * total_correct / total_samples,
    }


def maybe_start_swanlab_run(
    enabled: bool,
    run_name: str,
    config: Dict[str, Any],
    logger: Optional[logging.Logger] = None,
) -> bool:
    if not enabled:
        return False
    if swanlab is None:
        if logger is not None:
            logger.warning("SwanLab requested but not installed. Continuing with local logs only.")
        return False
    try:
        swanlab.init(project="cifar10_resnet_optimizer_compare", experiment_name=run_name, config=json_ready(config))
        return True
    except Exception as exc:
        if logger is not None:
            logger.warning("Failed to initialize SwanLab. Continuing with local logs only. error=%s", exc)
        return False


def finish_swanlab_run(active: bool, logger: Optional[logging.Logger] = None) -> None:
    if active and swanlab is not None:
        try:
            swanlab.finish()
        except Exception as exc:
            if logger is not None:
                logger.warning("Failed to finalize SwanLab cleanly. error=%s", exc)


def maybe_start_wandb_run(
    enabled: bool,
    run_name: str,
    config: Dict[str, Any],
    project: str,
    entity: Optional[str],
    logger: Optional[logging.Logger] = None,
) -> Any:
    if not enabled:
        return None
    if wandb is None:
        if logger is not None:
            logger.warning("wandb requested but not installed. Continuing with local logs only.")
        return None
    try:
        return wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            config=json_ready(config),
            reinit=True,
        )
    except Exception as exc:
        if logger is not None:
            logger.warning("Failed to initialize wandb. Continuing with local logs only. error=%s", exc)
        return None


def finish_wandb_run(run: Any, logger: Optional[logging.Logger] = None) -> None:
    if run is not None:
        try:
            run.finish()
        except Exception as exc:
            if logger is not None:
                logger.warning("Failed to finalize wandb cleanly. error=%s", exc)


def safe_log_swanlab_metrics(
    active: bool,
    metrics: Dict[str, Any],
    logger: Optional[logging.Logger] = None,
) -> bool:
    if not active or swanlab is None:
        return active
    try:
        swanlab.log(metrics)
        return True
    except Exception as exc:
        if logger is not None:
            logger.warning("SwanLab logging failed. Disabling SwanLab for the rest of this run. error=%s", exc)
        finish_swanlab_run(True, logger=logger)
        return False


def safe_log_wandb_metrics(
    run: Any,
    metrics: Dict[str, Any],
    logger: Optional[logging.Logger] = None,
) -> Any:
    if run is None:
        return None
    try:
        run.log(metrics)
        return run
    except Exception as exc:
        if logger is not None:
            logger.warning("wandb logging failed. Disabling wandb for the rest of this run. error=%s", exc)
        finish_wandb_run(run, logger=logger)
        return None


def run_single_experiment(
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
    use_swanlab: bool,
    use_wandb: bool,
    wandb_project: str,
    wandb_entity: Optional[str],
    logger: Optional[logging.Logger],
    show_progress: bool,
) -> Dict[str, Any]:
    set_seed(run_seed)
    pin_memory = device.type == "cuda"

    model = create_model(device=device, initial_state_path=initial_state_path)
    trainloader, testloader = get_cifar10_loaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(optimizer_name, model, copy.deepcopy(optimizer_config))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

    run_name = f"{phase}_{optimizer_name}_run{run_id}_seed{run_seed}"
    swanlab_active = maybe_start_swanlab_run(
        enabled=use_swanlab,
        run_name=run_name,
        config={
            "phase": phase,
            "optimizer": optimizer_name,
            "run_id": run_id,
            "seed": run_seed,
            "epochs": epochs,
            "batch_size": batch_size,
            "optimizer_config": optimizer_config,
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
            "batch_size": batch_size,
            "optimizer_config": optimizer_config,
        },
        project=wandb_project,
        entity=wandb_entity,
        logger=logger,
    )

    best_acc = float("-inf")
    best_epoch = -1
    epoch_metrics: List[Dict[str, Any]] = []
    start_time = time.time()

    log_info(logger, f"\n[{phase}] optimizer={optimizer_name} run={run_id} seed={run_seed}")
    log_info(logger, f"config={json.dumps(json_ready(optimizer_config), ensure_ascii=False)}")

    for epoch in range(epochs):
        epoch_start = time.time()
        train_metrics = train_one_epoch(
            model,
            trainloader,
            optimizer,
            criterion,
            device,
            progress_desc=f"{phase}:{optimizer_name}:run{run_id} epoch {epoch + 1}/{epochs}",
            show_progress=show_progress,
        )
        eval_metrics = evaluate_model(model, testloader, criterion, device)
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

        swanlab_active = safe_log_swanlab_metrics(swanlab_active, epoch_result, logger=logger)
        wandb_run = safe_log_wandb_metrics(wandb_run, epoch_result, logger=logger)

        scheduler.step()

        log_info(
            logger,
            f"epoch {epoch + 1:03d}/{epochs:03d} "
            f"train_acc={train_metrics['acc']:.2f} "
            f"test_acc={eval_metrics['acc']:.2f} "
            f"best_acc={best_acc:.2f} "
            f"lr={current_lr:.6f} "
            f"time={epoch_result['epoch_time_sec']:.2f}s"
        )

    finish_swanlab_run(swanlab_active, logger=logger)
    finish_wandb_run(wandb_run, logger=logger)
    total_time = time.time() - start_time

    return {
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
    }


def run_lr_grid_search(
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
    use_swanlab: bool,
    use_wandb: bool,
    wandb_project: str,
    wandb_entity: Optional[str],
    logger: Optional[logging.Logger],
    show_progress: bool,
) -> Dict[str, Any]:
    log_info(logger, "\n=== LR grid search ===")
    grid_results: Dict[str, Any] = {
        "enabled": True,
        "epochs": epochs,
        "search_space": {name: copy.deepcopy(DEFAULT_LR_SEARCH_SPACE[name]) for name in optimizer_order},
        "best_lrs": {},
        "trials": [],
    }

    for optimizer_name in optimizer_order:
        best_lr = None
        best_acc = float("-inf")
        base_config = copy.deepcopy(base_optimizer_configs[optimizer_name])

        for trial_id, lr in enumerate(DEFAULT_LR_SEARCH_SPACE[optimizer_name]):
            trial_config = copy.deepcopy(base_config)
            trial_config["lr"] = lr
            result = run_single_experiment(
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
                use_swanlab=use_swanlab,
                use_wandb=use_wandb,
                wandb_project=wandb_project,
                wandb_entity=wandb_entity,
                logger=logger,
                show_progress=show_progress,
            )
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
                math.isclose(result["best_acc"], best_acc) and (best_lr is None or lr < best_lr)
            ):
                best_acc = result["best_acc"]
                best_lr = lr

        grid_results["best_lrs"][optimizer_name] = best_lr
        log_info(logger, f"best lr for {optimizer_name}: {best_lr}")

    return grid_results


def summarize_results(per_run_results: List[Dict[str, Any]], optimizer_order: List[str]) -> Dict[str, Dict[str, Any]]:
    def safe_mean(values: List[float]) -> Optional[float]:
        if not values:
            return None
        return float(statistics.mean(values))

    def safe_std(values: List[float]) -> Optional[float]:
        if not values:
            return None
        return float(statistics.pstdev(values))

    grouped: Dict[str, List[Dict[str, Any]]] = {name: [] for name in optimizer_order}
    for result in per_run_results:
        grouped[result["optimizer"]].append(result)

    summary: Dict[str, Dict[str, Any]] = {}
    for optimizer_name, results in grouped.items():
        best_accs = [entry["best_acc"] for entry in results]
        final_accs = [entry["final_acc"] for entry in results]
        wall_times = [entry["wall_time_sec"] for entry in results]
        summary[optimizer_name] = {
            "num_runs": len(results),
            "mean_best_acc": safe_mean(best_accs),
            "std_best_acc": safe_std(best_accs),
            "mean_final_acc": safe_mean(final_accs),
            "std_final_acc": safe_std(final_accs),
            "mean_wall_time_sec": safe_mean(wall_times),
            "best_single_run_acc": float(max(best_accs)) if best_accs else None,
            "selected_config": copy.deepcopy(results[0]["optimizer_config"]) if results else None,
        }
    return summary


def print_summary(
    summary: Dict[str, Dict[str, Any]],
    optimizer_order: List[str],
    logger: Optional[logging.Logger] = None,
) -> None:
    def fmt(value: Optional[float], width: int) -> str:
        return f"{value:>{width}.2f}" if value is not None else f"{'n/a':>{width}}"

    log_info(logger, "\n=== Final summary ===")
    header = f"{'optimizer':<12} {'mean_best':>10} {'std_best':>10} {'mean_final':>11} {'time(s)':>10}"
    log_info(logger, header)
    log_info(logger, "-" * len(header))
    for optimizer_name in optimizer_order:
        result = summary[optimizer_name]
        log_info(
            logger,
            f"{optimizer_name:<12} "
            f"{fmt(result['mean_best_acc'], 10)} "
            f"{fmt(result['std_best_acc'], 10)} "
            f"{fmt(result['mean_final_acc'], 11)} "
            f"{fmt(result['mean_wall_time_sec'], 10)}"
        )


def build_output_path(requested_path: Optional[str]) -> Path:
    if requested_path:
        return Path(requested_path).expanduser().resolve()
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return (THIS_DIR / "results" / f"compare_resnet_optimizers_{timestamp}.json").resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare SGD, AdamW, and Muon-family optimizers on CIFAR-10 ResNet18.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=str, default=str(THIS_DIR / "data"))
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--use-swanlab", action="store_true")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="cifar10_resnet_optimizer_compare")
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
    device = resolve_device(args.device)
    selected_optimizer_order = resolve_optimizer_order(args.optimizers, args.exclude_optimizers)
    output_path = build_output_path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = build_log_path(output_path)
    logger = setup_logger(log_path, f"cifar10.compare.{output_path.stem}")

    try:
        log_info(logger, "CIFAR-10 ResNet18 optimizer comparison")
        log_info(logger, f"device={device}")
        log_info(logger, f"optimizers={selected_optimizer_order}")
        log_info(logger, f"output_json={output_path}")
        log_info(logger, f"log_path={log_path}")
        if device.type == "cuda":
            device_index = device.index if device.index is not None else torch.cuda.current_device()
            log_info(logger, f"gpu={torch.cuda.get_device_name(device_index)}")

        base_initial_state_path = save_initial_model_state(
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
            lr_grid_search_results = run_lr_grid_search(
                optimizer_order=selected_optimizer_order,
                base_optimizer_configs=optimizer_configs,
                epochs=args.grid_search_epochs,
                base_seed=args.seed,
                initial_state_path=base_initial_state_path,
                data_dir=args.data_dir,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                device=device,
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
            initial_state_path = save_initial_model_state(
                seed=run_seed,
                device=device,
                output_dir=output_path.parent / "initial_states",
            )
            for optimizer_name in selected_optimizer_order:
                result = run_single_experiment(
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
                    use_swanlab=args.use_swanlab,
                    use_wandb=args.use_wandb,
                    wandb_project=args.wandb_project,
                    wandb_entity=args.wandb_entity,
                    logger=logger,
                    show_progress=not args.disable_tqdm,
                )
                per_run_results.append(result)

        summary = summarize_results(per_run_results, selected_optimizer_order)
        print_summary(summary, selected_optimizer_order, logger=logger)

        payload = {
            "config": {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "num_runs": args.num_runs,
                "seed": args.seed,
                "data_dir": args.data_dir,
                "num_workers": args.num_workers,
                "device": device,
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
            },
            "lr_grid_search": lr_grid_search_results,
            "optimizer_summaries": summary,
            "per_run_metrics": per_run_results,
        }

        output_path.write_text(json.dumps(json_ready(payload), indent=2))
        log_info(logger, f"\nSaved results to {output_path}")
        log_info(logger, f"Saved logs to {log_path}")
    finally:
        close_logger(logger)


if __name__ == "__main__":
    main()
