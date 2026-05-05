import argparse
import copy
import inspect
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import wandb

import config
import distributed
from data.utils import DataReader, get_dataset
from models.utils import get_model
from optim.adafactor import Adafactor
from optim.ademamix import AdEMAMix
from optim.adamuon import AdaMuon
from optim.adopt import ADOPT
from optim.base import train
from optim.fismo import FISMO
from optim.lamb import Lamb
from optim.lion import Lion
from optim.mars import MARS
from optim.mousse import Mousse
from optim.muon import CombinedScheduler, DistributedMuon, Muon as MuonSplitLR
from optim.muon_kimi import Muon as Muon_Kimi
from optim.muoneq import MuonEq
from optim.muonplus import MuonPlus
from optim.prodigy import Prodigy
from optim.schedule import cos_inf_schedule, wsd_schedule
from optim.schedulefree import AdamWScheduleFree, SGDScheduleFree
from optim.scion import Scion, ScionLight, scion_partitions
from optim.sign import Signum
from optim.soap import SOAP
from optim.sophia import SophiaG


SUPPORTED_OPT_NAMES = (
    "sgd",
    "adamw",
    "soap",
    "adamuon",
    "mousse",
    "muon",
    "muon-splitlr",
    "fismo",
    "muonplus",
    "muoneq-rc",
    "muoneq-r",
    "muoneq-c",
    "d-muon",
    "ademamix",
    "lion",
    "sf-adamw",
    "sf-sgd",
    "signsgd",
    "signum",
    "prodigy",
    "sophiag",
    "adopt",
    "mars",
    "adafactor",
    "lamb",
    "scion",
    "scion-light",
    "muon-pytorch",
)


def get_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--config_format", default="base", choices=config.registered_formats()
    )

    args, rem_args = parser.parse_known_args()

    final_args = config.parse_args_with_format(
        format=args.config_format, base_parser=parser, args=rem_args, namespace=args
    )

    return final_args, parser


def main(args, parser):
    main_start_time = time.perf_counter()
    distributed_backend = distributed.make_backend_from_args(args)
    args = distributed_backend.get_adjusted_args_for_process(args)
    args.world_size = distributed_backend.get_world_size()

    if args.full_eval_at is None:
        args.full_eval_at = []

    # NOTE args.seed is offset per worker in get_adjusted_args_for_process
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if "cuda" in args.device:
        torch.cuda.set_device(torch.device(args.device))
    # torch.use_deterministic_algorithms(True)  # CUBLAS_WORKSPACE_CONFIG=:4096:8
    print("rank/world:", distributed_backend.rank, args.world_size, "device:", args.device)
    exp_name = get_exp_name(args, parser, distributed_backend)
    exp_dir = Path(args.results_base_folder) / exp_name
    if distributed_backend.is_master_process() and args.wandb:
        wandb.init(
            project=args.wandb_project,
            name=exp_name,
            config=vars(args),
            entity=args.wandb_entity,
        )
        # wandb to wandb 
        wandb.define_metric("iter")
        wandb.define_metric("train/*", step_metric="iter")
        wandb.define_metric("val/*", step_metric="iter")
        wandb.define_metric("lr", step_metric="iter")

    print(f"Starting Experiment: {exp_name}")
    print(f"Experiment Directory: {exp_dir}")
    print(f"Config:\n{vars(args)}\n")

    print(f"Loading dataset: '{args.dataset}'")
    datareaders = get_data_readers(args)

    model = get_model(args).to(
        args.device
    )  # todo: take care of initializing the model if args.use_pretrained != 'none'

    print(f"\nModel:\n{model}")

    # Print the number of trainable parameters.
    # Print the number of trainable parameters in millions.
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"\nParameters: {total_params:.2f}M total, {trainable_params:.2f}M trainable")
    model = distributed_backend.transform_model(model)
    print("param dtype:", next(distributed_backend.get_raw_model(model).parameters()).dtype)

    group_specs = distributed_backend.get_raw_model(model).get_parameter_group_specs(
        config=args
    )
    param_name_mapping = {p_name: p for p_name, p in model.named_parameters()}
    optimized_params_cnt = 0
    for g in group_specs:
        params = []
        for p_name in g["params"]:
            translated_p_names = (
                distributed_backend.translate_model_parameter_name_for_node(p_name)
            )
            params += [param_name_mapping[p_name] for p_name in translated_p_names]
        g["params"] = params
        optimized_params_cnt += sum([p.numel() for p in g["params"]])
    params_cnt = distributed_backend.get_raw_model(model).get_num_params()
    nonemb_param_cnt = (
        params_cnt
        - distributed_backend.get_raw_model(model).lm_head.weight.numel()
        - distributed_backend.get_raw_model(model).transformer.wte.weight.numel()
    )
    print("number of parameters: %.2fM" % (params_cnt / 1e6,))
    print("number of optimized parameters: %.2fM" % (optimized_params_cnt / 1e6,))
    print("number of non-embedding parameters: %.2fM" % (nonemb_param_cnt / 1e6,))
    if args.wandb and distributed_backend.is_master_process():
        wandb.log(
            {
                "parameters": params_cnt,
                "optimized_parameters": optimized_params_cnt,
                "non_embedding_parameters": nonemb_param_cnt,
            }
        )

    args.world_size = distributed_backend.get_world_size()

    if args.opt == "sgd":
        opt = torch.optim.SGD(
            group_specs,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov,
        )
    elif args.opt == "adamw":
        device_type = "cuda" if "cuda" in args.device else "cpu"
        use_fused = (device_type == "cuda") and (
            "fused" in inspect.signature(torch.optim.AdamW).parameters
        )
        print(f"using fused AdamW: {use_fused}")
        extra_args = dict(fused=True) if use_fused else dict()
        opt = torch.optim.AdamW(
            group_specs,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
            **extra_args,
        )
    elif args.opt == "soap":
        opt = SOAP(
            group_specs,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            shampoo_beta=args.shampoo_beta,
            weight_decay=args.weight_decay,
            precondition_frequency=args.precondition_frequency,
            max_precond_dim=args.max_precond_dim,
            merge_dims=args.merge_dims,
            precondition_1d=args.precondition_1d,
            normalize_grads=args.normalize_grads,
            data_format=args.soap_data_format,
            correct_bias=args.correct_bias,
        )

    elif args.opt == "adamuon":
        muon_params = [
            p
            for name, p in model.named_parameters()
            if p.ndim >= 2
            and "wte" not in name
            and "wpe" not in name
            and "lm_head" not in name
            and p.requires_grad
        ]
        adamw_params = [
            p
            for name, p in model.named_parameters()
            if not (
                p.ndim >= 2
                and "wte" not in name
                and "wpe" not in name
                and "lm_head" not in name
                and p.requires_grad
            )
        ]

        opt = AdaMuon(
            params=muon_params,
            lr=args.lr,
            weight_decay=args.weight_decay,
            momentum=args.momentum,
            nesterov=args.nesterov,
            ns_steps=args.muon_ns_steps,
            eps=args.adamuon_eps,
            rank=distributed_backend.rank,
            world_size=args.world_size,
            adamw_params=adamw_params,
            adamw_betas=(args.beta1, args.beta2),
        )
    elif args.opt == "mousse":
        muon_params = [
            p
            for name, p in model.named_parameters()
            if p.ndim >= 2
            and "wte" not in name
            and "wpe" not in name
            and "lm_head" not in name
            and p.requires_grad
        ]
        adamw_params = [
            p
            for name, p in model.named_parameters()
            if not (
                p.ndim >= 2
                and "wte" not in name
                and "wpe" not in name
                and "lm_head" not in name
                and p.requires_grad
            )
        ]

        opt = Mousse(
            lr=args.lr,
            wd=args.weight_decay,
            momentum=args.momentum,
            nesterov=args.nesterov,
            ns_steps=args.muon_ns_steps,
            zeropower_mode=args.zeropower_mode,
            adjust_lr=args.mousse_adjust_lr,
            flatten=args.mousse_flatten,
            shampoo_epsilon=args.mousse_shampoo_epsilon,
            shampoo_beta=args.mousse_shampoo_beta,
            shampoo_update_freq=args.mousse_shampoo_update_freq,
            shampoo_alpha=args.mousse_shampoo_alpha,
            lr_correction=args.mousse_lr_correction,
            apply_norm=args.mousse_apply_norm,
            use_lorr=args.mousse_use_lorr,
            muon_params=muon_params,
            adamw_betas=(args.beta1, args.beta2),
            adamw_params=adamw_params,
            adamw_eps=1e-8,
        )
    elif args.opt in {"muon", "muon-splitlr"}:
        muon_params = [
            p
            for name, p in model.named_parameters()
            if p.ndim >= 2
            and "wte" not in name
            and "wpe" not in name
            and "lm_head" not in name
            and p.requires_grad
        ]
        adamw_params = [
            p
            for name, p in model.named_parameters()
            if not (
                p.ndim >= 2
                and "wte" not in name
                and "wpe" not in name
                and "lm_head" not in name
                and p.requires_grad
            )
        ]

        if args.opt == "muon-splitlr":
            opt = MuonSplitLR(
                muon_params=muon_params,
                lr=args.muon_lr_factor,
                momentum=args.momentum,
                nesterov=args.nesterov,
                ns_steps=args.muon_ns_steps,
                adamw_params=adamw_params,
                adamw_lr=args.lr,
                adamw_betas=(args.beta1, args.beta2),
                adamw_eps=1e-8,
                adamw_wd=args.weight_decay,
            )
        else:
            opt = Muon_Kimi(
                lr=args.lr,
                wd=args.weight_decay,
                momentum=args.momentum,
                nesterov=args.nesterov,
                ns_steps=args.muon_ns_steps,
                zeropower_mode=args.zeropower_mode,
                muon_params=muon_params,
                adamw_betas=(args.beta1, args.beta2),
                adamw_params=adamw_params,
            )
    elif args.opt in {"fismo", "muonplus"}:
        muon_params = [
            p
            for name, p in model.named_parameters()
            if p.ndim >= 2 and "wte" not in name and "wpe" not in name and "lm_head" not in name and p.requires_grad
        ]
        adamw_params = [
            p
            for name, p in model.named_parameters()
            if not (
                p.ndim >= 2 and "wte" not in name and "wpe" not in name and "lm_head" not in name and p.requires_grad
            )
        ]

        if args.opt == "fismo":
            opt = FISMO(
                lr=args.lr,
                wd=args.weight_decay,
                momentum=args.momentum,
                ema_beta=args.fismo_ema_beta,
                damping=args.fismo_damping,
                ns_steps=args.muon_ns_steps,
                zeropower_mode=args.zeropower_mode,
                matrix_eps=args.fismo_matrix_eps,
                muon_params=muon_params,
                adamw_betas=(args.beta1, args.beta2),
                adamw_params=adamw_params,
            )
        else:
            opt = MuonPlus(
                lr=args.lr,
                wd=args.weight_decay,
                momentum=args.momentum,
                nesterov=args.nesterov,
                ns_steps=args.muon_ns_steps,
                normalize=args.muonplus_normalize,
                eps=args.muonplus_eps,
                muon_params=muon_params,
                adamw_betas=(args.beta1, args.beta2),
                adamw_params=adamw_params,
                zeropower_mode=args.zeropower_mode,
            )
    elif args.opt in {"muoneq-rc", "muoneq-r", "muoneq-c"}:
        muon_params = [
            p
            for name, p in model.named_parameters()
            if p.ndim >= 2 and "wte" not in name and "wpe" not in name and "lm_head" not in name and p.requires_grad
        ]
        adamw_params = [
            p
            for name, p in model.named_parameters()
            if not (
                p.ndim >= 2 and "wte" not in name and "wpe" not in name and "lm_head" not in name and p.requires_grad
            )
        ]

        if args.use_muonplus_update:
            raise ValueError(
                "--use_muonplus_update is no longer supported for MuonEq variants. "
                "Use --opt muonplus instead."
            )

        normalize_mode = {
            "muoneq-rc": "rowcol",
            "muoneq-r": "row",
            "muoneq-c": "col",
        }[args.opt]
        if args.muoneq_phase is not None and normalize_mode != "rowcol":
            raise ValueError("--muoneq_phase only applies to --opt muoneq-rc.")

        opt = MuonEq(
            lr=args.lr,
            wd=args.weight_decay,
            momentum=args.momentum,
            nesterov=args.nesterov,
            ns_steps=args.muon_ns_steps,
            muon_params=muon_params,
            adamw_betas=(args.beta1, args.beta2),
            adamw_params=adamw_params,
            normalize_mode=normalize_mode,
            phase=args.muoneq_phase if normalize_mode == "rowcol" else None,
            zeropower_mode=args.zeropower_mode,
        )
    elif args.opt == "d-muon":
        opt = DistributedMuon(
            group_specs,
            lr=args.lr,
            momentum=args.momentum,
            nesterov=args.nesterov,
            ns_steps=args.muon_ns_steps,
            adamw_betas=(args.beta1, args.beta2),
            adamw_eps=1e-8,
            weight_decay=args.weight_decay,
        )
    elif args.opt == "ademamix":
        opt = AdEMAMix(
            group_specs,
            lr=args.lr,
            betas=(args.beta1, args.beta2, args.adema_beta3),
            alpha=args.adema_alpha,
            beta3_warmup=args.adema_beta3_warmup,
            alpha_warmup=args.adema_alpha_warmup,
            weight_decay=args.weight_decay,
        )
    elif args.opt == "lion":
        opt = Lion(
            group_specs,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
        )
    elif args.opt == "sf-adamw":
        opt = AdamWScheduleFree(
            group_specs,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps,
            r=args.schedulefree_r,
            weight_lr_power=args.weight_lr_power,
        )  # without foreach argument
    elif args.opt == "sf-sgd":
        opt = SGDScheduleFree(
            group_specs,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps,
            r=args.schedulefree_r,
            weight_lr_power=args.weight_lr_power,
        )  # without foreach argument
    elif args.opt == "signsgd":
        opt = Signum(
            group_specs,
            lr=args.lr,
            momentum=0.0,  # always use zero momentum because its signSGD
            dampening=args.dampening,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov,
            sign_update=True,
        )
    elif args.opt == "signum":
        opt = Signum(
            group_specs,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            dampening=args.dampening,
            nesterov=args.nesterov,
            sign_update=True,
        )
    elif args.opt == "prodigy":
        opt = Prodigy(
            group_specs,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            beta3=args.prodigy_beta3,
            weight_decay=args.weight_decay,
            decouple=args.prodigy_decouple,
            use_bias_correction=args.prodigy_use_bias_correction,
            safeguard_warmup=args.prodigy_safeguard_warmup,
            fsdp_in_use=args.prodigy_fsdp_in_use,
        )
    elif args.opt == "sophiag":
        opt = SophiaG(
            group_specs,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
            rho=args.sophia_rho,
        )
    elif args.opt == "adopt":
        opt = ADOPT(
            group_specs,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            eps=args.adopt_eps,  # 1e-6
            weight_decay=args.weight_decay,
            decouple=args.adopt_decouple,
        )
    elif args.opt == "mars":
        opt = MARS(
            group_specs,
            lr=args.mars_lr,
            betas=(args.mars_beta1, args.mars_beta2),
            weight_decay=args.weight_decay,
            amsgrad=False,
            gamma=args.mars_vr_gamma,
            is_approx=args.mars_is_approx,
            mars_type=args.mars_type,
            optimize_1d=False,  # we set in order to optimize 1D parameters with AdamW
            lr_1d=args.lr,  # AdamW's lr when optimize_1d=False
            betas_1d=(args.beta1, args.beta2),  # AdamW's betas when optimize_1d=False
            weight_decay_1d=0.1,  # AdamW's weight decay
        )
    elif args.opt == "adafactor":
        opt = Adafactor(
            group_specs,
            lr=args.lr,
            decay_rate=args.adafactor_decay_rate,
            beta1=args.beta1,
            clip_threshold=1.0,
            weight_decay=args.weight_decay,
        )
    elif args.opt == "lamb":
        opt = Lamb(
            group_specs,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
            adam=False,
            bias_correction=args.lamb_use_bias_correction,
        )
    elif args.opt == "scion":
        scion_param_groups = scion_partitions(group_specs, model, args)
        scion_params_cnt = sum(
            p.numel() for group in scion_param_groups for p in group["params"]
        )
        print(f"Optimized parameters: {scion_params_cnt}")
        opt = Scion(
            scion_param_groups,
            lr=args.lr,
            momentum=args.momentum,
        )
    elif args.opt == "scion-light":
        scion_param_groups = scion_partitions(group_specs, model, args)
        scion_params_cnt = sum(
            p.numel() for group in scion_param_groups for p in group["params"]
        )
        print(f"Optimized parameters: {scion_params_cnt}")
        opt = ScionLight(
            scion_param_groups,
            lr=args.lr,
            momentum=args.momentum,
        )
    elif args.opt == "muon-pytorch":
        opt = torch.optim.Muon(
            group_specs,
            lr=args.lr,
            momentum=args.momentum,
            nesterov=args.nesterov,
            ns_steps=args.muon_ns_steps,
            ns_coefficients=(
                3.4445,
                -4.775,
                2.0315,
            ),  # someone might try to change it later
            eps=1e-7,  # muon pytorch uses smaller eps
            adjust_lr_fn=None,  # to make the orthogonalized update have a consistent RMS across rectangular matrices
        )
    else:
        raise ValueError(
            f"Unsupported optimizer: {args.opt!r}. "
            f"Supported values: {', '.join(SUPPORTED_OPT_NAMES)}"
        )
    print(f"\nOptimizer:\n{opt}")

    if args.scheduler != "none":
        assert (
            args.warmup_steps < args.iterations
        ), "Warmup steps must be < iterations."  # from schedules-and-scaling
        if args.scheduler in ["cos", "linear"]:
            # initial lr is args.lr / div_factor
            # final lr is initial_lr/final_div_factor = args.lr / div_factor / final_div_factor
            scheduler = (
                torch.optim.lr_scheduler.OneCycleLR(
                    optimizer=opt,
                    max_lr=[
                        group.get("lr", args.lr) for group in group_specs
                    ],  # it was args.lr
                    total_steps=args.iterations,
                    pct_start=args.warmup_steps
                    / args.iterations,  # it was args.warmup_percent
                    anneal_strategy=args.scheduler,
                    cycle_momentum=False,
                    div_factor=1e2,
                    final_div_factor=args.final_div_factor,
                )
                if args.opt != "muon"
                else CombinedScheduler(opt, args)
            )
        elif args.scheduler == "cos_inf":
            lambda_schedule = cos_inf_schedule(
                n_iterations=args.iterations,
                n_warmup=args.warmup_steps,
                n_inf=args.cos_inf_steps,
                div_factor=1e2,
                final_div_factor=0.1,
            )
            scheduler = (
                torch.optim.lr_scheduler.LambdaLR(opt, lambda_schedule)
                if args.opt != "muon"
                else CombinedScheduler(opt, args)
            )
        elif args.scheduler == "wsd":
            lambda_schedule = wsd_schedule(
                n_iterations=args.iterations,
                n_warmup=args.warmup_steps,
                fract_decay=args.wsd_fract_decay,
                init_div_factor=1e2,
                final_lr_factor=args.wsd_final_lr_scale,  # should be 0 here
                decay_type=args.decay_type,
            )
            scheduler = (
                torch.optim.lr_scheduler.LambdaLR(opt, lambda_schedule)
                if args.opt != "muon"
                else CombinedScheduler(opt, args)
            )
        else:
            raise NotImplementedError(f"Unknown scheduler type: {args.scheduler}.")
    else:
        scheduler = None

    if (exp_dir / "ckpts" / "latest" / "main.pt").exists():
        if not args.auto_resume:
            raise ValueError(
                f"The experiment dir {exp_dir} already exists. "
                + "To resume training, set auto_resume=True. "
                + "Otherwise, specify a different experiment name. "
            )
        else:
            # Auto resume overwrites resume_from
            args.resume_from = str(exp_dir / "ckpts" / "latest")
    elif distributed_backend.is_master_process():
        exp_dir.mkdir(parents=True, exist_ok=True)

    stats = train(
        model=model,
        opt=opt,
        datareaders=datareaders,
        scheduler=scheduler,
        exp_dir=exp_dir,
        distributed_backend=distributed_backend,
        cfg=args,
    )

    wall_clock_runtime_s = time.perf_counter() - main_start_time
    stats["wall_clock_runtime_s"] = wall_clock_runtime_s
    stats["wall_clock_runtime_hms"] = (
        f"{int(wall_clock_runtime_s // 3600)}:"
        f"{int((wall_clock_runtime_s % 3600) // 60):02d}:"
        f"{int(wall_clock_runtime_s % 60):02d}"
        if wall_clock_runtime_s >= 3600
        else f"{int(wall_clock_runtime_s // 60):02d}:{int(wall_clock_runtime_s % 60):02d}"
    )

    stats["args"] = vars(args)
    if distributed_backend.is_master_process():
        print(
            "Run finished: "
            f"wall_clock_runtime={stats['wall_clock_runtime_hms']} "
            f"train_runtime={stats.get('train_runtime_hms', '?')}"
        )
        if args.wandb and wandb.run is not None:
            wandb.run.summary["wall_clock_runtime_s"] = wall_clock_runtime_s
            wandb.run.summary["wall_clock_runtime_hms"] = stats["wall_clock_runtime_hms"]
        with open(exp_dir / "summary.json", "w") as fs:
            json.dump(stats, fs)
    distributed_backend.finalize()


def get_data_readers(args, verbose=True):
    data_srcs = get_dataset(args)
    train_reader = DataReader(
        data_src=data_srcs["train"],
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        seed=args.data_seed,
        with_replacement=False,
        auto_shard=True,
        keep_in_ram=args.data_in_ram,
    )
    val_reader = DataReader(
        data_src=data_srcs["val"],
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        seed=args.data_seed,
        with_replacement=False,
        auto_shard=False,  # NOTE Identical Per Rank
        keep_in_ram=args.data_in_ram,
    )

    if verbose:
        print(f"Num training tokens: {train_reader.num_tokens}")
        print(f"Num validation tokens: {val_reader.num_tokens}")

    return {
        "train": train_reader,
        "val": val_reader,
    }


def get_exp_name(
    args,
    parser,
    distributed_backend,
    key_args=["model", "dataset", "opt"],
    ignore_args=[
        "eval_interval",
        "full_eval_at",
        "distributed_backend",
        "latest_ckpt_interval",
        "permanent_ckpt_interval",
        "wandb",
        "wandb_project",
        "wandb_entity",
        "batch_size",
        "acc_steps",
        "results_base_folder",
        "run_prefix",
        "wandb_run_prefix",
        "seed",
        "device",
        "adema_beta3_warmup",
        "adema_alpha_warmup",
        "plot_router_logits",
        "weight_average",
        # "wa_interval",
        # "wa_horizon",
        "wa_dtype",
        "wa_use_temp_dir",
        "wa_sweep_horizon",
        # "max_num_wa_sweeps",
        "exponential_weight_average",
        # "ewa_interval",
        # "ewa_decay",
        # "ewa_after_warmup",
        "moe",
    ],
):
    if getattr(args, "experiment_name", None):
        return args.experiment_name

    # Get the default values
    defaults = vars(parser.parse_args([]))

    # rank = distributed_backend.rank # decided to remove rank from the exp name

    # Generate the prefix with key arguments
    prefix_parts = []
    for key in key_args:
        if hasattr(args, key):
            value = getattr(args, key)
            if key == "model":
                if getattr(args, "moe", False):
                    value = f"moe_{value}"
                if getattr(args, "weight_average", False):
                    value = f"{value}_WA"
                if getattr(args, "exponential_weight_average", False):
                    value = f"{value}_EWA"
            prefix_parts.append(f"{key}-{value}")

    prefix = "_".join(prefix_parts)
    prefix = f"{args.batch_size}x{args.acc_steps}_" + prefix  # rank={rank}

    # Generate the rest of the string with non-default arguments
    non_default_parts = []
    for key, value in vars(args).items():
        if key in ignore_args:
            continue
        if key not in defaults:
            print(f"Warning: {key} not in defaults")
            continue
        if key not in key_args and value != defaults[key]:
            non_default_parts.append(f"{key}-{value}")

    non_default_string = "_".join(non_default_parts)

    if args.run_prefix is not None:
        prefix = args.run_prefix + "_" + prefix

    # Combine prefix and non-default string
    if non_default_string:
        return f"{prefix}__{non_default_string}"
    else:
        return prefix


if __name__ == "__main__":
    args, parser = get_args()
    main(args, parser)
