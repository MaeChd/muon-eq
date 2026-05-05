from typing import Dict, Iterable

import torch

from .common import flatten_muon_tensor, zeropower_via_newtonschulz5


class CifarMuonMVR(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 5e-3,
        momentum: float = 0.9,
        gamma: float = 0.1,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
        adamw_betas: tuple[float, float] = (0.95, 0.99),
        eps: float = 1e-8,
    ) -> None:
        params = list(params)
        defaults = {
            "lr": lr,
            "momentum": momentum,
            "gamma": gamma,
            "ns_steps": ns_steps,
            "weight_decay": weight_decay,
            "adamw_betas": adamw_betas,
            "eps": eps,
        }
        super().__init__(params, defaults)

        for group in self.param_groups:
            for param in group["params"]:
                self.state[param]["use_muon"] = param.ndim >= 2

    @torch.no_grad()
    def update_last_grad(self) -> None:
        for group in self.param_groups:
            for param in group["params"]:
                if not self.state[param]["use_muon"]:
                    continue
                if param.grad is None:
                    continue

                state = self.state[param]
                grad_2d = flatten_muon_tensor(param.grad)
                if "last_grad" not in state:
                    state["last_grad"] = torch.zeros_like(grad_2d)
                state["last_grad"].copy_(grad_2d)

    @torch.no_grad()
    def _adamw_backup_step(self, param: torch.Tensor, grad: torch.Tensor, group: Dict[str, object]) -> None:
        state = self.state[param]
        if "step" not in state:
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(grad)
            state["exp_avg_sq"] = torch.zeros_like(grad)

        state["step"] += 1
        beta1, beta2 = group["adamw_betas"]
        eps = float(group["eps"])
        lr = float(group["lr"])
        weight_decay = float(group["weight_decay"])

        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]
        exp_avg.lerp_(grad, 1 - beta1)
        exp_avg_sq.lerp_(grad.square(), 1 - beta2)

        update = exp_avg / (eps + exp_avg_sq.sqrt())
        bias_correction1 = 1 - beta1 ** state["step"]
        bias_correction2 = 1 - beta2 ** state["step"]
        step_scale = bias_correction1 / (bias_correction2 ** 0.5)

        param.data.mul_(1 - lr * weight_decay)
        param.data.add_(update, alpha=-lr / step_scale)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = float(group["lr"])
            momentum = float(group["momentum"])
            gamma = float(group["gamma"])
            weight_decay = float(group["weight_decay"])
            ns_steps = int(group["ns_steps"])
            denom = max(1e-8, 1.0 - momentum)

            for param in group["params"]:
                grad = param.grad
                if grad is None:
                    continue

                if self.state[param]["use_muon"]:
                    grad_2d = flatten_muon_tensor(grad)
                    state = self.state[param]
                    if "exp_avg" not in state:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(grad_2d)
                        state["last_grad"] = torch.zeros_like(grad_2d)

                    state["step"] += 1
                    last_grad = state["last_grad"]
                    exp_avg = state["exp_avg"]

                    corrected_grad = (grad_2d - last_grad).mul(gamma * (momentum / denom)).add_(grad_2d)
                    grad_norm = corrected_grad.norm()
                    if grad_norm > 1.0:
                        corrected_grad = corrected_grad / grad_norm

                    exp_avg.mul_(momentum).add_(corrected_grad, alpha=1 - momentum)
                    update = zeropower_via_newtonschulz5(exp_avg.mul(1.0 / denom), steps=ns_steps)

                    param.data.mul_(1 - lr * weight_decay)
                    param.data.add_(update.view_as(param), alpha=-lr)
                else:
                    self._adamw_backup_step(param, grad, group)

        return loss
