from typing import Any, Dict, Iterable

import torch


def zeropower_via_newtonschulz5(grad_2d: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    if grad_2d.ndim != 2:
        raise ValueError(f"Muon expects a 2D matrix, got {grad_2d.ndim}D")

    a, b, c = (3.4445, -4.7750, 2.0315)
    work_dtype = torch.bfloat16 if grad_2d.is_cuda else torch.float32
    x = grad_2d.to(work_dtype)
    x = x / (x.norm() + eps)

    transposed = grad_2d.size(0) > grad_2d.size(1)
    if transposed:
        x = x.transpose(0, 1)

    for _ in range(steps):
        a_mat = x @ x.transpose(0, 1)
        b_mat = b * a_mat + c * a_mat @ a_mat
        x = a * x + b_mat @ x

    if transposed:
        x = x.transpose(0, 1)

    return x.to(grad_2d.dtype)


def flatten_muon_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim < 2:
        raise ValueError(f"Muon path only supports tensors with ndim >= 2, got {tensor.ndim}")
    if tensor.ndim == 2:
        return tensor
    return tensor.reshape(tensor.shape[0], -1)


class CifarMuonBase(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        *,
        lr: float,
        wd: float,
        adamw_betas: tuple[float, float],
        adamw_eps: float,
        **extra_defaults: Any,
    ) -> None:
        params = list(params)
        defaults = {
            "lr": lr,
            "wd": wd,
            "adamw_betas": adamw_betas,
            "adamw_eps": adamw_eps,
            **extra_defaults,
        }
        super().__init__(params, defaults)

        for group in self.param_groups:
            for param in group["params"]:
                self.state[param]["use_muon"] = param.ndim >= 2

    def _begin_muon_group(self, group: Dict[str, Any]) -> Dict[str, Any]:
        return {}

    def _muon_update(
        self,
        param: torch.Tensor,
        grad_2d: torch.Tensor,
        group: Dict[str, Any],
        muon_group_state: Dict[str, Any],
    ) -> torch.Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def _adamw_backup_step(self, param: torch.Tensor, grad: torch.Tensor, group: Dict[str, Any]) -> None:
        state = self.state[param]
        if "step" not in state:
            state["step"] = 0
            state["moment1"] = torch.zeros_like(grad)
            state["moment2"] = torch.zeros_like(grad)

        state["step"] += 1
        beta1, beta2 = group["adamw_betas"]
        eps = group["adamw_eps"]
        lr = group["lr"]
        weight_decay = group["wd"]

        buf1 = state["moment1"]
        buf2 = state["moment2"]
        buf1.lerp_(grad, 1 - beta1)
        buf2.lerp_(grad.square(), 1 - beta2)

        update = buf1 / (eps + buf2.sqrt())
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
            lr = group["lr"]
            wd = group["wd"]
            muon_group_state = self._begin_muon_group(group)
            for param in group["params"]:
                grad = param.grad
                if grad is None:
                    continue

                if self.state[param]["use_muon"]:
                    grad_2d = flatten_muon_tensor(grad)
                    update_2d = self._muon_update(param, grad_2d, group, muon_group_state)
                    param.data.mul_(1 - lr * wd)
                    param.data.add_(update_2d.view_as(param), alpha=-lr)
                else:
                    self._adamw_backup_step(param, grad, group)

        return loss
