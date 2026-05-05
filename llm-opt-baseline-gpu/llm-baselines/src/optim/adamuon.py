import torch
import torch.distributed as dist
from torch import Tensor

from .muon_fair_utils import adamw_backup_step


def zeropower_via_newtonschulz5(G: Tensor, steps: int) -> Tensor:
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


class AdaMuon(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=0.02,
        weight_decay=0.01,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        eps=1e-8,
        rank=None,
        world_size=None,
        adamw_params=None,
        adamw_betas=(0.95, 0.95),
        adamw_eps=1e-8,
    ):
        if rank is None:
            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        if world_size is None:
            world_size = (
                dist.get_world_size()
                if dist.is_available() and dist.is_initialized()
                else 1
            )
        self.rank = rank
        self.world_size = world_size
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            eps=eps,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
        )
        params: list[Tensor] = [*params]
        adamw_params = list(adamw_params) if adamw_params is not None else []
        param_groups = []
        if params:
            param_groups.append(dict(params=params, use_muon=True))
        if adamw_params:
            param_groups.append(dict(params=adamw_params, use_muon=False))
        super().__init__(param_groups, defaults)
        for p in params:
            self.state[p]["use_muon"] = True
        for p in adamw_params:
            self.state[p]["use_muon"] = False

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if not group.get("use_muon", False):
                for p in group["params"]:
                    adamw_backup_step(group, self.state[p], p)
                continue

            params: list[Tensor] = group["params"]
            for p in params:
                g = p.grad
                if g is None:
                    continue

                g_work = g.detach().float()
                if g_work.ndim > 2:
                    g_work = g_work.view(g_work.size(0), -1)

                state = self.state[p]
                if ("momentum_buffer" not in state) or (
                    state["momentum_buffer"].shape != g_work.shape
                ):
                    state["momentum_buffer"] = torch.zeros_like(
                        g_work, dtype=torch.float32
                    )

                buf: Tensor = state["momentum_buffer"]
                buf.mul_(group["momentum"]).add_(g_work)

                g_muon = (
                    g_work.add(buf, alpha=group["momentum"])
                    if group["nesterov"]
                    else buf
                )

                g_muon = zeropower_via_newtonschulz5(
                    torch.sign(g_muon).to(dtype=p.dtype),
                    steps=group["ns_steps"],
                ).float()

                if ("v_buffer" not in state) or (
                    state["v_buffer"].shape != g_muon.shape
                ):
                    state["v_buffer"] = torch.zeros_like(g_muon, dtype=torch.float32)
                v = state["v_buffer"]

                v.mul_(group["momentum"]).addcmul_(
                    g_muon, g_muon, value=1 - group["momentum"]
                )
                g_muon = g_muon.div(v.sqrt().add(group["eps"]))

                scale = 0.2 * (min(p.shape) * max(p.shape)) ** 0.5 / (
                    g_muon.norm() + group["eps"]
                )
                g_muon.mul_(scale)

                p.data.mul_(1 - group["lr"] * group["weight_decay"])
                p.data.add_(g_muon.view_as(p).to(dtype=p.dtype), alpha=-group["lr"])

        return loss
