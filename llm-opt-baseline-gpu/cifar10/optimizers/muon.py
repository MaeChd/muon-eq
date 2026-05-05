from typing import Any, Dict, Iterable

import torch

from .common import CifarMuonBase, zeropower_via_newtonschulz5


class CifarMuon(CifarMuonBase):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 5e-3,
        momentum: float = 0.9,
        nesterov: bool = True,
        ns_steps: int = 5,
        wd: float = 0.0,
        adamw_betas: tuple[float, float] = (0.95, 0.95),
        adamw_eps: float = 1e-8,
    ) -> None:
        super().__init__(
            params,
            lr=lr,
            wd=wd,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
        )

    def _muon_update(
        self,
        param: torch.Tensor,
        grad_2d: torch.Tensor,
        group: Dict[str, Any],
        muon_group_state: Dict[str, Any],
    ) -> torch.Tensor:
        state = self.state[param]
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros_like(grad_2d)

        buf = state["momentum_buffer"]
        momentum = float(group["momentum"])
        buf.mul_(momentum).add_(grad_2d)
        update = grad_2d.add(buf, alpha=momentum) if group["nesterov"] else buf
        return zeropower_via_newtonschulz5(update, steps=int(group["ns_steps"]))
