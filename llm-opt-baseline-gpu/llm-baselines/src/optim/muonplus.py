import math

import torch

from .muon_fair_utils import (
    adamw_backup_step,
    apply_zeropower_backend,
    canonicalize_zeropower_mode,
    norm_dir,
)


class MuonPlus(torch.optim.Optimizer):
    """
    Muon+ with the same matrix/scalar split used by muon-style optimizers in this repo.

    `nesterov` is accepted for CLI compatibility with existing compare scripts but is
    ignored by this implementation. The update follows the legacy combined
    implementation: an EMA-style momentum buffer is orthogonalized and then
    normalized before applying the matrix update.
    """

    def __init__(
        self,
        lr=1e-3,
        wd=0.1,
        muon_params=None,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        zeropower_mode="native",
        normalize="col_row",
        eps=1e-8,
        adamw_params=None,
        adamw_betas=(0.95, 0.95),
        adamw_eps=1e-8,
    ):
        if muon_params is None:
            raise ValueError("muon_params must be provided.")

        zeropower_mode = canonicalize_zeropower_mode(zeropower_mode)
        defaults = dict(
            lr=lr,
            wd=wd,
            momentum=momentum,
            ns_steps=ns_steps,
            nesterov=nesterov,
            zeropower_mode=zeropower_mode,
            normalize=normalize,
            eps=eps,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
        )

        muon_params = list(muon_params)
        adamw_params = list(adamw_params) if adamw_params is not None else []
        params = muon_params + adamw_params
        super().__init__(params, defaults)

        for p in muon_params:
            assert p.ndim >= 2, p.ndim
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
            muon_params = [
                p for p in group["params"] if self.state[p].get("use_muon", False)
            ]
            for p in muon_params:
                g = p.grad
                if g is None:
                    continue
                if g.ndim > 2:
                    g = g.view(g.size(0), -1)

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g, dtype=torch.float32)

                buf = state["momentum_buffer"]
                g_work = g.detach().float()
                buf.mul_(group["momentum"]).add_(g_work, alpha=1.0 - group["momentum"])

                u = apply_zeropower_backend(
                    buf.to(dtype=g.dtype),
                    steps=group["ns_steps"],
                    zeropower_mode=group["zeropower_mode"],
                ).float()
                o = norm_dir(u, direction=group["normalize"], eps=group["eps"])

                m, n = o.shape
                adjusted_lr = math.sqrt(m / n)
                p.data.mul_(1 - group["lr"] * group["wd"])
                p.data.add_(o.to(dtype=p.dtype), alpha=-group["lr"] * adjusted_lr)

            adamw_params = [
                p for p in group["params"] if not self.state[p].get("use_muon", False)
            ]
            for p in adamw_params:
                adamw_backup_step(group, self.state[p], p)

        return loss
