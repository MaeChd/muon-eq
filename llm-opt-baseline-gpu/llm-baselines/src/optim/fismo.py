import torch

from .muon_fair_utils import (
    adamw_backup_step,
    apply_zeropower_backend,
    canonicalize_zeropower_mode,
    matrix_inv_sqrt,
    matrix_inverse,
    sym_trace_normalize,
    trace,
)


class FISMO(torch.optim.Optimizer):
    """
    FISMO with a Muon-style matrix/scalar parameter split.
    """

    def __init__(
        self,
        lr=1e-3,
        wd=0.1,
        muon_params=None,
        momentum=0.95,
        ema_beta=0.95,
        damping=1e-4,
        ns_steps=5,
        zeropower_mode="native",
        matrix_eps=1e-6,
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
            ema_beta=ema_beta,
            damping=damping,
            ns_steps=ns_steps,
            zeropower_mode=zeropower_mode,
            matrix_eps=matrix_eps,
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
                g_work = g.detach().float()
                m, n = g_work.shape
                device = g_work.device
                eps = group["matrix_eps"]

                if "P" not in state:
                    state["P"] = torch.eye(m, device=device, dtype=torch.float32)
                    state["Q"] = torch.eye(n, device=device, dtype=torch.float32)
                    state["whitened_momentum"] = torch.zeros_like(g_work)

                P_prev = state["P"]
                Q_prev = state["Q"]
                M_prev = state["whitened_momentum"]

                eye_m = torch.eye(m, device=device, dtype=torch.float32)
                eye_n = torch.eye(n, device=device, dtype=torch.float32)

                Q_inv_prev = matrix_inverse(Q_prev, eps=eps)
                L_t = (g_work @ Q_inv_prev @ g_work.transpose(-2, -1)) / float(n)
                L_t = L_t + group["damping"] * (trace(P_prev) / float(m)) * eye_m

                P_tilde = group["ema_beta"] * P_prev + (1.0 - group["ema_beta"]) * L_t
                P_t = sym_trace_normalize(P_tilde, target_trace=m, eps=eps)

                P_inv_t = matrix_inverse(P_t, eps=eps)
                R_t = (g_work.transpose(-2, -1) @ P_inv_t @ g_work) / float(m)
                R_t = R_t + group["damping"] * (trace(Q_prev) / float(n)) * eye_n

                Q_tilde = group["ema_beta"] * Q_prev + (1.0 - group["ema_beta"]) * R_t
                Q_t = sym_trace_normalize(Q_tilde, target_trace=n, eps=eps)

                P_inv_sqrt = matrix_inv_sqrt(P_t, eps=eps)
                Q_inv_sqrt = matrix_inv_sqrt(Q_t, eps=eps)

                g_white = P_inv_sqrt @ g_work @ Q_inv_sqrt
                M_t = group["momentum"] * M_prev + (1.0 - group["momentum"]) * g_white

                U_t = apply_zeropower_backend(
                    M_t.to(dtype=g.dtype),
                    steps=group["ns_steps"],
                    zeropower_mode=group["zeropower_mode"],
                ).float()

                delta = P_inv_sqrt @ U_t @ Q_inv_sqrt

                p.data.mul_(1 - group["lr"] * group["wd"])
                p.data.add_(delta.to(dtype=p.dtype), alpha=-group["lr"])

                state["P"] = P_t
                state["Q"] = Q_t
                state["whitened_momentum"] = M_t

            adamw_params = [
                p for p in group["params"] if not self.state[p].get("use_muon", False)
            ]
            for p in adamw_params:
                adamw_backup_step(group, self.state[p], p)

        return loss
