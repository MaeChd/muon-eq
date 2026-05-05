## MuonPlus: Muon with optional post-polar normalization
## Based on muon.py from Moonlight / KellerJordan
##
## After polar factorization, optionally apply:
##   "col"     : column-wise L2 normalization  (divide each col by its L2 norm)
##   "row"     : row-wise L2 normalization     (divide each row by its L2 norm)
##   "col_row" : column-wise then row-wise
##   "row_col" : row-wise then column-wise
##   "none"    : no normalization (identical to original Muon)
##
## Controlled via yaml, e.g.:
##   optimizer:
##     name: MuonPlus
##     norm_mode: col_row   # or row / col / row_col / none
##     norm_eps: 1e-8

import torch
from functools import partial
import math

try:
    from .polar_express import PolarExpress, FastApplyPolarExpress
except ImportError:
    from polar_express import PolarExpress, FastApplyPolarExpress


# ---------------------------------------------------------------------------
# Polar factorization back-ends
# ---------------------------------------------------------------------------

@torch.compile
def jiacheng(G, steps):
    assert len(G.shape) >= 2
    abc_list = [
        (3955/1024, -8306/1024, 5008/1024),
        (3735/1024, -6681/1024, 3463/1024),
        (3799/1024, -6499/1024, 3211/1024),
        (4019/1024, -6385/1024, 2906/1024),
        (2677/1024, -3029/1024, 1162/1024),
        (2172/1024, -1833/1024,  682/1024)
    ]
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    if steps > len(abc_list):
        steps = len(abc_list)
    for a, b, c in abc_list[:steps]:
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.mT
    return X


@torch.compile
def zeropower_via_newtonschulz5(G, steps):
    assert len(G.shape) >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.mT
    return X


@torch.compile
def svd_exact_polar(G, _, cutoff=None, reverse=False):
    assert len(G.shape) >= 2
    U, Sigma, Vh = torch.linalg.svd(G.to(torch.float32), full_matrices=False)
    if cutoff is None:
        return (U @ Vh).to(G.dtype)
    else:
        Sigma = ((Sigma / Sigma.max()) >= cutoff).to(G.dtype)
        if reverse:
            Sigma = 2 * Sigma - 1
        return (U @ torch.diag(Sigma) @ Vh).to(G.dtype)


# ---------------------------------------------------------------------------
# Normalization helper
# ---------------------------------------------------------------------------

def apply_post_polar_norm(u: torch.Tensor, norm_mode: str, eps: float = 1e-8) -> torch.Tensor:
    """Apply row/col L2 normalization to the polar-factorized update u.

    eps is placed *inside* the sqrt: sqrt(sum_sq + 1e-7)

    Modes
    -----
    "col"     : col-wise L2 norm
    "row"     : row-wise L2 norm
    "col_row" : col-wise then row-wise
    "row_col" : row-wise then col-wise
    "none"    : no-op
    """
    if norm_mode in ("none", None, ""):
        return u

    # First normalization
    if norm_mode in ("col", "col_row"):
        u = u / torch.sqrt((u * u).sum(dim=-2, keepdim=True) + 1e-7)
    elif norm_mode in ("row", "row_col"):
        u = u / torch.sqrt((u * u).sum(dim=-1, keepdim=True) + 1e-7)
    else:
        raise ValueError(
            f"Unknown norm_mode '{norm_mode}'. "
            "Choose from: 'none', 'col', 'row', 'col_row', 'row_col'."
        )

    # Second normalization (combined modes only)
    if norm_mode == "col_row":
        u = u / torch.sqrt((u * u).sum(dim=-1, keepdim=True) + 1e-7)
    elif norm_mode == "row_col":
        u = u / torch.sqrt((u * u).sum(dim=-2, keepdim=True) + 1e-7)

    return u


# ---------------------------------------------------------------------------
# MuonPlus optimizer
# ---------------------------------------------------------------------------

class MuonPlus(torch.optim.Optimizer):
    """
    MuonPlus - Muon with optional post-polar normalization.

    Identical to the original Muon except that after polar factorization the
    update matrix ``u`` can be column-wise, row-wise, or both-wise L2-
    normalized before being applied to the parameters.

    Extra arguments vs. Muon
    ------------------------
    norm_mode : str
        One of ``"none"`` | ``"col"`` | ``"row"`` | ``"col_row"`` | ``"row_col"``.
        Controlled via yaml (default: ``"none"``).
    norm_eps : float
        Small constant for numerical stability in normalization (default 1e-8).
    """

    def __init__(
        self,
        named_params,
        lr=1e-3,
        weight_decay=0.1,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        rms_scaling=True,
        nuclear_scaling=False,
        polar_method="Keller",
        adamw_betas=(0.95, 0.95),
        adamw_eps=1e-8,
        split_heads=False,
        nheads=None,
        polar_args={},
        # Normalization
        norm_mode: str = "none",
        norm_eps: float = 1e-8,
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            rms_scaling=rms_scaling,
            nuclear_scaling=nuclear_scaling,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
        )

        muon_params, muon_params_names = [], []
        adamw_params, adamw_params_names = [], []
        for name, p in named_params:
            if p.ndim >= 2 and not any(
                excl in name
                for excl in ["embeddings", "embed_tokens", "wte", "lm_head", "wpe"]
            ):
                muon_params.append(p)
                muon_params_names.append(name)
            else:
                adamw_params.append(p)
                adamw_params_names.append(name)

        params = list(muon_params)
        params.extend(adamw_params)

        self.split_heads = split_heads
        if self.split_heads:
            assert nheads is not None, "nheads must be specified if split_heads is True"
            self.nheads = nheads

        super().__init__(params, defaults)

        for p, p_name in zip(muon_params, muon_params_names):
            if not self.split_heads:
                assert p.ndim == 2, p.ndim
            self.state[p]["use_muon"] = True
            if p_name.endswith("attn.c_attn.weight"):
                self.state[p]["is_W_QKV"] = True
            elif p_name.endswith("attn.c_proj.weight"):
                self.state[p]["is_W_O"] = True
            elif p_name.endswith("mlp.c_fc.weight"):
                self.state[p]["is_W_MLP1"] = True
            elif p_name.endswith("mlp.c_proj.weight"):
                self.state[p]["is_W_MLP2"] = True

        for p in adamw_params:
            self.state[p]["use_muon"] = False

        self.polar_factorizer = self._initialize_polar_factorizer(polar_method, polar_args)

        # Normalization settings
        valid_modes = ("none", "col", "row", "col_row", "row_col")
        if norm_mode not in valid_modes:
            raise ValueError(
                f"norm_mode must be one of {valid_modes}, got '{norm_mode}'."
            )
        self.norm_mode = norm_mode
        self.norm_eps = norm_eps

    # ---------------------------------------------------------------------------
    def _initialize_polar_factorizer(self, polar_method, polar_args):
        if polar_method == "Keller":
            return zeropower_via_newtonschulz5
        elif polar_method == "Jiacheng":
            return jiacheng
        elif polar_method == "polarexpress":
            return PolarExpress
        elif polar_method == "fast_polarexpress":
            return partial(FastApplyPolarExpress, restart_interval=3, shift_eps=1e-3)
        elif polar_method == "svd-exact":
            return partial(
                svd_exact_polar,
                cutoff=polar_args.get("svd_cutoff", None),
                reverse=polar_args.get("svd_reverse", False),
            )
        else:
            raise ValueError(f"Unknown polar method: {polar_method}")

    def adjust_lr_for_muon(self, lr, rms_scaling, nuclear_scaling, param_shape, grad, grad_sign):
        scale = 1.0
        if rms_scaling:
            fan_out, fan_in = param_shape[:2]
            scale *= math.sqrt(fan_out / fan_in)
        if nuclear_scaling:
            scale *= torch.trace(grad.T @ grad_sign)
        return lr * scale

    # ---------------------------------------------------------------------------
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            # ---------------------------------------------------------------------------
            #  Muon branch
            # ---------------------------------------------------------------------------
            params = [p for p in group["params"] if self.state[p]["use_muon"]]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]

            for p in params:
                g = p.grad
                if g is None:
                    continue
                if g.ndim > 2 and not self.split_heads:
                    g = g.view(g.size(0), -1)

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                if group["nesterov"]:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf

                # Reshape for split_heads
                if self.split_heads and state.get("is_W_QKV", False):
                    old_shape = g.shape
                    g = g.reshape(
                        3 * self.nheads,
                        g.shape[0] // (3 * self.nheads),
                        g.shape[1],
                    )
                elif self.split_heads and state.get("is_W_O", False):
                    old_shape = g.shape
                    g = g.reshape(
                        g.shape[0], self.nheads, g.shape[1] // self.nheads
                    ).transpose(0, 1)

                # ---- Polar factorization ----
                u = self.polar_factorizer(g, group["ns_steps"])

                # ---- Post-polar normalization ----
                u = apply_post_polar_norm(u, self.norm_mode, self.norm_eps)

                # Reshape back for split_heads
                if self.split_heads and state.get("is_W_QKV", False):
                    g = g.reshape(old_shape)
                    u = u.reshape(old_shape)
                elif self.split_heads and state.get("is_W_O", False):
                    g = g.transpose(0, 1).reshape(old_shape)
                    u = u.transpose(0, 1).reshape(old_shape)

                adjusted_lr = self.adjust_lr_for_muon(
                    lr,
                    group["rms_scaling"],
                    group["nuclear_scaling"],
                    p.shape,
                    g.bfloat16(),
                    u,
                )

                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(u, alpha=-adjusted_lr)

            # ---------------------------------------------------------------------------
            #  AdamW branch
            # ---------------------------------------------------------------------------
            params = [p for p in group["params"] if not self.state[p]["use_muon"]]
            lr = group["lr"]
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            weight_decay = group["weight_decay"]

            for p in params:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)
                state["step"] += 1
                step = state["step"]
                buf1 = state["moment1"]
                buf2 = state["moment2"]
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)

                g = buf1 / (eps + buf2.sqrt())

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5
                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(g, alpha=-lr / scale)

        return loss
