import math
import torch


ZEROPOWER_MODE_ALIASES = {
    "native": "native",
    "zeropower_via_newtonschulz5": "native",
    "spc": "spc",
    "zeropower_via_newtonschulz5_spc": "spc",
}


def zeropower_via_newtonschulz5(G, steps):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)

    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T

    X = X / (X.norm() + 1e-7)

    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T
    return X


def zeropower_via_newtonschulz5_spc(
    G: torch.Tensor,
    steps: int,
) -> torch.Tensor:
    """
    Exact J=2 spectral-proxy calibration + Newton-Schulz zeropower.

    Compared with the naive implementation:
      - always builds the Gram matrix on the short side
      - reuses the precomputed Gram and Gram^2 for NS step 1
      - therefore does NOT add extra GEMMs over vanilla NS
    """
    assert G.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315

    X = G.bfloat16()
    m, n = X.shape
    left = (m <= n)  # always build the smaller Gram

    def gram(x: torch.Tensor) -> torch.Tensor:
        return x @ x.T if left else x.T @ x

    # Build small-side Gram once
    A = gram(X)          # d x d, d = min(m, n)
    A2 = A @ A           # exact J=2 proxy needs this anyway

    # scale = ||A^2||_F^{-1/4}; do reduction in fp32 for stability
    inv_s = (A2.float().norm() + 1e-7).pow(-0.25)
    X = X * inv_s

    if steps == 0:
        return X

    # Reuse A and A2 for NS step 1:
    # after scaling, A -> inv_s^2 * A, A2 -> inv_s^4 * A2
    s2 = inv_s * inv_s
    s4 = s2 * s2

    # Reuse A2 buffer as B to save one allocation:
    # B = b * (s^2 A) + c * (s^4 A^2)
    B = A2.mul(c * s4).add_(A, alpha=b * s2)

    if left:
        X = a * X + B @ X
    else:
        X = a * X + X @ B

    # Remaining NS steps
    for _ in range(steps - 1):
        A = gram(X)
        B = A @ A            # reuse B buffer name for A^2
        B.mul_(c).add_(A, alpha=b)   # B = b A + c A^2

        if left:
            X = a * X + B @ X
        else:
            X = a * X + X @ B

    return X

class MuonEq(torch.optim.Optimizer):
    """
    Standalone MuonEq.

    Applies a lightweight stateless normalization preprocessing step to
    Muon's Nesterov update.
    Select the normalization strategy with normalize_mode:
        rowcol : G_tilde = D_r^{exp} G D_c^{exp}
        row    : G_tilde = D_r^{exp} G
        col    : G_tilde = G D_c^{exp}

    The default exp is -1/2, and rowcol_scale_exponent can change it to -1/4.
    When normalize_mode="row" and phase is set:
        step < phase  : use row/col normalization
        step >= phase : switch to row-only normalization
    Select the zeropower backend with zeropower_mode:
        native : zeropower_via_newtonschulz5
        spc    : zeropower_via_newtonschulz5_spc
    """

    def __init__(
        self,
        lr=1e-3,
        wd=0.1,
        muon_params=None,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        adamw_params=None,
        adamw_betas=(0.95, 0.95),
        adamw_eps=1e-8,
        rowcol_scale_exponent=-0.5,
        rowcol_eps=1e-8,
        rowcol_clip=None,
        normalize_mode="row",
        phase=None,
        zeropower_mode="native",
    ):
        normalize_mode = self._canonicalize_normalize_mode(normalize_mode)
        zeropower_mode = self._canonicalize_zeropower_mode(zeropower_mode)
        defaults = dict(
            lr=lr,
            wd=wd,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
            rowcol_scale_exponent=rowcol_scale_exponent,
            rowcol_eps=rowcol_eps,
            rowcol_clip=rowcol_clip,
            normalize_mode=normalize_mode,
            phase=phase,
            zeropower_mode=zeropower_mode,
        )

        muon_params = list(muon_params) if muon_params is not None else []
        adamw_params = list(adamw_params) if adamw_params is not None else []
        params = list(muon_params)
        params.extend(adamw_params)
        super().__init__(params, defaults)

        for p in muon_params:
            assert p.ndim == 2, p.ndim
            self.state[p]["use_muon"] = True

        for p in adamw_params:
            self.state[p]["use_muon"] = False

    def adjust_lr_for_muon(self, lr, param_shape):
        A, B = param_shape[:2]
        adjusted_ratio = 0.2 * math.sqrt(max(A, B))
        adjusted_lr = lr * adjusted_ratio
        return adjusted_lr

    @staticmethod
    def _canonicalize_normalize_mode(normalize_mode):
        aliases = {
            "rowcol": "rowcol",
            "row-col": "rowcol",
            "row_col": "rowcol",
            "row": "row",
            "col": "col",
            "column": "col",
        }
        canonical_mode = aliases.get(str(normalize_mode).lower())
        if canonical_mode is None:
            raise ValueError(
                f"Unsupported MuonEq normalize_mode={normalize_mode!r}. "
                "Expected one of: rowcol, row, col."
            )
        return canonical_mode

    @staticmethod
    def _canonicalize_zeropower_mode(zeropower_mode):
        canonical_mode = ZEROPOWER_MODE_ALIASES.get(str(zeropower_mode).lower())
        if canonical_mode is None:
            raise ValueError(
                f"Unsupported MuonEq zeropower_mode={zeropower_mode!r}. "
                "Expected one of: native, spc."
            )
        return canonical_mode

    @torch.no_grad()
    def _apply_scale_exponent_(self, scale, scale_exp, eps):
        scale.add_(eps)
        if scale_exp == -0.25:
            scale.rsqrt_().sqrt_()
        elif scale_exp == -0.5:
            scale.rsqrt_()
        else:
            scale.pow_(scale_exp)

    @torch.no_grad()
    def _clamp_scale_(self, scale, clip):
        if clip is None:
            return
        inv_clip = 1.0 / clip
        scale.clamp_(min=inv_clip, max=clip)

    @torch.no_grad()
    def _parallel_rowcol_normalize(self, g, group):
        assert g.ndim == 2

        clip = group["rowcol_clip"]
        scale_exp = float(group.get("rowcol_scale_exponent", -0.25))
        sq = g.square()
        row_scale = sq.sum(dim=1)
        col_scale = sq.sum(dim=0)
        self._apply_scale_exponent_(row_scale, scale_exp, group["rowcol_eps"])
        self._apply_scale_exponent_(col_scale, scale_exp, group["rowcol_eps"])
        self._clamp_scale_(row_scale, clip)
        self._clamp_scale_(col_scale, clip)

        g_tilde = g * row_scale[:, None]
        g_tilde.mul_(col_scale[None, :])
        return g_tilde

    @torch.no_grad()
    def _row_normalize(self, g, group):
        assert g.ndim == 2

        scale_exp = float(group.get("rowcol_scale_exponent", -0.25))
        row_scale = g.square().sum(dim=1)
        self._apply_scale_exponent_(row_scale, scale_exp, group["rowcol_eps"])
        self._clamp_scale_(row_scale, group["rowcol_clip"])
        return g * row_scale[:, None]

    @torch.no_grad()
    def _col_normalize(self, g, group):
        assert g.ndim == 2

        scale_exp = float(group.get("rowcol_scale_exponent", -0.25))
        col_scale = g.square().sum(dim=0)
        self._apply_scale_exponent_(col_scale, scale_exp, group["rowcol_eps"])
        self._clamp_scale_(col_scale, group["rowcol_clip"])
        return g * col_scale[None, :]

    @torch.no_grad()
    def _resolve_normalize_mode(self, group, step):
        normalize_mode = self._canonicalize_normalize_mode(group.get("normalize_mode", "rowcol"))
        phase = group.get("phase")
        if phase is not None and normalize_mode == "rowcol" and step >= phase:
            return "row"
        return normalize_mode

    @torch.no_grad()
    def _normalize_muon_update(self, g, group, step):
        normalize_mode = self._resolve_normalize_mode(group, step)
        if normalize_mode == "row":
            return self._row_normalize(g, group)
        if normalize_mode == "col":
            return self._col_normalize(g, group)
        return self._parallel_rowcol_normalize(g, group)

    @torch.no_grad()
    def _orthogonalize_muon_update(self, g, group):
        zeropower_mode = self._canonicalize_zeropower_mode(group.get("zeropower_mode", "native"))
        if zeropower_mode == "spc":
            return zeropower_via_newtonschulz5_spc(g, steps=group["ns_steps"])
        return zeropower_via_newtonschulz5(g, steps=group["ns_steps"])

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:

            params = [p for p in group["params"] if self.state[p]["use_muon"]]
            lr = group["lr"]
            wd = group["wd"]
            momentum = group["momentum"]
            muon_step = group.get("muon_step", 0) + 1
            group["muon_step"] = muon_step

            for p in params:
                g = p.grad
                if g is None:
                    continue

                if g.ndim > 2:
                    g = g.view(g.size(0), -1)
                assert g.ndim == 2

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)

                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)

                if group["nesterov"]:
                    update = g.add(buf, alpha=momentum)
                else:
                    update = buf

                update = self._normalize_muon_update(update, group, muon_step)
                u = self._orthogonalize_muon_update(update, group)

                adjusted_lr = self.adjust_lr_for_muon(lr, p.shape)

                p.data.mul_(1 - lr * wd)
                p.data.add_(u, alpha=-adjusted_lr)

            params = [p for p in group["params"] if not self.state[p]["use_muon"]]
            lr = group["lr"]
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            weight_decay = group["wd"]

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

                g_hat = buf1 / (eps + buf2.sqrt())

                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                scale = bias_correction1 / (bias_correction2 ** 0.5)

                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(g_hat, alpha=-lr / scale)

        return loss
