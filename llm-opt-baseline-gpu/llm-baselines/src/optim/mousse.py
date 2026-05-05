import math

import torch

from .muon_fair_utils import (
    adamw_backup_step,
    apply_zeropower_backend,
    canonicalize_zeropower_mode,
)


ADJUST_LR_ALIASES = {
    None: None,
    "none": None,
    "None": None,
    "spectral_norm": "spectral_norm",
    "rms_norm": "rms_norm",
}


class Mousse(torch.optim.Optimizer):
    """
    Simplified DDP-friendly Mousse.

    This keeps the core Mousse update path:
    momentum -> curvature EMA -> eigendecomposition -> whitening ->
    Newton-Schulz orthogonalization -> grafting -> parameter update.

    It intentionally drops the original async batching / DTensor / FSDP logic.
    Under DDP, gradients are already synchronized before `step()`, so each rank
    can run the same local optimizer update independently.
    """

    def __init__(
        self,
        lr=1e-3,
        wd=0.1,
        muon_params=None,
        momentum=0.95,
        nesterov=False,
        ns_steps=5,
        zeropower_mode="native",
        adjust_lr="rms_norm",
        flatten=False,
        shampoo_epsilon=1e-5,
        shampoo_beta=0.95,
        shampoo_update_freq=10,
        shampoo_alpha=0.125,
        lr_correction=True,
        apply_norm=True,
        use_lorr=0,
        adamw_params=None,
        adamw_betas=(0.95, 0.95),
        adamw_eps=1e-8,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if wd < 0.0:
            raise ValueError(f"Invalid weight decay: {wd}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if ns_steps < 0:
            raise ValueError(f"Invalid ns_steps: {ns_steps}")
        if shampoo_epsilon <= 0.0:
            raise ValueError(f"Invalid shampoo_epsilon: {shampoo_epsilon}")
        if not 0.0 <= shampoo_beta < 1.0:
            raise ValueError(f"Invalid shampoo_beta: {shampoo_beta}")
        if shampoo_update_freq < 1:
            raise ValueError(
                f"shampoo_update_freq must be >= 1, got {shampoo_update_freq}"
            )
        if use_lorr not in (0, 1, 2):
            raise ValueError(
                f"Unsupported use_lorr={use_lorr!r}. Expected one of: 0, 1, 2."
            )

        zeropower_mode = canonicalize_zeropower_mode(zeropower_mode)
        adjust_lr = self._canonicalize_adjust_lr(adjust_lr)

        defaults = dict(
            lr=lr,
            wd=wd,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            zeropower_mode=zeropower_mode,
            adjust_lr=adjust_lr,
            flatten=flatten,
            shampoo_epsilon=shampoo_epsilon,
            shampoo_beta=shampoo_beta,
            shampoo_update_freq=shampoo_update_freq,
            shampoo_alpha=shampoo_alpha,
            lr_correction=lr_correction,
            apply_norm=apply_norm,
            use_lorr=use_lorr,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
        )

        muon_params = list(muon_params) if muon_params is not None else []
        adamw_params = list(adamw_params) if adamw_params is not None else []
        params = muon_params + adamw_params
        super().__init__(params, defaults)

        for p in muon_params:
            if p.ndim < 2:
                raise ValueError(
                    "Mousse expects matrix-like parameters in muon_params; "
                    f"got ndim={p.ndim}."
                )
            self.state[p]["use_mousse"] = True
        for p in adamw_params:
            self.state[p]["use_mousse"] = False

    @staticmethod
    def _canonicalize_adjust_lr(adjust_lr):
        key = adjust_lr
        if isinstance(adjust_lr, str):
            key = adjust_lr.lower()
        if key not in ADJUST_LR_ALIASES:
            raise ValueError(
                f"Unsupported adjust_lr={adjust_lr!r}. "
                "Expected one of: spectral_norm, rms_norm, None."
            )
        return ADJUST_LR_ALIASES[key]

    @staticmethod
    def _reshape_to_matrix(tensor, flatten):
        if tensor.ndim == 2:
            return tensor, tensor.shape
        if flatten and tensor.ndim >= 2:
            return tensor.reshape(tensor.shape[0], -1), tensor.shape
        raise ValueError(
            "Simplified Mousse only supports 2D tensors unless flatten=True; "
            f"got shape={tuple(tensor.shape)}."
        )

    @staticmethod
    def _symmetrize(matrix):
        return 0.5 * (matrix + matrix.transpose(-2, -1))

    @staticmethod
    def _trace_normalize(matrix, target_trace, eps):
        matrix = Mousse._symmetrize(matrix)
        tr = torch.trace(matrix).clamp_min(eps)
        matrix = matrix * (float(target_trace) / tr)
        return Mousse._symmetrize(matrix)

    @staticmethod
    def _clean_eigenvalues(evals, eps):
        min_eig = evals.min()
        shift = torch.clamp(-min_eig, min=0.0) + eps
        return evals + shift

    @staticmethod
    def _adjust_lr(lr, matrix_shape, adjust_mode):
        if adjust_mode is None:
            return lr
        fan_out, fan_in = matrix_shape
        if adjust_mode == "rms_norm":
            return lr * (0.2 * math.sqrt(max(fan_out, fan_in)))
        if adjust_mode == "spectral_norm":
            return lr * math.sqrt(fan_out / fan_in)
        raise ValueError(f"Unsupported adjust_mode={adjust_mode!r}")

    def _init_mousse_state(self, param, grad_2d, group):
        state = self.state[param]
        if (
            "momentum_buffer" in state
            and state["momentum_buffer"].shape == grad_2d.shape
            and "shape_2d" in state
            and tuple(state["shape_2d"]) == tuple(grad_2d.shape)
        ):
            return state

        m_dim, n_dim = grad_2d.shape
        state["shape_2d"] = tuple(grad_2d.shape)
        state["momentum_buffer"] = torch.zeros_like(grad_2d, dtype=torch.float32)

        if group["use_lorr"] in (0, 1):
            state["L"] = torch.zeros(
                (m_dim, m_dim), device=grad_2d.device, dtype=torch.float32
            )
            state["eye_L"] = torch.eye(m_dim, device=grad_2d.device, dtype=torch.float32)
            state["eig_L"] = (
                torch.ones(m_dim, device=grad_2d.device, dtype=torch.float32),
                state["eye_L"].clone(),
            )

        if group["use_lorr"] in (0, 2):
            state["R"] = torch.zeros(
                (n_dim, n_dim), device=grad_2d.device, dtype=torch.float32
            )
            state["eye_R"] = torch.eye(n_dim, device=grad_2d.device, dtype=torch.float32)
            state["eig_R"] = (
                torch.ones(n_dim, device=grad_2d.device, dtype=torch.float32),
                state["eye_R"].clone(),
            )

        return state

    def _maybe_update_eigendecomp(self, state, grad_2d, group, step):
        beta = group["shampoo_beta"]
        eps = group["shampoo_epsilon"]
        use_left = group["use_lorr"] in (0, 1)
        use_right = group["use_lorr"] in (0, 2)

        if use_left:
            state["L"].mul_(beta)
            state["L"].add_(grad_2d @ grad_2d.T, alpha=1.0 - beta)
        if use_right:
            state["R"].mul_(beta)
            state["R"].add_(grad_2d.T @ grad_2d, alpha=1.0 - beta)

        if step % group["shampoo_update_freq"] != 1 and step != 1:
            return

        bias_correction = 1.0
        if group["lr_correction"]:
            bias_correction = 1.0 - beta**step

        if use_left:
            corrected = state["L"] / bias_correction
            corrected = self._trace_normalize(corrected, corrected.shape[0], eps)
            evals, evecs = torch.linalg.eigh(corrected + eps * state["eye_L"])
            state["eig_L"] = (self._clean_eigenvalues(evals, eps), evecs)

        if use_right:
            corrected = state["R"] / bias_correction
            corrected = self._trace_normalize(corrected, corrected.shape[0], eps)
            evals, evecs = torch.linalg.eigh(corrected + eps * state["eye_R"])
            state["eig_R"] = (self._clean_eigenvalues(evals, eps), evecs)

    def _mousse_direction(self, update_2d, grad_2d, state, group, step):
        update = update_2d.float()
        grad = grad_2d.float()
        self._maybe_update_eigendecomp(state, grad, group, step)

        use_left = group["use_lorr"] in (0, 1)
        use_right = group["use_lorr"] in (0, 2)
        alpha = group["shampoo_alpha"]

        if use_left:
            eval_L, evec_L = state["eig_L"]
            update = evec_L.T @ update
            scale_L = eval_L.abs().pow(alpha).clamp_min(group["shampoo_epsilon"])
            update = update / scale_L.unsqueeze(1)
        if use_right:
            eval_R, evec_R = state["eig_R"]
            update = update @ evec_R
            scale_R = eval_R.abs().pow(alpha).clamp_min(group["shampoo_epsilon"])
            update = update / scale_R.unsqueeze(0)

        update = apply_zeropower_backend(
            update,
            steps=group["ns_steps"],
            zeropower_mode=group["zeropower_mode"],
        ).float()

        target_norm = None
        if group["apply_norm"]:
            target_norm = update.norm()

        if use_left:
            update = update / scale_L.unsqueeze(1)
        if use_right:
            update = update / scale_R.unsqueeze(0)
        if use_left:
            update = evec_L @ update
        if use_right:
            update = update @ evec_R.T

        if target_norm is not None:
            update.mul_(target_norm / update.norm().clamp_min(1e-12))

        return update

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            mousse_params = [
                p for p in group["params"] if self.state[p].get("use_mousse", False)
            ]
            if mousse_params:
                step = group.get("mousse_step", 0) + 1
                group["mousse_step"] = step
                momentum = group["momentum"]

                for p in mousse_params:
                    g = p.grad
                    if g is None:
                        continue

                    g_2d, original_shape = self._reshape_to_matrix(g, group["flatten"])
                    g_2d = g_2d.float()
                    state = self._init_mousse_state(p, g_2d, group)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g_2d)

                    update = g_2d.add(buf, alpha=momentum) if group["nesterov"] else buf
                    update = self._mousse_direction(update, g_2d, state, group, step)

                    adjusted_lr = self._adjust_lr(
                        group["lr"], tuple(g_2d.shape), group["adjust_lr"]
                    )
                    p.data.mul_(1 - group["lr"] * group["wd"])
                    p.data.add_(
                        update.reshape(original_shape).to(dtype=p.dtype),
                        alpha=-adjusted_lr,
                    )

            adamw_params = [
                p for p in group["params"] if not self.state[p].get("use_mousse", False)
            ]
            for p in adamw_params:
                adamw_backup_step(group, self.state[p], p)

        return loss
