import torch


ZEROPOWER_MODE_ALIASES = {
    "native": "native",
    "zeropower_via_newtonschulz5": "native",
    "spc": "spc",
    "zeropower_via_newtonschulz5_spc": "spc",
}


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int) -> torch.Tensor:
    assert G.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315
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


def zeropower_via_newtonschulz5_spc(G: torch.Tensor, steps: int) -> torch.Tensor:
    assert G.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315

    X = G.bfloat16()
    m, n = X.shape
    left = m <= n

    def gram(x: torch.Tensor) -> torch.Tensor:
        return x @ x.T if left else x.T @ x

    A = gram(X)
    A2 = A @ A

    inv_s = (A2.float().norm() + 1e-7).pow(-0.25)
    X = X * inv_s

    if steps == 0:
        return X

    s2 = inv_s * inv_s
    s4 = s2 * s2
    B = A2.mul(c * s4).add_(A, alpha=b * s2)

    if left:
        X = a * X + B @ X
    else:
        X = a * X + X @ B

    for _ in range(steps - 1):
        A = gram(X)
        B = A @ A
        B.mul_(c).add_(A, alpha=b)
        if left:
            X = a * X + B @ X
        else:
            X = a * X + X @ B

    return X


def canonicalize_zeropower_mode(zeropower_mode: str) -> str:
    canonical_mode = ZEROPOWER_MODE_ALIASES.get(str(zeropower_mode).lower())
    if canonical_mode is None:
        raise ValueError(
            f"Unsupported zeropower_mode={zeropower_mode!r}. Expected one of: native, spc."
        )
    return canonical_mode


def apply_zeropower_backend(
    g: torch.Tensor, steps: int, zeropower_mode: str
) -> torch.Tensor:
    zeropower_mode = canonicalize_zeropower_mode(zeropower_mode)
    if zeropower_mode == "spc":
        return zeropower_via_newtonschulz5_spc(g, steps=steps)
    return zeropower_via_newtonschulz5(g, steps=steps)


def norm_dir(X: torch.Tensor, direction: str = "col", eps: float = 1e-8) -> torch.Tensor:
    if direction == "col":
        denom = (X.square().sum(dim=-2, keepdim=True) + eps).sqrt()
        return X / denom
    if direction == "row":
        denom = (X.square().sum(dim=-1, keepdim=True) + eps).sqrt()
        return X / denom
    if direction == "col_row":
        return norm_dir(norm_dir(X, "col", eps), "row", eps)
    if direction == "row_col":
        return norm_dir(norm_dir(X, "row", eps), "col", eps)
    raise ValueError(f"Unsupported normalization direction: {direction!r}")


def symmetrize(A: torch.Tensor) -> torch.Tensor:
    return 0.5 * (A + A.transpose(-2, -1))


def trace(A: torch.Tensor) -> torch.Tensor:
    return torch.diagonal(A, dim1=-2, dim2=-1).sum()


def sym_trace_normalize(A: torch.Tensor, target_trace: int, eps: float) -> torch.Tensor:
    A = symmetrize(A)
    tr = trace(A).clamp_min(eps)
    A = A * (float(target_trace) / tr)
    return symmetrize(A)


def matrix_inverse(A: torch.Tensor, eps: float) -> torch.Tensor:
    evals, evecs = torch.linalg.eigh(symmetrize(A))
    evals = evals.clamp_min(eps)
    inv_evals = evals.reciprocal()
    return (evecs * inv_evals.unsqueeze(0)) @ evecs.transpose(-2, -1)


def matrix_inv_sqrt(A: torch.Tensor, eps: float) -> torch.Tensor:
    evals, evecs = torch.linalg.eigh(symmetrize(A))
    evals = evals.clamp_min(eps)
    inv_sqrt_evals = evals.rsqrt()
    return (evecs * inv_sqrt_evals.unsqueeze(0)) @ evecs.transpose(-2, -1)


def adamw_backup_step(group, state, p):
    g = p.grad
    if g is None:
        return

    lr = group["lr"]
    beta1, beta2 = group["adamw_betas"]
    eps = group["adamw_eps"]
    weight_decay = group.get("wd", group.get("weight_decay", 0.0))

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

    bias_correction1 = 1 - beta1**step
    bias_correction2 = 1 - beta2**step
    scale = bias_correction1 / (bias_correction2**0.5)

    p.data.mul_(1 - lr * weight_decay)
    p.data.add_(g_hat, alpha=-lr / scale)
