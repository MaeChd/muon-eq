import math
from typing import Iterable, Optional, Tuple

import torch


# -----------------------------
# Core linear algebra utilities
# -----------------------------

def _normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm() + eps)


@torch.no_grad()
def power_iteration_top_singular(
    W: torch.Tensor,
    steps: int = 10,
    v0: Optional[torch.Tensor] = None,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Power iteration for top singular triplet of W (approx):
        u = argmax ||W v||, v = argmax ||W^T u||.

    Returns:
        sigma: scalar tensor
        u: (out_dim,)
        v: (in_dim,)
    """
    assert W.ndim == 2
    out_dim, in_dim = W.shape

    # Work in float32 for stability (on GPU this is typically fine; overhead is small vs matmuls)
    Wf = W.float()

    if v0 is None or v0.numel() != in_dim or v0.device != W.device:
        v = torch.randn(in_dim, device=W.device, dtype=torch.float32)
    else:
        v = v0.detach().float()

    v = _normalize(v, eps)

    for _ in range(steps):
        u = Wf @ v
        u = _normalize(u, eps)
        v = Wf.t() @ u
        v = _normalize(v, eps)

    # sigma ≈ u^T W v  (>=0)
    sigma = (u @ (Wf @ v)).abs()
    return sigma, u, v


@torch.no_grad()
def msign_polar_express_ns5(A: torch.Tensor, steps: int = 8, eps: float = 1e-7) -> torch.Tensor:
    """
    Approximate msign(A) = UV^T where A = U S V^T (polar factor).
    Uses a quintic Newton–Schulz style iteration (same coefficient style as your Muon snippet).

    Practical notes:
    - We rescale by Frobenius norm to ensure spectral norm <= 1 (conservative but stable).
    - On CUDA, we run the iteration in bf16 to reduce cost; on CPU keep float32.

    Returns:
        X approximating UV^T (same shape as A).
    """
    assert A.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)

    # Use bf16 on GPU for speed; float32 on CPU for correctness/stability
    work_dtype = torch.bfloat16 if A.is_cuda else torch.float32
    X = A.to(dtype=work_dtype)

    transposed = False
    if X.size(0) > X.size(1):
        X = X.t()
        transposed = True

    X = X / (X.norm() + eps)  # conservative scaling

    for _ in range(steps):
        AA = X @ X.t()
        B = b * AA + c * (AA @ AA)
        X = a * X + (B @ X)

    if transposed:
        X = X.t()

    return X.to(dtype=A.dtype)


def spectral_ball_scale_factor(dout: int, din: int, mode: str) -> float:
    """
    Learning-rate scaler R in paper Eq.(16).  :contentReference[oaicite:7]{index=7}
    """
    r = dout / din
    if mode == "spectral_mup":
        return math.sqrt(r)
    if mode == "align_adamw_rms":
        return 0.2 * math.sqrt(max(dout, din))
    if mode in ("shape_scaling", "spectral_kaiming"):
        return math.sqrt(max(1.0, r))
    raise ValueError(f"Unknown scale_mode: {mode}")


def target_spectral_radius(dout: int, din: int, radius_mode: str, radius_scaler: float) -> float:
    """
    Target radius for ||W||_2 = R. For spectral_mup: R = c * sqrt(dout/din). :contentReference[oaicite:8]{index=8}
    """
    r = dout / din
    if radius_mode == "spectral_mup":
        return radius_scaler * math.sqrt(r)
    if radius_mode == "identity":
        return float(radius_scaler)
    if radius_mode == "initialize":
        # In this simplified version we set it to (radius_scaler * initial_sigma) per-parameter (cached in state),
        # because we don't have the original helper util here.
        # The actual initialization heuristic can vary; this is a practical drop-in.
        return float("nan")  # placeholder; resolved per-parameter in optimizer using cached init_sigma
    raise ValueError(f"Unknown radius_mode: {radius_mode}")


@torch.no_grad()
def solve_lambda_bisection(
    G: torch.Tensor,
    Theta: torch.Tensor,
    msign_steps: int,
    tol_f: float,
    max_iter: int,
) -> float:
    """
    Solve for λ such that <Theta, msign(G + λ Theta)> = 0 using bisection.

    Paper shows h(λ)=<Θ,msign(G+λΘ)> is monotonic non-decreasing, and a root exists with |λ*| bounded. :contentReference[oaicite:9]{index=9}
    We bracket using an inexpensive upper bound via Frobenius -> nuclear norm inequality.
    """
    assert G.ndim == 2 and Theta.ndim == 2
    device = G.device
    dtype = G.dtype

    # Cheap bound: ||G||_* <= sqrt(rank) * ||G||_F, rank <= min(dout,din)
    frob = G.float().norm().item()
    rank_sqrt = math.sqrt(min(G.shape[0], G.shape[1]))
    bound = 2.0 * frob * rank_sqrt + 1e-6

    def h(lmbd: float) -> float:
        A = (G + (lmbd * Theta)).to(dtype=dtype, device=device)
        Phi = msign_polar_express_ns5(A, steps=msign_steps)
        return float((Theta * Phi).sum().float().item())

    lo, hi = -bound, bound
    f_lo, f_hi = h(lo), h(hi)

    # In theory should bracket; in practice numerical approx may fail. Expand a few times then fallback.
    expand = 0
    while f_lo * f_hi > 0.0 and expand < 6:
        bound *= 2.0
        lo, hi = -bound, bound
        f_lo, f_hi = h(lo), h(hi)
        expand += 1

    if f_lo * f_hi > 0.0:
        return 0.0

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = h(mid)
        if abs(f_mid) <= tol_f:
            return mid
        if f_lo * f_mid <= 0.0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid

    return 0.5 * (lo + hi)


@torch.no_grad()
def spectral_ball_direction(
    W: torch.Tensor,
    G: torch.Tensor,
    *,
    state: dict,
    lr: float,
    power_iteration_steps: int,
    msign_steps: int,
    solver_tolerance_f: float,
    solver_max_iterations: int,
    radius_mode: str,
    radius_scaler: float,
    scale_mode: str,
    retract_mode: str,
    retract_alpha: float,
) -> torch.Tensor:
    """
    Compute update direction U for SSO:
      1) power-iter on W -> sigma,u,v
      2) retract W toward ||W||2 = R
      3) Theta = u v^T
      4) solve λ: <Θ, msign(G + λΘ)> = 0
      5) Phi = msign(G + λΘ)
      6) return U = Phi * scale_factor
    """
    assert W.ndim == 2 and G.ndim == 2
    dout, din = W.shape

    # --- power iteration (cache v for warm start) ---
    v0 = state.get("pi_v", None)
    sigma, u, v = power_iteration_top_singular(W, steps=power_iteration_steps, v0=v0)
    state["pi_v"] = v.detach()

    sigma_val = float(sigma.item()) + 1e-12

    # --- target radius ---
    if radius_mode == "initialize":
        if "init_sigma" not in state:
            state["init_sigma"] = sigma_val
        R = radius_scaler * state["init_sigma"]
    else:
        R = target_spectral_radius(dout, din, radius_mode, radius_scaler)

    # --- retraction ---
    # Hard: W <- (R/sigma) W.  Dynamic: W <- (1 + alpha*lr*sign(R-sigma)) W.  :contentReference[oaicite:10]{index=10}
    if retract_mode == "hard":
        W.mul_(R / sigma_val)
    elif retract_mode == "dynamic":
        bias = math.copysign(1.0, (R - sigma_val))
        W.mul_(1.0 + retract_alpha * lr * bias)
        state["retract_bias"] = bias
        state["spectral_sigma"] = sigma_val
    else:
        raise ValueError(f"Unknown retract_mode: {retract_mode}")

    # --- Theta ---
    Theta = torch.outer(u, v).to(device=W.device, dtype=torch.float32)

    # --- solve λ & compute Phi ---
    Gf = G.to(dtype=torch.float32)
    lam = solve_lambda_bisection(
        G=Gf,
        Theta=Theta,
        msign_steps=msign_steps,
        tol_f=solver_tolerance_f,
        max_iter=solver_max_iterations,
    )
    Phi = msign_polar_express_ns5((Gf + lam * Theta).to(dtype=W.dtype), steps=msign_steps)

    # --- scale factor (Eq.(16)) ---
    sf = spectral_ball_scale_factor(dout, din, mode=scale_mode)
    return Phi * sf


# -----------------------------
# Optimizer (SSO + AdamW backup)
# -----------------------------

class SpectralBallOptimizer(torch.optim.Optimizer):
    """
    Simplified Spectral Ball / SSO optimizer.

    - sso_params: MUST be 2D tensors (Linear weights etc.)
    - adamw_params: all other params (embeddings, heads, norms, bias, etc.)

    Key hyperparams correspond to Megatron impl & paper defaults (commonly):
      - solver_max_iterations ~ 20
      - solver_tolerance_f ~ 2e-4
      - msign_steps ~ 8 (paper) :contentReference[oaicite:11]{index=11}
    """

    def __init__(
        self,
        *,
        lr: float = 3e-4,
        wd: float = 0.0,
        sso_params: Iterable[torch.nn.Parameter],
        momentum: float = 0.9,
        nesterov: bool = True,
        power_iteration_steps: int = 10,
        msign_steps: int = 8,
        solver_tolerance_f: float = 2e-4,
        solver_max_iterations: int = 20,
        radius_mode: str = "spectral_mup",
        radius_scaler: float = 1.0,
        scale_mode: str = "spectral_mup",  # "spectral_mup" / "align_adamw_rms" / "shape_scaling"
        retract_mode: str = "hard",        # "hard" / "dynamic"
        retract_alpha: float = 0.05,
        adamw_params: Optional[Iterable[torch.nn.Parameter]] = None,
        adamw_betas=(0.9, 0.999),
        adamw_eps: float = 1e-8,
    ):
        defaults = dict(
            lr=lr,
            wd=wd,
            momentum=momentum,
            nesterov=nesterov,
            power_iteration_steps=power_iteration_steps,
            msign_steps=msign_steps,
            solver_tolerance_f=solver_tolerance_f,
            solver_max_iterations=solver_max_iterations,
            radius_mode=radius_mode,
            radius_scaler=radius_scaler,
            scale_mode=scale_mode,
            retract_mode=retract_mode,
            retract_alpha=retract_alpha,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
        )

        sso_params = list(sso_params)
        adamw_params = list(adamw_params) if adamw_params is not None else []
        params = sso_params + adamw_params
        super().__init__(params, defaults)

        # Mark which params use SSO vs AdamW
        for p in sso_params:
            if p.ndim != 2:
                raise ValueError(f"SSO params must be 2D, got ndim={p.ndim}")
            self.state[p]["use_sso"] = True
        for p in adamw_params:
            self.state[p]["use_sso"] = False

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            wd = group["wd"]

            # -----------------------------
            # SSO params (2D)
            # -----------------------------
            for p in [x for x in group["params"] if self.state[x].get("use_sso", False)]:
                g = p.grad
                if g is None:
                    continue
                if g.ndim != 2:
                    raise ValueError(f"SSO only supports 2D grads, got {g.ndim}")

                st = self.state[p]

                # Momentum
                if "momentum_buffer" not in st:
                    st["momentum_buffer"] = torch.zeros_like(g)
                buf = st["momentum_buffer"]
                buf.mul_(group["momentum"]).add_(g)

                if group["nesterov"]:
                    G = g.add(buf, alpha=group["momentum"])
                else:
                    G = buf

                # Optional decoupled weight decay (often set 0 for 2D in SSO hard-retract setups)
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                # Compute SSO direction (this may retract p in-place)
                U = spectral_ball_direction(
                    W=p,
                    G=G,
                    state=st,
                    lr=lr,
                    power_iteration_steps=group["power_iteration_steps"],
                    msign_steps=group["msign_steps"],
                    solver_tolerance_f=group["solver_tolerance_f"],
                    solver_max_iterations=group["solver_max_iterations"],
                    radius_mode=group["radius_mode"],
                    radius_scaler=group["radius_scaler"],
                    scale_mode=group["scale_mode"],
                    retract_mode=group["retract_mode"],
                    retract_alpha=group["retract_alpha"],
                )

                # Apply update: W <- W - lr * U
                p.add_(U, alpha=-lr)

            # -----------------------------
            # AdamW backup params
            # -----------------------------
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]

            for p in [x for x in group["params"] if not self.state[x].get("use_sso", False)]:
                g = p.grad
                if g is None:
                    continue

                st = self.state[p]
                if "step" not in st:
                    st["step"] = 0
                    st["m1"] = torch.zeros_like(g)
                    st["m2"] = torch.zeros_like(g)
                st["step"] += 1
                t = st["step"]

                m1 = st["m1"]
                m2 = st["m2"]
                m1.lerp_(g, 1.0 - beta1)
                m2.lerp_(g.square(), 1.0 - beta2)

                m1_hat = m1 / (1.0 - beta1**t)
                m2_hat = m2 / (1.0 - beta2**t)

                update = m1_hat / (m2_hat.sqrt() + eps)

                # decoupled wd
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                p.add_(update, alpha=-lr)

        return loss
