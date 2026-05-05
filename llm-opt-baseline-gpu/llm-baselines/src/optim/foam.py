import math
from typing import Iterable, Tuple, Callable, Optional

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer





class FOAM_simple(Optimizer):
    """
    FOAM + AdamW hybrid optimizer.

    - Use standard FOAM for foam_params with folded first/second moments.
    - Use AdamW with bias correction as the backup for adamw_params.

    Typical usage, similar to Muon:

        foam_params = [
            p
            for name, p in model.named_parameters()
            if p.ndim >= 2
            and "embed_tokens" not in name
            and "lm_head" not in name
            and p.requires_grad
        ]
        adamw_params = [
            p
            for name, p in model.named_parameters()
            if not (
                p.ndim >= 2
                and "embed_tokens" not in name
                and "lm_head" not in name
                and p.requires_grad
            )
        ]

        optimizer = FOAM(
            lr=args.lr,
            wd=args.weight_decay,
            foam_params=foam_params,
            adamw_params=adamw_params,
            betas=args.foam_betas,        # FOAM (beta1, beta2)
            fold_level=args.fold_level,   # ℓ, block_size = 2**ℓ
            alpha=args.alpha,             # FOAM scaling factor
            eps=args.eps,                 # FOAM eps
            adamw_betas=args.betas,       # AdamW (beta1, beta2)
            adamw_eps=args.eps            # AdamW eps
        )
    """

    def __init__(
        self,
        lr: float,
        wd: float,
        foam_params: Iterable[nn.Parameter],
        adamw_params: Iterable[nn.Parameter],
        betas: Tuple[float, float] = (0.9, 0.999),      # FOAM betas
        fold_level: int = 1,                            # ℓ, block_size = 2**ℓ
        alpha: float = 1.0,                             # FOAM scaling
        eps: float = 1e-8,                              # FOAM eps
        adamw_betas: Optional[Tuple[float, float]] = None,
        adamw_eps: Optional[float] = None,
        foam_type = 'foam'
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if fold_level < 1:
            raise ValueError(f"fold_level must be >= 1, got {fold_level}")

        if adamw_betas is None:
            adamw_betas = betas
        if adamw_eps is None:
            adamw_eps = eps

        foam_params = list(foam_params)
        adamw_params = list(adamw_params)
        all_params = foam_params + adamw_params
        if len(all_params) == 0:
            raise ValueError("FOAM received no parameters.")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            fold_level=fold_level,
            alpha=alpha,
            wd=wd,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
            foam_type=foam_type
        )
        # Use one param_group; all hyperparameters come from defaults.
        super().__init__(all_params, defaults)

        foam_set = set(foam_params)
        # Mark whether each parameter uses FOAM in the state.
        for group in self.param_groups:
            for p in group["params"]:
                self.state[p]["use_foam"] = (p in foam_set)

    # ---------- General: arbitrary shape <-> (m, n) view ----------

    @staticmethod
    def _reshape_to_2d(t: torch.Tensor) -> Tuple[torch.Tensor, torch.Size]:
        """
        View any tensor as (m, n), where n is the last dimension, matching m×n in the paper.
        """
        orig_shape = t.shape
        if t.dim() == 1:
            t2d = t.view(1, -1)
        else:
            n = orig_shape[-1]
            m = t.numel() // n
            t2d = t.reshape(m, n)
        return t2d, orig_shape

    @staticmethod
    def _restore_from_2d(t2d: torch.Tensor, orig_shape: torch.Size) -> torch.Tensor:
        """
        Restore an (m, n) view to orig_shape.
        """
        return t2d.view(orig_shape)

    # ---------- FOAM: A^(ℓ) fold / compress ----------

    def _compress(self, tensor: torch.Tensor, fold_level: int) -> torch.Tensor:
        """
        Fold operator A^(ℓ): average each block_size = 2^ℓ adjacent elements along the last dimension.

        Given G_t ∈ R^{m×n}, return G̃_t ∈ R^{m×(n / 2^ℓ)}.
        If n is not divisible by 2^ℓ, pad zeros on the right.
        """
        block_size = 2 ** fold_level

        t2d, _ = self._reshape_to_2d(tensor)
        m, n = t2d.shape

        pad = (block_size - (n % block_size)) % block_size
        if pad > 0:
            t2d = torch.nn.functional.pad(t2d, (0, pad))
            n_padded = n + pad
        else:
            n_padded = n

        n_blocks = n_padded // block_size
        t_blocks = t2d.view(m, n_blocks, block_size)
        compressed = t_blocks.mean(dim=-1)  # (m, n_blocks)
        return compressed

    # ---------- FOAM: E^(ℓ) unfold / decompress ----------

    def _decompress(
        self,
        compressed: torch.Tensor,
        fold_level: int,
        original_shape: torch.Size,
    ) -> torch.Tensor:
        """
        Unfold operator E^(ℓ): repeat each compressed value block_size times along the last dimension,
        then crop back to the original length.

        Given M̃_t ∈ R^{m×(n/2^ℓ)}, return M̃_t E^(ℓ) ∈ R^{m×n}.
        """
        block_size = 2 ** fold_level
        n_orig = original_shape[-1]

        comp2d, _ = self._reshape_to_2d(compressed)
        m, n_blocks = comp2d.shape

        expanded = comp2d.unsqueeze(-1).expand(m, n_blocks, block_size)
        expanded = expanded.reshape(m, n_blocks * block_size)

        if expanded.size(1) != n_orig:
            expanded = expanded[:, :n_orig]

        return self._restore_from_2d(expanded, original_shape)

    # ---------- Main update: FOAM + AdamW ----------

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """
        Apply FOAM to foam_params and AdamW backup to adamw_params.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            # FOAM hyperparameters
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            fold_level = group["fold_level"]
            alpha = group["alpha"]

            # Shared weight decay
            lr = group["lr"]
            wd = group["wd"]
            foam_type = group['foam_type']
            # AdamW hyperparameters
            adamw_beta1, adamw_beta2 = group["adamw_betas"]
            adamw_eps = group["adamw_eps"]

            # -----------------
            # FOAM branch
            # -----------------
            foam_params = [
                p for p in group["params"] if self.state[p]["use_foam"]
            ]

            for p in foam_params:
                if p.grad is None:
                    continue
                g = p.grad
                if g.is_sparse:
                    raise RuntimeError("FOAM does not support sparse gradients.")

                # decoupled weight decay
                if wd > 0.0:
                    p.data.mul_(1 - lr * wd)

                # 1) Compress gradient G̃_t = G_t A^(ℓ)
                g_tilde = self._compress(g, fold_level)

                # 2) Residual R_t = G_t - G̃_t E^(ℓ)
                g_recon = self._decompress(g_tilde, fold_level, g.shape)
                residual = g - g_recon

                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    # state["exp_avg_compressed"] = torch.zeros_like(g_tilde)
                    # state["exp_avg_sq_compressed"] = torch.zeros_like(g_tilde)

                    # Record the last gradient.
                    state["last_grad"] = torch.zeros_like(g)

                state["step"] += 1
               
                update = torch.sign(residual)
                step_size = lr * alpha
                # p.data.addcdiv_(m_t, denom, value=-step_size)
                p.data.add_(update, value=-step_size)
                state['last_grad'].copy_(g)

            # -----------------------------
            # AdamW backup branch
            # -----------------------------
            adamw_params = [
                p for p in group["params"] if not self.state[p]["use_foam"]
            ]

            for p in adamw_params:
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

                # First/second moments
                buf1.lerp_(g, 1 - adamw_beta1)
                buf2.lerp_(g.square(), 1 - adamw_beta2)

                # Normalized gradient
                g_norm = buf1 / (adamw_eps + buf2.sqrt())

                # bias correction
                bias_correction1 = 1 - adamw_beta1 ** step
                bias_correction2 = 1 - adamw_beta2 ** step
                scale = bias_correction1 / math.sqrt(bias_correction2)

                # decoupled weight decay
                if wd > 0.0:
                    p.data.mul_(1 - lr * wd)

                # AdamW update
                p.data.add_(g_norm, alpha=-(lr / scale))

        return loss




class FOAM(Optimizer):
    """
    FOAM + AdamW hybrid optimizer.

    - Use standard FOAM for foam_params with folded first/second moments.
    - Use AdamW with bias correction as the backup for adamw_params.

    Typical usage, similar to Muon:

        foam_params = [
            p
            for name, p in model.named_parameters()
            if p.ndim >= 2
            and "embed_tokens" not in name
            and "lm_head" not in name
            and p.requires_grad
        ]
        adamw_params = [
            p
            for name, p in model.named_parameters()
            if not (
                p.ndim >= 2
                and "embed_tokens" not in name
                and "lm_head" not in name
                and p.requires_grad
            )
        ]

        optimizer = FOAM(
            lr=args.lr,
            wd=args.weight_decay,
            foam_params=foam_params,
            adamw_params=adamw_params,
            betas=args.foam_betas,        # FOAM (beta1, beta2)
            fold_level=args.fold_level,   # ℓ, block_size = 2**ℓ
            alpha=args.alpha,             # FOAM scaling factor
            eps=args.eps,                 # FOAM eps
            adamw_betas=args.betas,       # AdamW (beta1, beta2)
            adamw_eps=args.eps            # AdamW eps
        )
    """

    def __init__(
        self,
        lr: float,
        wd: float,
        foam_params: Iterable[nn.Parameter],
        adamw_params: Iterable[nn.Parameter],
        betas: Tuple[float, float] = (0.9, 0.999),      # FOAM betas
        fold_level: int = 1,                            # ℓ, block_size = 2**ℓ
        alpha: float = 1.0,                             # FOAM scaling
        eps: float = 1e-8,                              # FOAM eps
        adamw_betas: Optional[Tuple[float, float]] = None,
        adamw_eps: Optional[float] = None,
        foam_type: str = 'foam'
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if fold_level < 1:
            raise ValueError(f"fold_level must be >= 1, got {fold_level}")

        if adamw_betas is None:
            adamw_betas = betas
        if adamw_eps is None:
            adamw_eps = eps

        foam_params = list(foam_params)
        adamw_params = list(adamw_params)
        all_params = foam_params + adamw_params
        if len(all_params) == 0:
            raise ValueError("FOAM received no parameters.")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            fold_level=fold_level,
            alpha=alpha,
            wd=wd,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
            foam_type=foam_type
        )
        # Use one param_group; all hyperparameters come from defaults.
        super().__init__(all_params, defaults)

        foam_set = set(foam_params)
        # Mark whether each parameter uses FOAM in the state.
        for group in self.param_groups:
            for p in group["params"]:
                self.state[p]["use_foam"] = (p in foam_set)

    # ---------- General: arbitrary shape <-> (m, n) view ----------

    @staticmethod
    def _reshape_to_2d(t: torch.Tensor) -> Tuple[torch.Tensor, torch.Size]:
        """
        View any tensor as (m, n), where n is the last dimension, matching m×n in the paper.
        """
        orig_shape = t.shape
        if t.dim() == 1:
            t2d = t.view(1, -1)
        else:
            n = orig_shape[-1]
            m = t.numel() // n
            t2d = t.reshape(m, n)
        return t2d, orig_shape

    @staticmethod
    def _restore_from_2d(t2d: torch.Tensor, orig_shape: torch.Size) -> torch.Tensor:
        """
        Restore an (m, n) view to orig_shape.
        """
        return t2d.view(orig_shape)

    # ---------- FOAM: A^(ℓ) fold / compress ----------

    def _compress(self, tensor: torch.Tensor, fold_level: int) -> torch.Tensor:
        """
        Fold operator A^(ℓ): average each block_size = 2^ℓ adjacent elements along the last dimension.

        Given G_t ∈ R^{m×n}, return G̃_t ∈ R^{m×(n / 2^ℓ)}.
        If n is not divisible by 2^ℓ, pad zeros on the right.
        """
        block_size = 2 ** fold_level

        t2d, _ = self._reshape_to_2d(tensor)
        m, n = t2d.shape

        pad = (block_size - (n % block_size)) % block_size
        if pad > 0:
            t2d = torch.nn.functional.pad(t2d, (0, pad))
            n_padded = n + pad
        else:
            n_padded = n

        n_blocks = n_padded // block_size
        t_blocks = t2d.view(m, n_blocks, block_size)
        compressed = t_blocks.mean(dim=-1)  # (m, n_blocks)
        return compressed

    # ---------- FOAM: E^(ℓ) unfold / decompress ----------

    def _decompress(
        self,
        compressed: torch.Tensor,
        fold_level: int,
        original_shape: torch.Size,
    ) -> torch.Tensor:
        """
        Unfold operator E^(ℓ): repeat each compressed value block_size times along the last dimension,
        then crop back to the original length.

        Given M̃_t ∈ R^{m×(n/2^ℓ)}, return M̃_t E^(ℓ) ∈ R^{m×n}.
        """
        block_size = 2 ** fold_level
        n_orig = original_shape[-1]

        comp2d, _ = self._reshape_to_2d(compressed)
        m, n_blocks = comp2d.shape

        expanded = comp2d.unsqueeze(-1).expand(m, n_blocks, block_size)
        expanded = expanded.reshape(m, n_blocks * block_size)

        if expanded.size(1) != n_orig:
            expanded = expanded[:, :n_orig]

        return self._restore_from_2d(expanded, original_shape)

    # ---------- Main update: FOAM + AdamW ----------

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """
        Apply FOAM to foam_params and AdamW backup to adamw_params.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            # FOAM hyperparameters
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            fold_level = group["fold_level"]
            alpha = group["alpha"]

            # Shared weight decay
            lr = group["lr"]
            wd = group["wd"]
            foam_type = group['foam_type']
            # AdamW hyperparameters
            adamw_beta1, adamw_beta2 = group["adamw_betas"]
            adamw_eps = group["adamw_eps"]

            # -----------------
            # FOAM branch
            # -----------------
            foam_params = [
                p for p in group["params"] if self.state[p]["use_foam"]
            ]

            for p in foam_params:
                if p.grad is None:
                    continue
                g = p.grad
                if g.is_sparse:
                    raise RuntimeError("FOAM does not support sparse gradients.")

                # decoupled weight decay
                if wd > 0.0:
                    p.data.mul_(1 - lr * wd)

                # 1) Compress gradient G̃_t = G_t A^(ℓ)
                g_tilde = self._compress(g, fold_level)

                # 2) Residual R_t = G_t - G̃_t E^(ℓ)
                g_recon = self._decompress(g_tilde, fold_level, g.shape)
                residual = g - g_recon

                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["exp_avg_compressed"] = torch.zeros_like(g_tilde)
                    state["exp_avg_sq_compressed"] = torch.zeros_like(g_tilde)

                    # Record the last gradient.
                    state["last_grad"] = torch.zeros_like(g)

                state["step"] += 1
                m_tilde = state["exp_avg_compressed"]
                v_tilde = state["exp_avg_sq_compressed"]

                # 3) Update first/second moments in compressed space
                m_tilde.mul_(beta1).add_(g_tilde, alpha=(1.0 - beta1))
                v_tilde.mul_(beta2).addcmul_(g_tilde, g_tilde, value=(1.0 - beta2))

                
                step_size = lr * alpha
                if foam_type == 'foam':
                    # 4) Decompress and add the residual
                    m_t = self._decompress(m_tilde, fold_level, g.shape) + residual
                    v_t = self._decompress(v_tilde, fold_level, g.shape) + residual.pow(2)
                    # 5) FOAM update
                    denom = v_t.sqrt().add_(eps)
                    p.data.addcdiv_(m_t, denom, value=-step_size)
                elif foam_type == 'foam_m':
                    m_t = self._decompress(m_tilde, fold_level, g.shape) + residual
                    v_t = self._decompress(v_tilde, fold_level, g.shape)
                    denom = v_t.sqrt().add_(eps)
                    p.data.addcdiv_(m_t, denom, value=-step_size)
                elif foam_type == 'foam_v':
                    m_t = self._decompress(m_tilde, fold_level, g.shape) 
                    v_t = self._decompress(v_tilde, fold_level, g.shape) + residual.pow(2)
                    denom = v_t.sqrt().add_(eps)
                    p.data.addcdiv_(m_t, denom, value=-step_size)
                elif foam_type == 'foam_mr':
                    v_t = self._decompress(v_tilde, fold_level, g.shape) + residual.pow(2)
                    denom = v_t.sqrt().add_(eps)
                    p.data.addcdiv_(residual, denom, value=-step_size)
                elif foam_type == 'foam_vr':
                    m_t = self._decompress(m_tilde, fold_level, g.shape) + residual
                    v_t = residual.pow(2)
                    denom = v_t.sqrt().add_(eps)
                    p.data.addcdiv_(m_t, denom, value=-step_size)



                # update = torch.sign(residual)
                # p.data.add_(update,alpha=-step_size)


                state['last_grad'].copy_(g)

            # -----------------------------
            # AdamW backup branch
            # -----------------------------
            adamw_params = [
                p for p in group["params"] if not self.state[p]["use_foam"]
            ]

            for p in adamw_params:
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

                # First/second moments
                buf1.lerp_(g, 1 - adamw_beta1)
                buf2.lerp_(g.square(), 1 - adamw_beta2)

                # Normalized gradient
                g_norm = buf1 / (adamw_eps + buf2.sqrt())

                # bias correction
                bias_correction1 = 1 - adamw_beta1 ** step
                bias_correction2 = 1 - adamw_beta2 ** step
                scale = bias_correction1 / math.sqrt(bias_correction2)

                # decoupled weight decay
                if wd > 0.0:
                    p.data.mul_(1 - lr * wd)

                # AdamW update
                p.data.add_(g_norm, alpha=-(lr / scale))

        return loss



class SOAM(Optimizer):
    """
    SOAM (Shifted-Offset Approximate Moments) + AdamW hybrid optimizer.

    The interface stays consistent with FOAM:
    - foam_params use SOAM: cyclically shift gradients each step before fold/unfold, then inject residuals.
      First/second moments are still stored in compressed space after folding.
    - adamw_params use AdamW with bias correction as the backup.

    Key idea:
    - Cyclically shift each parameter gradient along the last dimension, with shift rolling over block_size=2**fold_level:
        shift_t = (step-1) % block_size
      This is equivalent to moving block boundaries over time and reduces fixed-block boundary artifacts.

    Hyperparameter meanings are the same as in FOAM.
    """

    def __init__(
        self,
        lr: float,
        wd: float,
        foam_params: Iterable[nn.Parameter],
        adamw_params: Iterable[nn.Parameter],
        betas: Tuple[float, float] = (0.9, 0.999),      # SOAM/FOAM betas
        fold_level: int = 1,                            # ℓ, block_size = 2**ℓ
        alpha: float = 1.0,                             # SOAM scaling
        eps: float = 1e-8,                              # SOAM eps
        adamw_betas: Optional[Tuple[float, float]] = None,
        adamw_eps: Optional[float] = None,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if fold_level < 1:
            raise ValueError(f"fold_level must be >= 1, got {fold_level}")

        if adamw_betas is None:
            adamw_betas = betas
        if adamw_eps is None:
            adamw_eps = eps

        foam_params = list(foam_params)
        adamw_params = list(adamw_params)
        all_params = foam_params + adamw_params
        if len(all_params) == 0:
            raise ValueError("SOAM received no parameters.")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            fold_level=fold_level,
            alpha=alpha,
            wd=wd,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
        )
        super().__init__(all_params, defaults)

        foam_set = set(foam_params)
        for group in self.param_groups:
            for p in group["params"]:
                self.state[p]["use_foam"] = (p in foam_set)

    # ---------- General: arbitrary shape <-> (m, n) view ----------

    @staticmethod
    def _reshape_to_2d(t: torch.Tensor) -> Tuple[torch.Tensor, torch.Size]:
        """
        View any tensor as (m, n), where n is the last dimension.
        """
        orig_shape = t.shape
        if t.dim() == 1:
            t2d = t.view(1, -1)
        else:
            n = orig_shape[-1]
            m = t.numel() // n
            t2d = t.reshape(m, n)
        return t2d, orig_shape

    @staticmethod
    def _restore_from_2d(t2d: torch.Tensor, orig_shape: torch.Size) -> torch.Tensor:
        """
        Restore an (m, n) view to orig_shape.
        """
        return t2d.view(orig_shape)

    # ---------- Fixed FOAM: A^(ℓ) fold / compress ----------

    def _compress(self, tensor: torch.Tensor, fold_level: int) -> torch.Tensor:
        """
        Fold operator A^(ℓ): average each block_size = 2^ℓ adjacent elements along the last dimension.

        Given G_t ∈ R^{m×n}, return G̃_t ∈ R^{m×(n / 2^ℓ)}.
        If n is not divisible by 2^ℓ, pad zeros on the right.
        """
        block_size = 2 ** fold_level

        t2d, _ = self._reshape_to_2d(tensor)
        m, n = t2d.shape

        pad = (block_size - (n % block_size)) % block_size
        if pad > 0:
            t2d = torch.nn.functional.pad(t2d, (0, pad))
            n_padded = n + pad
        else:
            n_padded = n

        n_blocks = n_padded // block_size
        t_blocks = t2d.view(m, n_blocks, block_size)
        compressed = t_blocks.mean(dim=-1)  # (m, n_blocks)
        return compressed

    # ---------- Fixed FOAM: E^(ℓ) unfold / decompress ----------

    def _decompress(
        self,
        compressed: torch.Tensor,
        fold_level: int,
        original_shape: torch.Size,
    ) -> torch.Tensor:
        """
        Unfold operator E^(ℓ): repeat each compressed value block_size times along the last dimension,
        then crop back to the original length.
        """
        block_size = 2 ** fold_level
        n_orig = original_shape[-1]

        comp2d, _ = self._reshape_to_2d(compressed)
        m, n_blocks = comp2d.shape

        expanded = comp2d.unsqueeze(-1).expand(m, n_blocks, block_size)
        expanded = expanded.reshape(m, n_blocks * block_size)

        if expanded.size(1) != n_orig:
            expanded = expanded[:, :n_orig]

        return self._restore_from_2d(expanded, original_shape)

    # ---------- SOAM: cyclic shift along the last dimension ----------

    @staticmethod
    def _roll_last_dim(t: torch.Tensor, shift: int) -> torch.Tensor:
        if shift == 0:
            return t
        return torch.roll(t, shifts=shift, dims=-1)

    @staticmethod
    def _compute_shift(step: int, fold_level: int, n_last: int) -> int:
        """
        shift_t = (step-1) % block_size, where block_size = 2**fold_level.
        Apply another % n_last to avoid an oversized shift when n_last < block_size.
        """
        block_size = 2 ** fold_level
        s = (step - 1) % block_size
        if n_last > 0:
            s = s % n_last
        return s

    # ---------- Main update: SOAM + AdamW ----------

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            # SOAM/FOAM hyperparameters
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            fold_level = group["fold_level"]
            alpha = group["alpha"]

            # Shared weight decay
            lr = group["lr"]
            wd = group["wd"]

            # AdamW hyperparameters
            adamw_beta1, adamw_beta2 = group["adamw_betas"]
            adamw_eps = group["adamw_eps"]

            # -----------------
            # SOAM branch for foam_params
            # -----------------
            foam_params = [p for p in group["params"] if self.state[p]["use_foam"]]

            for p in foam_params:
                if p.grad is None:
                    continue
                g = p.grad
                if g.is_sparse:
                    raise RuntimeError("SOAM does not support sparse gradients.")

                # decoupled weight decay
                if wd > 0.0:
                    p.data.mul_(1 - lr * wd)

                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0  # Parameter-level step count, consistent with original FOAM.

                # Increment step first, then compute shift from it.
                state["step"] += 1
                step_i = state["step"]

                # Compute shift, rolling with block_size as the period.
                n_last = g.shape[-1] if g.dim() > 0 else 0
                shift = self._compute_shift(step_i, fold_level, n_last)

                # 1) Shifted gradient
                g_shift = self._roll_last_dim(g, shift)

                # 2) Compressed gradient G̃_t = C_t(g_t) = C_0(shift(g_t))
                g_tilde = self._compress(g_shift, fold_level)

                # Initialize compressed moment states.
                if "exp_avg_compressed" not in state:
                    state["exp_avg_compressed"] = torch.zeros_like(g_tilde)
                    state["exp_avg_sq_compressed"] = torch.zeros_like(g_tilde)

                m_tilde = state["exp_avg_compressed"]
                v_tilde = state["exp_avg_sq_compressed"]

                # 3) Update first/second moments in compressed space, same as FOAM.
                m_tilde.mul_(beta1).add_(g_tilde, alpha=(1.0 - beta1))
                v_tilde.mul_(beta2).addcmul_(g_tilde, g_tilde, value=(1.0 - beta2))

                # 4) Decompress in shifted coordinates, shift back, then compute residual.
                g_recon_shift = self._decompress(g_tilde, fold_level, g.shape)
                g_recon = self._roll_last_dim(g_recon_shift, -shift)
                residual = g - g_recon

                # 5) Decode momentum/variance in shifted coordinates, shift back, and add residuals.
                m_shift = self._decompress(m_tilde, fold_level, g.shape)
                v_shift = self._decompress(v_tilde, fold_level, g.shape)

                m_t = self._roll_last_dim(m_shift, -shift) + residual
                v_t = self._roll_last_dim(v_shift, -shift) + residual.pow(2)

                # 6) SOAM update, same form as FOAM.
                denom = v_t.sqrt().add_(eps)
                step_size = lr * alpha
                p.data.addcdiv_(m_t, denom, value=-step_size)

            # -----------------------------
            # AdamW backup branch, unchanged.
            # -----------------------------
            adamw_params = [p for p in group["params"] if not self.state[p]["use_foam"]]

            for p in adamw_params:
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

                # First/second moments
                buf1.lerp_(g, 1 - adamw_beta1)
                buf2.lerp_(g.square(), 1 - adamw_beta2)

                # Normalized gradient
                g_norm = buf1 / (adamw_eps + buf2.sqrt())

                # bias correction
                bias_correction1 = 1 - adamw_beta1 ** step
                bias_correction2 = 1 - adamw_beta2 ** step
                scale = bias_correction1 / math.sqrt(bias_correction2)

                # decoupled weight decay
                if wd > 0.0:
                    p.data.mul_(1 - lr * wd)

                # AdamW update
                p.data.add_(g_norm, alpha=-(lr / scale))

        return loss


## FOAMuon



# -------- Muon Newton-Schulz orthogonalization --------

def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int) -> torch.Tensor:
    """
    Same zeropower_via_newtonschulz5 implementation as original Muon:
    - Input G is a 2D matrix, usually a momentum-adjusted gradient.
    - The iterated output is close to orthogonal in spectral norm.
    """
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)

    # Use bfloat16 in Muon to save memory while maintaining numerical stability.
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T

    # Ensure spectral norm <= 1.
    X = X / (X.norm() + 1e-7)

    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)  # Quintic iteration.
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T
    return X


class FOAMuon(Optimizer):
    """
    FOAMuon: FOAM + Muon + AdamW hybrid optimizer.

    - For foam_params:
        * Use FOAM folded first momentum, compressing only M and not maintaining V.
        * Use the FOAM-decompressed momentum with residuals as Muon's momentum gradient.
        * Optional Nesterov: direction = g + beta1 * m_full.
        * Use zeropower_via_newtonschulz5 for orthogonalization so the update direction has spectral norm 1.
        * Use Muon's size-adaptive LR, 0.2 * sqrt(max(A,B)), as the step size.
        * Use decoupled weight decay, p *= 1 - lr * wd.

    - For adamw_params:
        * Use AdamW with bias correction as the backup.

    Important constraint:
        - Parameters in foam_params must be 2D, as in Muon.

    Typical usage:

        foam_params = [
            p
            for name, p in model.named_parameters()
            if p.ndim == 2
            and "embed_tokens" not in name
            and "lm_head" not in name
            and p.requires_grad
        ]
        adamw_params = [
            p
            for name, p in model.named_parameters()
            if p not in foam_params
        ]

        optimizer = FOAMuon(
            lr=args.lr,
            wd=args.weight_decay,
            foam_params=foam_params,
            adamw_params=adamw_params,
            betas=args.foam_betas,        # FOAM (beta1, beta2); only beta1 is used as momentum here
            fold_level=args.fold_level,   # ℓ, block_size = 2**ℓ
            alpha=args.alpha,             # Kept for interface compatibility; not used by FOAMuon's Muon branch
            eps=args.eps,                 # Not used by the FOAM branch; only AdamW uses adamw_eps
            adamw_betas=args.betas,       # AdamW (beta1, beta2)
            adamw_eps=args.eps,           # AdamW eps
            nesterov=True,                # Whether to use Nesterov
            ns_steps=5,                   # Number of Newton-Schulz iterations
        )
    """

    def __init__(
        self,
        lr: float,
        wd: float,
        foam_params: Iterable[nn.Parameter],
        adamw_params: Iterable[nn.Parameter],
        betas: Tuple[float, float] = (0.9, 0.999),      # FOAM emphasizes beta1 (momentum); beta2 is unused in this class.
        fold_level: int = 1,                            # ℓ, block_size = 2**ℓ
        alpha: float = 1.0,                             # Kept for interface compatibility; not used by the Muon update itself.
        eps: float = 1e-8,                              # Kept for interface compatibility; FOAM first-moment mode does not use eps.
        adamw_betas: Optional[Tuple[float, float]] = None,
        adamw_eps: Optional[float] = None,
        nesterov: bool = True,
        ns_steps: int = 5,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if fold_level < 1:
            raise ValueError(f"fold_level must be >= 1, got {fold_level}")

        if adamw_betas is None:
            adamw_betas = betas
        if adamw_eps is None:
            adamw_eps = eps

        foam_params = list(foam_params)
        adamw_params = list(adamw_params)
        all_params = foam_params + adamw_params
        if len(all_params) == 0:
            raise ValueError("FOAMuon received no parameters.")

        defaults = dict(
            lr=lr,
            wd=wd,
            betas=betas,
            eps=eps,
            fold_level=fold_level,
            alpha=alpha,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
            nesterov=nesterov,
            ns_steps=ns_steps,
        )
        super().__init__(all_params, defaults)

        # Use id to mark which parameters take the FOAM+Muon path and avoid Tensor __eq__ issues.
        foam_ids = {id(p) for p in foam_params}
        for group in self.param_groups:
            for p in group["params"]:
                use_foam = id(p) in foam_ids
                self.state[p]["use_foam"] = use_foam
                if use_foam and p.ndim != 2:
                    raise ValueError(
                        f"FOAMuon expects foam_params to be 2D tensors, got ndim={p.ndim}"
                    )

    # ---------- FOAM A^(ℓ) / E^(ℓ) implementation, used only to fold first momentum ----------

    @staticmethod
    def _reshape_to_2d(t: torch.Tensor):
        orig_shape = t.shape
        if t.dim() == 1:
            t2d = t.view(1, -1)
        else:
            n = orig_shape[-1]
            m = t.numel() // n
            t2d = t.reshape(m, n)
        return t2d, orig_shape

    @staticmethod
    def _restore_from_2d(t2d: torch.Tensor, orig_shape: torch.Size) -> torch.Tensor:
        return t2d.view(orig_shape)

    def _compress(self, tensor: torch.Tensor, fold_level: int) -> torch.Tensor:
        """
        A^(ℓ): average each block_size = 2^ℓ adjacent elements along the last dimension.
        """
        block_size = 2 ** fold_level
        t2d, _ = self._reshape_to_2d(tensor)
        m, n = t2d.shape

        pad = (block_size - (n % block_size)) % block_size
        if pad > 0:
            t2d = torch.nn.functional.pad(t2d, (0, pad))
            n_padded = n + pad
        else:
            n_padded = n

        n_blocks = n_padded // block_size
        t_blocks = t2d.view(m, n_blocks, block_size)
        compressed = t_blocks.mean(dim=-1)  # (m, n_blocks)
        return compressed

    def _decompress(
        self,
        compressed: torch.Tensor,
        fold_level: int,
        original_shape: torch.Size,
    ) -> torch.Tensor:
        """
        E^(ℓ): repeat each compressed value block_size times along the last dimension, then crop back to the original length.
        """
        block_size = 2 ** fold_level
        n_orig = original_shape[-1]

        comp2d, _ = self._reshape_to_2d(compressed)
        m, n_blocks = comp2d.shape

        expanded = comp2d.unsqueeze(-1).expand(m, n_blocks, block_size)
        expanded = expanded.reshape(m, n_blocks * block_size)

        if expanded.size(1) != n_orig:
            expanded = expanded[:, :n_orig]

        return self._restore_from_2d(expanded, original_shape)

    # ---------- Muon LR adjustment ----------

    @staticmethod
    def adjust_lr_for_muon(lr: float, param_shape: torch.Size) -> float:
        """
        Scale LR by matrix size according to the Muon paper:
            lr_eff = lr * (0.2 * sqrt(max(A, B)))
        """
        if len(param_shape) < 2:
            return lr
        A, B = param_shape[:2]
        adjusted_ratio = 0.2 * math.sqrt(max(A, B))
        return lr * adjusted_ratio

    # ---------- Main step: FOAM + Muon + AdamW ----------

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            wd = group["wd"]

            beta1, _beta2 = group["betas"]      # Only beta1 is used as the FOAM / Muon momentum coefficient.
            fold_level = group["fold_level"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]

            adamw_beta1, adamw_beta2 = group["adamw_betas"]
            adamw_eps = group["adamw_eps"]

            # -----------------
            # FOAM + Muon branch
            # -----------------
            foam_params = [p for p in group["params"] if self.state[p]["use_foam"]]

            for p in foam_params:
                g = p.grad
                if g is None:
                    continue
                if g.is_sparse:
                    raise RuntimeError("FOAMuon does not support sparse gradients.")
                if g.ndim != 2:
                    # foam_params are already restricted to 2D; this is only a double-check.
                    g = g.view(g.size(0), -1)

                # Decoupled weight decay, matching Muon and using the original LR.
                if wd > 0.0:
                    p.data.mul_(1 - lr * wd)

                # FOAM: compress the current gradient.
                g_tilde = self._compress(g, fold_level)
                g_recon = self._decompress(g_tilde, fold_level, g.shape)
                residual = g - g_recon

                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    # Folded first momentum M̃_0 = 0.
                    state["exp_avg_compressed"] = torch.zeros_like(g_tilde)

                state["step"] += 1
                m_tilde = state["exp_avg_compressed"]

                # FOAM momentum update: M̃_t = beta1 * M̃_{t-1} + (1 - beta1) * G̃_t
                # m_tilde.mul_(beta1).add_(g_tilde, alpha=(1.0 - beta1))
                m_tilde.mul_(beta1).add_(g_tilde)


                # Decompress and add residual: M_t ≈ M̃_t E^(ℓ) + R_t
                m_full = self._decompress(m_tilde, fold_level, g.shape) + residual

                # Nesterov: direction = g + beta1 * m_full; otherwise direction = m_full.
                if nesterov:
                    direction = g + beta1 * m_full
                else:
                    direction = m_full

                # Muon: apply Newton-Schulz orthogonalization to direction.
                u = zeropower_via_newtonschulz5(direction, steps=ns_steps)

                # Size-adaptive LR.
                adjusted_lr = self.adjust_lr_for_muon(lr, p.shape)

                # Apply update; spectral norm is approximately adjusted_lr.
                p.data.add_(u, alpha=-adjusted_lr)

            # -----------------
            # AdamW backup branch
            # -----------------
            adamw_params = [p for p in group["params"] if not self.state[p]["use_foam"]]

            for p in adamw_params:
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

                # Standard Adam first/second moments.
                buf1.lerp_(g, 1 - adamw_beta1)
                buf2.lerp_(g.square(), 1 - adamw_beta2)

                g_norm = buf1 / (adamw_eps + buf2.sqrt())

                # bias correction
                bias_correction1 = 1 - adamw_beta1 ** step
                bias_correction2 = 1 - adamw_beta2 ** step
                scale = bias_correction1 / math.sqrt(bias_correction2)

                # decoupled weight decay
                if wd > 0.0:
                    p.data.mul_(1 - lr * wd)

                # AdamW update
                p.data.add_(g_norm, alpha=-(lr / scale))

        return loss




class LP_FOAM(Optimizer):
    """
    LP-FOAM optimizer based on numerical quantization with low-precision mapping.
    
    Mathematical modeling:
    1. g_lp = Quantize(g)  [corresponds to g_t^c in the theory]
    2. g_recon = Dequantize(g_lp) [corresponds to g_t^cd in the theory]
    3. r_t = g - g_recon (quantization residual) [corresponds to r_t in the theory, cite: 3]
    4. Store low-precision m_lp and v_lp to save memory.
    5. Apply error feedback during update: m_t = Dequantize(m_lp) + r_t [cite: 4]
    """

    def __init__(
        self,
        lr: float,
        wd: float,
        foam_params: Iterable[nn.Parameter],
        adamw_params: Iterable[nn.Parameter],
        betas: Tuple[float, float] = (0.9, 0.999),
        lp_dtype: torch.dtype = torch.bfloat16,  # Low-precision dtype for storage.
        alpha: float = 1.0,
        eps: float = 1e-8,
        adamw_betas: Optional[Tuple[float, float]] = None,
        adamw_eps: Optional[float] = None,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        
        if adamw_betas is None:
            adamw_betas = betas
        if adamw_eps is None:
            adamw_eps = eps

        foam_params = list(foam_params)
        adamw_params = list(adamw_params)
        all_params = foam_params + adamw_params
        
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            lp_dtype=lp_dtype,
            alpha=alpha,
            wd=wd,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
        )
        super().__init__(all_params, defaults)

        foam_set = set(foam_params)
        for group in self.param_groups:
            for p in group["params"]:
                self.state[p]["use_foam"] = (p in foam_set)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            lp_dtype = group["lp_dtype"]
            alpha = group["alpha"]
            lr = group["lr"]
            wd = group["wd"]

            # -----------------
            # LP-FOAM branch with low-precision storage and error feedback.
            # -----------------
            foam_params = [p for p in group["params"] if self.state[p]["use_foam"]]

            for p in foam_params:
                if p.grad is None:
                    continue
                g = p.grad.float() # Keep high precision for residual computation.
                
                # 1. Numerical quantization mapping as a low-precision projection.
                # Simulate C(g) [cite: 3].
                g_lp = g.to(lp_dtype)
                
                # 2. Compute residual r_t = g - D(C(g)).
                # Theory correspondence: r_t = (I - P)g [cite: 47, 48].
                g_recon = g_lp.to(g.dtype)
                residual = g - g_recon

                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    # Store only low-precision momentum to save 50% memory.
                    state["exp_avg_lp"] = torch.zeros_like(g_lp)
                    state["exp_avg_sq_lp"] = torch.zeros_like(g_lp)

                state["step"] += 1
                m_lp = state["exp_avg_lp"]
                v_lp = state["exp_avg_sq_lp"]

                # 3. Accumulate momentum in low-precision space [cite: 3].
                # n_t^c = beta1 * m_{t-1}^c + (1-beta1) * g_t^c
                m_lp.mul_(beta1).add_(g_lp, alpha=(1.0 - beta1))
                v_lp.mul_(beta2).addcmul_(g_lp, g_lp, value=(1.0 - beta2))

                # 4. Error-feedback reconstruction.
                # m_t^cd = D(m_t^c) + r_t [cite: 4]
                # v_t^cd = D(v_t^c) + r_t^2 [cite: 5, 103]
                m_t = m_lp.to(g.dtype) + residual
                v_t = v_lp.to(g.dtype) + residual.pow(2)

                # 5. Apply update [cite: 7, 106].
                if wd > 0.0:
                    p.data.mul_(1 - lr * wd)
                
                denom = v_t.sqrt().add_(eps)
                p.data.addcdiv_(m_t, denom, value=-(lr * alpha))

            # -----------------------------
            # AdamW backup for non-matrix parameters.
            # -----------------------------
            adamw_beta1, adamw_beta2 = group["adamw_betas"]
            adamw_eps = group["adamw_eps"]
            adamw_params = [p for p in group["params"] if not self.state[p]["use_foam"]]

            for p in adamw_params:
                if p.grad is None: continue
                g = p.grad
                
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)

                state["step"] += 1
                s = state["step"]
                m, v = state["moment1"], state["moment2"]

                m.lerp_(g, 1 - adamw_beta1)
                v.lerp_(g.square(), 1 - adamw_beta2)

                # Bias correction
                bc1 = 1 - adamw_beta1 ** s
                bc2 = 1 - adamw_beta2 ** s
                
                if wd > 0.0:
                    p.data.mul_(1 - lr * wd)
                
                update = (m / bc1) / (v.sqrt() / math.sqrt(bc2) + adamw_eps)
                p.data.add_(update, alpha=-lr)

        return loss




class LOAM(Optimizer):
    """
    LOAM: Low-precision Optimizer with Adam-style Momentum
    Implements INT8-level momentum storage, block-wise dynamic quantization, and error-feedback compensation.
    Memory use: about 2 bytes per parameter for optimizer state.
    """
    def __init__(
        self,
        params: Iterable[nn.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        block_size: int = 2048, # Block size for block-wise quantization.
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, block_size=block_size)
        super().__init__(params, defaults)

    def _quantize_blockwise(self, tensor: torch.Tensor, block_size: int, is_positive: bool = False):
        """Quantize a tensor block-wise into int8/uint8."""
        original_shape = tensor.shape
        flat_tensor = tensor.view(-1)
        # Pad to a multiple of block_size.
        num_elements = flat_tensor.numel()
        padding_size = (block_size - num_elements % block_size) % block_size
        if padding_size > 0:
            flat_tensor = torch.cat([flat_tensor, torch.zeros(padding_size, device=tensor.device)])
        
        reshaped = flat_tensor.view(-1, block_size)
        
        if is_positive:
            # Use uint8 (0-255) for second moments.
            scales = reshaped.max(dim=1, keepdim=True)[0] / 255.0
            scales = scales.clamp(min=1e-12)
            quantized = (reshaped / scales).round().clamp(0, 255).to(torch.uint8)
        else:
            # Use int8 (-128-127) for first moments.
            scales = reshaped.abs().max(dim=1, keepdim=True)[0] / 127.0
            scales = scales.clamp(min=1e-12)
            quantized = (reshaped / scales).round().clamp(-128, 127).to(torch.int8)
            
        return quantized, scales, padding_size, original_shape

    def _dequantize_blockwise(self, quantized, scales, padding_size, original_shape):
        """Restore float32 values from the quantized format."""
        dequantized = (quantized.float() * scales).view(-1)
        if padding_size > 0:
            dequantized = dequantized[:-padding_size]
        return dequantized.view(original_shape)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            lr = group["lr"]
            wd = group["weight_decay"]
            block_size = group["block_size"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                state = self.state[p]

                # 1. Initialize state.
                if len(state) == 0:
                    state["step"] = 0
                    # Store int8/uint8 tensors and scales.
                    m_q, m_s, m_p, _ = self._quantize_blockwise(torch.zeros_like(grad), block_size)
                    # v_q, v_s, v_p, _ = self._quantize_blockwise(torch.zeros_like(grad), block_size, is_positive=True)
                    state["m_q"], state["m_s"], state["m_p"] = m_q, m_s, m_p
                    # state["v_q"], state["v_s"], state["v_p"] = v_q, v_s, v_p
                    state["v"] = torch.zeros_like(grad)

                state["step"] += 1
                
                # 2. Compute quantization error feedback.
                # Simulate the quantization loss of the current gradient: r = g - D(C(g)).
                g_q, g_s, g_p, _ = self._quantize_blockwise(grad, block_size)
                g_recon = self._dequantize_blockwise(g_q, g_s, g_p, grad.shape)
                residual = grad - g_recon

                # 3. Dequantize momentum for computation as FP32 in registers/temporary memory.
                m = self._dequantize_blockwise(state["m_q"], state["m_s"], state["m_p"], grad.shape)
                # v = self._dequantize_blockwise(state["v_q"], state["v_s"], state["v_p"], grad.shape)
                v = state['v']
                # 4. Update momentum with FP32 computation.
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # 5. Re-quantize and store.
                state["m_q"], state["m_s"], _, _ = self._quantize_blockwise(m, block_size)
                # state["v_q"], state["v_s"], _, _ = self._quantize_blockwise(v, block_size, is_positive=True)

                # 6. Bias correction and error compensation.
                # Add residual during the update so discarded precision can affect parameter updates.
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                
                # m_t and v_t include current-step error compensation.
                m_t = m / bias_correction1 + residual
                v_t = v / bias_correction2 + residual.pow(2)

                # 7. Parameter update.
                if wd != 0:
                    p.data.mul_(1 - lr * wd)

                denom = v_t.sqrt().add_(eps)
                p.data.addcdiv_(m_t, denom, value=-lr)

        return loss
