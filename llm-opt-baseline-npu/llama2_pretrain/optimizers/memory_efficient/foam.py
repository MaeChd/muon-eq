import math
from typing import Iterable, Tuple, Callable, Optional

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer


class FOAM(Optimizer):
    """
    Hybrid FOAM + AdamW optimizer.

    - Use standard FOAM for foam_params with folded first/second moments.
    - Use AdamW with bias correction as the backup path for adamw_params.

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
            fold_level=args.fold_level,   # ell, block_size = 2**ell
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
        fold_level: int = 1,                            # ell, block_size = 2**ell
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
        # Use a single param_group; all hyperparameters come from defaults.
        super().__init__(all_params, defaults)

        foam_set = set(foam_params)
        # Mark whether each parameter uses FOAM in the optimizer state.
        for group in self.param_groups:
            for p in group["params"]:
                self.state[p]["use_foam"] = (p in foam_set)

    # ---------- Utilities: arbitrary shape <-> (m, n) view ----------

    @staticmethod
    def _reshape_to_2d(t: torch.Tensor) -> Tuple[torch.Tensor, torch.Size]:
        """
        View an arbitrary tensor as (m, n), where n is the last dimension,
        matching the m x n notation in the paper.
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
        Restore an (m, n) view back to orig_shape.
        """
        return t2d.view(orig_shape)

    # ---------- FOAM: A^(ell) folding / compression ----------

    def _compress(self, tensor: torch.Tensor, fold_level: int) -> torch.Tensor:
        """
        Fold operator A^(ell): average adjacent elements along the last
        dimension with block_size = 2^ell.

        Given G_t in R^{m x n}, return G_tilde in R^{m x (n / 2^ell)}.
        If n is not divisible by 2^ell, pad zeros on the right.
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

    # ---------- FOAM: E^(ell) unfolding / decompression ----------

    def _decompress(
        self,
        compressed: torch.Tensor,
        fold_level: int,
        original_shape: torch.Size,
    ) -> torch.Tensor:
        """
        Unfold operator E^(ell): repeat each compressed value block_size
        times along the last dimension, then crop back to the original length.

        Given M_tilde in R^{m x (n / 2^ell)}, return
        M_tilde E^(ell) in R^{m x n}.
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
        Apply FOAM to foam_params and the AdamW backup path to adamw_params.
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
            # FOAM path
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

                # 1) Compress gradient: G_tilde = G_t A^(ell)
                g_tilde = self._compress(g, fold_level)

                # 2) Residual: R_t = G_t - G_tilde E^(ell)
                g_recon = self._decompress(g_tilde, fold_level, g.shape)
                residual = g - g_recon

                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["exp_avg_compressed"] = torch.zeros_like(g_tilde)
                    state["exp_avg_sq_compressed"] = torch.zeros_like(g_tilde)
                    state["exp_avg_sq"] = torch.zeros_like(g)


                    # Store the last gradient.
                    state["last_grad"] = torch.zeros_like(g)

                state["step"] += 1
                m_tilde = state["exp_avg_compressed"]
                v_tilde = state["exp_avg_sq_compressed"]

                # 3) Update first/second moments in the compressed space.
                m_tilde.mul_(beta1).add_(g_tilde, alpha=(1.0 - beta1))
                v_tilde.mul_(beta2).addcmul_(g_tilde, g_tilde, value=(1.0 - beta2))

                
                step_size = lr * alpha
                if foam_type == 'foam':
                    # 4) Decompress and add the residual.
                    m_t = self._decompress(m_tilde, fold_level, g.shape) + residual
                    v_t = self._decompress(v_tilde, fold_level, g.shape) + residual.pow(2)
                    # 5) FOAM update.
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
                elif foam_type == 'foam_va':
                    m_t = self._decompress(m_tilde, fold_level, g.shape) + residual
                    v_t = state['exp_avg_sq']
                    v_t.mul_(beta2).addcmul_(g, g, value=(1.0 - beta2))
                    denom = v_t.sqrt().add_(eps)
                    p.data.addcdiv_(m_t, denom, value=-step_size)



                # update = torch.sign(residual)
                # p.data.add_(update,alpha=-step_size)


                state['last_grad'].copy_(g)

            # -----------------------------
            # AdamW backup path
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
