'''
The official code of "Mano: Restriking Manifold Optimization for LLM Training"

References: https://github.com/MoonshotAI/Moonlight

This implementation of Mano closely replicated the MoonshotAI's implmentation style of Muon.
The Newton-Schulz iterations is replaced by a cheaper manifold normalization operation with enhanced performance.
'''

import math
import torch

class Mano(torch.optim.Optimizer):
    """
    Mano: Manifold Normalized Optimizer

    Arguments:
        mano_params: The parameters to be optimized by mano.
        lr: The learning rate. The updates will have spectral norm of `lr`, default as 1e-3.
        momentum: The momentum coefficient used by the internal SGD, default as 0.95.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD.
        adamw_params: The parameters to be optimized by AdamW. Any parameters in `mano_params` which are
            {0,1}-D or are detected as being the embed or lm_head will be optimized by AdamW as well.
        adamw_betas: The betas for the internal AdamW.
        adamw_eps: The epsilon for the internal AdamW.
    """

    def __init__(
        self,
        lr=1e-3,
        wd=0.1,
        eps=1e-8,
        mano_params=None,
        momentum=0.95,
        nesterov=False,
        adamw_params=None,
        adamw_betas=(0.9, 0.95),
        adamw_eps=1e-8
    ):
        defaults = dict(
            lr=lr,
            wd=wd,
            eps=eps,
            momentum=momentum,
            nesterov=nesterov,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
            steps=0,
        )

        params = list(mano_params)
        adamw_params = list(adamw_params) if adamw_params is not None else []
        params.extend(adamw_params)
        super().__init__(params, defaults)
        # Sort parameters into those for which we will use mano, and those for which we will not
        for p in mano_params:
            # Use mano for every parameter in mano_params which is >= 2D and doesn't look like an embedding or head layer
            assert p.ndim == 2, p.ndim
            self.state[p]["use_mano"] = True
        for p in adamw_params:
            # Do not use mano for parameters in adamw_params
            self.state[p]["use_mano"] = False
    
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:

            ############################
            #           Mano           #
            ############################

            params = [p for p in group["params"] if self.state[p]["use_mano"]]
            # import pdb; pdb.set_trace()
            lr = group["lr"]
            wd = group["wd"]
            eps = group["eps"]
            momentum = group["momentum"]

            # generate weight updates in distributed fashion
            for p in params:
                # sanity check
                g = p.grad
                if g is None:
                    continue
                if g.ndim > 2:
                    g = g.view(g.size(0), -1)
                assert g is not None
                
                # calc update
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                
                # Use Nesterov accelerated gradient
                g = g.add(buf, alpha=momentum) if group["nesterov"] else buf

                ################################################################################
                # 0. Rotating Dimension intermittently across each step
                dim = int(group["steps"] % 2) 
                
                # 1. Project the momentum onto the Tangent Space
                p_unit = p.data / torch.clamp(torch.norm(p.data, p=2, dim=dim, keepdim=True), min=eps)
                tangent_momentum = g - (torch.sum(g * p_unit, dim=dim, keepdim=True) * p_unit)
                
                # 2. Mapping to the Oblique Manifold via column/row-wise normalization.
                u = tangent_momentum / torch.clamp(torch.norm(tangent_momentum, p=2, dim=dim, keepdim=True), min=eps)
                ################################################################################

                # Apply weight decay
                p.data.mul_(1 - lr * wd)

                # Apply update via rescaling update RMS to 0.2
                adjusted_lr = lr * 0.2 * math.sqrt(g.shape[dim])
                p.data.add_(u, alpha=-adjusted_lr)

                group["steps"] += 1
                
            ############################
            #       AdamW backup       #
            ############################

            params = [p for p in group["params"] if not self.state[p]["use_mano"]]
            lr = group['lr']
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

                g = buf1 / (eps + buf2.sqrt())

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5
                p.data.mul_(1 - lr * weight_decay)
                
                step_size = lr / scale
                p.data.add_(g, alpha=-step_size)

        return loss