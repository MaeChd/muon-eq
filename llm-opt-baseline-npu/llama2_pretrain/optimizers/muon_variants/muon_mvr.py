import os
import math
import torch
from loguru import logger

# @torch.compile
def zeropower_via_newtonschulz5(G, steps):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T
    # Ensure spectral norm is at most 1
    X = X / (X.norm() + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.T
        B = (
            b * A + c * A @ A
        )  # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T
    return X


class MuonMVR(torch.optim.Optimizer):
    def __init__(self, 
                 muon_params=None, 
                 adamw_params=None,
                 lr=3e-3, 
                 momentum = 0.95 ,
                 adamw_betas=(0.95, 0.99), 
                 eps=1e-8, 
                 weight_decay=0.0, 
                 gamma=0.025, 
                 ns_steps = 5,
                 is_approx=False
        ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum parameter: {momentum}")
        if not 0.0 <= adamw_betas[0] < 1.0:
            raise ValueError(f"Invalid beta1 parameter: {adamw_betas[0]}")
        if not 0.0 <= adamw_betas[1] < 1.0:
            raise ValueError(f"Invalid beta2 parameter: {adamw_betas[1]}")
        
        defaults = dict(
            muon_params=muon_params,
            adamw_params=adamw_params,
            lr=lr, 
            momentum=momentum,
            adamw_betas=adamw_betas, 
            eps=eps,
            weight_decay=weight_decay, 
            gamma=gamma,
            ns_steps=ns_steps
        )
        params = []
        muon_params = list(muon_params) if muon_params is not None else []
        adamw_params = list(adamw_params) if adamw_params is not None else []
        params.extend(muon_params)
        params.extend(adamw_params)

        super().__init__(params, defaults)
        self.is_approx = is_approx

         # Sort parameters into those for which we will use Muon, and those for which we will not
        for p in muon_params:
            # Use Muon for every parameter in muon_params which is >= 2D and doesn't look like an embedding or head layer
            assert p.ndim == 2, p.ndim
            self.state[p]["use_muon"] = True
        for p in adamw_params:
            # Do not use Muon for parameters in adamw_params
            self.state[p]["use_muon"] = False

    
    def adjust_lr_for_muon(self, lr, param_shape):
        A, B = param_shape[:2]
        # We adjust the learning rate and weight decay based on the size of the parameter matrix
        # as describted in the paper
        adjusted_ratio = 0.2 * math.sqrt(max(A, B))
        # adjusted_ratio = math.sqrt(A*B)
        adjusted_lr = lr * adjusted_ratio
        return adjusted_lr
    
    @torch.no_grad()
    def update_last_grad(self):
        if not self.is_approx:
            for group in self.param_groups:
                params = [p for p in group["params"] if self.state[p]["use_muon"]]
                for p in params:
                    state = self.state[p]
                    # if "last_grad" not in state:
                    #     state["last_grad"] = torch.zeros_like(p)
                    if p.grad is not None:
                        state["last_grad"].zero_().add_(p.grad, alpha=1.0)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['adamw_betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            gamma = group['gamma']

            ############################
            #           Muon           #
            ############################

            params = [p for p in group["params"] if self.state[p]["use_muon"]]

            for p in params:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                # State initialization
                if "step" not in state:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['last_grad'] = torch.zeros_like(p)
            
                state['step'] += 1
                last_grad = state['last_grad']
                exp_avg = state['exp_avg']
                
                # Compute momentum-like term with correction
                c_t = (grad - last_grad).mul(gamma * (momentum / (1. - momentum))).add(grad)
                #### MARS grad clip ######
                c_t_norm = torch.norm(c_t)
                if c_t_norm > 1.:
                    c_t = c_t / c_t_norm

                #########################
                # Update moving averages
                exp_avg.mul_(momentum).add_(c_t, alpha=1 - momentum)
                update = zeropower_via_newtonschulz5(exp_avg.mul(1./(1.- momentum)),steps=group['ns_steps']) # whiten the update
                adjusted_lr = self.adjust_lr_for_muon(lr, p.shape)
                p.data.mul_(1 - lr * weight_decay)
                # p.data.mul_(1 - adjusted_lr * weight_decay)

                p.data.add_(update, alpha=-adjusted_lr)



                if self.is_approx:
                    state['last_grad'].copy_(grad)
            
            ############################
            #       AdamW backup       #
            ############################

            params = [p for p in group["params"] if not self.state[p]["use_muon"]]
            for p in params:
                grad = p.grad
                if grad is None:
                    continue
                state = self.state[p]
                # State initialization
                if "step" not in state:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    # state['last_grad'] = torch.zeros_like(p)
                    # state['previous_grad'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                # For bias vectors - use simple update
                state['step'] += 1
                step = state['step']
                # Compute momentum-like term with correction
                # MRAS-like
                # last_grad = state['last_grad']
                # c_t = (grad - last_grad).mul(gamma * (beta1 / (1. - beta1))).add(grad)
                # c_t_norm = torch.norm(c_t)
                # if c_t_norm > 1.:
                #     c_t = c_t / c_t_norm
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                # exp_avg.lerp_(c_t, 1 - beta1)
                # exp_avg_sq.lerp_(c_t.square(), 1 - beta2)
                exp_avg.lerp_(grad, 1 - beta1)
                exp_avg_sq.lerp_(grad.square(), 1 - beta2)
                g = exp_avg / (eps + exp_avg_sq.sqrt())
                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5
                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(g, alpha=-lr / scale)

                # if self.is_approx:
                #     state['last_grad'].copy_(grad)

        
        return loss
