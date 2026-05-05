# File: ldm/optimizers/signsgd.py
import torch
from torch.optim.optimizer import Optimizer

class SignSGD(Optimizer):
    """Sign SGD optimizer.
    
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    """
    
    def __init__(self, 
                 params, 
                 lr=1e-3, 
                 momentum=0.0, 
                 weight_decay=0.0,
                 adjust_lr = False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= momentum:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
            
        defaults = dict(
            lr=lr, 
            momentum=momentum, 
            weight_decay=weight_decay,
            adjust_lr=adjust_lr
            )
        super(SignSGD, self).__init__(params, defaults)



    def adjust_global_lr_for_signsgd(self, lr):
        adjusted_lr = lr * 0.2 
        return adjusted_lr
    

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        
        Arguments:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
            
        for group in self.param_groups:
            wd = group['weight_decay']
            momentum = group['momentum']
            lr = group['lr']
            adjust_lr = group['adjust_lr']
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = p.grad.data
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(p.data)
                
                buf = state['momentum_buffer']
                # Apply momentum to the sign of gradients
                sign_grad = torch.sign(grad)
                buf.mul_(momentum).add_(grad,alpha=1 - momentum)
                
                p.data.mul_(1 - lr * wd)
                # Update parameters using the sign
                ad_lr = self.adjust_global_lr_for_signsgd(lr) if adjust_lr else lr
                if momentum > 0:
                    p.data.add_(torch.sign(buf), alpha=-ad_lr)
                else:
                    p.data.add_(sign_grad, alpha=-ad_lr)
        
        return loss
