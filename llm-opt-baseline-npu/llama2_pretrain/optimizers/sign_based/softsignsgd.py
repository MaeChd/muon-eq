import torch
from torch.optim import Optimizer


class SoftSignSGD(Optimizer):
    '''
    SoftSignSGD
    '''

    def __init__(self, 
                params,
                lr=1e-3,
                betas=(0.9, 2),
                eps=1e-8,
                weight_decay=0,
                ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))


        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay)
        super().__init__(params, defaults)
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Optimizer does not support sparse gradients')

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p.data)
                    state['s'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_p= state['m'], state['s']
                beta, power = group['betas']
                eps = group['eps']
                wd = group['weight_decay']
                lr = group['lr']
                state['step'] += 1

                if  wd!= 0:
                    p.data.mul_(1 - lr * wd)
                exp_avg.mul_(beta).add_(grad, alpha=1 - beta)
                exp_avg_p.mul_(beta).add_(torch.pow(grad.abs(),power).add_(eps), alpha=1 - beta )

                ## nesterov momentum
                nesterov = beta * exp_avg + (1-beta) * grad
                b_nesterov = torch.pow((beta * exp_avg_p + (1-beta) * torch.pow(grad.abs(),power).add_(eps)), 1 / power).add_(eps)
                # update = nesterov / b_nesterov
                p.data.addcdiv_(nesterov, b_nesterov, value=-lr)
        return loss