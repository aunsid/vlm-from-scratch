import torch

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), weight_decay=0.01, eps=1e-8):
        if lr < 0:
            raise ValueError(f"Invalid learing rate: {lr}")
        beta1, beta2 = betas
        defaults = {
            "lr": lr,
            "beta1": beta1,
            "beta2": beta2,
            "weight_decay": weight_decay,
            "eps": eps
            }
        super().__init__(params, defaults)
    
    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            group_lr = group["lr"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 1)
                grad = p.grad.data 
                m1 = state.get("m1", 0)
                m2 = state.get("m2", 0)
                m1 = beta1 * m1 + (1-beta1) * grad
                m2 = beta2 * m2 + (1-beta2) * grad ** 2
                corrected_lr = group_lr * (1- beta2 ** t)** 0.5/ (1 - beta1 ** t)
                decay_grad = group_lr * p.data * weight_decay
                p.data  -= corrected_lr * m1 / (m2 ** 0.5 + eps) + decay_grad
                state["t"]  = t + 1
                state["m1"] = m1
                state["m2"] = m2
        return loss
    



class Adam(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.95, weight_decay=0.01, eps=1e-8):
        if lr < 0:
            raise ValueError(f"Invalid learing rate: {lr}")
        defaults = {
            "lr": lr,
            "beta1": beta1,
            "beta2": beta2,
            "weight_decay": weight_decay,
            "eps": eps
            }
        super().__init__(params, defaults)
    
    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 1)
                grad = p.grad.data + p.data * weight_decay
                m1 = state.get("m1", 0)
                m2 = state.get("m2", 0)
                m1 = beta1 * m1 + (1-beta1) * grad
                m2 = beta2 * m2 + (1-beta2) * grad ** 2
                lr = lr * (1- beta2 ** t)** 0.5/ (1 - beta1 ** t)
                p.data  -= lr * m1 / (m2 ** 0.5 + eps)
                state["t"]  = t + 1
                state["m1"] = m1
                state["m2"] = m2
        return loss