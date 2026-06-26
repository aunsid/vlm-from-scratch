import torch

def softmax(x: torch.Tensor, dim: int=-1):
    # get the maximum value and subtract for numerical stabitility
    maxes, _ = torch.max(x, dim=dim, keepdim=True)
    x = x -  maxes
    exps = torch.exp(x)
    total = exps.sum(dim=dim, keepdim=True)
    return exps/total

def sigmoid(x: torch.Tensor):
    return 1 / (1 + torch.exp(-x))

def SiLU(x: torch.Tensor):
    silu = x * sigmoid(x)
    return silu