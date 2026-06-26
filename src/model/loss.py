import torch

def cross_entropy_loss(
        logits: torch.Tensor,
        targets: torch.Tensor
    ):
    """
    Computes the cross-entropy loss between logits and targets
    L = -sum(y * log p)
    """
    n = targets.shape[0]
    # get max and subtract logits from max for numerical stability
    logit_maxes = logits.max(dim=-1, keepdim=True).values
    norm_logits = logits - logit_maxes
    exps = norm_logits.exp()
    total_exps = exps.sum(-1, keepdim=True)
    log_probs = norm_logits - total_exps.log()
    loss = -log_probs[range(n), targets.view(-1)].mean()
    return loss