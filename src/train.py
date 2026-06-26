import os
import math

import numpy as np
from typing import Optional, Callable

import torch

from dataloader.dataloader import get_batch
from src.optimizer import AdamW
from src.model.loss import cross_entropy_loss


def training_loop(
    model: torch.nn.Module,
    data_path: Optional[str] = None,
    total_tokens: int = 5,
    context_length: int = 8,
    batch_size: int = 4,
    optimizer: Optional[torch.optim.Optimizer] = None,
    criterion: Optional[Callable] = None,  
    max_l2_norm: float = 1.0,              
    max_learning_rate: float = 1e-1,        
    min_learning_rate: float = 1e-2,        
    warmup_iters: int = 20,
    cosine_cycle_iters: int = 9000,
    out: str = '/tmp/',
    device: str = 'cpu',
    on_checkpoint: Optional[Callable] = None,
    log_metrics: Optional[Callable] = None,
    overfit: bool = False,
    eval_interval: int = 1000,
    checkpoint_interval: int = 1000,
    keep_last_n_checkpoints: int = 3,
    val_steps: Optional[int] = None,
    weight_decay: float = 0.01,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
    resume_from: Optional[str] = None,
    wandb_run_id: Optional[str] = None,
):
    """
    Mainly training loop that reads the data breaks it into chunks and does model  training.
    """
    
    # split dataset step
    data = np.memmap(data_path, dtype=np.uint16, mode='r')
    dataset_size = len(data)

    # overfit to test if training steps have no bugs
    if overfit:
        slice_size = batch_size * context_length * 4
        train_data = data[:slice_size]
        # make val and train data to be the same
        val_data = train_data
        val_steps = 1
    else:
        train_size = int(0.9 * dataset_size)
        train_data = data[:train_size]
        val_data = data[train_size+1:]
        if val_steps is None:
            val_steps = int(0.1 * dataset_size // batch_size)

    # llm pretraining number of tokens you want to train on
    total_steps = int(total_tokens / (batch_size * context_length))

    # init adamw optimizer
    optimizer = optimizer if optimizer else AdamW(
        model.parameters(),
        weight_decay=weight_decay,
        betas=betas,
        eps=eps,
    )
    # init cross entropy loss
    criterion = criterion if criterion else cross_entropy_loss

    train_losses = []
    val_losses = []
    saved_checkpoints: list[str] = []

    # if training fails - resume from last checkpoint 
    start_iter = 0
    if resume_from is not None:
        _, _, loaded_iter = load_checkpoint(resume_from, model, optimizer)
        start_iter = loaded_iter + 1
        print(f"Resuming from {resume_from} at step {start_iter}")

    activation_norms, hook_handles = register_activation_hooks(model)

    # run training
    for it in range(start_iter, total_steps):
        # get learning rate
        current_lr = get_lr_cosine_schedule(
            it,
            max_learning_rate,
            min_learning_rate,
            warmup_iters,
            cosine_cycle_iters,
        )
        # update lr in params
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        # train on one batch
        training_loss, grad_norm, update_ratios = train(
            train_data,
            batch_size,
            context_length,
            model,
            optimizer,
            max_l2_norm,
            criterion,
            device,
            overfit=overfit,
        )
        train_losses.append(training_loss)

        # Early stopping if NaNs/Infs encountered
        if not math.isfinite(grad_norm) or not math.isfinite(training_loss):
            print(f"Non-finite gradient/loss at step {it} (grad_norm={grad_norm}, loss={training_loss}); stopping.")
            if log_metrics is not None:
                log_metrics({"diverged": 1.0, "grad/total_norm": grad_norm, "train/loss": training_loss}, it)
            break
        
        # log metrics
        if log_metrics is not None:
            metrics = {
                "train/loss": training_loss,
                "train/lr": current_lr,
                "grad/total_norm": grad_norm,
                "update/p_mean_ratio": sum(update_ratios) / len(update_ratios) if update_ratios else 0.0,
                "update/p_max_ratio": max(update_ratios) if update_ratios else 0.0,
            }
            if it % eval_interval == 0:
                metrics.update(compute_weight_norms(model))
                metrics.update(activation_norms)
            log_metrics(metrics, it)

        # Evaluate on validation data
        if it != 0 and it % eval_interval == 0:
            avg_validation_loss = val(
                val_data,
                val_steps,
                batch_size,
                context_length,
                model,
                criterion,
                device,
                overfit=overfit,
            )
            val_losses.append(avg_validation_loss)

            if log_metrics is not None:
                log_metrics({"val/loss": avg_validation_loss}, it)

        # store checkpoints
        if it != 0 and it % checkpoint_interval == 0:
            os.makedirs(out, exist_ok=True)
            checkpoint_path = os.path.join(out, f"checkpoint_step_{it}.pt")
            save_checkpoint(model, optimizer, it, checkpoint_path,
                            extra={"wandb_run_id": wandb_run_id})
            saved_checkpoints.append(checkpoint_path)

            # store last n checkpoints if something fails
            while len(saved_checkpoints) > keep_last_n_checkpoints:
                stale = saved_checkpoints.pop(0)
                try:
                    os.remove(stale)
                except FileNotFoundError:
                    pass

            if on_checkpoint is not None:
                on_checkpoint()

    # final checkpoint at end of training; kept outside the rolling-retention list.
    if total_steps > 0:
        last_it = total_steps - 1
        os.makedirs(out, exist_ok=True)
        final_path = os.path.join(out, f"checkpoint_final_step_{last_it}.pt")
        save_checkpoint(model, optimizer, last_it, final_path,
                        extra={"wandb_run_id": wandb_run_id})
        if on_checkpoint is not None:
            on_checkpoint()

    for h in hook_handles:
        h.remove()

    return train_losses, val_losses


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str,
    extra: Optional[dict] = None,
):
    """
    Dumps model and optimizer params in specified output dir.
    """
    mapping = {
        "model":  model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    if extra:
        mapping.update(extra)
    torch.save(mapping, out)


def find_latest_checkpoint(out_dir: str) -> Optional[str]:
    """Return path to the highest-step *.pt in out_dir, or None if none."""
    if not os.path.isdir(out_dir):
        return None
    pairs = []
    for f in os.listdir(out_dir):
        if not f.endswith('.pt'):
            continue
        try:
            step = int(f[:-3].rsplit('_', 1)[-1])
        except ValueError:
            continue
        pairs.append((step, os.path.join(out_dir, f)))
    return max(pairs)[1] if pairs else None


def peek_checkpoint(src: str) -> dict:
    """Load checkpoint metadata to CPU without applying to a model."""
    return torch.load(src, map_location='cpu', weights_only=False)


def load_checkpoint(
    src: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer
): 
    """
    Loads model and optmizer checkpoint
    """
    map_location = next(model.parameters()).device
    vals = torch.load(src, map_location=map_location, weights_only=False)
    model.load_state_dict(vals["model"])
    optimizer.load_state_dict(vals["optimizer"])
    iteration = vals["iteration"]
    return model, optimizer, iteration


def train(
        train_data: torch.Tensor,
        batch_size: int,
        context_length: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        max_l2_norm: float,
        criterion: Callable,
        device: str,
        overfit: bool = False,
):  
    """
    Training on a single batch
    """
    model.train()
    train_loss = 0.0
    samples = 0

    # get next batch of data
    inputs, targets = get_batch(train_data, batch_size, context_length, device, overfit=overfit)

    # zero previous grads
    optimizer.zero_grad()
    # forward pass
    outputs= model(inputs)
    # compute loss
    loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
    # backward pass - compute gradients
    loss.backward()
    # clip norms
    grad_norm = gradient_clipping(model.parameters(), max_l2_norm)

    # params data and grads for checking instability
    params_before = {
        n: p.data.clone() for n, p in model.named_parameters() if p.grad is not None
    }
    # update params
    optimizer.step()

    # compute gradient update ratios to track stability
    update_ratios = []
    for n, p in model.named_parameters():
        if n not in params_before:
            continue
        weight_norm = p.data.norm()
        if weight_norm > 0:
            update_ratios.append(((p.data - params_before[n]).norm() / weight_norm).item())

    # compute total loss
    train_loss += loss.item() * inputs.size(0)
    samples += inputs.size(0)

    return train_loss / samples, grad_norm, update_ratios


def val(
        val_data: torch.Tensor,
        steps: int,
        batch_size: int,
        context_length: int,
        model: torch.nn.Module,
        criterion: callable,
        device: str,
        overfit: bool = False,
    ):
    """
    Computes loss on the validation data
    """
    # eval mode for no gradient updates
    model.eval()
    total = 0
    validation_loss = 0.0

    with torch.no_grad():
        for _ in range(steps):
            # get model outputs - forward pass
            inputs, targets = get_batch(val_data, batch_size, context_length, device, overfit=overfit)
            outputs = model(inputs)
            # compute loss
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            # aggregate validation loss for the batch
            validation_loss += loss.item() * inputs.size(0)
            total += inputs.size(0)

    return validation_loss/total


def get_lr_cosine_schedule(
        it: int,
        max_learning_rate: int, 
        min_learning_rate: int,
        warmup_iters: int,
        cosine_cycle_iters: int,
):
    """
    Cosine learning rate schedule

    if t < warmup_iters(Tw): lr = (t/Tw) * lr_max
    elif Tw < t < cosine_cycle_iters(Tc): lr = lr_min + 0.5 * (1+cos((t-Tw/(Tc-Tw)* pi) (lr_max - lr_min)
    else: lr = lr_min
    """
    if it < warmup_iters:
        return it / warmup_iters * max_learning_rate
    elif warmup_iters <= it <= cosine_cycle_iters:
        cos_term = (it-warmup_iters)/(cosine_cycle_iters - warmup_iters) * math.pi
        return min_learning_rate + 0.5 * (1 + math.cos(cos_term)) * (max_learning_rate - min_learning_rate)
    else:
        return min_learning_rate

def gradient_clipping(
    parameters,
    max_l2_norm: int,
):
    """
    Given the gradient (for all parameters) g, we compute its l2 norm
    If this norm < max_l2_norm, dont do anything
    norm > max_l2_norm -> scale g down  by. (M/(g+eps))
    """
    params = [p for p in parameters if p.grad is not None]
    total = sum((p.grad ** 2).sum() for p in params) ** 0.5

    if total > max_l2_norm:
        scale = max_l2_norm / (total + 1e-6)
        for p in params:
            p.grad *= scale

    return float(total)


def compute_weight_norms(model):
    return {f"weights/{n}": float(p.detach().norm()) for n, p in model.named_parameters()}


def register_activation_hooks(model):
    """Attach forward hooks on leaf modules; returns (storage, handles)."""
    storage = {}
    handles = []
    for name, module in model.named_modules():
        if name == "" or any(True for _ in module.children()):
            continue
        def make_hook(n):
            def hook(_mod, _inp, out):
                t = out[0] if isinstance(out, tuple) else out
                if isinstance(t, torch.Tensor):
                    storage[f"activations/{n}"] = float(t.detach().norm())
            return hook
        handles.append(module.register_forward_hook(make_hook(name)))
    return storage, handles






