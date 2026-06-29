import modal
import wandb
"""
Pretraining from assignments
"""
image = (
    modal.Image.debian_slim()
    .pip_install("torch", "numpy", "wandb", "regex", "jaxtyping", "einops")
    .add_local_python_source("src", "dataloader")
)

app = modal.App("LLM-from-scratch", image=image)

data_vol = modal.Volume.from_name("tinystories-data", create_if_missing=True)
ckpt_vol = modal.Volume.from_name("tinystories-checkpoints", create_if_missing=True)

def get_model_config():
    return {
        "vocab_size": 10000,
        "context_length": 256,
        "d_model": 512,
        "d_ff": 1344,
        "rope_theta": 10000,
        "num_layers": 4,
        "num_heads": 16,
    }


def get_train_config():
    # OLMo-style cadence by default
    return {
        "max_learning_rate": 3e-4,
        "min_learning_rate": 3e-5,
        "warmup_iters": 500,
        "batch_size": 64,
        "max_l2_norm": 1.0,
        "eval_interval": 1000,
        "checkpoint_interval": 1000,
        "keep_last_n_checkpoints": 3,
        "val_steps": 500,
        "weight_decay": 0.01,
        "betas": (0.9, 0.95),
        "eps": 1e-8,
    }


@app.function(
    gpu="A10G",
    volumes={"/data": data_vol, "/ckpt": ckpt_vol},
    secrets=[modal.Secret.from_name("wandb")],
    timeout=60 * 60 * 6,
)
def overfit_model(
        data_path: str,
        total_tokens: int,
):
    
    from src.model.transformer import TransformerLM
    from src.train import training_loop

    model_config = get_model_config()
    train_config = get_train_config()
    total_steps = total_tokens // (train_config['batch_size'] * model_config['context_length'])
    wandb.init(
        project="LLM-from-scratch",
        name="overfit_bz_64",
        config={**model_config, **train_config, "total_steps": total_steps},
    )
    model = TransformerLM(**model_config, device="cuda")
    training_loop(
        model=model,
        data_path=data_path,
        total_tokens=total_tokens,
        context_length=model_config['context_length'],
        batch_size=train_config['batch_size'],
        max_l2_norm=train_config['max_l2_norm'],
        max_learning_rate=train_config['max_learning_rate'],
        min_learning_rate=train_config['min_learning_rate'],
        warmup_iters=train_config['warmup_iters'],
        cosine_cycle_iters=total_steps,
        out='/ckpt/',
        device='cuda',
        on_checkpoint=ckpt_vol.commit,
        log_metrics=lambda metrics, step: wandb.log(metrics, step=step),
        overfit=True,
        # overfit debug: eval every step (cheap, same fixed batch), checkpoint sparsely
        eval_interval=1,
        checkpoint_interval=max(total_steps // 4, 1),
        keep_last_n_checkpoints=2,
        weight_decay=train_config['weight_decay'],
        betas=train_config['betas'],
        eps=train_config['eps'],
    )
    ckpt_vol.commit()


@app.function(
    gpu="A10G",
    volumes={"/data": data_vol, "/ckpt": ckpt_vol},
    secrets=[modal.Secret.from_name("wandb")],
    timeout=60 * 60 * 6,
)
def default_run(
        data_path: str,
        total_tokens: int,
):

    from src.model.transformer import TransformerLM
    from src.train import training_loop, find_latest_checkpoint, peek_checkpoint

    model_config = get_model_config()
    train_config = get_train_config()
    total_steps = total_tokens // (train_config['batch_size'] * model_config['context_length'])

    resume_from = find_latest_checkpoint('/ckpt/')
    wandb_resume_id = None
    if resume_from is not None:
        wandb_resume_id = peek_checkpoint(resume_from).get('wandb_run_id')
        print(f"Found checkpoint {resume_from}; wandb_run_id={wandb_resume_id}")

    wandb.init(
        project="LLM-from-scratch",
        name="default_param_run",
        id=wandb_resume_id,
        resume="allow" if wandb_resume_id else None,
        config={**model_config, **train_config, "total_steps": total_steps},
    )
    model = TransformerLM(**model_config, device="cuda")
    training_loop(
        model=model,
        data_path=data_path,
        total_tokens=total_tokens,
        context_length=model_config['context_length'],
        batch_size=train_config['batch_size'],
        max_l2_norm=train_config['max_l2_norm'],
        max_learning_rate=train_config['max_learning_rate'],
        min_learning_rate=train_config['min_learning_rate'],
        warmup_iters=train_config['warmup_iters'],
        cosine_cycle_iters=total_steps,
        out='/ckpt/',
        device='cuda',
        on_checkpoint=ckpt_vol.commit,
        log_metrics=lambda metrics, step: wandb.log(metrics, step=step),
        overfit=False,
        eval_interval=train_config['eval_interval'],
        checkpoint_interval=train_config['checkpoint_interval'],
        keep_last_n_checkpoints=train_config['keep_last_n_checkpoints'],
        val_steps=train_config['val_steps'],
        weight_decay=train_config['weight_decay'],
        betas=train_config['betas'],
        eps=train_config['eps'],
        resume_from=resume_from,
        wandb_run_id=wandb.run.id,
    )
    ckpt_vol.commit()

@app.function(
    gpu="A10G",
    volumes={"/data": data_vol, "/ckpt": ckpt_vol},
    secrets=[modal.Secret.from_name("wandb")],
    timeout=60 * 60 * 6,
)
def learning_rate_sweep(
    data_path: str,
    total_tokens: int,
):

    from src.model.transformer import TransformerLM
    from src.train import training_loop, find_latest_checkpoint, peek_checkpoint

    model_config = get_model_config()
    train_config = get_train_config()
    total_steps = total_tokens // (train_config['batch_size'] * model_config['context_length'])

    for i in [1, 2, 4, 8, 16, 32, 64, 128]:
        wandb.init(
        project="LLM-from-scratch",
        name=f"learning_rate_sweep_x{i}",
        config={**model_config, **train_config, "total_steps": total_steps},
        )

        model = TransformerLM(**model_config, device="cuda")
        
        training_loop(
            model=model,
            data_path=data_path,
            total_tokens=total_tokens,
            context_length=model_config['context_length'],
            batch_size=train_config['batch_size'],
            max_l2_norm= 1e9, #train_config['max_l2_norm'],
            max_learning_rate=train_config['max_learning_rate']*i,
            min_learning_rate=train_config['min_learning_rate'],
            warmup_iters=train_config['warmup_iters'],
            cosine_cycle_iters=total_steps,
            out='/ckpt/',
            device='cuda',
            on_checkpoint=ckpt_vol.commit,
            log_metrics=lambda metrics, step: wandb.log(metrics, step=step),
            overfit=False,
            eval_interval=train_config['eval_interval'],
            checkpoint_interval=int(1e9),#train_config['checkpoint_interval'],
            # keep_last_n_checkpoints=train_config['keep_last_n_checkpoints'],
            val_steps=train_config['val_steps'],
            weight_decay=train_config['weight_decay'],
            betas=train_config['betas'],
            eps=train_config['eps'],
            wandb_run_id=wandb.run.id,
        )
        wandb.run.summary["max_learning_rate"] = train_config['max_learning_rate'] * i
        wandb.finish()
    

@app.function(
    gpu="A10G",
    volumes={"/data": data_vol, "/ckpt": ckpt_vol},
    secrets=[modal.Secret.from_name("wandb")],
    timeout=60 * 60 * 6,
)
def normalization_abalation(
    data_path: str,
    total_tokens: int,
):
    # import torch.nn as nn
    # from src.model import transformer as tfm
    # tfm.RMSNorm = lambda *a, **kw: nn.Identity()  # ablation: drop RMSNorm
    from src.model.transformer import TransformerLM
    from src.train import training_loop, find_latest_checkpoint, peek_checkpoint

    model_config = get_model_config()
    train_config = get_train_config()

    # model = TransformerLM(**model_config, device="cuda")
    # assert not any(type(m).__name__ == "RMSNorm" for m in model.modules()), "patch failed"

    total_steps = total_tokens // (train_config['batch_size'] * model_config['context_length'])
    
    wandb.init(
    project="LLM-from-scratch",
    name=f"post_norm",
    config={**model_config, **train_config, "total_steps": total_steps},
    )

    model = TransformerLM(**model_config, device="cuda", norm_position="post")
    i = 8
    training_loop(
        model=model,
        data_path=data_path,
        total_tokens=total_tokens,
        context_length=model_config['context_length'],
        batch_size=train_config['batch_size'],
        max_l2_norm= 1e9, #train_config['max_l2_norm'],
        max_learning_rate=train_config['max_learning_rate']*i,
        min_learning_rate=train_config['min_learning_rate'],
        warmup_iters=train_config['warmup_iters'],
        cosine_cycle_iters=total_steps,
        out='/ckpt/',
        device='cuda',
        on_checkpoint=ckpt_vol.commit,
        log_metrics=lambda metrics, step: wandb.log(metrics, step=step),
        overfit=False,
        eval_interval=train_config['eval_interval'],
        checkpoint_interval=int(1e9),#train_config['checkpoint_interval'],
        # keep_last_n_checkpoints=train_config['keep_last_n_checkpoints'],
        val_steps=train_config['val_steps'],
        weight_decay=train_config['weight_decay'],
        betas=train_config['betas'],
        eps=train_config['eps'],
        wandb_run_id=wandb.run.id,
    )
    wandb.finish()


@app.function(
    gpu="A10G",
    volumes={"/data": data_vol, "/ckpt": ckpt_vol},
    secrets=[modal.Secret.from_name("wandb")],
    timeout=60 * 60 * 6,
)
def batch_size_sweep(
    data_path: str,
    total_tokens: int,
):
    from src.model.transformer import TransformerLM
    from src.train import training_loop

    model_config = get_model_config()
    train_config = get_train_config()
    context_length = model_config['context_length']

    for bs in [16, 32, 64, 128, 256, 512]:
        total_steps = total_tokens // (bs * context_length)
        # Clamp warmup so the run isn't mostly-warmup at large batch sizes.
        warmup_iters = min(train_config['warmup_iters'], max(1, total_steps // 4))

        wandb.init(
            project="LLM-from-scratch",
            name=f"batch_size_sweep_bs{bs}",
            config={
                **model_config, **train_config,
                "batch_size": bs,
                "total_steps": total_steps,
                "warmup_iters": warmup_iters,
            },
        )

        model = TransformerLM(**model_config, device="cuda")

        training_loop(
            model=model,
            data_path=data_path,
            total_tokens=total_tokens,
            context_length=context_length,
            batch_size=bs,
            max_l2_norm=train_config['max_l2_norm'],
            max_learning_rate=train_config['max_learning_rate'],
            min_learning_rate=train_config['min_learning_rate'],
            warmup_iters=warmup_iters,
            cosine_cycle_iters=total_steps,
            out='/ckpt/',
            device='cuda',
            on_checkpoint=ckpt_vol.commit,
            log_metrics=lambda metrics, step, _bs=bs: wandb.log(
                {**metrics, "tokens_seen": step * _bs * context_length},
                step=step,
            ),
            overfit=False,
            eval_interval=train_config['eval_interval'],
            checkpoint_interval=int(1e9),
            val_steps=train_config['val_steps'],
            weight_decay=train_config['weight_decay'],
            betas=train_config['betas'],
            eps=train_config['eps'],
            wandb_run_id=wandb.run.id,
        )
        wandb.run.summary["batch_size"] = bs
        wandb.finish()


@app.function(
    gpu="A10G",
    volumes={"/data": data_vol, "/ckpt": ckpt_vol},
    secrets=[modal.Secret.from_name("wandb")],
    timeout=60 * 60 * 6,
)
def position_embedding_abalation(
    data_path: str,
    total_tokens: int,
):
    from src.model.transformer import TransformerLM
    from src.train import training_loop

    # Disable RoPE: MHSA only constructs it when both theta and max_seq_len are truthy.
    model_config = {**get_model_config(), "rope_theta": None}
    train_config = get_train_config()
    total_steps = total_tokens // (train_config['batch_size'] * model_config['context_length'])

    wandb.init(
        project="LLM-from-scratch",
        name="no_position",
        config={**model_config, **train_config, "total_steps": total_steps},
    )

    model = TransformerLM(**model_config, device="cuda")
    assert not any(type(m).__name__ == "RoPE" for m in model.modules()), "RoPE present despite ablation"

    training_loop(
        model=model,
        data_path=data_path,
        total_tokens=total_tokens,
        context_length=model_config['context_length'],
        batch_size=train_config['batch_size'],
        max_l2_norm=train_config['max_l2_norm'],
        max_learning_rate=train_config['max_learning_rate'],
        min_learning_rate=train_config['min_learning_rate'],
        warmup_iters=train_config['warmup_iters'],
        cosine_cycle_iters=total_steps,
        out='/ckpt/',
        device='cuda',
        on_checkpoint=ckpt_vol.commit,
        log_metrics=lambda metrics, step: wandb.log(metrics, step=step),
        overfit=False,
        eval_interval=train_config['eval_interval'],
        checkpoint_interval=int(1e9),
        val_steps=train_config['val_steps'],
        weight_decay=train_config['weight_decay'],
        betas=train_config['betas'],
        eps=train_config['eps'],
        wandb_run_id=wandb.run.id,
    )
    wandb.finish()


@app.function(
    gpu="A10G",
    volumes={"/data": data_vol, "/ckpt": ckpt_vol},
    secrets=[modal.Secret.from_name("wandb")],
    timeout=60 * 60 * 6,
)
def best_run(
    data_path: str,
    total_tokens: int,
):
    from src.model.transformer import TransformerLM
    from src.train import training_loop, find_latest_checkpoint, peek_checkpoint

    model_config = get_model_config()
    train_config = {
        **get_train_config(),
        "batch_size": 128,                       # max we can fit
        "max_learning_rate": 3e-4 * 8 * 2**0.5,  # x8 sweep winner at bs=64, scaled by sqrt(128/64)
    }
    total_steps = total_tokens // (train_config['batch_size'] * model_config['context_length'])

    # Dedicated subdir so old experiment checkpoints don't get resumed by accident.
    out_dir = '/ckpt/best_run/'

    resume_from = find_latest_checkpoint(out_dir)
    wandb_resume_id = None
    if resume_from is not None:
        wandb_resume_id = peek_checkpoint(resume_from).get('wandb_run_id')
        print(f"Found checkpoint {resume_from}; wandb_run_id={wandb_resume_id}")

    wandb.init(
        project="LLM-from-scratch",
        name="best_run_bs512_lrx8",
        id=wandb_resume_id,
        resume="allow" if wandb_resume_id else None,
        config={**model_config, **train_config, "total_steps": total_steps},
    )
    model = TransformerLM(**model_config, device="cuda")
    training_loop(
        model=model,
        data_path=data_path,
        total_tokens=total_tokens,
        context_length=model_config['context_length'],
        batch_size=train_config['batch_size'],
        max_l2_norm=train_config['max_l2_norm'],
        max_learning_rate=train_config['max_learning_rate'],
        min_learning_rate=train_config['min_learning_rate'],
        warmup_iters=train_config['warmup_iters'],
        cosine_cycle_iters=total_steps,
        out=out_dir,
        device='cuda',
        on_checkpoint=ckpt_vol.commit,
        log_metrics=lambda metrics, step: wandb.log(metrics, step=step),
        overfit=False,
        eval_interval=train_config['eval_interval'],
        checkpoint_interval=train_config['checkpoint_interval'],
        keep_last_n_checkpoints=train_config['keep_last_n_checkpoints'],
        val_steps=train_config['val_steps'],
        weight_decay=train_config['weight_decay'],
        betas=train_config['betas'],
        eps=train_config['eps'],
        resume_from=resume_from,
        wandb_run_id=wandb.run.id,
    )
    ckpt_vol.commit()


@app.local_entrypoint()
def main():
    tokenized_data_path = "/data/TinyStoriesV2-GPT4-train-BPE-tokenized.bin"

    # full run with best params: bs=512, LR=3e-4*8, ~541M tokens (~2061 steps).
    tokens = 541271275
    best_run.remote(tokenized_data_path, tokens)


    