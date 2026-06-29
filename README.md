# vlm-from-scratch

Building a small/nano vision-language model from scratch. The transformer LM backbone was written while working through [Stanford CS336](https://stanford-cs336.github.io/spring2025/) (cs336n); post-training and the vision stack are in progress.

## Layout

- `src/model/` — transformer, activations, loss
- `src/tokenizer/` — BPE tokenizer (Python and a faster variant)
- `src/train.py`, `src/optimizer.py` — training loop and optimizer
- `dataloader/` — data loading utilities
- `experiments/pretraining/` — pretraining runs, ablations, and results ([writeup](experiments/pretraining/experiments.md))
- `experiments/post_training/` — post-training experiments (in progress)

## Experiments

- [Pretraining notes](experiments/pretraining/experiments.md) — hyperparameter selection, LR sweep, batch-size effects, LayerNorm/RoPE ablations, and tuned-run results on TinyStories.
