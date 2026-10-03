from __future__ import annotations

import torch
from torch import nn

from .precision import Precision


def train_step(
    model: nn.Module,
    idx: torch.Tensor,
    targets: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    precision: Precision,
    scaler: torch.amp.GradScaler,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    max_grad_norm: float | None = None,
) -> float:
    """One optimizer update with mixed precision; returns the loss.

    Forward (and therefore backward) runs under autocast; the loss is
    scaled for fp16 (scaler is a no-op otherwise) and gradients are
    unscaled before clipping, so max_grad_norm applies to the true
    gradients. If fp16 gradients overflowed, the scaler skips the update.
    """
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with precision.autocast():
        _, loss = model(idx, targets)
    scaler.scale(loss).backward()
    if max_grad_norm is not None:
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    scaler.step(optimizer)
    scaler.update()
    if scheduler is not None:
        scheduler.step()
    return loss.item()
