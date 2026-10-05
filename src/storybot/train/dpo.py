from __future__ import annotations

import copy
import json
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from storybot.model import GPT
from storybot.tokenizer import BPETokenizer

from .checkpoint import save_checkpoint
from .precision import resolve_precision
from .pretrain import make_optimizer
from .schedule import d2z_scheduler


@dataclass
class PreferencePair:
    prompt: str
    chosen: str
    rejected: str


def load_pairs(path: str | Path) -> list[PreferencePair]:
    """Read a JSONL file of {"prompt", "chosen", "rejected"} objects."""
    pairs = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                pairs.append(PreferencePair(d["prompt"], d["chosen"], d["rejected"]))
    return pairs


def encode_pairs(
    tokenizer: BPETokenizer,
    pairs: Sequence[PreferencePair],
    max_len: int,
) -> list[tuple[list[int], list[int], list[int]]]:
    """Tokenize to (prompt_ids, chosen_ids, rejected_ids).

    Each response ends with <|endoftext|>, so preferring a response also
    teaches where it should stop. Responses are cut so prompt + response
    fits in max_len; pairs whose prompt alone leaves no room are dropped.
    """
    eot_id = tokenizer.special_tokens[tokenizer.EOT_TOKEN]
    encoded = []
    for pair in pairs:
        prompt = tokenizer.encode(pair.prompt)
        room = max_len - len(prompt)
        if room < 1 or not prompt:
            continue
        chosen = (tokenizer.encode(pair.chosen) + [eot_id])[:room]
        rejected = (tokenizer.encode(pair.rejected) + [eot_id])[:room]
        encoded.append((prompt, chosen, rejected))
    return encoded


def collate(
    examples: Sequence[tuple[list[int], list[int], list[int]]], pad_id: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack the chosen sequences of all pairs, then all rejected ones.

    Returns idx (2B, T) right-padded with pad_id, and response_mask (2B, T)
    that is 1 exactly on response tokens: the tokens whose probability DPO
    compares. Prompt and padding positions are 0. Right padding never
    affects the real tokens, since attention is causal.
    """
    seqs, masks = [], []
    for which in (1, 2):  # chosen first, then rejected
        for ex in examples:
            prompt, response = ex[0], ex[which]
            seqs.append(prompt + response)
            masks.append([0] * len(prompt) + [1] * len(response))
    width = max(map(len, seqs))
    idx = torch.full((len(seqs), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), width), dtype=torch.float32)
    for i, (s, m) in enumerate(zip(seqs, masks)):
        idx[i, : len(s)] = torch.tensor(s)
        mask[i, : len(m)] = torch.tensor(m, dtype=torch.float32)
    return idx, mask


def sequence_logprobs(
    model: GPT, idx: torch.Tensor, response_mask: torch.Tensor
) -> torch.Tensor:
    """Sum of log p(token | everything before it) over response tokens.

    Token t is predicted by the logits at position t-1, so logits and the
    mask are shifted against each other by one. Returns shape (batch,).
    """
    logits, _ = model(idx)
    logp = F.log_softmax(logits[:, :-1].float(), dim=-1)
    token_logp = logp.gather(-1, idx[:, 1:, None]).squeeze(-1)
    return (token_logp * response_mask[:, 1:]).sum(dim=-1)


def dpo_loss(
    policy_chosen: torch.Tensor,
    policy_rejected: torch.Tensor,
    ref_chosen: torch.Tensor,
    ref_rejected: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """DPO loss (Rafailov et al. 2023).

    The implicit reward of a response is beta * (log pi - log pi_ref): how
    much more likely the policy makes it than the frozen reference does.
    The loss -log sigmoid(r_chosen - r_rejected) pushes the chosen reward
    above the rejected one, while measuring both relative to the reference
    keeps the policy from drifting far from it.
    """
    chosen_reward = beta * (policy_chosen - ref_chosen)
    rejected_reward = beta * (policy_rejected - ref_rejected)
    margin = chosen_reward - rejected_reward
    loss = -F.logsigmoid(margin).mean()
    metrics = {
        "loss": loss.item(),
        "reward_accuracy": (margin > 0).float().mean().item(),
        "reward_margin": margin.mean().item(),
        "chosen_reward": chosen_reward.mean().item(),
        "rejected_reward": rejected_reward.mean().item(),
    }
    return loss, metrics


def make_reference(model: GPT) -> GPT:
    """Frozen copy of the starting policy."""
    ref = copy.deepcopy(model)
    ref.eval()
    ref.requires_grad_(False)
    return ref


def dpo_batch_loss(
    policy: GPT,
    reference: GPT,
    idx: torch.Tensor,
    mask: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """DPO loss for a collated batch (chosen half, then rejected half)."""
    half = idx.shape[0] // 2
    policy_logp = sequence_logprobs(policy, idx, mask)
    with torch.no_grad():
        ref_logp = sequence_logprobs(reference, idx, mask)
    return dpo_loss(
        policy_logp[:half], policy_logp[half:], ref_logp[:half], ref_logp[half:], beta
    )


@dataclass
class DPOConfig:
    epochs: int = 1
    batch_size: int = 8
    beta: float = 0.1
    peak_lr: float = 1e-5
    warmup_steps: int = 10
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.95)
    max_grad_norm: float = 1.0
    max_len: int | None = None  # None -> model.config.max_seq_len
    precision: str = "auto"
    seed: int = 0


@torch.no_grad()
def evaluate_dpo(
    policy: GPT,
    reference: GPT,
    examples: Sequence[tuple[list[int], list[int], list[int]]],
    beta: float,
    batch_size: int,
    pad_id: int,
) -> dict[str, float]:
    """DPO metrics averaged over examples (weighted by batch size)."""
    was_training = policy.training
    policy.eval()
    totals: dict[str, float] = {}
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        idx, mask = collate(batch, pad_id)
        _, metrics = dpo_batch_loss(policy, reference, idx, mask, beta)
        for k, v in metrics.items():
            totals[k] = totals.get(k, 0.0) + v * len(batch)
    policy.train(was_training)
    return {k: v / len(examples) for k, v in totals.items()}


def train_dpo(
    policy: GPT,
    tokenizer: BPETokenizer,
    train_pairs: Sequence[PreferencePair],
    val_pairs: Sequence[PreferencePair],
    config: DPOConfig,
    out_path: str | Path | None = None,
    log: Callable[[str], None] = print,
) -> list[dict]:
    """Fine-tune policy on preference pairs with DPO.

    The reference is a frozen copy of policy as passed in (typically the
    pretrained or SFT model). Validation metrics are logged after every
    epoch; returns that history.
    """
    rng = random.Random(config.seed)
    torch.manual_seed(config.seed)
    max_len = config.max_len or policy.config.max_seq_len
    pad_id = tokenizer.special_tokens[tokenizer.EOT_TOKEN]
    train = encode_pairs(tokenizer, train_pairs, max_len)
    val = encode_pairs(tokenizer, val_pairs, max_len)
    if not train:
        raise ValueError("no usable training pairs")

    reference = make_reference(policy)
    policy.train()
    device = next(policy.parameters()).device
    precision = resolve_precision(device, config.precision)
    optimizer = make_optimizer(policy, config.peak_lr, config.weight_decay, config.betas)
    steps_per_epoch = -(-len(train) // config.batch_size)
    total_steps = steps_per_epoch * config.epochs
    scheduler = d2z_scheduler(optimizer, total_steps, min(config.warmup_steps, total_steps - 1))
    scaler = precision.make_grad_scaler()

    log(
        f"DPO on {len(train)} pairs ({len(val)} val), beta {config.beta}, "
        f"{total_steps} steps, precision {precision.name}"
    )
    history = []
    start = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        order = list(range(len(train)))
        rng.shuffle(order)
        for s in range(0, len(order), config.batch_size):
            batch = [train[i] for i in order[s : s + config.batch_size]]
            idx, mask = collate(batch, pad_id)
            optimizer.zero_grad(set_to_none=True)
            with precision.autocast():
                loss, _ = dpo_batch_loss(policy, reference, idx.to(device), mask.to(device), config.beta)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(policy.parameters(), config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
        entry = {"epoch": epoch}
        if val:
            entry.update(
                evaluate_dpo(policy, reference, val, config.beta, config.batch_size, pad_id)
            )
        history.append(entry)
        log(
            f"epoch {epoch}/{config.epochs} | "
            + " | ".join(f"{k} {v:.4f}" for k, v in entry.items() if k != "epoch")
            + f" | {time.perf_counter() - start:.1f}s"
        )
    if out_path is not None:
        save_checkpoint(out_path, policy, dpo_config=vars(config), history=history)
    return history
