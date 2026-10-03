"""Pretrain the GPT (next-token prediction).

Trains on data/train.bin and validates on data/val.bin (both written by
prepare_data.py). Without a validation file, the tail of the training
tokens is held out instead (--val-fraction).

Usage:
    python scripts/pretrain.py --steps 2000
    python scripts/pretrain.py --steps 2000 --resume   # continue a run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.stdout.reconfigure(encoding="utf-8")

import torch  # noqa: E402

from storybot.data import load_token_file, split_train_val  # noqa: E402
from storybot.model import GPT, GPTConfig  # noqa: E402
from storybot.tokenizer import BPETokenizer  # noqa: E402
from storybot.train import PretrainConfig, pretrain  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-tokens", default=str(ROOT / "data" / "train.bin"))
    p.add_argument("--val-tokens", default=str(ROOT / "data" / "val.bin"))
    p.add_argument("--tokenizer", default=str(ROOT / "data" / "tokenizer.json"))
    p.add_argument("--checkpoint", default=str(ROOT / "data" / "checkpoint.pt"))
    p.add_argument("--resume", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # model
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-layers", type=int, default=6)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--max-seq-len", type=int, default=512)
    # training
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--eval-interval", type=int, default=100)
    p.add_argument("--eval-batches", type=int, default=20)
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--precision", default="auto", choices=["auto", "fp32", "bf16", "fp16"])
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if Path(args.val_tokens).exists():
        train_tokens = load_token_file(args.train_tokens)
        val_tokens = load_token_file(args.val_tokens)
    else:
        train_tokens, val_tokens = split_train_val(
            load_token_file(args.train_tokens), args.val_fraction
        )
    print(f"tokens: {len(train_tokens)} train / {len(val_tokens)} val")

    torch.manual_seed(args.seed)
    vocab_size = len(BPETokenizer.load(args.tokenizer).vocab)
    model = GPT(
        GPTConfig(
            vocab_size=vocab_size,
            max_seq_len=args.max_seq_len,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
        )
    )
    config = PretrainConfig(
        total_steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        peak_lr=args.lr,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        eval_interval=args.eval_interval,
        eval_batches=args.eval_batches,
        precision=args.precision,
        seed=args.seed,
    )
    pretrain(
        model,
        train_tokens,
        val_tokens,
        config,
        device=args.device,
        checkpoint_path=args.checkpoint,
        resume=args.resume,
    )
    print(f"saved checkpoint to {args.checkpoint}")


if __name__ == "__main__":
    main()
