"""Fine-tune a checkpoint with DPO on preference pairs.

The pairs file is JSONL, one {"prompt", "chosen", "rejected"} per line.
The starting checkpoint also serves as the frozen reference model.

Usage:
    python scripts/dpo.py --pairs data/dpo/pairs.jsonl --epochs 1
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from storybot.tokenizer import BPETokenizer  # noqa: E402
from storybot.train import DPOConfig, load_model, load_pairs, train_dpo  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", required=True)
    p.add_argument("--checkpoint", default=str(ROOT / "data" / "checkpoint.pt"))
    p.add_argument("--tokenizer", default=str(ROOT / "data" / "tokenizer.json"))
    p.add_argument("--out", default=str(ROOT / "data" / "checkpoint_dpo.pt"))
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--warmup-steps", type=int, default=10)
    p.add_argument("--max-len", type=int, default=None)
    p.add_argument("--precision", default="auto", choices=["auto", "fp32", "bf16", "fp16"])
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    pairs = load_pairs(args.pairs)
    random.Random(args.seed).shuffle(pairs)
    n_val = int(len(pairs) * args.val_fraction)
    val, train = pairs[:n_val], pairs[n_val:]
    print(f"pairs: {len(train)} train / {len(val)} val")

    model = load_model(args.checkpoint)
    tokenizer = BPETokenizer.load(args.tokenizer)
    config = DPOConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        beta=args.beta,
        peak_lr=args.lr,
        warmup_steps=args.warmup_steps,
        max_len=args.max_len,
        precision=args.precision,
        seed=args.seed,
    )
    train_dpo(model, tokenizer, train, val, config, out_path=args.out)
    print(f"saved checkpoint to {args.out}")


if __name__ == "__main__":
    main()
