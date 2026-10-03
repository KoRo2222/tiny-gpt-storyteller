"""Generate text from a pretrained checkpoint.

Usage:
    python scripts/generate.py --prompt "むかしむかし、"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.stdout.reconfigure(encoding="utf-8")

import torch  # noqa: E402

from storybot.tokenizer import BPETokenizer  # noqa: E402
from storybot.train import load_model  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=str(ROOT / "data" / "checkpoint.pt"))
    p.add_argument("--tokenizer", default=str(ROOT / "data" / "tokenizer.json"))
    p.add_argument("--prompt", default="むかしむかし、")
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--num-samples", type=int, default=1)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
    tokenizer = BPETokenizer.load(args.tokenizer)
    model = load_model(args.checkpoint)
    eot_id = tokenizer.special_tokens[tokenizer.EOT_TOKEN]
    prompt = torch.tensor([tokenizer.encode(args.prompt)])
    for i in range(args.num_samples):
        out = model.generate(
            prompt,
            args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            eot_id=eot_id,
        )[0].tolist()
        if eot_id in out:
            out = out[: out.index(eot_id)]
        print(f"--- sample {i + 1} ---")
        print(tokenizer.decode(out))


if __name__ == "__main__":
    main()
