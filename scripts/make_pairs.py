"""Build DPO preference pairs from the model's own stories.

For each story opening, sample several continuations from the checkpoint
and pick a chosen/rejected pair by rules, by an LLM judge (Claude), or by
both (rule pair kept only if the LLM agrees). Pairs are appended to the
output JSONL as they are made, so an interrupted run keeps its progress.

The LLM judge needs Anthropic API credentials (ANTHROPIC_API_KEY, or an
`ant auth login` profile).

Usage:
    python scripts/make_pairs.py --judge rule --num-prompts 500
    python scripts/make_pairs.py --judge both --num-prompts 200
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.stdout.reconfigure(encoding="utf-8")

import torch  # noqa: E402

from storybot.prefs import (  # noqa: E402
    DEFAULT_MODEL,
    LLMJudge,
    distinct_candidates,
    make_pair,
    sample_candidates,
    story_openings,
)
from storybot.tokenizer import BPETokenizer  # noqa: E402
from storybot.train import load_model  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=str(ROOT / "data" / "checkpoint.pt"))
    p.add_argument("--tokenizer", default=str(ROOT / "data" / "tokenizer.json"))
    p.add_argument(
        "--prompts-from",
        default=str(ROOT / "data" / "raw" / "validation-00000-of-00001.jsonl"),
        help="JSONL stories whose first sentences become the prompts",
    )
    p.add_argument("--num-prompts", type=int, default=100)
    p.add_argument("--samples", type=int, default=4, help="continuations per prompt")
    p.add_argument("--max-new-tokens", type=int, default=400)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--judge", choices=["rule", "llm", "both"], default="rule")
    p.add_argument("--judge-model", default=DEFAULT_MODEL)
    p.add_argument("--effort", default="low", choices=["low", "medium", "high", "xhigh", "max"])
    p.add_argument("--min-gap", type=float, default=0.2, help="minimum rule-score gap")
    p.add_argument("--out", default=str(ROOT / "data" / "dpo" / "pairs.jsonl"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    judge = None
    if args.judge in ("llm", "both"):
        import anthropic

        judge = LLMJudge(anthropic.Anthropic(), model=args.judge_model, effort=args.effort)

    torch.manual_seed(args.seed)
    tokenizer = BPETokenizer.load(args.tokenizer)
    model = load_model(args.checkpoint)
    prompts = story_openings(args.prompts_from, args.num_prompts)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    stats: Counter = Counter()
    start = time.perf_counter()
    with out.open("a", encoding="utf-8") as f:
        for i, prompt in enumerate(prompts, 1):
            candidates = distinct_candidates(
                sample_candidates(
                    model,
                    tokenizer,
                    prompt,
                    args.samples,
                    args.max_new_tokens,
                    args.temperature,
                    args.top_k,
                )
            )
            pair = make_pair(prompt, candidates, args.judge, judge, args.min_gap, stats)
            if pair is not None:
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")
                f.flush()
            if i % 10 == 0 or i == len(prompts):
                print(
                    f"{i}/{len(prompts)} prompts | {stats['kept']} pairs | "
                    f"{dict(stats)} | {time.perf_counter() - start:.0f}s"
                )
    print(f"appended {stats['kept']} pairs to {out}")


if __name__ == "__main__":
    main()
