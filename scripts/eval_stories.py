"""Measure the quality of a checkpoint's stories with the rule-based checks.

Continues held-out story openings (by default lines from 1000 on, which
make_pairs.py does not use unless asked for >1000 prompts) and reports
averages, so checkpoints before and after DPO can be compared on the same
prompts with the same sampling seed.

Usage:
    python scripts/eval_stories.py --checkpoint data/checkpoint.pt
    python scripts/eval_stories.py --checkpoint data/checkpoint_dpo.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from storybot.prefs import sample_candidates, score_story, story_openings  # noqa: E402
from storybot.prefs.rules import repeated_ngram_ratio, repeated_sentence_ratio  # noqa: E402
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
    )
    p.add_argument("--skip", type=int, default=1000, help="skip stories used for pairs")
    p.add_argument("--num-prompts", type=int, default=100)
    p.add_argument("--samples", type=int, default=2)
    p.add_argument("--max-new-tokens", type=int, default=400)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--show", type=int, default=2, help="print this many samples")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    tokenizer = BPETokenizer.load(args.tokenizer)
    model = load_model(args.checkpoint)
    prompts = story_openings(args.prompts_from, args.num_prompts, skip=args.skip)

    rows = []
    shown = 0
    for prompt in prompts:
        for c in sample_candidates(
            model, tokenizer, prompt, args.samples, args.max_new_tokens,
            args.temperature, args.top_k,
        ):
            row = {
                "finished": c.finished,
                "chars": len(c.text),
                "repeated_sentences": repeated_sentence_ratio(c.text),
                "repeated_ngrams": repeated_ngram_ratio(c.text),
            }
            if c.finished:
                s = score_story(prompt, c)
                row["rule_score"] = s.score
                row["bad_ending"] = "bad_ending" in s.penalties
                row["lost_protagonist"] = "lost_protagonist" in s.penalties
            rows.append(row)
            if shown < args.show:
                print(f"--- {prompt}\n{c.text}\n")
                shown += 1

    def mean(key, only_finished=False):
        vals = [r[key] for r in rows if key in r and (r["finished"] or not only_finished)]
        return float(np.mean(vals)) if vals else float("nan")

    print(f"checkpoint: {args.checkpoint}")
    print(f"stories: {len(rows)} ({len(prompts)} prompts x {args.samples})")
    print(f"finished (emitted EOT):     {mean('finished'):.3f}")
    print(f"rule score (finished only): {mean('rule_score'):.3f}")
    print(f"repeated sentences ratio:   {mean('repeated_sentences'):.3f}")
    print(f"repeated 6-gram ratio:      {mean('repeated_ngrams'):.3f}")
    print(f"bad ending (finished only): {mean('bad_ending'):.3f}")
    print(f"lost protagonist:           {mean('lost_protagonist'):.3f}")
    print(f"mean length (chars):        {mean('chars'):.1f}")


if __name__ == "__main__":
    main()
