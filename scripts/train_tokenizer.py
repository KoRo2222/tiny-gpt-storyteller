"""Train the BPE tokenizer on a corpus and save it.

The corpus is any mix of .txt files (one document each) and .jsonl files
(one story per line, e.g. TinyStories-JA), given as files or directories.

Usage:
    python scripts/train_tokenizer.py --corpus data/raw/train-00000-of-00004.jsonl \
        --vocab-size 8000 --max-docs 200000
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from storybot.data import iter_documents  # noqa: E402
from storybot.tokenizer import BPETokenizer  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", nargs="+", default=[str(ROOT / "data" / "corpus")])
    parser.add_argument("--jsonl-field", default="text_ja")
    parser.add_argument("--max-docs", type=int, default=None, help="train on the first N docs only")
    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument("--out", default=str(ROOT / "data" / "tokenizer.json"))
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--chunk-chars",
        type=int,
        default=1 << 20,
        help="characters read from a corpus file / sent to a worker at a time",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 1,
        help="processes used to pretokenize the corpus",
    )
    args = parser.parse_args()

    stats: Counter = Counter()
    docs = iter_documents(
        args.corpus,
        jsonl_field=args.jsonl_field,
        max_docs=args.max_docs,
        chunk_chars=args.chunk_chars,
        stats=stats,
    )
    start = time.perf_counter()
    tokenizer = BPETokenizer()
    tokenizer.train(
        docs,
        vocab_size=args.vocab_size,
        verbose=args.verbose,
        num_workers=args.workers,
        piece_chars=args.chunk_chars,
    )
    if stats["kept"] == 0:
        raise SystemExit(f"no documents found in {args.corpus}")
    tokenizer.save(args.out)
    print(
        f"trained on {stats['kept']} docs ({stats['dropped']} filtered out) "
        f"in {time.perf_counter() - start:.0f}s"
    )
    print(f"saved tokenizer ({len(tokenizer.vocab)} tokens) to {args.out}")

    sample = "むかしむかし、ある国に物語の好きな王様がいました。"
    ids = tokenizer.encode(sample)
    print(f"\nsample: {sample!r}")
    print(f"encoded ({len(ids)} tokens): {[tokenizer.decode([i]) for i in ids]}")
    print(f"decoded matches original: {tokenizer.decode(ids) == sample}")


if __name__ == "__main__":
    main()
