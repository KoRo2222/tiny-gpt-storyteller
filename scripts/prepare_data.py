"""Tokenize a corpus into one flat token file for pretraining.

Documents (.txt files, or one story per line of .jsonl files) are separated
by <|endoftext|>. Run once for the training split and once for validation.

Usage:
    python scripts/prepare_data.py --corpus data/raw/train-00000-of-00004.jsonl --out data/train.bin
    python scripts/prepare_data.py --corpus data/raw/validation-00000-of-00001.jsonl --out data/val.bin
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from storybot.data import iter_documents, write_token_file  # noqa: E402
from storybot.tokenizer import BPETokenizer  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", nargs="+", default=[str(ROOT / "data" / "corpus")])
    parser.add_argument("--jsonl-field", default="text_ja")
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--tokenizer", default=str(ROOT / "data" / "tokenizer.json"))
    parser.add_argument("--out", default=str(ROOT / "data" / "tokens.bin"))
    parser.add_argument("--chunk-chars", type=int, default=1 << 20)
    args = parser.parse_args()

    tokenizer = BPETokenizer.load(args.tokenizer)
    stats: Counter = Counter()
    docs = iter_documents(
        args.corpus,
        jsonl_field=args.jsonl_field,
        max_docs=args.max_docs,
        chunk_chars=args.chunk_chars,
        stats=stats,
    )
    start = time.perf_counter()
    n = write_token_file(tokenizer, docs, args.out)
    if stats["kept"] == 0:
        raise SystemExit(f"no documents found in {args.corpus}")
    print(
        f"wrote {n} tokens from {stats['kept']} docs ({stats['dropped']} filtered out) "
        f"to {args.out} in {time.perf_counter() - start:.0f}s"
    )


if __name__ == "__main__":
    main()
