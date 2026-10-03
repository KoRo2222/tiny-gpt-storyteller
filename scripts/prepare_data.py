"""Tokenize the corpus in data/corpus into one flat token file for pretraining.

Each .txt file is one document; documents are separated by <|endoftext|>.

Usage:
    python scripts/prepare_data.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from storybot.data import write_token_file  # noqa: E402
from storybot.tokenizer import BPETokenizer  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def read_chunks(path: Path, chunk_chars: int):
    with path.open(encoding="utf-8") as f:
        while chunk := f.read(chunk_chars):
            yield chunk


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-dir", default=str(ROOT / "data" / "corpus"))
    parser.add_argument("--tokenizer", default=str(ROOT / "data" / "tokenizer.json"))
    parser.add_argument("--out", default=str(ROOT / "data" / "tokens.bin"))
    parser.add_argument("--chunk-chars", type=int, default=1 << 20)
    args = parser.parse_args()

    paths = sorted(Path(args.corpus_dir).glob("*.txt"))
    if not paths:
        raise SystemExit(f"no .txt files found under {args.corpus_dir}")
    tokenizer = BPETokenizer.load(args.tokenizer)
    n = write_token_file(
        tokenizer, (read_chunks(p, args.chunk_chars) for p in paths), args.out
    )
    print(f"wrote {n} tokens from {len(paths)} docs to {args.out}")


if __name__ == "__main__":
    main()
