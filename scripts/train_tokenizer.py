"""Train the BPE tokenizer on the corpus in data/corpus and save it.

Usage:
    python scripts/train_tokenizer.py --vocab-size 512
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from storybot.tokenizer import BPETokenizer  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def read_chunks(path: Path, chunk_chars: int) -> Iterator[str]:
    """Read a text file piece by piece so large corpora never sit in memory."""
    with path.open(encoding="utf-8") as f:
        while chunk := f.read(chunk_chars):
            yield chunk


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-dir", default=str(ROOT / "data" / "corpus"))
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument(
        "--out", default=str(ROOT / "data" / "tokenizer.json")
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--chunk-chars",
        type=int,
        default=1 << 20,
        help="characters read from a corpus file at a time",
    )
    args = parser.parse_args()

    corpus_dir = Path(args.corpus_dir)
    paths = sorted(corpus_dir.glob("*.txt"))
    if not paths:
        raise SystemExit(f"no .txt files found under {corpus_dir}")

    tokenizer = BPETokenizer()
    tokenizer.train(
        (read_chunks(p, args.chunk_chars) for p in paths),
        vocab_size=args.vocab_size,
        verbose=args.verbose,
    )
    tokenizer.save(args.out)
    print(f"saved tokenizer ({len(tokenizer.vocab)} tokens) to {args.out}")

    sample = "むかしむかし、ある国に物語の好きな王様がいました。"
    ids = tokenizer.encode(sample)
    decoded = tokenizer.decode(ids)
    print(f"\nsample: {sample!r}")
    print(f"encoded ({len(ids)} tokens): {ids}")
    print(f"decoded matches original: {decoded == sample}")

    # Pretraining will consume every corpus doc back to back, separated by
    # <|endoftext|>; count the resulting stream without materializing it.
    eot_id = tokenizer.special_tokens[tokenizer.EOT_TOKEN]
    n_tokens = sum(
        sum(1 for _ in tokenizer.encode_chunks(read_chunks(p, args.chunk_chars)))
        + 1
        for p in paths
    )
    print(
        f"\npacked {len(paths)} docs into {n_tokens} tokens "
        f"({len(paths)} <|endoftext|> boundaries, id={eot_id})"
    )


if __name__ == "__main__":
    main()
