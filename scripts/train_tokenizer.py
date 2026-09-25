"""Train the BPE tokenizer on the corpus in data/corpus and save it.

Usage:
    python scripts/train_tokenizer.py --vocab-size 512
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from storybot.tokenizer import BPETokenizer  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-dir", default=str(ROOT / "data" / "corpus"))
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument(
        "--out", default=str(ROOT / "data" / "tokenizer.json")
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    corpus_dir = Path(args.corpus_dir)
    texts = [
        p.read_text(encoding="utf-8") for p in sorted(corpus_dir.glob("*.txt"))
    ]
    if not texts:
        raise SystemExit(f"no .txt files found under {corpus_dir}")

    tokenizer = BPETokenizer()
    tokenizer.train(texts, vocab_size=args.vocab_size, verbose=args.verbose)
    tokenizer.save(args.out)
    print(f"saved tokenizer ({len(tokenizer.vocab)} tokens) to {args.out}")

    sample = "むかしむかし、ある国に物語の好きな王様がいました。"
    ids = tokenizer.encode(sample)
    decoded = tokenizer.decode(ids)
    print(f"\nsample: {sample!r}")
    print(f"encoded ({len(ids)} tokens): {ids}")
    print(f"decoded matches original: {decoded == sample}")

    # This is the shape pretraining will consume: every corpus doc back to
    # back, separated by <|endoftext|>, as one flat id stream.
    packed = tokenizer.encode_with_eot(texts)
    eot_id = tokenizer.special_tokens[tokenizer.EOT_TOKEN]
    print(
        f"\npacked {len(texts)} docs into {len(packed)} tokens "
        f"({packed.count(eot_id)} <|endoftext|> boundaries, id={eot_id})"
    )


if __name__ == "__main__":
    main()
