from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path

_LATIN = re.compile(r"[A-Za-z]")


def expand_corpus_paths(paths: Iterable[str | Path]) -> list[Path]:
    """Files given directly, plus every .txt / .jsonl inside given dirs."""
    files: list[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            files.extend(
                sorted(f for f in p.iterdir() if f.suffix in (".txt", ".jsonl"))
            )
        elif p.exists():
            files.append(p)
        else:
            raise FileNotFoundError(p)
    return files


def _read_chunks(path: Path, chunk_chars: int) -> Iterator[str]:
    with path.open(encoding="utf-8") as f:
        while chunk := f.read(chunk_chars):
            yield chunk


def keep_story(text: str, max_latin_ratio: float) -> bool:
    """Drop empty stories and ones that are mostly untranslated
    (machine-translated corpora occasionally leave English behind)."""
    text = text.strip()
    if not text:
        return False
    return len(_LATIN.findall(text)) / len(text) <= max_latin_ratio


def iter_documents(
    paths: Iterable[str | Path],
    jsonl_field: str = "text_ja",
    max_docs: int | None = None,
    max_latin_ratio: float = 0.05,
    chunk_chars: int = 1 << 20,
    stats: Counter | None = None,
) -> Iterator[str | Iterator[str]]:
    """Stream the documents of a corpus, for tokenizer training and encoding.

    A .txt file is one document, yielded as an iterator of text chunks so
    a large file is never read whole. A .jsonl file holds one document per
    line in jsonl_field (TinyStories-JA style); those are filtered with
    keep_story. stats, if given, counts kept and dropped documents.
    """
    stats = stats if stats is not None else Counter()
    for path in expand_corpus_paths(paths):
        if path.suffix == ".txt":
            if max_docs is not None and stats["kept"] >= max_docs:
                return
            stats["kept"] += 1
            yield _read_chunks(path, chunk_chars)
            continue
        with path.open(encoding="utf-8") as f:
            for line in f:
                if max_docs is not None and stats["kept"] >= max_docs:
                    return
                if not line.strip():
                    continue
                text = json.loads(line).get(jsonl_field) or ""
                if keep_story(text, max_latin_ratio):
                    stats["kept"] += 1
                    yield text.strip()
                else:
                    stats["dropped"] += 1
