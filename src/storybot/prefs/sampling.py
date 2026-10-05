from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from storybot.model import GPT
from storybot.tokenizer import BPETokenizer


@dataclass
class Candidate:
    text: str
    finished: bool  # the model emitted <|endoftext|> within the token budget


def story_openings(path: str | Path, n: int, field: str = "text_ja") -> list[str]:
    """First sentence of each of the first n stories in a JSONL corpus, as
    prompts the model continues. Stories without a usable first sentence
    are skipped."""
    openings = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if len(openings) >= n:
                break
            text = (json.loads(line).get(field) or "").strip()
            end = text.find("。")
            if 5 <= end <= 80:
                openings.append(text[: end + 1])
    return openings


@torch.no_grad()
def sample_candidates(
    model: GPT,
    tokenizer: BPETokenizer,
    prompt: str,
    num_samples: int,
    max_new_tokens: int = 300,
    temperature: float = 1.0,
    top_k: int | None = 50,
) -> list[Candidate]:
    """Sample several continuations of one prompt (the prompt itself is not
    included in the returned text)."""
    eot_id = tokenizer.special_tokens[tokenizer.EOT_TOKEN]
    prompt_ids = tokenizer.encode(prompt)
    idx = torch.tensor([prompt_ids] * num_samples)
    out = model.generate(
        idx, max_new_tokens, temperature=temperature, top_k=top_k, eot_id=eot_id
    )
    candidates = []
    for row in out[:, len(prompt_ids) :].tolist():
        finished = eot_id in row
        if finished:
            row = row[: row.index(eot_id)]
        candidates.append(Candidate(tokenizer.decode(row), finished))
    return candidates


def distinct_candidates(candidates: Sequence[Candidate]) -> list[Candidate]:
    """Drop exact duplicate texts (identical samples make useless pairs)."""
    seen, unique = set(), []
    for c in candidates:
        if c.text not in seen:
            seen.add(c.text)
            unique.append(c)
    return unique
