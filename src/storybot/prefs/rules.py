from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from .sampling import Candidate

_SENTENCE_END = re.compile(r"(?<=[。！？!?」])")
_KATAKANA_NAME = re.compile(r"[ァ-ヺー]{2,}")
_GOOD_ENDINGS = ("。", "！", "？", "」", "!", "?")


@dataclass
class RuleScore:
    """Rule-based quality score of a story continuation, in [0, 1].

    Each check contributes a penalty in [0, 1] times its weight; the score
    is 1 minus the weighted penalties (clamped at 0). Penalties are kept so
    a pair can be explained.
    """

    score: float
    penalties: dict[str, float] = field(default_factory=dict)


WEIGHTS = {
    "repeated_sentences": 0.35,  # the same sentence said again
    "repeated_ngrams": 0.25,  # loops like 「大好きでした。大好きでした。」
    "bad_ending": 0.2,  # ended the story mid-sentence
    "broken_text": 0.1,  # undecodable bytes (U+FFFD)
    "lost_protagonist": 0.1,  # the named character in the prompt never returns
    "too_short": 0.1,
}


def character_names(prompt: str) -> list[str]:
    """Katakana words in the prompt that look like names. Reduplicated
    words (ピカピカ, キラキラ) are onomatopoeia, not characters."""
    names = []
    for word in _KATAKANA_NAME.findall(prompt):
        half = len(word) // 2
        if len(word) % 2 == 0 and word[:half] == word[half:]:
            continue
        names.append(word)
    return names


def sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_END.split(text) if s.strip()]


def repeated_sentence_ratio(text: str) -> float:
    """Fraction of sentences that duplicate an earlier one."""
    sents = sentences(text)
    if len(sents) < 2:
        return 0.0
    return 1 - len(set(sents)) / len(sents)


def repeated_ngram_ratio(text: str, n: int = 6) -> float:
    """Fraction of character n-grams that occur more than once. Natural
    prose repeats a few (names, set phrases); degenerate loops repeat
    most of them."""
    grams = [text[i : i + n] for i in range(len(text) - n + 1)]
    if not grams:
        return 0.0
    counts = Counter(grams)
    return sum(c for c in counts.values() if c > 1) / len(grams)


def score_story(prompt: str, candidate: Candidate, min_chars: int = 80) -> RuleScore:
    """Score a finished continuation. (Continuations cut off by the token
    budget are not scored: penalizing them would reward short stories.)"""
    text = candidate.text.strip()
    p: dict[str, float] = {}
    p["repeated_sentences"] = min(1.0, 2 * repeated_sentence_ratio(text))
    # Up to ~15% repeated 6-grams is normal storytelling; beyond that it's a loop.
    p["repeated_ngrams"] = min(1.0, max(0.0, repeated_ngram_ratio(text) - 0.15) / 0.35)
    p["bad_ending"] = 0.0 if text.endswith(_GOOD_ENDINGS) else 1.0
    p["broken_text"] = 1.0 if "�" in text else 0.0
    names = character_names(prompt)
    p["lost_protagonist"] = (
        1.0 if names and not any(name in text for name in names) else 0.0
    )
    p["too_short"] = max(0.0, 1 - len(text) / min_chars)
    score = max(0.0, 1 - sum(WEIGHTS[k] * v for k, v in p.items()))
    return RuleScore(score, {k: v for k, v in p.items() if v > 0})


def best_and_worst(
    prompt: str, candidates: list[Candidate], min_gap: float = 0.2
) -> tuple[Candidate, Candidate, RuleScore, RuleScore] | None:
    """The highest- and lowest-scoring candidates, if they differ by at
    least min_gap (smaller gaps are too close to call by rules alone)."""
    if len(candidates) < 2:
        return None
    scored = sorted(
        ((score_story(prompt, c), c) for c in candidates), key=lambda sc: sc[0].score
    )
    (worst_s, worst), (best_s, best) = scored[0], scored[-1]
    if best_s.score - worst_s.score < min_gap:
        return None
    return best, worst, best_s, worst_s
