from __future__ import annotations

from collections import Counter
from typing import Literal

from .llm_judge import LLMJudge
from .rules import best_and_worst
from .sampling import Candidate

JudgeMode = Literal["rule", "llm", "both"]


def make_pair(
    prompt: str,
    candidates: list[Candidate],
    mode: JudgeMode,
    judge: LLMJudge | None = None,
    min_gap: float = 0.2,
    stats: Counter | None = None,
) -> dict | None:
    """Turn sampled continuations of one prompt into a preference pair.

    rule: best vs worst by rule score, if they differ by min_gap.
    llm:  the first two finished candidates, judged by the LLM in both orders.
    both: the rule's best/worst pair, kept only if the LLM agrees -- rules
          catch obvious defects cheaply, the LLM vetoes cases where the
          rule-preferred story is actually worse (e.g. tidy but incoherent).
    Returns a JSON-ready dict with prompt/chosen/rejected plus the reasons,
    or None when no confident preference exists.
    """
    stats = stats if stats is not None else Counter()
    if mode in ("llm", "both") and judge is None:
        raise ValueError(f"mode {mode!r} needs an LLM judge")
    # Only stories the model actually ended are compared: one cut off by
    # the token budget can't be judged fairly against a finished one, and
    # preferring finished ones would just teach the model to stop early.
    finished = [c for c in candidates if c.finished]
    stats["unfinished_dropped"] += len(candidates) - len(finished)
    candidates = finished

    if mode == "llm":
        if len(candidates) < 2:
            stats["too_few_candidates"] += 1
            return None
        x, y = candidates[0], candidates[1]
        j = judge.judge_pair(prompt, x.text, y.text)
        if j.preferred is None:
            stats["llm_undecided"] += 1
            return None
        chosen, rejected = (x, y) if j.preferred == "x" else (y, x)
        stats["kept"] += 1
        return {
            "prompt": prompt,
            "chosen": chosen.text,
            "rejected": rejected.text,
            "judge": "llm",
            "llm_reasons": [v.reason for v in j.verdicts],
        }

    picked = best_and_worst(prompt, candidates, min_gap)
    if picked is None:
        stats["rule_gap_too_small"] += 1
        return None
    best, worst, best_s, worst_s = picked
    pair = {
        "prompt": prompt,
        "chosen": best.text,
        "rejected": worst.text,
        "judge": mode,
        "rule_scores": [round(best_s.score, 3), round(worst_s.score, 3)],
        "rule_penalties": [best_s.penalties, worst_s.penalties],
    }
    if mode == "both":
        j = judge.judge_pair(prompt, best.text, worst.text)
        if j.preferred != "x":
            stats["llm_disagreed" if j.preferred == "y" else "llm_undecided"] += 1
            return None
        pair["llm_reasons"] = [v.reason for v in j.verdicts]
    stats["kept"] += 1
    return pair

