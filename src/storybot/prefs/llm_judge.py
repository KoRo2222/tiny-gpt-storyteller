from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

DEFAULT_MODEL = "claude-opus-5-5"

SYSTEM_PROMPT = """\
あなたは子ども向けの短い物語を評価する審査員です。
同じ書き出しに続けて書かれた2つの物語(AとB)を読み、より良い方を選んでください。

評価の観点(重要な順):
1. 話の筋が通っているか(出来事のつながり、原因と結果)
2. 登場人物・持ち物・場所が途中で食い違わないか
3. 書き出しから自然に続いているか
4. 日本語として自然か(文法、言葉づかい)
5. 同じ文や言い回しの繰り返し、途中で途切れた終わり方がないか

長さそのものは評価しないでください。差がほとんどない場合だけ "tie" を選んでください。"""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "reason": {"type": "string", "description": "判断の理由(日本語で1〜2文)"},
        "winner": {"type": "string", "enum": ["A", "B", "tie"]},
    },
    "required": ["reason", "winner"],
    "additionalProperties": False,
}


@dataclass
class Verdict:
    winner: Literal["A", "B", "tie"] | None  # None: the model declined
    reason: str


@dataclass
class PairJudgment:
    """Outcome of judging x vs y in both orders."""

    preferred: Literal["x", "y"] | None  # None: tie, disagreement, or refusal
    verdicts: tuple[Verdict, Verdict]


class LLMJudge:
    """Compares two story continuations with Claude.

    client is an anthropic.Anthropic (or anything with the same
    beta.messages.create surface, e.g. a test double).
    """

    def __init__(self, client: Any, model: str = DEFAULT_MODEL, effort: str = "low"):
        self.client = client
        self.model = model
        self.effort = effort

    def compare(self, prompt: str, story_a: str, story_b: str) -> Verdict:
        user = (
            f"【書き出し】\n{prompt}\n\n"
            f"【物語A】\n{prompt}{story_a}\n\n"
            f"【物語B】\n{prompt}{story_b}"
        )
        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user}],
            output_config={
                "effort": self.effort,
                "format": {"type": "json_schema", "schema": RESPONSE_SCHEMA},
            },
            # If a safety classifier declines, re-run on Anthropic's
            # recommended fallback model instead of losing the judgment.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        if response.stop_reason == "refusal":
            return Verdict(None, "refused")
        text = next(b.text for b in response.content if b.type == "text")
        data = json.loads(text)
        return Verdict(data["winner"], data["reason"])

    def judge_pair(self, prompt: str, x: str, y: str) -> PairJudgment:
        """Ask twice with the stories swapped; keep the result only if both
        orders agree. This cancels the judge's position bias (a tendency
        to favour whichever story is shown first)."""
        first = self.compare(prompt, x, y)  # x is A
        second = self.compare(prompt, y, x)  # x is B
        preferred: Literal["x", "y"] | None = None
        if first.winner == "A" and second.winner == "B":
            preferred = "x"
        elif first.winner == "B" and second.winner == "A":
            preferred = "y"
        return PairJudgment(preferred, (first, second))
