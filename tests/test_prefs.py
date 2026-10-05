import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from storybot.model import GPT, GPTConfig
from storybot.prefs import (
    Candidate,
    LLMJudge,
    best_and_worst,
    distinct_candidates,
    make_pair,
    sample_candidates,
    score_story,
    story_openings,
)
from storybot.prefs.rules import character_names, repeated_ngram_ratio, repeated_sentence_ratio
from storybot.tokenizer import BPETokenizer

PROMPT = "むかしむかし、リリーという女の子がいました。"
GOOD = "リリーはお花を育てるのが大好きでした。ある日、リリーは庭できれいな赤い花を見つけました。リリーはその花をお母さんに見せて、二人でにっこり笑いました。それからリリーは毎日、花に水をあげました。"
LOOP = "リリーは大好きでした。大好きでした。大好きでした。大好きでした。大好きでした。大好きでした。"


def C(text, finished=True):
    return Candidate(text, finished)


# --- rule scoring ----------------------------------------------------------


def test_coherent_story_scores_full_marks():
    s = score_story(PROMPT, C(GOOD))
    assert s.score == pytest.approx(1.0)
    assert s.penalties == {}


def test_repetition_loop_scores_far_lower():
    good, loop = score_story(PROMPT, C(GOOD)), score_story(PROMPT, C(LOOP))
    assert loop.score < good.score - 0.4
    assert {"repeated_sentences", "repeated_ngrams"} <= loop.penalties.keys()


def test_each_defect_is_penalized():
    base = score_story(PROMPT, C(GOOD)).score
    cases = {
        "bad_ending": GOOD[:-1] + "、そして",
        "broken_text": GOOD.replace("赤い", "赤�"),
        "lost_protagonist": GOOD.replace("リリー", "ボブ"),
        "too_short": "リリーは笑いました。",
    }
    for penalty, text in cases.items():
        s = score_story(PROMPT, C(text))
        assert penalty in s.penalties, penalty
        assert s.score < base, penalty


def test_repetition_measures():
    assert repeated_sentence_ratio("あ。い。う。") == 0
    assert repeated_sentence_ratio("あ。あ。あ。あ。") == pytest.approx(0.75)
    assert repeated_ngram_ratio("あいうえおかきくけこ") == 0
    assert repeated_ngram_ratio("ねこねこねこねこねこねこ") > 0.9


def test_onomatopoeia_is_not_a_character_name():
    assert character_names("スポットはピカピカの車とキラキラの星を見ました。") == ["スポット"]


def test_best_and_worst_needs_a_clear_gap():
    best, worst, bs, ws = best_and_worst(PROMPT, [C(LOOP), C(GOOD)])
    assert (best.text, worst.text) == (GOOD, LOOP)
    assert bs.score > ws.score
    # Two equally good stories: no confident preference.
    assert best_and_worst(PROMPT, [C(GOOD), C(GOOD + "")]) is None
    assert best_and_worst(PROMPT, [C(GOOD)]) is None


# --- LLM judge (with a fake client) ------------------------------------------


class FakeClient:
    """Stands in for anthropic.Anthropic: answers from a scripted function
    and records every request."""

    def __init__(self, decide):
        self.decide = decide  # (story_a, story_b) -> "A" | "B" | "tie" | "refuse"
        self.requests = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        content = kwargs["messages"][0]["content"]
        a = content.split("【物語A】\n")[1].split("\n\n【物語B】")[0]
        b = content.split("【物語B】\n")[1]
        winner = self.decide(a, b)
        if winner == "refuse":
            return SimpleNamespace(stop_reason="refusal", content=[])
        text = json.dumps({"reason": "理由", "winner": winner}, ensure_ascii=False)
        return SimpleNamespace(
            stop_reason="end_turn", content=[SimpleNamespace(type="text", text=text)]
        )


def prefers_good(a, b):
    return "A" if GOOD in a else "B"


def test_judge_request_shape():
    client = FakeClient(prefers_good)
    LLMJudge(client).compare(PROMPT, GOOD, LOOP)
    req = client.requests[0]
    assert req["model"] == "claude-opus-5-5"
    assert req["output_config"]["effort"] == "low"
    assert req["output_config"]["format"]["schema"]["properties"]["winner"]["enum"] == ["A", "B", "tie"]
    assert req["fallbacks"] == "default"
    assert "server-side-fallback-2026-07-01" in req["betas"]
    # Stories are shown as continuations of the prompt.
    assert PROMPT + GOOD in req["messages"][0]["content"]


def test_judge_pair_asks_both_orders_and_keeps_consistent_verdicts():
    client = FakeClient(prefers_good)
    j = LLMJudge(client).judge_pair(PROMPT, LOOP, GOOD)
    assert j.preferred == "y"
    assert len(client.requests) == 2
    first, second = (r["messages"][0]["content"] for r in client.requests)
    assert first.index(LOOP) < first.index(GOOD)  # x shown as A first...
    assert second.index(GOOD) < second.index(LOOP)  # ...then as B


def test_position_biased_judge_yields_no_preference():
    # A judge that always picks whatever is shown first says nothing real.
    j = LLMJudge(FakeClient(lambda a, b: "A")).judge_pair(PROMPT, GOOD, LOOP)
    assert j.preferred is None


def test_refusal_and_tie_yield_no_preference():
    assert LLMJudge(FakeClient(lambda a, b: "refuse")).judge_pair(PROMPT, GOOD, LOOP).preferred is None
    assert LLMJudge(FakeClient(lambda a, b: "tie")).judge_pair(PROMPT, GOOD, LOOP).preferred is None


# --- pair construction --------------------------------------------------------


def test_rule_mode_pair():
    stats = Counter()
    pair = make_pair(PROMPT, [C(LOOP), C(GOOD)], "rule", stats=stats)
    assert (pair["chosen"], pair["rejected"]) == (GOOD, LOOP)
    assert pair["rule_scores"][0] > pair["rule_scores"][1]
    assert stats["kept"] == 1


def test_unfinished_candidates_are_not_compared():
    # A cut-off story must not become "rejected" just for being long.
    stats = Counter()
    assert make_pair(PROMPT, [C(GOOD), C(GOOD + "そして", finished=False)], "rule", stats=stats) is None
    assert stats["unfinished_dropped"] == 1


def test_both_mode_keeps_rule_pair_only_if_llm_agrees():
    stats = Counter()
    agree = LLMJudge(FakeClient(prefers_good))
    assert make_pair(PROMPT, [C(LOOP), C(GOOD)], "both", agree, stats=stats)["chosen"] == GOOD

    prefers_loop = LLMJudge(FakeClient(lambda a, b: "A" if LOOP in a else "B"))
    assert make_pair(PROMPT, [C(LOOP), C(GOOD)], "both", prefers_loop, stats=stats) is None
    assert stats["llm_disagreed"] == 1


def test_llm_mode_uses_llm_preference():
    judge = LLMJudge(FakeClient(prefers_good))
    pair = make_pair(PROMPT, [C(LOOP), C(GOOD)], "llm", judge)
    assert (pair["chosen"], pair["rejected"]) == (GOOD, LOOP)
    assert len(pair["llm_reasons"]) == 2


def test_llm_modes_require_a_judge():
    with pytest.raises(ValueError):
        make_pair(PROMPT, [C(GOOD), C(LOOP)], "both")


# --- sampling -----------------------------------------------------------------


def test_story_openings_take_first_sentence(tmp_path):
    path = tmp_path / "s.jsonl"
    rows = [{"text_ja": "むかしむかし、猫がいました。猫は歌いました。"}, {"text_ja": "短。"}, {"text_ja": "ある日、犬が走りました。"}]
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    assert story_openings(path, 5) == ["むかしむかし、猫がいました。", "ある日、犬が走りました。"]


def test_sample_candidates_strip_prompt_and_mark_finished():
    tok = BPETokenizer()
    tok.train([GOOD, LOOP], vocab_size=300)
    eot = tok.special_tokens[tok.EOT_TOKEN]
    prompt_ids = tok.encode("リリーは")
    a, b = tok.encode("笑いました。"), tok.encode("歌い")

    class FakeModel:
        """generate() returns fixed rows: one ends with EOT (+ padding),
        one runs out of tokens, one repeats the first."""

        def generate(self, idx, max_new_tokens, **kw):
            assert idx.tolist() == [prompt_ids] * 3
            rows = [a + [eot, eot], b + b + b, a + [eot, eot]]
            width = max(map(len, rows))
            rows = [r + [eot] * (width - len(r)) if r[-1] == eot else r for r in rows]
            return torch.tensor([prompt_ids + r for r in rows])

    cands = sample_candidates(FakeModel(), tok, "リリーは", num_samples=3, max_new_tokens=6)
    assert [(c.text, c.finished) for c in cands] == [
        ("笑いました。", True),
        ("歌い歌い歌い", False),
        ("笑いました。", True),
    ]
    assert [c.text for c in distinct_candidates(cands)] == ["笑いました。", "歌い歌い歌い"]
