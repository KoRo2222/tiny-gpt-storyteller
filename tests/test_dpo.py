import json
import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from storybot.model import GPT, GPTConfig
from storybot.tokenizer import BPETokenizer
from storybot.train import (
    DPOConfig,
    PreferencePair,
    collate,
    dpo_loss,
    encode_pairs,
    load_pairs,
    make_reference,
    sequence_logprobs,
    train_dpo,
)
from storybot.train.dpo import dpo_batch_loss

VOCAB = 40
PAD = 0


def _model(seed=0, vocab=VOCAB):
    torch.manual_seed(seed)
    return GPT(GPTConfig(vocab_size=vocab, max_seq_len=32, d_model=32, n_layers=2, n_heads=4))


# --- batching --------------------------------------------------------------


def test_collate_stacks_chosen_then_rejected_with_response_masks():
    examples = [([5, 6], [7, 8, 9], [10]), ([11], [12], [13, 14])]
    idx, mask = collate(examples, pad_id=PAD)
    assert idx.tolist() == [
        [5, 6, 7, 8, 9],  # pair 0 chosen
        [11, 12, PAD, PAD, PAD],  # pair 1 chosen
        [5, 6, 10, PAD, PAD],  # pair 0 rejected
        [11, 13, 14, PAD, PAD],  # pair 1 rejected
    ]
    assert mask.tolist() == [
        [0, 0, 1, 1, 1],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 1, 1, 0, 0],
    ]


def test_encode_pairs_appends_eot_and_truncates_responses():
    tok = BPETokenizer()
    tok.train(["むかしむかし王様がいました。"], vocab_size=260)
    eot = tok.special_tokens[tok.EOT_TOKEN]
    pair = PreferencePair("むかし", "王様がいました。", "王様")
    ((prompt, chosen, rejected),) = encode_pairs(tok, [pair], max_len=100)
    assert prompt == tok.encode("むかし")
    assert chosen == tok.encode("王様がいました。") + [eot]
    assert rejected == tok.encode("王様") + [eot]

    ((prompt, chosen, rejected),) = encode_pairs(tok, [pair], max_len=len(prompt) + 2)
    assert len(chosen) == len(rejected) == 2  # cut to fit max_len
    # A prompt that leaves no room for any response is dropped.
    assert encode_pairs(tok, [pair], max_len=len(prompt)) == []


def test_load_pairs(tmp_path):
    path = tmp_path / "pairs.jsonl"
    path.write_text(
        json.dumps({"prompt": "p", "chosen": "c", "rejected": "r"}, ensure_ascii=False) + "\n\n",
        encoding="utf-8",
    )
    assert load_pairs(path) == [PreferencePair("p", "c", "r")]


# --- log-probabilities and loss ---------------------------------------------


def test_sequence_logprobs_sums_only_response_tokens():
    model = _model()
    examples = [([3, 4], [5, 6, 7], [8]), ([9, 10, 11], [12], [13, 14])]
    idx, mask = collate(examples, PAD)
    got = sequence_logprobs(model, idx, mask)

    # Reference: score each sequence on its own, token by token.
    expected = []
    for prompt, response in [(e[0], e[w]) for w in (1, 2) for e in examples]:
        seq = torch.tensor([prompt + response])
        logp = F.log_softmax(model(seq)[0][0], dim=-1)
        total = sum(
            logp[len(prompt) + j - 1, tok].item() for j, tok in enumerate(response)
        )
        expected.append(total)
    torch.testing.assert_close(got, torch.tensor(expected), atol=1e-4, rtol=1e-5)


def test_dpo_loss_matches_formula():
    pc, pr = torch.tensor([-5.0, -3.0]), torch.tensor([-6.0, -2.0])
    rc, rr = torch.tensor([-5.5, -3.0]), torch.tensor([-5.0, -2.5])
    beta = 0.5
    loss, m = dpo_loss(pc, pr, rc, rr, beta)
    margins = beta * ((pc - rc) - (pr - rr))  # [1.0, -0.25]
    expected = -torch.log(torch.sigmoid(margins)).mean()
    torch.testing.assert_close(loss, expected)
    assert m["reward_accuracy"] == 0.5
    assert m["reward_margin"] == pytest.approx(margins.mean().item())


def test_loss_is_log2_when_policy_equals_reference():
    # No preference learned yet: margin 0, loss = -log sigmoid(0) = log 2.
    policy = _model()
    reference = make_reference(policy)
    idx, mask = collate([([1, 2], [3, 4], [5, 6])], PAD)
    loss, m = dpo_batch_loss(policy, reference, idx, mask, beta=0.1)
    assert loss.item() == pytest.approx(math.log(2), abs=1e-6)
    assert m["reward_margin"] == pytest.approx(0.0, abs=1e-6)


def test_swapping_chosen_and_rejected_flips_the_margin():
    pc, pr, rc, rr = (torch.tensor([v]) for v in (-4.0, -6.0, -5.0, -5.0))
    _, a = dpo_loss(pc, pr, rc, rr, 0.1)
    _, b = dpo_loss(pr, pc, rr, rc, 0.1)
    assert a["reward_margin"] == pytest.approx(-b["reward_margin"])


def test_gradient_step_raises_chosen_and_lowers_rejected():
    # The direction DPO must move the policy: one SGD step makes the chosen
    # response more likely and the rejected one less likely.
    policy = _model()
    reference = make_reference(policy)
    idx, mask = collate([([1, 2, 3], [4, 5, 6], [7, 8, 9])], PAD)
    with torch.no_grad():
        before = sequence_logprobs(policy, idx, mask)
    opt = torch.optim.SGD(policy.parameters(), lr=0.5)
    loss, _ = dpo_batch_loss(policy, reference, idx, mask, beta=0.1)
    loss.backward()
    opt.step()
    with torch.no_grad():
        after = sequence_logprobs(policy, idx, mask)
    assert after[0] > before[0]  # chosen up
    assert after[1] < before[1]  # rejected down


def test_reference_is_frozen_copy():
    policy = _model()
    reference = make_reference(policy)
    assert all(not p.requires_grad for p in reference.parameters())
    assert all(a is not b for a, b in zip(policy.parameters(), reference.parameters()))
    with torch.no_grad():
        next(policy.parameters()).add_(1.0)
    assert not torch.equal(next(policy.parameters()), next(reference.parameters()))


# --- end to end ------------------------------------------------------------


def test_train_dpo_learns_the_preference_and_keeps_reference(tmp_path):
    texts = ["王様は笑いました。", "王様は泣きました。", "むかしむかし、"]
    tok = BPETokenizer()
    tok.train(texts, vocab_size=300)
    policy = _model(vocab=len(tok.vocab))
    pairs = [PreferencePair("むかしむかし、", "王様は笑いました。", "王様は泣きました。")] * 16
    ref_before = [p.detach().clone() for p in policy.parameters()]

    history = train_dpo(
        policy,
        tok,
        pairs,
        pairs[:4],
        DPOConfig(epochs=8, batch_size=4, beta=0.1, peak_lr=1e-3, warmup_steps=2, precision="fp32"),
        out_path=tmp_path / "dpo.pt",
        log=lambda s: None,
    )
    assert history[-1]["reward_accuracy"] == 1.0
    assert history[-1]["loss"] < 0.5 * math.log(2)
    assert history[-1]["reward_margin"] > history[0]["reward_margin"] > 0
    # The policy moved; the saved checkpoint holds it.
    assert any(not torch.equal(a, b) for a, b in zip(ref_before, policy.parameters()))
    assert (tmp_path / "dpo.pt").exists()


def test_train_dpo_rejects_empty_training_set():
    tok = BPETokenizer()
    tok.train(["あいう"], vocab_size=260)
    with pytest.raises(ValueError):
        train_dpo(_model(vocab=len(tok.vocab)), tok, [], [], DPOConfig(), log=lambda s: None)
