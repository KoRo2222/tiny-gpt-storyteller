import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from storybot.tokenizer import BPETokenizer

CORPUS = [
    "def add(a, b):\n    return a + b\n",
    "def sub(a, b):\n    return a - b\n",
    "def add_three(a, b, c):\n    return a + b + c\n",
]


def test_vocab_size_matches_request():
    tok = BPETokenizer()
    tok.train(CORPUS, vocab_size=280)
    # +1 for the <|endoftext|> special token appended after BPE training.
    assert len(tok.vocab) == 281


def test_eot_inserted_between_documents():
    tok = BPETokenizer()
    tok.train(CORPUS, vocab_size=280)
    eot_id = tok.special_tokens[tok.EOT_TOKEN]

    ids = tok.encode_with_eot(CORPUS)

    assert ids.count(eot_id) == len(CORPUS)
    assert ids[-1] == eot_id
    assert ids == tok.encode(CORPUS[0]) + [eot_id] + tok.encode(CORPUS[1]) + [
        eot_id
    ] + tok.encode(CORPUS[2]) + [eot_id]


def test_eot_round_trips_through_decode():
    tok = BPETokenizer()
    tok.train(CORPUS, vocab_size=280)
    eot_id = tok.special_tokens[tok.EOT_TOKEN]
    assert tok.decode([eot_id]) == tok.EOT_TOKEN


def test_encode_decode_roundtrip():
    tok = BPETokenizer()
    tok.train(CORPUS, vocab_size=280)
    for text in CORPUS + ["def unseen(x):\n    return x * 2\n", "日本語もOK?"]:
        assert tok.decode(tok.encode(text)) == text


def test_common_substring_gets_own_token():
    tok = BPETokenizer()
    tok.train(CORPUS, vocab_size=280)
    ids = tok.encode("    return a + b\n")
    assert len(ids) < len("    return a + b\n")


def test_save_and_load_roundtrip(tmp_path):
    tok = BPETokenizer()
    tok.train(CORPUS, vocab_size=280)
    path = tmp_path / "tokenizer.json"
    tok.save(path)

    loaded = BPETokenizer.load(path)
    text = "def add(a, b):\n    return a + b\n"
    assert loaded.encode(text) == tok.encode(text)
    assert loaded.decode(tok.encode(text)) == text


def test_vocab_size_below_256_rejected():
    tok = BPETokenizer()
    try:
        tok.train(CORPUS, vocab_size=100)
    except ValueError:
        return
    raise AssertionError("expected ValueError for vocab_size < 256")
