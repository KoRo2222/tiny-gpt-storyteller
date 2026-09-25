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


JA_CORPUS = [
    "むかしむかし、ある国に王様がいました。王様は毎晩、物語を聞きました。",
    "王様は言いました。「今夜も話をしておくれ。」",
    "ドラゴンは王様の城にやってきました。すごーい、と子どもたちは言いました。",
]


def test_japanese_split_at_script_boundaries():
    tok = BPETokenizer()
    assert tok._pretokenize("むかし、王様がドラゴンを見た。") == [
        "むかし", "、", "王様", "が", "ドラゴン", "を", "見", "た", "。",
    ]


def test_long_vowel_mark_stays_with_kana():
    tok = BPETokenizer()
    assert tok._pretokenize("すごーい") == ["すごーい"]
    assert tok._pretokenize("スーパー") == ["スーパー"]


def test_japanese_repeated_word_gets_single_token():
    tok = BPETokenizer()
    tok.train(JA_CORPUS, vocab_size=300)
    # "王様" appears in every document, so BPE should merge its 6 UTF-8 bytes
    # into one token.
    assert len(tok.encode("王様")) == 1


def test_learned_tokens_never_cross_script_boundaries():
    tok = BPETokenizer()
    tok.train(JA_CORPUS, vocab_size=400)
    for token_id in range(256, len(tok.merges) + 256):
        text = tok.decode([token_id])
        if "�" in text:  # partial UTF-8 sequence; can't pretokenize
            continue
        assert tok._pretokenize(text) == [text], text


def test_japanese_roundtrip():
    tok = BPETokenizer()
    tok.train(JA_CORPUS, vocab_size=300)
    for text in JA_CORPUS + ["　全角スペースとｶﾀｶﾅと１２３。\n\n次の夜。"]:
        assert tok.decode(tok.encode(text)) == text


def _reference_merges(tok, texts, vocab_size):
    """Naive BPE: recount every pair from scratch before each merge."""
    word_freqs = {}
    for text in texts:
        for token in tok._pretokenize(text):
            word = tok._to_byte_symbols(token)
            word_freqs[word] = word_freqs.get(word, 0) + 1
    merges = []
    for _ in range(vocab_size - 256):
        counts = tok._get_pair_counts(word_freqs)
        if not counts:
            break
        best = max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
        word_freqs = {
            tok._merge_word(w, best, "".join(best)): f
            for w, f in word_freqs.items()
        }
        merges.append(best)
    return merges


def test_incremental_training_matches_full_recount():
    import random

    rng = random.Random(0)
    # Small alphabet -> many overlapping repeats ("aaaa") and count ties,
    # which are the cases where incremental updates are easiest to get wrong.
    random_texts = [
        " ".join(
            "".join(rng.choice("abあい") for _ in range(rng.randint(1, 8)))
            for _ in range(20)
        )
        for _ in range(10)
    ]
    for texts in (CORPUS, JA_CORPUS, random_texts, ["aaaaaaa aaa aaaa"]):
        tok = BPETokenizer()
        tok.train(texts, vocab_size=400)
        assert tok.merges == _reference_merges(BPETokenizer(), texts, 400)


# Text full of chunk-boundary hazards: contractions ("'ll" split as "'l"),
# whitespace runs whose split depends on the next char, leading-space tokens,
# and mixed scripts.
TRICKY_TEXT = (
    "We'll see, they're here!!  \n\n\n  むかし  王様は言いました。「すごーい!」\n"
    "   ドラゴン123と１２３ ｶﾀｶﾅ　全角 I'd've... \t\n end   "
)


def _split(text, size):
    return [text[i : i + size] for i in range(0, len(text), size)]


def test_chunked_pretokenize_matches_whole_text_for_every_chunk_size():
    tok = BPETokenizer()
    expected = tok._pretokenize(TRICKY_TEXT)
    for size in range(1, len(TRICKY_TEXT) + 1):
        got = list(tok._pretokenize_chunks(_split(TRICKY_TEXT, size)))
        assert got == expected, size


def test_training_on_chunked_generator_matches_whole_strings():
    texts = JA_CORPUS + [TRICKY_TEXT]
    whole = BPETokenizer()
    whole.train(texts, vocab_size=400)

    chunked = BPETokenizer()
    # A generator of documents, each itself a generator of 7-char chunks:
    # the corpus is never held as full strings.
    chunked.train((iter(_split(t, 7)) for t in texts), vocab_size=400)

    assert chunked.merges == whole.merges
    assert chunked.vocab == whole.vocab


def test_encode_chunks_matches_encode():
    tok = BPETokenizer()
    tok.train(JA_CORPUS + [TRICKY_TEXT], vocab_size=400)
    for size in (1, 3, 7, 50):
        assert list(tok.encode_chunks(_split(TRICKY_TEXT, size))) == tok.encode(
            TRICKY_TEXT
        )
