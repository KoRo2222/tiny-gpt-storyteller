from __future__ import annotations

import json
import re
from pathlib import Path

# GPT-2's pre-tokenization pattern (ASCII-approximated: no \p{L}/\p{N} unicode
# classes, since the training corpus is Python source code).
_SPLIT_PATTERN = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+"""
)


def _bytes_to_unicode() -> dict[int, str]:
    """Map every byte value to a printable unicode char (GPT-2's byte encoder).

    Keeps merge rules and vocab entries as plain strings while still being
    able to round-trip arbitrary bytes, including ones with no printable
    glyph (control chars, etc.).
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


class BPETokenizer:
    """Byte-level BPE tokenizer, trained from scratch (GPT-2 style)."""

    # Document-boundary marker for pretraining data (never produced by BPE
    # merges themselves; only inserted explicitly via encode_with_eot).
    EOT_TOKEN = "<|endoftext|>"

    def __init__(self) -> None:
        self.byte_encoder = _bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.merges: list[tuple[str, str]] = []
        self.ranks: dict[tuple[str, str], int] = {}
        self.vocab: dict[str, int] = {}
        self.inverse_vocab: dict[int, str] = {}
        self.special_tokens: dict[str, int] = {}

    def _pretokenize(self, text: str) -> list[str]:
        return _SPLIT_PATTERN.findall(text)

    def _to_byte_symbols(self, token: str) -> tuple[str, ...]:
        return tuple(self.byte_encoder[b] for b in token.encode("utf-8"))

    @staticmethod
    def _get_pair_counts(
        word_freqs: dict[tuple[str, ...], int]
    ) -> dict[tuple[str, str], int]:
        counts: dict[tuple[str, str], int] = {}
        for word, freq in word_freqs.items():
            for pair in zip(word, word[1:]):
                counts[pair] = counts.get(pair, 0) + freq
        return counts

    @staticmethod
    def _merge_word(
        word: tuple[str, ...], pair: tuple[str, str], merged: str
    ) -> tuple[str, ...]:
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                new_word.append(merged)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        return tuple(new_word)

    def train(self, texts: list[str], vocab_size: int, verbose: bool = False) -> None:
        base_symbols = [self.byte_encoder[b] for b in range(256)]
        if vocab_size < len(base_symbols):
            raise ValueError(f"vocab_size must be >= {len(base_symbols)}")

        word_freqs: dict[tuple[str, ...], int] = {}
        for text in texts:
            for token in self._pretokenize(text):
                word = self._to_byte_symbols(token)
                word_freqs[word] = word_freqs.get(word, 0) + 1

        self.merges = []
        num_merges = vocab_size - len(base_symbols)
        for i in range(num_merges):
            pair_counts = self._get_pair_counts(word_freqs)
            if not pair_counts:
                break
            best_pair = max(pair_counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
            merged_symbol = "".join(best_pair)
            word_freqs = {
                self._merge_word(word, best_pair, merged_symbol): freq
                for word, freq in word_freqs.items()
            }
            self.merges.append(best_pair)
            if verbose:
                print(
                    f"merge {i + 1}/{num_merges}: {best_pair} -> "
                    f"{merged_symbol!r} (count={pair_counts[best_pair]})"
                )

        self.ranks = {pair: i for i, pair in enumerate(self.merges)}
        self._build_vocab(base_symbols)

    def _build_vocab(self, base_symbols: list[str]) -> None:
        self.vocab = {sym: i for i, sym in enumerate(base_symbols)}
        for a, b in self.merges:
            self.vocab["".join((a, b))] = len(self.vocab)
        self.vocab[self.EOT_TOKEN] = len(self.vocab)
        self.special_tokens = {self.EOT_TOKEN: self.vocab[self.EOT_TOKEN]}
        self.inverse_vocab = {i: sym for sym, i in self.vocab.items()}

    def _bpe_word(self, word: tuple[str, ...]) -> tuple[str, ...]:
        while len(word) > 1:
            pairs = list(zip(word, word[1:]))
            candidate = min(pairs, key=lambda p: self.ranks.get(p, float("inf")))
            if candidate not in self.ranks:
                break
            word = self._merge_word(word, candidate, "".join(candidate))
        return word

    def encode(self, text: str) -> list[int]:
        ids = []
        for token in self._pretokenize(text):
            word = self._to_byte_symbols(token)
            for symbol in self._bpe_word(word):
                ids.append(self.vocab[symbol])
        return ids

    def encode_with_eot(self, texts: list[str]) -> list[int]:
        """Encode multiple documents into one id stream, EOT-separated.

        This is the shape pretraining needs: a flat sequence of token ids
        with an explicit boundary marker between otherwise-unrelated
        documents, so the model learns "this is the end" rather than
        blending the tail of one snippet into the head of the next.
        """
        eot_id = self.special_tokens[self.EOT_TOKEN]
        ids: list[int] = []
        for text in texts:
            ids.extend(self.encode(text))
            ids.append(eot_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        symbols = [self.inverse_vocab[i] for i in ids]
        byte_seq = bytearray()
        for symbol in symbols:
            byte_seq.extend(self.byte_decoder[ch] for ch in symbol)
        return byte_seq.decode("utf-8", errors="replace")

    def save(self, path: str | Path) -> None:
        data = {
            "merges": [list(pair) for pair in self.merges],
            "vocab": self.vocab,
        }
        Path(path).write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        tok = cls()
        tok.merges = [tuple(pair) for pair in data["merges"]]
        tok.ranks = {pair: i for i, pair in enumerate(tok.merges)}
        tok.vocab = data["vocab"]
        tok.inverse_vocab = {i: sym for sym, i in tok.vocab.items()}
        tok.special_tokens = {tok.EOT_TOKEN: tok.vocab[tok.EOT_TOKEN]}
        return tok
