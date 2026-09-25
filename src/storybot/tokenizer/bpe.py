from __future__ import annotations

import heapq
import json
import re
from collections import Counter, deque
from collections.abc import Iterable, Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from pathlib import Path

# Character classes for pre-tokenization. Japanese has no spaces between
# words, so script boundaries (hiragana / katakana / kanji / punctuation)
# stand in for word boundaries. "ー" belongs to both kana classes so that
# long vowels stay attached ("すごーい", "ドラゴン").
_LATIN = "A-Za-zＡ-Ｚａ-ｚ"
_DIGIT = "0-9０-９"
_HIRAGANA = "ぁ-ゖゝゞー"
_KATAKANA = "ァ-ヺーヽヾｦ-ﾟ"
_KANJI = "一-鿿㐀-䶿々〆"

# GPT-2's pre-tokenization pattern, extended with Japanese script classes.
_SPLIT_PATTERN = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d"""
    rf"""| ?[{_LATIN}]+| ?[{_DIGIT}]+"""
    rf"""| ?[{_HIRAGANA}]+| ?[{_KATAKANA}]+| ?[{_KANJI}]+"""
    rf"""| ?[^\s{_LATIN}{_DIGIT}{_HIRAGANA}{_KATAKANA}{_KANJI}]+"""
    r"""|\s+(?!\S)|\s+"""
)

_CLASS_PATTERNS = [
    re.compile(f"[{cls}]") for cls in (_LATIN, _DIGIT, _HIRAGANA, _KATAKANA, _KANJI)
]
_SPACE = re.compile(r"\s")


def _char_classes(c: str) -> frozenset[int]:
    """Indices of the pretokenizer character classes c belongs to (5 = the
    catch-all symbol class). "ー" belongs to two."""
    found = frozenset(i for i, p in enumerate(_CLASS_PATTERNS) if p.match(c))
    return found or frozenset((len(_CLASS_PATTERNS),))


def _is_safe_split(text: str, i: int) -> bool:
    """True if pretokenizing text[:i] and text[i:] separately gives exactly
    the pretokens of text.

    The pattern has no lookbehind, so matching from i is unaffected by what
    precedes it; it only has to be guaranteed that a pretoken boundary falls
    at i. That holds when both neighbours are non-space and share no
    character class (so no run can span them), except after "'", where a
    contraction such as "'ll" may be in progress.
    """
    a, b = text[i - 1], text[i]
    if a == "'" or _SPACE.match(a) or _SPACE.match(b):
        return False
    return not (_char_classes(a) & _char_classes(b))


def _safe_pieces(chunks: Iterable[str], target_chars: int) -> Iterator[str]:
    """Regroup one document's chunks into pieces of about target_chars that
    can be pretokenized independently (split only at _is_safe_split points).
    A piece grows past target_chars when no safe point is found."""
    buf = ""
    scan_from = target_chars  # cut at the first safe point at or after this
    for chunk in chunks:
        buf += chunk
        while len(buf) > scan_from:
            cut = next(
                (
                    i
                    for i in range(max(scan_from, 1), len(buf))
                    if _is_safe_split(buf, i)
                ),
                None,
            )
            if cut is None:
                # Wait for more text; don't rescan what was already checked.
                scan_from = len(buf)
                break
            yield buf[:cut]
            buf = buf[cut:]
            scan_from = target_chars
    if buf:
        yield buf


def _count_pretokens(text: str) -> Counter[str]:
    """Worker task for parallel training (module-level so it can be pickled)."""
    return Counter(_SPLIT_PATTERN.findall(text))


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


class _Desc:
    """Heap key that orders pairs in reverse, so heapq (a min-heap) pops the
    lexicographically largest pair first among equal counts."""

    __slots__ = ("pair",)

    def __init__(self, pair: tuple[str, str]) -> None:
        self.pair = pair

    def __lt__(self, other: "_Desc") -> bool:
        return self.pair > other.pair


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

    @staticmethod
    def _pretokenize_chunks(chunks: Iterable[str]) -> Iterator[str]:
        """Pretokenize one document given as consecutive text chunks.

        Yields exactly what _pretokenize("".join(chunks)) would, without ever
        holding the whole document. The last two matches of each buffer are
        held back and re-matched with the next chunk: only they can change
        once more text arrives (a match that ends at the buffer edge may keep
        growing, and a contraction like "'ll" cut to "'l" shows up as two
        short matches). Every earlier match starts at least 3 chars before
        the edge, so its alternatives and lookahead see the same text either
        way.
        """
        buf = ""
        for chunk in chunks:
            buf += chunk
            held: deque[re.Match[str]] = deque()
            for m in _SPLIT_PATTERN.finditer(buf):
                held.append(m)
                if len(held) > 2:
                    yield held.popleft().group()
            if held:
                buf = buf[held[0].start():]
        yield from _SPLIT_PATTERN.findall(buf)

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

    @staticmethod
    def _count_parallel(
        docs: Iterable[Iterable[str]], num_workers: int, piece_chars: int
    ) -> Counter[str]:
        pieces = (
            piece for chunks in docs for piece in _safe_pieces(chunks, piece_chars)
        )
        token_freqs: Counter[str] = Counter()
        # Keep at most 2 pieces per worker in flight so a large corpus is
        # never queued into memory ahead of the workers.
        max_pending = 2 * num_workers
        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            pending: set[Future[Counter[str]]] = set()
            for piece in pieces:
                pending.add(pool.submit(_count_pretokens, piece))
                if len(pending) >= max_pending:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for f in done:
                        token_freqs.update(f.result())
            for f in pending:
                token_freqs.update(f.result())
        return token_freqs

    def train(
        self,
        texts: Iterable[str | Iterable[str]],
        vocab_size: int,
        verbose: bool = False,
        num_workers: int = 1,
        piece_chars: int = 1 << 20,
    ) -> None:
        """Learn merges from documents.

        Each document is either a str or an iterable of str chunks (e.g. a
        large file read piece by piece), and texts itself may be a generator,
        so the corpus is streamed: only the counts of distinct pretokens are
        kept in memory, never the corpus text.

        With num_workers > 1, pretokenizing and counting run in worker
        processes on pieces of about piece_chars characters. The learned
        merges are identical to the single-process result.
        """
        base_symbols = [self.byte_encoder[b] for b in range(256)]
        if vocab_size < len(base_symbols):
            raise ValueError(f"vocab_size must be >= {len(base_symbols)}")

        docs = ((doc,) if isinstance(doc, str) else doc for doc in texts)
        if num_workers > 1:
            token_freqs = self._count_parallel(docs, num_workers, piece_chars)
        else:
            token_freqs = Counter()
            for chunks in docs:
                token_freqs.update(self._pretokenize_chunks(chunks))
        word_freqs: dict[tuple[str, ...], int] = {}
        for token, freq in token_freqs.items():
            word = self._to_byte_symbols(token)
            word_freqs[word] = word_freqs.get(word, 0) + freq
        del token_freqs

        words = list(word_freqs)
        freqs = [word_freqs[w] for w in words]

        # Pair counts are computed once, then kept up to date incrementally:
        # a merge only touches the words that contain the merged pair, so only
        # those words' pairs are subtracted and re-added. pair_to_words is an
        # index from pair to candidate words (it may hold stale entries for
        # words that no longer contain the pair; merging those is a no-op).
        pair_counts = self._get_pair_counts(word_freqs)
        pair_to_words: dict[tuple[str, str], set[int]] = {}
        for idx, word in enumerate(words):
            for pair in zip(word, word[1:]):
                pair_to_words.setdefault(pair, set()).add(idx)

        # Max-heap over (count, pair) with lazy deletion: an entry is valid
        # only if its count still matches pair_counts. Ties break toward the
        # lexicographically larger pair, same as max(key=(count, pair)).
        heap = [(-count, _Desc(pair)) for pair, count in pair_counts.items()]
        heapq.heapify(heap)

        self.merges = []
        num_merges = vocab_size - len(base_symbols)
        for i in range(num_merges):
            best_pair = None
            while heap:
                neg_count, desc = heapq.heappop(heap)
                if pair_counts.get(desc.pair) == -neg_count:
                    best_pair, best_count = desc.pair, -neg_count
                    break
            if best_pair is None:
                break

            merged_symbol = "".join(best_pair)
            changed: set[tuple[str, str]] = set()
            for idx in pair_to_words.pop(best_pair, ()):
                word = words[idx]
                new_word = self._merge_word(word, best_pair, merged_symbol)
                if new_word == word:
                    continue
                freq = freqs[idx]
                for pair in zip(word, word[1:]):
                    pair_counts[pair] -= freq
                    changed.add(pair)
                for pair in zip(new_word, new_word[1:]):
                    pair_counts[pair] = pair_counts.get(pair, 0) + freq
                    pair_to_words.setdefault(pair, set()).add(idx)
                    changed.add(pair)
                words[idx] = new_word

            for pair in changed:
                count = pair_counts[pair]
                if count > 0:
                    heapq.heappush(heap, (-count, _Desc(pair)))
                else:
                    del pair_counts[pair]

            self.merges.append(best_pair)
            if verbose:
                print(
                    f"merge {i + 1}/{num_merges}: {best_pair} -> "
                    f"{merged_symbol!r} (count={best_count})"
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

    def encode_chunks(self, chunks: Iterable[str]) -> Iterator[int]:
        """Lazily encode one document given as consecutive text chunks.

        Same ids as encode("".join(chunks)), without materializing the text
        or the full id list; callers can write ids out as they arrive.
        """
        for token in self._pretokenize_chunks(chunks):
            for symbol in self._bpe_word(self._to_byte_symbols(token)):
                yield self.vocab[symbol]

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
