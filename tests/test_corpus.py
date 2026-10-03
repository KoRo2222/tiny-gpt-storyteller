import json
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from storybot.data import expand_corpus_paths, iter_documents, keep_story


def _write_jsonl(path, stories, field="text_ja"):
    path.write_text(
        "\n".join(json.dumps({field: s, "text_en": "x"}, ensure_ascii=False) for s in stories)
        + "\n",
        encoding="utf-8",
    )


def _materialize(docs):
    return [d if isinstance(d, str) else "".join(d) for d in docs]


def test_jsonl_yields_one_document_per_line(tmp_path):
    stories = ["むかしむかし、王様がいました。", "ある日、猫が歌いました。\n\n終わり。"]
    _write_jsonl(tmp_path / "s.jsonl", stories)
    assert _materialize(iter_documents([tmp_path / "s.jsonl"])) == stories


def test_untranslated_and_empty_stories_are_dropped(tmp_path):
    stories = [
        "むかしむかし、王様がいました。",
        "",
        "   ",
        "Once upon a time there was a king.",  # left in English
        "リリーはおもちゃで遊びました。",
    ]
    _write_jsonl(tmp_path / "s.jsonl", stories)
    stats = Counter()
    docs = _materialize(iter_documents([tmp_path / "s.jsonl"], stats=stats))
    assert docs == [stories[0], stories[4]]
    assert stats == Counter(kept=2, dropped=3)


def test_keep_story_threshold():
    assert keep_story("ボブはABCを習いました。", max_latin_ratio=0.3)
    assert not keep_story("ボブはABCを習いました。", max_latin_ratio=0.1)


def test_max_docs_stops_early_across_files(tmp_path):
    _write_jsonl(tmp_path / "a.jsonl", ["一話目。", "二話目。"])
    _write_jsonl(tmp_path / "b.jsonl", ["三話目。", "四話目。"])
    docs = _materialize(iter_documents([tmp_path], max_docs=3))
    assert docs == ["一話目。", "二話目。", "三話目。"]


def test_txt_file_is_one_streamed_document(tmp_path):
    text = "むかしむかし。" * 100
    (tmp_path / "story.txt").write_text(text, encoding="utf-8")
    (docs,) = list(iter_documents([tmp_path], chunk_chars=10))
    assert not isinstance(docs, str)  # streamed in chunks, not read whole
    assert "".join(docs) == text


def test_directories_expand_to_corpus_files_only(tmp_path):
    (tmp_path / "a.txt").write_text("あ", encoding="utf-8")
    (tmp_path / "b.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "notes.md").write_text("x", encoding="utf-8")
    assert [p.name for p in expand_corpus_paths([tmp_path])] == ["a.txt", "b.jsonl"]
    with pytest.raises(FileNotFoundError):
        expand_corpus_paths([tmp_path / "missing.jsonl"])
