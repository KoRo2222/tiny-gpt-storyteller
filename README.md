# tiny-gpt-storyteller

BPEトークナイザーからGPT-2アーキテクチャ、学習まで全てゼロから自前実装する小型LLM。毎晩前の晩の続きを語る「千夜一夜」型の語り部Botを目指している。

## 構成

- **トークナイザー**(`src/storybot/tokenizer`) — バイトレベルBPEを学習アルゴリズムから自前実装。GPT-2方式のプリトークナイズを日本語向けに拡張(ひらがな・カタカナ・漢字・記号の境目で分割)、ペア頻度の差分更新による高速な学習、`<|endoftext|>`による文書区切り

## 現状

トークナイザーのみ実装済み。モデル・学習・語り部としての対話部分はこれから。

## セットアップ

```
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt

# data/corpus 以下のコーパスで学習し、data/tokenizer.json に保存
.venv/Scripts/python scripts/train_tokenizer.py --vocab-size 1024

# テスト実行
.venv/Scripts/python -m pytest tests/ -v
```

## License

Copyright (c) 2026 KoRo2. All rights reserved.
