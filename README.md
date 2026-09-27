# tiny-gpt-storyteller

BPEトークナイザーからTransformerモデル、学習まで全てゼロから自前実装する小型LLM。毎晩前の晩の続きを語る「千夜一夜」型の語り部Botを目指している。

## 構成

- **トークナイザー**(`src/storybot/tokenizer`) — バイトレベルBPEを学習アルゴリズムから自前実装。GPT-2方式のプリトークナイズを日本語向けに拡張(ひらがな・カタカナ・漢字・記号の境目で分割)、ペア頻度の差分更新による高速な学習、コーパスを分割して読み込むストリーミング処理(学習・エンコードとも)、事前トークン化のマルチプロセス並列化、`<|endoftext|>`による文書区切り
- **モデル**(`src/storybot/model`) — RoPE(回転位置埋め込み)。クエリ/キーを位置に比例した角度で回転させ、注意スコアが相対位置だけに依存するようにする。生成時に新しいトークンだけを正しい位置で回転できるオフセット指定に対応

## 現状

トークナイザーと、モデルの部品としてRoPEを実装済み。Attentionなど残りのモデル部分・学習・語り部としての対話部分はこれから。

## セットアップ

```
python -m venv .venv
# CPU版torchを明示的に入れる(付けないとCUDA同梱の巨大なwheelが入る)
.venv/Scripts/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/pip install -r requirements.txt

# data/corpus 以下のコーパスで学習し、data/tokenizer.json に保存
.venv/Scripts/python scripts/train_tokenizer.py --vocab-size 1024

# テスト実行
.venv/Scripts/python -m pytest tests/ -v
```

## License

Copyright (c) 2026 KoRo2. All rights reserved.
