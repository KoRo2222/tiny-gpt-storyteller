# tiny-gpt-storyteller

BPEトークナイザーからTransformerモデル、学習まで全てゼロから自前実装する小型LLM。毎晩前の晩の続きを語る「千夜一夜」型の語り部Botを目指している。

## 構成

- **トークナイザー**(`src/storybot/tokenizer`) — バイトレベルBPEを学習アルゴリズムから自前実装。GPT-2方式のプリトークナイズを日本語向けに拡張(ひらがな・カタカナ・漢字・記号の境目で分割)、ペア頻度の差分更新による高速な学習、コーパスを分割して読み込むストリーミング処理(学習・エンコードとも)、事前トークン化のマルチプロセス並列化、`<|endoftext|>`による文書区切り
- **モデル**(`src/storybot/model`) — LLaMA方式に改良したデコーダー型Transformer(`GPT`)。GPT-2からの主な変更点は以下
  - **RoPE**(回転位置埋め込み) — 位置埋め込みテーブルの代わりに、クエリ/キーを位置に比例した角度で回転させ、注意スコアが相対位置だけに依存するようにする
  - **SwiGLU** — FFNをSiLUを通したゲート枝と線形枝の要素積にし、隠れ次元を 8/3·d に縮めて通常の4倍幅FFNとパラメータ数をそろえる
  - **RMSNorm** — 平均を引かずRMSだけで割り、学習可能なゲインを掛ける正規化(統計量は半精度入力でもfloat32で計算)。Pre-Norm構成
  - バイアスなし、dropoutなし、入力埋め込みと出力層(LM head)の重みは共有しない
  - 残差に書き込む射影の初期値を層数に応じて縮小(GPT-2方式)
  - KVキャッシュ付きの生成(温度・top-k・`<|endoftext|>`での停止)

## 現状

トークナイザーとモデル本体を実装済み。事前学習・語り部としての対話部分はこれから。

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
