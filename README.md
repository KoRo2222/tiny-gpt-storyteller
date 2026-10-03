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
  - 混合精度でも数値が崩れやすい箇所(Attentionのsoftmax、RoPEの回転、RMSNormの統計量、損失)はfloat32で計算
- **データ**(`src/storybot/data`) — コーパスをストリーミングでトークン化し、文書を`<|endoftext|>`で区切った1本のトークン列としてバイナリ(語彙が収まればuint16)に書き出す。学習時はメモリマップで読み、末尾の一定割合を検証用に分離(訓練と重ならない)。訓練バッチはランダム位置の窓、評価は毎回同じ固定の窓
- **学習**(`src/storybot/train`) — 次トークン予測による事前学習。AdamW(重み減衰は行列のみ、RMSNormのゲインには掛けない)。学習率スケジュールはD2Z(Decay-to-Zero)。線形ウォームアップでピークまで上げた後、最終ステップでちょうど0になるよう線形に減衰させる。混合精度学習(autocastでbf16/fp16計算、重みとオプティマイザー状態はfloat32のまま、fp16では勾配スケーリングを行いクリップ前にunscale)。`auto`指定ではGPUならbf16(非対応ならfp16)、CPUならfp32を選ぶ(bf16命令を持たないCPUではbf16の方が大幅に遅いため)。一定間隔で検証損失を測り、モデル・オプティマイザー・スケジューラー・バッチ乱数の状態をチェックポイントに保存。中断後に再開しても、中断しなかった場合と同じ結果になる

## 現状

トークナイザー、モデル本体、事前学習の一式(データ準備・学習ループ・チェックポイント・生成)を実装済み。物語コーパスでの本格的な事前学習と、語り部としての対話部分はこれから。

## セットアップ

```
python -m venv .venv
# CPU版torchを明示的に入れる(付けないとCUDA同梱の巨大なwheelが入る)
.venv/Scripts/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/pip install -r requirements.txt

# data/corpus 以下の .txt(1ファイル=1文書)でトークナイザーを学習し、data/tokenizer.json に保存
.venv/Scripts/python scripts/train_tokenizer.py --vocab-size 1024

# コーパスをトークン化し、data/tokens.bin に保存
.venv/Scripts/python scripts/prepare_data.py

# 事前学習(data/checkpoint.pt に保存。--resume で中断したところから再開)
.venv/Scripts/python scripts/pretrain.py --steps 2000

# 学習済みチェックポイントから物語を生成
.venv/Scripts/python scripts/generate.py --prompt "むかしむかし、"

# テスト実行
.venv/Scripts/python -m pytest tests/ -v
```

## License

Copyright (c) 2026 KoRo2. All rights reserved.
