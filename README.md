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
- **データ**(`src/storybot/data`) — コーパス(1行1話のJSONL、または1ファイル1文書のテキスト。空の話や英語が翻訳されずに残った話は除外)をストリーミングでトークン化し、文書を`<|endoftext|>`で区切った1本のトークン列としてバイナリ(語彙が収まればuint16)に書き出す。学習時はメモリマップで読む。検証用は別ファイルで渡すか、なければ末尾の一定割合を分離(訓練と重ならない)。訓練バッチはランダム位置の窓、評価は毎回同じ固定の窓
- **学習**(`src/storybot/train`) — 次トークン予測による事前学習。AdamW(重み減衰は行列のみ、RMSNormのゲインには掛けない)。学習率スケジュールはD2Z(Decay-to-Zero)。線形ウォームアップでピークまで上げた後、最終ステップでちょうど0になるよう線形に減衰させる。混合精度学習(autocastでbf16/fp16計算、重みとオプティマイザー状態はfloat32のまま、fp16では勾配スケーリングを行いクリップ前にunscale)。`auto`指定ではGPUならbf16(非対応ならfp16)、CPUならfp32を選ぶ(bf16命令を持たないCPUではbf16の方が大幅に遅いため)。一定間隔で検証損失を測り、モデル・オプティマイザー・スケジューラー・バッチ乱数の状態をチェックポイントに保存。中断後に再開しても、中断しなかった場合と同じ結果になる

## 現状

トークナイザー、モデル本体、事前学習の一式(データ準備・学習ループ・チェックポイント・生成)を実装済み。

TinyStories-JAの訓練データ1/4(約53万話、1.39億トークン)のうち約1,600万トークンを使い、約900万パラメータのモデル(d=256、6層、8ヘッド、語彙8,000)をCPUで約1時間事前学習した。検証損失は2.09(パープレキシティ8.1)。「むかしむかし、ルーシーという名前の小さな女の子がいました。」のような童話らしい文体と、文法的に自然な日本語の文は書けるが、文をまたいだ話の筋(誰が何をしているか)はまだ崩れやすい。

語り部としての対話部分(前の晩の続きを語る仕組み)はこれから。

## セットアップ

```
python -m venv .venv
# CPU版torchを明示的に入れる(付けないとCUDA同梱の巨大なwheelが入る)
.venv/Scripts/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/pip install -r requirements.txt

# コーパス: TinyStories-JA(英語のTinyStoriesを日本語に機械翻訳した子ども向けの短い物語、CDLA-Sharing-1.0)
# https://huggingface.co/datasets/shibatch/TinyStories-JA から data/raw/ にダウンロード
mkdir -p data/raw
curl -L -o data/raw/train-00000-of-00004.jsonl https://huggingface.co/datasets/shibatch/TinyStories-JA/resolve/main/data/train-00000-of-00004.jsonl
curl -L -o data/raw/validation-00000-of-00001.jsonl https://huggingface.co/datasets/shibatch/TinyStories-JA/resolve/main/data/validation-00000-of-00001.jsonl

# トークナイザーを学習し、data/tokenizer.json に保存
# (コーパスは .jsonl(1行1話、本文は text_ja)と .txt(1ファイル1文書)のどちらも可)
.venv/Scripts/python scripts/train_tokenizer.py --corpus data/raw/train-00000-of-00004.jsonl --max-docs 200000 --vocab-size 8000

# 訓練用・検証用をそれぞれトークン化し、data/train.bin, data/val.bin に保存
.venv/Scripts/python scripts/prepare_data.py --corpus data/raw/train-00000-of-00004.jsonl --out data/train.bin
.venv/Scripts/python scripts/prepare_data.py --corpus data/raw/validation-00000-of-00001.jsonl --out data/val.bin

# 事前学習(data/checkpoint.pt に保存。--resume で中断したところから再開)
.venv/Scripts/python scripts/pretrain.py --steps 4000 --eval-interval 250

# 学習済みチェックポイントから物語を生成
.venv/Scripts/python scripts/generate.py --prompt "むかしむかし、"

# テスト実行
.venv/Scripts/python -m pytest tests/ -v
```

## License

Copyright (c) 2026 KoRo2. All rights reserved.
