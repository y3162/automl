# ASR-guided SEMamba++ v1

16 kHz の音声を SEMamba++ で強調する。学習設定では固定した `nvidia/parakeet-tdt-0.6b-v2` のエンコーダ特徴、認識誤りによるスペクトル損失の重み付け、強調音声から求める TDT 損失を個別に指定できる。

実装は `methods.asr_guided_se.v1` にある。`components.model.settings.backbone` の既定値は `semambapp`。共通の `model.py` がモデルを選択し、`semambapp.py` が生成器・判別器の構築と生成器の `forward` を担当する。新しいモデルを追加するときは、そのモジュールに設定の `Architecture` と構築関数 `build_modules` を定義し、`model.py` の `_BACKBONES` に登録する。`build_modules` が返す辞書のキーがチェックポイント内のモジュール名になる。

## 環境と音源の準備

Linux x86_64 の CUDA 12 環境では次を実行する。`uv` が Python 3.12 の仮想環境を作成し、NeMo 3.0.0、PyTorch と torchaudio 2.6.0+cu124 を導入する。Parakeet の重みは初回使用時に Hugging Face の固定リビジョンから取得する。取得済みなら `HF_HUB_OFFLINE=1` で実行できる。[NeMo のリリース](https://pypi.org/project/nemo-toolkit/3.0.0/)

```sh
scripts/setup_speech_v0.sh
source .venvs/speech-v0/bin/activate
```

このワークスペースでは、参照環境の Mamba 2.3.2.post1 から Mamba-1 の演算を、causal-conv1d 1.7.0 から因果畳み込みを取り込み、コンパイル済みの CUDA 拡張と一緒に `.venvs/speech-v0` にインストールする。学習時の依存先はこの仮想環境内で完結する。他の場所にある同じ参照環境を使うときは `SPEECH_REFERENCE_PYTHON` に Python のパスを指定する。参照環境がない場合は Mamba 2.3.1 と causal-conv1d 1.7.0 の公式 CUDA ホイールを使う。その場合はホイールに対応する新しい glibc が必要になる。

A0〜A7 は参照実装の VoiceBank −15 dB 条件を使う。既存のスナップショットから、発話順、切り出し位置、音量係数、原音の SHA-256、正解テキストを含むマニフェストを準備する。`--corpora` には VCTK、DEMAND48、Babble、SSN の親ディレクトリを渡す。このワークスペースでの準備コマンドは次の通り。

```sh
python -m methods.asr_guided_se.v1.data \
  --snapshot ../audio-db/data/db/snapshots/820178d409919018fdc24e5d3d3e32e2c6d4d181ea11969068fe2917be791250 \
  --corpora /net/spring/work/seki/database
```

学習はローカルのマニフェストと原音を使う。48 kHz で浮動小数点のペアを再構成し、16 kHz に変換してから両方を noisy 音声のピークで正規化する。訓練は28話者の11,572発話、保存点の選択は参照実験と同じ公式テスト824発話のコーパス WER を使う。この WER は選択に使ったデータ上の値となる。

−10 dB スナップショットにも同じ準備コマンドが使える。出力した `voicebank_m10db.jsonl` を `components.dataset.settings.manifest` に指定する。配布 WAV のペアを使う場合は `.env` の `CORPUS_VOICEBANK_DEMAND` を設定し、次のコマンドで索引とマニフェストを作る。この場合の `manifest` は `data/work/asr_guided_se/voicebank28.jsonl`、`corpus` は `data/corpora/voicebank_demand` とする。

```sh
make setup-data VENV=.venvs/speech-v0 CORPUS=voicebank_demand
python -m methods.asr_guided_se.v1.data
```

## 学習

次の例は単一 GPU で3機能を有効にし、3秒の音声窓で判別器と生成器を1回ずつ更新して保存する。`CUDA_VISIBLE_DEVICES` で使用する GPU を指定できる。

```python
from automl.experiment import Experiment

experiment = Experiment("data/store")
run = experiment.start(experiment.branch(), "configs/asr_guided_se/v1/train.yaml")
print(run, experiment.wait(run)["status"])
```

100エポックの条件は `configs/asr_guided_se/v1/ablation/A0.yaml` から `A7.yaml` で選ぶ。論理バッチは8発話で、A1、A3、A5 は4発話ずつ勾配を蓄積する。最終バッチも実際の発話数で加重する。

| 条件 | エンコーダ特徴 | TDT 損失 | 認識誤りの重み付け |
| --- | --- | --- | --- |
| A0 | 無効 | 無効 | 無効 |
| A1 | 有効 | 無効 | 無効 |
| A2 | 無効 | 有効 | 無効 |
| A3 | 無効 | 無効 | 有効 |
| A4 | 有効 | 有効 | 無効 |
| A5 | 有効 | 無効 | 有効 |
| A6 | 無効 | 有効 | 有効 |
| A7 | 有効 | 有効 | 有効 |

特徴融合は noisy 音声の第15層（0始まり）を128次元に射影し、STFT のフレーム中心に最も近い ASR フレームを dense encoder の直後で結合する。誤りマスクは clean 全発話と強調した窓の認識結果から作る。TDT の教師は中心時刻が窓に入る clean 側のトークンで、勾配は固定した Parakeet を通って生成器に達する。

判別器は MRD と CQT で1個の AdamW を共有し、生成器は別の AdamW を使う。両方とも学習率は 0.0002、betas は (0.8, 0.99)、weight decay は 0.01。エポック終了時に学習率を 0.99 倍する。全条件が有効フレーム上で損失を集計する。参照実装の `all_off` の全フレーム平均は、A0 の `valid_frame_aggregation` を `false` にして選択できる。

## 保存点と音声強調

AutoML のモデル状態は `torch.save` 形式のバイト列で、`generator`、`mrd`、`cqt` に各モジュール本来の PyTorch `state_dict` を格納する。Parakeet は固定リビジョンの外部重みとして読み込む。学習状態には両 optimizer、両 scheduler と乱数状態を保存する。エポック、次の発話位置、更新数は `progress` に記録する。

構築した `SpeechEnhancementModel` の `model.generator(clean, noisy, lengths)` は、強調波形 `audio` と学習用のスペクトルを返す。推論では `model.generator.eval().enhance(waveform, sample_rate)` で、元のサンプリングレートと長さのモノラル波形を取得できる。
