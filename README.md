# automl

`make setup-data` で、`.env` に指定した LibriSpeech、VoiceBank+DEMAND など19種類の音声コーパスから検索用の DuckDB 索引を作成できる。[音声コーパスの準備](docs/90_operations_and_maintenance/02_make_and_scripts.md#音声コーパスの準備)

[MNIST の CPU 実験](docs/80_examples/mnist/README.md)では、外部データの準備から数字分類・GAN の学習、保存点の再評価と成果物の確認までを実行できる。

[ASR-guided SEMamba++](docs/80_examples/asr_guided_se/README.md) は、VoiceBank 音声を使った学習と、Parakeet の特徴・TDT 損失・認識誤り重み付けの8条件を実行できる。

## XOR のモデル比較

同じ CSV の訓練320件を使い、TinyMLP と、2つの入力の積を特徴にするロジスティック回帰を比較した。TinyMLP は25更新ごとの検証で第50更新に初めて96/96に達し、第100更新で学習を終了した。積を使うモデルは第1更新で96/96に達して早期終了した。別のテスト96件では両モデルとも全問正解だった。[TinyMLP の設定](configs/examples/xor/full_batch.yaml)・[積を使うモデルの設定](configs/examples/xor/product_target.yaml)

テスト評価の実行 ID は TinyMLP が `R764eccd695eb4f6d9a0c27829973c77e`、積を使うモデルが `R6212d10d2ad744a19561b42701e17d4c`。`automl history --run 実行ID --summary` で評価に使った保存点と結果を確認できる。
