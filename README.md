# automl

機械学習の実験をローカルで実行し、実行 ID から使用データ、設定、評価結果、保存点をたどるための実験基盤。人もエージェントも、同じ CLI または Python API で学習を開始し、保存点から再開・再評価できる。[Experiment の公開 API](docs/30_detailed_design/01_experiment_api.md)

## 最初の実験（XOR）

Python 3.11 以上を用意し、リポジトリのルートで次を実行する。訓練・検証・テスト用のデータは各4件で、設定ファイルに含まれる。[XOR の設定](configs/examples/xor/inline.yaml)

```bash
make venv
source .venvs/test/bin/activate

# CLI が返す JSON から ID を取り出す。
json_field() {
    python -c 'import json, sys; print(json.load(sys.stdin)[sys.argv[1]])' "$1"
}

branch_id=$(automl branch | json_field branch_id)
run_id=$(automl start --branch "$branch_id" configs/examples/xor/inline.yaml | json_field run_id)
automl wait "$run_id"
automl history --run "$run_id" --summary
```

`branch` は実験履歴に新しい枝を作り、`start` はその枝で学習と評価を開始する。`wait` の `status` が `succeeded` になれば完了。同じ実行 ID で後から `history` を呼ぶと、評価に使った保存点と集計結果を確認できる。

## 実験例とデータ準備

- **XOR のモデル比較**。同梱の CSV を使った比較では、TinyMLP と入力の積を使うロジスティック回帰が、ともにテスト96件に全問正解した。[設定・学習経過・実行 ID](docs/80_examples/xor/README.md)
- **MNIST**。CPU で数字分類と GAN を学習し、保存点の再評価と成果物の確認まで試せる。[MNIST の CPU 実験](docs/80_examples/mnist/README.md)
- **音声コーパスの準備**。`.env.example` をもとに利用するコーパスのパスを `.env` に書き、`make setup-data` で検索用の DuckDB 索引を作る。LibriSpeech、VoiceBank+DEMAND など19種類に対応する。[音声コーパスの準備](docs/90_operations_and_maintenance/02_make_and_scripts.md#音声コーパスの準備)
- **ASR-guided SEMamba++**。VoiceBank 音声の強調を学習し、Parakeet の特徴・TDT 損失・認識誤りによる重み付けを組み合わせた8条件を比較できる。[環境構築と実験設定](docs/80_examples/asr_guided_se/README.md)

## 開発と運用

`make venv` の後、`make test` でテストを実行する。実装の配置、実験履歴の保存と復元、エージェントの作業指示は次を参照する。[手法の配置と版管理](docs/90_operations_and_maintenance/03_source_layout_and_versions.md)・[データの配置とバックアップ](docs/90_operations_and_maintenance/01_data_and_backup.md)・[エージェント向け作業指示](AGENTS.md)
