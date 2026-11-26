# MNIST で AutoML の実行経路を試す

[数字分類](classifier/README.md) と [GAN](gan/README.md) は、同じ MNIST 版を共有する CPU 実験。少数サンプルで学習し、データ使用履歴、保存点、評価集計、予測・生成物が実際につながるかを確認する。

## 初回の準備

プロジェクトルートで `make venv` を実行する。`.env` がなければ `.env.example` をコピーし、次の設定を実データの置き場に合わせる。このディレクトリには非圧縮の IDX ファイル 4 個（train と t10k の images・labels）を置く。

```sh
MNIST_ROOT=/net/spring/work/seki/database
```

```sh
make setup-data
readlink data/corpora/mnist
. .venvs/test/bin/activate
export CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
```

`make setup-data` は IDX ヘッダー、件数、ラベルを検査し、`data/work/mnist/metadata.sqlite3` にサンプルごとの位置・ラベルと各ファイルの SHA-256 を記録する。配布時の訓練データ 60,000 件のうち先頭 55,000 件が train、末尾 5,000 件が validation。公式テスト 10,000 件が test に対応する。設定の件数制限は各分割を開いた後に適用する。

## 学習と検証

上の環境を有効にしたシェルで実行する。各タスクのモデル計算・乱数生成は NumPy を使う。実行 ID と段階ごとの状態が表示される。

```sh
python - <<'PY'
from automl.experiment import Experiment

experiment = Experiment("data/store")
for task in ("classifier", "gan"):
    config = f"configs/examples/mnist/{task}/train.yaml"
    run = experiment.start(experiment.branch(), config)
    status = experiment.wait(run)
    print(task, run, status["stages"])
    if status["status"] != "succeeded":
        raise SystemExit(f"{run} ended with {status['status']}: {status['error']}")
    check = experiment.verify(run)
    if not check["ready"]:
        raise SystemExit(check["issues"])
PY
```

`automl history --summary` で分岐元・評価に使った保存点・集計値を一覧できる。`automl verify` は正本全体の履歴と成果物を照合する。復元時は `data/store/` をコピーし、同じ版の外部 IDX を復元先の `.env` に設定して `make setup-data` を実行する。各実行の `automl verify --run RUN` が `ready: true` を返すことを確認する。

実データの配置に依存しない回帰試験は `python -m pytest -q tests/examples/mnist`、全体は `make test` で実行できる。

MLflow を使う場合は、同じ仮想環境で `python -m pip install -e '.[mlflow]'` を実行し、`python -m automl.mlflow` で同期する。表示用の DB は `data/work/mlflow/tracking.db` に作られる。分類では訓練損失と正解率、GAN では生成器・識別器それぞれの訓練損失と、検証・テストの識別器損失を比較できる。

## 次に追加したい機能

- **評価結果の一括登録**。全件評価すると、現在の `record_result` は画像ごとに書き込みトランザクションを開く。バッチ単位で登録でき、再送時も各測定 ID で重複を判定できれば、10,000 件の公式テストにも同じ履歴を残しやすい。
- **MLflow に表示する成果物の選択**。GAN は検証用とテスト用の生成画像を正本に保存する。設定で `generated.pgm` のような少数の出力を選んでプレビューできると、実行間で生成結果を比較しやすい。
