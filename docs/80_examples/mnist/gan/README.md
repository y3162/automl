# MNIST の GAN

[共通手順](../README.md)の GAN 設定は、訓練分割の先頭 512 枚から毎回 32 枚を復元抽出して、識別器と生成器を交互に 24 回更新する。8 更新ごとに両モデルの重み・進捗・乱数状態を一緒に保存する。最終保存点の validation と test ではそれぞれ実画像 16 枚と、固定シードから生成した 16 枚の組を測定する。集計値は実画像・生成画像に対する識別器の二値交差エントロピーの平均。[学習設定](../../../../configs/examples/mnist/gan/train.yaml)・[評価設定](../../../../configs/examples/mnist/gan/evaluate.yaml)

## テストの生成画像を開く

共通手順で表示された GAN の実行 ID をシェル変数 `RUN` に設定する。テスト評価の成果物 ID から、16 枚を並べたグレースケール画像のパスを取得できる。

```sh
python - "$RUN" <<'PY'
import sys
from automl.experiment import Experiment

ex = Experiment("data/store", read_only=True)
run = sys.argv[1]
evaluations = ex.list_evaluations(run, limit=100)["evaluations"]
test = next(e for e in evaluations if e["conditions"].get("split") == "test")
grid_id = test["result"]["sample_grid"]
artifact = next(a for a in ex.history(run)["artifacts"] if a["id"] == grid_id)
print(ex.root / artifact["files"]["generated.pgm"]["path"])
PY
```

別の枝で保存点から再生成するには、`automl branch` で得た枝 ID を `BRANCH` に設定し、`automl start --branch "$BRANCH" --evaluate "$RUN"` を実行する。`automl wait` の後に同じ手順で結果を参照できる。

## 更新数と学習率の比較

初期化、訓練画像512枚、バッチサイズ32を固定した。生成器の比較には、実画像と生成画像をそれぞれ平均し、2枚の平均画像の対応する784画素について平均絶対誤差（MAE）を求めた。生成側は保存した PGM を28×28の画像に切り分け、画素値を255で割った。検証には先頭16枚と乱数シード11を使った。

| 更新数 | 学習率 | 保存間隔（更新） | 検証の識別器損失 | 検証の平均画像 MAE |
|---:|---:|---:|---:|---:|
| 24 | 0.05 | 8 | 0.08374 | 0.38124 |
| 96 | 0.05 | 32 | 0.02703 | 0.37880 |
| 96 | 0.2 | 32 | 0.01008 | 0.36453 |
| 384 | 0.2 | 96 | 0.01506 | 0.28210 |

検証の平均画像 MAE で384更新を選んだ。テストの先頭64枚とシード12では、基準の24更新と比べて MAE が 0.38398 から 0.29236 に下がった。生成画像では背景が暗くなり、中央に明るい領域が現れた。個々の数字の輪郭はまだ不鮮明だった。共通手順の学習設定を切り替えると、この比較で選んだ条件を実行できる。[384更新の設定](../../../../configs/examples/mnist/gan/384_steps.yaml)

このストアでのテスト評価の実行 ID は、基準が `R2abb00691a2948ceb1d9260e85d2723b`、384更新が `R59a2d1e53d204008bde2a3c841c596c2`。各実行の `history` の `decision_recorded` に平均画像 MAE と画像の成果物 ID を記録した。上の手順で両方の生成画像を開ける。
