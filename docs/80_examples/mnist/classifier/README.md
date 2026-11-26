# MNIST の数字分類

[共通手順](../README.md)の分類設定は、訓練分割の先頭 256 枚をエポックごとにシャッフルし、32 枚ずつ 2 エポック学習する。画像を 784 次元に展開し、10 クラスのソフトマックス回帰をミニバッチ SGD で更新する。各エポックで validation の 32 件を測定し、正解率が最も高い保存点を test の 64 件に適用する。実際に使ったサンプル ID はミニバッチごとに記録する。[学習設定](../../../../configs/examples/mnist/classifier/train.yaml)・[評価設定](../../../../configs/examples/mnist/classifier/evaluate.yaml)

## 保存した予測を再採点する

共通手順で表示された分類実行の ID をシェル変数 `RUN` に設定する。次の例は、テストで保存した予測 JSON の成果物 ID を使って再採点する。

```sh
python - "$RUN" <<'PY'
import sys
from automl.experiment import Experiment

ex = Experiment("data/store")
run = sys.argv[1]
evaluations = ex.list_evaluations(run, limit=100)["evaluations"]
test = next(e for e in evaluations if e["result"].get("prediction_artifact"))
again = ex.start(ex.branch(), "configs/examples/mnist/classifier/evaluate.yaml",
                 prediction_artifact=test["result"]["prediction_artifact"])
status = ex.wait(again)
assert status["status"] == "succeeded", status["error"]
print(ex.history(again, summary=True)["runs"][0]["evaluations"])
PY
```

保存点から学習条件を変えて試す場合は `automl start configs/examples/mnist/classifier/train.yaml --fork "$RUN"` で枝を作る。`automl stop` で学習途中に停止したら、`automl wait` で中断状態と保存点を確認し、`automl resume BRANCH` で同じ枝の学習を続けられる。

## 線形モデルと MLP の比較

各試行は新しい枝で初期化した。モデルのシードを7、シャッフルのシードを17、学習率を0.25、バッチサイズを32に固定し、毎エポック、同じ検証分割の先頭128枚で正解率を測った。MLP は784次元の入力と10クラスの出力の間に、幅128の ReLU 隠れ層を持つ。

| モデル | 訓練枚数 | 学習エポック数 | 選ばれたエポック | 検証正解数 |
|---|---:|---:|---:|---:|
| 線形（基準） | 256 | 2 | 2 | 100/128 |
| 線形 | 2,048 | 2 | 1 | 111/128 |
| MLP | 2,048 | 2 | 1 | 112/128 |
| 線形 | 2,048 | 6 | 6 | 115/128 |
| MLP | 2,048 | 6 | 5 | 118/128 |

6エポックの線形モデルと MLP では、使用履歴に記録した384バッチの画像 ID と順序がすべて一致した。MLP は第6エポックで117/128だったため、第5エポックの保存点を採用した。共通手順の学習設定を `configs/examples/mnist/classifier/mlp.yaml` に切り替えると、検証で選んだ保存点を公式テスト全件に適用できる。[MLP の学習・評価設定](../../../../configs/examples/mnist/classifier/mlp.yaml)

| 保存点 | 公式テスト正解数 | 正解率 | テスト評価の実行 ID |
|---|---:|---:|---|
| 基準の線形モデル・第2エポック | 7,199/10,000 | 71.99% | `Rdd732b6d76f34e5aafc56e575296b753` |
| 2,048枚の線形モデル・第6エポック | 8,817/10,000 | 88.17% | `Rc4b8222b4ac447368f3c722ed1aece84` |
| 2,048枚の MLP・第5エポック | 8,987/10,000 | 89.87% | `Re9d0306e42ab4fe3b65f60f199763379` |

同じ2,048枚で学習した2モデルの予測を画像 ID ごとに突き合わせると、線形モデルの誤りを MLP が正した画像は389枚、逆に MLP が誤った画像は219枚で、差し引き170枚の改善だった。MLP に残った誤りで最も多かったのは、数字の3を5と予測した122枚だった。採用した保存点 ID は `C92e3f59744b74406bdae5cc3fe0859aa`。
