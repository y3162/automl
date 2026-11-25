# 手法の配置と実装の版管理

`src/automl/` には実行基盤と共通 API を役割ごとに置く。たとえば、実行や履歴を管理する Experiment のパッケージ名は `automl.experiment` とする。`src/methods/<手法名>/` には、その手法に固有のモデルや学習手順を置く。複数の手法で同じ Trainer を使う場合は、各設定から同じ実装を参照できる。`src/examples/xor/` は小さな実行例に使う。

## 手法の中で版を分ける目安

初版は `src/methods/baseline_mlp/model.py` のように始める。これからの実験で新旧の実装を選んで使いたいときに、一緒に切り替える処理の範囲に応じて版を分ける。

| 変更内容 | 置き方の例 |
| --- | --- |
| 隠れ層の幅や学習率を変える | 同じ実装を使い、部品の `settings` に値を指定する。実行ごとの確定値は履歴から取得できる。 |
| モデルの計算だけを変える | 初版の `model.py` と新版の `model_v2.py` を置く。新版を使う設定では `components.model.implementation` に `methods.baseline_mlp.model_v2.MLP` を指定する。 |
| モデル、損失関数、正規化を一組で変える | 手法内に `v2/` を設け、`model.py`、`loss.py`、`normalize.py` をまとめる。初版は従来のインポート先から選べるようにする。 |

日々の編集は Git で管理する。過去の実行の再現には、実行 ID に対応する確定設定と、未コミットの変更も含めて当時のソースを復元できるよう `data/store/` に保存した情報を使う。 [実行時ソースの復元](../20_basic_design/06_execution_and_replay.md#別のクローンでの履歴の参照と実験の再実行)

## 別の版のモジュールを使う場合

同じ手法に `v1/` と `v2/` があり、`v2` でも `v1` の正規化を使うなら、そのモジュールを直接 import する。

```python
# src/methods/baseline_mlp/v2/model.py
from ..v1.normalize import normalize
```

この例の依存は `v2 → v1` の一方向である。どちらの版を選んでも当時の正規化が使われるように、`v1.normalize` とそこで呼ぶ補助関数の挙動を保つ。たとえば平均・分散の求め方を変えて比較するなら、`v3/normalize.py` に新しい計算を置き、`v3` から参照する。手法全体で仕様が共通の処理は、必要に応じて手法直下の共通モジュールに置ける。 [実装者による版管理](../20_basic_design/03_configuration_and_implementations.md#実装者による版管理)
