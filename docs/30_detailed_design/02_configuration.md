# configの詳細設計（SS）

Experimentが起動前に読み込むYAMLの構造、継承、パス解決、確定後の保存を定める。[設定と個別実装](../20_basic_design/03_configuration_and_implementations.md)

## YAMLの構造

| 項目 | 内容 |
|---|---|
| `plan` | 実行する処理の順序。`[train]`、`[evaluate]`、`[train, evaluate]` のいずれかを指定する |
| `components` | 部品名をキーにした辞書。各部品に個別実装のインポート先 `implementation` を指定する。`settings` は実装に渡す辞書で、省略時は空の辞書とする |
| `extends` | 継承する親YAMLへのパス。継承する場合に指定する |

親configを展開した後の `plan` と `components` を検証する。学習には `trainer`、評価には `evaluator` の定義を用いる。他の部品に必要な名前と設定は、利用する学習・評価手順が定める。分岐元や初期化・評価に使う保存点は起動時に指定する。[Experimentの公開インターフェース](01_experiment_api.md)

## ハイパーパラメータ

モデル構成や学習に使うハイパーパラメータは、個別実装に固定する値を減らし、親または子configの `settings` に明示することを推奨する。どの値を設定として公開するか、どの粒度で指定できるようにするかは、個別実装の作者が決める。

共通の値は親configに置ける。既定値が自然に定まる場合は、個別実装の作者が設けてもよい。継承と上書きの解決後、指定されていない項目に実装の既定値を適用する。実装が必須とする項目に値がなければ、受付前に理由を返す。

## 継承と上書き

1. `extends` に指定された親を読み込み、その親の継承も先に解決する。親を取得できない場合や、継承が循環する場合は、実行の受付前に理由を返す。
2. 辞書はキーごとに継承し、子で指定した項目を上書きする。リストは子のリスト全体で置き換える。`plan` もこのルールで扱う。
3. 部品の `implementation` を子で別の実装に差し替えた場合は、その部品の設定を丸ごと置き換える。差し替え先の実装が定める既定値と、子で明示した値から適用設定を決める。

保存点を再評価する場合は、引き継ぐ評価条件を基に、継承元の親YAMLから起動時の子YAMLまで、各ファイルで明示した項目を順に適用する。同じ項目は子YAMLの指定を優先し、同じ実装でどのYAMLにも指定がない項目は引き継いだ値を使う。途中で部品の実装を差し替えたときは、その段階で上記の差し替え規則を適用する。

## パスの解決

YAMLに記載するファイルパスは、`configs/` や `data/` があるプロジェクトルートからの相対パスとする。`extends` と、個別実装がファイルパスとして受け取る設定項目に同じ基準を適用する。再現実験の YAML は `configs/` に置く。公式の分割やデータ作成手順を使う場合は `recipes/` の定義を、外部コーパスのファイルを読む場合は `data/corpora/<corpus_id>/` を通るパスを指定する。 [正本と実験の定義](../90_operations_and_maintenance/01_data_and_backup.md#正本と実験の定義)

Python APIの `source` がプロジェクトルートを指定する。省略時とCLIからの起動時はカレントディレクトリを使う。Experimentは、別プロセスで実行する際にも、受付時に確定したプロジェクトルートをパス解決に使う。`root` は、履歴や成果物を保存するデータルートを指定する。

```yaml
# configs/examples/xor/full_batch.yaml の一部
plan: [train, evaluate]
components:
  dataset:
    implementation: examples.xor.components.data.CsvDataset
    settings:
      path: data/corpora/xor_csv/samples.csv
  model:
    implementation: examples.xor.components.model.TinyMLP
    settings:
      width: 8
```

```yaml
# configs/examples/xor/minibatch.yaml
extends: configs/examples/xor/full_batch.yaml
components:
  trainer:
    implementation: examples.xor.components.learning.MiniBatch
    settings:
      rounds: 20
      rate: 0.35
      batch_size: 32
```

## 確定済みconfig

Experimentは、継承、上書き、実装の既定値を反映して検証した設定を、実行ごとの `runs/<実行ID>/config.yaml` としてデータルートに保存する。保存するYAMLに `extends` は含めず、実際に適用した値をすべて記録する。パスはプロジェクトルート基準のまま保持する。

確定済みconfigは、元の親ファイルを参照せずに読み込める。次の試行で親configとして指定することもできる。別のクローンでは、同じディレクトリ配置のもとで同じデータの版を取得できることを確認する。[履歴と成果物](../20_basic_design/02_history_and_artifacts.md)

同一 YAML マッピングでキーを重複して定義すると、受付前にファイル名・行番号・キー名を含む `invalid_config` を返す。親 YAML の設定は子 YAML から上書きできる。Trainer が検証結果の `selection_metric` を使う場合、検証用 Evaluator は `selection_metrics(settings)` で選択可能な指標名を宣言する。設定検証では指定名をこの一覧と照合する。
