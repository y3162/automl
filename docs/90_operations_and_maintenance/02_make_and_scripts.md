# Makefile と運用スクリプト

Makefile は、人がプロジェクトルートで定型作業を実行するための入口とする。通常の操作はターゲット名だけで完結し、その計算機の仮想環境と標準のデータ配置を使う。実験ごとの起動・再開や条件の指定には、Experiment の公開 CLI・Python API を使う。 [Experiment の公開インターフェース](../30_detailed_design/01_experiment_api.md)

## Make ターゲット

以下は実装予定の4操作で、ターゲット名は運用設計上の案である。

| 操作 | 内容 |
| --- | --- |
| `make venv` | プロジェクト用の Python 仮想環境を用意し、データ準備・テスト・可視化に必要な依存パッケージを導入する。他のターゲットもこの環境を使う。 |
| `make setup-data` | `.env` に実体パスを設定したコーパスのリンクを `data/corpora/<corpus_id>` に作成・更新し、そのコーパスのメタデータを作成する。 |
| `make test` | `tests/` に実装したテストを実行し、結果を端末に表示する。テストが失敗した場合は終了ステータスに反映する。 |
| `make mlflow-ui` | AutoML の正本から比較用の記録を MLflow Tracking に反映し、結果を閲覧する UI を起動する。閲覧先を端末に表示する。 |

新しい計算機では、まず `make venv` を実行する。`.env.example` をひな形に `.env` を記入し、`make setup-data` を実行すると、リンク先のコーパスから相対パスや配布元の ID などを取得できる。メタデータの項目と読み取り方はコーパスごとの実装が定める。同じ入力で繰り返し実行しても同じ内容の索引を得られるようにする。再構築できる索引 DB は `data/work/` に置き、`data/store/` に確定したデータセット版、登録済みレシピ、人手による訂正を保持する。 [コーパスの索引](../30_detailed_design/05_dataset.md#コーパスの索引)

テストで使う小さな入力例は Git で管理する。`make test` は `make venv` の後、実データ用の `.env` を設定する前でも実行できる。テスト用の DB と正本には `data/mock/store/`、中間ファイルには `data/mock/work/` を使う。 [テスト用の正本](01_data_and_backup.md#テスト用の正本)

MLflow への反映は、Experiment の実行 ID を MLflow run に対応付ける同期処理が担う。表示用 DB と成果物のコピーは `data/work/` に置く。バックアップを復元した直後も、正本から同期して `make mlflow-ui` で実験結果を比較できるようにする。 [MLflow Tracking への記録](../20_basic_design/02_history_and_artifacts.md#mlflow-trackingへの記録)

## scripts/ と tools/

- `scripts/` には、このプロジェクトで繰り返す作業の手順を置く。たとえば `make setup-data` から呼ぶスクリプトが、リンクを用意した後に各コーパスのメタデータ作成処理を呼ぶ。Makefile の記述は作業の呼び出しに絞る。
- `tools/` には、入力を指定して単独で使う道具を置く。たとえば、二つの公式分割表から追加・除外されたサンプルを調べる道具が該当する。

両方のディレクトリのコードを Git で管理する。
