# Makefile と運用スクリプト

Makefile は、人がプロジェクトルートで定型作業を実行するための入口とする。通常の操作はターゲット名だけで完結し、その計算機の仮想環境と標準のデータ配置を使う。実験ごとの起動・再開や条件の指定には、Experiment の公開 CLI・Python API を使う。 [Experiment の公開インターフェース](../30_detailed_design/01_experiment_api.md)

## Make ターゲット

プロジェクトルートで使うターゲットの用途を示す。

| 操作 | 内容 |
| --- | --- |
| `make venv` / `make venv-examples` | XOR・MNIST と共通基盤のテスト用に `.venvs/examples` を作る。 |
| `make venv-mlflow` | MLflow のテスト用に `.venvs/mlflow` を作る。 |
| `make setup-data` | `.env` に実体パスを設定したコーパスのリンクを `data/corpora/<corpus_id>` に作成・更新し、そのコーパスのメタデータを作成する。 |
| `make test` | `tests/` に実装したテストを実行し、結果を端末に表示する。テストが失敗した場合は終了ステータスに反映する。 |
| `make clean-mock` | テスト用の `data/mock/` を削除する。 |
| `make mlflow-ui`（予定） | AutoML の正本から比較用の記録を MLflow Tracking に反映し、結果を閲覧する UI を起動する。閲覧先を端末に表示する。 |

新しい計算機では、まず `make venv` を実行する。`.env.example` をひな形に `.env` を記入し、`make setup-data` を実行すると、リンク先のコーパスから相対パスや配布元の ID などを取得できる。メタデータの項目と読み取り方はコーパスごとの実装が定める。同じ入力で繰り返し実行しても同じ内容の索引を得られるようにする。再構築できる索引 DB は `data/work/` に置き、`data/store/` に確定したデータセット版、登録済みレシピ、人手による訂正を保持する。 [コーパスの索引](../30_detailed_design/05_dataset.md#コーパスの索引)

`make setup-data` は既定で `.venvs/examples` を使い、`VENV` で変更できる。`make test` は共通基盤・コーパス・examples を `.venvs/examples`、MLflow を `.venvs/mlflow`、音声手法を [speech-v0 環境](../80_examples/asr_guided_se/README.md)で実行する。3環境を先に準備する。

テストで使う小さな入力例は Git で管理する。`make test` は各環境の準備後、実データ用の `.env` を設定する前でも実行できる。テスト用の DB と正本には `data/mock/store/`、中間ファイルには `data/mock/work/` を使う。 [テスト用の正本](01_data_and_backup.md#テスト用の正本)

MLflow への反映は、Experiment の実行 ID を MLflow run に対応付ける同期処理が担う。表示用 DB と成果物のコピーは `data/work/` に置く。バックアップを復元した直後も、正本から同期して `make mlflow-ui` で実験結果を比較できるようにする。 [MLflow Tracking への記録](../20_basic_design/02_history_and_artifacts.md#mlflow-trackingへの記録)

### 音声コーパスの準備

`make venv` を実行し、利用するコーパスの展開済みディレクトリを `.env` に記入する。空欄のコーパスはスキップする。相対パスはプロジェクトルートを基準に解決する。たとえば次の2件を設定できる。[設定のひな形](../../.env.example)

```sh
CORPUS_LIBRISPEECH="/datasets/LibriSpeech"
CORPUS_VOICEBANK_DEMAND="/datasets/VoiceBank+DEMAND"
```

| コマンド | 対象 |
| --- | --- |
| `make setup-data` | パスを設定した全コーパス。`MNIST_ROOT` を指定すると MNIST の IDX 索引も作る。 |
| `make setup-data CORPUS=librispeech` | LibriSpeech のみ。[コーパスIDと環境変数の対応](../../src/corpora/catalog.py) |
| `make setup-data JOBS=2` | 音声コーパスを最大2件並列で構築する。既定は4件。 |

音声ファイルは `data/corpora/<corpus_id>` から参照する。`data/work/corpus_index.duckdb` の `corpora.<corpus_id>` には、各コーパス内の相対パスと配布元に由来する属性が入る。LibriSpeech は発話単位、VCTK はマイク別の録音単位、VoiceBank+DEMAND は clean・noisy ペア単位で1行になる。たとえば次の SQL で LibriSpeech の公式サブセットごとの発話数を確認できる。

```sql
SELECT official_subset, count(*) AS utterances
FROM corpora.librispeech
GROUP BY official_subset
ORDER BY official_subset;
```

各コーパスを一時 DB に読み込み、完成したテーブルと検証済みの実体へのリンクを反映する。リンク先を変更するときも同じコマンドを使う。読み取りに失敗したコーパスは元のリンクとテーブルを保持し、成功したコーパスはその場で利用できる。エラーに表示されたコーパスは `CORPUS=<corpus_id>` で個別に再実行できる。

## scripts/ と tools/

- `scripts/` には、このプロジェクトで繰り返す作業の手順を置く。たとえば `make setup-data` から呼ぶスクリプトが、設定されたコーパスを選んで各セットアップ処理を呼ぶ。Makefile の記述は作業の呼び出しに絞る。
- `tools/` には、入力を指定して単独で使う道具を置く。たとえば、二つの公式分割表から追加・除外されたサンプルを調べる道具が該当する。

両方のディレクトリのコードを Git で管理する。
