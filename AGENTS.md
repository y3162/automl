# エージェント向け作業指示

Cursor、Codex、Claude Code に共通する指示はこのファイルで管理する。複数の作業で繰り返す手順は `.agents/skills/<名前>/SKILL.md` に置き、依頼に合うスキルを使う。[共通スキルの追加方法](.agents/skills/README.md)

ツール固有の指示や設定は必要になったときに追加する。Claude Code 固有の指示は `CLAUDE.md` の `@AGENTS.md` の下に追記する。Cursor と Codex の追加先は各ディレクトリに示す。[Cursor](.cursor/README.md)・[Codex](.codex/README.md)

<!-- CODEGRAPH_START -->
## CodeGraph

In repositories indexed by CodeGraph (a `.codegraph/` directory exists at the repo root), reach for it BEFORE grep/find or reading files when you need to understand or locate code:

- **MCP tool** (when available): `codegraph_explore` answers most code questions in one call — the relevant symbols' verbatim source plus the call paths between them, including dynamic-dispatch hops grep can't follow. Name a file or symbol in the query to read its current line-numbered source. If it's listed but deferred, load it by name via tool search.
- **Shell** (always works): `codegraph explore "<symbol names or question>"` prints the same output.

If there is no `.codegraph/` directory, skip CodeGraph entirely — indexing is the user's decision.
<!-- CODEGRAPH_END -->

## 実装と設定

- 共通の実行基盤と公開 API は `src/automl/`、手法固有の実装は `src/methods/`、コーパスの読み取りと索引作成は `src/corpora/`、小さな実行例は `src/examples/` に置く。
- 学習率や更新回数を変えるときは、まず `configs/` の対象 YAML で該当する部品の `settings` を調べる。新旧の実装を選んで比較する場合は、既存の設定から従来の版を指定できる形で新しい版を追加する。[手法の配置と版管理](docs/90_operations_and_maintenance/03_source_layout_and_versions.md)

## 実験とテスト

- 実験の起動・再開・評価と履歴の参照には `automl` CLI または `automl.experiment.Experiment` を使う。結果を報告するときは実行 ID を添え、比較では評価に使ったデータ分割、保存点、指標を確認する。[Experiment の公開 API](docs/30_detailed_design/01_experiment_api.md)
- 履歴と保持する成果物の正本は `data/store/`、再生成できる索引や中間ファイルは `data/work/` に置く。外部コーパスのパスは計算機ごとの `.env` に設定する。[データの配置とバックアップ](docs/90_operations_and_maintenance/01_data_and_backup.md)
- 変更した領域は対応する `tests/` 配下のテストで確認する。テスト用の履歴や成果物には `data/mock/store/`、中間ファイルには `data/mock/work/` を使う。初回はプロジェクトルートで `make venv` を実行し、全体の確認には `make test` を使う。[テスト設計](docs/50_unit_testing/01_unit_test_design.md)
