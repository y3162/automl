# Experimentの公開インターフェース（SS）

初版の実行操作はPython APIで定義する。CLIは引数を読み取り、同じ操作を呼び出し、結果をJSONで返す。実行の受付、状態の管理、履歴の記録はExperimentが担う。[実行フロー](../20_basic_design/01_execution_flow.md)

## 操作

Pythonでは `Experiment(root, *, source=None, read_only=False)` を使う。パスには文字列または `Path` を指定できる。`root` はプロジェクトデータのルート、`source` は実装の読み込みとYAML内のパス解決の基準となるプロジェクトルートで、省略時はカレントディレクトリとする。CLIはカレントディレクトリをプロジェクトルートとし、共通オプション `--root ROOT` でデータルートを指定する。データルートの省略時は `data/store` とする。XOR の例も既定のストアを使う。[configの詳細設計](02_configuration.md)

| Python API | CLI | 成功時の戻り値 |
|---|---|---|
| `branch(origin_run=None, *, checkpoint=None, request_id=None)` | `branch [--from-run RUN] [--checkpoint REF] [--request-id ID]` | Pythonは枝ID、CLIは `branch_id` を含むJSON |
| `start(branch=None, config=None, *, initialize=None, evaluate=None, predictions=None, retain=None, fork=None, current_source=False, request_id=None)` | `start [CONFIG] (--branch BRANCH または --fork REF) [--initialize REF / --evaluate REF / --predictions PATH] [--retain NAME]... [--current-source] [--request-id ID]` | Pythonは通常は実行ID、`fork` と要求IDを併用した場合は `branch_id` と `run_id` を持つ辞書。CLIは `run_id`、`branch_id`、`origin_run`、`origin_checkpoint`、`initial_checkpoint` を含むJSON |
| `resume(branch, *, request_id=None)` | `resume BRANCH [--request-id ID]` | 同じ枝に追加した実行のID。CLIは `run_id` を含むJSON |
| `stop(run)` | `stop RUN` | Pythonは `None`、CLIは `run_id` と `stop_requested: true` を含むJSON |
| `status(run, *, reconcile=False)` | `status RUN` | 指定実行の状態 |
| `record_decision(run, evaluation, decision, *, aggregate_id=None, followup_run=None, request_id=None)` | `record-decision RUN EVALUATION DECISION [--aggregate-id ID] [--followup-run RUN] [--request-id ID]` | Pythonは判断ID、CLIは `decision_id` を含むJSON |
| `history(run=None, *, summary=False)` | `history [--run RUN] [--summary]` | 枝、実行、イベント、成果物の記録 |
| `tracking_revisions()` | | 実行IDごとの同期用の変更識別子 |
| `tracking_snapshot(run)` | | 同一時点の詳細履歴（処理状態、実際に使った保存点IDを含む） |
| `verify(run=None)` | `verify [--run RUN]` | 全体は履歴照合の件数。実行を指定すると `ready` と不足・不一致を示す `issues` |

実行に属する評価の検索、測定や集計の個別・分割取得、判断根拠の逆引きには評価履歴の参照APIを使う。[評価履歴の参照API](10_evaluation_history_api.md)

枝ID・実行ID・成果物IDは文字列として扱う。`REF` にはチェックポイントの成果物ID、実行ID、または `RUN:latest` を指定できる。実行IDだけを指定した場合は `RUN:latest` として扱う。`latest` は対象の実行で保持している再開可能な保存点のうち学習順で最後のものを指す。成績に基づく選択は個別実装が行い、保存点IDを指定する。参照が解決された先の成果物IDを履歴に残す。分岐元の実行とチェックポイントを両方指定する場合は、チェックポイントがその実行の成果物であることを確認する。[チェックポイントと学習再開](../20_basic_design/05_checkpoints_and_resume.md)

保存点の削除は公開APIから保存点ID単位で依頼する。使用中ならその旨を返して依頼を終え、削除途中なら同じIDへの依頼で続行する。完了後はそのIDが削除済みと分かる。重みを必要とする操作に削除済みのIDを明示した場合は、指定IDと削除済みである理由を返す。[使用中の保存点と削除](04_checkpoint_management.md#使用中の保存点と削除)

`stop(run)` は指定した実行IDで進行中の学習と評価に中断を要求する。逐次評価中、学習と評価が並行している間、学習成功後に評価だけが残っている間、評価のみの実行でも受け付ける。後日追加した評価は自身の実行IDで停止する。学習の途中再開に未対応のTrainerにも停止を要求できる。

`stop` の成功応答は中断要求の受付を示す。対象の処理すべてが安全に停止したことは `status` で確認する。学習が成功済みならその状態と成果物を保持し、要求で停止した評価と実行全体は `interrupted` とする。並行する一方がすでに失敗している場合は、他方の安全な停止を確認して実行全体を `failed` とする。各処理の `failed` と `interrupted` は、それぞれの終了理由とともに履歴から確認できる。中断した学習の履歴には再開元に使える保存点IDまたは再開できない理由を示す。CLIの `wait RUN [--timeout SECONDS]` は `status` を繰り返し呼び出し、終了時または待ち時間の上限に達した時点の状態を返す。

## 要求の再送

個々の評価開始に指定する `request_id` は、その評価が属する実行IDの中で照合する。学習中のTrainerからの再送も、同じ依頼なら初回の評価IDを返す。[評価開始の再送](09_evaluator.md#評価開始の再送)

枝作成・起動・再開では、同じプロジェクト内で受付済みの `request_id` を照合する。操作と指定内容が同じなら、枝の占有や再開可否を改めて判定せず、初回に受け付けた枝・実行のIDを返す。たとえば `resume` の応答を失い、再開した実行の終了後に再送しても、初回の再開で作成した実行IDを返す。初回に `latest` などの参照を解決した場合も、そのときの受付結果を使う。

同じIDで操作・対象・設定の指定などを変えた場合は `invalid_request` とし、要求IDと食い違った内容を示す。意図して同じ条件の実験をもう一度依頼する場合は、新しい要求IDを使う。要求IDを省略した場合は従来どおり新規要求として扱う。[受付と枝の占有](03_execution_state_and_processes.md#受付と枝の占有)

## 実行計画と起動時の指定

configの `plan` に実行する処理を順に指定する。`[train]`、`[evaluate]`、`[train, evaluate]` を受け付ける。学習中に起動する評価は、個別実装に応じて逐次または並行で実行できる。同じ実行の計画に含めた並行評価が進行中であれば、学習の成功後も実行全体は `running` となる。後日評価は新しい評価のみの実行として受け付け、元の学習実行と評価対象を参照する。configには各処理で使う実装と設定を記載する。分岐や過去の状態への参照は起動時に指定する。

| 実行計画 | 起動時に指定する対象 |
|---|---|
| 学習のみ、または学習の後に評価 | 必要に応じて `initialize` で学習の初期状態を指定する。部分初期化では `retain` に引き継ぐ名前を指定する。学習後の評価に使う保存点はタスク側が選び、ExperimentにIDを渡す。候補が0件なら、意図に基づく見送りか失敗かを理由とともに報告する |
| モデルまたは予測結果を使う評価のみの実行 | `evaluate` に保存済みチェックポイントの参照、または `predictions` に予測結果のファイルを指定する。指定はどちらか一つとする |

保存済みの測定結果の再集計には、入力の評価と採用した結果を使う。参照用に保存する段階で新しい評価IDを付け、モデルのロードや再推論は不要とする。[評価の続行・訂正・再集計](09_evaluator.md#評価の続行訂正再集計)

`--fork REF` は一操作でその保存点から枝を作り、その重みで新規学習を始める。`--branch` と `--initialize` を使えば由来と初期化元を個別に選べる。

`--evaluate REF` では config を省略できる。保存元からモデル、データ版、分割、前処理、評価実装と設定を引き継ぐ。評価だけ行った親 run を分岐元に指定し、同じ保存点を渡すと、その親の評価条件を引き継ぐ。比較時は YAML に変更項目を書く。XOR CSV の版を変えるときは新しい SHA-256 を指定する。新しいファイルから版を確定する場合は `dataset.settings` に `path` と `sha256: null` を指定する。前処理を外すには `automl.transform.Identity` を指定する。`evaluation_conditions` には元の run と変更した部品の設定を残す。明示的に変更したモデルは受付前に保存済みの重みを復元して互換性を検査する。

保存後に追加・編集した部品を試す場合は `--evaluate REF --current-source` で現在のプロジェクトソースを指定する。変更前後のソースの識別子を記録し、指定した重みを各 Model が復元できることを受付前に確認する。

実行を受け付ける前に、Experimentがconfigと起動時の指定の組み合わせを確認する。チェックポイントなど、起動時に選んだ対象は具体的な参照に解決して記録する。予測結果をファイルで渡す場合は、その実体を成果物として保持する。[設定と個別実装](../20_basic_design/03_configuration_and_implementations.md)

## 状態と履歴の戻り値

`record_decision` は、`run` 内の評価 `evaluation` の結果を根拠とした判断内容を一件記録する。特定の集計値を使った場合は、その評価内の `aggregate_id` を任意で指定できる。同じ評価を指定して別の判断も記録できる。指標の計算前に失敗した評価IDも根拠にできる。`request_id` と内容が同じ再送には元の判断IDを返し、同じIDの内容が変われば `invalid_request` を返す。対応する後続の実行がある場合は、その実行IDを `followup_run` に指定できる。指定した実行・評価・集計が見つからない場合は `not_found` を返す。`history(run)` の `events` から、判断ID、内容、評価への参照、指定された集計IDや後続の実行IDを確認できる。

`status` は実行IDの `id`、枝IDの `branch`、実行全体の `status`、処理ごとの `stages` を返す。`status` は `accepted`、`running`、`succeeded`、`failed`、`interrupted` のいずれかとする。`stages` にはconfigで実行を指定した `train` や `evaluate` の状態が入る。開始待ちは `pending` とする。逐次計画で学習が失敗または中断し、後続評価を開始しないことが確定した場合は、評価の状態 `skipped` と、その理由「前段の学習が失敗したため」または「前段の学習が中断されたため」を返す。並行実行中の一方が失敗し、他方が続行中の場合、該当する処理の状態はそれぞれ `failed` と `running`、実行全体は `running` とする。学習中にTrainerが呼んだ個々の評価は、学習実行から参照できる評価として履歴に記録する。

状態を確認できない間は、最後に確認した状態を返し、`state_confirmed` を `false` とする。確認できる場合は `true` とする。失敗時の `error` には、エラーコードと原因を含める。[実行状態とプロセス管理](03_execution_state_and_processes.md)

読み取り専用のExperimentでは、`status` はその時点の観測を返す。`reconcile=True` を指定すると、異常終了を実行ホストで再確認し、そのrunがまだ受付・実行中なら既存DBに失敗を記録する。CLIの `status` と `wait` はこの動作を使う。実行中の `execution_host` は状態を確認するためのホスト名を示す。

`history` は `branches`、`runs`、`events`、`artifacts` を返す。実行IDを指定した場合は、その実行、所属する枝、関連するイベントと成果物を返す。分岐元や別の実行の成果物は、その参照を通じてたどれる。指標、評価結果、生成されたログはイベントに含まれる。[履歴と成果物](../20_basic_design/02_history_and_artifacts.md)

保存に失敗した場合は、`history(run)` の `events` から保存対象の学習時点と失敗理由を確認できる。学習成功後に判明した任意の保存の失敗も、その学習実行から参照できる。[Trainerの詳細設計](08_trainer.md#停止条件と中断要求)

公開APIから再開経路の各区間と全実行の記録を取得できる。進行中の評価を複数回に分けて読む場合も、一連の取得では読み出し開始時点の確定範囲を保つ。続行・訂正・保存した再集計は新しい評価IDから入力の評価と採用した結果をたどれる。[学習履歴の経路](../20_basic_design/02_history_and_artifacts.md#再開分岐後の学習履歴)・[評価中の記録](09_evaluator.md#評価中の記録)

`history --summary` は run ごとに分岐元・初期化元・ステージ状態と各評価部品の直近の評価を返す。評価ID、保存点ID、進捗、分割、集計値、およびタスクが報告した選択基準を確認できる。サンプル ID と例外の詳細は `history --run RUN` の対応するイベントから取得する。

## 成果物の整合性検証

`verify()` は保存済みの記録と成果物を読み取り、各実行の由来と、保持する成果物の内容を検証する。学習や評価を再実行する操作は含まない。

既存の保存点を使った場合は、実行履歴に記録した成果物IDから元の保存点の記録と生成元の実行をたどれることを確認する。開始状態の検証には、保存完了と学習・評価の開始の記録を使う。予測結果を入力にした評価では、入力に使った予測結果を同じ規則で照合する。[保存点と実行開始状態](04_checkpoint_management.md)

| 実行履歴 | 開始状態の判定 |
|---|---|
| 保存完了前に失敗し、学習・評価も始まっていない | コピーがなくても履歴は整合する。実行の状態は `failed` |
| 保存完了を記録した | 利用側のコピーがすべて揃い、利用側で保存した時点の内容と一致することを確認する。その後に実行が失敗した場合も同じ判定を行う |
| 学習・評価の開始記録があり、それに先立つ保存完了記録がない | 記録順序の不整合として検証失敗とする |

たとえば親の保存点 A から子を開始し、親の A が削除済みであっても、由来の記録と子のコピーが揃い、内容が一致していれば、この検証は成功する。由来をたどれない場合や、子のコピーに欠落・内容の不一致がある場合は失敗する。親に A のファイルが残っていても子のコピーの欠落は検出する。

`verify` は移行先でも保存点ごとの削除記録を照合する。「削除済み」は意図した削除、「削除中」は操作未完了と報告し、保持対象の欠落は検証エラーとする。利用側の開始状態は独立した保持対象として照合する。

`verify --run RUN` は保存されたコードの再構築、環境の一致、Dataset が公開した入力と使用したデータ版、開始状態と入力保存点を照合する。不足・破損・旧形式で版の記録がない場合は `ready: false` と対象別の理由を返し、CLI は終了コード 1 を返す。

## エラー

Pythonで操作を拒否するときは `ExperimentError` を送出する。CLIでは同じエラー情報を標準エラーにJSONで出力し、終了コードを `1` とする。エラーには機械判定用の `code` と `details`、人向けの `message` を含める。

| コード | 用途 | 主な補足情報 |
|---|---|---|
| `branch_busy` | 対象の枝ですでに実行を受け付けている | `active_run_id` |
| `invalid_config` | configの解決・検証や指定された実装の読み込みに失敗した | 特定できる場合は対象の設定項目や実装 |
| `not_found` | 指定された枝、実行、成果物を特定できない | 対象の種類と参照 |
| `invalid_request` | 実行計画と起動時の指定が適合しない、または現在の状態では操作できない | 対象の操作や実行の状態 |

CLI の必須引数の欠落も標準エラーの JSON と終了コード 1 で返す。受付前に拒否された要求には新しい実行IDを割り当てない。受付後の失敗は、発行済みの実行IDの状態と `error` に記録する。起動の失敗には `launch_failed`、起動後の処理の失敗には `execution_failed` を使い、失敗した処理が分かる場合は `details` に含める。

CLIで状態を取得できた場合は取得操作の終了コードを `0` とし、実行そのものの成否は返された状態で示す。`wait` は実行が失敗した場合に `1`、待ち時間の上限に達した場合に `2`、成功または中断を確認した場合に `0` を返す。いずれの場合も取得した状態を標準出力へ返す。
