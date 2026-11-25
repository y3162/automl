# Experimentの公開インターフェース（SS）

初版の実行操作はPython APIで定義する。CLIは引数を読み取り、同じ操作を呼び出し、結果をJSONで返す。実行の受付、状態の管理、履歴の記録はExperimentが担う。[実行フロー](../20_basic_design/01_execution_flow.md)

## 操作

Pythonでは `Experiment(root, *, source=None, read_only=False)` を使う。パスには文字列または `Path` を指定できる。`root` は正本を保存するデータルート、`source` は実装の読み込みと YAML 内のパス解決の基準となるプロジェクトルートで、省略時はカレントディレクトリとする。CLI はカレントディレクトリをプロジェクトルートとし、共通オプション `--root ROOT` を省略すると `data/store/` を使う。 [config の詳細設計](02_configuration.md)

XOR の通常実行例も標準の `data/store/` を使う。自動テストでは `root` または `--root` に `data/mock/store/` 配下のテスト専用ディレクトリを指定する。 [テスト用の正本](../90_operations_and_maintenance/01_data_and_backup.md#テスト用の正本)

| Python API | CLI | 成功時の戻り値 |
|---|---|---|
| `branch(origin_run=None, *, checkpoint=None, request_id=None)` | `branch [--from-run RUN] [--checkpoint REF] [--request-id ID]` | Pythonは枝ID、CLIは `branch_id` を含むJSON |
| `start(branch=None, config=None, *, initialize=None, evaluate=None, predictions=None, prediction_artifact=None, retain=None, fork=None, current_source=False, request_id=None)` | `start [CONFIG] (--branch BRANCH または --fork REF) [--initialize REF / --evaluate REF / --predictions PATH / --prediction-artifact ID] [--retain NAME]... [--current-source] [--request-id ID]` | Pythonは既存の枝なら実行ID、`fork` を指定した場合は要求IDの有無にかかわらず `branch_id` と `run_id` を持つ辞書。CLIは `run_id`、`branch_id`、`origin_run`、`origin_checkpoint`、`initial_checkpoint` を含むJSON |
| `resume(branch, *, request_id=None)` | `resume BRANCH [--request-id ID]` | 同じ枝に追加した実行のID。CLIは `run_id` を含むJSON |
| `stop(run)` | `stop RUN` | Pythonは `None`、CLIは `run_id` と `stop_requested: true` を含むJSON |
| `status(run, *, reconcile=False)` | `status RUN` | 指定実行の状態 |
| | `wait RUN [--timeout SECONDS]` | 終了または待ち時間の上限に達した時点の状態 |
| `delete_checkpoint(checkpoint_id)` | `delete-checkpoint CHECKPOINT_ID` | `checkpoint_id` と `result`。使用中は `in_use`、削除完了時と削除済みのIDへの再依頼は `deleted` |
| `record_decision(run, evaluation, decision, *, aggregate_id=None, followup_run=None, request_id=None)` | `record-decision RUN EVALUATION DECISION [--aggregate-id ID] [--followup-run RUN] [--request-id ID]` | Pythonは判断ID、CLIは `decision_id` を含むJSON |
| `history(run=None, *, summary=False)` | `history [--run RUN] [--summary]` | 枝、実行、イベント、成果物の記録 |
| `tracking_revisions()` | | 実行IDごとの同期用の変更識別子 |
| `tracking_snapshot(run)` | | 同一時点の詳細履歴（処理状態、実際に使った保存点IDを含む） |
| `verify(run=None)` | `verify [--run RUN]` | プロジェクト全体では検証件数、不整合の一覧 `issues`、削除未完了の保存点ID一覧 `pending_deletions`。実行を指定した場合は `ready` と `issues` |

実行に属する評価の検索、測定や集計の個別・分割取得には評価履歴の参照APIを使う。`list_uses` は、指定した測定を直接使った集計、または指定した集計を直接使った集計・判断を返す。[評価履歴の参照API](10_evaluation_history_api.md#判断と集計の逆引き)

枝ID・実行ID・成果物IDは文字列として扱う。`REF` にはチェックポイントの成果物ID、実行ID、または `RUN:latest` を指定できる。実行IDだけを指定した場合は `RUN:latest` として扱う。`latest` は対象の実行で保持している再開可能な保存点のうち学習順で最後のものを指す。成績に基づく選択は個別実装が行い、保存点IDを指定する。参照が解決された先の成果物IDを履歴に残す。分岐元の実行とチェックポイントを両方指定する場合は、チェックポイントがその実行の成果物であることを確認する。[チェックポイントと学習再開](../20_basic_design/05_checkpoints_and_resume.md)

`delete_checkpoint("C200")` の受付時に評価側がその保存点をコピーしている途中なら、通常の戻り値 `{"checkpoint_id": "C200", "result": "in_use"}` を返し、削除依頼はそこで終了する。コピーの完了後、管理側は同じIDで改めて削除を依頼できる。削除途中で停止した場合も同じIDで続行できる。[使用中の保存点と削除](04_checkpoint_management.md#使用中の保存点と削除)

削除が先に始まった保存点を指定して新たに評価を起動すると、受付前に `invalid_request` を返す。`details` には `checkpoint_id` と `checkpoint_state`（`deleting` または `deleted`）を含める。新しい実行IDは発行しない。指定したIDに該当する保存点の記録自体がなければ `not_found` とする。

`stop(run)` は受付済みの実行IDに対して起動前から中断要求を受け付ける。起動を取り消せたら `interrupted` を記録して枝を解放する。すでに起動していれば、進行中の学習・評価の安全な停止を確認する。逐次評価中、学習と評価が並行している間、学習成功後に評価だけが残っている間、評価のみの実行でも受け付ける。後日追加した評価は自身の実行IDで停止する。学習の途中再開に未対応のTrainerにも停止を要求できる。

`stop` の成功応答は中断要求の受付を示す。対象の処理すべてが安全に停止したことは `status` で確認する。学習が成功済みならその状態と成果物を保持し、要求で停止した評価と実行全体は `interrupted` とする。並行する一方がすでに失敗している場合は、他方の安全な停止を確認して実行全体を `failed` とする。各処理の `failed` と `interrupted` は、それぞれの終了理由とともに履歴から確認できる。中断した学習の履歴には再開元に使える保存点IDまたは再開できない理由を示す。CLIの `wait RUN [--timeout SECONDS]` は `status` を繰り返し呼び出し、終了時または待ち時間の上限に達した時点の状態を返す。

学習を含む `start(branch=...)` と `resume(branch)` の新しい依頼は、評価のみの実行を含め、枝が占有されていれば `branch_busy` を返す。枝の解放後は次の規則を使う。同じ要求ID・同じ引数での再送には、受付済みの結果を返す。[同じ枝での再開](../20_basic_design/05_checkpoints_and_resume.md#同じ枝での再開)

| 枝の学習履歴 | 学習の起動方法 |
|---|---|
| 一度も学習を開始していないことが確認できた | 同じ枝で新たに `start` できる。最初の実行が `launch_failed`、起動前の取消、モデル構築失敗で終わった場合を含む。元の実行と終了理由は履歴に残る |
| 学習を開始した実行がある | 続きは `resume` で依頼する。最後に学習開始を記録した実行を選び、今回の受付時点で有効な `latest` の保存点IDと再開元の実行IDを記録する。後続の評価のみの実行や学習開始前に失敗した再開試行は読み飛ばす。再開元が条件を満たさなければ理由を返す |
| 学習開始後に最初からやり直す | 同じ枝への新規学習の `start` は受け付けない。新しい枝で `start` する |

## 要求の再送

個々の評価開始に指定する `request_id` は、その評価が属する実行IDの中で照合する。学習中のTrainerからの再送も、同じ依頼なら初回の評価IDを返す。[評価開始の再送](09_evaluator.md#評価開始の再送)

枝作成・起動・再開では、同じプロジェクト内で受付済みの `request_id` を照合する。操作と指定内容が同じなら、枝の占有や再開可否を改めて判定せず、初回に受け付けた枝・実行のIDを返す。たとえば `resume` の応答を失い、再開した実行の終了後に再送しても、初回の再開で作成した実行IDを返す。初回に `latest` などの参照を解決した場合も、そのときの受付結果を使う。

同じIDで操作・対象・設定の指定などを変えた場合は `invalid_request` とし、要求IDと食い違った内容を示す。意図して同じ条件の実験をもう一度依頼する場合は、新しい要求IDを使う。要求IDを省略した場合は従来どおり新規要求として扱う。[受付と枝の占有](03_execution_state_and_processes.md#受付と枝の占有)

受付後の再送は初回の操作と引数で照合する。たとえば `--predictions predictions.json` の受付後に同じパスが上書きされても、同じ要求ID・同じ引数には初回の実行IDを返し、その評価には受付時に確保した内容を使う。更新後の内容を評価するときは新しい要求として起動する。

## 実行計画と起動時の指定

configの `plan` に実行する処理を順に指定する。`[train]`、`[evaluate]`、`[train, evaluate]` を受け付ける。学習中に起動する評価は、個別実装に応じて逐次または並行で実行できる。同じ実行の計画に含めた並行評価が進行中であれば、学習の成功後も実行全体は `running` となる。後日評価は評価のみの新しい実行として受け付ける。その実行から評価対象を参照でき、プロジェクト内の成果物を使った場合は生成元の実行もたどれる。configには各処理で使う実装と設定を記載する。分岐や過去の状態への参照は起動時に指定する。

| 実行計画 | 起動時に指定する対象 |
|---|---|
| 学習のみ、または学習の後に評価 | 必要に応じて `initialize` で学習の初期状態を指定する。部分初期化では `retain` に引き継ぐ名前を指定する。学習後の評価に使う保存点はタスク側が選び、ExperimentにIDを渡す。候補が0件なら、実行中のタスクが終了前に、実験の意図に応じた見送りか失敗かと理由を報告する |
| モデルまたは予測結果を使う評価のみの実行 | チェックポイント参照（`evaluate`）、プロジェクト内の予測成果物ID（`prediction_artifact`）、外部の予測ファイル（`predictions`）のいずれか一つを指定する |
| 保存済み測定だけを再集計する評価のみの実行 | 評価設定で再集計の手順を選ぶ。モデル状態や予測ファイルの指定は省略できる |

保存済みの測定結果の再集計には、入力の評価と採用した結果を使う。訂正・再集計を参照用に保存する段階で、それを行った実行に新しい評価IDを付ける。後日行う場合は新しい評価のみの実行に属する。保存済みの測定だけを使う再集計は、モデルをロードせずに行える。[評価の続行・訂正・再集計](09_evaluator.md#評価の続行訂正再集計)

`--fork REF` は一操作でその保存点から枝を作り、保存点のモデル状態を初期値として新規学習を始める。Trainerの学習状態は新しい枝で初期化する。`--branch` と `--initialize` を使えば由来と初期化元を個別に選べる。部分引き継ぎで `retain` を指定した場合、Experimentは指定名を実行履歴に残し、Trainerを経由してModelの初期化設定に渡す。名前が指す範囲と引き継ぎの可否は個別のModelが判定する。[モデルの構築](07_model.md#モデルの構築)

`--evaluate REF` ではconfigを省略できる。保存点を直接指定すると、その生成元の実行からモデル、データ版、分割、前処理、Evaluatorのコード版と設定を引き継ぐ。評価のみの実行を分岐元にして同じ保存点を指定したときは、その評価で実際に使ったEvaluatorの版と評価条件を引き継ぐ。どちらの場合も、比較用の設定は、この条件を土台に親YAML、子YAMLの順で明示項目を上書きする。[設定の継承と上書き](02_configuration.md#継承と上書き)

XOR CSV の版を変えるときは新しい SHA-256 を指定する。新しいファイルから版を確定する場合は `dataset.settings` に `path` と `sha256: null` を指定する。前処理を外すには `automl.transform.Identity` を指定する。`evaluation_conditions` には元の run と変更した部品の設定を残す。明示的に変更したモデルは受付前に保存済みの重みを復元して互換性を検査する。

保存後に追加・編集した部品を試す場合は `--evaluate REF --current-source` で現在のプロジェクトソースを指定する。変更前後のソースの識別子を記録し、指定した重みを各 Model が復元できることを受付前に確認する。

実行を受け付ける前に、Experimentがconfigと起動時の指定の組み合わせを確認する。チェックポイントなど、起動時に選んだ対象は具体的な参照に解決して記録する。`--prediction-artifact ID` では評価履歴から指定した予測成果物IDとその生成元の実行IDをたどれる。`--predictions PATH` では受付時のファイル内容を成果物として保持し、外部入力と記録する。[設定と個別実装](../20_basic_design/03_configuration_and_implementations.md)

## 状態と履歴の戻り値

`record_decision` は、`run` 内の評価 `evaluation` の結果を根拠とした判断内容を一件記録する。特定の集計値を使った場合は、その評価内の `aggregate_id` を任意で指定できる。同じ評価を指定して別の判断も記録できる。指標の計算前に失敗した評価IDも根拠にできる。`request_id` と内容が同じ再送には元の判断IDを返し、同じIDの内容が変われば `invalid_request` を返す。対応する後続の実行がある場合は、その実行IDを `followup_run` に指定できる。指定した実行・評価・集計が見つからない場合は `not_found` を返す。`history(run)` の `events` から、判断ID、内容、評価への参照、指定された集計IDや後続の実行IDを確認できる。

`status` は実行IDの `id`、枝IDの `branch`、実行全体の `status`、処理ごとの `stages` を返す。`status` は `accepted`、`running`、`succeeded`、`failed`、`interrupted` のいずれかとする。`stages` にはconfigで実行を指定した `train` や `evaluate` の状態が入る。開始待ちは `pending` とする。逐次計画で学習が失敗または中断し、後続評価を開始しないことが確定した場合は、評価の状態 `skipped` と、その理由「前段の学習が失敗したため」または「前段の学習が中断されたため」を返す。並行実行中の一方が失敗し、他方が続行中の場合、該当する処理の状態はそれぞれ `failed` と `running`、実行全体は `running` とする。学習中にTrainerが呼んだ個々の評価は、学習実行から参照できる評価として履歴に記録する。

学習成功後に評価候補が0件となった場合、`status` の評価段と `history` の処理記録にタスクが報告した理由を含める。見送りなら評価は `skipped`、全体は `succeeded` となる。必要な評価を実施できなかった場合は評価と全体が `failed` となる。

状態を確認できない間は、最後に確認した状態を返し、`state_confirmed` を `false` とする。確認できる場合は `true` とする。失敗時の `error` には、エラーコードと原因を含める。[実行状態とプロセス管理](03_execution_state_and_processes.md)

読み取り専用の Experiment では、`status` はその時点の観測を返す。書き込み可能な Experiment で `reconcile=True` を指定すると、受付・実行中の実行について、実行ホストの起動・終了情報と記録済みの処理結果を照合し、状態遷移表に従って確認できた遷移をDBに記録する。起動や終了の有無を確認できない間は枝の占有を維持する。CLIの `status` と `wait` はこの動作を使う。実行中の `execution_host` は状態を確認するためのホスト名を示す。

`history` は `branches`、`runs`、`events`、`artifacts` を返す。実行IDを指定した場合は、その実行、所属する枝、関連するイベントと成果物を返す。`artifacts` の各保存点には `checkpoint_state` があり、`retained`、`deleting`、`deleted` のいずれかを返す。コピー中に削除依頼へ `in_use` を返した保存点は `retained` のままである。削除後も同じIDから過去の進捗、評価結果、使用履歴をたどれる。分岐元や別の実行の成果物は、その参照を通じてたどれる。指標、評価結果、生成されたログはイベントに含まれる。[履歴と成果物](../20_basic_design/02_history_and_artifacts.md)

保存に失敗した場合は、`history(run)` の `events` から保存対象の学習時点と失敗理由を確認できる。学習成功後に判明した任意の保存の失敗も、その学習実行から参照できる。[Trainerの詳細設計](08_trainer.md#停止条件と中断要求)

公開APIから再開経路の各区間と全実行の記録を取得できる。進行中の評価を複数回に分けて読む場合も、一連の取得では読み出し開始時点の確定範囲を保つ。続行・訂正・保存した再集計は新しい評価IDから入力の評価と採用した結果をたどれる。[学習履歴の経路](../20_basic_design/02_history_and_artifacts.md#再開分岐後の学習履歴)・[評価中の記録](09_evaluator.md#評価中の記録)

`history --summary` は run ごとに分岐元・初期化元・ステージ状態と各評価部品の直近の評価を返す。評価ID、保存点ID、進捗、分割、集計値、およびタスクが報告した選択基準を確認できる。サンプル ID と例外の詳細は `history --run RUN` の対応するイベントから取得する。

### 再開前後の履歴の取得例

XORの学習実行 `R1` を更新120で中断し、同じ枝の `R2` で学習を再開して更新200に達したとする。次の順に履歴を読むと、`R2` がどの時点のモデルを引き継いだかが分かる。

| 呼び出し | 読み取る記録 | この例で得られる内容 |
|---|---|---|
| `history(R2)` | `runs` にある `R2` の再開元 | 元の学習実行IDは `R1`、再開に使った保存点IDは `C100` |
| `history(R1)` | `artifacts` にある `C100` の保存点記録 | 保存時の進捗は更新100。対応するデータ使用範囲は `R1` の更新100までにモデルに反映された分 |

最終モデルに反映されたのは、`R1` の更新1〜100と `R2` の更新101〜200の計200更新である。`R1` の更新101〜120の記録も元の実行に残るため、実際に計算した更新は `R1` の120回と `R2` の100回を合わせて220回と分かる。

同じ例で、最初の再開試行 `R2` がモデル復元中に失敗し、学習を開始できなかった場合を考える。同じ要求IDでの再送には失敗した `R2` のIDを返す。枝が空いた後、新しい要求IDで `resume` すると、`R1` の現在の `latest` を改めて選ぶ。`C100` が残っていれば新しい実行 `R3` は更新101から始まり、`history(R3)` の再開元は `R1` と `C100` になる。`R2` の失敗理由は `history(R2)` に残る。`C100` が削除されていれば `R1` の残る有効な保存点を学習順で選び、候補がなければ再開できない理由を返す。

## 成果物の整合性検証

`verify()` は保存済みの記録と成果物を読み取り、プロジェクト全体で各実行の由来と保持する成果物の内容を検証する。検証件数に加え、見つかった不整合をすべて `issues` に返す。各項目には対象を特定する実行IDや成果物IDと理由を含める。たとえば、ある実行の保存点の欠落と別の実行の予測ファイルの破損は同じ応答で確認できる。

既存の保存点を使った場合は、実行履歴に記録した成果物IDから元の保存点の記録と生成元の実行をたどれることを確認する。以下は、モデル・学習の開始状態または入力予測の実体を保存する実行について、保存完了と学習・評価の開始の記録を照合する規則である。保存済み測定だけを使った再集計の入力は、集計 ID から元の評価と採用した結果をたどる。 [保存点と実行開始状態](04_checkpoint_management.md)・[集計値と入力](10_evaluation_history_api.md#集計値と入力をたどる)

| 実行履歴 | 開始状態の判定 |
|---|---|
| 保存完了前に失敗し、学習・評価も始まっていない | コピーがなくても履歴は整合する。実行の状態は `failed` |
| 起動前の停止要求で起動を取り消し、開始状態の保存も始まっていない | コピーがなくても履歴は整合する。実行の状態は `interrupted` |
| 保存完了を記録した | 利用側のコピーがすべて揃い、利用側で保存した時点の内容と一致することを確認する。その後に実行が失敗した場合も同じ判定を行う |
| 学習・評価の開始記録があり、それに先立つ保存完了記録がない | 記録順序の不整合として検証失敗とする |

たとえば親の保存点 A から子を開始し、親の A が削除済みであっても、由来の記録と子のコピーが揃い、内容が一致していれば、この検証は成功する。由来をたどれない場合や、子のコピーに欠落・内容の不一致がある場合は失敗する。親に A のファイルが残っていても子のコピーの欠落は検出する。

`verify()` は移行先でも保存点の `checkpoint_state` を照合する。`deleting` の保存点IDは `pending_deletions` に含める。削除途中のファイルだけが残っていても、ほかに不整合がなければ `issues` は空で、CLIの終了コードは `0` となる。`deleted` の記録と実体の削除が一致すれば正常とする。`retained` の保存点や利用側の独立した開始状態に欠落・破損があれば `issues` に含める。CLIは全体の照合結果を返し、`issues` が1件以上あれば終了コードを `1` とする。

`verify(run)`（CLIでは `verify --run RUN`）は保存されたコードの再構築、依存環境、Datasetが公開した入力と使用したデータ版、開始状態と入力保存点を照合する。依存環境の照合項目と一致基準はプロジェクト・タスクが定める。基準を満たさない場合は `ready: false` とし、`issues` に項目名、実行時の値、移行先の値を含める。必要な記録・成果物の不足や破損、旧形式で版の記録がない場合も対象別の理由を返す。CLIは復元可能なら `0`、`ready: false` なら `1` を返す。

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
