# 単体テスト設計

実験を再現するには、使ったデータ、学習を再開する状態、評価に採用した測定結果を正しく特定できる必要がある。そこで、正常時の戻り値に加え、途中で失敗した時点で何が確定しているか、同じ依頼を再送すると何が返るかを確認する。[プロジェクト共通要件](../10_requirements/00_project_requirements.md)

以下の「既存」は該当する振る舞いを確認できるテスト、「追加」は直接検証するケースとその配置案を示す。要件 ID は、そのケースが担当する受け入れ基準の観点を示す。

## テストの単位とデータ

- 一つのテストで判定するのは、一つの部品の公開操作、または一つの状態判定とする。Trainer と Evaluator には、小さな入力と、使用した ID・報告順・保存状態を記録できるテスト用コンテキストを渡す。
- 設定ファイルには `tmp_path`、使用履歴の SQL にはテストごとに開いたインメモリ SQLite を使う。トランザクションや成果物の実体を検証するときは、`data/mock/store/` 配下にケース専用の正本を作る。一時プロジェクトを使うケースも、その中の `data/mock/store/` と `data/mock/work/` で完結する。[テスト用の正本](../90_operations_and_maintenance/01_data_and_backup.md#テスト用の正本)
- XOR はインラインの4点または Git 管理の CSV を使う。MNIST の計算には固定配列、読み出しにはテスト中に生成した小さな IDX を使う。ID・順序・件数・保存済みバイト列は完全一致で照合し、数値計算は手法ごとの基準で照合する。[XOR の入力](../../configs/examples/xor/samples.csv)・[MNIST のフィクスチャ](../../tests/examples/mnist/conftest.py)
- 時間に依存する競合は `Barrier` や `Event` で操作順を作る。書き込み失敗は対象の書き込みだけを一度失敗させ、同じ要求を再送する。実 worker の起動・監視、別クローンからの再実行、CLI からの学習・評価の一巡は結合・システムテストで確認する。[プロセス監視](../../tests/automl/experiment/test_supervision.py)・[再実行](../../tests/automl/experiment/test_replay.py)・[XOR の一連の実験](../../tests/examples/xor/test_experiment.py)

## Experiment の設定・受付・状態判定

設定の解決は YAML と小さな部品クラスだけで試せる。枝の占有には専用の Store を使い、終了判定には worker の終了コードと、学習・評価それぞれの確定済みの状態を与える。[設定の詳細設計](../30_detailed_design/02_configuration.md)・[実行状態の遷移](../30_detailed_design/03_execution_state_and_processes.md)・[Experiment の受け入れ基準](../10_requirements/experiment/05_acceptance_criteria.md)

| 対象 | 入力と操作 | 合格条件 | テスト |
| --- | --- | --- | --- |
| `load_config` の継承 | 保存済みの実装 B を親 YAML で A、子で再び B にする。別ケースで同じ実装の辞書とリストを上書きする。 | 戻した B の設定には子が明示した値だけが残る。同じ実装の辞書は未指定キーを保持し、リストと `plan` は子の値になる。 | 既存 [test_config.py](../../tests/automl/experiment/test_config.py) |
| YAML の重複キー | 同じマッピングの4行目に `width: 4`、5行目に `width: 9` を書く。 | `invalid_config` の `details` にファイル名、行番号 5、キー名 `width` が入る。 | 既存 [test_config.py](../../tests/automl/experiment/test_config.py) |
| `resolve_config` の受付条件 | ① `plan=[evaluate, train]` ② `plan=[train]` で `trainer` に `Identity` を指定する。 | 両方とも `invalid_config`。①は `field=plan`、②は `component=trainer` を返し、設定のどこを直せばよいか特定できる。 | 追加 [test_config.py](../../tests/automl/experiment/test_config.py) |
| 同じ枝の受付（EX-A-010・014） | 空の枝に同時に3件を要求し、うち2件の要求 ID と引数を同じにする。 | 作られる実行は1件。受理した要求の再送は同一 ID、競合した別要求は `branch_busy` と占有中の実行 ID を返す。起動前の失敗を確定すると枝が空く。 | 既存 [test_store.py](../../tests/automl/experiment/test_store.py) |
| 全体の終了判定（EX-A-019・020） | `engine._outcome` に `running` の実行と終了コード 0 を渡す。学習・評価の順に ① `failed/interrupted` ② `interrupted/failed` ③ `succeeded/interrupted` を与える。さらに両段成功・終了コード 1 を試す。 | ①②は全体が `failed`、③は全体が `interrupted`。いずれも両段の結果を保持する。異常終了では両段の成功が残り、全体は `failed` で理由は `execution_failed`。 | 追加 [test_supervision.py](../../tests/automl/experiment/test_supervision.py) |

## Dataset・Transform

データの読み出しは版 ID を、使用の記録は使用 ID と位置を起点に観測する。同じサンプルを二度使った場合と、一回の使用を二度報告した場合で件数が変わる。[Dataset の受け入れ基準](../10_requirements/dataset/05_acceptance_criteria.md)・[Transform の受け入れ基準](../10_requirements/transform/05_acceptance_criteria.md)

| 対象 | 入力と操作 | 合格条件 | テスト |
| --- | --- | --- | --- |
| 固定版の読み出し（DS-A-001・003） | CSV のサンプル `104` のラベルを 1 から 0 に変更する。変更前の版作成要求を再送し、新しい要求 ID でも版を作る。 | 再送は変更前の版 A を返す。開いておいた A では引き続き 1、新しい版 B では 0 が得られる。 | 既存 [test_data.py](../../tests/examples/xor/test_data.py) |
| 生成サンプルの登録（DS-A-002・003・006・015） | 同じ要求 ID とレシピを再送する。別ケースで入力素材の欠落、実体だけの登録を試す。 | 再送は同じサンプル ID を返し、記録した条件での再生成は照合に通る。入力欠落は登録失敗となる。実体だけのサンプルはレシピを要求され、対応付け後は同じ ID から取得できる。 | 既存 [生成登録のテスト](../../tests/examples/xor/test_data.py#L80) |
| 分散した使用履歴（DS-A-010・011） | 同じバッチでプロセス 0 が A・B を報告し、プロセス 1 の未着時に `commit`。次に 1 が B・C を同じ使用 ID で二度送り、さらに C・B に変えて再送する。 | 未着時は `missing_processes=[1]` で確定範囲は空。到着後は A・B と B・C の計4使用になり、順序を変えた再送は `sample_ids` の相違で拒否される。同じ範囲を再確定しても `commit` は1件。 | 既存 [test_usage.py](../../tests/automl/dataset/test_usage.py) |
| バッチのない使用（DS-A-014） | A を位置 1 と 2 で使う。異なる使用 ID で `batch_id` を省略して記録し、`read_usage` で実行を読む。 | A の使用が元の順に2件返る。両方の `batch_id` は `None`、`batches` は空になる。 | 追加 [test_usage.py](../../tests/automl/dataset/test_usage.py) |
| 恒等変換（TR-A-004） | 教師情報付きの辞書を `Identity.apply` に渡し、返った再現情報で再適用する。教師情報のない配列でも同じ操作を行う。 | 両方でデータと付随情報が保たれる。再現情報に未知の条件を渡すと `ValueError`。 | 既存は辞書、配列を追加 [test_identity.py](../../tests/automl/transform/test_identity.py) |
| 変換条件の再適用（TR-A-001・005 の再適用操作） | `ScaleFeatures(2)` で `[[1, -1]]` を変換し、返った条件で再適用する。さらに `scale=3` の再現情報を渡す。 | 初回と再適用はともに `[[2, -2]]`、記録した条件は `{"scale": 2}`。異なる条件では `ValueError` から尺度の不一致が分かる。 | 追加 [test_data.py](../../tests/examples/xor/test_data.py) |

## Model・Trainer

モデルは取り出した状態を比較し、Trainer は保存点の前後で実際に使ったデータと更新後の状態を比較する。[Model の受け入れ基準](../10_requirements/model/05_acceptance_criteria.md)・[Trainer の受け入れ基準](../10_requirements/trainer/05_acceptance_criteria.md)・[停止条件と中断要求](../30_detailed_design/08_trainer.md#停止条件と中断要求)

| 対象 | 入力と操作 | 合格条件 | テスト |
| --- | --- | --- | --- |
| 完全復元と部分引き継ぎ（MD-A-003・005・006） | 学習済みの `TinyMLP` を完全復元する。同じ初期化条件で `hidden_weight` だけを引き継ぐ。幅を変えたモデルでも同じ引き継ぎを試す。 | 完全復元時は全状態が保存値に一致する。部分引き継ぎでは指定した重みだけが保存値に一致し、残りは新規初期値になる。幅が異なる場合は該当名を伴う失敗となる。 | 既存 [test_xor_training.py](../../tests/examples/xor/test_xor_training.py) |
| `Model.build` の構築モード（MD-A-002・003・004） | 重み・移動平均の初期値が (1, 0)、保存値が (7, 3) の小さな具象 Model に `new`、`restore`、`initialize` を指定する。新規に状態を渡す、復元で状態を省く、復元に `retain` を渡す、の3条件も試す。 | 新規の状態は (1, 0)、復元は (7, 3)。引き継ぎでは具象 Model の `initialize_from` が報告した内訳を `initialization_report` から取得できる。不正な組み合わせは各々 `ValueError`。 | 追加 `tests/automl/model/test_build.py` |
| XOR のミニバッチ再開（TN-A-002・005） | 4点を seed 7、バッチサイズ 2 で6ラウンド学習する。通し実行と、第2ラウンドで停止して保存状態から再開する実行を比較する。 | 停止前と再開後につないだサンプル ID の列が、バッチの区切りも含めて通し実行と一致する。最終モデルと学習状態（完了ラウンド・乱数状態）も一致し、進捗の矛盾や不正な乱数状態は理由付きで拒否される。 | 既存 [ミニバッチ再開のテスト](../../tests/examples/xor/test_xor_training.py#L70) |
| XOR の停止判定 | 検証結果を順に「未完了」「目標未達」「目標値と等しい」とする。正解率の最大化と誤り率の最小化をそれぞれ試す。 | 3回目の検証で停止し、当該評価 ID と指標値が保存点に対応する。目標到達済みの状態からの再開では追加更新が0件となる。 | 既存 [test_xor_training.py](../../tests/examples/xor/test_xor_training.py) |
| MNIST の数値計算 | 固定した3入力で MLP の1回の更新量から勾配を求め、交差エントロピーの中心差分と比べる。GAN では固定ノイズと入力画像で識別器・生成器を順に1回ずつ更新する。 | MLP の代表的な重み・バイアスの勾配は既存の絶対許容差 `0.0001` で一致する。GAN では識別器の偽画像に対するロジットが下がり、生成器更新後は出力が 0～1 の範囲で生成器の損失が下がる。 | 既存 [分類モデル](../../tests/examples/mnist/test_classifier.py)・[GAN](../../tests/examples/mnist/test_gan.py) |

## Evaluator・評価履歴

Evaluator が返す完了判定、Experiment が保存した確定件数、履歴を読み始めた時点の評価状態をそれぞれ観測する。[Evaluator の受け入れ基準](../10_requirements/evaluator/05_acceptance_criteria.md)・[評価履歴の参照 API](../30_detailed_design/10_evaluation_history_api.md)

| 対象 | 入力と操作 | 合格条件 | テスト |
| --- | --- | --- | --- |
| `automl.evaluator.evaluate` の評価対象（EV-A-001・002） | 呼び出し引数を記録する Evaluator を用い、モデル単独、予測バイト列単独、両方、両方省略を試す。省略時は `uses_saved_measurements` を切り替える。 | 単独入力はそのまま1回渡る。両方指定と、保存済み測定を使わない Evaluator への両方省略は、呼び出し0回で `ValueError`。保存済み測定からの再集計なら両方省略で1回呼ばれる。 | 追加 `tests/automl/evaluator/test_evaluate.py` |
| 同 API の終了報告（EV-A-004） | 上の Evaluator で有効な対象を受け、`complete` が `True`、`False`、整数 0、欠落の辞書を順に返す。 | 真偽値の2件は報告内容をそのまま返す。整数と欠落は `complete` を示す `ValueError` になる。 | 追加 `tests/automl/evaluator/test_evaluate.py` |
| XOR の予測欠落（EV-A-003・004） | 予定96件のうち `104` の予測だけを欠き、残り95件は正解とする。必要件数を既定値と 95 に切り替える。 | いずれも正解率は 95/95。既定値では `complete=False` で暫定値、95では `complete=True` で確定値。失敗した ID・理由・試行回数 1 が得られる。 | 既存 [test_scoring.py](../../tests/examples/xor/test_scoring.py) |
| 結果の再送と評価終了（EV-A-009・010） | 結果 `r1=0.8` を保存し、必要な集計 `mean` の公開前に成功を報告する。公開後に成功を再送し、終了後に同一結果、値を変えた結果、新しい `r2` を送る。 | 初回の成功報告はエラーに未公開の集計 ID を含み、完了判定は `None`。公開後は終了でき、同一結果は初回の値を返す。値の変更と終了後の新規追加は拒否され、確定件数は1件。 | 既存 [test_evaluations.py](../../tests/automl/experiment/test_evaluations.py) |
| 読み出し中の追記（EV-A-010） | 進行中に `x, y, z` が確定した状態で1件読み、`later` を追加して評価を成功終了する。元のカーソルで続け、次に `after=z` で新しく読む。 | 続きは `y, z` で状態は `running`。新しい読み出しは `later` だけで状態は `succeeded`。 | 既存 [カーソルと正本コピーのテスト](../../tests/automl/experiment/test_evaluations.py#L71) |
| 分割取得の境界（EV-A-010） | 空の評価で `read_results` を呼ぶ。次に `r1`、`r2` を登録し、`after=r2`、未知の ID、`limit=1` を試す。得たカーソルで件数を 0、上限超過、1 の順に指定する。 | 空と末尾は `results=[]`、`next_cursor=None`、状態は `running`。未知 ID は `not_found`、件数違反は `invalid_request` で、上限超過時は許容上限を示す。同じカーソルでの再試行は `r2` だけを返す。 | 追加 [test_evaluations.py](../../tests/automl/experiment/test_evaluations.py) |
| 訂正と判断の由来（EV-A-012・014） | 別実行の訂正版で元の集計を二つの役割で参照し、新しい集計を根拠に判断を記録する。 | 訂正元・理由を取得できる。新しい集計の入力には二つの役割が順に残る。元の集計から利用先を逆引きすると新しい集計は1件で、そこから判断 ID を取得できる。 | 既存 [実行をまたぐ参照のテスト](../../tests/automl/experiment/test_evaluations.py#L122) |

## 保存点・再現情報・外部出力

保存操作は、呼び出し後にもう一度一覧・状態・実体を読み、外から何が利用できるかで判定する。[保存点の詳細設計](../30_detailed_design/04_checkpoint_management.md)・[CLI の応答](../30_detailed_design/01_experiment_api.md)・[MLflow Tracking への記録](../20_basic_design/02_history_and_artifacts.md#mlflow-trackingへの記録)

| 対象 | 入力と操作 | 合格条件 | テスト |
| --- | --- | --- | --- |
| 状態の直列化 | 辞書とバイト列を持つ名前付きモデル、学習状態、進捗を `encode_states` に渡す。モデル名の挿入順を逆にしたものと、途中で切った保存データも試す。 | 挿入順によらず保存バイト列が一致し、`decode_states` で全状態が戻る。切断データは `ValueError`。 | 既存 [状態の往復テスト](../../tests/automl/experiment/test_training.py#L30) |
| 保存失敗と `latest`（EX-A-021、TN-A-009） | 更新301、300の順に保存した後、301のファイルを破損させる。別ケースで保存トランザクションを公開前にロールバックし、同じ ID で再試行する。 | `latest` は最初は301、破損後は300を指す。301の明示指定はその ID を示して失敗する。ロールバック後は一覧と保存先が空で、再試行後に内容を読み出せる。 | 既存 [test_artifacts.py](../../tests/automl/experiment/test_artifacts.py) |
| 使用中の保存点の削除（EX-A-022・023） | 利用者2件の片方だけを解放して削除を依頼する。全解放後、ファイル削除の途中で失敗させ、同じ保存点 ID で再依頼する。 | 使用中は `in_use` で `retained`。途中失敗では `deleting` として検出され、新規利用が拒否される。再依頼後は `deleted` で、利用側の独立コピーを読める。 | 既存 [test_artifacts.py](../../tests/automl/experiment/test_artifacts.py) |
| 再実行用入力の確保（PR-F-003） | 検証時に記録した CSV の SHA-256 と異なる内容を用意し、`build_manifest` に初めて保存を依頼する。 | 入力が変わった理由を含む例外となり、正本の `inputs/` に当該入力は公開されない。 | 既存 [入力変更のテスト](../../tests/automl/experiment/test_replay.py#L119) |
| CLI の終了コード | `cli.main` に不正な引数を渡す。次に Experiment の応答を差し替え、失敗済み実行に `status` と `wait`、進行中実行に `wait --timeout 0` を呼ぶ。 | 引数不正は終了コード 1、stderr にエラーの JSON、Store の生成は0回。実行状態の応答は stdout に JSON、終了コードは順に 0、1、2。 | 既存 [test_cli.py](../../tests/automl/test_cli.py) |
| MLflow に渡す指標（PR-F-006） | 同期済みイベントの後に、step 10 の `loss=0.25` と、同じ評価の暫定・確定集計を別 ID で置く。指標に文字列も混ぜ、`_metrics(events, after)` を呼ぶ。 | 新着分から `train.loss` が step 10・値 0.25 で1件得られる。暫定集計は `provisional.`、確定集計は `eval.` で始まる名前になり、文字列は指標から除かれる。ローカル Tracking への登録と再同期は既存の結合テストが担う。 | 追加 [test_mlflow.py](../../tests/automl/test_mlflow.py) |

保存点の公開と Dataset の `commit` を一緒に検証する結合テストでは、使用記録を1件遅らせて保存を失敗させる。残りが届いてから同じ要求を同時再送し、履歴にある保存点 ID と指定範囲の `commit` が各1件であることを確認する（TN-A-012）。[保存要求の再送テスト](../../tests/automl/experiment/test_training.py#L40)

## 実装順と実行方法

追加分は、まず `Model.build` の復元条件、`evaluate` の入力と完了判定、実行全体の終了判定から着手する。これらを取り違えると、別の状態から学習を再開したり、失敗した実行を成功として後続の実験に使ったりする。次に、設定検証、使用履歴、変換の再適用、ページ境界、表示用指標を補う。

プロジェクトルートで `make venv` を実行するとテスト用環境ができる。既存の設定・使用履歴・恒等変換のケースを選んで実行する例を示す。単体・結合・システムを含む既存スイート全体の回帰確認には `make test` を使う。

```sh
.venvs/test/bin/python -m pytest -q \
  tests/automl/experiment/test_config.py \
  tests/automl/dataset/test_usage.py \
  tests/automl/transform/test_identity.py
```

新しい手法を対象に加えるときは、その手法で再開後に一致させる状態と数値の許容差、評価の完了条件を実装者と決め、入力例と期待値をケースに追記する。[Trainer の未確定事項](../10_requirements/trainer/06_open_questions.md)・[Evaluator の未確定事項](../10_requirements/evaluator/06_open_questions.md)
