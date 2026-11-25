# モジュール間の実行時の契約

学習や評価で、構築済みのモデルと取得済みのデータをどう受け渡すかを定める。[実行フロー](01_execution_flow.md)

## モデルの利用

Modelの共通APIは、モデルの構築・初期化、現在のモデル状態の取り出し、完全復元、部分引き継ぎを担う。推論などで構築済みモデルに求める操作と入出力はプロジェクト・タスクごとに定める。TrainerやEvaluatorの個別実装は、必要とする操作と入出力の条件を定め、選ばれたモデルが適合するかを確認する。[Modelの機能要件](../10_requirements/model/02_functional_requirements.md)

学習中の評価では、Trainerが評価時点のモデルをEvaluatorに渡せる。並行評価では、その後の学習更新が評価中のモデル状態に影響しない形で渡す。後日評価ではModelのAPIで指定した状態を復元するか、保存済みの予測結果をEvaluatorに渡す。

## データの受け渡し

Datasetの公開APIから、データ本体、サンプルID、データの解釈に必要な付随情報を取得できるようにする。データ本体の表現と必要な項目は、タスクと処理に応じて定める。Transformの呼び出し側が処理対象を選んで引き渡し、Transformは処理結果と再現に必要な情報を返す。[Datasetの機能要件](../10_requirements/dataset/02_functional_requirements.md) · [Transformの機能要件](../10_requirements/transform/02_functional_requirements.md)

## 適合性の確認

設定や指定された実装から分かる不整合は、Experimentによる実行の受付前に検出し、呼び出し側に理由を返す。たとえば、Trainerが要求する操作を指定モデルが提供できない場合は、その組み合わせが適合しない理由を返す。

実データを見て初めて分かる不整合は、そのデータを受け取る処理で検出する。たとえばTransformに必要な項目が欠けていれば、対象サンプルを特定できる情報と理由を呼び出し側に伝える。評価中の不整合は、設定されたエラー対応に従って扱う。[設定と個別実装](03_configuration_and_implementations.md) · [Evaluatorの機能要件](../10_requirements/evaluator/02_functional_requirements.md)

## 生成データの登録

学習に使う生成データについて、Trainerなどの呼び出し側は、入力のID・版、処理順、各Transformが返した版・適用条件をDatasetに渡す。Datasetは再現情報をDBに保存し、使用時のレシピの取得と、プロジェクト・タスクが指定した項目の適用値による検索を担う。中間結果と最終結果の実体の保管はプロジェクト・タスクの方針に従う。[Transformの詳細設計](../30_detailed_design/06_transform.md)

Trainerは実際に使った入力を実行に対応付けてDatasetに伝える（`add`）。バッチの有無と範囲は呼び出し側が決める。再開に対応する手法では、Datasetが候補の保存点に反映された使用履歴の範囲を対応付ける（`commit`）。Experimentはモデル状態・学習状態・進捗の保存とこの対応付けの両方が完了したことを確認してから、再開可能な保存点を公開する。[Datasetの詳細設計](../30_detailed_design/05_dataset.md)・[チェックポイントと学習再開](05_checkpoints_and_resume.md)

## 実行の成果物

Trainer と Evaluator は `context.save_artifact(name, data)` にファイル名とバイト列を渡せる。Experiment が実行に紐付けて保存し、成果物IDを返す。評価中の成果物は個々の評価IDにも紐付ける。
