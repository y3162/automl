## 未確定事項

今回の議論で挙がった共通要件は確定済み。実装・版・設定の指定はDataset・Transformと共通とし、構築後の推論などの操作と入出力はプロジェクト・タスクごとに定める。[担当範囲](01_scope.md)

モデル状態の取り出し・完全復元と部分引き継ぎはModelの共通契約に含める。再開に使う保存点は、Trainerが同じ地点で用意したモデル状態と学習状態をExperimentが対応付けて管理する。[機能要件](02_functional_requirements.md)・[Modelの詳細設計](../../30_detailed_design/07_model.md)
