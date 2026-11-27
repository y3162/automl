# XOR のモデル比較

同じ CSV の訓練320件を使い、TinyMLP と、2つの入力の積を特徴にするロジスティック回帰を比較した。TinyMLP は25更新ごとの検証で第50更新に初めて96/96に達し、第100更新で学習を終了した。積を使うモデルは第1更新で96/96に達して早期終了した。別のテスト96件では両モデルとも全問正解だった。[比較に使用した CSV](../../../configs/examples/xor/samples.csv)・[TinyMLP の設定](../../../configs/examples/xor/full_batch.yaml)・[積を使うモデルの設定](../../../configs/examples/xor/product_target.yaml)

この比較の履歴がある `data/store/` では、次のコマンドでテスト評価に使った保存点と結果を確認できる。

```sh
# TinyMLP
automl history --run R764eccd695eb4f6d9a0c27829973c77e --summary

# 入力の積を使うロジスティック回帰
automl history --run R6212d10d2ad744a19561b42701e17d4c --summary
```
