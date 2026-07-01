# ML Helpdesk Triage

業務問い合わせ・障害チケットの一次切り分けを支援する、教師あり機械学習アプリの開発プロジェクトです。

## GitHubでの運用方針

GitHubには、ソースコード、ダミーデータ、設計メモ、README、開発ログを載せます。
学習済みモデル、評価レポート、Pythonキャッシュは再生成できるため、`.gitignore` で管理対象から外します。

この方針により、採用担当者には実装内容と再現手順を見せつつ、生成物でリポジトリが重くならない構成にします。

## 現在できていること

- ダミーチケットCSVの作成
- TF-IDFによるテキスト特徴量化
- Logistic Regressionによる多クラス分類
- 以下3タスクのベースライン評価
  - 問い合わせカテゴリ分類
  - 優先度分類
  - 推奨担当部署分類
- 評価レポート出力
- 学習済みモデルの保存
- 任意の問い合わせ文に対するCLI予測

## データ

- `data/tickets_dummy.csv`
- 240件
- 8カテゴリ x 30件
- 実在する企業、顧客、社員、問い合わせ内容とは関係しない完全なダミーデータ

データは以下のコマンドで再生成できます。

```bash
python src/generate_dummy_data.py
```

## ベースラインモデル

`src/train_baseline.py` で以下を実行します。

- 文字1〜4gramのTF-IDF
- 多クラスLogistic Regression
- stratified train/test split
- Accuracy、Macro F1、混同行列、予測結果CSVの出力

`scikit-learn` がない環境でも動かせるよう、TF-IDFとLogistic Regressionは `numpy` ベースで実装しています。

## 実行方法

プロジェクトフォルダへ移動してから実行します。

```bash
python src/train_baseline.py
```

学習時の出力例です。

```text
category: accuracy=0.7344 macro_f1=0.7305 train=176 test=64
priority: accuracy=1.0 macro_f1=1.0 high_recall=1.0 train=180 test=60
department: accuracy=0.7541 macro_f1=0.72 train=179 test=61
```

学習後、以下のようなコマンドで新しい問い合わせ文を分類できます。

```bash
python src/predict.py --text "出勤打刻が全社で利用できず業務が止まっています" --impact-scope 全社 --requester-role 管理者 --channel Slack
```

出力例です。

```text
予測結果
- category: 勤怠 (0.5602)
  - 勤怠: 0.5602
  - 端末: 0.0963
  - 権限: 0.0848
- priority: High (0.8932)
  - High: 0.8932
  - Middle: 0.0727
  - Low: 0.0341
- department: 総務 (0.5699)
  - 総務: 0.5699
  - 情シス: 0.2103
  - 開発: 0.0930
```

## 現在の評価結果

| target | accuracy | macro_f1 | high_recall | train_count | test_count |
|---|---:|---:|---:|---:|---:|
| category | 0.7344 | 0.7305 | - | 176 | 64 |
| priority | 1.0000 | 1.0000 | 1.0000 | 180 | 60 |
| department | 0.7541 | 0.7200 | - | 179 | 61 |

## レポート出力先

以下は `python src/train_baseline.py` の実行時に生成されます。

- `reports/baseline/summary.csv`
- `reports/baseline/category_metrics.json`
- `reports/baseline/category_confusion_matrix.csv`
- `reports/baseline/category_predictions.csv`
- `reports/baseline/priority_metrics.json`
- `reports/baseline/priority_confusion_matrix.csv`
- `reports/baseline/priority_predictions.csv`
- `reports/baseline/department_metrics.json`
- `reports/baseline/department_confusion_matrix.csv`
- `reports/baseline/department_predictions.csv`

## モデル出力先

以下は `python src/train_baseline.py` の実行時に生成されます。

- `models/baseline/category_metadata.json`
- `models/baseline/category_arrays.npz`
- `models/baseline/priority_metadata.json`
- `models/baseline/priority_arrays.npz`
- `models/baseline/department_metadata.json`
- `models/baseline/department_arrays.npz`

## 次に改善すること

- カテゴリごとの表現ゆれをさらに増やす
- 誤分類例を分析し、特徴量設計を改善する
- Streamlitで予測画面を作る
- 予測結果をユーザーが修正できるフィードバック機能を作る
