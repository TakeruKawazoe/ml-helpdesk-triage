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
- ローカルWeb画面からの予測
- Web画面での予測履歴保存
- 予測結果に対する修正フィードバック保存
- 修正フィードバックを利用した安全な自動再学習
- Middle・Highの問い合わせをSlackへ担当者メンション付きで通知

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

## 学習方法

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

## Web画面で使う方法

学習済みモデルを作成したあと、以下を実行します。

```bash
python src/web_app.py
```

ブラウザで以下を開きます。

```text
http://127.0.0.1:8000
```

Web画面では、問い合わせ文、影響範囲、依頼者、経路を入力して、カテゴリ・優先度・担当部署を確認できます。

## Notion連携

予測結果をNotionのデータベースへ自動登録し、修正フィードバックを同じ行へ反映できます。連携先データベースに不足している列は、初回同期時に自動で追加されます。

NotionのDeveloper Portalで内部インテグレーションを作成し、以下の権限を有効にしてください。

- Read content
- Insert content
- Update content

次に、連携先データベースのメニューからインテグレーションを追加します。トークンはソースコードやGitへ保存せず、Webアプリを起動するPowerShellで環境変数へ設定してください。

```powershell
$env:NOTION_API_TOKEN="Notionのインテグレーショントークン"
$env:NOTION_DATABASE_ID="3a5d3d7e828c80d59124c41c212013e9"
python src/web_app.py
```

Windowsのユーザー環境変数へ上記2項目を保存済みの場合は、次のコマンドで起動できます。

```powershell
powershell -ExecutionPolicy Bypass -File .\run_web_app.ps1
```

データベースに複数のデータソースが存在する場合だけ、`NOTION_DATA_SOURCE_ID` も設定します。環境変数が未設定の場合はNotion連携を無効として明示し、予測とローカル履歴保存は継続します。APIエラーの場合もローカル履歴を保持し、画面と履歴に同期失敗を表示します。

- [Notion APIの認証設定](https://developers.notion.com/guides/get-started/authorization)
- [Notion APIでページを作成する](https://developers.notion.com/reference/post-page)

## Slack通知

予測優先度が`Middle`または`High`の場合、共通のSlackチャンネルへ、予測担当部署の担当者をメンションして通知します。通知には問い合わせ文、優先度、カテゴリ、担当部署、Notionリンクを含めます。`Low`は通知対象外です。

Slack Appを作成し、Bot Token Scopeに`chat:write`を追加してワークスペースへインストールします。Botを通知先チャンネルへ招待したあと、次のWindowsユーザー環境変数を設定してください。

```text
SLACK_BOT_TOKEN
SLACK_CHANNEL_ID
SLACK_MENTION_SOMU
SLACK_MENTION_KEIRI
SLACK_MENTION_JOSYS
SLACK_MENTION_DEVELOPMENT
SLACK_MENTION_INFRASTRUCTURE
```

`SLACK_BOT_TOKEN`には`xoxb-`で始まるBot User OAuth Token、`SLACK_CHANNEL_ID`には共通通知チャンネルのID、`SLACK_MENTION_...`には各部署担当者のSlackメンバーIDを設定します。トークンをソースコード、README、Gitへ保存しないでください。

環境変数という仕組み自体はWindows専用ではありません。別のPCでは同じ変数名をそのPCへ設定し、Dockerやクラウドでは実行環境のシークレット設定から同じ変数名を渡します。コードと認証情報を分離するため、実行環境ごとに認証情報の設定が必要です。

環境変数がすべて未設定の場合はSlack通知を無効として予測を継続します。一部だけ設定されている場合やSlack APIでエラーが発生した場合も、予測・ローカル履歴・Notion登録は保持し、画面と履歴に通知失敗を表示します。

- [Slack Appの設定](https://docs.slack.dev/app-management/quickstart-app-settings/)
- [chat.postMessage API](https://api.slack.com/methods/chat.postMessage)
- [OAuthトークンの安全な管理](https://api.slack.com/docs/oauth-safety)

## 履歴と修正フィードバック

Web画面から予測すると、予測内容は `storage/prediction_feedback.csv` に保存されます。
このファイルには、問い合わせ文、入力条件、予測カテゴリ、予測優先度、予測担当部署、修正後の正解ラベル、メモを保存します。

予測結果の下に表示される「修正フィードバック」から、カテゴリ・優先度・担当部署の正解ラベルを登録できます。
保存後は「予測履歴」に反映され、あとから誤分類例として見直せます。

`storage/` はローカル実行時の履歴保存先のため、Git管理対象から除外しています。
採用担当者には実装内容を見せつつ、個人の試行履歴や実行時データはリポジトリへ含めない方針です。

### Web API

- `POST /api/predict`: 問い合わせ文を予測し、予測履歴を作成する
- `GET /api/history?limit=10`: 最新の予測履歴を取得する
- `POST /api/feedback`: 予測IDに対して修正フィードバックを保存する
- `GET /api/retraining`: 自動再学習の進行状況と直近の評価結果を取得する

## フィードバックによる自動再学習

カテゴリ、優先度、担当部署の正解ラベルがすべて保存されたフィードバックを、再学習用データとして利用します。未確認の予測結果やラベルが一部しかない履歴は学習へ混ぜません。

新規または更新されたフィードバックが10件たまると、Webアプリがバックグラウンドで候補モデルを学習します。学習に約1分かかっても、予測画面は引き続き利用できます。

候補モデルはすぐ本番へ反映せず、乱数シードを固定した既存ダミーデータのテスト部分で現行モデルと比較します。以下をすべて満たした場合だけ自動採用します。

- カテゴリ、優先度、担当部署のMacro F1が現行モデルより低下しない
- 優先度`High`のRecallが現行モデルより低下しない

基準を満たさない候補は破棄し、現行モデルを維持します。採用時はモデル読込を一時的にロックし、3種類のモデルをまとめて切り替えます。旧モデルは`models/archive/`へ退避されます。

再学習の状態と評価結果は以下へ保存されます。いずれもローカル実行データのためGit管理対象外です。

- `storage/retraining_status.json`: 待機、学習中、採用、不採用、失敗の状態
- `storage/retraining_state.json`: 処理済みフィードバックの識別情報
- `reports/retraining/{実行日時}/metrics.json`: 現行モデルと候補モデルの比較結果

10件に達するまでは画面に残り件数を表示し、再学習は実行しません。

## テスト

履歴保存、修正フィードバック、Notion連携、Slack通知の回帰テストは以下で実行できます。

```bash
python -m unittest discover -s tests
```

## CLIで使う方法

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
- 実利用者による修正フィードバックを増やし、評価用データも実データへ段階的に置き換える
- 履歴画面にカテゴリ・優先度・担当部署の絞り込みを追加する
- 小規模なデプロイ環境で第三者が触れる状態にする
