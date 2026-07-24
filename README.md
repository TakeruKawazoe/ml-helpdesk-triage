# ML Helpdesk Triage

業務問い合わせ・障害チケットの一次切り分けを支援する、教師あり機械学習アプリです。カテゴリ、優先度、推奨担当部署を予測し、信頼度が評価済み基準を下回る場合は「要確認（一次受付）」へ回します。

## 主な機能

- TF-IDFと線形分類器による3タスク分類
- 文字TF-IDF、単語+文字TF-IDF、Logistic Regression、LinearSVC、クラス重みの同一条件比較
- 意味テンプレート単位の分割による類似文漏洩の防止
- モデル選定に使わない固定未知文による最終評価
- 信頼度に基づく自動振り分けと要確認の切り替え
- 予測履歴の検索・絞り込み・ページング・CSV出力
- 理由と実行者を記録する履歴の論理削除・復元
- 人が確認した修正フィードバックの保存
- Notionへの予測・修正結果の同期
- Middle・HighのSlack担当者通知
- 品質ゲートを通過したフィードバックだけを使う自動再学習

## データ

| ファイル | 件数 | 用途 |
|---|---:|---|
| `data/tickets_dummy.csv` | 540 | 学習、モデル選定、信頼度基準の調整 |
| `data/tickets_holdout.csv` | 81 | 採用モデル決定後の固定未知文評価 |

対象は9カテゴリです。

- 勤怠
- 請求
- 権限
- システム障害
- ネットワーク
- アカウント
- データ連携
- 端末
- その他・対象外

空調、照明、備品などをITカテゴリへ無理に分類しないため、`その他・対象外`を設けています。現在は一次受付先を総務としています。

すべて完全なダミーデータであり、実在する企業、顧客、社員、問い合わせとは関係しません。詳細は`data/dataset_design.md`を参照してください。

```bash
python src/generate_dummy_data.py
```

## 学習とモデル比較

必要パッケージを準備し、改善モデルを学習します。

```bash
pip install -r requirements.txt
python src/train_improved.py
```

学習処理は次の順で行います。

1. 同じ意味テンプレートから作った言い換え文を同じグループに固定する
2. グループ単位で開発、検証、信頼度調整に分割する
3. 5系統7候補を検証データで比較する
4. Macro F1を主指標として採用候補を決める
5. 優先度が同点の場合はHighのRecallと正解Highへの確信度を優先する
6. 信頼度調整データで自動振り分けの最低信頼度を計測する
7. 採用決定後にだけ固定未知文を評価する

### 比較候補

- 文字1〜4gram TF-IDF + Logistic Regression
- 単語1〜2gramと文字1〜4gram TF-IDF + Logistic Regression
- 上記2構成の`class_weight=balanced`
- 単語+文字TF-IDF + LinearSVC + 確率校正
- Highを重くした優先度用Logistic Regression

### 現在の固定未知文評価

| target | 採用モデル | Accuracy | Macro F1 | High見逃し | 部署誤振り分け |
|---|---|---:|---:|---:|---:|
| category | hybrid LinearSVC | 1.0000 | 1.0000 | - | - |
| priority | hybrid LinearSVC | 1.0000 | 1.0000 | 0 | - |
| department | char Logistic Regression | 1.0000 | 1.0000 | - | 0 |

これは81件の合成固定未知文に対する結果であり、実務精度を保証しません。実データを取得した段階で、固定評価データを実データへ置き換えて再計測する必要があります。

学習量を各カテゴリ20件、40件、60件に変えた学習曲線も`reports/improved/learning_curve.csv`へ出力します。精度だけでなく学習時間と1件当たり推論時間も記録します。

## 信頼度と要確認

信頼度基準は固定値を勘で設定せず、信頼度調整データ上で「自動振り分けした予測の正答率90%以上」を満たす範囲から計測します。

3タスクのうち1つでも基準未満なら、ラベル候補は表示したまま`要確認（一次受付）`とします。履歴、Notion、Slackにも状態と理由を残します。

現在の合成固定未知文では、自動振り分け対象の正答率は全タスクで100%でした。優先度は誤判定を抑えるため自動振り分け対象が約33%に限定され、残りは要確認になります。精度と自動化率のトレードオフを隠さない設計です。

## Web画面

学習後、次のコマンドで起動します。

```powershell
powershell -ExecutionPolicy Bypass -File .\run_web_app.ps1
```

ブラウザで`http://127.0.0.1:8000`を開きます。

Notion・Slackの連携設定がない環境では、次のコマンドでもローカル機能のみを起動できます。

```bash
python src/web_app.py
```

## 履歴管理

履歴画面では、キーワード、登録日、カテゴリ、優先度、担当部署、振り分け、フィードバック、Notion、Slack、削除状態で絞り込めます。カテゴリ、優先度、担当部署は、修正フィードバックがある場合は修正後の値を検索条件に使います。CSV出力にも画面上の絞り込みと並び順を適用し、Excelで開けるUTF-8 BOM付きで出力します。

削除はデータを消去しない論理削除です。実行者ID、理由、日時を記録し、削除済み表示から復元できます。削除済み履歴は通常表示、CSV出力、自動再学習から除外されます。Notion上の行は監査情報として残し、ローカル履歴の削除とは連動させません。

## 修正フィードバックと自動再学習

修正フィードバックには、正解カテゴリ、正解優先度、正解担当部署、確認者ID、修正理由が必要です。予測のまま未確認の履歴と削除済み履歴は学習に混ぜません。

再学習を開始する条件は次のとおりです。

- 新規または更新フィードバックが20件以上
- カテゴリ、優先度、担当部署の全ラベルが各2件以上
- 確認者が2名以上
- 全件に修正理由がある

候補モデルはすぐ採用せず、固定評価データで現行モデルと比較します。

- 3タスクのMacro F1が低下しない
- 優先度HighのRecallが低下せず、見逃し件数が増えない
- 担当部署の誤振り分け件数が増えない
- 自動振り分け対象の正答率が90%以上
- 自動振り分け対象が20%以上で、現行より15ポイントを超えて低下しない

採用・却下の判定理由と指標は`reports/retraining/{実行日時}/decision.json`へ保存します。旧モデルは採用時に`models/archive/`へ退避します。現在の固定評価データは合成データであり、実データではないことも判定レポートへ記録します。

## Notion連携

Notionの内部インテグレーションにRead content、Insert content、Update contentを付与し、対象データベースへ接続します。認証情報はGitへ保存せず、実行環境の環境変数へ設定します。

```text
NOTION_API_TOKEN
NOTION_DATABASE_ID
```

データベースに複数データソースがある場合だけ`NOTION_DATA_SOURCE_ID`も設定します。必要な列は初回同期時に追加されます。同期失敗時もローカル履歴と予測は残ります。

- [Notion APIの認証設定](https://developers.notion.com/guides/get-started/authorization)
- [Notion APIでページを作成する](https://developers.notion.com/reference/post-page)

## Slack通知

予測優先度がMiddleまたはHighの場合、共通チャンネルへ担当者メンション付きで通知します。

```text
SLACK_BOT_TOKEN
SLACK_CHANNEL_ID
SLACK_MENTION_SOMU
SLACK_MENTION_KEIRI
SLACK_MENTION_JOSYS
SLACK_MENTION_DEVELOPMENT
SLACK_MENTION_INFRASTRUCTURE
```

トークンやメンバーIDはコードへ埋め込まず、各PC、Docker、クラウドのシークレット設定から同じ変数名で渡します。通知失敗時も予測、Notion登録、ローカル履歴は残ります。

- [Slack Appの設定](https://docs.slack.dev/app-management/quickstart-app-settings/)
- [chat.postMessage API](https://api.slack.com/methods/chat.postMessage)
- [OAuthトークンの安全な管理](https://api.slack.com/docs/oauth-safety)

## テスト

```bash
python -m unittest discover -s tests
```

## CLI予測

```bash
python src/predict.py --text "全社でシステムにログインできず、業務が停止しています。" --impact-scope 全社 --requester-role 管理者 --channel Slack --json
```

## 生成物

GitHubにはコード、ダミーデータ、設計、テストを保存します。学習済みモデル、評価レポート、予測履歴は再生成またはローカル実行で作られるため、`.gitignore`で除外しています。

- `models/improved/`: 採用モデルと信頼度基準
- `reports/improved/model_comparison.csv`: 全候補比較
- `reports/improved/summary.csv`: 固定未知文評価
- `reports/improved/learning_curve.csv`: 学習曲線
- `storage/prediction_feedback.csv`: 予測履歴と修正フィードバック
- `storage/history_events.csv`: 履歴の削除・復元監査イベント

## 残る課題

- 実利用者から許諾を得た匿名化データで固定評価セットを作る
- 実データ上で信頼度基準を再計測する
- 確認者IDを認証済みユーザーIDへ置き換える
- 実データの分布変化とラベル偏りを定期監視する
- 第三者が試せる小規模環境へデプロイする
