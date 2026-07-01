# GitHub Publish Steps

## 推奨リポジトリ名

`ml-helpdesk-triage`

## 公開前に確認すること

- 実在する企業名、社員名、顧客名、個人情報を含めない
- `data/tickets_dummy.csv` が完全なダミーデータであることを確認する
- `models/`、`reports/`、`__pycache__/` はGitHubに含めない
- READMEの実行手順で、第三者が再現できる状態にする

## 初回公開の流れ

GitHub上で `ml-helpdesk-triage` という空のリポジトリを作成してから、ローカルで以下を実行します。

```bash
git init -b main
git add .
git commit -m "feat: 教師あり学習の問い合わせ分類ベースラインを追加"
git remote add origin https://github.com/TakeruKawazoe/ml-helpdesk-triage.git
git push -u origin main
```

## 2回目以降の開発フロー

```bash
git status
git add .
git commit -m "feat: 変更内容を短く書く"
git push
```

## コミットメッセージ例

- `feat: 予測CLIを追加`
- `add: ダミーデータ設計を追加`
- `fix: 優先度予測の評価処理を修正`
- `docs: READMEに実行手順を追記`
