from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "tickets_dummy.csv"


@dataclass(frozen=True)
class CategorySpec:
    category: str
    department: str
    requester_roles: list[str]
    objects: list[str]


CATEGORY_SPECS = [
    CategorySpec(
        category="勤怠",
        department="総務",
        requester_roles=["社員", "管理者"],
        objects=[
            "出勤打刻",
            "退勤打刻",
            "休憩時間",
            "有給申請",
            "残業申請",
            "勤務表",
            "勤怠集計",
            "休日出勤申請",
            "リモート勤務区分",
            "承認ワークフロー",
        ],
    ),
    CategorySpec(
        category="請求",
        department="経理",
        requester_roles=["経理担当", "管理者"],
        objects=[
            "請求書PDF",
            "月末請求",
            "請求金額",
            "税率計算",
            "請求書番号",
            "取引先マスタ",
            "入金ステータス",
            "一括出力",
            "送付先メール",
            "会社印プレビュー",
        ],
    ),
    CategorySpec(
        category="権限",
        department="情シス",
        requester_roles=["管理者", "社員"],
        objects=[
            "管理画面",
            "ユーザー追加",
            "閲覧権限",
            "承認者権限",
            "権限グループ",
            "外部パートナー権限",
            "部署異動後の権限",
            "プロジェクトフォルダ",
            "経理メニュー",
            "権限変更履歴",
        ],
    ),
    CategorySpec(
        category="システム障害",
        department="開発",
        requester_roles=["社員", "管理者", "開発者"],
        objects=[
            "受注管理システム",
            "検索画面",
            "ダッシュボード",
            "保存処理",
            "夜間バッチ",
            "一覧画面",
            "決済処理",
            "添付ファイルアップロード",
            "申請データ",
            "お知らせ欄",
        ],
    ),
    CategorySpec(
        category="ネットワーク",
        department="インフラ",
        requester_roles=["社員", "管理者", "開発者"],
        objects=[
            "社内WiFi",
            "VPN接続",
            "拠点間ネットワーク",
            "社内DNS",
            "インターネット接続",
            "クラウド環境接続",
            "会議室LAN",
            "ゲストWiFi",
            "ルーター",
            "ファイル共有通信",
        ],
    ),
    CategorySpec(
        category="アカウント",
        department="情シス",
        requester_roles=["社員", "管理者"],
        objects=[
            "ログイン",
            "多要素認証",
            "新入社員アカウント",
            "アカウントロック",
            "退職者アカウント",
            "メールエイリアス",
            "共有アカウント",
            "表示名",
            "チャットツールアカウント",
            "不審ログイン通知",
        ],
    ),
    CategorySpec(
        category="データ連携",
        department="開発",
        requester_roles=["管理者", "開発者", "社員"],
        objects=[
            "外部システム連携",
            "CSV取込",
            "マスタ反映",
            "API連携",
            "夜間連携",
            "商品コード取込",
            "認証トークン",
            "連携履歴画面",
            "テスト環境マスタ",
            "レスポンス時間",
        ],
    ),
    CategorySpec(
        category="端末",
        department="情シス",
        requester_roles=["社員", "管理者"],
        objects=[
            "会社PC",
            "プリンタ",
            "Webカメラ",
            "貸与PC",
            "全社配布アプリ",
            "キーボード",
            "セキュリティ警告",
            "モニター接続",
            "標準ブラウザ",
            "端末交換",
        ],
    ),
]

LOW_ISSUES = [
    "の操作方法を確認したいです",
    "の入力項目の意味を知りたいです",
    "の表示名を変更したいです",
    "の設定手順を教えてください",
    "の過去データを確認したいです",
    "の通知設定を変更したいです",
    "の画面の場所が分かりません",
    "の軽微な表示文言を修正したいです",
]

MIDDLE_ISSUES = [
    "が一覧に反映されません",
    "を保存するとエラーになります",
    "の検索結果が表示されません",
    "の通知が担当者へ届きません",
    "を開くと白い画面になります",
    "の処理に時間がかかり業務効率が落ちています",
    "の一部データが更新されていません",
    "のステータスを変更できません",
    "で特定条件だけ処理が失敗します",
    "の承認先が想定と異なります",
    "のダウンロードに失敗します",
    "で部署内の複数名に影響が出ています",
]

HIGH_ISSUES = [
    "が全社で利用できず業務が止まっています",
    "の締め処理が完了せず期限に間に合いません",
    "の障害により複数部署の作業が停止しています",
    "でセキュリティリスクがある状態です",
    "の夜間処理が失敗し本日の業務に影響しています",
    "の重要データが欠落しています",
    "が大量に失敗して顧客対応に影響しています",
    "で業務継続に必要な処理が実行できません",
    "の不具合が本番環境で継続発生しています",
    "の復旧を本日中に行わないと月次処理に影響します",
]

PRIORITY_SETTINGS = {
    "Low": {
        "issues": LOW_ISSUES,
        "impact_scopes": ["個人", "個人", "部署"],
        "channels": ["問い合わせフォーム", "Slack", "メール"],
    },
    "Middle": {
        "issues": MIDDLE_ISSUES,
        "impact_scopes": ["個人", "部署", "部署", "複数部署"],
        "channels": ["Slack", "問い合わせフォーム", "メール"],
    },
    "High": {
        "issues": HIGH_ISSUES,
        "impact_scopes": ["全社", "複数部署", "全社", "部署"],
        "channels": ["電話", "メール", "Slack"],
    },
}


def build_records() -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    ticket_number = 1

    for spec in CATEGORY_SPECS:
        for priority in ["Low", "Middle", "High"]:
            setting = PRIORITY_SETTINGS[priority]
            issues = setting["issues"]
            impact_scopes = setting["impact_scopes"]
            channels = setting["channels"]

            for index, issue in enumerate(issues):
                object_name = spec.objects[index % len(spec.objects)]
                requester_role = spec.requester_roles[index % len(spec.requester_roles)]
                record = {
                    "ticket_id": f"TKT-{ticket_number:04d}",
                    "inquiry_text": f"{object_name}{issue}",
                    "label_category": spec.category,
                    "label_priority": priority,
                    "label_department": spec.department,
                    "impact_scope": impact_scopes[index % len(impact_scopes)],
                    "requester_role": requester_role,
                    "channel": channels[index % len(channels)],
                }
                records.append(record)
                ticket_number += 1

    return records


def main() -> None:
    records = build_records()
    if len(records) != 240:
        raise ValueError(f"Expected 240 records, but got {len(records)}.")

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with DATA_PATH.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    print(f"wrote {len(records)} records to {DATA_PATH}")


if __name__ == "__main__":
    main()
