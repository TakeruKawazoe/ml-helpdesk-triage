from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
TRAIN_DATA_PATH = ROOT_DIR / "data" / "tickets_dummy.csv"
HOLDOUT_DATA_PATH = ROOT_DIR / "data" / "tickets_holdout.csv"
PRIORITIES = ["Low", "Middle", "High"]
TRAIN_CASES_PER_PRIORITY = 10
TRAIN_VARIANTS_PER_CASE = 2
HOLDOUT_CASES_PER_PRIORITY = 3


@dataclass(frozen=True)
class CategorySpec:
    category: str
    department: str
    requester_roles: tuple[str, ...]
    objects: tuple[str, ...]
    holdout_objects: tuple[str, ...]


CATEGORY_SPECS = [
    CategorySpec(
        "勤怠",
        "総務",
        ("社員", "管理者"),
        (
            "出勤打刻", "退勤打刻", "休憩時間", "有給申請", "残業申請",
            "勤務表", "勤怠集計", "休日出勤申請", "リモート勤務区分", "承認ワークフロー",
        ),
        ("タイムカード", "出退勤記録", "休暇管理"),
    ),
    CategorySpec(
        "請求",
        "経理",
        ("経理担当", "管理者"),
        (
            "請求書PDF", "月末請求", "請求金額", "税率計算", "請求書番号",
            "取引先マスタ", "入金ステータス", "一括出力", "送付先メール", "会社印プレビュー",
        ),
        ("インボイス帳票", "請求明細", "支払通知書"),
    ),
    CategorySpec(
        "権限",
        "情シス",
        ("管理者", "社員"),
        (
            "管理画面", "ユーザー追加", "閲覧権限", "承認者権限", "権限グループ",
            "外部パートナー権限", "部署異動後の権限", "プロジェクトフォルダ", "経理メニュー", "権限変更履歴",
        ),
        ("ロール設定", "アクセス制御", "管理者メニュー"),
    ),
    CategorySpec(
        "システム障害",
        "開発",
        ("社員", "管理者", "開発者"),
        (
            "受注管理システム", "検索画面", "ダッシュボード", "保存処理", "夜間バッチ",
            "一覧画面", "決済処理", "添付ファイルアップロード", "申請データ", "お知らせ欄",
        ),
        ("注文登録", "顧客検索", "集計ジョブ"),
    ),
    CategorySpec(
        "ネットワーク",
        "インフラ",
        ("社員", "管理者", "開発者"),
        (
            "社内WiFi", "VPN接続", "拠点間ネットワーク", "社内DNS", "インターネット接続",
            "クラウド環境接続", "会議室LAN", "ゲストWiFi", "ルーター", "ファイル共有通信",
        ),
        ("無線LAN", "在宅接続", "拠点VPN"),
    ),
    CategorySpec(
        "アカウント",
        "情シス",
        ("社員", "管理者"),
        (
            "ログイン", "多要素認証", "新入社員アカウント", "アカウントロック", "退職者アカウント",
            "メールエイリアス", "共有アカウント", "表示名", "チャットツールアカウント", "不審ログイン通知",
        ),
        ("サインイン", "パスワード再設定", "認証コード"),
    ),
    CategorySpec(
        "データ連携",
        "開発",
        ("管理者", "開発者", "社員"),
        (
            "外部システム連携", "CSV取込", "マスタ反映", "API連携", "夜間連携",
            "商品コード取込", "認証トークン", "連携履歴画面", "テスト環境マスタ", "レスポンス時間",
        ),
        ("SFTP取込", "Webhook連携", "データ同期"),
    ),
    CategorySpec(
        "端末",
        "情シス",
        ("社員", "管理者"),
        (
            "会社PC", "プリンタ", "Webカメラ", "貸与PC", "全社配布アプリ",
            "キーボード", "セキュリティ警告", "モニター接続", "標準ブラウザ", "端末交換",
        ),
        ("ノートPC", "外部ディスプレイ", "USB機器"),
    ),
    CategorySpec(
        "その他・対象外",
        "総務",
        ("社員", "管理者"),
        (
            "オフィス空調", "会議室の照明", "事務机", "社員証ケース", "給湯室",
            "郵便物", "備品在庫", "座席表", "社内掲示板", "清掃依頼",
        ),
        ("エアコン", "会議室設備", "オフィス照明"),
    ),
]


TRAIN_ISSUES = {
    "Low": (
        "の操作方法を確認したいです", "の入力項目の意味を教えてください", "の表示名を変更したいです",
        "の設定手順を知りたいです", "の過去データを確認したいです", "の通知設定を変更したいです",
        "の画面の場所が分かりません", "の説明文を修正したいです", "の利用方法を共有してください",
        "について急ぎではない質問があります",
    ),
    "Middle": (
        "が一覧に反映されません", "を保存するとエラーになります", "の検索結果が表示されません",
        "の通知が担当者へ届きません", "を開くと白い画面になります", "の処理が遅く業務に支障があります",
        "の一部データが更新されていません", "の状態を変更できません", "で特定条件だけ処理が失敗します",
        "で部署内の複数名に影響が出ています",
    ),
    "High": (
        "が全社で利用できず業務が止まっています", "の締め処理が完了せず期限に間に合いません",
        "の障害で複数部署の作業が停止しています", "で重大なセキュリティリスクが発生しています",
        "の夜間処理が失敗し本日の業務に影響しています", "の重要データが欠落しています",
        "が大量に失敗して顧客対応に影響しています", "で業務継続に必要な処理を実行できません",
        "の不具合が本番環境で継続発生しています", "を本日中に復旧しないと月次処理に影響します",
    ),
}


HOLDOUT_ISSUES = {
    "Low": (
        "について手順書の参照先だけ知りたいです",
        "の表示を自分向けに調整する方法はありますか",
        "に関する一般的な使い方を確認させてください",
    ),
    "Middle": (
        "を実行しても反応がなく、担当業務を進められません",
        "が昨日から不安定で、チームの作業が遅れています",
        "で入力内容が消えるため調査をお願いします",
    ),
    "High": (
        "が停止し、全拠点で通常業務を開始できません",
        "で情報漏えいにつながる恐れがあり、緊急対応が必要です",
        "の障害が顧客対応を全面的に止めています",
    ),
}


PRIORITY_METADATA = {
    "Low": {
        "impact_scopes": ("個人", "個人", "部署"),
        "channels": ("問い合わせフォーム", "Slack", "メール"),
    },
    "Middle": {
        "impact_scopes": ("個人", "部署", "複数部署"),
        "channels": ("Slack", "問い合わせフォーム", "メール"),
    },
    "High": {
        "impact_scopes": ("全社", "複数部署", "全社"),
        "channels": ("電話", "メール", "Slack"),
    },
}


def render_training_text(object_name: str, issue: str, variant: int) -> str:
    if variant == 0:
        return f"{object_name}{issue}"
    if variant == 1:
        return f"{object_name}について問い合わせます。現在、{object_name}{issue}"
    raise ValueError(f"Unsupported training text variant: {variant}")


def build_training_records() -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    ticket_number = 1
    for spec in CATEGORY_SPECS:
        for priority in PRIORITIES:
            issues = TRAIN_ISSUES[priority]
            if len(issues) != TRAIN_CASES_PER_PRIORITY:
                raise ValueError(f"{priority} must have {TRAIN_CASES_PER_PRIORITY} issues.")
            metadata = PRIORITY_METADATA[priority]
            for case_index, issue in enumerate(issues):
                template_group = f"{spec.category}-{priority}-{case_index:02d}"
                for variant in range(TRAIN_VARIANTS_PER_CASE):
                    object_name = (
                        spec.objects[case_index]
                        if variant == 0
                        else spec.holdout_objects[case_index % len(spec.holdout_objects)]
                    )
                    record = make_record(
                        ticket_id=f"TKT-{ticket_number:04d}",
                        inquiry_text=render_training_text(object_name, issue, variant),
                        spec=spec,
                        priority=priority,
                        case_index=case_index,
                        template_group=template_group,
                        data_origin="synthetic_train",
                        metadata=metadata,
                    )
                    records.append(record)
                    ticket_number += 1
    return records


def build_holdout_records() -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    ticket_number = 1
    for spec in CATEGORY_SPECS:
        for priority in PRIORITIES:
            issues = HOLDOUT_ISSUES[priority]
            if len(issues) != HOLDOUT_CASES_PER_PRIORITY:
                raise ValueError(f"{priority} must have {HOLDOUT_CASES_PER_PRIORITY} holdout issues.")
            metadata = PRIORITY_METADATA[priority]
            for case_index, issue in enumerate(issues):
                records.append(
                    make_record(
                        ticket_id=f"HLD-{ticket_number:04d}",
                        inquiry_text=f"{spec.holdout_objects[case_index]}{issue}",
                        spec=spec,
                        priority=priority,
                        case_index=case_index,
                        template_group=f"holdout-{spec.category}-{priority}-{case_index:02d}",
                        data_origin="synthetic_holdout",
                        metadata=metadata,
                    )
                )
                ticket_number += 1
    return records


def make_record(
    *,
    ticket_id: str,
    inquiry_text: str,
    spec: CategorySpec,
    priority: str,
    case_index: int,
    template_group: str,
    data_origin: str,
    metadata: dict[str, tuple[str, ...]],
) -> dict[str, str]:
    return {
        "ticket_id": ticket_id,
        "inquiry_text": inquiry_text,
        "label_category": spec.category,
        "label_priority": priority,
        "label_department": spec.department,
        "impact_scope": metadata["impact_scopes"][case_index % len(metadata["impact_scopes"])],
        "requester_role": spec.requester_roles[case_index % len(spec.requester_roles)],
        "channel": metadata["channels"][case_index % len(metadata["channels"])],
        "template_group": template_group,
        "data_origin": data_origin,
    }


def write_records(path: Path, records: list[dict[str, str]]) -> None:
    if not records:
        raise ValueError(f"No records to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    training_records = build_training_records()
    holdout_records = build_holdout_records()
    expected_training_count = (
        len(CATEGORY_SPECS)
        * len(PRIORITIES)
        * TRAIN_CASES_PER_PRIORITY
        * TRAIN_VARIANTS_PER_CASE
    )
    expected_holdout_count = (
        len(CATEGORY_SPECS) * len(PRIORITIES) * HOLDOUT_CASES_PER_PRIORITY
    )
    if len(training_records) != expected_training_count:
        raise ValueError(
            f"Expected {expected_training_count} training records, got {len(training_records)}."
        )
    if len(holdout_records) != expected_holdout_count:
        raise ValueError(
            f"Expected {expected_holdout_count} holdout records, got {len(holdout_records)}."
        )
    training_texts = {record["inquiry_text"] for record in training_records}
    holdout_texts = {record["inquiry_text"] for record in holdout_records}
    overlap = training_texts & holdout_texts
    if overlap:
        raise ValueError(f"Training and holdout texts overlap: {sorted(overlap)}")

    write_records(TRAIN_DATA_PATH, training_records)
    write_records(HOLDOUT_DATA_PATH, holdout_records)
    print(f"wrote {len(training_records)} records to {TRAIN_DATA_PATH}")
    print(f"wrote {len(holdout_records)} records to {HOLDOUT_DATA_PATH}")


if __name__ == "__main__":
    main()
