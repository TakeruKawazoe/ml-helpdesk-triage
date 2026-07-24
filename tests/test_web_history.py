from __future__ import annotations

import csv
import io
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as Namespace


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

import web_app  # noqa: E402


def SimpleNamespace(*, target: str, label: str, confidence: float) -> Namespace:
    return Namespace(
        target=target,
        label=label,
        confidence=confidence,
        threshold=0.5,
        requires_review=False,
    )


class WebHistoryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.original_storage_dir = web_app.STORAGE_DIR
        self.original_feedback_path = web_app.FEEDBACK_PATH
        self.original_history_event_path = web_app.HISTORY_EVENT_PATH
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.addCleanup(self.restore_history_paths)

        storage_dir = Path(self.temp_dir.name) / "storage"
        web_app.STORAGE_DIR = storage_dir
        web_app.FEEDBACK_PATH = storage_dir / "prediction_feedback.csv"
        web_app.HISTORY_EVENT_PATH = storage_dir / "history_events.csv"

    def restore_history_paths(self) -> None:
        web_app.STORAGE_DIR = self.original_storage_dir
        web_app.FEEDBACK_PATH = self.original_feedback_path
        web_app.HISTORY_EVENT_PATH = self.original_history_event_path

    def test_prediction_history_and_feedback_round_trip(self) -> None:
        prediction = web_app.save_prediction(
            {
                "text": "VPN接続後に社内システムへアクセスできません",
                "impact_scope": "個人",
                "requester_role": "社員",
                "channel": "問い合わせフォーム",
            },
            [
                SimpleNamespace(target="category", label="ネットワーク", confidence=0.8),
                SimpleNamespace(target="priority", label="Middle", confidence=0.7),
                SimpleNamespace(target="department", label="インフラ", confidence=0.9),
            ],
        )

        updated = web_app.save_feedback(
            {
                "prediction_id": prediction["prediction_id"],
                "corrected_category": "ネットワーク",
                "corrected_priority": "Middle",
                "corrected_department": "インフラ",
                "note": "VPN問い合わせとして修正",
                "reviewer_id": "reviewer-01",
            }
        )
        history = web_app.read_history(limit=10)

        self.assertEqual(updated["note"], "VPN問い合わせとして修正")
        self.assertEqual(history[0]["prediction_id"], prediction["prediction_id"])
        self.assertEqual(history[0]["corrected_category"], "ネットワーク")
        self.assertEqual(history[0]["corrected_priority"], "Middle")
        self.assertEqual(history[0]["corrected_department"], "インフラ")
        self.assertEqual(history[0]["note"], "VPN問い合わせとして修正")
        self.assertTrue(history[0]["feedback_saved_at"])
        self.assertEqual(history[0]["category_confidence"], "0.800000")
        self.assertEqual(history[0]["routing_status"], "自動振り分け")
        self.assertEqual(history[0]["reviewer_id"], "reviewer-01")

    def test_feedback_update_preserves_other_history_rows(self) -> None:
        first = web_app.save_prediction(
            {
                "text": "請求書の一括出力が失敗します",
                "impact_scope": "部署",
                "requester_role": "経理担当",
                "channel": "メール",
            },
            [
                SimpleNamespace(target="category", label="請求", confidence=0.8),
                SimpleNamespace(target="priority", label="Middle", confidence=0.7),
                SimpleNamespace(target="department", label="経理", confidence=0.9),
            ],
        )
        second = web_app.save_prediction(
            {
                "text": "API連携で取引先データが欠落しています",
                "impact_scope": "複数部署",
                "requester_role": "管理者",
                "channel": "メール",
            },
            [
                SimpleNamespace(target="category", label="データ連携", confidence=0.8),
                SimpleNamespace(target="priority", label="High", confidence=0.7),
                SimpleNamespace(target="department", label="開発", confidence=0.9),
            ],
        )

        web_app.save_feedback(
            {
                "prediction_id": first["prediction_id"],
                "corrected_category": "請求",
                "corrected_priority": "Middle",
                "corrected_department": "経理",
                "note": "",
                "reviewer_id": "reviewer-01",
            }
        )
        history_ids = {row["prediction_id"] for row in web_app.read_history(limit=10)}

        self.assertEqual(history_ids, {first["prediction_id"], second["prediction_id"]})

    def test_notion_sync_result_is_saved_to_history(self) -> None:
        prediction = web_app.save_prediction(
            {
                "text": "全社でログインできません",
                "impact_scope": "全社",
                "requester_role": "管理者",
                "channel": "Slack",
            },
            [
                SimpleNamespace(target="category", label="アカウント", confidence=0.8),
                SimpleNamespace(target="priority", label="High", confidence=0.9),
                SimpleNamespace(target="department", label="情シス", confidence=0.7),
            ],
        )

        updated = web_app.save_notion_sync_result(
            prediction["prediction_id"],
            web_app.NotionSyncResult(status="synced", page_id="page-id"),
        )

        self.assertEqual(updated["notion_page_id"], "page-id")
        self.assertEqual(updated["notion_sync_status"], "synced")
        self.assertTrue(updated["notion_synced_at"])

    def test_slack_notification_result_is_saved_to_history(self) -> None:
        prediction = web_app.save_prediction(
            {
                "text": "全社でログインできません",
                "impact_scope": "全社",
                "requester_role": "管理者",
                "channel": "Slack",
            },
            [
                SimpleNamespace(target="category", label="アカウント", confidence=0.8),
                SimpleNamespace(target="priority", label="High", confidence=0.9),
                SimpleNamespace(target="department", label="情シス", confidence=0.7),
            ],
        )

        updated = web_app.save_slack_notification_result(
            prediction["prediction_id"],
            web_app.SlackNotificationResult(status="sent", message_ts="123.456"),
        )

        self.assertEqual(updated["slack_notification_status"], "sent")
        self.assertEqual(updated["slack_message_ts"], "123.456")
        self.assertTrue(updated["slack_notified_at"])

    def test_existing_history_file_is_migrated(self) -> None:
        web_app.STORAGE_DIR.mkdir(parents=True)
        web_app.FEEDBACK_PATH.write_text(
            "prediction_id,created_at,text\nold-id,2026-07-22T10:00:00+09:00,問い合わせ\n",
            encoding="utf-8-sig",
        )

        history = web_app.read_history(limit=10)

        self.assertEqual(history[0]["prediction_id"], "old-id")
        self.assertEqual(history[0]["notion_sync_status"], "")
        self.assertEqual(history[0]["slack_notification_status"], "")
        self.assertEqual(history[0]["routing_status"], "")
        self.assertEqual(history[0]["deleted_at"], "")

    def test_history_query_uses_corrected_labels_and_pagination(self) -> None:
        prediction = web_app.save_prediction(
            {
                "text": "VPN接続を確認してください",
                "impact_scope": "個人",
                "requester_role": "社員",
                "channel": "問い合わせフォーム",
            },
            [
                SimpleNamespace(target="category", label="その他・対象外", confidence=0.6),
                SimpleNamespace(target="priority", label="Low", confidence=0.7),
                SimpleNamespace(target="department", label="情シス", confidence=0.8),
            ],
        )
        web_app.save_feedback(
            {
                "prediction_id": prediction["prediction_id"],
                "corrected_category": "ネットワーク",
                "corrected_priority": "Middle",
                "corrected_department": "インフラ",
                "note": "VPN障害として確認",
                "reviewer_id": "reviewer-01",
            }
        )

        result = web_app.query_history(
            web_app.HistoryQuery(
                keyword="vpn",
                category="ネットワーク",
                priority="Middle",
                department="インフラ",
                feedback_status="with_feedback",
                page=1,
                page_size=1,
            )
        )

        self.assertEqual(result["total"], 1)
        self.assertEqual(result["total_pages"], 1)
        self.assertEqual(result["items"][0]["prediction_id"], prediction["prediction_id"])

    def test_history_can_be_logically_deleted_and_restored(self) -> None:
        prediction = web_app.save_prediction(
            {
                "text": "削除対象の問い合わせ",
                "impact_scope": "個人",
                "requester_role": "社員",
                "channel": "メール",
            },
            [
                SimpleNamespace(target="category", label="端末", confidence=0.8),
                SimpleNamespace(target="priority", label="Low", confidence=0.8),
                SimpleNamespace(target="department", label="情シス", confidence=0.8),
            ],
        )

        deleted = web_app.delete_history_record(
            prediction["prediction_id"],
            "operator-01",
            "テストデータのため",
        )

        self.assertTrue(deleted["deleted_at"])
        self.assertEqual(web_app.read_history(limit=10), [])
        deleted_result = web_app.query_history(
            web_app.HistoryQuery(deleted_status="deleted")
        )
        self.assertEqual(deleted_result["total"], 1)
        with self.assertRaises(ValueError):
            web_app.save_feedback(
                {
                    "prediction_id": prediction["prediction_id"],
                    "corrected_category": "端末",
                    "corrected_priority": "Low",
                    "corrected_department": "情シス",
                    "note": "削除済み",
                    "reviewer_id": "reviewer-01",
                }
            )

        restored = web_app.restore_history_record(
            prediction["prediction_id"],
            "operator-02",
            "誤削除だったため",
        )

        self.assertEqual(restored["deleted_at"], "")
        self.assertEqual(len(web_app.read_history(limit=10)), 1)
        with web_app.HISTORY_EVENT_PATH.open(
            "r", encoding="utf-8-sig", newline=""
        ) as csv_file:
            events = list(csv.DictReader(csv_file))
        self.assertEqual([event["event_type"] for event in events], ["deleted", "restored"])

    def test_history_export_uses_filters_and_prevents_formula_injection(self) -> None:
        prediction = web_app.save_prediction(
            {
                "text": "=SUM(1,1)",
                "impact_scope": "部署",
                "requester_role": "管理者",
                "channel": "Slack",
            },
            [
                SimpleNamespace(target="category", label="請求", confidence=0.8),
                SimpleNamespace(target="priority", label="High", confidence=0.8),
                SimpleNamespace(target="department", label="経理", confidence=0.8),
            ],
        )

        content = web_app.build_history_export(
            web_app.HistoryQuery(priority="High")
        )
        rows = list(csv.DictReader(io.StringIO(content.decode("utf-8-sig"))))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["予測ID"], prediction["prediction_id"])
        self.assertEqual(rows[0]["問い合わせ文"], "'=SUM(1,1)")

    def test_history_query_rejects_invalid_date_range(self) -> None:
        with self.assertRaises(ValueError):
            web_app.parse_history_query(
                {
                    "date_from": ["2026-07-25"],
                    "date_to": ["2026-07-24"],
                }
            )

    def test_low_confidence_prediction_is_routed_for_review(self) -> None:
        result = web_app.routing_result(
            [
                Namespace(
                    target="category",
                    label="その他・対象外",
                    confidence=0.4,
                    threshold=0.6,
                    requires_review=True,
                )
            ]
        )

        self.assertTrue(result["review_required"])
        self.assertEqual(result["status"], "要確認（一次受付）")
        self.assertIn("category", result["reasons"][0])

    def test_validate_string_field_rejects_non_string_value(self) -> None:
        with self.assertRaises(ValueError):
            web_app.validate_string_field("note", 123)

    def test_parse_history_limit_rejects_out_of_range_value(self) -> None:
        with self.assertRaises(ValueError):
            web_app.parse_history_limit({"limit": ["51"]})


if __name__ == "__main__":
    unittest.main()
