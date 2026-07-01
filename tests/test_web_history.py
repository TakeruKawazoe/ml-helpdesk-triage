from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

import web_app  # noqa: E402


class WebHistoryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.original_storage_dir = web_app.STORAGE_DIR
        self.original_feedback_path = web_app.FEEDBACK_PATH
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.addCleanup(self.restore_history_paths)

        storage_dir = Path(self.temp_dir.name) / "storage"
        web_app.STORAGE_DIR = storage_dir
        web_app.FEEDBACK_PATH = storage_dir / "prediction_feedback.csv"

    def restore_history_paths(self) -> None:
        web_app.STORAGE_DIR = self.original_storage_dir
        web_app.FEEDBACK_PATH = self.original_feedback_path

    def test_prediction_history_and_feedback_round_trip(self) -> None:
        prediction = web_app.save_prediction(
            {
                "text": "VPN接続後に社内システムへアクセスできません",
                "impact_scope": "個人",
                "requester_role": "社員",
                "channel": "問い合わせフォーム",
            },
            [
                SimpleNamespace(target="category", label="ネットワーク"),
                SimpleNamespace(target="priority", label="Middle"),
                SimpleNamespace(target="department", label="インフラ"),
            ],
        )

        updated = web_app.save_feedback(
            {
                "prediction_id": prediction["prediction_id"],
                "corrected_category": "ネットワーク",
                "corrected_priority": "Middle",
                "corrected_department": "インフラ",
                "note": "VPN問い合わせとして修正",
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

    def test_feedback_update_preserves_other_history_rows(self) -> None:
        first = web_app.save_prediction(
            {
                "text": "請求書の一括出力が失敗します",
                "impact_scope": "部署",
                "requester_role": "経理担当",
                "channel": "メール",
            },
            [
                SimpleNamespace(target="category", label="請求"),
                SimpleNamespace(target="priority", label="Middle"),
                SimpleNamespace(target="department", label="経理"),
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
                SimpleNamespace(target="category", label="データ連携"),
                SimpleNamespace(target="priority", label="High"),
                SimpleNamespace(target="department", label="開発"),
            ],
        )

        web_app.save_feedback(
            {
                "prediction_id": first["prediction_id"],
                "corrected_category": "請求",
                "corrected_priority": "Middle",
                "corrected_department": "経理",
                "note": "",
            }
        )
        history_ids = {row["prediction_id"] for row in web_app.read_history(limit=10)}

        self.assertEqual(history_ids, {first["prediction_id"], second["prediction_id"]})

    def test_validate_string_field_rejects_non_string_value(self) -> None:
        with self.assertRaises(ValueError):
            web_app.validate_string_field("note", 123)

    def test_parse_history_limit_rejects_out_of_range_value(self) -> None:
        with self.assertRaises(ValueError):
            web_app.parse_history_limit({"limit": ["51"]})


if __name__ == "__main__":
    unittest.main()
