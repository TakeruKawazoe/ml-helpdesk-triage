from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

import retrain_from_feedback as retrain  # noqa: E402


FEEDBACK_FIELDS = [
    "prediction_id",
    "text",
    "impact_scope",
    "requester_role",
    "channel",
    "corrected_category",
    "corrected_priority",
    "corrected_department",
    "note",
    "reviewer_id",
    "feedback_saved_at",
    "deleted_at",
]


def feedback_row(prediction_id: str, saved_at: str = "2026-07-23T10:00:00+09:00") -> dict[str, str]:
    return {
        "prediction_id": prediction_id,
        "text": "VPNへ接続できません",
        "impact_scope": "個人",
        "requester_role": "社員",
        "channel": "問い合わせフォーム",
        "corrected_category": "ネットワーク",
        "corrected_priority": "Middle",
        "corrected_department": "インフラ",
        "note": "VPN障害として確認しました",
        "reviewer_id": "reviewer-01",
        "feedback_saved_at": saved_at,
        "deleted_at": "",
    }


class RetrainFromFeedbackTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        root = Path(self.temp_dir.name)
        self.feedback_path = root / "prediction_feedback.csv"
        self.state_path = root / "retraining_state.json"
        self.status_path = root / "retraining_status.json"
        self.original_paths = (
            retrain.FEEDBACK_PATH,
            retrain.STATE_PATH,
            retrain.STATUS_PATH,
        )
        retrain.FEEDBACK_PATH = self.feedback_path
        retrain.STATE_PATH = self.state_path
        retrain.STATUS_PATH = self.status_path
        self.addCleanup(self.restore_paths)
        self.addCleanup(self.release_retraining_lock)

    def restore_paths(self) -> None:
        (
            retrain.FEEDBACK_PATH,
            retrain.STATE_PATH,
            retrain.STATUS_PATH,
        ) = self.original_paths

    def release_retraining_lock(self) -> None:
        if retrain.RETRAINING_LOCK.locked():
            retrain.RETRAINING_LOCK.release()

    def write_feedback(self, rows: list[dict[str, str]]) -> None:
        with self.feedback_path.open("w", encoding="utf-8-sig", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=FEEDBACK_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    def test_load_complete_feedback_ignores_unreviewed_rows(self) -> None:
        complete = feedback_row("complete")
        incomplete = feedback_row("incomplete")
        incomplete["corrected_priority"] = ""
        self.write_feedback([complete, incomplete])

        rows = retrain.load_complete_feedback()

        self.assertEqual([row["prediction_id"] for row in rows], ["complete"])

    def test_load_complete_feedback_ignores_deleted_rows(self) -> None:
        active = feedback_row("active")
        deleted = feedback_row("deleted")
        deleted["deleted_at"] = "2026-07-24T12:00:00+09:00"
        self.write_feedback([active, deleted])

        rows = retrain.load_complete_feedback()

        self.assertEqual([row["prediction_id"] for row in rows], ["active"])

    def test_changed_feedback_becomes_pending_again(self) -> None:
        row = feedback_row("ticket-1")
        processed = {"ticket-1": retrain.feedback_version(row)}
        self.assertEqual(retrain.pending_feedback_rows([row], processed), [])

        row["corrected_department"] = "情シス"

        self.assertEqual(
            [item["prediction_id"] for item in retrain.pending_feedback_rows([row], processed)],
            ["ticket-1"],
        )

    def test_start_retraining_waits_until_threshold(self) -> None:
        self.write_feedback([feedback_row(f"ticket-{index}") for index in range(5)])

        status = retrain.start_retraining_if_ready()

        self.assertEqual(status.status, "waiting")
        self.assertEqual(status.complete_feedback_count, 5)
        self.assertEqual(status.pending_feedback_count, 5)
        self.assertEqual(status.threshold, 20)
        saved_status = json.loads(self.status_path.read_text(encoding="utf-8"))
        self.assertEqual(saved_status["status"], "waiting")

    def test_start_retraining_launches_background_worker_at_threshold(self) -> None:
        self.write_feedback([feedback_row(f"ticket-{index}") for index in range(20)])

        with patch.object(retrain, "feedback_gate_reasons", return_value=[]), patch.object(
            retrain, "Thread"
        ) as thread_class:
            status = retrain.start_retraining_if_ready()

        self.assertEqual(status.status, "running")
        thread_class.assert_called_once()
        thread_class.return_value.start.assert_called_once_with()

    def test_candidate_requires_no_macro_f1_regression(self) -> None:
        metrics = {
            "category": {
                "current": {"macro_f1": 0.70, "accepted_precision": 0.95, "coverage": 0.80},
                "candidate": {"macro_f1": 0.71, "accepted_precision": 0.95, "coverage": 0.80},
            },
            "priority": {
                "current": {
                    "macro_f1": 0.90,
                    "accepted_precision": 0.95,
                    "coverage": 0.80,
                    "high_recall": 1.0,
                    "high_false_negatives": 0,
                },
                "candidate": {
                    "macro_f1": 0.90,
                    "accepted_precision": 0.95,
                    "coverage": 0.80,
                    "high_recall": 1.0,
                    "high_false_negatives": 0,
                },
            },
            "department": {
                "current": {
                    "macro_f1": 0.72,
                    "accepted_precision": 0.95,
                    "coverage": 0.80,
                    "misroutes": 1,
                },
                "candidate": {
                    "macro_f1": 0.71,
                    "accepted_precision": 0.95,
                    "coverage": 0.80,
                    "misroutes": 1,
                },
            },
        }

        self.assertFalse(retrain.candidate_passes(metrics))

        metrics["department"]["candidate"]["macro_f1"] = 0.72
        self.assertTrue(retrain.candidate_passes(metrics))

    def test_feedback_dataframe_uses_corrected_labels(self) -> None:
        data = retrain.feedback_to_dataframe([feedback_row("ticket-1")])

        self.assertEqual(data.iloc[0]["label_category"], "ネットワーク")
        self.assertEqual(data.iloc[0]["label_priority"], "Middle")
        self.assertEqual(data.iloc[0]["label_department"], "インフラ")
        self.assertIn("影響範囲=個人", data.iloc[0]["model_text"])

    def test_feedback_gate_rejects_single_reviewer_and_label_bias(self) -> None:
        rows = [feedback_row(f"ticket-{index}") for index in range(20)]

        reasons = retrain.feedback_gate_reasons(rows, pending_count=20)

        self.assertTrue(any("確認者" in reason for reason in reasons))
        self.assertTrue(any("category" in reason for reason in reasons))


if __name__ == "__main__":
    unittest.main()
