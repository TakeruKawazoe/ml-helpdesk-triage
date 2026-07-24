from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from notion_sync import (  # noqa: E402
    NotionConfig,
    NotionConfigurationError,
    NotionSyncError,
    NotionSyncService,
)


DATABASE_ID = "3a5d3d7e-828c-80d5-9124-c41c212013e9"
DATA_SOURCE_ID = "3a5d3d7e-828c-80f4-8ee3-000c4814880d"
PAGE_ID = "5a5d3d7e-828c-80f4-8ee3-000c4814880d"


class FakeNotionClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, object] | None]] = []

    def request(
        self,
        method: str,
        path: str,
        payload: dict[str, object] | None = None,
    ) -> dict[str, object]:
        self.calls.append((method, path, payload))
        if method == "GET" and path.startswith("/databases/"):
            return {"data_sources": [{"id": DATA_SOURCE_ID, "name": "テーブルビュー"}]}
        if method == "GET" and path.startswith("/data_sources/"):
            return {"properties": {"名前": {"type": "title", "title": {}}}}
        if method == "PATCH" and path.startswith("/data_sources/"):
            return {"id": DATA_SOURCE_ID}
        if method == "POST" and path == "/pages":
            return {"id": PAGE_ID}
        if method == "PATCH" and path.startswith("/pages/"):
            return {"id": PAGE_ID}
        raise AssertionError(f"Unexpected request: {method} {path}")


class NotionSyncTest(unittest.TestCase):
    def setUp(self) -> None:
        self.client = FakeNotionClient()
        self.service = NotionSyncService(
            NotionConfig(api_token="secret", database_id=DATABASE_ID, data_source_id=""),
            client=self.client,
        )
        self.record = {
            "prediction_id": "prediction-id",
            "created_at": "2026-07-22T10:00:00+09:00",
            "text": "全社でログインできず業務が停止しています",
            "impact_scope": "全社",
            "requester_role": "管理者",
            "channel": "Slack",
            "predicted_category": "アカウント",
            "predicted_priority": "High",
            "predicted_department": "情シス",
            "category_confidence": "0.800000",
            "priority_confidence": "0.900000",
            "department_confidence": "0.700000",
            "routing_status": "自動振り分け",
            "review_reasons": "",
            "corrected_category": "アカウント",
            "corrected_priority": "High",
            "corrected_department": "情シス",
            "note": "確認済み",
            "reviewer_id": "reviewer-01",
            "feedback_saved_at": "2026-07-22T10:05:00+09:00",
            "notion_page_id": PAGE_ID,
        }

    def test_create_prediction_adds_schema_and_page(self) -> None:
        page_id = self.service.create_prediction(self.record)

        self.assertEqual(page_id, PAGE_ID)
        schema_payload = self.client.calls[2][2]
        self.assertIn("問い合わせ文", schema_payload["properties"])
        create_payload = self.client.calls[3][2]
        self.assertEqual(
            create_payload["parent"]["data_source_id"],
            DATA_SOURCE_ID,
        )
        self.assertEqual(
            create_payload["properties"]["優先度"]["select"]["name"],
            "High",
        )
        self.assertEqual(
            create_payload["properties"]["カテゴリ信頼度"]["number"],
            0.8,
        )

    def test_feedback_updates_created_page(self) -> None:
        self.service.create_prediction(self.record)

        self.service.update_feedback(PAGE_ID, self.record)

        method, path, payload = self.client.calls[-1]
        self.assertEqual(method, "PATCH")
        self.assertEqual(path, f"/pages/{PAGE_ID}")
        self.assertEqual(
            payload["properties"]["修正担当部署"]["select"]["name"],
            "情シス",
        )

    def test_partial_environment_configuration_is_rejected(self) -> None:
        with patch.dict(os.environ, {"NOTION_API_TOKEN": "secret"}, clear=True):
            with self.assertRaises(NotionConfigurationError):
                NotionConfig.from_environment()

    def test_wrong_existing_property_type_is_rejected(self) -> None:
        class WrongSchemaClient(FakeNotionClient):
            def request(self, method, path, payload=None):
                if method == "GET" and path.startswith("/data_sources/"):
                    return {
                        "properties": {
                            "名前": {"type": "title", "title": {}},
                            "優先度": {"type": "rich_text", "rich_text": {}},
                        }
                    }
                return super().request(method, path, payload)

        service = NotionSyncService(
            NotionConfig(
                api_token="secret",
                database_id="",
                data_source_id=DATA_SOURCE_ID,
            ),
            client=WrongSchemaClient(),
        )

        with self.assertRaises(NotionSyncError):
            service.ensure_schema()


if __name__ == "__main__":
    unittest.main()
