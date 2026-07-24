from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from slack_notify import (  # noqa: E402
    SlackClient,
    SlackConfig,
    SlackConfigurationError,
    SlackNotificationError,
    build_notification_text,
    notify_prediction,
)


class FakeResponse:
    def __init__(self, body: dict[str, object]) -> None:
        self.body = json.dumps(body).encode("utf-8")

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        return None

    def read(self) -> bytes:
        return self.body


class SlackNotificationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.record = {
            "text": "VPN接続後も社内サーバーへアクセスできません",
            "predicted_category": "ネットワーク",
            "predicted_priority": "Middle",
            "predicted_department": "インフラ",
            "notion_page_id": "3a5d3d7e-828c-81f3-9e7e-f3c6bd72681e",
            "routing_status": "自動振り分け",
            "review_reasons": "",
        }

    def test_low_priority_is_skipped_without_configuration(self) -> None:
        record = {**self.record, "predicted_priority": "Low"}

        with patch.dict(os.environ, {}, clear=True):
            result = notify_prediction(record)

        self.assertEqual(result.status, "skipped")

    def test_middle_priority_is_disabled_when_configuration_is_absent(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            result = notify_prediction(self.record)

        self.assertEqual(result.status, "disabled")

    def test_partial_environment_configuration_is_rejected(self) -> None:
        with patch.dict(os.environ, {"SLACK_BOT_TOKEN": "xoxb-test"}, clear=True):
            with self.assertRaises(SlackConfigurationError):
                SlackConfig.from_environment()

    def test_notification_text_contains_mention_and_notion_link(self) -> None:
        message = build_notification_text(self.record, "U0000000005")

        self.assertIn("<@U0000000005>", message)
        self.assertIn("*優先度:* Middle", message)
        self.assertIn("*担当部署:* インフラ", message)
        self.assertIn("*振り分け:* 自動振り分け", message)
        self.assertIn(
            "https://www.notion.so/3a5d3d7e828c81f39e7ef3c6bd72681e",
            message,
        )

    def test_client_posts_message_and_returns_timestamp(self) -> None:
        captured_request = None

        def opener(request, timeout):
            nonlocal captured_request
            captured_request = request
            self.assertEqual(timeout, 10)
            return FakeResponse({"ok": True, "ts": "123.456"})

        client = SlackClient("xoxb-test-token", opener=opener)

        message_ts = client.post_message("C0123456789", "test message")

        self.assertEqual(message_ts, "123.456")
        self.assertEqual(
            captured_request.get_header("Authorization"),
            "Bearer xoxb-test-token",
        )
        payload = json.loads(captured_request.data.decode("utf-8"))
        self.assertEqual(payload["channel"], "C0123456789")
        self.assertEqual(payload["text"], "test message")

    def test_client_rejects_slack_api_error(self) -> None:
        client = SlackClient(
            "xoxb-test-token",
            opener=lambda request, timeout: FakeResponse(
                {"ok": False, "error": "not_in_channel"}
            ),
        )

        with self.assertRaisesRegex(SlackNotificationError, "not_in_channel"):
            client.post_message("C0123456789", "test message")

    def test_client_converts_timeout_to_notification_error(self) -> None:
        def timeout_opener(request, timeout):
            raise TimeoutError("timed out")

        client = SlackClient("xoxb-test-token", opener=timeout_opener)

        with self.assertRaisesRegex(SlackNotificationError, "timed out"):
            client.post_message("C0123456789", "test message")


if __name__ == "__main__":
    unittest.main()
