from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


SLACK_POST_MESSAGE_URL = "https://slack.com/api/chat.postMessage"
SLACK_TIMEOUT_SECONDS = 10
NOTIFY_PRIORITIES = {"High", "Middle"}
DEPARTMENT_MENTION_ENV = {
    "総務": "SLACK_MENTION_SOMU",
    "経理": "SLACK_MENTION_KEIRI",
    "情シス": "SLACK_MENTION_JOSYS",
    "開発": "SLACK_MENTION_DEVELOPMENT",
    "インフラ": "SLACK_MENTION_INFRASTRUCTURE",
}


class SlackConfigurationError(ValueError):
    pass


class SlackNotificationError(RuntimeError):
    pass


@dataclass(frozen=True)
class SlackConfig:
    bot_token: str
    channel_id: str
    department_mentions: dict[str, str]

    @classmethod
    def from_environment(cls) -> SlackConfig | None:
        variable_names = [
            "SLACK_BOT_TOKEN",
            "SLACK_CHANNEL_ID",
            *DEPARTMENT_MENTION_ENV.values(),
        ]
        values = {
            name: os.environ.get(name, "").strip()
            for name in variable_names
        }
        if not any(values.values()):
            return None

        missing = [name for name, value in values.items() if not value]
        if missing:
            raise SlackConfigurationError(
                f"Missing Slack environment variables: {', '.join(missing)}"
            )
        if not values["SLACK_BOT_TOKEN"].startswith("xoxb-"):
            raise SlackConfigurationError(
                "SLACK_BOT_TOKEN must be a bot token starting with xoxb-."
            )
        if not values["SLACK_CHANNEL_ID"].startswith(("C", "G")):
            raise SlackConfigurationError("SLACK_CHANNEL_ID must be a Slack channel ID.")

        department_mentions = {
            department: values[environment_name]
            for department, environment_name in DEPARTMENT_MENTION_ENV.items()
        }
        invalid_mentions = [
            department
            for department, member_id in department_mentions.items()
            if not member_id.startswith("U")
        ]
        if invalid_mentions:
            raise SlackConfigurationError(
                "Slack member IDs must start with U: " + ", ".join(invalid_mentions)
            )

        return cls(
            bot_token=values["SLACK_BOT_TOKEN"],
            channel_id=values["SLACK_CHANNEL_ID"],
            department_mentions=department_mentions,
        )


@dataclass(frozen=True)
class SlackNotificationResult:
    status: str
    message_ts: str = ""
    error: str = ""


class SlackClient:
    def __init__(
        self,
        bot_token: str,
        timeout_seconds: int = SLACK_TIMEOUT_SECONDS,
        opener: Callable[..., object] = urlopen,
    ) -> None:
        self.bot_token = bot_token
        self.timeout_seconds = timeout_seconds
        self.opener = opener

    def post_message(self, channel_id: str, text: str) -> str:
        request = Request(
            url=SLACK_POST_MESSAGE_URL,
            data=json.dumps(
                {
                    "channel": channel_id,
                    "text": text,
                    "unfurl_links": False,
                },
                ensure_ascii=False,
            ).encode("utf-8"),
            method="POST",
            headers={
                "Authorization": f"Bearer {self.bot_token}",
                "Content-Type": "application/json; charset=utf-8",
            },
        )

        try:
            with self.opener(request, timeout=self.timeout_seconds) as response:
                response_body = response.read().decode("utf-8")
        except HTTPError as error:
            error_body = error.read().decode("utf-8", errors="replace")
            raise SlackNotificationError(
                f"Slack API returned HTTP {error.code}: {error_body[:500]}"
            ) from error
        except URLError as error:
            raise SlackNotificationError(
                f"Slack API connection failed: {error.reason}"
            ) from error
        except TimeoutError as error:
            raise SlackNotificationError("Slack API request timed out.") from error

        try:
            decoded = json.loads(response_body)
        except json.JSONDecodeError as error:
            raise SlackNotificationError("Slack API returned invalid JSON.") from error
        if not isinstance(decoded, dict):
            raise SlackNotificationError(
                "Slack API returned an unexpected response format."
            )
        if decoded.get("ok") is not True:
            error_code = decoded.get("error")
            if not isinstance(error_code, str) or not error_code:
                error_code = "unknown_error"
            raise SlackNotificationError(f"Slack API rejected the message: {error_code}")

        message_ts = decoded.get("ts")
        if not isinstance(message_ts, str) or not message_ts:
            raise SlackNotificationError("Slack API did not return a message timestamp.")
        return message_ts


def notify_prediction(record: dict[str, str]) -> SlackNotificationResult:
    if record["predicted_priority"] not in NOTIFY_PRIORITIES:
        return SlackNotificationResult(status="skipped")

    config = SlackConfig.from_environment()
    if config is None:
        return SlackNotificationResult(status="disabled")

    department = record["predicted_department"]
    if department not in config.department_mentions:
        raise SlackConfigurationError(
            f"No Slack assignee is configured for: {department}"
        )

    client = SlackClient(config.bot_token)
    message_ts = client.post_message(
        config.channel_id,
        build_notification_text(record, config.department_mentions[department]),
    )
    return SlackNotificationResult(status="sent", message_ts=message_ts)


def build_notification_text(record: dict[str, str], member_id: str) -> str:
    notion_line = "Notion登録失敗（ローカル履歴を確認してください）"
    if record["notion_page_id"]:
        notion_page_id = record["notion_page_id"].replace("-", "")
        notion_line = f"https://www.notion.so/{notion_page_id}"

    return "\n".join(
        [
            f"<@{member_id}> 新しい問い合わせです",
            f"*振り分け:* {escape_slack_text(record['routing_status'])}",
            *(
                [f"*要確認理由:* {escape_slack_text(record['review_reasons'])}"]
                if record["review_reasons"]
                else []
            ),
            f"*問い合わせ:* {escape_slack_text(record['text'])}",
            f"*優先度:* {escape_slack_text(record['predicted_priority'])}",
            f"*カテゴリ:* {escape_slack_text(record['predicted_category'])}",
            f"*担当部署:* {escape_slack_text(record['predicted_department'])}",
            f"*Notion:* {notion_line}",
        ]
    )


def escape_slack_text(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
