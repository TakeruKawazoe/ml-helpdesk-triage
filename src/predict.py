from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from train_baseline import (
    MODEL_DIR,
    SoftmaxLogisticRegression,
    TfidfVectorizer,
    build_text,
)


TARGETS = ["category", "priority", "department"]
IMPACT_SCOPE_CHOICES = ["個人", "部署", "複数部署", "全社"]
REQUESTER_ROLE_CHOICES = ["社員", "管理者", "経理担当", "開発者"]
CHANNEL_CHOICES = ["Slack", "問い合わせフォーム", "メール", "電話"]


@dataclass
class PredictionResult:
    target: str
    label: str
    confidence: float
    ranking: list[dict[str, object]]


def build_model_text(
    inquiry_text: str,
    impact_scope: str,
    requester_role: str,
    channel: str,
) -> str:
    row = {
        "inquiry_text": inquiry_text,
        "impact_scope": impact_scope,
        "requester_role": requester_role,
        "channel": channel,
    }
    return build_text(row)


def load_target(target: str) -> tuple[TfidfVectorizer, SoftmaxLogisticRegression, list[str]]:
    metadata_path = MODEL_DIR / f"{target}_metadata.json"
    arrays_path = MODEL_DIR / f"{target}_arrays.npz"
    if not metadata_path.exists() or not arrays_path.exists():
        raise FileNotFoundError(
            f"Model artifacts for '{target}' are missing. "
            "Run src/train_baseline.py before prediction."
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    arrays = np.load(arrays_path)

    vectorizer = TfidfVectorizer(
        vocab_=metadata["vocab"],
        idf_=arrays["idf"],
        min_n=metadata["min_n"],
        max_n=metadata["max_n"],
    )
    model = SoftmaxLogisticRegression(weights_=arrays["weights"])
    classes = metadata["classes"]
    return vectorizer, model, classes


def predict_target(target: str, model_text: str, top_k: int) -> PredictionResult:
    vectorizer, model, classes = load_target(target)
    x_values = vectorizer.transform([model_text])
    probabilities = model.predict_proba(x_values)[0]
    ranking_indexes = probabilities.argsort()[::-1][:top_k]
    ranking = [
        {
            "label": classes[index],
            "confidence": round(float(probabilities[index]), 4),
        }
        for index in ranking_indexes
    ]
    best = ranking[0]
    return PredictionResult(
        target=target,
        label=str(best["label"]),
        confidence=float(best["confidence"]),
        ranking=ranking,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict category, priority, and department for a helpdesk ticket."
    )
    parser.add_argument("--text", required=True, help="問い合わせ本文")
    parser.add_argument("--impact-scope", required=True, choices=IMPACT_SCOPE_CHOICES)
    parser.add_argument("--requester-role", required=True, choices=REQUESTER_ROLE_CHOICES)
    parser.add_argument("--channel", required=True, choices=CHANNEL_CHOICES)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--json", action="store_true", help="JSON形式で出力する")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_k < 1:
        raise ValueError("--top-k must be greater than or equal to 1.")

    model_text = build_model_text(
        inquiry_text=args.text,
        impact_scope=args.impact_scope,
        requester_role=args.requester_role,
        channel=args.channel,
    )
    results = [
        predict_target(target=target, model_text=model_text, top_k=args.top_k)
        for target in TARGETS
    ]

    if args.json:
        print(
            json.dumps(
                {
                    "input": {
                        "text": args.text,
                        "impact_scope": args.impact_scope,
                        "requester_role": args.requester_role,
                        "channel": args.channel,
                    },
                    "predictions": [result.__dict__ for result in results],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    print("予測結果")
    for result in results:
        print(f"- {result.target}: {result.label} ({result.confidence:.4f})")
        for rank in result.ranking:
            print(f"  - {rank['label']}: {rank['confidence']:.4f}")


if __name__ == "__main__":
    main()
