from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from threading import RLock

import joblib

from modeling import MODEL_OUTPUT_DIR
from train_baseline import build_text


TARGETS = ["category", "priority", "department"]
IMPACT_SCOPE_CHOICES = ["個人", "部署", "複数部署", "全社"]
REQUESTER_ROLE_CHOICES = ["社員", "管理者", "経理担当", "開発者"]
CHANNEL_CHOICES = ["Slack", "問い合わせフォーム", "メール", "電話"]
MODEL_DIR = MODEL_OUTPUT_DIR
MODEL_ACCESS_LOCK = RLock()
MODEL_CACHE: dict[tuple[str, int, float], LoadedTarget] = {}


@dataclass(frozen=True)
class LoadedTarget:
    pipeline: object
    classes: list[str]
    confidence_threshold: float


@dataclass(frozen=True)
class PredictionResult:
    target: str
    label: str
    confidence: float
    threshold: float
    requires_review: bool
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


def load_manifest(model_dir: Path = MODEL_DIR) -> dict[str, object]:
    manifest_path = model_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Improved model manifest is missing: {manifest_path}. "
            "Run src/train_improved.py before prediction."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["format"] != "sklearn-joblib-v1":
        raise ValueError(f"Unsupported model format: {manifest['format']}")
    return manifest


def load_target(target: str, model_dir: Path = MODEL_DIR) -> LoadedTarget:
    with MODEL_ACCESS_LOCK:
        manifest = load_manifest(model_dir)
        target_metadata = manifest["targets"][target]
        model_path = model_dir / target_metadata["model_file"]
        if not model_path.exists():
            raise FileNotFoundError(f"Model file is missing: {model_path}")
        confidence_threshold = float(target_metadata["confidence_threshold"])
        resolved_model_path = str(model_path.resolve())
        cache_key = (
            resolved_model_path,
            model_path.stat().st_mtime_ns,
            confidence_threshold,
        )
        cached = MODEL_CACHE.get(cache_key)
        if cached is not None:
            return cached

        pipeline = joblib.load(model_path)
        classes = [str(value) for value in pipeline.classes_]
        expected_classes = [str(value) for value in target_metadata["classes"]]
        if classes != expected_classes:
            raise ValueError(f"Model classes do not match the manifest for {target}.")
        loaded = LoadedTarget(
            pipeline=pipeline,
            classes=classes,
            confidence_threshold=confidence_threshold,
        )
        stale_keys = [
            key
            for key in MODEL_CACHE
            if key[0] == resolved_model_path and key != cache_key
        ]
        for stale_key in stale_keys:
            del MODEL_CACHE[stale_key]
        MODEL_CACHE[cache_key] = loaded
        return loaded


def clear_model_cache() -> None:
    with MODEL_ACCESS_LOCK:
        MODEL_CACHE.clear()


def predict_target(
    target: str,
    model_text: str,
    top_k: int,
    model_dir: Path = MODEL_DIR,
) -> PredictionResult:
    if target not in TARGETS:
        raise ValueError(f"Unknown prediction target: {target}")
    if top_k < 1:
        raise ValueError("top_k must be greater than or equal to 1.")

    loaded = load_target(target, model_dir=model_dir)
    probabilities = loaded.pipeline.predict_proba([model_text])[0]
    ranking_indexes = probabilities.argsort()[::-1][:top_k]
    ranking = [
        {
            "label": loaded.classes[index],
            "confidence": round(float(probabilities[index]), 4),
        }
        for index in ranking_indexes
    ]
    best_index = int(ranking_indexes[0])
    confidence = float(probabilities[best_index])
    return PredictionResult(
        target=target,
        label=loaded.classes[best_index],
        confidence=round(confidence, 4),
        threshold=round(loaded.confidence_threshold, 4),
        requires_review=confidence < loaded.confidence_threshold,
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
                    "routing_status": (
                        "要確認（一次受付）"
                        if any(result.requires_review for result in results)
                        else "自動振り分け"
                    ),
                    "predictions": [result.__dict__ for result in results],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    print(
        "振り分け: "
        + (
            "要確認（一次受付）"
            if any(result.requires_review for result in results)
            else "自動振り分け"
        )
    )
    for result in results:
        review_label = " / 要確認" if result.requires_review else ""
        print(
            f"- {result.target}: {result.label} ({result.confidence:.4f})"
            f" / 基準={result.threshold:.4f}{review_label}"
        )
        for rank in result.ranking:
            print(f"  - {rank['label']}: {rank['confidence']:.4f}")


if __name__ == "__main__":
    main()
