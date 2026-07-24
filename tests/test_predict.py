import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from predict import (  # noqa: E402
    LoadedTarget,
    clear_model_cache,
    load_target,
    predict_target,
)


class DummyPipeline:
    def __init__(self, classes, probabilities):
        self.classes_ = np.array(classes)
        self.probabilities = np.array([probabilities])

    def predict_proba(self, texts):
        if len(texts) != 1:
            raise ValueError("DummyPipeline expects one text.")
        return self.probabilities


class PredictTest(unittest.TestCase):
    def tearDown(self):
        clear_model_cache()

    def test_review_decision_uses_unrounded_confidence(self):
        loaded = LoadedTarget(
            pipeline=DummyPipeline(["Low", "High"], [0.05344, 0.94656]),
            classes=["Low", "High"],
            confidence_threshold=0.94657,
        )
        with patch("predict.load_target", return_value=loaded):
            result = predict_target("priority", "test", top_k=2)

        self.assertEqual(result.confidence, 0.9466)
        self.assertTrue(result.requires_review)

    def test_cache_keeps_models_for_different_targets(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            model_dir = Path(temporary_directory)
            (model_dir / "category.joblib").write_bytes(b"category")
            (model_dir / "priority.joblib").write_bytes(b"priority")
            manifest = {
                "format": "sklearn-joblib-v1",
                "targets": {
                    "category": {
                        "model_file": "category.joblib",
                        "classes": ["A", "B"],
                        "confidence_threshold": 0.5,
                    },
                    "priority": {
                        "model_file": "priority.joblib",
                        "classes": ["Low", "High"],
                        "confidence_threshold": 0.6,
                    },
                },
            }
            (model_dir / "manifest.json").write_text(
                json.dumps(manifest),
                encoding="utf-8",
            )
            pipelines = {
                "category.joblib": DummyPipeline(["A", "B"], [0.2, 0.8]),
                "priority.joblib": DummyPipeline(["Low", "High"], [0.3, 0.7]),
            }

            def fake_load(path):
                return pipelines[Path(path).name]

            with patch("predict.joblib.load", side_effect=fake_load) as load_mock:
                load_target("category", model_dir=model_dir)
                load_target("priority", model_dir=model_dir)
                load_target("category", model_dir=model_dir)

            self.assertEqual(load_mock.call_count, 2)


if __name__ == "__main__":
    unittest.main()
