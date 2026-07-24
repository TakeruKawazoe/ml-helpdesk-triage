from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from modeling import (  # noqa: E402
    MODEL_SPECS,
    TRAIN_DATA_PATH,
    choose_confidence_threshold,
    load_dataset,
    select_model_row,
    split_development_data,
)


class ModelingTest(unittest.TestCase):
    def test_group_split_keeps_paraphrases_together(self) -> None:
        data = load_dataset(TRAIN_DATA_PATH)

        development, validation, calibration = split_development_data(data)

        development_groups = set(development["template_group"])
        validation_groups = set(validation["template_group"])
        calibration_groups = set(calibration["template_group"])
        self.assertTrue(development_groups.isdisjoint(validation_groups))
        self.assertTrue(development_groups.isdisjoint(calibration_groups))
        self.assertTrue(validation_groups.isdisjoint(calibration_groups))
        self.assertEqual(len(development) + len(validation) + len(calibration), len(data))

    def test_threshold_is_measured_from_accepted_precision(self) -> None:
        y_true = np.array(["A", "A", "B", "B", "A", "B"])
        y_pred = np.array(["A", "B", "B", "B", "A", "A"])
        probabilities = np.array(
            [
                [0.95, 0.05],
                [0.55, 0.45],
                [0.10, 0.90],
                [0.20, 0.80],
                [0.75, 0.25],
                [0.60, 0.40],
            ]
        )

        result = choose_confidence_threshold(
            y_true,
            y_pred,
            probabilities,
            desired_precision=1.0,
        )

        self.assertEqual(result["threshold"], 0.75)
        self.assertEqual(result["accepted_count"], 4)
        self.assertEqual(result["accepted_precision"], 1.0)

    def test_priority_has_explicit_high_sensitive_candidates(self) -> None:
        high_sensitive = [
            spec for spec in MODEL_SPECS if "high_weighted" in spec.name
        ]

        self.assertGreaterEqual(len(high_sensitive), 2)
        for spec in high_sensitive:
            self.assertEqual(spec.class_weight["High"], 2.5)

    def test_model_tie_break_does_not_depend_on_measured_runtime(self) -> None:
        rows = [
            {
                "model": "hybrid_linear_svc",
                "validation_macro_f1": 1.0,
                "validation_accuracy": 1.0,
                "inference_ms_per_row": 10.0,
            },
            {
                "model": "hybrid_linear_svc_balanced",
                "validation_macro_f1": 1.0,
                "validation_accuracy": 1.0,
                "inference_ms_per_row": 1.0,
            },
        ]

        winner = select_model_row("category", rows)

        self.assertEqual(winner["model"], "hybrid_linear_svc")


if __name__ == "__main__":
    unittest.main()
