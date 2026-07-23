from __future__ import annotations

import sys
import unittest
from collections import Counter
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from generate_dummy_data import (  # noqa: E402
    CATEGORY_SPECS,
    HOLDOUT_CASES_PER_PRIORITY,
    PRIORITIES,
    TRAIN_CASES_PER_PRIORITY,
    TRAIN_VARIANTS_PER_CASE,
    build_holdout_records,
    build_training_records,
)


class GenerateDummyDataTest(unittest.TestCase):
    def test_training_data_is_balanced_and_grouped_by_semantic_case(self) -> None:
        records = build_training_records()

        expected_count = (
            len(CATEGORY_SPECS)
            * len(PRIORITIES)
            * TRAIN_CASES_PER_PRIORITY
            * TRAIN_VARIANTS_PER_CASE
        )
        self.assertEqual(len(records), expected_count)
        self.assertEqual(
            Counter(record["label_category"] for record in records),
            {spec.category: 60 for spec in CATEGORY_SPECS},
        )
        self.assertEqual(
            set(Counter(record["template_group"] for record in records).values()),
            {TRAIN_VARIANTS_PER_CASE},
        )

    def test_holdout_uses_novel_text_and_all_labels(self) -> None:
        training_records = build_training_records()
        holdout_records = build_holdout_records()

        expected_count = (
            len(CATEGORY_SPECS) * len(PRIORITIES) * HOLDOUT_CASES_PER_PRIORITY
        )
        self.assertEqual(len(holdout_records), expected_count)
        self.assertEqual(
            {record["label_category"] for record in holdout_records},
            {spec.category for spec in CATEGORY_SPECS},
        )
        self.assertEqual(
            {record["label_priority"] for record in holdout_records},
            set(PRIORITIES),
        )
        self.assertTrue(
            {record["inquiry_text"] for record in training_records}.isdisjoint(
                record["inquiry_text"] for record in holdout_records
            )
        )

    def test_out_of_scope_category_has_triage_owner(self) -> None:
        out_of_scope = next(
            spec for spec in CATEGORY_SPECS if spec.category == "その他・対象外"
        )

        self.assertEqual(out_of_scope.department, "総務")
        self.assertIn("エアコン", out_of_scope.holdout_objects)


if __name__ == "__main__":
    unittest.main()
