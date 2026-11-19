"""Signed and nonnegative tasks cannot share a completed-model checkpoint."""
from pathlib import Path
import unittest
from unittest.mock import Mock

import pandas as pd

from pavement_analytics.io import write_json
from pavement_analytics.model_tasks import task_key, verified_task
from pavement_analytics.security import SafeError
from temp_support import temporary_directory


class SignedTaskCheckpointTests(unittest.TestCase):
    def setUp(self):
        temporary = temporary_directory(); self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.arguments = {
            "table": pd.DataFrame({"row_id": [0, 1], "group_id": ["001", "002"], "target": [1., 2.], "x": [3., 4.]}),
            "plan": pd.DataFrame({"group_id": ["001", "002"], "partition": ["development", "holdout"], "fold": [0, -1]}),
            "features": ["x"], "climate": [], "target": "target", "seed": 42,
            "weight_column": "delta_days", "options": {"candidate_ids": ["enet_0.01"]},
        }

    def test_kind_changes_checkpoint_identity_even_for_identical_aliases_and_values(self):
        deduct = task_key(**self.arguments)
        self.assertEqual(deduct, task_key(**self.arguments, target_kind="deduct_total"))
        self.assertNotEqual(deduct, task_key(**self.arguments, target_kind="signed_pci_rate"))
        with self.assertRaises(SafeError):
            task_key(**self.arguments, target_kind="signed_typo")

    def test_signed_checkpoint_preserves_predictions_and_never_reuses_deduct_task(self):
        cache = self.root/"cache"
        for target_kind, value in (("deduct_total", 2.), ("signed_pci_rate", -3.)):
            with self.subTest(target_kind=target_kind):
                key = task_key(**self.arguments, target_kind=target_kind)
                def build(destination):
                    destination.mkdir()
                    summary = {"target": "target", "target_kind": target_kind, "status": "supported"}
                    write_json(destination/"summary.json", summary)
                    pd.DataFrame({"prediction": [value]}).to_csv(destination/"candidate_predictions.csv", index=False)
                    return summary
                builder = Mock(side_effect=build)
                first = verified_task(cache, key, builder, self.root/(target_kind+"_first"))
                builder.assert_called_once()
                must_reuse = Mock(side_effect=AssertionError("a complete task must not refit"))
                repeated = verified_task(cache, key, must_reuse, self.root/(target_kind+"_replayed"))
                must_reuse.assert_not_called()
                self.assertEqual(first, repeated)
                self.assertEqual(repeated["target_kind"], target_kind)
                predictions = pd.read_csv(self.root/(target_kind+"_replayed")/"candidate_predictions.csv")
                self.assertEqual(predictions.prediction.tolist(), [value])
        self.assertEqual(len([path for path in cache.iterdir() if path.is_dir()]), 2)


if __name__ == "__main__":
    unittest.main()
