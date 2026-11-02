"""Small deterministic checks for preprocessing, alignment, and publication guards."""
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import random_forest_training as training


class FoldTrainingTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({"speed": np.arange(80, dtype=float),
            "flow": np.tile([np.nan, 1., 2., 3.], 20), "dow": np.arange(80) % 5,
            "y_known": [True] * 60 + [False] * 20, "y": np.arange(80) % 2})
        self.frame.loc[60:, "speed"] = 1_000_000.
        self.temp = tempfile.TemporaryDirectory(dir=training.LOCAL_DIR)
        self.output = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_train_only_imputation_unknown_validation_and_checkpoint(self):
        rows = np.array([75, 61, 79, 60])
        result = training.fit_fold(self.frame, ["speed", "flow", "dow"], np.arange(60), rows,
            self.output, model_parameters=dict(n_estimators=5, n_jobs=1, min_samples_leaf=1))
        model = joblib.load(self.output / "model.joblib")
        self.assertEqual(model.named_steps['prepare'].named_transformers_['numeric'].statistics_[0], 29.5)
        np.testing.assert_array_equal(result['positions'], rows)
        prediction = model.predict_proba(self.frame.iloc[rows][['speed', 'flow', 'dow']].astype('float32'))[:, 1]
        np.testing.assert_allclose(result['probability'], prediction)
        self.assertEqual(len(result['probability']), 4)

    def test_forbid_leaky_or_overlapping_training(self):
        with self.assertRaises(ValueError):
            training.fit_fold(self.frame, ['speed', 'y'], np.arange(20), np.arange(60, 80), self.output)
        with self.assertRaises(ValueError):
            training.fit_fold(self.frame, ['speed'], np.arange(20), np.arange(10, 30), self.output)
        with self.assertRaises(ValueError):
            training.fit_fold(self.frame, ['speed'], np.arange(80), np.array([1]), self.output)

    def test_quick_and_incomplete_runs_cannot_publish(self):
        self.assertIsNone(training.publish_team_results(self.output, 'test', pd.DataFrame(), [], {}, quick_run=True))
        with self.assertRaises(ValueError):
            training.publish_team_results(self.output, 'test', pd.DataFrame(index=['one']), [],
                                           {'expected_fold_names': ['one']}, quick_run=False)
        self.assertFalse((self.output / 'notebooks/results/summary.csv').exists())


if __name__ == '__main__':
    unittest.main()
