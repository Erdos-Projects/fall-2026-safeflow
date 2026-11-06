import unittest
import torch  # Initialize native DLLs before tabular worker tests enter threadpool control.
from pathlib import Path
import hashlib,json
from src.analysis.study_release import ROOT,config
from src.analysis.release_pipeline import validate

class ReleaseTests(unittest.TestCase):
    def test_no_winner_and_recorded_configuration(self):
        cfg=config();validate(cfg)
        self.assertIsNone(cfg['selected_final_model'])
        self.assertEqual(set(cfg['models']),{'random_forest','boosted_trees','transformer'})

    def test_portable_checkpoints_and_frozen_feature_statistics(self):
        from src.analysis.release_verification import verify
        result=verify()
        self.assertEqual(len(result),3)
        self.assertTrue(all(r['training_feature_statistics_reproduce'] for r in result))

if __name__=='__main__':unittest.main()
