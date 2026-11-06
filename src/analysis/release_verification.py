"""Portable release verification independent of original run directories."""
from pathlib import Path
import json
from .study_release import ROOT,MODELS,CandidateBundle,digest

def verify():
    # Load Torch before Arrow/threadpool initialization on Windows.
    import torch
    import numpy as np
    import pandas as pd
    fixtures=ROOT/'tests/fixtures'
    source=json.loads((fixtures/'README.json').read_text())
    assert digest(fixtures/'feature_frame.parquet')==source['feature_frame_sha256']
    expected=pd.read_parquet(fixtures/'expected_probabilities.parquet')
    frame=pd.read_parquet(fixtures/'feature_frame.parquet');rows=np.load(fixtures/'endpoint_rows.npy')
    results=[]
    for model in ['transformer','random_forest','boosted_trees']:
        bundle=CandidateBundle(ROOT/'artifacts'/model)
        assert not bundle.metadata['selected_final_model']
        assert bundle.metadata['training_window_end_before']=='2025-03-25'
        probability=bundle.predict_features(frame,rows)
        error=float(np.max(np.abs(probability-expected[model].to_numpy())))
        tolerance=1e-5 if model=='transformer' else 1e-10
        assert error<=tolerance,(model,error,tolerance)
        restored=bundle.apply_training_statistics(frame)
        for feature in ['typical_congestion_30','speed_anomaly','occ_anomaly','min_speed_down_anomaly']:
            np.testing.assert_allclose(restored[feature],frame[feature],rtol=1e-6,atol=1e-6,equal_nan=True)
        results.append(dict(model=model,endpoints=len(rows),max_probability_error=error,tolerance=tolerance,
            training_feature_statistics_reproduce=True))
    manifest=json.loads((ROOT/'results/final/manifest.json').read_text())
    for item in manifest['files']:
        assert digest(ROOT/'results/final'/item['path'])==item['sha256'],item['path']
    return results
