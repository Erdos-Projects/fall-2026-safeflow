"""Synthetic tabular worker integration checks; timings are not research results."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from . import local_benchmark as worker

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"notebooks"))

class LocalBenchmarkTests(unittest.TestCase):
    def test_tabular_workers_save_reload_and_reproduce_probabilities(self):
        rng=np.random.default_rng(7)
        time=pd.date_range("2025-04-01 06:00",periods=180,freq="5min")
        frame=pd.DataFrame(dict(station=np.repeat([1,2],180),timestamp=np.tile(time,2),
            speed=rng.uniform(20,70,360).astype("float32"),flow=rng.normal(100,10,360).astype("float32"),dow=1))
        frame["date"]=frame.timestamp.dt.normalize()
        frame["y"]=frame.speed.lt(40)
        frame["y_known"]=True;frame["real"]=True;frame["congested_now"]=False;frame["onset"]=False
        frame.loc[[150,330],"onset"]=True
        frame.loc[::19,"flow"]=np.nan
        train=np.r_[np.arange(120),np.arange(180,300)]
        val=np.r_[np.arange(120,180),np.arange(300,360)]
        for model in ["random_forest","boosted_trees"]:
            with self.subTest(model=model):
                parent=ROOT/"src"/model/"results"
                parent.mkdir(exist_ok=True)
                # Temporary cleanup stays inside the explicitly verified workspace path.
                with tempfile.TemporaryDirectory(prefix="benchmark_test_",dir=parent) as name:
                    folder=Path(name).resolve()
                    self.assertTrue(folder.is_relative_to(ROOT))
                    frame.to_parquet(folder/"frame.parquet",index=False)
                    np.save(folder/"train.npy",train);np.save(folder/"val.npy",val)
                    config=dict(threads=1,quick=True,features=["speed","flow","dow"],
                        frame=str(folder/"frame.parquet"),train_rows=str(folder/"train.npy"),val_rows=str(folder/"val.npy"),
                        fixture=str(folder/"frame.parquet"),fixture_rows=str(folder/"val.npy"),
                        model_dirs={model:str(folder)},warmup=1,repetitions=2,threshold=.5,missing_feature_tests=True)
                    # psutil is optional in this validation environment. Mock RSS only;
                    # real estimators, serialization, inference and reproduction checks run.
                    with patch.object(worker,"memory",return_value=dict(rss_bytes=0,peak_rss_bytes=0)),contextlib.redirect_stdout(io.StringIO()):
                        worker.train(config,model)
                        worker.inference(config,model)
                    result=json.loads((folder/"inference_benchmark.json").read_text())
                    self.assertLessEqual(result["max_reproduction_error"],1e-10)
                    self.assertTrue((folder/"model.joblib").exists())
                    self.assertEqual(len(pd.read_csv(folder/"latency.csv")),6)
                    self.assertEqual(len(pd.read_csv(folder/"feature_stress.csv")),10)
                    self.assertEqual(len(pd.read_parquet(folder/"validation_predictions.parquet")),len(val))

if __name__=="__main__":unittest.main()
