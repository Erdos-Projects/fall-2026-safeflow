"""Holdout isolation checks using synthetic data only."""
import tempfile
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from .evaluate_holdout import ROOT,window_files
import safeflow_features as sf

class HoldoutIsolationTests(unittest.TestCase):
    def test_holdout_file_window_excludes_development_and_next_year(self):
        with tempfile.TemporaryDirectory(prefix='holdout_window_',dir=ROOT/'.venv') as name:
            path=Path(name).resolve()
            self.assertTrue(path.is_relative_to(ROOT))
            for day in ['2025_06_30','2025_07_01','2025_12_31','2026_01_01']:
                (path/f'd12_text_station_5min_{day}.parquet').touch()
            self.assertEqual([p.stem[-10:] for p in window_files(path)],['2025_07_01','2025_12_31'])

    def test_future_outcomes_cannot_change_training_typical_features(self):
        f=pd.DataFrame(dict(station=1,tod=480,date=pd.to_datetime(['2025-03-01','2025-03-02','2025-03-03','2025-07-01','2025-07-02']),
            speed=[55.,65.,75.,20.,30.],occupancy=[.1,.2,.3,.9,.8],min_speed_down=[50.,60.,70.,10.,20.],
            real=True,y=[False,True,False,True,True],y_known=True,y_speed=[50.,60.,70.,10.,20.],y_speed_real=True))
        train=pd.Series([True,True,True,False,False])
        a=f.copy();sf.add_typical_features(a,train)
        b=f.copy();b.loc[~train,'y']=False;b.loc[~train,['speed','occupancy','min_speed_down','y_speed']]=999.
        sf.add_typical_features(b,train)
        np.testing.assert_allclose(a.loc[~train,'typical_congestion_30'],b.loc[~train,'typical_congestion_30'])
        np.testing.assert_allclose(a.loc[~train,'typical_congestion_30'],1/3)
        np.testing.assert_allclose(a.loc[~train,'speed']-a.loc[~train,'speed_anomaly'],65.)
        np.testing.assert_allclose(b.loc[~train,'speed']-b.loc[~train,'speed_anomaly'],65.)

if __name__=='__main__':unittest.main()
