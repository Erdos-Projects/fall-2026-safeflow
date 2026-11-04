import unittest
import numpy as np
import pandas as pd
from .paired_uncertainty import paired_bootstrap

class PairedBootstrapTests(unittest.TestCase):
    def sample(self):
        rows=[]
        for model in ["A","B"]:
            for i in range(14):
                rows.append(dict(model=model,fold="Q1" if i<7 else "Q2",date=pd.Timestamp("2024-01-01")+pd.Timedelta(days=i),
                    days=1,onsets=10,caught=5,false_episodes=2,wrong_rows=6,rare_onsets=2,rare_caught=1))
        return pd.DataFrame(rows)

    def test_identical_models_have_exactly_zero_paired_uncertainty(self):
        result,draws=paired_bootstrap(self.sample(),"A","B",repetitions=100)
        np.testing.assert_array_equal(draws,np.zeros_like(draws))
        self.assertTrue(result.ci_low.eq(0).all())

    def test_known_constant_difference_and_reproducibility(self):
        d=self.sample();d.loc[d.model.eq("A"),"caught"]+=2
        result,draws=paired_bootstrap(d,"A","B",block="day",repetitions=100)
        np.testing.assert_allclose(draws[:,0],.2)
        _,again=paired_bootstrap(d,"A","B",block="day",repetitions=100)
        np.testing.assert_array_equal(draws,again)
        self.assertAlmostEqual(result.iloc[0].difference,.2)

    def test_different_days_in_same_week_are_not_treated_as_paired(self):
        d=self.sample().iloc[[0,7,14,21]].copy()
        d.iloc[0,d.columns.get_loc("date")]+=pd.Timedelta(days=1)
        with self.assertRaises(ValueError):paired_bootstrap(d,"A","B",repetitions=100)

    def test_unpaired_population_rejected(self):
        d=self.sample();d.loc[0,"onsets"]+=1
        with self.assertRaises(ValueError):paired_bootstrap(d,"A","B",repetitions=100)

if __name__ == "__main__":unittest.main()
