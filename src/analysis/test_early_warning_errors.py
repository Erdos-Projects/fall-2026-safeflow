import unittest
import numpy as np
import pandas as pd
from .early_warning_errors import ExactWarningSweep, align_predictions, forward_budget_validation


class WarningSweepTests(unittest.TestCase):
    def frame(self):
        times = pd.date_range('2024-01-02 08:00', periods=15, freq='5min')
        return pd.DataFrame(dict(station=1, timestamp=times, date=times.normalize(),
            y=[False]*8+[True]*7, y_known=True, congested_now=False, real=True,
            onset=[False]*12+[True]+[False]*2))

    def test_exact_costs_and_early_catches_against_brute_force(self):
        frame = self.frame()
        p = np.array([.9,.1,.9,.2,.8,.1,.8,.2,.4,.9,.2,.1,.6,.4,.9])
        sweep = ExactWarningSweep(frame, p)
        thresholds = np.r_[np.linspace(0,1,21), np.nextafter(np.unique(p),np.inf)]
        for row in sweep.evaluate(thresholds).itertuples():
            wrong = (p >= row.threshold) & frame.y_known.to_numpy() & ~frame.y.to_numpy()
            starts = wrong & ~np.r_[False, wrong[:-1]]
            self.assertEqual(row.false_episodes_per_day, starts.sum())
            self.assertAlmostEqual(row.wrong_alert_hours_per_day, wrong.sum()*5/60)
            self.assertEqual(row.caught_count, int((p[[6,7,8,9]] >= row.threshold).any()))

    def test_episode_cost_is_not_monotone_and_global_budget_is_feasible(self):
        frame = self.frame().iloc[:3].copy()
        sweep = ExactWarningSweep(frame, [.9,.1,.9])
        costs = sweep.evaluate([.05,.5,.95]).false_episodes_per_day.tolist()
        self.assertEqual(costs, [1.,2.,0.])
        policy, _ = sweep.operating_points(episode_budget=1, hours_budget=.2)
        self.assertTrue(policy.loc[policy.budget_type.eq('episodes'), 'false_episodes_per_day'].le(1).all())

    def test_key_alignment_and_label_guard(self):
        frame = self.frame()
        prediction = frame.assign(probability=np.linspace(0,1,len(frame)))
        actual = align_predictions(frame, prediction.iloc[::-1])
        np.testing.assert_allclose(actual, np.linspace(0,1,len(frame)))
        prediction.loc[0,'y'] = True
        with self.assertRaises(ValueError):
            align_predictions(frame, prediction)


class ForwardValidationTests(unittest.TestCase):
    def frame(self):
        base = WarningSweepTests().frame()
        parts = []
        for i, fold in enumerate(['2024-01','2024-04','2024-07']):
            part = base.copy()
            part['timestamp'] += pd.Timedelta(days=90*i)
            part['date'] = part.timestamp.dt.normalize()
            part['fold'] = fold
            parts.append(part)
        return pd.concat(parts,ignore_index=True)

    def test_future_scores_cannot_change_selected_threshold(self):
        f = self.frame()
        p = np.tile(np.linspace(.05,.95,15),3)
        first = forward_budget_validation(f,p,episode_budget=2,hours_budget=.2)
        altered = p.copy()
        altered[f.fold.eq('2024-04')] = .999
        second = forward_budget_validation(f,altered,episode_budget=2,hours_budget=.2)
        np.testing.assert_array_equal(first.loc[first.fold.eq('2024-04'),'threshold'],
                                      second.loc[second.fold.eq('2024-04'),'threshold'])
        self.assertTrue((pd.to_datetime(first.calibration_end)<pd.to_datetime(first.test_start)).all())
        # Frozen thresholds allow future budget breaches rather than retuning them away.
        self.assertTrue(second.loc[second.fold.eq('2024-04'),'budget_exceeded'].any())

    def test_unsorted_frame_is_rejected(self):
        f=self.frame().iloc[::-1]
        with self.assertRaises(ValueError):
            forward_budget_validation(f,np.full(len(f),.5))


if __name__ == '__main__':
    unittest.main()
