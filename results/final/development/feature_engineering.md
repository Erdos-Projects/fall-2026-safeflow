# Feature engineering story and its limits

The final shared schema contains a 16-feature core followed by 13 additions. The historical selection analysis is preserved in [feature_selection_results.ipynb](../../../notebooks/archive/analysis/feature_selection_results.ipynb); these are development decisions, not feature choices made using holdout results.

| Decision | Reason in the historical development analysis |
| --- | --- |
| Create `speed_anomaly` and `occ_anomaly` | Express departure from a station's usual time-of-day conditions |
| Keep `typical_congestion_30` | Represent how recurrent congestion is at that station/time, computed from training only |
| Create `up_minus_here` and `down_minus_here` | Express neighboring speed relative to the current station |
| Replace raw `tod` with `tod_sin` and `tod_cos` | Represent the daily cycle without an artificial midnight discontinuity |
| Drop `upstream_occ` and `downstream_occ` from the core | They largely duplicate local occupancy in the historical analysis |
| Replace raw downstream minimum speed with its anomaly in the core | Reduce redundancy with downstream jam distance |
| Retain downstream jam distance and its change | Describe approaching congestion along the corridor |
| Later add raw speed/occupancy, extra speed deltas, distance and recent-history measures | Enrich the shared 29-feature inputs used by all three retained models |

The archived drop-a-group study on the decorrelated-plus-downstream feature set reported positive boosted-tree contribution in all six folds for current speed, typical congestion, time/weekday, and downstream-jam groups. Postmile and ramp evidence was less consistent, and logistic regression did not benefit uniformly from the same groups. These findings support a modeling journey, not a claim that each of the final 29 features is independently necessary.

The later additions reintroduce raw speed/occupancy and temporal information, so the low-correlation rationale for the original core must not be generalized to the complete 29-feature set. Typical statistics remain training-only. Earlier endpoint/horizon experiments used different targets or cohorts and are not directly comparable final AP results. No new feature search was conducted on the now-used holdout.
