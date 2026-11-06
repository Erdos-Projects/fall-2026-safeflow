# Executive summary

## Purpose and decision context

SafeFlow studies whether freeway measurements can provide useful advance warning of congestion. The intended users are transportation agencies and traffic management center operators who need to prioritize corridor monitoring without being overwhelmed by false warnings. We evaluate prediction and warning burden; we have not demonstrated that acting on these warnings reduces congestion or travel times.

For each station every five minutes, the target is whether measured speed will be **strictly below 40 mph at any of the next six readings**, covering the next 30 minutes. A warning is counted as early when it precedes a congestion onset by 15-30 minutes.

## Data and evaluation

The current study uses D12 I-5 measurements from 46 mainline stations. Features describe current conditions, recent changes, neighboring detectors, ramp flows and typical traffic patterns. All three models use 29 logical features; the Transformer additionally consumes historical sequences.

The frozen-checkpoint test uses training target windows ending before March 25, 2025, April-June for threshold calibration, and July-December for holdout evaluation. The holdout contains 966,000 rows and 3,497 onsets across 125 eligible days. These checkpoints were not refitted through June. The holdout has now been used; subsequent improvements need new independent validation.

The feature-engineering story includes replacing redundant raw quantities with anomalies, neighboring speed differences and cyclic time encoding; dropping duplicate neighboring occupancy from the core; and later enriching the shared set with raw measurements and recent history. See [the documented decisions](../results/final/development/feature_engineering.md); the holdout was not used for feature selection.

## What policy and budget mean

A **warning policy** is the rule that turns a predicted probability into an alert: issue a warning when the probability is at least a model-specific threshold. The threshold is chosen on calibration data and then frozen. This is an alerting rule, not a policy for controlling traffic.

A **warning budget** is a target for acceptable false-warning burden, not a financial allocation or a guaranteed limit on future traffic. Our study uses two illustrative targets that require stakeholder validation:

- **Duration policy:** select the threshold for greatest early detection while calibration wrong-alert duration is at most four hours/day, summed across stations.
- **Episode policy:** select the threshold for greatest early detection while calibration false-warning episodes are at most 20/day.

One uninterrupted 30-minute false warning at one station is **one episode and 0.5 wrong-alert hours**. Six separated five-minute false warnings are **six episodes and the same 0.5 hours**. Thus, episode count measures repeated interruptions, while duration measures time spent warning incorrectly. Simultaneous warnings at different stations contribute separately to duration.

In the original tradeoff plots, circles represent the duration policy, squares the episode policy, and triangles the fixed 0.5 diagnostic threshold. Colors identify models. Equal calibration targets do not imply identical realized holdout burden.

## Holdout results

AP measures ranking on measured, currently uncongested origins with known future outcomes. It is separate from the fraction of congestion events warned early.

| Model | Holdout AP | Onsets warned at least 15 min early, duration policy | Actual wrong-alert hours/day |
| --- | --- | --- | --- |
| Boosted trees | 0.7412 | 54.3% | 4.87 |
| Transformer | 0.7380 | 56.7% | 4.97 |
| Random Forest | 0.7216 | 43.9% | 4.81 |

Boosted trees lead on AP and Brier score. Transformer detects more onsets early under the frozen duration policy: its advantage over boosted trees is 2.34 percentage points, with a paired weekly 95% interval of +1.19 to +3.49. Its wrong-alert duration is slightly higher; the interval for that duration difference spans zero. This is not an exactly matched-cost comparison.

**None of the models met the four-hours/day duration target on holdout.** All three also exceeded the separate 20-episode/day target. No threshold was retuned to hide those breaches.

The local CPU benchmark found that boosted trees trained much faster and processed the full fixture faster. Transformer had the lowest warm small-batch latency, but required about 130 minutes for fitting, prediction and saving, versus 0.81 minutes for boosted trees. Inference cost depends on batch size and preprocessing scope.

## Conclusion and next steps

**No single final model is designated. All three models are retained, with boosted trees and Transformer as the leading candidates.** While neither leading candidate satisfies the target warning budget on holdout, both demonstrate the predictive feasibility of the proposed target design. They do not yet establish budget-compliant operational readiness.

Further improvements, additional data and retraining may help, but must be assessed on new independent validation data; retraining is not a demonstrated remedy for the budget breach. Non-recurrent congestion remains a substantial weakness. Results are limited to this D12 cohort, and missing-feature diagnostics do not establish robustness to real detector outages.

The release preserves model artifacts, frozen policies, calibration and error plots, and reproduction commands. Historical six-fold results remain separately labeled provenance-backed evidence; the frozen final workflow does not reconstruct historical fits whose original feature-source snapshots are unavailable.

## Evidence

- [Curated holdout report](../results/final/holdout/HOLDOUT_REPORT.md)
- [Final score tables](../results/final/holdout/holdout_metrics.csv)
- [Recomputed metrics and generated figures](../results/final/README.md)
- [Final results showcase](../notebooks/final_results.ipynb)
- [All three artifact bundles](../artifacts/README.md)
