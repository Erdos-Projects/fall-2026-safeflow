# Early-warning error analysis

`notebooks/archive/analysis/error_analysis_models.ipynb` extends the original `notebooks/archive/error_analysis.ipynb`
to saved model probabilities. It does not fit models or read the final holdout.

The Random Forest full-run export is detected automatically. Set
`TRANSFORMER_INPUT` to the transformer's run directory containing six
`<quarter>/validation_predictions.parquet` files, or to a combined Parquet file.
The expected columns are `station`, `timestamp`, `y`, `y_known`,
`congested_now`, `real`, `onset`, and `probability`. Every validation row is
required; row order may differ. The notebook checks keys and outcome flags
against the shared six-fold development reference before comparing models.
Absence of a transformer input is explicitly recorded as pending, rather than
substituting aggregate scores or another model's predictions.

Completed boosted-tree exports under `src/boosted_trees/results/` are detected
automatically. Their feature list, shared-code hash, and fold definitions are
checked against the common contract. `BOOSTED_INPUT` can also point to a combined
legacy export from the root notebook. The probability column is inferred as
`probability` for standardized exports or `29 features` for legacy exports; set
`BOOSTED_PROBABILITY_COLUMN` explicitly if the schema is ambiguous.

The notebook produces recurrence counts, group-specific AP and episode recall,
station/time/condition breakdowns, and corridor-speed case studies. Recurrence
uses each fold's training-only schedule 15 minutes before the onset. Global
budget policies include all onsets; group analyses report missing recurrence
context separately.

For exact warning-cost computation, wrong-alert rows form disjoint paths in
station/time order. At a score threshold, the number of false-warning episodes
is the number of active wrong-alert rows minus active adjacent pairs. A pair
activates at the minimum of its two probabilities. This supports a full
score-breakpoint sweep without assuming monotone episode counts. Early catch
is determined by the maximum probability 15/20/25/30 minutes before onset.
The selected policies are checked against `sf.alert_metrics`.

Thresholds are selected and reported on pooled development predictions, so the
results are exploratory. They are not independent next-quarter or holdout
performance estimates. Case examples are selected using Random Forest outcomes,
with all available models subsequently plotted on those same examples.

Run helper checks with:

```text
python -m unittest src.analysis.test_early_warning_errors -v
```

Outputs are under `results/error_analysis_models/<timestamp>/`. Reference caches
are fingerprinted and Git-ignored. Compact tables, plots, input provenance, and
pending-model status are saved separately. The shared feature module and
existing team summary remain unchanged.

## Forward evaluation

`notebooks/archive/analysis/forward_warning_budget_validation.ipynb` calibrates each policy on expanding earlier out-of-fold quarters and evaluates the frozen threshold in the next quarter (five evaluated quarters). It reuses the verified error-analysis context cache and prediction provenance. Tables report actual future burden, budget breaches and onset-weighted recall; costs are weighted by days. This is development-only threshold transfer, not final holdout performance. Artifacts are saved in `results/forward_warning_budget_validation/<timestamp>/`.

## Paired uncertainty and operational readiness

`notebooks/archive/analysis/model_decision_analysis.ipynb` reconstructs daily event/cost counts from the frozen forward policies. It checks their totals against the forward tables, then uses 5,000 paired calendar-week resamples within each quarter, with a daily-resampling sensitivity analysis. All stations are retained together. The main contrast is Transformer minus boosted trees under the duration-budget policy; Random Forest and the episode policy are secondary comparisons. Ratios are computed from summed counts in every replicate, rather than averaging daily recalls.

Intervals condition on saved fitted models and thresholds, exclude model/feature-selection uncertainty, and do not establish performance at identical realized cost. Local model checkpoint availability, feature-source hashes and historical timing evidence are audited separately. Missing fitted checkpoints prevent a measured latency/memory/outage comparison; the notebook records the exact exports needed. Outputs are under `results/model_decision_analysis/<timestamp>/`.

```text
python -m unittest src.analysis.test_paired_uncertainty src.analysis.test_early_warning_errors -v
```

See [the operational benchmark protocol](operational_benchmark_protocol.md) for required checkpoints, probability reproduction, controlled timing/memory measurements and causal outage stress tests.

## Local fresh-fit benchmark

`notebooks/archive/analysis/local_model_benchmark.ipynb` uses `local_benchmark.py` workers to train and benchmark RF, boosted trees and Transformer on the same last development fold. Set `QUICK_RUN = False` for established full model sizes and the 2-million training cap. Install `requirements-benchmark.txt` in the selected kernel. Each model runs in separate training/inference processes, with common thread limits and checkpoint reproduction checks. Outputs are under `results/local_model_benchmark/<run>/`; fitted models stay in each model's local results directory. No team metrics are published automatically.

## Frozen holdout evaluation

`evaluate_holdout.py` evaluates the unchanged local checkpoint fits, using April-June 2025 for calibration and July-December 2025 for testing. It writes a frozen protocol and hashes before reading holdout files, uses only the original training mask for typical features, and checks calibration-feature reproduction. The shared module's default holdout exclusion is unchanged.

The completed report is [HOLDOUT_REPORT.md](../../results/holdout_evaluation/20261104_160843_077282/HOLDOUT_REPORT.md). Holdout data has now been evaluated and must not be described as untouched. There was no final refit through June and no holdout-based retuning. To regenerate tables/plots from the existing frozen artifacts, use the report-only stage:

```text
python -m src.analysis.evaluate_holdout --stage report --run-dir results/holdout_evaluation/20261104_160843_077282
```
