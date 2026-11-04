# Operational comparison protocol

The current development exports contain predictions but no fitted Transformer or boosted-tree checkpoints. `model_decision_analysis.ipynb` audits this explicitly. Prediction Parquet size and architectural input counts must not be reported as model memory or speed measurements.

## Fresh local comparison

Use [local_model_benchmark.ipynb](../../notebooks/analysis/local_model_benchmark.ipynb) to train fresh versions of all three models with the current feature source, instead of recovering old checkpoints. Start with quick mode, then run the full last development fold for a meaningful comparison. These are newly evaluated fits and do not establish deployment costs for the exact previously published checkpoints.

The notebook runs separate training and inference subprocesses, checks checkpoint prediction reproduction, records CPU/thread/environment/source provenance, and measures warm latency, checkpoint size and absolute process peak RSS. Raw-to-engineered-feature extraction is excluded from per-model inference timing. Optional missing-feature diagnostics are not a replacement for causal detector-outage tests.

## Required artifacts

Use the 2025 Q2 fold from these exact development runs:

- Transformer: `src/transformer/results/transformer_29_features_20261031_085746_133888/2025-04/model.pt`. Current training saves weights, architecture, ordered logical features and fold-specific preprocessing together. Also provide the original `safeflow_features.py` snapshot matching the recorded configuration hash.
- Boosted trees: `src/boosted_trees/results/boosted_trees_29_features_full_20261104_000723/2025-04/model.joblib`, with the existing manifest and ordered features. The training notebook's `SAVE_MODELS = True` saves this locally.

The notebooks already support saving these files; predictions alone cannot reconstruct the fitted models. Retrieve original files where possible rather than fitting a different replacement and attributing its performance to the saved run.

## Fixed development fixture and reproduction gate

Use a deterministic sample from the same 2025 Q2 development endpoints for both models. Preserve the Transformer historical context up to each endpoint and use its saved training-only preprocessing. Build engineered features using the exact recorded feature source and the fold's training-only typical values. Do not open data dated July 2025 or later. Save endpoint keys and a source fingerprint with the fixture.

Before any timings, reproduce each model's saved clean probabilities on that fixture. Record maximum/mean absolute error and investigate discrepancies beyond expected numerical precision. Missing checkpoints, source mismatch or failed reproduction must be reported as pending, not replaced with synthetic or randomly initialized models.

## Controlled measurements

Use the same CPU machine and fixed thread count for both models. Record processor, RAM, OS, library versions, thread settings and model/source hashes. Run each model in a separate process. Use 20 warm-up batches followed by at least 200 measured batches; repeat on the same endpoint sequence at batch sizes 1 and 46. Report median and p95 batch latency, endpoints/second and variation between repeats. Record load/cold-start time separately. GPU results may be supplementary; do not compare CPU boosted trees with GPU Transformer as the sole comparison.

Report feature preparation, Transformer history construction, and model-only inference separately, then measure the full prediction path. Report serialized checkpoint bytes and process peak resident memory with a process baseline. Do not infer full process memory from tensor input bytes or Python allocations alone.

## Missing-input robustness

Use the same predeclared detector/time blocks, unmodified outcomes and frozen forward thresholds. Test outages of 5, 15 and 30 minutes separately for current, upstream and downstream detectors. Mask the underlying measurements and recompute every affected lag, delta, spatial and history feature causally; changing only one derived feature can leave leaked outage information in related features. Do not refit or recalculate typical values using evaluation outcomes. Apply the original preprocessing and history-gap behavior.

Report clean versus perturbed early detection, false episodes/day and wrong-alert hours/day over the common population, together with prediction coverage and paired block intervals. Keep outcome availability fixed using the original measured labels so masked input readings do not silently remove difficult evaluation cases. These stress tests assess simulated outages, not all real sensor failures.

## Decision

Treat the current confidence intervals as conditional development evidence. Compare detection and actual burden jointly, then assess whether any benefit justifies measured deployment cost. Agree the final fitting dates, threshold calibration procedure and reporting metrics before accessing the final holdout.
