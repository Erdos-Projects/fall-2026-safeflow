# Random Forest with the shared 29 features

Run `notebooks/train_random_forest.ipynb` from the project root or `notebooks/`.
Use `QUICK_RUN = True` for the last-fold smoke test, then `False` for the complete
six-quarter walk-forward experiment. `notebooks/random_forest_results.ipynb`
reloads completed results and draws the comparison charts without retraining.

The shared module remains `notebooks/safeflow_features.py`; no model-specific
features or label changes are introduced here. All 29 current shared features
are passed to training-only preprocessing: numeric median imputation with
missingness indicators, and categorical weekday encoding. Scaling is unnecessary
for the forest. A new unfitted pipeline is created for every fold.

Full defaults: 300 trees, depth 12, minimum leaf size 10, bootstrap quarter-samples,
sqrt feature selection, no class weighting, seed 42. Up to 2 million known-label
training endpoints are selected per fold with seed 0, matching the team's sampling
convention. Quick runs use 50 trees and at most 200,000 training endpoints.

Artifacts live in `src/random_forest/results/<timestamped-run>/`: manifest, fitted
pipelines, preprocessing details, validation probabilities, fold metrics, and
figures. These large local run folders are ignored by Git. Completed full runs
also publish `<MODEL_NAME>_folds.csv` and `<MODEL_NAME>_manifest.json` to
`notebooks/results/` and update that directory's existing `summary.csv`, matching
the transformer workflow. Quick runs never alter the shared summary.

The manifest records the exact feature list, shared-code hash, model parameters,
fold definitions, versions, and sampling settings. Score comparisons should check
these contracts, not just model names. Existing boosted-tree results use 16
features; the transformer row uses 29. The final holdout is not loaded or scored.

Impurity importance is diagnostic and can distribute credit across correlated
features; it is not an estimate of a feature's causal effect.
