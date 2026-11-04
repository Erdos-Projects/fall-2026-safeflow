# Boosted trees with the shared 29 features

Run `notebooks/train_boosted_trees.ipynb` from the project root or `notebooks/`.
Use `QUICK_RUN = True` for a last-fold smoke test, then `False` for the complete
six-quarter walk-forward experiment.

The shared module remains `notebooks/safeflow_features.py`. No model-specific
features or label changes are introduced here. All 29 current shared features go
straight into scikit-learn's `HistGradientBoostingClassifier`, which handles
missing values natively and needs no scaling. Day of week enters as the number 0-4.
A new unfitted model is created for every fold.

Settings: 300 trees, learning rate 0.05, 31 leaves per tree, minimum leaf size 50,
no early stopping (so every fold trains exactly 300 trees), seed 0. Up to 2 million
known-label training rows are sampled per fold with seed 0, matching the team's
sampling convention. The settings were fixed in advance, not tuned on the folds.

Artifacts live in `src/boosted_trees/results/<timestamped-run>/`: manifest, fitted
models, validation predictions, fold metrics, trade-off and calibration tables.
Only each full run's `manifest.json` and per-fold `validation_predictions.parquet`
are committed (see `.gitignore`). Completed full runs also publish
`boosted_trees_folds.csv` and `boosted_trees_manifest.json` to `notebooks/results/`
and replace the boosted-trees row of `summary.csv`. Quick runs never alter the
shared summary.

The manifest records the exact feature list, shared-code and training-code hashes,
model parameters, seeds, sampling, fold definitions, per-fold row counts, a hash
of each fold's validation rows and labels, a hash of the data files, and package
versions. The July-December 2025 holdout is not loaded or scored.

Reruns on another machine can differ in the fourth decimal of AP because of
floating-point differences, even with the same seeds.
