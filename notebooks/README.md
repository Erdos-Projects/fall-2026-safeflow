# Notebook guide

## Current training workflow

Training entry points remain in this directory: [Random Forest](train_random_forest.ipynb), [Transformer](train_transformer.ipynb), [boosted trees](train_boosted_trees.ipynb), and the [shared template](train_and_evaluate.ipynb). All use `safeflow_features.py` here. `results/` remains the team publication location.

Read the [results summary](../results/RESULTS_SUMMARY.md) for the consolidated findings and any still-pending local benchmark measurements.

## Analyses to read

- [Feature selection evidence](analysis/feature_selection_results.ipynb): the selection rationale and saved findings.
- [Published metric comparison](analysis/compare_models_29_features.ipynb): a saved comparison snapshot; consult `results/summary.csv` for subsequent updates.
- [Random Forest results](analysis/random_forest_results.ipynb): diagnostics from the saved full run.
- [All-model error analysis](analysis/error_analysis_models.ipynb): aligned prediction comparisons, recurrence and station/time breakdowns; pooled budget results are exploratory.
- [Forward warning budgets](analysis/forward_warning_budget_validation.ipynb): earlier-quarter calibration, frozen next-quarter evaluation, realized budget breaches and recurrence breakdowns.
- [Model decision analysis](analysis/model_decision_analysis.ipynb): paired weekly confidence intervals, daily sensitivity and operational artifact readiness for Transformer versus boosted trees.
- [Local model benchmark](analysis/local_model_benchmark.ipynb): fresh last-fold fits of all three models, isolated CPU training/inference measurements and optional missing-feature diagnostics. Start in quick mode; see `requirements-benchmark.txt`.

Run the all-model error analysis before forward validation if its ignored development-context cache has not been generated. Saved prediction exports for all six quarters are required. Neither analysis retrains models or accesses the final holdout.

## EDA and historical work

`eda/` contains the data exploration and `data_and_features.ipynb` extraction workflow. The final models instead build their dataset through the shared feature module. `archive/` contains older baselines, horizon/speed-difference experiments, feature-selection variants and the original teammate error-analysis notebook. Saved outputs are preserved; old experiments may require their original dependencies and datasets.

[notebook_locations.json](notebook_locations.json) maps every relocated notebook. Data and output directories were not relocated. Open the new path in the IDE if an old tab points at a moved file. Launch notebooks from anywhere inside the repository; Python packages still need to be installed in the selected kernel.
