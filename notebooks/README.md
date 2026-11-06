# Notebook guide

## Final study showcase

[final_results.ipynb](final_results.ipynb) is the clean, executable comparative-study report. It calls versioned Python modules/scripts to regenerate figures and verify all frozen holdout scores. It does not designate a winning model. Full training replay is optional and off by default because it can take hours; the report documents exactly how to run it.

The primary execution interfaces are now:

```text
python scripts/generate_report.py
python scripts/verify_release.py
python scripts/reproduce_results.py
```

Run from the repository root. The first two use the packaged evidence and checkpoints. The last rebuilds features/fits from raw data in a separate reproduction workspace. See [README](../README.md), [executive summary](../presentation/summary.md) and [results/final](../results/final/README.md).

## EDA retained unchanged

All files under `eda/` have been retained as-is, including the team's numbering and filenames. They preserve the original exploratory work and can have historical dataset/dependency requirements. They are not silently converted or rewritten by the release pipeline.

## Archived research notebooks

Earlier training notebooks are under `archive/training/`; comparison, error and benchmark notebooks are under `archive/analysis/`. Older baselines and feature experiments remain in `archive/`. These remain available as research provenance. Existing model implementations and their run artifacts are preserved under `src/`.

`notebook_locations.json` maps relocated paths. The byte-preserved `safeflow_features.py` remains here for historical source-hash compatibility; new code uses `from src.features import shared` to reach that same implementation. There is one feature registry and one set of formulas.

Development averages, last-fold measurements and holdout results have different populations and should not be merged into an unlabeled comparison. The holdout has now been evaluated; future model/policy improvements require new independent validation.
