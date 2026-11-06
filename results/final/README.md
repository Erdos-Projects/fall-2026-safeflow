# Final comparative-study evidence

`final` identifies the finalized evidence package, **not a selected model**. Random Forest, boosted trees and Transformer are all retained. Boosted trees and Transformer are leading candidates; neither frozen warning policy met the holdout target budget.

| Folder/file | Scope |
| --- | --- |
| `development/` | Published six-fold tables and historical forward/uncertainty evidence |
| `development/predictions/` | April-June calibration predictions from the current frozen checkpoints |
| `holdout/` | July-December frozen predictions, policy definitions, score tables and original provenance |
| `operational_benchmark/` | Completed fresh last-fold CPU measurements |
| `figures/` | Regenerated PR/ROC, reliability, warning-budget/error and operational plots |
| `recomputed_holdout_metrics.csv` | Independently regenerated scores verified against all nine frozen policy rows |
| `calibration.csv` | Descriptive equal-width reliability bins; no calibrator fitted to holdout |
| `manifest.json` | File hashes and source-run locations |
| `report_manifest.json` | Report-generation scope and score-verification status |

A **warning policy** is the rule for issuing an alert when predicted probability reaches a threshold. A **budget** is a target for false-warning burden used to select that threshold on earlier calibration data. It is not a hard future limit or a financial budget. Duration policies target at most 4 wrong-alert hours/day; episode policies target at most 20 false-warning episodes/day. Threshold 0.5 is a separate diagnostic rule. See [the full definitions](../../kpis.md) and [executive summary](../../presentation/summary.md).

Regenerate and verify this package's figures and holdout score tables without raw data or training:

```text
python scripts/generate_report.py
```

Run this from the repository root, or invoke the script by absolute path. Reproduction of raw feature construction and fitting is separate:

```text
python scripts/reproduce_results.py
```

That command writes a new workspace under ignored `results/reproductions/` and preserves the curated release and published team tables. It replays the pre-April fit, April-June calibration and already-used July-December test; it is not a new independent evaluation. Full CPU fitting can take hours. Historical six-fold tables remain provenance-backed snapshots: original historical fits, including the Transformer with a different feature-source hash, are not reconstructed by that command.

Original timestamped runs are preserved. Paths inside their copied manifests describe the original execution machine; the portable commands above use the packaged predictions, repository-relative configuration and artifact metadata. Large raw inputs/caches are not required for fast reporting and are not bundled here.

[Release verification](release_verification.json) records preservation of EDA, the executed showcase, checkpoint/feature reproduction, and byte-identical full-size prepared inputs. The multi-hour training replay was not repeated during packaging.
