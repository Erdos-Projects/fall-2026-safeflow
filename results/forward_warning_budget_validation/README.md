# Forward warning-budget validation

Run `notebooks/analysis/forward_warning_budget_validation.ipynb` after the all-model error analysis has generated its local context cache. Completed run `20261104_102956_926783` includes the executed notebook, per-quarter policies, pooled metrics, recurrence tables and plots. The empty earlier run was interrupted before artifacts were produced.

Thresholds are selected on earlier OOF quarters and evaluated without adjustment on the next quarter. Calibration budgets are not guaranteed future costs. No final holdout data or model retraining is involved.
