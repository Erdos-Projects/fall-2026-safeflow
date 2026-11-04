# Model decision analysis

Completed run: `20261104_104058_694019`. Includes the executed notebook, paired daily counts, 5,000-resample confidence intervals, a comparison figure, operational readiness/source audits and provenance. The earlier run preceded extension of the source audit.

Primary comparison: Transformer minus boosted trees under the frozen forward duration-budget policy. Weekly intervals preserve all stations and within-week observations; daily resampling is a sensitivity check. Intervals condition on existing fits and thresholds. Operational latency/memory/outage measurements await the original fitted checkpoints. No final holdout data was read.
