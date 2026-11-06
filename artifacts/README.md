# Retained model artifacts

All three candidates are retained in separate directories. There is no `final_model` alias and no selected winner.

Each bundle contains its trusted checkpoint, ordered feature schema, model configuration, frozen training-period typical-feature statistics, station/ramp mapping, threshold policies and provenance. RF and boosted trees use `model.joblib`; Transformer uses `model.pt` with architecture and preprocessing plus a portable loader. These are intentional per-model formats, not renamed file extensions.

```python
from src.analysis.study_release import CandidateBundle
bundle = CandidateBundle("artifacts/transformer")
probability = bundle.predict_features(engineered_frame, endpoint_rows)
```

Initialize the Transformer bundle before reading Arrow fixtures when using Windows to avoid native DLL import conflicts. `engineered_frame` must be sorted by station/time, follow the recorded feature schema, and retain the necessary history rows. The input is engineered features, not raw PeMS measurements. `apply_training_statistics` restores frozen typical/anomaly features from station/time lookups when the required current measurements are supplied. Raw causal feature construction is exposed through `src.features.shared`.

Joblib/PyTorch checkpoints must only be loaded from trusted sources. Checkpoint hashes are verified on load. Version pins are in [requirements-lock.txt](../requirements-lock.txt). These bundles reproduce the pre-April fits evaluated with April-June calibration and July-December testing; they are not refits through June. The original timestamped artifacts remain untouched as provenance.
