"""Local fold artifacts and team-result export for train_random_forest.ipynb."""
from __future__ import annotations

import hashlib
import json
import platform
from datetime import date
from datetime import datetime
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import sklearn

from .safeflow_random_forest import make_model

LOCAL_DIR = Path(__file__).resolve().parent
FUTURE_COLUMNS = {"y", "y_speed", "y_speed_real", "y_known", "y_known_any", "onset", "window_end"}


def local_path(path):
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(LOCAL_DIR):
        raise ValueError(f"Local run artifacts must remain under {LOCAL_DIR}")
    return resolved


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def feature_hash(columns):
    return hashlib.sha256(json.dumps(list(columns), separators=(",", ":")).encode()).hexdigest()


def write_json(path, payload):
    """Atomic local metadata/checkpoint write."""
    path = local_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def _predictors(frame, rows, columns):
    # Apply the same nonfinite policy at fit and prediction time; never drop rows.
    return frame.iloc[rows][columns].replace([np.inf, -np.inf], np.nan).astype("float32")


def fit_fold(frame, feature_columns, train_rows, validation_rows, output_dir,
             *, model_parameters=None, save_model=True, prediction_chunk_size=100_000):
    """Fit a fresh pipeline and return probabilities aligned with validation_rows.

    Shared label construction, fold masks, training-only typical features, and
    scoring remain owned by safeflow_features.py and the notebook.
    """
    output_dir = local_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = list(feature_columns)
    if FUTURE_COLUMNS.intersection(columns):
        raise ValueError("Target/outcome columns cannot be predictors")
    train_rows = np.asarray(train_rows, dtype=np.int64)
    validation_rows = np.asarray(validation_rows, dtype=np.int64)
    for name, rows in [("train", train_rows), ("validation", validation_rows)]:
        if not len(rows) or len(np.unique(rows)) != len(rows):
            raise ValueError(f"{name} endpoints must be nonempty and unique")
        if rows.min() < 0 or rows.max() >= len(frame):
            raise ValueError(f"{name} endpoint outside the dataframe")
    if np.intersect1d(train_rows, validation_rows).size:
        raise ValueError("Training and validation endpoints overlap")
    if not frame.iloc[train_rows]["y_known"].all():
        raise ValueError("Training endpoints require measured future outcomes")
    target = frame.iloc[train_rows]["y"].to_numpy(dtype=np.int8)
    if np.unique(target).size != 2:
        raise ValueError("Training requires both target classes")
    start = perf_counter()
    model = make_model(columns, **(model_parameters or {}))
    print(f"  fitting {len(train_rows):,} rows / {len(columns)} logical features", flush=True)
    model.fit(_predictors(frame, train_rows, columns), target)
    fit_seconds = perf_counter() - start
    print(f"  fit complete in {fit_seconds:.1f}s; predicting {len(validation_rows):,} rows", flush=True)
    probabilities = np.empty(len(validation_rows), dtype=np.float64)
    for first in range(0, len(validation_rows), prediction_chunk_size):
        last = min(first + prediction_chunk_size, len(validation_rows))
        part = model.predict_proba(_predictors(frame, validation_rows[first:last], columns))
        positive_column = np.flatnonzero(model.named_steps["model"].classes_ == 1)[0]
        probabilities[first:last] = part[:, positive_column]
    if not np.isfinite(probabilities).all() or not ((probabilities >= 0) & (probabilities <= 1)).all():
        raise ValueError("Model returned invalid probabilities")
    encoded_columns = model.named_steps["prepare"].get_feature_names_out().tolist()
    importance = pd.DataFrame({"encoded_feature": encoded_columns,
                               "impurity_importance": model.named_steps["model"].feature_importances_})
    importance.sort_values("impurity_importance", ascending=False).to_csv(output_dir / "feature_importance.csv", index=False)
    write_json(output_dir / "preprocessing.json", dict(logical_features=columns, encoded_features=encoded_columns,
        numeric_medians=model.named_steps["prepare"].named_transformers_["numeric"].statistics_.tolist()))
    if save_model:
        joblib.dump(model, output_dir / "model.joblib", compress=3)
    return dict(probability=probabilities, positions=validation_rows.copy(), fit_seconds=fit_seconds,
                total_seconds=perf_counter()-start, encoded_features=len(encoded_columns))


def summary_row(model_name, fold_results, features):
    return {
        "model": model_name, "date run": date.today().isoformat(), "features": len(features),
        "AP onsets (mean)": fold_results["AP, onsets"].mean(),
        "AP onsets (std)": fold_results["AP, onsets"].std(),
        "AP onsets (worst fold)": fold_results["AP, onsets"].min(),
        "Brier": fold_results["Brier"].mean(),
        "caught ≥15 min @0.5": fold_results["caught ≥15 min"].mean(),
        "false alarms/day @0.5": fold_results["false alarms / day"].mean(),
        "wrong-alert hours/day @0.5": fold_results["wrong-alert hours / day"].mean(),
    }


def publish_team_results(project_dir, model_name, fold_results, features, manifest, *, quick_run):
    """Publish only a completed full run to the same location as the transformer."""
    if quick_run:
        print("Quick run: local artifacts retained; team summary unchanged.", flush=True)
        return None
    expected = manifest["expected_fold_names"]
    if list(fold_results.index) != expected or len(expected) != 6:
        raise ValueError("Only all six completed shared folds can enter the team table")
    directory = Path(project_dir).resolve() / "notebooks" / "results"
    directory.mkdir(parents=True, exist_ok=True)
    summary_file = directory / "summary.csv"
    existing = pd.read_csv(summary_file) if summary_file.exists() else pd.DataFrame()
    if len(existing):
        existing = existing.loc[existing["model"] != model_name]
    summary = pd.concat([existing, pd.DataFrame([summary_row(model_name, fold_results, features)])], ignore_index=True)
    summary = summary.sort_values("AP onsets (mean)", ascending=False)
    fold_results.round(4).to_csv(directory / f"{model_name}_folds.csv")
    (directory / f"{model_name}_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    temporary = summary_file.with_suffix(".csv.tmp")
    summary.round(4).to_csv(temporary, index=False)
    temporary.replace(summary_file)
    return summary


def versions():
    return dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__)


def load_dataset(shared, data_dir, *, use_cache=True):
    """Cache only shared, fold-independent development data with input fingerprints."""
    directory = Path(data_dir).resolve()
    inventory = []
    for path in sorted(directory.rglob("d12_text_station_5min_*.parquet")):
        day = datetime.strptime(path.stem[-10:], "%Y_%m_%d").date()
        if day >= pd.Timestamp(shared.HOLDOUT_START).date():
            continue  # never open or fingerprint holdout file contents
        stat = path.stat()
        inventory.append((str(path.relative_to(directory)), stat.st_size, stat.st_mtime_ns))
    signature = dict(data_dir=str(directory), inventory=inventory,
                     shared_code_sha256=sha256_file(shared.__file__),
                     metadata_sha256=sha256_file(directory / shared.META_FILE_NAME),
                     features=list(shared.SHARED_FEATURES), holdout_start=shared.HOLDOUT_START)
    digest = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    cache = LOCAL_DIR / "cache" / f"development_{digest[:16]}.parquet"
    if use_cache and cache.exists():
        print("Loading matching local development cache:", cache, flush=True)
        frame = pd.read_parquet(cache)
    else:
        print("Building shared development dataset; holdout excluded", flush=True)
        frame = shared.build_dataset(directory)
        if use_cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(cache, index=False)
    if frame.empty or frame["timestamp"].max() >= pd.Timestamp(shared.HOLDOUT_START):
        raise ValueError("Dataset is empty or includes holdout observations")
    if frame.duplicated(shared.KEY).any():
        raise ValueError("Duplicate station/timestamp keys in shared dataset")
    return frame, digest
