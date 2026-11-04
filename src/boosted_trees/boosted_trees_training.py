"""Reusable code for notebooks/train_boosted_trees.ipynb: the model, one fold's fit, run provenance and the team table."""
import hashlib
import inspect
import json
import platform
from datetime import datetime
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier

LABEL_COLUMNS = ["y", "y_known", "congested_now", "real", "onset"]
EXPORT_COLUMNS = ["station", "timestamp", "date", "fold", *LABEL_COLUMNS, "probability"]


# ---- Model -----------------------------------------------------------------------------------------------------------------
def make_model(**parameters):
    """A new, unfitted model. Handles blanks (NaN) natively, so no imputing or scaling. dow goes in as the number 0-4."""
    return HistGradientBoostingClassifier(**parameters)


def fit_fold(frame, features, train_rows, validation_rows, fold_dir, *, model_parameters, save_model=False):
    """
    Fit a new model on frame.iloc[train_rows] and return probabilities for frame.iloc[validation_rows], in that order.
    train_rows keep the order DataFrame.sample gives them (not sorted), which reproduces the original notebook run exactly.
    """
    fold_dir = Path(fold_dir)
    fold_dir.mkdir(parents=True, exist_ok=True)
    train = frame.iloc[train_rows]
    if not train["y_known"].all():
        raise ValueError("Training rows need measured future outcomes")
    if np.intersect1d(train_rows, validation_rows).size:
        raise ValueError("Training and validation rows overlap")

    start = perf_counter()
    model = make_model(**model_parameters).fit(train[features], train["y"])
    fit_seconds = perf_counter() - start
    probability = model.predict_proba(frame.iloc[validation_rows][features])[:, 1]
    if not (np.isfinite(probability).all() and ((probability >= 0) & (probability <= 1)).all()):
        raise ValueError("Model returned invalid probabilities")
    if save_model:
        joblib.dump(model, fold_dir / "model.joblib", compress=3)
    return dict(probability=probability.astype("float64"), fit_seconds=fit_seconds, n_iter=int(model.n_iter_))


def export_predictions(validation, probability, fold_name, path):
    """Write every validation row (unknown outcomes included) in the team's shared format."""
    export = validation[["station", "timestamp", "date", *LABEL_COLUMNS]].copy()
    export["fold"] = fold_name
    export["probability"] = probability
    export = export[EXPORT_COLUMNS].sort_values(["station", "timestamp"]).reset_index(drop=True)
    export[LABEL_COLUMNS] = export[LABEL_COLUMNS].astype(bool)
    export.to_parquet(path, index=False)
    return export


# ---- Provenance ------------------------------------------------------------------------------------------------------------
def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def feature_hash(features):
    return hashlib.sha256(json.dumps(list(features), separators=(",", ":")).encode()).hexdigest()


def versions():
    return dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__)


def data_inventory(shared, data_dir):
    """Content hashes of the development data files. Holdout files are skipped by their date, never opened."""
    holdout = pd.Timestamp(shared.HOLDOUT_START).date()
    files = []
    for path in sorted(Path(data_dir).rglob("d12_text_station_5min_*.parquet"), key=lambda p: p.name):
        if datetime.strptime(path.stem[-10:], "%Y_%m_%d").date() >= holdout:
            continue
        files.append([path.name, path.stat().st_size, sha256_file(path)])
    digest = hashlib.sha256(json.dumps(files, separators=(",", ":")).encode()).hexdigest()
    return dict(files=len(files), first=files[0][0], last=files[-1][0], sha256=digest,
                metadata_sha256=sha256_file(Path(data_dir) / shared.META_FILE_NAME))


def rows_fingerprint(export):
    """Hash of the validation rows and labels, so two models' exports can be checked for identical rows."""
    keyed = export[["station", "timestamp", *LABEL_COLUMNS]].sort_values(["station", "timestamp"])
    return hashlib.sha256(pd.util.hash_pandas_object(keyed, index=False).to_numpy().tobytes()).hexdigest()


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def factory_source():
    return inspect.getsource(make_model)


# ---- Team table ------------------------------------------------------------------------------------------------------------
def summary_row(model_name, fold_results, features):
    return {
        "model": model_name,
        "date run": datetime.now().date().isoformat(),
        "features": len(features),
        "AP onsets (mean)": fold_results["AP, onsets"].mean(),
        "AP onsets (std)": fold_results["AP, onsets"].std(),
        "AP onsets (worst fold)": fold_results["AP, onsets"].min(),
        "Brier": fold_results["Brier"].mean(),
        "caught ≥15 min @0.5": fold_results["caught ≥15 min"].mean(),
        "false alarms/day @0.5": fold_results["false alarms / day"].mean(),
        "wrong-alert hours/day @0.5": fold_results["wrong-alert hours / day"].mean(),
    }


def publish_team_results(results_dir, model_name, fold_results, metric_columns, features, manifest):
    """
    Write <model>_folds.csv and <model>_manifest.json, and replace this model's row in summary.csv. Other models' rows are kept.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    if len(fold_results) != 6 or manifest.get("status") != "complete":
        raise ValueError("Only a complete six-fold run can enter the team table")
    fold_results[metric_columns].round(4).to_csv(results_dir / f"{model_name}_folds.csv")
    write_json(results_dir / f"{model_name}_manifest.json", manifest)
    summary_file = results_dir / "summary.csv"
    summary = pd.read_csv(summary_file) if summary_file.exists() else pd.DataFrame()
    if not summary.empty:
        summary = summary.loc[summary["model"] != model_name]
    summary = pd.concat([summary, pd.DataFrame([summary_row(model_name, fold_results, features)])], ignore_index=True)
    summary = summary.sort_values("AP onsets (mean)", ascending=False)
    summary.round(4).to_csv(summary_file, index=False)
    return summary
