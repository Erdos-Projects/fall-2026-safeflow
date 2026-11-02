"""Random Forest factory for the shared SafeFlow tabular feature contract."""
from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def make_model(feature_columns, *, n_estimators=300, max_depth=12,
               min_samples_leaf=10, max_samples=0.25, n_jobs=-1,
               random_state=42):
    """New training-only preprocessing and an unfitted probability forest.

    No class reweighting: probabilities are learned at the observed training
    prevalence, consistent with the shared unweighted probability task.
    Each tree bootstraps a quarter of the selected training rows to bound cost.
    """
    columns = list(feature_columns)
    if not columns or len(set(columns)) != len(columns):
        raise ValueError("Provide a nonempty, unique feature list")
    numeric = [column for column in columns if column != "dow"]
    transformers = [("numeric", SimpleImputer(strategy="median", add_indicator=True,
                                               keep_empty_features=True), numeric)]
    if "dow" in columns:
        transformers.append(("weekday", OneHotEncoder(drop="first", handle_unknown="ignore",
                                                      sparse_output=False, dtype=np.float32), ["dow"]))
    prepare = ColumnTransformer(transformers, remainder="drop", sparse_threshold=0)
    forest = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        min_samples_leaf=min_samples_leaf, max_features="sqrt",
        bootstrap=True, max_samples=max_samples, class_weight=None,
        n_jobs=n_jobs, random_state=random_state,
    )
    return Pipeline([("prepare", prepare), ("model", forest)])
