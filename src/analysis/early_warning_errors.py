"""Prediction alignment and exact budget sweeps for early-warning error analysis."""
from __future__ import annotations

import numpy as np
import pandas as pd

KEY = ["station", "timestamp"]
LABEL_COLUMNS = ["y", "y_known", "congested_now", "real", "onset"]


def align_predictions(reference, predictions, probability_column="probability"):
    """Require all reference rows and identical labels; align by keys, not row order."""
    required = KEY + LABEL_COLUMNS + [probability_column]
    missing = set(required) - set(predictions.columns)
    if missing:
        raise ValueError(f"Prediction file missing columns: {sorted(missing)}")
    pred = predictions[required].copy()
    pred["timestamp"] = pd.to_datetime(pred["timestamp"])
    if pred.duplicated(KEY).any() or reference.duplicated(KEY).any():
        raise ValueError("Station/timestamp keys must be unique")
    if len(pred) != len(reference):
        raise ValueError(f"Need every validation row: {len(reference):,}; got {len(pred):,}")
    aligned = pred.set_index(KEY).reindex(pd.MultiIndex.from_frame(reference[KEY]))
    if aligned.isna().any().any():
        raise ValueError("Prediction keys, labels, or probabilities are missing")
    for column in LABEL_COLUMNS:
        if not np.array_equal(aligned[column].to_numpy(), reference[column].to_numpy()):
            raise ValueError(f"Prediction labels disagree with reference: {column}")
    p = aligned[probability_column].to_numpy(dtype=float)
    if not np.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all():
        raise ValueError("Probabilities must be finite and within [0, 1]")
    return p


def warning_leads(frame, probability, onset_rows, threshold, window_minutes=30):
    """Same earliest-alert lead convention as safeflow_features.alert_metrics."""
    index = pd.MultiIndex.from_frame(frame[KEY])
    scores = pd.Series(np.asarray(probability, dtype=float), index=index)
    lead = np.zeros(len(onset_rows), dtype=np.int16)
    for minutes in range(5, window_minutes + 1, 5):
        wanted = pd.MultiIndex.from_arrays([onset_rows.station.to_numpy(),
                   (onset_rows.timestamp - pd.Timedelta(minutes=minutes)).to_numpy()])
        earlier = scores.reindex(wanted).to_numpy()
        lead = np.where(np.isfinite(earlier) & (earlier >= threshold), minutes, lead)
    return lead


class ExactWarningSweep:
    """Exact false episodes/duration and early-catch rates at score breakpoints.

    Wrong-alert rows form disjoint paths in station/time order. At threshold T,
    number of episodes = active vertices - active adjacent edges. An edge is
    active when both scores exceed T, so its activation score is min(p_i,p_j).
    This identity avoids a false monotonicity assumption for episode counts.
    """
    def __init__(self, frame, probability, *, min_lead=15, window_minutes=30):
        frame = frame.reset_index(drop=True)
        if frame.duplicated(KEY).any():
            raise ValueError("Duplicate validation keys")
        expected = frame.sort_values(KEY).index.to_numpy()
        if not np.array_equal(expected, np.arange(len(frame))):
            raise ValueError("Sort validation frame by station and timestamp first")
        p = np.asarray(probability, dtype=float)
        if len(p) != len(frame) or not np.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all():
            raise ValueError("Need one finite probability in [0,1] per validation row")
        self.days = frame.date.nunique()
        if self.days == 0:
            raise ValueError("No validation days")
        negative = (frame.y_known & ~frame.y).to_numpy(dtype=bool)
        same_station = frame.station.to_numpy()[1:] == frame.station.to_numpy()[:-1]
        adjacent_time = np.diff(frame.timestamp.to_numpy(dtype="datetime64[ns]")) == np.timedelta64(5, "m")
        edges = same_station & adjacent_time & negative[1:] & negative[:-1]
        self.vertex_scores = np.sort(p[negative])
        self.edge_scores = np.sort(np.minimum(p[1:], p[:-1])[edges])
        onsets = frame.loc[frame.onset, KEY]
        self.n_onsets = len(onsets)
        index = pd.MultiIndex.from_frame(frame[KEY])
        scores = pd.Series(p, index=index)
        best = np.full(self.n_onsets, -np.inf)
        for minutes in range(min_lead, window_minutes + 1, 5):
            wanted = pd.MultiIndex.from_arrays([onsets.station.to_numpy(),
                (onsets.timestamp - pd.Timedelta(minutes=minutes)).to_numpy()])
            q = scores.reindex(wanted).to_numpy()
            best = np.maximum(best, np.where(np.isfinite(q), q, -np.inf))
        self.early_scores = np.sort(best)

    def evaluate(self, thresholds):
        t = np.atleast_1d(np.asarray(thresholds, dtype=float))
        vertices = len(self.vertex_scores) - np.searchsorted(self.vertex_scores, t, side="left")
        edges = len(self.edge_scores) - np.searchsorted(self.edge_scores, t, side="left")
        caught = self.n_onsets - np.searchsorted(self.early_scores, t, side="left")
        return pd.DataFrame(dict(threshold=t, caught_count=caught,
            caught_15min=caught / self.n_onsets if self.n_onsets else np.full(len(t), np.nan),
            false_episodes_per_day=(vertices-edges) / self.days,
            wrong_alert_hours_per_day=vertices * 5 / 60 / self.days))

    def operating_points(self, episode_budget=20, hours_budget=4):
        # Include both equality and immediately-above cases for every breakpoint.
        values = np.unique(np.r_[0., 1., self.vertex_scores, self.early_scores[np.isfinite(self.early_scores)]])
        thresholds = np.unique(np.r_[values, np.nextafter(values, np.inf)])
        table = self.evaluate(thresholds)
        selected = []
        for name, column, budget in [("episodes", "false_episodes_per_day", episode_budget),
                                      ("duration", "wrong_alert_hours_per_day", hours_budget)]:
            eligible = table.loc[table[column] <= budget + 1e-12]
            # Optimize early detection globally, then reduce duration/episodes.
            best_count = eligible.caught_count.max()
            best = eligible.loc[eligible.caught_count.eq(best_count)].sort_values(
                ["wrong_alert_hours_per_day", "false_episodes_per_day", "threshold"],
                ascending=[True, True, False]).iloc[0].to_dict()
            selected.append(dict(budget_type=name, budget=budget, **best))
        positions = np.unique(np.linspace(0, len(table)-1, min(3000, len(table))).astype(int))
        return pd.DataFrame(selected), table.iloc[positions].reset_index(drop=True)


def forward_budget_validation(frame, probability, episode_budget=20, hours_budget=4):
    """Select on all earlier OOF quarters; freeze and score the next quarter.

    Budgets constrain calibration only. Realized future cost is never retuned.
    Each quarter is scored independently, including its available onset history.
    """
    if not frame.reset_index(drop=True).equals(frame.sort_values(KEY).reset_index(drop=True)):
        raise ValueError("Sort validation frame and probabilities together first")
    frame = frame.reset_index(drop=True)
    # Callers must supply probabilities in the original frame order: require sorted input.
    # Alignment is deliberately handled by align_predictions before this function.
    p = np.asarray(probability, dtype=float)
    quarters = sorted(frame.fold.unique())
    rows = []
    for i, quarter in enumerate(quarters[1:], 1):
        earlier = frame.fold.isin(quarters[:i]).to_numpy()
        later = frame.fold.eq(quarter).to_numpy()
        calibration, test = frame.loc[earlier], frame.loc[later]
        if calibration.timestamp.max() >= test.timestamp.min():
            raise ValueError("Calibration must strictly precede evaluation")
        policies, _ = ExactWarningSweep(calibration, p[earlier]).operating_points(episode_budget, hours_budget)
        evaluator = ExactWarningSweep(test, p[later])
        for policy in policies.to_dict("records"):
            measured = evaluator.evaluate([policy["threshold"]]).iloc[0].to_dict()
            cost = "false_episodes_per_day" if policy["budget_type"] == "episodes" else "wrong_alert_hours_per_day"
            rows.append(dict(fold=quarter, calibration_folds=",".join(map(str, quarters[:i])),
                calibration_end=str(calibration.timestamp.max()), test_start=str(test.timestamp.min()),
                budget_type=policy["budget_type"], budget=policy["budget"],
                calibration_catch=policy["caught_15min"], calibration_cost=policy[cost],
                days=evaluator.days, onsets=evaluator.n_onsets,
                budget_exceeded=bool(measured[cost] > policy["budget"] + 1e-12), **measured))
    return pd.DataFrame(rows)
