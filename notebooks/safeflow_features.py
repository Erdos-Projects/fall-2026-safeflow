"""
Shared features for the I-5 northbound (District 12) congestion early-warning project.

Question being predicted, for every mainline station every 5 minutes:
    "Will this station be congested (measured speed < 40 mph) at any point in the next 30 minutes?"

Typical use (train_and_evaluate.ipynb does all of this for a model):

    import safeflow_features as sf

    df = sf.build_dataset()                        # loads Jan 2023 - Jun 2025 (the Jul-Dec 2025 holdout is never read)
    for fold in sf.folds(df):                      # six quarterly walk-forward folds
        sf.add_typical_features(df, fold.train)    # REQUIRED once per fold, before using the features
        train = df[fold.train & df["y_known"]]     # train only on measured outcomes
        val = df[fold.val]                         # predict on ALL validation rows
        model.fit(train[sf.SHARED_FEATURES], train["y"])
        probability = model.predict_proba(val[sf.SHARED_FEATURES])[:, 1]
        print(fold.name, sf.score(val, probability))

Rules
- Call add_typical_features(df, train_mask) once per fold. Those features summarize the training days only. Computing them on
  all the data would leak the validation period into the features.
- Train on rows with y_known (all six future readings measured). Predict on every validation row. score() picks the scored rows.
- Never use the July-December 2025 holdout until the final evaluation.

Adding a feature to the shared set
- Add a function to the TEAM FEATURES section at the bottom of this file, with @feature (computed once when the dataset is built)
  or @training_feature (recomputed in every fold). Its name is added to SHARED_FEATURES automatically, so everyone gets it on
  their next build_dataset(). Instructions and an example are in the TEAM FEATURES section.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

# ---- Settings (shared by the whole team, change only together) ----------------------------------------------------------
META_FILE_NAME = "d12_text_meta_2021_05_18.txt"        # station metadata, inside the data folder
HOLDOUT_START = "2025-07-01"                           # files on or after this date are never read
HOUR_START, HOUR_END = 6, 19                           # keep readings from 6:00 to 19:55
WINDOW_MIN = 30                                        # "congested at any point in the next 30 minutes"
SLOW_MPH = 40                                          # congested = measured speed below this
ONSET_CLEAR_MIN = 30                                   # an onset = congested now, after at least this long uncongested
DOWNSTREAM_MILES = 5.0                                 # how far ahead the downstream-jam features look
ALERT_THRESHOLD = 0.5                                  # default alert threshold used by score()
MIN_LEAD_MIN = 15                                      # an onset counts as "caught" if an alert came at least this early

TRAIN_START = "2023-01-01"
VALIDATION_QUARTERS = [
    ("2024-01-01", "2024-04-01"),
    ("2024-04-01", "2024-07-01"),
    ("2024-07-01", "2024-10-01"),
    ("2024-10-01", "2025-01-01"),
    ("2025-01-01", "2025-04-01"),
    ("2025-04-01", "2025-07-01"),
]
GAP_DAYS = 7                                           # days left out between training and each validation quarter

# Freeway-to-freeway connectors: is each one an exit from I-5 or an entrance onto it?
CONNECTOR_ROLES = {
    1208990: "exit", 1209013: "exit", 1209012: "entrance", 1210020: "exit",
    1205315: "exit", 1212015: "entrance", 1213419: "exit",
}

# ---- Feature lists ------------------------------------------------------------------------------------------------------
# The feature set the whole team uses. Team features (bottom of this file) are appended to it automatically.
SHARED_FEATURES = [
    "speed_anomaly",            # speed - this station's typical speed at this time of day            (needs add_typical_features)
    "occ_anomaly",              # occupancy - its typical value (occupancy is a fraction, 0-1)        (needs add_typical_features)
    "typical_congestion_30",    # share of training days congested within 30 min, this station & time (needs add_typical_features)
    "speed_delta_2",            # speed now - speed 10 minutes ago
    "flow",                     # vehicles per 5 minutes, all lanes
    "down_minus_here",          # next downstream station's speed - this station's speed
    "up_minus_here",            # next upstream station's speed - this station's speed
    "tod_sin", "tod_cos",       # time of day on a 24-hour circle
    "dow",                      # day of week, 0 = Monday (one-hot encode it for linear models)
    "pm",                       # absolute postmile
    "onramp_flow_up",           # flow on the nearest on-ramp(s) at or before the station
    "offramp_flow_down",        # flow on the nearest off-ramp(s) at or after the station
    "jam_distance_down",        # miles to the nearest congested station within DOWNSTREAM_MILES downstream (DOWNSTREAM_MILES if none)
    "min_speed_down_anomaly",   # lowest speed within DOWNSTREAM_MILES downstream - its typical value  (needs add_typical_features)
    "jam_distance_change_10",   # jam_distance_down now - 10 minutes ago (negative = a jam is getting closer)
]

KEY = ["station", "timestamp"]
FIVE_MIN = pd.Timedelta(minutes=5)
STEPS_AHEAD = WINDOW_MIN // 5

# Raw per-lane columns available in the station files. A team feature that needs them adds them to EXTRA_COLUMNS.
LANE_SPEED_COLUMNS = [f"lane{i}_speed" for i in range(1, 9)]
LANE_FLOW_COLUMNS = [f"lane{i}_flow" for i in range(1, 9)]
LANE_OCCUPANCY_COLUMNS = [f"lane{i}_occupancy" for i in range(1, 9)]
EXTRA_COLUMNS: list[str] = []                   # raw columns loaded in addition to speed, flow and occupancy

# Columns that look into the future (used by the optional check_no_lookahead helper)
FUTURE_COLUMNS = ["y_speed", "y_speed_real", "y", "y_known", "y_known_any", "onset", "window_end"]


# ---- Registry for team features -----------------------------------------------------------------------------------------
_LAST_CONTEXT = None                            # stations, ramps and ramp readings from the last build_dataset() call
STATIC_FEATURES: dict[str, Callable] = {}       # name -> function(df, context), computed once in build_dataset()
TRAINING_FEATURES: dict[str, Callable] = {}     # name -> function(df, train_mask), recomputed per fold in add_typical_features()


def feature(name):
    """
    Register a feature computed once, at the end of build_dataset(), and add it to SHARED_FEATURES. The function receives the
    full df and a context with `stations`, `nearest_ramps` and `ramp_readings`, and returns the new column.
    """
    def register(function):
        STATIC_FEATURES[name] = function
        if name not in SHARED_FEATURES:
            SHARED_FEATURES.append(name)
        return function
    return register


def training_feature(name):
    """
    Register a feature that depends on the fold, and add it to SHARED_FEATURES. The function receives df and the fold's
    train_mask and returns the new column.
    It is recomputed by add_typical_features() in every fold. typical_value() is available for per-station, per-time averages.
    """
    def register(function):
        TRAINING_FEATURES[name] = function
        if name not in SHARED_FEATURES:
            SHARED_FEATURES.append(name)
        return function
    return register


# ---- Loading -----------------------------------------------------------------------------------------------------------
def find_data_dir(data_dir=None) -> Path:
    """The given folder, or else a folder named "Data" in the current folder or any parent folder."""
    if data_dir is not None:
        return Path(data_dir).expanduser()
    here = Path.cwd()
    for folder in [here, *here.parents]:
        if (folder / "Data").is_dir():
            return folder / "Data"
    return here / "Data"


def _load_readings(data_dir: Path, verbose: bool, extra_columns=()):
    columns = ["timestamp", "station", "lane_type", "speed", "flow", "occupancy", "pct_observed"] + list(extra_columns)
    files = sorted(data_dir.rglob("d12_text_station_5min_*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No station files (d12_text_station_5min_*.parquet) found in {data_dir.resolve()}. The current folder is "
            f"{Path.cwd()}. Pass data_dir= the folder that contains 'D12 2023', 'D12 2024' and 'D12 2025'.")
    holidays = set(USFederalHolidayCalendar().holidays("2015-01-01", "2035-12-31").date)
    holdout_start = pd.Timestamp(HOLDOUT_START).date()

    mainline_parts, ramp_parts = [], []
    days_used, skipped = 0, 0
    start = time.time()
    for i, path in enumerate(files, 1):
        day = datetime.strptime(path.stem[-10:], "%Y_%m_%d").date()
        if day >= holdout_start:
            skipped += 1                               # never opened
            continue
        if day.weekday() >= 5 or day in holidays:
            continue
        readings = pd.read_parquet(path, columns=columns)
        readings = readings[readings["timestamp"].dt.hour.between(HOUR_START, HOUR_END)]
        is_ramp = readings["lane_type"].isin(["OR", "FR"]) | readings["station"].isin(CONNECTOR_ROLES)
        mainline_parts.append(readings[readings["lane_type"] == "ML"].drop(columns="lane_type"))
        ramp_parts.append(readings[is_ramp].drop(columns=["lane_type", *extra_columns]))
        days_used += 1
        if verbose and i % 100 == 0:
            print(f"read {i}/{len(files)} files ({time.time() - start:.0f}s)")

    def combine(parts):
        combined = pd.concat(parts, ignore_index=True)
        for column in ["speed", "flow", "occupancy"]:
            combined[column] = combined[column].astype("float32")
        combined["real"] = combined["pct_observed"] >= 100       # measured, not filled in by PeMS
        return combined.drop(columns="pct_observed")

    mainline, ramps = combine(mainline_parts), combine(ramp_parts)
    assert mainline["timestamp"].max() < pd.Timestamp(HOLDOUT_START), "holdout data was loaded"
    if verbose:
        print(f"{days_used} weekdays used out of {len(files)} files ({skipped} holdout files skipped, never read)")
    return mainline, ramps


def _stations_and_ramps(meta_file: Path, mainline: pd.DataFrame):
    meta = pd.read_csv(meta_file, sep="\t")
    meta = meta[(meta["Fwy"] == 5) & meta["Dir"].isin(["N", "S"])].copy()
    meta["Abs_PM"] = pd.to_numeric(meta["Abs_PM"], errors="coerce")
    meta = meta.dropna(subset=["Abs_PM"])
    meta["role"] = meta["Type"].map({"OR": "entrance", "FR": "exit"})
    is_connector = meta["Type"] == "FF"
    meta.loc[is_connector, "role"] = meta.loc[is_connector, "ID"].map(CONNECTOR_ROLES)
    meta["travel_pm"] = np.where(meta["Dir"] == "N", meta["Abs_PM"], -meta["Abs_PM"])    # increases along travel

    stations = meta[(meta["Type"] == "ML") & meta["ID"].isin(mainline["station"].unique())]
    stations = stations.sort_values(["Dir", "travel_pm"]).copy()
    stations["upstream_id"] = stations.groupby("Dir")["ID"].shift(1)
    stations["downstream_id"] = stations.groupby("Dir")["ID"].shift(-1)

    nearest = []
    for _, station in stations.iterrows():
        road = meta[meta["Dir"] == station["Dir"]]
        before = road[(road["role"] == "entrance") & (road["travel_pm"] <= station["travel_pm"])]
        after = road[(road["role"] == "exit") & (road["travel_pm"] >= station["travel_pm"])]
        nearest.append({"station": station["ID"],
                        "on_ids": sorted(before.loc[before["travel_pm"] == before["travel_pm"].max(), "ID"]),
                        "off_ids": sorted(after.loc[after["travel_pm"] == after["travel_pm"].min(), "ID"])})
    return stations, pd.DataFrame(nearest)


def _look_up(table, stations, times):
    """Rows of `table` (indexed by station and timestamp) at the given stations and times (NaN where there is no reading)."""
    return table.reindex(pd.MultiIndex.from_arrays([stations, times]))


# ---- Building the dataset ------------------------------------------------------------------------------------------------
def build_dataset(data_dir=None, verbose=True, extra_columns=None) -> pd.DataFrame:
    """
    Load the development data (holdout excluded) and build every feature that doesn't depend on the training period,
    plus the target, plus any features registered with @feature. Returns one row per mainline station per 5 minutes,
    sorted by station then time.

    extra_columns: raw columns to keep from the station files (default: EXTRA_COLUMNS), e.g. LANE_SPEED_COLUMNS.
    Target columns: y (congested within 30 min), y_known (all six future readings measured, train and score on these),
    congested_now, onset (a jam starting after 30+ measured uncongested minutes), real (current reading measured).
    """
    data_dir = find_data_dir(data_dir)
    meta_file = data_dir / META_FILE_NAME
    if not meta_file.exists():
        raise FileNotFoundError(f"Station metadata not found: {meta_file.resolve()}. Put {META_FILE_NAME} in {data_dir}.")
    extra_columns = list(EXTRA_COLUMNS if extra_columns is None else extra_columns)
    mainline, ramp_readings = _load_readings(data_dir, verbose, extra_columns)
    stations, nearest_ramps = _stations_and_ramps(meta_file, mainline)

    df = mainline[KEY + ["speed", "occupancy", "flow", "real"] + extra_columns].sort_values(KEY).reset_index(drop=True)
    by_key = df.set_index(KEY)

    def same_station_at(columns, minutes):
        later = df["timestamp"] + pd.Timedelta(minutes=minutes)
        return _look_up(by_key[columns], df["station"], later).reset_index(drop=True)

    # Recent change in speed
    df["speed_delta_2"] = df["speed"] - same_station_at(["speed"], -10)["speed"].to_numpy()

    # Ramp flows, summed over tied nearest ramps
    ramp_flow = ramp_readings[KEY + ["flow"]].rename(columns={"station": "ramp_id"})
    for list_column, name in [("on_ids", "onramp_flow_up"), ("off_ids", "offramp_flow_down")]:
        pairs = nearest_ramps[["station", list_column]].explode(list_column).dropna()
        pairs = pairs.rename(columns={list_column: "ramp_id"})
        pairs["n"] = pairs.groupby("station").cumcount() + 1
        per_ramp = []
        for n in range(1, pairs["n"].max() + 1):
            matched = (df[KEY]
                       .merge(pairs.loc[pairs["n"] == n, ["station", "ramp_id"]], on="station", how="left")
                       .merge(ramp_flow, on=["ramp_id", "timestamp"], how="left"))
            per_ramp.append(matched["flow"].to_numpy())
        df[name] = pd.DataFrame(np.column_stack(per_ramp)).sum(axis=1, min_count=1).to_numpy()

    # Neighboring stations' current speed and occupancy
    for id_column, side in [("upstream_id", "upstream"), ("downstream_id", "downstream")]:
        neighbor_of = stations[["ID", id_column]].rename(columns={"ID": "station", id_column: "neighbor"})
        readings = mainline[KEY + ["speed", "occupancy"]].rename(
            columns={"station": "neighbor", "speed": f"{side}_speed", "occupancy": f"{side}_occ"})
        matched = (df[KEY].merge(neighbor_of, on="station", how="left")
                   .merge(readings, on=["neighbor", "timestamp"], how="left"))
        df[f"{side}_speed"] = matched[f"{side}_speed"].to_numpy()
        df[f"{side}_occ"] = matched[f"{side}_occ"].to_numpy()

    # When and where
    df["tod"] = (df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute).astype("int16")
    df["dow"] = df["timestamp"].dt.dayofweek.astype("int8")
    df["pm"] = df["station"].map(stations.set_index("ID")["Abs_PM"]).astype("float32")
    df["down_minus_here"] = (df["downstream_speed"] - df["speed"]).astype("float32")
    df["up_minus_here"] = (df["upstream_speed"] - df["speed"]).astype("float32")
    angle = 2 * np.pi * df["tod"] / (24 * 60)
    df["tod_sin"] = np.sin(angle).astype("float32")
    df["tod_cos"] = np.cos(angle).astype("float32")

    _add_downstream_features(df, stations)
    df["date"] = df["timestamp"].dt.normalize()

    # The target, and the speed 30 minutes ahead (used for typical_in_30)
    ahead = same_station_at(["speed", "real"], WINDOW_MIN)
    df["y_speed"] = ahead["speed"].to_numpy()
    df["y_speed_real"] = ahead["real"].fillna(False).astype(bool).to_numpy()
    _add_labels(df)
    df["window_end"] = df["timestamp"] + pd.Timedelta(minutes=WINDOW_MIN)

    # Team features registered with @feature
    context = SimpleNamespace(stations=stations, nearest_ramps=nearest_ramps, ramp_readings=ramp_readings)
    for name, function in STATIC_FEATURES.items():
        df[name] = _as_column(function(df, context))
        if verbose:
            print(f"added team feature {name}")
    global _LAST_CONTEXT
    _LAST_CONTEXT = context

    if verbose:
        known = df["y_known"]
        print(f"{len(df):,} rows | {df['station'].nunique()} stations | scored rows {known.mean():.0%} | "
              f"onsets {df['onset'].sum():,}")
    return df


def _add_downstream_features(df, stations):
    speed_wide = df.pivot(index="timestamp", columns="station", values="speed")
    speed_wide_10_ago = speed_wide.reindex(speed_wide.index - pd.Timedelta(minutes=10)).set_axis(speed_wide.index)
    min_speed = pd.DataFrame(np.nan, index=speed_wide.index, columns=speed_wide.columns)
    distance_now, distance_before = min_speed.copy(), min_speed.copy()

    def miles_to_first_jam(speeds, miles):
        jammed = speeds.to_numpy() < SLOW_MPH
        return np.where(jammed.any(axis=1), miles[jammed.argmax(axis=1)], DOWNSTREAM_MILES)

    for _, road in stations.groupby("Dir"):
        ids, positions = road["ID"].to_numpy(), road["travel_pm"].to_numpy()
        for i, station in enumerate(ids):
            ahead = (positions > positions[i]) & (positions - positions[i] <= DOWNSTREAM_MILES)
            if not ahead.any():
                continue                                       # last station: nothing downstream in the data
            down_ids, miles = ids[ahead], positions[ahead] - positions[i]
            min_speed[station] = speed_wide[down_ids].min(axis=1).to_numpy()
            distance_now[station] = miles_to_first_jam(speed_wide[down_ids], miles)
            distance_before[station] = miles_to_first_jam(speed_wide_10_ago[down_ids], miles)

    def to_rows(wide):
        long = wide.stack(future_stack=True).swaplevel()
        return _look_up(long, df["station"], df["timestamp"]).to_numpy().astype("float32")

    df["min_speed_down"] = to_rows(min_speed)
    df["jam_distance_down"] = to_rows(distance_now)
    df["jam_distance_change_10"] = df["jam_distance_down"] - to_rows(distance_before)


def _add_labels(df):
    status = pd.DataFrame({"congested": (df["speed"] < SLOW_MPH).astype(float), "measured": df["real"].astype(float)})
    status.index = pd.MultiIndex.from_frame(df[KEY])

    def status_at(steps):
        found = _look_up(status, df["station"], df["timestamp"] + steps * FIVE_MIN)
        return found["congested"].to_numpy(), found["measured"].to_numpy()

    any_congested = np.zeros(len(df))
    all_exist = np.ones(len(df), dtype=bool)
    all_measured = np.ones(len(df), dtype=bool)
    for step in range(1, STEPS_AHEAD + 1):
        congested, measured = status_at(step)
        all_exist &= ~np.isnan(congested)
        all_measured &= measured == 1
        any_congested = np.maximum(any_congested, np.nan_to_num(congested))

    clear_before = np.ones(len(df), dtype=bool)
    for step in range(1, ONSET_CLEAR_MIN // 5 + 1):
        congested, measured = status_at(-step)
        clear_before &= (measured == 1) & (congested == 0)

    now = (df["speed"] < SLOW_MPH).to_numpy()
    df["congested_now"] = now
    df["y"] = any_congested.astype(bool)
    df["y_known"] = all_exist & all_measured
    df["y_known_any"] = all_exist
    df["onset"] = now & df["real"].to_numpy() & clear_before


# ---- Features that depend on the training period ------------------------------------------------------------------------
def _leave_one_day_out_mean(values, usable, train_mask, keys):
    """Average of `values` over training days, per (station, time of day). Training rows get the average without their own day."""
    counted = (train_mask & usable).to_numpy()
    counted_values = np.where(counted, values.astype(float).to_numpy(), 0.0)
    groups = pd.MultiIndex.from_arrays(keys)

    def group_total(x):
        return pd.Series(x, index=groups).groupby(level=[0, 1]).sum().reindex(groups).to_numpy()

    total, count = group_total(counted_values), group_total(counted.astype(float))
    in_training = train_mask.to_numpy()
    total = np.where(in_training, total - counted_values, total)
    count = np.where(in_training, count - counted, count)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(count > 0, total / count, np.nan)
    return mean.astype("float32")


def add_typical_features(df, train_mask):
    """
    Add the features that summarize the training period: typical_in_30, typical_congestion_30, speed_anomaly, occ_anomaly,
    min_speed_down_anomaly. Call once per fold, with that fold's training mask, before using any of these features.
    """
    same_time = [df["station"], df["tod"]]
    df["typical_in_30"] = _leave_one_day_out_mean(df["y_speed"], df["y_speed_real"], train_mask,
                                                  [df["station"], df["tod"] + WINDOW_MIN])
    df["typical_congestion_30"] = _leave_one_day_out_mean(df["y"], df["y_known"], train_mask, same_time)
    typical_speed = _leave_one_day_out_mean(df["speed"], df["real"], train_mask, same_time)
    typical_occ = _leave_one_day_out_mean(df["occupancy"], df["real"], train_mask, same_time)
    typical_min_down = _leave_one_day_out_mean(df["min_speed_down"], df["min_speed_down"].notna(), train_mask, same_time)
    df["speed_anomaly"] = (df["speed"] - typical_speed).astype("float32")
    df["occ_anomaly"] = (df["occupancy"] - typical_occ).astype("float32")
    df["min_speed_down_anomaly"] = (df["min_speed_down"] - typical_min_down).astype("float32")
    for name, function in TRAINING_FEATURES.items():          # team features registered with @training_feature
        df[name] = _as_column(function(df, train_mask))


# ---- Helpers for writing features -----------------------------------------------------------------------------------------
def _as_column(values):
    return values.to_numpy() if hasattr(values, "to_numpy") else values


def same_station_at(df, column, minutes):
    """Values of `column` at the same station `minutes` later (negative = earlier), lined up with df's rows."""
    table = df.set_index(KEY)[column]
    return _look_up(table, df["station"], df["timestamp"] + pd.Timedelta(minutes=minutes)).to_numpy()


def typical_value(df, column, train_mask, usable=None):
    """
    This station's average of `column` at this time of day over the training days (training rows leave out their own day).
    `usable`: which rows may enter the average (default: measured readings with a value).
    """
    values = df[column]
    if usable is None:
        usable = df["real"] & values.notna()
    return _leave_one_day_out_mean(values, usable, train_mask, [df["station"], df["tod"]])


# ---- Folds ---------------------------------------------------------------------------------------------------------------
@dataclass
class Fold:
    name: str                   # e.g. "2024-01 to 2024-03"
    train: pd.Series            # boolean mask over df rows
    val: pd.Series              # boolean mask over df rows


def training_mask(df, validation_start):
    """Everything from TRAIN_START until GAP_DAYS before validation_start (the whole 30-minute window included)."""
    cutoff = pd.Timestamp(validation_start) - pd.Timedelta(days=GAP_DAYS)
    return (df["timestamp"] >= TRAIN_START) & (df["window_end"] < cutoff)


def folds(df, only_last=False):
    """The six expanding-window validation folds (or just the last one, for quick checks)."""
    quarters = VALIDATION_QUARTERS[-1:] if only_last else VALIDATION_QUARTERS
    for start, end in quarters:
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        name = f"{start:%Y-%m} to {end - pd.Timedelta(days=1):%Y-%m}"
        yield Fold(name, training_mask(df, start), (df["timestamp"] >= start) & (df["window_end"] < end))


# ---- Checking a new feature ---------------------------------------------------------------------------------------------------
def check_no_lookahead(df, name, n_checks=2, context=None):
    """
    Optional: report whether a feature registered with @feature changes when every reading after a cutoff time is removed,
    i.e. whether it uses readings from after the forecast time.
    """
    function = STATIC_FEATURES[name]
    context = context or _LAST_CONTEXT
    days = df["date"].drop_duplicates().sort_values().to_numpy()
    for day in days[np.linspace(len(days) // 4, 3 * len(days) // 4, n_checks).astype(int)]:
        cutoff = pd.Timestamp(day) + pd.Timedelta(hours=16)
        past = df[df["timestamp"] <= cutoff].drop(columns=FUTURE_COLUMNS + [name], errors="ignore").reset_index(drop=True)
        recomputed = np.asarray(function(past, context), dtype=float)
        rows = (df["timestamp"] <= cutoff) & (df["date"] == pd.Timestamp(day))
        full = df.loc[rows, name].to_numpy(dtype=float)
        check = recomputed[(past["date"] == pd.Timestamp(day)).to_numpy()]
        if not np.allclose(full, check, equal_nan=True, atol=1e-4):
            print(f"{name!r} changes when readings after {cutoff} are removed: it uses readings from after the forecast time")
            return False
    print(f"{name!r}: unchanged when later readings are removed ({n_checks} cutoffs checked)")
    return True


def correlation_report(df, features, train_mask, new=None):
    """Largest correlation of each feature (or of each `new` one) with the others, and each feature's VIF, on training rows."""
    X = df.loc[train_mask, features]
    pearson = X.corr().abs()
    pearson = pearson.where(~np.eye(len(features), dtype=bool))
    complete = X.dropna()
    complete = complete.sample(min(len(complete), 300_000), random_state=0).to_numpy(float)
    vif = np.diag(np.linalg.inv(np.corrcoef(complete, rowvar=False)))
    report = pd.DataFrame({"largest |Pearson|": pearson.max(), "with": pearson.idxmax(), "VIF": vif}, index=features)
    report = report.loc[new] if new else report.sort_values("largest |Pearson|", ascending=False)
    return report.round(2)


def ap_onsets(val, probability):
    """Average precision on validation rows that are scored and not congested now (the main development metric)."""
    from sklearn.metrics import average_precision_score
    probability = np.asarray(probability)
    rows = (val["y_known"] & ~val["congested_now"] & val["real"]).to_numpy()
    return average_precision_score(val["y"].to_numpy()[rows], probability[rows])


def compare_feature_sets(df, feature_sets, make_model, only_last=False, train_sample=2_000_000):
    """
    Fit the same model with each feature set in every fold and compare AP on onsets. The first set is the reference.
    make_model: a function that takes a feature list and returns a new, unfitted model (with fit and predict_proba).
    """
    make = make_model

    rows = []
    for fold in folds(df, only_last):
        add_typical_features(df, fold.train)
        train = df[fold.train & df["y_known"]]
        if train_sample and len(train) > train_sample:
            train = train.sample(train_sample, random_state=0)
        val = df[fold.val]
        for name, features in feature_sets.items():
            fitted = make(features).fit(train[features], train["y"])
            rows.append({"fold": fold.name, "set": name, "AP, onsets": ap_onsets(val, fitted.predict_proba(val[features])[:, 1])})
    table = pd.DataFrame(rows).pivot(index="fold", columns="set", values="AP, onsets")[list(feature_sets)]
    reference = list(feature_sets)[0]
    for name in list(feature_sets)[1:]:
        difference = table[name] - table[reference]
        print(f"{name} vs {reference}: mean {difference.mean():+.4f}, better in {(difference > 0).sum()} of {len(table)} folds")
    return table.round(4)


# ---- Scoring ----------------------------------------------------------------------------------------------------------------
def _shifted_flags(flags, rows, minutes):
    found = _look_up(flags, rows["station"], rows["timestamp"] + pd.Timedelta(minutes=minutes))
    return found.fillna(False).astype(bool).to_numpy()


def alert_metrics(val, probability, threshold=ALERT_THRESHOLD):
    """How the map would perform alerting whenever probability >= threshold. `val`: all validation rows (df[fold.val])."""
    alert = pd.Series(np.asarray(probability) >= threshold, index=pd.MultiIndex.from_frame(val[KEY]))

    onsets = val[val["onset"]]
    lead_minutes = np.zeros(len(onsets))
    for step in range(1, STEPS_AHEAD + 1):
        lead_minutes = np.where(_shifted_flags(alert, onsets, -5 * step), 5 * step, lead_minutes)
    caught = (lead_minutes >= MIN_LEAD_MIN).mean() if len(onsets) else np.nan
    median_lead = np.median(lead_minutes[lead_minutes > 0]) if (lead_minutes > 0).any() else np.nan

    known = val["y_known"].to_numpy()
    false_alarm = pd.Series(alert.to_numpy() & known & ~val["y"].to_numpy(), index=alert.index)
    episodes = (false_alarm.to_numpy() & ~_shifted_flags(false_alarm, val, -5)).sum()
    n_false = false_alarm.to_numpy().sum()
    n_known_alerts = (alert.to_numpy() & known).sum()
    days = val["date"].nunique()
    return {
        "caught ≥15 min": caught,
        "false alarms / day": episodes / days,
        "wrong-alert hours / day": n_false * 5 / 60 / days,
        "alerts right": 1 - n_false / n_known_alerts if n_known_alerts else np.nan,
        "median lead (min)": median_lead,
        "onsets": len(onsets),
    }


def score(val, probability, threshold=ALERT_THRESHOLD):
    """
    Score probabilities for ALL validation rows (in df order). "AP, onsets" is the main development metric: average precision
    on stations not congested now (random guessing scores about 0.05).
    """
    from sklearn.metrics import average_precision_score, brier_score_loss

    probability = np.asarray(probability)
    if len(probability) != len(val):
        raise ValueError(f"need one probability per validation row ({len(val):,}), got {len(probability):,}")
    known = val["y_known"].to_numpy()
    y, p = val["y"].to_numpy()[known], probability[known]
    not_congested_now = (~val["congested_now"].to_numpy() & val["real"].to_numpy())[known]
    return {
        "AP, onsets": average_precision_score(y[not_congested_now], p[not_congested_now]),
        "AP, all": average_precision_score(y, p),
        "Brier": brier_score_loss(y, p),
        **alert_metrics(val, probability, threshold),
    }


# ---- TEAM FEATURES ----------------------------------------------------------------------------------------------------------
# Add shared features here. Each one is added to SHARED_FEATURES automatically.
#
#   @feature("name")             computed once when the dataset is built. Receives df (every column) and context
#                                (stations, nearest_ramps, ramp_readings). Return one value per row of df.
#   @training_feature("name")    recomputed in every fold by add_typical_features(). Receives df and the fold's train_mask.
#
# If a feature needs raw columns that aren't loaded by default (e.g. per-lane speeds), add them to EXTRA_COLUMNS.
# Example (remove the # signs to activate):
#
# EXTRA_COLUMNS += LANE_SPEED_COLUMNS
#
# @feature("lane_speed_spread")
# def lane_speed_spread(df, context):
#     lanes = df[LANE_SPEED_COLUMNS]
#     return lanes.max(axis=1) - lanes.min(axis=1)
#
# @training_feature("flow_anomaly")
# def flow_anomaly(df, train_mask):
#     return df["flow"] - typical_value(df, "flow", train_mask)
