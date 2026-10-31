from __future__ import annotations
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class HistoryIndex:
    # Time/station bookkeeping used across all folds
    # frame must be sorted by station first then timestamp
    def __init__(self, frame, target_column="y", step="5min"):
        station = frame["station"].to_numpy()
        timestamp = frame["timestamp"].to_numpy(dtype="datetime64[ns]")

        if not len(frame) or np.isnat(timestamp).any():
            raise ValueError("Need nonempty rows with valid timestamps")
        
        change = station[1:] != station[:-1]
        difference = np.diff(timestamp).astype("timedelta64[ns]").astype(np.int64)

        if np.any(station[1:] < station[:-1]) or np.any((~change) & (difference <= 0)):
            raise ValueError("Rows must be unique and sorted by station, timestamp")
        
        starts_run = np.r_[True, change | (difference != pd.Timedelta(step).value)]
        positions = np.arange(len(frame), dtype=np.int64)

        self.run_start = np.maximum.accumulate(np.where(starts_run, positions, 0))
        self.targets = frame[target_column].to_numpy(dtype=np.float32, copy=True)
        self.n_rows = len(frame)


class HistoryWindows(Dataset):
    # Return (features, padding_mask, target, original_row_position)
    def __init__(self, index, features, end_positions, lookback=24):
        if lookback < 1:
            raise ValueError("lookback must be positive")
        
        self.index = index
        self.features = np.asarray(features, dtype=np.float32)

        if self.features.ndim != 2 or self.features.shape[0] != index.n_rows:
            raise ValueError("Feature matrix must align with the entire input frame")
        
        raw = np.asarray(end_positions)

        if raw.ndim != 1 or not np.issubdtype(raw.dtype, np.integer):
            raise ValueError("end_positions must be a one-dimensional integer array")
        
        self.end_positions = raw.astype(np.int64, copy=True)

        if len(raw) and (raw.min() < 0 or raw.max() >= index.n_rows):
            raise ValueError("Endpoint outside input rows")
        
        self.lookback = lookback

    def __len__(self):
        return len(self.end_positions)

    def __getitem__(self, item):
        end = int(self.end_positions[item])
        start = max(int(self.index.run_start[end]), end - self.lookback + 1)
        length = end - start + 1

        x = np.zeros((self.lookback, self.features.shape[1]), dtype=np.float32)
        x[-length:] = self.features[start:end + 1]

        if not np.isfinite(x[-length:]).all():
            raise ValueError("Impute/scale all input features before making windows")
        
        padding = np.ones(self.lookback, dtype=bool)
        padding[-length:] = False

        return (torch.from_numpy(x), torch.from_numpy(padding), torch.tensor(self.index.targets[end]), end)

# Build history dataset from selected features and endpoints
def build_history_windows(index, features, end_positions, lookback=24):
    return HistoryWindows(index, features, end_positions, lookback=lookback)
