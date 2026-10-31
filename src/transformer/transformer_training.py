"""Small training/preprocessing helpers for train_transformer.ipynb.

No data loading or training runs on import. All caches/checkpoints requested by
these helpers must be inside this local transformer directory.
"""
from __future__ import annotations
import gc
import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from .build_history_windows import build_history_windows
from .safeflow_transformer import SafeFlowTransformer

LOCAL_DIR = Path(__file__).resolve().parent


def local_path(path):
    path = Path(path).resolve()
    if not path.is_relative_to(LOCAL_DIR):
        raise ValueError(f"Outputs must stay inside {LOCAL_DIR}")
    return path


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class FoldPreprocessor:
    """Training-only medians/scales, missing flags, and weekday one-hot encoding.

    The logical feature list remains sf.SHARED_FEATURES. Encoded tensor channels
    can be more numerous because weekday and missingness get separate channels.
    """
    def fit(self, frame, rows, feature_columns):
        self.feature_columns = list(feature_columns)
        if len(set(self.feature_columns)) != len(self.feature_columns):
            raise ValueError("Duplicate feature names")
        self.numeric = [c for c in self.feature_columns if c != "dow"]
        sample = frame.iloc[rows][self.numeric].replace([np.inf, -np.inf], np.nan)
        if not len(sample):
            raise ValueError("No training rows for preprocessing")
        self.median = sample.median().fillna(0.0)
        filled = sample.fillna(self.median)
        self.mean = filled.mean()
        self.scale = filled.std(ddof=0).replace(0, 1).fillna(1)
        self.indicators = [c for c in self.numeric if sample[c].isna().any()]
        self.weekdays = [1, 2, 3, 4] if "dow" in self.feature_columns else []
        self.tensor_columns = (self.numeric + [f"dow_{i}" for i in self.weekdays]
                               + [f"{c}_missing" for c in self.indicators])
        return self

    def to_dict(self):
        return {"logical_features": self.feature_columns, "numeric": self.numeric,
                "median": self.median.to_dict(), "mean": self.mean.to_dict(),
                "scale": self.scale.to_dict(), "missing_indicators": self.indicators,
                "weekdays": self.weekdays, "tensor_columns": self.tensor_columns}

    @classmethod
    def from_dict(cls, parameters):
        """Restore preprocessing saved in a checkpoint for later predictions."""
        obj = cls()
        obj.feature_columns = parameters["logical_features"]
        obj.numeric = parameters["numeric"]
        for field in ("median", "mean", "scale"):
            setattr(obj, field, pd.Series(parameters[field]).reindex(obj.numeric))
        obj.indicators = parameters["missing_indicators"]
        obj.weekdays = parameters["weekdays"]
        obj.tensor_columns = parameters["tensor_columns"]
        return obj

    def transform(self, frame, destination=None, chunk_size=100_000):
        shape = (len(frame), len(self.tensor_columns))
        if destination is None:
            result = np.empty(shape, dtype=np.float32)
        else:
            destination = local_path(destination)
            destination.parent.mkdir(parents=True, exist_ok=True)
            result = np.lib.format.open_memmap(destination, mode="w+", dtype=np.float32, shape=shape)
        med = self.median.to_numpy(dtype=np.float32)
        mean = self.mean.to_numpy(dtype=np.float32)
        scale = self.scale.to_numpy(dtype=np.float32)
        flag_indices = [self.numeric.index(c) for c in self.indicators]
        for start in range(0, len(frame), chunk_size):
            end = min(start + chunk_size, len(frame))
            part = frame.iloc[start:end]
            values = part[self.numeric].to_numpy(dtype=np.float32, copy=True)
            absent = ~np.isfinite(values)
            values = (np.where(absent, med, values) - mean) / scale
            blocks = [values]
            if self.weekdays:
                dow = part["dow"].to_numpy()
                if not np.isin(dow, [0, 1, 2, 3, 4]).all():
                    raise ValueError("Expected weekday dow values 0 through 4")
                blocks.append(np.column_stack([(dow == i).astype(np.float32) for i in self.weekdays]))
            if flag_indices:
                blocks.append(absent[:, flag_indices].astype(np.float32))
            result[start:end] = np.column_stack(blocks)
            if not np.isfinite(result[start:end]).all():
                raise ValueError("Non-finite values after preprocessing")
        if isinstance(result, np.memmap):
            result.flush()
        return result


def make_loader(dataset, batch_size, shuffle=False, seed=42, device="cpu"):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=torch.device(device).type == "cuda",
                      generator=torch.Generator().manual_seed(seed))


def predict_probabilities(model, loader, device):
    """Return probabilities plus keys-as-positions; preserve loader endpoint order."""
    probabilities, positions = [], []
    model.eval()
    with torch.inference_mode():
        for x, padding, _, rows in loader:
            logits = model(x.to(device), padding_mask=padding.to(device))
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
            positions.append(rows.numpy())
    if not probabilities:
        raise ValueError("Empty prediction dataset")
    return np.concatenate(probabilities), np.concatenate(positions)


def fit_fold(frame, index, feature_columns, train_rows, validation_rows, output_dir,
             *, lookback=24, epochs=10, batch_size=256, learning_rate=1e-3,
             weight_decay=0.01, seed=42, device="cpu", model_parameters=None,
             keep_cache=False, verbose=True):
    """Fresh model/preprocessing per fold; fixed epochs; validation predicted once.

    train_rows selects only eligible endpoints; frame/index retain unknown-label
    rows as context. Validation scores do not select an epoch/checkpoint here.
    """
    output_dir = local_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_rows = np.asarray(train_rows, dtype=np.int64)
    validation_rows = np.asarray(validation_rows, dtype=np.int64)
    if not len(train_rows) or not len(validation_rows):
        raise ValueError("Training and validation endpoints must be nonempty")
    if not frame["y_known"].to_numpy()[train_rows].all():
        raise ValueError("Training endpoints must have measured future outcomes")
    if np.intersect1d(train_rows, validation_rows).size:
        raise ValueError("Training and validation endpoints overlap")
    if epochs < 1:
        raise ValueError("epochs must be positive")
    seed_everything(seed)
    prepare = FoldPreprocessor().fit(frame, train_rows, feature_columns)
    cache_file = output_dir / "features.npy"
    x = prepare.transform(frame, cache_file)
    train_data = build_history_windows(index, x, train_rows, lookback)
    val_data = build_history_windows(index, x, validation_rows, lookback)
    train_loader = make_loader(train_data, batch_size, True, seed, device)
    val_loader = make_loader(val_data, batch_size * 2, False, seed, device)
    architecture = dict(model_parameters or {})
    model = SafeFlowTransformer(n_features=x.shape[1], max_history=lookback, **architecture).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    history = []
    import time
    for epoch in range(1, epochs + 1):
        start = time.monotonic()
        model.train()
        loss_sum = 0.0
        count = 0
        for bx, padding, by, _ in train_loader:
            bx, padding, by = bx.to(device), padding.to(device), by.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(bx, padding_mask=padding)
            if logits.shape != by.shape:
                raise ValueError("Logit and label shapes differ")
            loss = criterion(logits, by)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += loss.detach().item() * len(by)
            count += len(by)
        row = {"epoch": epoch, "train_BCE": loss_sum / count,
               "seconds": time.monotonic() - start}
        history.append(row)
        pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
        if verbose:
            print(f"  epoch {epoch:02d}/{epochs}: BCE={row['train_BCE']:.4f}, {row['seconds']:.1f}s", flush=True)
    probability, positions = predict_probabilities(model, val_loader, device)
    if not np.array_equal(positions, validation_rows):
        raise ValueError("Prediction rows are not aligned to validation rows")
    # Save the last fixed epoch, not a checkpoint selected on the reported validation set.
    checkpoint = {"model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                  "architecture": architecture, "n_features": x.shape[1], "lookback": lookback,
                  "preprocessing": prepare.to_dict(), "epochs": epochs,
                  "logical_features": list(feature_columns)}
    torch.save(checkpoint, output_dir / "model.pt")
    (output_dir / "preprocessing.json").write_text(json.dumps(prepare.to_dict(), indent=2))
    result = {"probability": probability, "positions": positions,
              "history": pd.DataFrame(history), "n_tensor_channels": x.shape[1]}
    # No millions-of-windows tensor is materialized. Discard each fold's feature cache.
    del train_loader, val_loader, train_data, val_data, model, optimizer
    gc.collect()
    if isinstance(x, np.memmap):
        x.flush()
        x._mmap.close()
    del x
    if not keep_cache:
        cache_file.unlink()
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()
    return result
