"""Subprocess workers for local_model_benchmark.ipynb; no work on import."""
from __future__ import annotations
import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path


def memory():
    import psutil
    info = psutil.Process().memory_info()
    if hasattr(info, "peak_wset"):
        peak = info.peak_wset
    else:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak *= 1 if sys.platform == "darwin" else 1024
    return {"rss_bytes": info.rss, "peak_rss_bytes": int(peak)}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def train(config, model):
    import numpy as np
    import pandas as pd
    from threadpoolctl import threadpool_limits
    if model == "random_forest":
        from src.random_forest import random_forest_training as rf
    elif model == "boosted_trees":
        from src.boosted_trees import boosted_trees_training as bt
    else:
        from src.transformer import transformer_training as tf
        from src.transformer.build_history_windows import HistoryIndex
        import torch
        torch.set_num_threads(config["threads"])
        torch.set_num_interop_threads(1)
    baseline = memory()
    start = time.perf_counter()
    frame = pd.read_parquet(config["frame"])
    train_rows = np.load(config["train_rows"])
    val_rows = np.load(config["val_rows"])
    directory = Path(config["model_dirs"][model])
    directory.mkdir(parents=True, exist_ok=True)
    load_seconds = time.perf_counter() - start
    start = time.perf_counter()
    with threadpool_limits(limits=config["threads"]):
        if model == "random_forest":
            result = rf.fit_fold(frame, config["features"], train_rows, val_rows, directory,
                model_parameters=dict(n_estimators=25 if config["quick"] else 300,
                    n_jobs=config["threads"]), save_model=True)
        elif model == "boosted_trees":
            result = bt.fit_fold(frame, config["features"], train_rows, val_rows, directory,
                model_parameters=dict(max_iter=30 if config["quick"] else 300, learning_rate=.05,
                    max_leaf_nodes=31, min_samples_leaf=50, early_stopping=False, random_state=0), save_model=True)
        elif model == "transformer":
            result = tf.fit_fold(frame, HistoryIndex(frame), config["features"], train_rows, val_rows, directory,
                lookback=24, epochs=1 if config["quick"] else 10, batch_size=256,
                seed=42,device="cpu",model_parameters=dict(d_model=64,nhead=4,num_layers=2,d_ff=128,dropout=.1))
        else:
            raise ValueError(model)
    elapsed = time.perf_counter() - start
    prediction = frame.iloc[val_rows][["station","timestamp","date","y","y_known","real","congested_now","onset"]].copy()
    prediction["probability"] = result["probability"]
    prediction.to_parquet(directory / "validation_predictions.parquet",index=False)
    write_json(directory / "training_benchmark.json",dict(model=model,quick=config["quick"],
        worker_pid=os.getpid(),threads=config["threads"],device="cpu",load_seconds=load_seconds,
        fit_predict_save_seconds=elapsed,fit_seconds=result.get("fit_seconds"),
        scope="fit plus validation prediction and artifact save; data load separate",
        baseline_rss_bytes=baseline["rss_bytes"],**memory()))
    print("Training complete:",model,flush=True)


def inference(config, model):
    import joblib
    import numpy as np
    import pandas as pd
    from threadpoolctl import threadpool_limits
    if model == "transformer":
        from src.transformer import transformer_training as tf
        from src.transformer.safeflow_transformer import SafeFlowTransformer
        from src.transformer.build_history_windows import HistoryIndex, build_history_windows
        import torch
        torch.set_num_threads(config["threads"])
        torch.set_num_interop_threads(1)
    directory = Path(config["model_dirs"][model])
    frame = pd.read_parquet(config["fixture"])
    rows = np.load(config["fixture_rows"])
    features = config["features"]
    # Fixture-load baseline is recorded separately; peaks are absolute process RSS.
    baseline = memory()
    start = time.perf_counter()
    if model == "transformer":
        checkpoint = directory / "model.pt"
        # Load only our own checkpoint written by the training worker.
        ck = torch.load(checkpoint,map_location="cpu",weights_only=False)
        prepare = tf.FoldPreprocessor.from_dict(ck["preprocessing"])
        fitted = SafeFlowTransformer(ck["n_features"],max_history=ck["lookback"],**ck["architecture"])
        fitted.load_state_dict(ck["model_state"])
        fitted.eval()
        index = HistoryIndex(frame)
    else:
        checkpoint = directory / "model.joblib"
        fitted = joblib.load(checkpoint)
    load_seconds = time.perf_counter()-start

    def encode(source):
        if model == "transformer":
            return prepare.transform(source)
        values=source[features].replace([np.inf,-np.inf],np.nan)
        return values.astype("float32") if model == "random_forest" else values

    def batch(encoded, endpoints):
        if model != "transformer":
            return encoded.iloc[endpoints]
        dataset=build_history_windows(index,encoded,np.asarray(endpoints,dtype=np.int64),24)
        items=[dataset[i] for i in range(len(dataset))]
        return torch.stack([item[0] for item in items]),torch.stack([item[1] for item in items])

    def predict_batch(value):
        if model == "transformer":
            with torch.inference_mode():
                return torch.sigmoid(fitted(value[0],padding_mask=value[1])).numpy()
        return fitted.predict_proba(value)[:,1]

    def predict_all(source):
        encoded=encode(source)
        return np.concatenate([predict_batch(batch(encoded,rows[i:i+512])) for i in range(0,len(rows),512)])

    with threadpool_limits(limits=config["threads"]):
        start=time.perf_counter()
        encoded=encode(frame)
        encoding_seconds=time.perf_counter()-start
        start=time.perf_counter()
        probability=predict_all(frame)
        full_fixture_seconds=time.perf_counter()-start
        saved=pd.read_parquet(directory/"validation_predictions.parquet").set_index(["station","timestamp"])
        wanted=pd.MultiIndex.from_frame(frame.iloc[rows][["station","timestamp"]])
        expected=saved.probability.reindex(wanted).to_numpy()
        if not np.isfinite(expected).all():raise ValueError("Fixture keys absent from saved predictions")
        error=float(np.max(np.abs(expected-probability)))
        tolerance=1e-5 if model=="transformer" else 1e-10
        if error>tolerance:raise ValueError(f"Checkpoint reproduction failed: max error {error}")
        timings=[]
        rng=np.random.default_rng(42)
        for size in [1,46]:
            # Same endpoint sequence across models; batches can span timestamps.
            positions=[rng.choice(rows,size=size,replace=False) for _ in range(config["warmup"]+config["repetitions"])]
            model_times=[]; batch_times=[]; total_times=[]
            for i,endpoints in enumerate(positions):
                start=time.perf_counter(); value=batch(encoded,endpoints)
                prepared=time.perf_counter(); predict_batch(value); end=time.perf_counter()
                if i>=config["warmup"]:
                    model_times.append(end-prepared);batch_times.append(prepared-start);total_times.append(end-start)
            for scope,values in [("model_only",model_times),("batch_build_plus_model",total_times),("batch_build",batch_times)]:
                timings.append(dict(model=model,batch_size=size,scope=scope,
                    median_ms=1000*float(np.median(values)),p95_ms=1000*float(np.quantile(values,.95)),
                    endpoints_per_second=size/float(np.mean(values)),repetitions=config["repetitions"]))
        pd.DataFrame(timings).to_csv(directory/"latency.csv",index=False)
        evaluation=frame.iloc[rows].copy()
        import safeflow_features as sf
        metrics=dict(model=model,scenario="clean",AP_onsets=sf.ap_onsets(evaluation,probability),
            threshold=config["threshold"],**sf.alert_metrics(evaluation,probability,config["threshold"]))
        scenarios=[metrics]
        if config["missing_feature_tests"]:
            families={
                "current":[f for f in features if f in ["speed","occupancy","flow","speed_anomaly","occ_anomaly","slow_last_30","minutes_since_slow","speed_lag_30","down_minus_here","up_minus_here"] or f.startswith("speed_delta_")],
                "upstream":[f for f in features if f.startswith("upstream_") or f in ["up_minus_here","onramp_flow_up"]],
                "downstream":[f for f in features if f.startswith("downstream_") or f in ["down_minus_here","jam_distance_down","jam_distance_change_10","min_speed_down_anomaly","offramp_flow_down"]]}
            minutes=frame.timestamp.dt.hour*60+frame.timestamp.dt.minute
            for family,columns in families.items():
                for duration in [5,15,30]:
                    modified=frame.copy()
                    mask=minutes.ge(11*60)&minutes.lt(11*60+duration)
                    modified.loc[mask,columns]=np.nan
                    changed=predict_all(modified)
                    scenarios.append(dict(model=model,scenario=f"{family}_missing_features_{duration}min",
                        AP_onsets=sf.ap_onsets(evaluation,changed),threshold=config["threshold"],
                        **sf.alert_metrics(evaluation,changed,config["threshold"])))
        pd.DataFrame(scenarios).to_csv(directory/"feature_stress.csv",index=False)
    write_json(directory/"inference_benchmark.json",dict(model=model,device="cpu",threads=config["threads"],
        worker_pid=os.getpid(),checkpoint_bytes=checkpoint.stat().st_size,checkpoint_load_seconds=load_seconds,
        encode_fixture_seconds=encoding_seconds,fixture_encode_build_predict_seconds=full_fixture_seconds,
        fixture_endpoints=len(rows),max_reproduction_error=error,reproduction_tolerance=tolerance,
        baseline_rss_bytes=baseline["rss_bytes"],**memory(),
        memory_scope="fresh inference process; includes imports, fixture, encoded cache, model, timings and optional stress tests",
        latency_scope="engineered features already computed; raw-data feature extraction excluded"))
    print("Inference benchmark complete:",model,flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",required=True)
    parser.add_argument("--model",choices=["random_forest","boosted_trees","transformer"],required=True)
    parser.add_argument("--stage",choices=["train","inference"],required=True)
    args=parser.parse_args()
    config=json.loads(Path(args.config).read_text(encoding="utf-8"))
    root=Path(config["root"])
    sys.path[:0]=[str(root),str(root/"notebooks")]
    print(args.stage,args.model,"PID",os.getpid(),flush=True)
    (train if args.stage=="train" else inference)(config,args.model)

if __name__=="__main__":main()
