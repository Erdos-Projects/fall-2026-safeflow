"""Create a compact, portable checkpoint-reproduction fixture from frozen evidence."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
import numpy as np
import pandas as pd
from src.transformer.build_history_windows import HistoryIndex
from src.analysis.study_release import config,digest,dump,MODELS
cfg=config();benchmark=ROOT/cfg['source_runs']['benchmark']
worker=json.loads((benchmark/'worker_config.json').read_text())
frame=pd.read_parquet(worker['fixture']);endpoints=np.load(worker['fixture_rows'])
chosen=endpoints[np.linspace(0,len(endpoints)-1,48,dtype=int)]
index=HistoryIndex(frame);keep=np.zeros(len(frame),dtype=bool)
for lag in range(24):keep[np.maximum(index.run_start[chosen],chosen-lag)]=True
mapping=np.full(len(frame),-1,dtype=np.int64);mapping[np.flatnonzero(keep)]=np.arange(keep.sum())
fixture=frame.loc[keep].reset_index(drop=True)
# Include one static measurement needed to reconstruct all training-dependent anomalies.
cache=next((ROOT/'src/random_forest/cache').glob('development_'+json.loads((benchmark/'manifest.json').read_text())['dataset_signature'][:16]+'.parquet'))
min_down=pd.read_parquet(cache,columns=['station','timestamp','min_speed_down'],filters=[('timestamp','>=',fixture.timestamp.min()),('timestamp','<=',fixture.timestamp.max())]).set_index(['station','timestamp'])
fixture['min_speed_down']=min_down.reindex(pd.MultiIndex.from_frame(fixture[['station','timestamp']])).min_speed_down.to_numpy()
folder=ROOT/'tests/fixtures';folder.mkdir(exist_ok=True)
fixture.to_parquet(folder/'feature_frame.parquet',index=False);np.save(folder/'endpoint_rows.npy',mapping[chosen])
expected=fixture.iloc[mapping[chosen]][['station','timestamp']].copy().reset_index(drop=True)
for model in MODELS:
 pred=pd.read_parquet(Path(worker['model_dirs'][model])/'validation_predictions.parquet').set_index(['station','timestamp'])
 expected[model]=pred.probability.reindex(pd.MultiIndex.from_frame(expected[['station','timestamp']])).to_numpy()
expected.to_parquet(folder/'expected_probabilities.parquet',index=False)
dump(folder/'README.json',dict(scope='48 fixed April-June development endpoints with causal historical context; no new holdout sampling',feature_frame_sha256=digest(folder/'feature_frame.parquet'),labels='Targets are carried for compatibility but are not model inputs',selection='48 evenly spaced original fixture endpoints'))
print('Created portable fixture:',len(fixture),'context rows and',len(expected),'endpoints')
