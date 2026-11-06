"""Packaging, portable inference, and report data for the comparative release."""
from __future__ import annotations
import hashlib
import json
import shutil
from pathlib import Path
import sys
import yaml

ROOT=Path(__file__).resolve().parents[2]
MODELS=('random_forest','boosted_trees','transformer')
NAMES=dict(zip(MODELS,('Random Forest','Boosted trees','Transformer')))

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda:stream.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def config(path=None):
    p=Path(path) if path else ROOT/'configs/evaluation.yaml'
    return yaml.safe_load(p.read_text(encoding='utf-8-sig'))

def dump(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,indent=2,default=lambda v: v.item() if hasattr(v, 'item') else str(v)),encoding='utf-8')

def package_results(cfg):
    """Copy completed evidence, never overwrite source runs or team tables."""
    final=ROOT/cfg['release']['results']
    sources=cfg['source_runs']
    manifest={'study':'comparative study; no selected final model','selected_final_model':None,
        'models':list(MODELS),'holdout_already_evaluated':True,'source_runs':sources,'files':[]}
    transfers=[(ROOT/sources['development_tables'],'development',lambda p:p.suffix=='.csv'),
       (ROOT/sources['forward'],'development/forward',lambda p:p.suffix in ('.csv','.json','.png')),
       (ROOT/sources['uncertainty'],'development/uncertainty',lambda p:p.suffix in ('.csv','.json','.png')),
       (ROOT/sources['benchmark'],'operational_benchmark',lambda p:p.suffix in ('.csv','.json','.png') and not p.name.startswith('partial_')),
       (ROOT/sources['holdout'],'holdout',lambda p:p.suffix in ('.csv','.json','.png','.md','.sha256','.parquet'))]
    for source,subfolder,include in transfers:
        for p in sorted(source.iterdir()):
            if not p.is_file() or not include(p):continue
            dest=final/subfolder/p.name;dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(p,dest)
            manifest['files'].append({'path':str(dest.relative_to(final)).replace('\\','/'),
                'sha256':digest(dest),'source':str(p.relative_to(ROOT)).replace('\\','/')})
    # The final report generator must not depend on ignored model-run folders.
    bench=json.loads((ROOT/sources['benchmark']/'worker_config.json').read_text())
    for model in MODELS:
        p=Path(bench['model_dirs'][model])/'validation_predictions.parquet'
        dest=final/'development/predictions'/f'{model}_calibration.parquet'
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
        manifest['files'].append({'path':str(dest.relative_to(final)).replace('\\','/'),'sha256':digest(dest),
            'source':str(p.relative_to(ROOT)).replace('\\','/')})
    # Preserve the original manifest untouched and add a portable release index.
    dump(final/'manifest.json',manifest)
    return final

def package_artifacts(cfg):
    import pandas as pd
    import numpy as np
    from src.features import shared as sf
    from src.random_forest import random_forest_training as rf
    bench=ROOT/cfg['source_runs']['benchmark']
    original=json.loads((bench/'worker_config.json').read_text())
    protocol=json.loads((ROOT/cfg['source_runs']['holdout']/'frozen_protocol.json').read_text())
    assert list(sf.SHARED_FEATURES)==protocol['features']
    assert digest(sf.__file__)==protocol['shared_code_sha256']
    df,signature=rf.load_dataset(sf,ROOT/cfg['data_dir'],use_cache=True)
    assert signature==protocol['dataset_signature']
    train=sf.training_mask(df,cfg['training']['validation_start'])
    # Match the original shared counted-sum/count definition, including its NaN policy.
    stats=None
    for name,column,usable in [('typical_congestion_30','y',df.y_known),
         ('typical_speed','speed',df.real),('typical_occupancy','occupancy',df.real),
         ('typical_min_speed_down','min_speed_down',df.min_speed_down.notna())]:
        counted=(train&usable).to_numpy()
        values=np.where(counted,df[column].to_numpy(dtype=float),0.)
        part=pd.DataFrame(dict(station=df.station,tod=df.tod,value=values,count=counted.astype(int)))
        grouped=part.groupby(['station','tod'])[['value','count']].sum()
        result=(grouped.value/grouped['count'].replace(0,np.nan)).rename(name)
        stats=result.to_frame() if stats is None else stats.join(result)
    stats=stats.reset_index()
    stations,ramps=sf._stations_and_ramps(ROOT/cfg['data_dir']/sf.META_FILE_NAME,pd.DataFrame({'station':protocol['stations']}))
    versions=json.loads((ROOT/cfg['source_runs']['holdout']/'verification.json').read_text())['versions']
    outputs=[]
    for model in MODELS:
        directory=ROOT/cfg['release']['artifacts']/model;directory.mkdir(parents=True,exist_ok=True)
        checkpoint=Path(protocol['checkpoints'][model]['path'])
        assert digest(checkpoint)==protocol['checkpoints'][model]['sha256']
        shutil.copy2(checkpoint,directory/checkpoint.name)
        prep=checkpoint.parent/'preprocessing.json'
        if prep.exists():shutil.copy2(prep,directory/'preprocessing.json')
        stats.to_parquet(directory/'feature_statistics.parquet',index=False)
        stations.to_csv(directory/'station_mapping.csv',index=False)
        dump(directory/'ramp_mapping.json',ramps.to_dict('records'))
        dump(directory/'model_config.json',yaml.safe_load((ROOT/'configs/models'/f'{model}.yaml').read_text()))
        metadata=dict(model=model,selected_final_model=False,role='retained comparative candidate',
            checkpoint=checkpoint.name,checkpoint_sha256=digest(checkpoint),features=protocol['features'],
            feature_code_sha256=digest(sf.__file__),feature_implementation='notebooks/safeflow_features.py',
            dataset_signature=signature,training_window_end_before='2025-03-25',
            calibration_start='2025-04-01',calibration_end='2025-07-01',
            holdout_start='2025-07-01',holdout_end='2026-01-01',versions=versions,
            policies=[p for p in protocol['policies'] if p['model']==model],
            input_contract='Engineered station/time rows, sorted by station and timestamp; include history rows for Transformer.',
            predict_api='src.analysis.study_release.CandidateBundle.predict_features',
            no_budget_compliant_model=True)
        dump(directory/'metadata.json',metadata)
        outputs.append(directory)
    return outputs

class CandidateBundle:
    """Portable loader for our trusted release checkpoints; no refitting.

    Feature building remains causal and external. This interface accepts engineered
    feature rows; apply_training_statistics restores the frozen fold baselines.
    """
    def __init__(self,directory,threads=4):
        self.directory=Path(directory)
        self.metadata=json.loads((self.directory/'metadata.json').read_text())
        self.name=self.metadata['model'];self.features=self.metadata['features'];self.threads=threads
        checkpoint=self.directory/self.metadata['checkpoint']
        if digest(checkpoint)!=self.metadata['checkpoint_sha256']:raise ValueError('Checkpoint hash mismatch')
        if self.name=='transformer':
            # Before Arrow loads native DLLs on Windows.
            import torch
            torch.set_num_threads(threads)
            from src.transformer.transformer_training import FoldPreprocessor
            from src.transformer.safeflow_transformer import SafeFlowTransformer
            ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
            self.prepare=FoldPreprocessor.from_dict(ck['preprocessing'])
            self.lookback=ck['lookback']
            self.model=SafeFlowTransformer(ck['n_features'],max_history=self.lookback,**ck['architecture'])
            self.model.load_state_dict(ck['model_state']);self.model.eval()
        else:
            import joblib
            self.model=joblib.load(checkpoint)
            if self.name=='random_forest':self.model.set_params(model__n_jobs=threads)

    def apply_training_statistics(self,frame):
        import pandas as pd
        f=frame.copy()
        if 'tod' not in f:f['tod']=f.timestamp.dt.hour*60+f.timestamp.dt.minute
        stat=pd.read_parquet(self.directory/'feature_statistics.parquet').set_index(['station','tod'])
        lookup=stat.reindex(pd.MultiIndex.from_frame(f[['station','tod']]))
        f['typical_congestion_30']=lookup.typical_congestion_30.to_numpy(dtype='float32')
        for dest,column,baseline in [('speed_anomaly','speed','typical_speed'),('occ_anomaly','occupancy','typical_occupancy'),('min_speed_down_anomaly','min_speed_down','typical_min_speed_down')]:
            f[dest]=(f[column].to_numpy()-lookup[baseline].to_numpy(dtype='float32')).astype('float32')
        return f

    def predict_features(self,frame,rows=None,batch_size=512):
        import numpy as np
        from threadpoolctl import threadpool_limits
        if frame.duplicated(['station','timestamp']).any():raise ValueError('Duplicate keys')
        if not frame.reset_index(drop=True).equals(frame.sort_values(['station','timestamp']).reset_index(drop=True)):
            raise ValueError('Sort feature rows and endpoints together')
        rows=np.arange(len(frame),dtype=np.int64) if rows is None else np.asarray(rows,dtype=np.int64)
        if not len(rows):return np.array([],dtype=float)
        with threadpool_limits(limits=self.threads):
            if self.name=='transformer':
                from src.transformer.build_history_windows import HistoryIndex,build_history_windows
                from src.transformer.transformer_training import make_loader,predict_probabilities
                f=frame.copy()
                if 'y' not in f:f['y']=False
                x=self.prepare.transform(f)
                ds=build_history_windows(HistoryIndex(f),x,rows,self.lookback)
                p,positions=predict_probabilities(self.model,make_loader(ds,batch_size,False,42,'cpu'),'cpu')
                assert np.array_equal(rows,positions)
                return p
            parts=[]
            for first in range(0,len(rows),100000):
                values=frame.iloc[rows[first:first+100000]][self.features].replace([np.inf,-np.inf],np.nan)
                if self.name=='random_forest':values=values.astype('float32')
                parts.append(self.model.predict_proba(values)[:,1])
            return np.concatenate(parts)
