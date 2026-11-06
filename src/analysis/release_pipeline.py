"""Replay the recorded last-fold train/calibrate/test recipe without notebook execution."""
from pathlib import Path
import gc,json,os,subprocess,sys,time
from datetime import datetime
from .study_release import ROOT,MODELS,config,dump,digest,package_results


def validate(cfg):
    import yaml
    expected={'random_forest':{'n_estimators':300,'max_depth':12,'min_samples_leaf':10,'max_samples':.25},
      'boosted_trees':{'max_iter':300,'learning_rate':.05,'max_leaf_nodes':31,'min_samples_leaf':50,'early_stopping':False},
      'transformer':{'d_model':64,'nhead':4,'num_layers':2,'d_ff':128,'dropout':.1}}
    for model in MODELS:
        m=yaml.safe_load((ROOT/'configs/models'/f'{model}.yaml').read_text())
        assert m['seed']==(0 if model=='boosted_trees' else 42)
        if model=='transformer':
            for k,v in {'lookback':24,'epochs':10,'batch_size':256,'learning_rate':.001,'weight_decay':.01,'device':'cpu'}.items():assert m[k]==v
        assert m['parameters']==expected[model],f'{model}: this command replays the recorded settings, not a new tuning experiment'
    assert cfg['target']=={'speed_below_mph':40,'future_readings':6,'interval_minutes':5,'early_warning_minutes':15}
    assert cfg['policies']=={'duration_hours_per_day':4,'false_episodes_per_day':20,'diagnostic_threshold':.5,'primary':'duration'}
    assert cfg['training']['max_endpoints']==2000000 and cfg['training']['sample_seed']==0
    assert cfg['training']['validation_start']=='2025-04-01'
    assert cfg['calibration']=={'start':'2025-04-01','end':'2025-07-01'}
    assert cfg['holdout']=={'start':'2025-07-01','end':'2026-01-01'}
    assert cfg['selected_final_model'] is None and set(cfg['models'])==set(MODELS)


def prepare_training(cfg,run,use_cache=False):
    import numpy as np
    import pandas as pd
    from src.features import shared as sf
    from src.random_forest import random_forest_training as training
    # Initialize Torch before Arrow's DLLs, as in the validated notebook.
    from src.transformer.build_history_windows import HistoryIndex
    validate(cfg)
    inputs=run/'inputs';inputs.mkdir(parents=True,exist_ok=True)
    df,signature=training.load_dataset(sf,ROOT/cfg['data_dir'],use_cache=use_cache)
    df=df.sort_values(sf.KEY).reset_index(drop=True)
    fold=next(sf.folds(df,only_last=True));sf.add_typical_features(df,fold.train)
    candidates=np.flatnonzero((fold.train&df.y_known).to_numpy())
    train=pd.Series(candidates).sample(min(cfg['training']['max_endpoints'],len(candidates)),random_state=cfg['training']['sample_seed']).to_numpy()
    val=np.flatnonzero(fold.val.to_numpy())
    index=HistoryIndex(df);endpoints=np.r_[train,val];keep=np.zeros(len(df),dtype=bool)
    for lag in range(24):keep[np.maximum(index.run_start[endpoints],endpoints-lag)]=True
    original=np.flatnonzero(keep);mapping=np.full(len(df),-1,dtype=np.int64);mapping[original]=np.arange(len(original))
    cols=list(dict.fromkeys(sf.KEY+['date','y','y_known','real','congested_now','onset']+list(sf.SHARED_FEATURES)))
    frame=df.loc[keep,cols].reset_index(drop=True);train=mapping[train];val=mapping[val]
    frame.to_parquet(inputs/'development_frame.parquet',index=False)
    np.save(inputs/'train_rows.npy',train);np.save(inputs/'val_rows.npy',val)
    days=sorted(frame.iloc[val].date.unique())[:5]
    fixture_endpoints=val[frame.iloc[val].date.isin(days).to_numpy()]
    idx=HistoryIndex(frame);mask=np.zeros(len(frame),dtype=bool)
    for lag in range(24):mask[np.maximum(idx.run_start[fixture_endpoints],fixture_endpoints-lag)]=True
    positions=np.flatnonzero(mask);remap=np.full(len(frame),-1,dtype=np.int64);remap[positions]=np.arange(len(positions))
    frame.loc[mask].reset_index(drop=True).to_parquet(inputs/'inference_fixture.parquet',index=False)
    np.save(inputs/'fixture_rows.npy',remap[fixture_endpoints])
    directories={m:str(ROOT/'src'/m/'results'/('reproduction_'+run.parent.name)/'2025-04') for m in MODELS}
    for value in directories.values():
        p=Path(value);p.mkdir(parents=True,exist_ok=True);(p.parent/'.gitignore').write_text('*\n!.gitignore\n')
    worker=dict(root=str(ROOT),quick=False,threads=cfg['training']['threads'],features=list(sf.SHARED_FEATURES),
        frame=str(inputs/'development_frame.parquet'),train_rows=str(inputs/'train_rows.npy'),val_rows=str(inputs/'val_rows.npy'),
        fixture=str(inputs/'inference_fixture.parquet'),fixture_rows=str(inputs/'fixture_rows.npy'),model_dirs=directories,
        warmup=20,repetitions=200,threshold=.5,missing_feature_tests=True)
    paths=['notebooks/safeflow_features.py','src/analysis/local_benchmark.py','src/random_forest/safeflow_random_forest.py',
      'src/random_forest/random_forest_training.py','src/boosted_trees/boosted_trees_training.py',
      'src/transformer/transformer_training.py','src/transformer/safeflow_transformer.py','src/transformer/build_history_windows.py']
    manifest=dict(status='prepared',quick_run=False,fold=fold.name,holdout_used=False,dataset_signature=signature,
        features=list(sf.SHARED_FEATURES),source_hashes={p:digest(ROOT/p) for p in paths},
        train_rows=len(train),validation_rows=len(val),input_sha256={p.name:digest(p) for p in inputs.iterdir() if p.is_file()})
    dump(run/'worker_config.json',worker);dump(run/'manifest.json',manifest)
    sf._LAST_CONTEXT=None
    print('Prepared frozen replay:',len(train),'training endpoints;',len(val),'validation endpoints',flush=True)
    return worker,manifest


def run_process(command,log,threads=4):
    env=os.environ.copy()
    for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:env[key]=str(threads)
    env['PYTHONIOENCODING']='utf-8'
    # Match activated Conda native-library search when launched from IDE Python.
    native=Path(sys.prefix)/'Library/bin'
    if native.is_dir():env['PATH']=str(native)+os.pathsep+env.get('PATH','')
    start=time.monotonic();last=start;offset=0
    with Path(log).open('w',encoding='utf-8') as stream:
        p=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        try:
            while True:
                value=Path(log).read_text(encoding='utf-8',errors='replace')
                if len(value)>offset:print(value[offset:],end='',flush=True);offset=len(value)
                if p.poll() is not None:break
                if time.monotonic()-last>30:print(f'Worker active: {time.monotonic()-start:.0f}s',flush=True);last=time.monotonic()
                time.sleep(2)
        except BaseException:
            p.terminate()
            try:p.wait(timeout=10)
            except subprocess.TimeoutExpired:p.kill();p.wait()
            raise
    if p.returncode:raise RuntimeError(f'Worker failed; see {log}')


def replay(prepare_only=False,use_cache=False):
    cfg=config();validate(cfg)
    root=ROOT/'results/reproductions'/datetime.now().strftime('%Y%m%d_%H%M%S')
    benchmark=root/'benchmark';benchmark.mkdir(parents=True)
    worker,manifest=prepare_training(cfg,benchmark,use_cache)
    gc.collect()
    if prepare_only:return root
    for model in MODELS:
        for stage in ['train','inference']:
            run_process([sys.executable,'-u','-m','src.analysis.local_benchmark','--config',str(benchmark/'worker_config.json'),
                '--model',model,'--stage',stage],benchmark/f'{model}_{stage}.log',worker['threads'])
    import pandas as pd
    from src.features import shared as sf
    train=[];infer=[];latency=[];stress=[];quality=[]
    for model,value in worker['model_dirs'].items():
        directory=Path(value)
        train.append(json.loads((directory/'training_benchmark.json').read_text()))
        infer.append(json.loads((directory/'inference_benchmark.json').read_text()))
        latency.append(pd.read_csv(directory/'latency.csv'));stress.append(pd.read_csv(directory/'feature_stress.csv'))
        pred=pd.read_parquet(directory/'validation_predictions.parquet')
        quality.append(dict(model=model,AP_onsets=sf.ap_onsets(pred,pred.probability),threshold=.5,**sf.alert_metrics(pred,pred.probability,.5)))
    for name,table in [('training',pd.DataFrame(train)),('inference',pd.DataFrame(infer)),('latency',pd.concat(latency)),('feature_stress',pd.concat(stress)),('predictive_metrics',pd.DataFrame(quality))]:table.to_csv(benchmark/f'{name}.csv',index=False)
    manifest['status']='complete';dump(benchmark/'manifest.json',manifest)
    from .evaluate_holdout import freeze,prepare,report
    holdout=root/'holdout';(holdout/'inputs').mkdir(parents=True)
    plan,development=freeze(holdout,benchmark);prepare(holdout,plan,development)
    del development;gc.collect()
    for model in MODELS:
        run_process([sys.executable,'-u','-m','src.analysis.evaluate_holdout','--stage','predict','--model',model,'--run-dir',str(holdout)],holdout/f'{model}.log',worker['threads'])
    report(holdout,plan)
    release_cfg=dict(cfg);release_cfg['source_runs']=dict(cfg['source_runs'],benchmark=str(benchmark.relative_to(ROOT)),holdout=str(holdout.relative_to(ROOT)))
    release_cfg['release']=dict(cfg['release'],results=str((root/'release').relative_to(ROOT)))
    final=package_results(release_cfg)
    from .release_reporting import generate
    generate(final,source=final)
    dump(root/'reproduction.json',dict(scope='Replay of an already-used holdout, not a new independent test',
        config_sha256=digest(ROOT/'configs/evaluation.yaml'),release=str(final),
        historical_cv='Published evidence copied with provenance; original historical fits are not reconstructed'))
    return root
