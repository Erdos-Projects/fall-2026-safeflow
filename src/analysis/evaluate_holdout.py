"""Frozen-checkpoint D12 holdout evaluation. No fitting or holdout-based tuning."""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'notebooks')]
MODELS=['random_forest','boosted_trees','transformer']
NAMES={'random_forest':'Random Forest','boosted_trees':'Boosted trees','transformer':'Transformer'}
START='2025-07-01'
END='2026-01-01'

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,default=str),encoding='utf-8')

def window_files(directory,start=START,end=END):
    import pandas as pd
    lo,hi=pd.Timestamp(start).date(),pd.Timestamp(end).date()
    return [p for p in sorted(Path(directory).rglob('d12_text_station_5min_*.parquet'))
            if lo<=datetime.strptime(p.stem[-10:],'%Y_%m_%d').date()<hi]

def freeze(out,source_run):
    import numpy as np
    import pandas as pd
    import safeflow_features as sf
    from src.random_forest import random_forest_training as training
    from src.analysis.early_warning_errors import align_predictions,ExactWarningSweep
    source=json.loads((source_run/'manifest.json').read_text())
    config=json.loads((source_run/'worker_config.json').read_text())
    assert source['status']=='complete' and not source['quick_run'] and not source['holdout_used']
    assert config['features']==list(sf.SHARED_FEATURES)
    for relative,expected in source['source_hashes'].items():
        assert digest(ROOT/relative.replace('\\','/'))==expected,relative
    data=ROOT/'data'
    development,signature=training.load_dataset(sf,data,use_cache=True)
    assert signature==source['dataset_signature'],'Development inputs differ from checkpoint training'
    development=development.sort_values(sf.KEY).reset_index(drop=True)
    reference=None;policies=[];checkpoints={};prediction_files={}
    for model in MODELS:
        directory=Path(config['model_dirs'][model])
        checkpoint=directory/('model.pt' if model=='transformer' else 'model.joblib')
        prediction_file=directory/'validation_predictions.parquet'
        prediction=pd.read_parquet(prediction_file)
        if reference is None:reference=prediction.sort_values(sf.KEY).reset_index(drop=True)
        p=align_predictions(reference,prediction)
        assert reference.timestamp.min()>=pd.Timestamp('2025-04-01')
        assert reference.timestamp.max()<pd.Timestamp(START)
        selected,_=ExactWarningSweep(reference,p).operating_points(episode_budget=20,hours_budget=4)
        for row in selected.to_dict('records'):
            policies.append(dict(model=model,policy=row['budget_type'],threshold=float(row['threshold']),
                budget=float(row['budget']),calibration_catch=float(row['caught_15min']),
                calibration_false_episodes_per_day=float(row['false_episodes_per_day']),
                calibration_wrong_alert_hours_per_day=float(row['wrong_alert_hours_per_day'])))
        policies.append(dict(model=model,policy='default_0.5',threshold=.5,budget=None))
        checkpoints[model]=dict(path=str(checkpoint),sha256=digest(checkpoint))
        prediction_files[model]=dict(path=str(prediction_file),sha256=digest(prediction_file))
    stations=sorted(int(x) for x in reference.station.unique())
    assert len(stations)==46 and len(reference)==486864
    files=window_files(data)
    assert files,'No holdout paths found'
    plan=dict(status='frozen_before_holdout_read',frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        source_benchmark=str(source_run.relative_to(ROOT)),features=config['features'],
        shared_code_sha256=digest(sf.__file__),metadata_sha256=digest(data/sf.META_FILE_NAME),
        dataset_signature=signature,checkpoints=checkpoints,calibration_predictions=prediction_files,
        training='Existing local checkpoint fits only; no retraining',
        training_window_end_before='2025-03-25',calibration_start='2025-04-01',calibration_end=START,
        holdout_start=START,holdout_end=END,stations=stations,threads=4,
        primary_warning_policy='duration',primary_warning_metric='onsets warned at least 15 minutes early',
        ranking_metric='AP on measured currently uncongested origins with known future labels',
        policies=policies,threshold_selection='April-June checkpoint predictions only; no holdout adaptation',
        evaluation='Same observed station/time rows for all models; no quarterly recalibration on holdout',
        quality_policy='Fail on duplicate keys or nonfinite measured speed; shared known-label rules unchanged',
        history='Same day 04:00-19:55 inputs for recent features; Transformer resets history at time gaps',
        typical_features='Frozen checkpoint training-period statistics from full unsampled development training mask',
        holdout_paths=[str(p.relative_to(data)) for p in files],
        holdout_loaded=False)
    # This file and its digest are persisted before opening any holdout Parquet.
    write(out/'frozen_protocol.json',plan)
    (out/'frozen_protocol.sha256').write_text(digest(out/'frozen_protocol.json')+'\n')
    pd.DataFrame(policies).to_csv(out/'frozen_policies.csv',index=False)
    reference[['station','timestamp','y','y_known','real','congested_now','onset']].to_parquet(out/'inputs/calibration_keys.parquet',index=False)
    print('Protocol and 9 policies frozen before holdout access:',out,flush=True)
    return plan,development

def prepare(out,plan,development):
    import numpy as np
    import pandas as pd
    from pandas.tseries.holiday import USFederalHolidayCalendar
    import safeflow_features as sf
    assert digest(sf.__file__)==plan['shared_code_sha256']
    data=ROOT/'data'
    assert digest(data/sf.META_FILE_NAME)==plan['metadata_sha256']
    stations=set(plan['stations'])
    holidays=set(USFederalHolidayCalendar().holidays(START,END).date)
    paths=[data/p for p in plan['holdout_paths']]
    columns=['timestamp','station','lane_type','speed','flow','occupancy','pct_observed']
    accessed=[]
    def loader(directory,verbose,extra_columns=()):
        main=[];ramps=[];history=[]
        for path in paths:
            day=datetime.strptime(path.stem[-10:],'%Y_%m_%d').date()
            if day.weekday()>=5 or day in holidays:continue
            if not accessed:
                write(out/'holdout_access.json',dict(first_access_at_utc=datetime.now(timezone.utc).isoformat(),
                    protocol_sha256=digest(out/'frozen_protocol.json'),holdout_used=True))
            readings=pd.read_parquet(path,columns=columns+list(extra_columns))
            accessed.append(dict(path=str(path.relative_to(data)),sha256=digest(path)))
            ml=readings.lane_type.eq('ML') & readings.station.isin(stations)
            measured=readings.pct_observed.ge(100)
            if (ml & measured & ~np.isfinite(readings.speed)).any():
                raise ValueError(f'Nonfinite measured mainline speed: {path}')
            if readings.loc[ml].duplicated(sf.KEY).any():raise ValueError(f'Duplicate ML keys: {path}')
            hour=readings.timestamp.dt.hour
            history.append(readings.loc[ml & hour.between(sf.HISTORY_INPUT_START_HOUR,sf.HOUR_END),sf.KEY+['speed']])
            origin=hour.between(sf.HOUR_START,sf.HOUR_END)
            main.append(readings.loc[ml & origin].drop(columns='lane_type'))
            ramp=(readings.lane_type.isin(['OR','FR']) | readings.station.isin(sf.CONNECTOR_ROLES)) & origin
            ramps.append(readings.loc[ramp].drop(columns=['lane_type',*extra_columns]))
            if len(accessed)%20==0:print('Read holdout weekdays:',len(accessed),flush=True)
        def combine(parts):
            result=pd.concat(parts,ignore_index=True)
            for column in ['speed','flow','occupancy']:result[column]=result[column].astype('float32')
            result['real']=result.pct_observed.ge(100)
            return result.drop(columns='pct_observed')
        hist=pd.concat(history,ignore_index=True);hist['speed']=hist.speed.astype('float32')
        return combine(main),combine(ramps),hist
    original_loader=sf._load_readings
    original_stations=sf._stations_and_ramps
    sf._load_readings=loader
    # Keep the development station graph even if station availability changes in holdout.
    sf._stations_and_ramps=lambda meta,unused:original_stations(meta,pd.DataFrame({'station':plan['stations']}))
    try:holdout=sf.build_dataset(data,verbose=True)
    finally:sf._load_readings=original_loader;sf._stations_and_ramps=original_stations
    assert holdout.timestamp.min()>=pd.Timestamp(START) and holdout.timestamp.max()<pd.Timestamp(END)
    assert not holdout.duplicated(sf.KEY).any()
    # Shared formulas, with the ORIGINAL checkpoint training mask, never test labels.
    combined=pd.concat([development,holdout],ignore_index=True).sort_values(sf.KEY).reset_index(drop=True)
    mask=sf.training_mask(combined,'2025-04-01')
    assert combined.loc[mask,'window_end'].max()<pd.Timestamp(plan['training_window_end_before'])
    sf.add_typical_features(combined,mask)
    # Recomputed Q2 feature inputs must equal those used by the fitted checkpoints.
    source_run=ROOT/plan['source_benchmark']
    config=json.loads((source_run/'worker_config.json').read_text())
    saved=pd.read_parquet(config['frame'],columns=sf.KEY+plan['features'],
        filters=[('timestamp','>=',pd.Timestamp('2025-04-01')),('timestamp','<',pd.Timestamp(START))]).set_index(sf.KEY).sort_index()
    current=combined.loc[combined.timestamp.ge('2025-04-01') & combined.timestamp.lt(START),sf.KEY+plan['features']].set_index(sf.KEY).sort_index()
    pd.testing.assert_frame_equal(current,saved,check_dtype=False,rtol=1e-6,atol=1e-6)
    frame=combined.loc[combined.timestamp.ge(START)&combined.window_end.lt(END)].copy()
    columns=list(dict.fromkeys(sf.KEY+['date','y','y_known','real','congested_now','onset']+plan['features']))
    frame=frame[columns].sort_values(sf.KEY).reset_index(drop=True)
    frame.to_parquet(out/'inputs/holdout_frame.parquet',index=False)
    months=[]
    for month,group in frame.groupby(frame.timestamp.dt.to_period('M')):
        months.append(dict(month=str(month),rows=len(group),known_rows=int(group.y_known.sum()),
            eligible_AP_rows=int((group.y_known&group.real&~group.congested_now).sum()),onsets=int(group.onset.sum()),days=group.date.nunique()))
    pd.DataFrame(months).to_csv(out/'monthly_coverage.csv',index=False)
    write(out/'holdout_input_provenance.json',dict(files=accessed,rows=len(frame),days=frame.date.nunique(),
        stations=frame.station.nunique(),min_time=str(frame.timestamp.min()),max_time=str(frame.timestamp.max()),
        onsets=int(frame.onset.sum()),calibration_feature_reproduction=True,training_statistics_used_test_labels=False))
    sf._LAST_CONTEXT=None
    del combined,holdout,current,saved,development,frame
    gc.collect()
    print('Prepared common holdout inputs; Q2 features reproduce checkpoint inputs.',flush=True)

def predict(out,plan,model):
    # Initialize PyTorch's native libraries before Arrow/threadpool DLL loading.
    if model=='transformer':
        import torch
        torch.set_num_threads(plan['threads']);torch.set_num_interop_threads(1)
    import numpy as np
    import pandas as pd
    from threadpoolctl import threadpool_limits
    import joblib
    frame=pd.read_parquet(out/'inputs/holdout_frame.parquet')
    item=plan['checkpoints'][model]
    assert digest(item['path'])==item['sha256'],'Frozen checkpoint changed'
    start=time.perf_counter()
    with threadpool_limits(limits=plan['threads']):
        if model=='transformer':
            import torch
            from src.transformer.transformer_training import FoldPreprocessor,make_loader,predict_probabilities
            from src.transformer.safeflow_transformer import SafeFlowTransformer
            from src.transformer.build_history_windows import HistoryIndex,build_history_windows
            ck=torch.load(item['path'],map_location='cpu',weights_only=False)
            assert ck['logical_features']==plan['features']
            prep=FoldPreprocessor.from_dict(ck['preprocessing'])
            x=prep.transform(frame)
            rows=np.arange(len(frame),dtype=np.int64)
            ds=build_history_windows(HistoryIndex(frame),x,rows,ck['lookback'])
            fitted=SafeFlowTransformer(ck['n_features'],max_history=ck['lookback'],**ck['architecture'])
            fitted.load_state_dict(ck['model_state'])
            loader=make_loader(ds,512,False,42,'cpu')
            p,positions=predict_probabilities(fitted,loader,'cpu')
            assert np.array_equal(rows,positions)
        else:
            fitted=joblib.load(item['path']);parts=[]
            for first in range(0,len(frame),100_000):
                values=frame.iloc[first:first+100_000][plan['features']].replace([np.inf,-np.inf],np.nan)
                if model=='random_forest':values=values.astype('float32')
                parts.append(fitted.predict_proba(values)[:,1])
                print(model,'predicted',min(first+100_000,len(frame)),'/',len(frame),flush=True)
            p=np.concatenate(parts)
    assert len(p)==len(frame) and np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    export=frame[['station','timestamp','date','y','y_known','real','congested_now','onset']].copy()
    export['fold']='2025-07 to 2025-12';export['probability']=p
    export.to_parquet(out/(model+'_holdout_predictions.parquet'),index=False)
    write(out/(model+'_prediction_manifest.json'),dict(checkpoint_sha256=item['sha256'],rows=len(p),seconds=time.perf_counter()-start,refitted=False))
    print('Holdout predictions saved:',model,flush=True)

def report(out,plan):
    import numpy as np
    import pandas as pd
    import safeflow_features as sf
    from src.analysis.early_warning_errors import align_predictions,warning_leads
    from src.analysis.paired_uncertainty import paired_bootstrap
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve
    frame=pd.read_parquet(out/'inputs/holdout_frame.parquet')
    scores={};metrics=[];monthly=[];group_rows=[];daily_parts=[]
    for model in MODELS:
        p=align_predictions(frame,pd.read_parquet(out/(model+'_holdout_predictions.parquet')))
        scores[model]=p
        for policy in [r for r in plan['policies'] if r['model']==model]:
            row=dict(model=NAMES[model],policy=policy['policy'],threshold=policy['threshold'],budget=policy['budget'],**sf.score(frame,p,policy['threshold']))
            cost='wrong-alert hours / day' if policy['policy']=='duration' else 'false alarms / day'
            row['budget_exceeded']=None if policy['budget'] is None else row[cost]>policy['budget']+1e-12
            metrics.append(row)
            for month,positions in frame.groupby(frame.timestamp.dt.to_period('M')).groups.items():
                positions=np.asarray(positions)
                monthly.append(dict(model=NAMES[model],policy=policy['policy'],month=str(month),threshold=policy['threshold'],**sf.score(frame.iloc[positions],p[positions],policy['threshold'])))
            onsets=frame.loc[frame.onset,['station','timestamp']]
            leads=warning_leads(frame,p,onsets,policy['threshold'])
            wanted=pd.MultiIndex.from_arrays([onsets.station,onsets.timestamp-pd.Timedelta(minutes=15)])
            recurrence=frame.set_index(sf.KEY).typical_congestion_30.reindex(wanted).to_numpy()
            groups=np.where(~np.isfinite(recurrence),'unknown',np.where(recurrence<.1,'non-recurrent',np.where(recurrence<.4,'occasional','recurrent')))
            for group in ['non-recurrent','occasional','recurrent','unknown']:
                use=groups==group;n=int(use.sum());caught=int(((leads>=15)&use).sum())
                group_rows.append(dict(model=NAMES[model],policy=policy['policy'],group=group,onsets=n,caught=caught,catch_rate=caught/n if n else None))
            if policy['policy']=='duration':
                wrong=(p>=policy['threshold']) & frame.y_known.to_numpy() & ~frame.y.to_numpy()
                adjacent=np.r_[False,(frame.station.to_numpy()[1:]==frame.station.to_numpy()[:-1])&(np.diff(frame.timestamp.to_numpy(dtype='datetime64[ns]'))==np.timedelta64(5,'m'))]
                starts=wrong & ~(np.r_[False,wrong[:-1]]&adjacent)
                costs=pd.DataFrame(dict(date=pd.to_datetime(frame.date).dt.normalize(),wrong_rows=wrong.astype(int),false_episodes=starts.astype(int))).groupby('date').sum()
                rare=groups=='non-recurrent'
                events=pd.DataFrame(dict(date=onsets.timestamp.dt.normalize().to_numpy(),onsets=1,caught=(leads>=15).astype(int),rare_onsets=rare.astype(int),rare_caught=(rare&(leads>=15)).astype(int))).groupby('date').sum()
                daily=costs.join(events,how='left').fillna(0).reset_index()
                daily['days']=1;daily['fold']=daily.date.dt.to_period('Q').astype(str);daily['model']=NAMES[model]
                assert daily.onsets.sum()==int(frame.onset.sum())
                assert np.isclose(daily.caught.sum()/daily.onsets.sum(),row['caught '+chr(8805)+'15 min'])
                daily_parts.append(daily)
    metrics=pd.DataFrame(metrics);monthly=pd.DataFrame(monthly);groups=pd.DataFrame(group_rows)
    daily=pd.concat(daily_parts,ignore_index=True)
    intervals,_=paired_bootstrap(daily,'Transformer','Boosted trees',repetitions=5000,seed=42)
    for name,value in [('holdout_metrics',metrics),('monthly_metrics',monthly),('recurrence_metrics',groups),('daily_duration_counts',daily),('paired_duration_intervals',intervals)]:value.to_csv(out/(name+'.csv'),index=False)
    colors={'Random Forest':'#2878b5','Boosted trees':'#1b9e77','Transformer':'#e87932'}
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    eligible=frame.y_known & frame.real & ~frame.congested_now
    for model,p in scores.items():
        name=NAMES[model]
        precision,recall,_=precision_recall_curve(frame.loc[eligible,'y'],p[eligible])
        ap=metrics.loc[metrics.model.eq(name),'AP, onsets'].iloc[0]
        axes[0].plot(recall,precision,label=f'{name} (AP {ap:.3f})',color=colors[name])
        m=monthly.loc[monthly.model.eq(name)&monthly.policy.eq('duration')]
        axes[1].plot(m.month,m['AP, onsets'],marker='o',label=name,color=colors[name])
    axes[0].axhline(frame.loc[eligible,'y'].mean(),color='gray',linestyle='--',label='Positive prevalence')
    axes[0].set(xlabel='Recall on eligible forecast origins',ylabel='Precision',title='Holdout probability ranking');axes[0].legend(fontsize=8)
    axes[1].set(title='Monthly AP: same frozen checkpoints',ylabel='Average precision');axes[1].tick_params(axis='x',rotation=25)
    fig.savefig(out/'holdout_ranking.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    for policy,marker in [('duration','o'),('episodes','s'),('default_0.5','^')]:
        for r in metrics.loc[metrics.policy.eq(policy)].to_dict('records'):
            catch=100*r['caught '+chr(8805)+'15 min']
            axes[0].scatter(r['wrong-alert hours / day'],catch,color=colors[r['model']],marker=marker,s=65)
            axes[1].scatter(r['false alarms / day'],catch,color=colors[r['model']],marker=marker,s=65)
    from matplotlib.lines import Line2D
    handles=[Line2D([0],[0],color=c,marker='o',linestyle='',label=m) for m,c in colors.items()]
    handles+=[Line2D([0],[0],color='gray',marker=marker,linestyle='',label=p) for p,marker in [('duration policy','o'),('episode policy','s'),('threshold 0.5','^')]]
    axes[0].legend(handles=handles,fontsize=7)
    axes[0].axvline(4,color='gray',linestyle='--');axes[1].axvline(20,color='gray',linestyle='--')
    axes[0].set(xlabel='Realized wrong-alert hours/day',ylabel='Onsets warned at least 15 min early (%)',title='Frozen warning policies')
    axes[1].set(xlabel='Realized false episodes/day',ylabel='Onsets warned at least 15 min early (%)',title='Episode and duration tradeoffs')
    for name in colors:
        g=groups.loc[groups.model.eq(name)&groups.policy.eq('duration')].set_index('group')
        axes[2].plot(['non-recurrent','occasional','recurrent'],100*g.reindex(['non-recurrent','occasional','recurrent']).catch_rate,marker='o',color=colors[name],label=name)
    axes[2].set(ylabel='Early catch (%)',title='Duration-policy recurrence groups');axes[2].tick_params(axis='x',rotation=25)
    fig.savefig(out/'holdout_warning_tradeoffs.png',dpi=160);plt.close(fig)
    write(out/'evaluation_manifest.json',dict(status='complete',holdout_used=True,holdout_start=START,holdout_end=END,
        completed_at_utc=datetime.now(timezone.utc).isoformat(),protocol_sha256=digest(out/'frozen_protocol.json'),
        models=MODELS,refitted=False,thresholds_changed_after_access=False,rows=len(frame),onsets=int(frame.onset.sum()),days=frame.date.nunique(),
        calibration_feature_reproduction=True,team_summary_modified=False,
        note='Holdout has now been evaluated and must not be described as untouched. Future tuning requires a new independent test period.'))
    print(metrics.to_string(index=False),flush=True)
    print('Completed frozen-checkpoint holdout evaluation:',out,flush=True)

def main():
    import warnings
    warnings.filterwarnings('once',category=FutureWarning)
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['all','predict','report'],default='all')
    parser.add_argument('--model',choices=MODELS)
    parser.add_argument('--run-dir',type=Path)
    args=parser.parse_args()
    if args.stage=='report':
        plan=json.loads((args.run_dir/'frozen_protocol.json').read_text());report(args.run_dir,plan);return
    if args.stage=='predict':
        plan=json.loads((args.run_dir/'frozen_protocol.json').read_text());predict(args.run_dir,plan,args.model);return
    out=ROOT/'results/holdout_evaluation'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    (out/'inputs').mkdir(parents=True)
    (out.parent/'.gitignore').write_text('*/inputs/\n',encoding='utf-8')
    summary_before=(ROOT/'notebooks/results/summary.csv').read_bytes()
    source=ROOT/'results/local_model_benchmark/20261104_115016_080613'
    plan,development=freeze(out,source)
    prepare(out,plan,development)
    del development;gc.collect()
    env=os.environ.copy()
    for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:env[name]='4'
    env['PYTHONIOENCODING']='utf-8'
    for model in MODELS:
        print('Predicting frozen holdout:',model,flush=True)
        subprocess.run([sys.executable,'-u','-m','src.analysis.evaluate_holdout','--stage','predict','--model',model,'--run-dir',str(out)],cwd=ROOT,env=env,check=True)
    report(out,plan)
    assert (ROOT/'notebooks/results/summary.csv').read_bytes()==summary_before

if __name__=='__main__':main()
