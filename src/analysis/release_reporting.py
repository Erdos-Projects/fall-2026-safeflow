"""Regenerate final figures and score tables from packaged immutable predictions."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from src.features import shared as sf
from .study_release import ROOT,MODELS,NAMES,config,dump,digest


def generate(output=None,verify=True,source=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve,roc_curve,roc_auc_score
    cfg=config();base=Path(source) if source else ROOT/cfg['release']['results'];out=Path(output) if output else base
    out.mkdir(parents=True,exist_ok=True);figures=out/'figures';figures.mkdir(exist_ok=True)
    policies=pd.read_csv(base/'holdout/frozen_policies.csv',float_precision='round_trip')
    expected=pd.read_csv(base/'holdout/holdout_metrics.csv')
    colors={'random_forest':'#2878b5','boosted_trees':'#1b9e77','transformer':'#e87932'}
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    ranking,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    calibration,calax=plt.subplots(figsize=(7,5),layout='constrained')
    calax.plot([0,1],[0,1],'--',color='gray',label='Perfect reliability')
    metrics=[];calibration_rows=[]
    for model in MODELS:
        frame=pd.read_parquet(base/'holdout'/f'{model}_holdout_predictions.parquet')
        p=frame.probability.to_numpy(dtype=float)
        use=(frame.y_known&frame.real&~frame.congested_now).to_numpy()
        y=frame.y.to_numpy()[use];q=p[use]
        precision,recall,_=precision_recall_curve(y,q)
        fpr,tpr,_=roc_curve(y,q)
        for row in policies.loc[policies.model.eq(model)].itertuples():
            score=sf.score(frame,p,row.threshold)
            actual=dict(model=NAMES[model],policy=row.policy,threshold=row.threshold,**score)
            if verify:
                saved=expected.loc[expected.model.eq(NAMES[model])&expected.policy.eq(row.policy)].iloc[0]
                for metric,value in score.items():
                    if not np.isclose(value,saved[metric],atol=1e-10,rtol=1e-9,equal_nan=True):
                        raise AssertionError(f'Released score changed: {model}/{row.policy}/{metric}')
            metrics.append(actual)
        ap=metrics[-1]['AP, onsets']
        axes[0].plot(recall,precision,label=f'{NAMES[model]} AP {ap:.3f}',color=colors[model])
        axes[1].plot(fpr,tpr,label=f'{NAMES[model]} AUC {roc_auc_score(y,q):.3f}',color=colors[model])
        bins=np.minimum((q*10).astype(int),9)
        part=pd.DataFrame(dict(bin=bins,probability=q,outcome=y)).groupby('bin').agg(mean_probability=('probability','mean'),observed_rate=('outcome','mean'),count=('outcome','size')).reset_index()
        part['model']=NAMES[model];part['bin_lower']=part.bin/10;part['bin_upper']=(part.bin+1)/10
        calibration_rows.extend(part.to_dict('records'))
        calax.plot(part.mean_probability,part.observed_rate,marker='o',color=colors[model],label=NAMES[model])
    axes[0].axhline(y.mean(),color='gray',ls='--',label='Positive prevalence')
    axes[1].plot([0,1],[0,1],'--',color='gray')
    axes[0].set(title='Holdout precision-recall',xlabel='Recall',ylabel='Precision')
    axes[1].set(title='Holdout ROC (supplementary)',xlabel='False positive rate',ylabel='True positive rate')
    for ax in axes:ax.legend(fontsize=8)
    ranking.savefig(figures/'holdout_ranking.png',dpi=160);plt.close(ranking)
    calax.set(title='Holdout reliability: eligible uncongested origins',xlabel='Mean predicted probability',ylabel='Observed positive fraction',xlim=(0,1),ylim=(0,1));calax.legend(fontsize=9)
    calibration.savefig(figures/'holdout_calibration.png',dpi=160);plt.close(calibration)
    pd.DataFrame(metrics).to_csv(out/'recomputed_holdout_metrics.csv',index=False)
    pd.DataFrame(calibration_rows).to_csv(out/'calibration.csv',index=False)
    # Warning rules and their burden targets, not hard constraints imposed on holdout.
    caught='caught '+chr(8805)+'15 min'
    duration=expected.loc[expected.policy.eq('duration')].set_index('model').reindex([NAMES[m] for m in MODELS])
    groups=pd.read_csv(base/'holdout/recurrence_metrics.csv')
    figure,axs=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    c=[colors[m] for m in MODELS]
    for ax,column,title,line in [(axs[0],caught,'Onsets warned at least 15 min early',None),(axs[1],'wrong-alert hours / day','Realized wrong-alert hours/day',4)]:
        vals=duration[column]*(100 if column==caught else 1)
        bars=ax.bar(range(3),vals,color=c)
        ax.bar_label(bars,fmt='%.1f',padding=3)
        ax.set_xticks(range(3),[NAMES[m] for m in MODELS],rotation=20,ha='right');ax.set_title(title,fontsize=10)
        ax.set_ylim(0,vals.max()*1.2)
        if line:ax.axhline(line,ls='--',color='gray',label='Calibration budget: 4');ax.legend(fontsize=8)
    for model in MODELS:
        g=groups.loc[groups.model.eq(NAMES[model])&groups.policy.eq('duration')].set_index('group')
        axs[2].plot(['Non-recurrent','Occasional','Recurrent'],100*g.reindex(['non-recurrent','occasional','recurrent']).catch_rate,marker='o',color=colors[model],label=NAMES[model])
    axs[2].set_title('Early detection by recurrence group',fontsize=10);axs[2].set_ylabel('Early catch (%)');axs[2].tick_params(axis='x',rotation=20);axs[2].legend(fontsize=8)
    figure.suptitle('Frozen duration policy: target burden calibrated before holdout; all models exceed it')
    figure.savefig(figures/'holdout_errors_and_budget.png',dpi=160);plt.close(figure)
    # Regenerate development context and operational comparisons from packaged tables.
    published=pd.read_csv(base/'development/summary.csv')
    fig,axs=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    identifiers={'random_forest':'random_forest_shared_29_features','boosted_trees':'boosted_trees','transformer':'transformer_shared_plus_agreed_features'}
    for i,m in enumerate(MODELS):
        r=published.loc[published.model.eq(identifiers[m])].iloc[0]
        axs[0].errorbar(r['AP onsets (mean)'],i,xerr=r['AP onsets (std)'],fmt='o',color=colors[m],capsize=5)
        folds=pd.read_csv(base/'development'/f'{identifiers[m]}_folds.csv')
        axs[1].plot(range(6),folds['AP, onsets'],marker='o',color=colors[m],label=NAMES[m])
    axs[0].set_yticks(range(3),[NAMES[m] for m in MODELS]);axs[0].set_xlabel('Mean AP; whiskers = fold SD, not confidence intervals',fontsize=9)
    axs[0].set_title('Historical published development ranking')
    axs[1].set_xticks(range(6),['2024 Q1','Q2','Q3','Q4','2025 Q1','Q2']);axs[1].set_ylabel('Average precision');axs[1].legend(fontsize=8)
    fig.savefig(figures/'development_ranking.png',dpi=160);plt.close(fig)
    training=pd.read_csv(base/'operational_benchmark/training.csv').set_index('model')
    inference=pd.read_csv(base/'operational_benchmark/inference.csv').set_index('model')
    latency=pd.read_csv(base/'operational_benchmark/latency.csv')
    fig,axs=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    series=[training.reindex(MODELS).fit_predict_save_seconds,
        latency.loc[latency.scope.eq('batch_build_plus_model')&latency.batch_size.eq(46)].set_index('model').reindex(MODELS).median_ms,
        inference.reindex(MODELS).peak_rss_bytes/2**20]
    for ax,values,title in zip(axs,series,['Fit/predict/save seconds (log)','46-row warm batch median (ms)','Inference process peak RSS (MiB)']):
        bars=ax.bar(range(3),values,color=c);ax.bar_label(bars,fmt='%.1f',padding=3)
        ax.set_xticks(range(3),[NAMES[m] for m in MODELS],rotation=20,ha='right');ax.set_title(title,fontsize=10)
    axs[0].set_yscale('log');axs[0].set_ylim(10,training.fit_predict_save_seconds.max()*2)
    fig.savefig(figures/'operational_benchmark.png',dpi=160);plt.close(fig)
    # Additional public-cover plots use only the packaged historical tables.
    forward=pd.read_csv(base/'development/forward/quarterly_policies.csv')
    fc={'Random Forest':colors['random_forest'],'Transformer':colors['transformer'],'Boosted trees (29)':colors['boosted_trees'],'Timetable':'gray'}
    fig,axs=plt.subplots(2,2,figsize=(12,7),layout='constrained')
    for col,(policy,cost,budget) in enumerate([('episodes','false_episodes_per_day',20),('duration','wrong_alert_hours_per_day',4)]):
        for name,c in fc.items():
            f=forward.loc[forward.model.eq(name)&forward.budget_type.eq(policy)]
            x=range(len(f));axs[0,col].plot(x,100*f.caught_15min,marker='o',color=c,label=name)
            axs[1,col].plot(x,f[cost],marker='o',color=c)
        axs[0,col].set(title=f'{policy.capitalize()} calibration policy',ylabel='Onsets caught early (%)')
        axs[1,col].axhline(budget,color='gray',ls='--');axs[1,col].set_ylabel('False episodes/day' if policy=='episodes' else 'Wrong-alert hours/day')
        for ax in axs[:,col]:ax.set_xticks(range(5),['2024 Q2','Q3','Q4','2025 Q1','Q2'])
    axs[0,0].legend(fontsize=8);fig.savefig(figures/'forward_warning_budget.png',dpi=160);plt.close(fig)
    intervals=pd.read_csv(base/'development/uncertainty/paired_intervals.csv')
    fig,axs=plt.subplots(1,3,figsize=(12,3.5),layout='constrained')
    for ax,(metric,scale,label) in zip(axs,[('catch_15min',100,'Early catch difference (pp)'),('false_episodes_per_day',1,'False episodes/day difference'),('wrong_alert_hours_per_day',1,'Wrong hours/day difference')]):
        r=intervals.loc[intervals.model_b.eq('Boosted trees (29)')&intervals.budget_type.eq('duration')&intervals.block.eq('week')&intervals.metric.eq(metric)].iloc[0]
        m,lo,hi=scale*r.difference,scale*r.ci_low,scale*r.ci_high
        ax.axvline(0,color='gray',ls='--');ax.errorbar(m,0,xerr=[[m-lo],[hi-m]],fmt='o',capsize=6,color=colors['transformer']);ax.set_yticks([]);ax.set_xlabel(label)
        ax.set_title(f'{m:+.3f} [{lo:+.3f}, {hi:+.3f}]',fontsize=10)
    fig.suptitle('Development duration policy: Transformer minus boosted trees; paired weekly 95% intervals')
    fig.savefig(figures/'paired_uncertainty.png',dpi=160);plt.close(fig)
    recurrence=pd.read_csv(base/'development/forward/recurrence_summary.csv')
    fig,ax=plt.subplots(figsize=(10,4),layout='constrained')
    for i,model in enumerate(MODELS):
        name='Boosted trees (29)' if model=='boosted_trees' else NAMES[model]
        g=recurrence.loc[recurrence.model.eq(name)&recurrence.budget_type.eq('duration')].set_index('group')
        v=100*g.reindex(['non-recurrent','occasional','recurrent']).catch_15min
        bars=ax.bar(np.arange(3)+(i-1)*.24,v,width=.22,color=colors[model],label=NAMES[model]);ax.bar_label(bars,fmt='%.1f',padding=3)
    ax.set_xticks(range(3),['Non-recurrent','Occasional','Recurrent']);ax.set_ylim(0,100);ax.set_ylabel('Onsets caught early (%)');ax.legend(fontsize=8)
    ax.set_title('Development forward duration policy by recurrence group')
    fig.savefig(figures/'development_recurrence.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    shapes={'duration':'o','episodes':'s','default_0.5':'^'}
    for r in expected.to_dict('records'):
        model=next(m for m in MODELS if NAMES[m]==r['model'])
        for ax,cost in zip(axs,['wrong-alert hours / day','false alarms / day']):
            ax.scatter(r[cost],100*r[caught],marker=shapes[r['policy']],color=colors[model],s=70)
    from matplotlib.lines import Line2D
    handles=[Line2D([0],[0],color=colors[m],marker='o',linestyle='',label=NAMES[m]) for m in MODELS]
    handles+=[Line2D([0],[0],color='gray',marker=marker,linestyle='',label=policy) for policy,marker in shapes.items()]
    axs[0].legend(handles=handles,fontsize=7)
    for ax,xlabel,budget in zip(axs,['Wrong-alert hours/day','False episodes/day'],[4,20]):
        ax.axvline(budget,ls='--',color='gray');ax.set(xlabel=xlabel,ylabel='Onsets caught at least 15 min early (%)')
    fig.suptitle('Holdout: color = model, shape = warning policy; dashed lines = calibration budgets')
    fig.savefig(figures/'holdout_warning_tradeoffs.png',dpi=160);plt.close(fig)
    dump(out/'report_manifest.json',{'score_reproduction_passed':True,'selected_final_model':None,
        'input_release_manifest_sha256':digest(base/'manifest.json'),
        'generator_sha256':digest(Path(__file__)),
        'figure_sha256':{p.name:digest(p) for p in sorted(figures.glob('*.png'))},
        'figures':[p.name for p in sorted(figures.glob('*.png'))],
        'historical_cv':'Imported published evidence, not reconstructed historical training fits',
        'calibration_plot':'Descriptive reliability only; no calibrator fitted to holdout'})
    print('Generated final report figures and reproduced all 9 holdout metric rows:',out)
    return out
