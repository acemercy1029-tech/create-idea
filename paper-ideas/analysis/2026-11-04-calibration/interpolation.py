"""Frozen ordinary interpolation, evaluated from held-out fixed-policy means.

No independent new simulation or confidence intervals are claimed here.
"""
import sys
import csv
import numpy as np
from scipy.stats import binom
import run_calibration as r

def read(name):
    with (r.OUT/name).open(encoding='utf-8') as f:return list(csv.DictReader(f))

if '--freeze' in sys.argv:
    train=read('training_summary.csv')
    decisions=[]
    for c in [256,1024]:
        estimates=np.clip((2*np.arange(c+1)/c-1)/r.d.KAPPA,0,1)
        for th in [False,True]:
            pred=[]
            for policy in r.POLICIES:
                values=[]
                for lam in r.GRID:
                    rows=[z for z in train if float(z['lam'])==lam and int(z['calibration_users'])==c and z['threshold']==str(th) and z['policy']==policy and z['period']=='full']
                    assert len(rows)==4
                    values.append(np.mean([float(z['mse']) for z in rows]))
                pred.append(np.interp(estimates,r.GRID,values))
            chosen=np.argmin(np.asarray(pred),axis=0)
            for k,p in enumerate(chosen):
                decisions.append(dict(calibration_users=c,threshold=th,positive_reports=k,estimated_lambda=estimates[k],policy=list(r.POLICIES)[p]))
    r.save('frozen_interpolation_decisions.csv',decisions)
    print('Frozen interpolation decisions using training only:',len(decisions))
    raise SystemExit

decisions=read('frozen_interpolation_decisions.csv')
fixed=read('fixed_validation.csv')
idx={(float(z['lam']),z['scene'],int(z['calibration_users']),z['threshold'],z['policy'],z['period']):float(z['mse']) for z in fixed}
rows=[];choices=[]
for lam in r.HOLDOUT:
    for scene in r.SCENES:
        for c in [256,1024]:
            rawmean=lam+(1-lam)*.36/r.d.M if scene=='constant' else lam
            pmf=binom.pmf(np.arange(c+1),c,(1+r.d.KAPPA*rawmean)/2)
            pmf/=pmf.sum()
            for th in ['False','True']:
                table=[z for z in decisions if int(z['calibration_users'])==c and z['threshold']==th]
                assert len(table)==c+1
                weights={p:0. for p in r.POLICIES}
                for z in table:weights[z['policy']]+=pmf[int(z['positive_reports'])]
                for p,w in weights.items():choices.append(dict(lam=lam,scene=scene,calibration_users=c,threshold=th,policy=p,probability=w))
                periods=sorted(set(z['period'] for z in fixed if float(z['lam'])==lam and z['scene']==scene))
                for period in periods:
                    mean=sum(w*idx[(lam,scene,c,th,p,period)] for p,w in weights.items())
                    rows.append(dict(lam=lam,scene=scene,calibration_users=c,threshold=th,period=period,mse=mean))
r.save('interpolation_scenes.csv',rows);r.save('interpolation_selection_probabilities.csv',choices)
summary=[]
for scope,lams in [('training_grid',r.GRID),('unseen_lambda',[.15,.45,.7])]:
    for c in [256,1024]:
        for th in ['False','True']:
            selected=[z['mse'] for z in rows if z['lam'] in lams and z['calibration_users']==c and z['threshold']==th and z['period']=='full']
            assert len(selected)==len(lams)*4
            summary.append(dict(scope=scope,calibration_users=c,threshold=th,mse=np.mean(selected),note='Frozen on training; derived from same held-out fixed-policy risks; no new independent experiment or CI'))
r.save('interpolation_aggregate.csv',summary)
for z in summary:print(z)
