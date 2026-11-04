"""Ordinary private calibration with reserved users; full-horizon expected risk.

Correct marginal distributions, cached Monte Carlo integration, not trajectories.
Calibration and final-publication randomization are integrated analytically.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
import sys
import csv
import json
import time
import numpy as np
from scipy.stats import binom
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(OUT.parent/'2026-11-03-allocation-grid'))
import run_grid as g
d=g.d
GRID=[0.,.3,.6,.8]
HOLDOUT=[0.,.3,.6,.8,.15,.45,.7]
COSTS=[0,256,1024]
SCENES=['constant','migration8','migration32','migration96']
POLICIES={'local16':('local',16,10240),'local32':('local',32,10240),
          'local64':('local',64,10240),'local128':('local',128,10240),
          'server32':('server',32,15360),'server64':('server',64,15360),
          'server128':('server',128,10240)}

def save(name,rows):
    with (OUT/name).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def nphase(n,stride,t):
    return n//stride+int(t%stride<n%stride)

def nearest(value):
    return int(np.argmin(abs(np.asarray(GRID)-value)))

def calibration_probs(lam,scene,c):
    rawmean=lam+(1-lam)*.36/d.M if scene=='constant' else lam
    pmf=binom.pmf(np.arange(c+1),c,(1+d.KAPPA*rawmean)/2)
    estimate=np.clip((2*np.arange(c+1)/c-1)/d.KAPPA,0,1)
    choice=np.argmin(abs(estimate[:,None]-np.asarray(GRID)[None,:]),axis=1)
    answer=np.bincount(choice,weights=pmf,minlength=4)
    assert abs(answer.sum()-1)<1e-10
    return answer/answer.sum()

def truth(scene):
    pa,pb=np.full(d.T,.5),np.full(d.T,.5)
    if scene=='constant':pa[:]=.8
    else:
        duration=int(scene[len('migration'):])
        pa[96:160]=.8;pb[160:160+duration]=.8
    return pa,pb

class Selector:
    def __init__(self,lam,reps,seed,li):
        self.lam,self.reps=lam,reps
        self.rng=np.random.default_rng(np.random.SeedSequence([seed,li]))
        self.cache={}
        self.hits=0
    def chosen(self,policy,pa,pb,t,start):
        kind,size,ns=POLICIES[policy]
        if t<start or (kind=='local' and t<size-1):return None
        if kind=='local':
            ends=range(max(start,size-1,t-3),t+1)
            data=tuple((nphase(ns,d.W+size-1,end),tuple(pa[end-size+1:end+1]),tuple(pb[end-size+1:end+1])) for end in ends)
            key=(policy,data)
        else:
            begin=max(start,t-size+1)
            data=(tuple(pa[begin:t+1]),tuple(pb[begin:t+1]))
            key=(policy,data)
        if key in self.cache:
            self.hits+=1;return self.cache[key]
        if kind=='local':
            aggregate=np.zeros((self.reps,d.M))
            for n,aa,bb in data:
                qa,qb=d.winners(d.raw_counts(aa,self.lam),d.raw_counts(bb,self.lam),self.lam)
                vote=np.full(d.M,(1-qa-qb)/(d.M-2));vote[d.A],vote[d.B]=qa,qb
                mu=d.H@vote
                probs=np.r_[1+d.KAPPA*mu,1-d.KAPPA*mu]/(2*d.M)
                count=self.rng.multinomial(n,probs/probs.sum(),size=self.reps)
                aggregate+=count[:,:d.M]-count[:,d.M:]
            pick=d.iid.argmax_random(d.iid.fwht(aggregate),self.rng)
        else:
            k=ns//(d.W*d.M)
            pmfs=[g.server_counts(z,self.lam,k) for z in [data[0],data[1],(.5,)*len(data[0])]]
            count=self.rng.choice(len(pmfs[2]),size=(self.reps,d.M),p=pmfs[2])
            count[:,d.A]=self.rng.choice(len(pmfs[0]),size=self.reps,p=pmfs[0])
            count[:,d.B]=self.rng.choice(len(pmfs[1]),size=self.reps,p=pmfs[1])
            pick=d.iid.argmax_random(2*count-k*len(data[0]),self.rng,absolute=True)
        self.cache[key]=pick.astype(np.int16)
        return self.cache[key]

def fixed_risks(selector,scene):
    pa,pb=truth(scene)
    active=(pa!=.5)|(pb!=.5)
    masks={'full':np.ones(d.T,dtype=bool),'startup':np.arange(d.T)<64,
           'late':np.arange(d.T)>=256}
    if scene!='constant':
        duration=int(scene[len('migration'):])
        masks.update(old=(np.arange(d.T)>=96)&(np.arange(d.T)<160),
                     new=(np.arange(d.T)>=160)&(np.arange(d.T)<160+duration),
                     new_first8=(np.arange(d.T)>=160)&(np.arange(d.T)<168))
    output={}
    for policy,(kind,size,ns) in POLICIES.items():
        for c in COSTS:
            start=0 if c==0 else 2
            risks={th:np.zeros((selector.reps,d.T)) for th in [False,True]}
            for t in range(d.T):
                available=t>=start and (kind=='server' or t>=size-1)
                if not available:
                    for th in risks:risks[th][:,t]=.36 if active[t] else 0.
                    continue
                n=nphase(d.N-c-ns,d.W,t)
                if active[t]:
                    target=d.A if pa[t]!=.5 else d.B
                    correct=selector.chosen(policy,pa,pb,t,start)==target
                    for th in risks:risks[th][:,t]=g.single_risk(correct,n,th)
                else:
                    for th in risks:risks[th][:,t]=g.pub_moments(0.,n,th)[1]
            for th in risks:
                for period,mask in masks.items():
                    output[(c,th,policy,period)]=risks[th][:,mask].mean(axis=1)
    return output,masks

def row_from(vals,**labels):
    mean,se=float(vals.mean()),float(vals.std(ddof=1)/np.sqrt(len(vals)))
    return dict(**labels,mse=mean,mc_se=se,low=mean-1.96*se,high=mean+1.96*se)

def validate():
    rows=[]
    assert abs(np.log((1+d.KAPPA)/(1-d.KAPPA))-1)<1e-12
    for lam in HOLDOUT:
        for p in [.5,.8]:
            matrix=d.trans(p,p,lam)
            exact=sum((1-p if z0==0 else p)*matrix[z0,z1]*(2*z0-1)*(2*z1-1) for z0 in [0,1] for z1 in [0,1])
            expected=lam+(1-lam)*(2*p-1)**2
            assert abs(exact-expected)<1e-12
            rows.append(dict(test='markov_product_mean',setting=str((lam,p)),error=abs(exact-expected),tolerance=1e-12))
    rng=np.random.default_rng(2026110499)
    reps=100000
    for lam in [0.,.15,.3,.45,.6,.7,.8]:
        for scene in ['constant','migration96']:
            for c in [256,1024]:
                exact=calibration_probs(lam,scene,c)
                rawmean=lam+(1-lam)*.36/d.M if scene=='constant' else lam
                count=rng.binomial(c,(1+d.KAPPA*rawmean)/2,reps)
                est=np.clip((2*count/c-1)/d.KAPPA,0,1)
                chosen=np.argmin(abs(est[:,None]-np.asarray(GRID)[None,:]),axis=1)
                empirical=np.bincount(chosen,minlength=4)/reps
                tolerance=6*np.sqrt(exact*(1-exact)/reps)+6/reps
                assert np.all(abs(empirical-exact)<tolerance)
                rows.append(dict(test='calibration_decision_law',setting=str((lam,scene,c)),error=float(max(abs(empirical-exact))),tolerance=float(max(tolerance))))
    for c in COSTS:
        for policy,(kind,size,ns) in POLICIES.items():
            npub=d.N-c-ns
            assert sum(nphase(npub,d.W,t) for t in range(d.W))==npub
            stride=d.W+size-1 if kind=='local' else d.W
            assert sum(nphase(ns,stride,t) for t in range(stride))==ns
            maxbits=0
            for t in range(d.T):
                available=t>=(2 if c else 0) and (kind=='server' or t>=size-1)
                bits=c if c and t==1 else 0
                if available:bits+=nphase(ns,stride,t)+nphase(npub,d.W,t)
                maxbits=max(maxbits,bits)
            assert maxbits<=1024
            rows.append(dict(test='round_uplink_and_disjoint_population',setting=str((c,policy)),error=maxbits,tolerance=1024))
    save('kernel_checks.csv',rows)
    print('New checks passed',len(rows),flush=True)

def main():
    started=time.monotonic();validate()
    training=[];average={};cache=[]
    for li,lam in enumerate(GRID):
        sel=Selector(lam,300,2026110418,li)
        grouped={}
        for scene in SCENES:
            risk,masks=fixed_risks(sel,scene)
            for (c,th,policy,period),vals in risk.items():
                training.append(row_from(vals,lam=lam,scene=scene,calibration_users=c,threshold=th,policy=policy,period=period))
                if period=='full':grouped.setdefault((c,th,policy),[]).append(vals)
            print('Training',lam,scene,'kernels',len(sel.cache),'seconds',round(time.monotonic()-started,1),flush=True)
        for key,vals in grouped.items():average[(lam,)+key]=np.mean(vals,axis=0)
        cache.append(dict(stage='training',lam=lam,repeats=300,unique_selection_laws=len(sel.cache),cache_hits=sel.hits))
        save('training_summary.csv',training)
    lookup=[];maps={}
    for c in COSTS:
        for th in [False,True]:
            for lam in GRID:
                best=min(POLICIES,key=lambda policy:average[(lam,c,th,policy)].mean())
                maps[(c,th,lam)]=best
                lookup.append(dict(calibration_users=c,threshold=th,training_lambda=lam,policy=best,training_full_risk=float(average[(lam,c,th,best)].mean())))
    baselines={th:min(POLICIES,key=lambda p:np.mean([average[(lam,0,th,p)].mean() for lam in GRID])) for th in [False,True]}
    save('frozen_lookup.csv',lookup)
    (OUT/'frozen_baselines.json').write_text(json.dumps({str(k):v for k,v in baselines.items()},indent=2),encoding='utf-8')
    results=[];differences=[];summary=[];overall={};calrows=[];fixed=[]
    for li,lam in enumerate(HOLDOUT):
        sel=Selector(lam,1500,2026110419,li)
        group={};gridlam=GRID[nearest(lam)]
        for scene in SCENES:
            risk,masks=fixed_risks(sel,scene)
            for (c,th,policy,period),vals in risk.items():
                fixed.append(row_from(vals,lam=lam,scene=scene,calibration_users=c,threshold=th,policy=policy,period=period))
            for c in [256,1024]:
                prob=calibration_probs(lam,scene,c)
                calrows.append(dict(lam=lam,scene=scene,calibration_users=c,prob_grid0=prob[0],prob_grid03=prob[1],prob_grid06=prob[2],prob_grid08=prob[3],nearest_grid_accuracy=prob[nearest(lam)]))
            for th in [False,True]:
                for period,mask in masks.items():
                    values={'fixed_full_population':risk[(0,th,baselines[th],period)],
                            'known_grid_no_cost':risk[(0,th,maps[(0,th,gridlam)],period)],
                            'zero':np.full(sel.reps,.36*float(((truth(scene)[0]!=.5)|(truth(scene)[1]!=.5))[mask].mean()))}
                    for c in [256,1024]:
                        prob=calibration_probs(lam,scene,c)
                        values['private_cal'+str(c)]=sum(prob[i]*risk[(c,th,maps[(c,th,z)],period)] for i,z in enumerate(GRID))
                        values['known_grid_cost'+str(c)]=risk[(c,th,maps[(c,th,gridlam)],period)]
                    for name,vals in values.items():
                        results.append(row_from(vals,lam=lam,scene=scene,threshold=th,method=name,period=period))
                        if period=='full':group.setdefault((th,name),[]).append(vals)
                    for c in [256,1024]:
                        pairs=[('private_cal'+str(c),'fixed_full_population'),('private_cal'+str(c),'known_grid_cost'+str(c)),('known_grid_cost'+str(c),'known_grid_no_cost')]
                        for left,right in pairs:
                            differences.append(row_from(values[left]-values[right],lam=lam,scene=scene,threshold=th,method=left,reference=right,period=period))
            print('Validation',lam,scene,'kernels',len(sel.cache),'seconds',round(time.monotonic()-started,1),flush=True)
        for key,vals in group.items():
            avg=np.mean(vals,axis=0);overall[(lam,)+key]=avg
            summary.append(row_from(avg,lam=lam,threshold=key[0],method=key[1],period='four_scenes_full_horizon'))
        cache.append(dict(stage='validation',lam=lam,repeats=1500,unique_selection_laws=len(sel.cache),cache_hits=sel.hits))
        save('validation_scenes.csv',results);save('validation_differences.csv',differences)
        save('validation_summary.csv',summary);save('fixed_validation.csv',fixed)
    aggregate=[]
    for scope,lams in [('training_grid',GRID),('unseen_lambda',[.15,.45,.7])]:
        for th in [False,True]:
            names=[name for flag,name in group if flag==th]
            for name in names:
                per=[overall[(lam,th,name)] for lam in lams]
                avg=np.mean(per,axis=0)
                # Independent RNGs across lambda; same replicate indexing forms
                # a legitimate independent sum for Monte Carlo integration SE.
                aggregate.append(row_from(avg,scope=scope,threshold=th,method=name,kind='risk',reference=''))
            for c in [256,1024]:
                left='private_cal'+str(c)
                for right in ['fixed_full_population','known_grid_cost'+str(c),'known_grid_no_cost']:
                    avg=np.mean([overall[(lam,th,left)]-overall[(lam,th,right)] for lam in lams],axis=0)
                    aggregate.append(row_from(avg,scope=scope,threshold=th,method=left,kind='difference',reference=right))
    save('aggregate.csv',aggregate);save('calibration_probabilities.csv',calrows);save('cache_metadata.csv',cache)
    meta=dict(client_date='2026-11-04',training_seed=2026110418,validation_seed=2026110419,training_repeats=300,validation_repeats=1500,training_lambdas=GRID,unseen_lambdas=[.15,.45,.7],calibration_users=COSTS,full_horizon=[0,319],scenes=SCENES,elapsed_seconds=time.monotonic()-started,note='Expected-risk integration. Reused kernel draws retain dependence in MC SE; not full trajectories. Calibration/publication noise integrated exactly.')
    (OUT/'metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(json.dumps(meta,indent=2),flush=True)

if __name__=='__main__':main()
