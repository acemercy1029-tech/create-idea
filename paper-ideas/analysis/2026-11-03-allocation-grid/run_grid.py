"""Fixed allocation/mixture audit. Oracle tuning is diagnostic, not online DP."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
from pathlib import Path
import sys
import json
import csv
import time
from functools import lru_cache
import numpy as np
from scipy.stats import binom

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT.parent / '2026-10-30-temporal-dependence'))
import run_dependence as d
SIZES = [8, 16, 32, 64, 128]
LAMBDAS = [0., .3, .6, .8]
ANCHORS = ['local16_a2', 'local64_a2', 'server64_a2', 'server128_a2']
CONFIGS = {}
for kind in ['local', 'server']:
    for size in SIZES:
        for a in [1, 2, 3]:
            CONFIGS[kind + str(size) + '_a' + str(a)] = (kind, size, a)
for b in SIZES:
    for h in SIZES:
        CONFIGS['mix_b' + str(b) + '_h' + str(h)] = ('mix', b, h)


def save(name, rows):
    with (OUT / name).open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


@lru_cache(None)
def pub_moments(rho, n, threshold):
    x = np.arange(n + 1)
    raw = (2 * x / n - 1) / d.KAPPA
    tau = np.sqrt(2 * np.log(200) / (n * d.KAPPA ** 2)) if threshold else 0.
    z = np.where(abs(raw) > tau, np.clip(raw, -1, 1), 0.)
    pmf = binom.pmf(x, n, (1 + d.KAPPA * rho) / 2)
    return float(pmf @ z), float(pmf @ z ** 2)


def single_risk(correct, n, th):
    mean, second = pub_moments(.6, n, th)
    _, second0 = pub_moments(0., n, th)
    return np.where(correct, second - 1.2 * mean + .36, second0 + .36)


def mixture_risk(left, right, th):
    mean, second = pub_moments(.6, 256, th)
    _, second0 = pub_moments(0., 256, th)
    return (.36 - .6 * mean * (left.astype(int) + right.astype(int))
            + .25 * (np.where(left, second, second0) + np.where(right, second, second0))
            + .5 * mean ** 2 * (left & right))


@lru_cache(None)
def server_one(ps, lam):
    """One user per coordinate and phase. Independent RR across repeat visits."""
    step = [np.eye(2)] + [d.trans(ps[i-1], ps[i], lam) for i in range(1, len(ps))]
    total = np.ones(1)
    emission = np.array([(1 - d.KAPPA) / 2, (1 + d.KAPPA) / 2])
    for phase in range(min(d.W, len(ps))):
        times = list(range(phase, len(ps), d.W))
        matrices = []
        for t in times[1:]:
            mat = np.eye(2)
            for j in range(t - d.W + 1, t + 1):
                mat = mat @ step[j]
            matrices.append(mat)
        pmf = d.count_chain(ps[times[0]], matrices, emission)
        total = np.convolve(total, pmf)
    assert len(total) == len(ps) + 1 and abs(total.sum() - 1) < 1e-10
    return total / total.sum()


@lru_cache(None)
def server_counts(ps, lam, k):
    base = server_one(ps, lam)
    total = np.ones(1)
    for _ in range(k):
        total = np.convolve(total, base)
    return total / total.sum()


def validate():
    rows = []
    for b in SIZES:
        for active in [0, b // 2, b]:
            qa, _ = d.winners(d.raw_counts((.5,) * (b-active) + (.8,) * active, 0.), d.raw_counts((.5,) * b, 0.), 0.)
            delta = abs(qa - d.iid.winner_prob(b, active, .6))
            assert delta < 1e-10
            rows.append(dict(test='iid_winner', setting=str((b, active)), error=delta, tolerance=1e-10))
        stride, maximum = d.W + b - 1, 0
        for phase in range(stride):
            ends = np.arange(-3 * stride + phase, 4 * stride + phase, stride)
            for start in range(stride):
                maximum = max(maximum, int(np.sum((ends >= start) & (ends - b + 1 <= start + d.W - 1))))
        assert maximum == 1
        rows.append(dict(test='affected_history_reports', setting=str(b), error=maximum, tolerance=1))
    for lam in LAMBDAS:
        for k in [1, 2, 3]:
            for h in [8, 32, 128]:
                p = .8
                pmf = server_counts((p,) * h, lam, k)
                x = np.arange(len(pmf))
                mean = k * h * (1 + d.KAPPA * .6) / 2
                var = k * h * (1 - (d.KAPPA * .6) ** 2) / 4
                var += 2 * k * sum((h-lag) * d.KAPPA ** 2 * p * (1-p) * lam ** lag for lag in range(d.W, h, d.W))
                error = max(abs(x @ pmf - mean), abs(((x-mean)**2) @ pmf - var))
                assert error < 1e-8
                rows.append(dict(test='server_moments', setting=str((lam,k,h)), error=error, tolerance=1e-8))
        ps = np.r_[np.full(150, .8), np.full(170, .5)]
        law = d.StreamLaw(ps, lam)
        for h in [32, 128]:
            error = float(max(abs(law.reported_counts(180-h,179) - server_counts(tuple(ps[180-h:180]),lam,2))))
            assert error < 1e-10
            rows.append(dict(test='server_previous_nonstationary', setting=str((lam,h)), error=error, tolerance=1e-10))
    rng = np.random.default_rng(2026110399)
    reps = 200000
    for th in [False, True]:
        for i, j in [(0,0),(0,1),(1,0),(1,1),(1,2)]:
            # target coordinate 0; wrong equal/different coordinates both checked.
            estimates = []
            for chosen in [i,j]:
                rho = .6 if chosen == 0 else 0.
                x = rng.binomial(256, (1+d.KAPPA*rho)/2, reps)
                raw = (2*x/256-1)/d.KAPPA
                tau = np.sqrt(2*np.log(200)/(256*d.KAPPA**2)) if th else 0.
                estimates.append(np.where(abs(raw)>tau, np.clip(raw,-1,1),0.))
            left,right=estimates
            vals = .25*(left**2+right**2) + .5*left*right*(i==j) - .6*(left*(i==0)+right*(j==0)) + .36
            exact = float(mixture_risk(np.array([i==0]), np.array([j==0]), th)[0])
            se = vals.std(ddof=1)/np.sqrt(reps)
            assert abs(vals.mean()-exact)<6*se+1e-10
            rows.append(dict(test='mixture_vector_simulation', setting=str((th,i,j)), error=abs(vals.mean()-exact), tolerance=6*se+1e-10))
    for n in [256,512,768]:
        for th in [False,True]:
            assert abs(pub_moments(0.,n,th)[0])<1e-12
    save('kernel_checks.csv', rows)
    print('Checks passed:',len(rows),flush=True)


def run(lam, li, scene, names, repeats, stage):
    rng=np.random.default_rng(np.random.SeedSequence([2026110317+stage,li,int(scene=='migration')]))
    p_a,p_b,p0=np.full(d.T,.5),np.full(d.T,.5),np.full(d.T,.5)
    if scene=='constant':
        p_a[:]=.8
        times,target=[159],d.A
    else:
        p_a[96:160]=.8
        p_b[160:256]=.8
        times,target=list(range(160,256)),d.B
    keys=set()
    for name in names:
        kind,size,a=CONFIGS[name]
        if kind=='mix':
            keys.update([('local',size,1),('server',a,1)])
        else:
            keys.add((kind,size,a))
    keys=sorted(keys)
    correctness={key:np.zeros((repeats,len(times)),dtype=bool) for key in keys}
    for ti,t in enumerate(times):
        for kind,size,a in keys:
            if kind=='local':
                stride=d.W+size-1
                screen=d.N*a//4
                hist=np.zeros((repeats,d.M))
                for end in range(t-3,t+1):
                    n=screen//stride+int(end%stride<screen%stride)
                    qa,qb=d.winners(d.raw_counts(tuple(p_a[end-size+1:end+1]),lam),d.raw_counts(tuple(p_b[end-size+1:end+1]),lam),lam)
                    vote=np.full(d.M,(1-qa-qb)/(d.M-2))
                    vote[d.A],vote[d.B]=qa,qb
                    row_mean=d.H@vote
                    probs=np.r_[1+d.KAPPA*row_mean,1-d.KAPPA*row_mean]/(2*d.M)
                    counts=rng.multinomial(n,probs/probs.sum(),size=repeats)
                    hist+=counts[:,:d.M]-counts[:,d.M:]
                chosen=d.iid.argmax_random(d.iid.fwht(hist),rng)
            else:
                pmfs=[server_counts(tuple(ps[t-size+1:t+1]),lam,a) for ps in [p_a,p_b,p0]]
                counts=rng.choice(len(pmfs[2]),size=(repeats,d.M),p=pmfs[2])
                counts[:,d.A]=rng.choice(len(pmfs[0]),size=repeats,p=pmfs[0])
                counts[:,d.B]=rng.choice(len(pmfs[1]),size=repeats,p=pmfs[1])
                chosen=d.iid.argmax_random(2*counts-a*size,rng,absolute=True)
            correctness[(kind,size,a)][:,ti]=chosen==target
        if scene=='migration' and (ti+1)%32==0:
            print('stage={} lambda={} snapshots {}/96'.format(stage,lam,ti+1),flush=True)
    periods={'stable':np.ones(1,dtype=bool)} if scene=='constant' else {'first8':np.arange(96)<8,'first32':np.arange(96)<32,'first96':np.ones(96,dtype=bool),'last8':np.arange(96)>=88}
    rows, averages=[],{}
    for th in [False,True]:
        for name in names+['zero']:
            if name=='zero':
                vals=np.full((repeats,len(times)),.36)
                family='zero'
            else:
                kind,size,a=CONFIGS[name]
                family=kind
                if kind=='mix':
                    vals=mixture_risk(correctness[('local',size,1)],correctness[('server',a,1)],th)
                else:
                    vals=single_risk(correctness[(kind,size,a)],(4-a)*256,th)
            for period,mask in periods.items():
                avg=vals[:,mask].mean(axis=1)
                mu,se=float(avg.mean()),float(avg.std(ddof=1)/np.sqrt(repeats))
                rows.append(dict(stage=stage,lam=lam,scene=scene,period=period,threshold=th,method=name,family=family,repeats=repeats,mse=mu,mc_se=se,low=mu-1.96*se,high=mu+1.96*se))
                averages[(th,name,period)]=avg
    return rows,averages


def main():
    started=time.monotonic()
    validate()
    pilot=[]
    names=list(CONFIGS)
    for li,lam in enumerate(LAMBDAS):
        for scene in ['constant','migration']:
            rows,_=run(lam,li,scene,names,300,0)
            pilot.extend(rows)
            save('pilot_summary.csv',pilot)
            print('Pilot completed',lam,scene,'elapsed',round(time.monotonic()-started,1),flush=True)
    choices=[]
    for lam in LAMBDAS:
        for scene,period in [('constant','stable'),('migration','first96')]:
            for th in [False,True]:
                for family in ['local','server','mix']:
                    possible=[r for r in pilot if r['lam']==lam and r['scene']==scene and r['period']==period and r['threshold']==th and r['family']==family]
                    best=min(possible,key=lambda r:r['mse'])
                    choices.append(dict(lam=lam,scene=scene,threshold=th,family=family,method=best['method'],pilot_mse=best['mse']))
    save('frozen_choices.csv',choices)
    confirmed,diffs=[],[]
    for li,lam in enumerate(LAMBDAS):
        for scene in ['constant','migration']:
            selected=[r for r in choices if r['lam']==lam and r['scene']==scene]
            names=sorted(set(ANCHORS+[r['method'] for r in selected]))
            repeats=5000 if scene=='constant' else 1500
            rows,avg=run(lam,li,scene,names,repeats,1)
            confirmed.extend(rows)
            save('confirmation_summary.csv',confirmed)
            periods=['stable'] if scene=='constant' else ['first8','first32','first96','last8']
            for th in [False,True]:
                pick={r['family']:r['method'] for r in selected if r['threshold']==th}
                pairs=[(pick['local'],pick['server']),(pick['mix'],pick['server']),(pick['mix'],pick['local'])]
                pairs.extend((pick[family],anchor) for family in ['local','server','mix'] for anchor in ANCHORS)
                for left,right in pairs:
                    for period in periods:
                        x=avg[(th,left,period)]-avg[(th,right,period)]
                        mu,se=float(x.mean()),float(x.std(ddof=1)/np.sqrt(repeats))
                        diffs.append(dict(lam=lam,scene=scene,period=period,threshold=th,method=left,reference=right,difference=mu,mc_se=se,low=mu-1.96*se,high=mu+1.96*se))
            save('confirmation_differences.csv',diffs)
            print('Confirmation completed',lam,scene,'elapsed',round(time.monotonic()-started,1),flush=True)
    resources=[]
    nulls=[]
    for name,(kind,size,a) in CONFIGS.items():
        if kind=='mix':
            bits=5120/(d.W+size-1)+5120/d.W+512
            pubs='256+256'
        else:
            bits=d.N*a/4/(d.W+size-1 if kind=='local' else d.W)+(4-a)*256
            pubs=str((4-a)*256)
        resources.append(dict(method=name,family=kind,publication_reports=pubs,mean_uplink_bits=bits,max_per_user_per_round_bits=1,epsilon=d.EPS))
        for th in [False,True]:
            nullrisk=.5*pub_moments(0.,256,th)[1] if kind=='mix' else pub_moments(0.,(4-a)*256,th)[1]
            nulls.append(dict(method=name,threshold=th,null_total_mse=nullrisk))
    save('resources.csv',resources)
    save('exact_null_risk.csv',nulls)
    save('publication_moments.csv',[dict(rho=rho,n=n,threshold=th,mean=pub_moments(rho,n,th)[0],second=pub_moments(rho,n,th)[1]) for rho in [0.,.6] for n in [256,512,768] for th in [False,True]])
    metadata=dict(client_date='2026-11-03',n=d.N,m=d.M,w=d.W,epsilon=d.EPS,lambdas=LAMBDAS,sizes=SIZES,screening_fractions=[.25,.5,.75],fixed_configs=len(CONFIGS),pilot_repeats=300,confirmation_migration_repeats=1500,confirmation_stable_repeats=5000,seeds=[2026110317,2026110318],elapsed_seconds=time.monotonic()-started,note='Independent per-time snapshots, not path trajectories. Frozen choices know environment; not an online algorithm.')
    (OUT/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(json.dumps(metadata,indent=2),flush=True)


if __name__=='__main__':
    main()
