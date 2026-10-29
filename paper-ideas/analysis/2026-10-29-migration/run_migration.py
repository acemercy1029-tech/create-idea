"""Predeclared migration audit with ordinary two-scale/two-candidate baselines."""
from pathlib import Path
from functools import lru_cache
import csv
import itertools
import json
import time
import sys
import numpy as np
from scipy.stats import binom

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT.parent / '2026-10-29-local-history'))
from run import fwht, abs_pmf, winner_prob, argmax_random

N, M, W, EPS, T, R, SEED = 20480, 256, 20, 1., 320, 500, 2026102916
A, B, START, SWITCH, WARM = 37, 121, 96, 160, 64
KAPPA = np.tanh(EPS/2)
WINDOWS = [8, 16, 32, 64, 128]


def save(name, rows):
    with (OUT/name).open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


@lru_cache(None)
def special_abs(b, active, amplitude=.6):
    pmf = np.convolve(binom.pmf(np.arange(active+1), active, (1+amplitude)/2),
                      binom.pmf(np.arange(b-active+1), b-active, .5))
    return abs_pmf(pmf, b)


@lru_cache(None)
def tie_weights(b, dims):
    null = abs_pmf(binom.pmf(np.arange(b+1), b, .5), b)
    cdf = np.minimum(1., np.cumsum(null))
    n = dims-2
    k = np.arange(n+1)
    one, two = np.zeros(b+1), np.zeros(b+1)
    for a in range(b+1):
        if cdf[a] > 0:
            q = min(1., null[a]/cdf[a])
            mass = cdf[a]**n * binom.pmf(k, n, q)
            one[a], two[a] = mass @ (1/(k+1)), mass @ (1/(k+2))
    return one, two


@lru_cache(None)
def two_winners(b, ka, kb, dims=M):
    pa, pb = special_abs(b, ka), special_abs(b, kb)
    one, two = tie_weights(b, dims)
    lower_a = np.maximum(0., np.cumsum(pa)-pa)
    lower_b = np.maximum(0., np.cumsum(pb)-pb)
    qa = float(pa @ (lower_b*one + pb*two))
    qb = float(pb @ (lower_a*one + pa*two))
    assert qa >= -1e-12 and qb >= -1e-12 and qa+qb <= 1+1e-12
    return qa, qb


@lru_cache(None)
def rr_risk(signal, k, threshold):
    n = N//2//W//k
    x = np.arange(n+1)
    z = (2*x/n-1)/KAPPA
    if threshold:
        tau = np.sqrt(2*np.log(2*k/.01)/(n*KAPPA**2))
        z = np.where(np.abs(z)>tau, z, 0.)
    z = np.clip(z, -1, 1)
    p = binom.pmf(x, n, (1+KAPPA*signal)/2)
    return float(p @ ((z-signal)**2))


def validate():
    rows = []
    for b, ka, kb in [(2,1,1), (3,2,1), (4,3,1)]:
        pmfs = []
        for active in [ka,kb,0]:
            pmfs.append(np.convolve(binom.pmf(np.arange(active+1),active,.8),
                                    binom.pmf(np.arange(b-active+1),b-active,.5)))
        expected = np.zeros(2)
        for counts in itertools.product(range(b+1), repeat=3):
            scores = np.abs(2*np.array(counts)-b)
            mass = np.prod([pmfs[j][counts[j]] for j in range(3)])
            for j in [0,1]:
                if scores[j] == max(scores):
                    expected[j] += mass/sum(scores==scores[j])
        exact = two_winners(b,ka,kb,3)
        assert np.max(np.abs(expected-exact)) < 1e-12
        rows.append(dict(check='full_small_enumeration',b=b,ka=ka,kb=kb,
                         exact_a=exact[0],exact_b=exact[1],observed_a=expected[0],observed_b=expected[1]))
    rng = np.random.default_rng(SEED+100)
    for b,ka,kb in [(16,8,8),(64,56,8),(64,32,32),(64,0,64)]:
        observed = np.zeros(2)
        for _ in range(50):
            counts = rng.binomial(b,.5,size=(1000,M))
            for idx,active in [(A,ka),(B,kb)]:
                counts[:,idx] = rng.binomial(active,.8,1000)+rng.binomial(b-active,.5,1000)
            pick = argmax_random(2*counts-b,rng,True)
            observed += [np.sum(pick==A),np.sum(pick==B)]
        observed /= 50000
        exact = np.array(two_winners(b,ka,kb))
        assert np.all(np.abs(observed-exact) < 6*np.sqrt(exact*(1-exact)/50000)+1e-10)
        if ka==0:
            assert abs(exact[1]-winner_prob(b,kb,.6))<1e-10
        rows.append(dict(check='direct_count_simulation_50000',b=b,ka=ka,kb=kb,
                         exact_a=exact[0],exact_b=exact[1],observed_a=observed[0],observed_b=observed[1]))
    save('kernel_checks.csv',rows)


def top_two(v,rng,absolute=False):
    z = np.abs(v) if absolute else v
    z = z+rng.uniform(-1e-12,1e-12,z.shape)
    # sorted descending within the two selected indices
    idx = np.argpartition(z,-2,axis=1)[:,-2:]
    order = np.argsort(np.take_along_axis(z,idx,axis=1),axis=1)[:,::-1]
    return np.take_along_axis(idx,order,axis=1)


def simulate(duration, scene_id):
    rng = np.random.default_rng(np.random.SeedSequence([SEED,scene_id]))
    truth = np.full(T,-1,dtype=int)
    truth[START:SWITCH]=A; truth[SWITCH:SWITCH+duration]=B
    local_states = {}
    for label,b,users in [('full16',16,N//2),('full64',64,N//2),('half16',16,N//4),('half64',64,N//4)]:
        local_states[label] = dict(b=b,users=users,ring=np.zeros((R,4,M)),numbers=np.zeros(4),
                                   sums=np.zeros((R,M)),count=0.)
    server_ring = np.zeros((R,128,M))
    server_sums = {h:np.zeros((R,M)) for h in WINDOWS}
    base_methods = ['server_h%d'%h for h in WINDOWS]+['local_b16_top1','local_b16_top2',
        'local_b64_top1','local_b64_top2','mix_average','mix_union']
    methods = base_methods+[x+'_threshold' for x in base_methods]+['all_rotation_hmm','zero']
    risks = {m:np.zeros((R,T)) for m in methods}
    coverage = {m:np.zeros((R,T)) for m in base_methods}
    cardinality = {m:np.ones((R,T)) for m in base_methods}
    amplitudes=np.array([-.6,-.4,-.2,.2,.4,.6])
    lp,lm=np.log1p(KAPPA*amplitudes),np.log1p(-KAPPA*amplitudes)
    weights=np.full((R,M,6),.5/(M*6)); none=np.full(R,.5)
    law_rows=[]
    for t in range(T):
        mean=np.zeros(M)
        if truth[t]>=0:mean[truth[t]]=.6
        signed=2*rng.binomial(2,(1+KAPPA*mean)/2,size=(R,M))-2
        selected={}
        for h in WINDOWS:
            if t>=h:server_sums[h]-=server_ring[:,(t-h)%128]
            server_sums[h]+=signed
            selected['server_h%d'%h]=top_two(server_sums[h],rng,True)[:,:1]
        server_ring[:,t%128]=signed
        frequencies={}
        for label,state in local_states.items():
            b,users=state['b'],state['users']; stride=W+b-1
            pos=t%4
            state['sums']-=state['ring'][:,pos]
            state['count']-=state['numbers'][pos]
            delta=np.zeros((R,M)); n=0
            if t>=b-1:
                n=users//stride+int(t%stride<users%stride)
                recent=truth[t-b+1:t+1]
                ka,kb=int(np.sum(recent==A)),int(np.sum(recent==B))
                qa,qb=two_winners(b,ka,kb)
                vote=np.full(M,(1-qa-qb)/(M-2)); vote[A]=qa; vote[B]=qb
                hm=fwht(vote)
                probabilities=np.concatenate([(1+KAPPA*hm)/(2*M),(1-KAPPA*hm)/(2*M)])
                probabilities/=probabilities.sum()
                counts=rng.multinomial(n,probabilities,size=R)
                delta=fwht(counts[:,:M]-counts[:,M:])
                if label.startswith('full'):
                    law_rows.append(dict(duration=duration,time=t,batch=b,old_rows=ka,new_rows=kb,
                                         old_vote_probability=qa,new_vote_probability=qb,reporters=n))
            state['ring'][:,pos]=delta; state['numbers'][pos]=n
            state['sums']+=delta; state['count']+=n
            frequencies[label]=state['sums']/(max(1,state['count'])*KAPPA)
        for b in [16,64]:
            picks=top_two(frequencies['full%d'%b],rng)
            selected['local_b%d_top1'%b]=picks[:,:1]
            selected['local_b%d_top2'%b]=picks
        short=top_two(frequencies['half16'],rng)[:,0]
        long=top_two(frequencies['half64'],rng)[:,0]
        union=np.stack([short,np.where(short==long,-1,long)],axis=1)
        selected['mix_union']=union
        selected['mix_average']=top_two((frequencies['half16']+frequencies['half64'])/2,rng)[:,:1]
        for method,picks in selected.items():
            k=np.sum(picks>=0,axis=1)
            contains=np.any(picks==truth[t],axis=1) if truth[t]>=0 else np.zeros(R,dtype=bool)
            coverage[method][:,t]=contains
            cardinality[method][:,t]=k
            for threshold in [False,True]:
                target=method+('_threshold' if threshold else '')
                v0=np.array([rr_risk(0.,int(x),threshold) for x in k])
                vs=np.array([rr_risk(.6,int(x),threshold) for x in k])
                risks[target][:,t]=np.where(contains,vs+(k-1)*v0,k*v0+(.36 if truth[t]>=0 else 0.))
        raw_all=rng.binomial(4,(1+KAPPA*mean)/2,size=(R,M))
        if t:
            active=weights.sum(axis=(1,2)); prev_none=none.copy()
            none=prev_none*(1-1/64)+active/64
            weights=weights*(1-2/64)+(prev_none+active)[:,None,None]/(64*M*6)
        x=raw_all[:,:,None]
        weights*=np.exp(x*lp+(4-x)*lm)
        normalizer=none+weights.sum(axis=(1,2))
        weights/=normalizer[:,None,None]; none/=normalizer
        assert np.max(np.abs(none+weights.sum(axis=(1,2))-1))<1e-12
        estimate=(weights*amplitudes).sum(axis=2)
        risks['all_rotation_hmm'][:,t]=((estimate-mean)**2).sum(axis=1)
        risks['zero'][:,t]=np.sum(mean**2)
    periods=dict(evaluation=np.arange(WARM,T),old_active=np.arange(START,SWITCH),
        new_active=np.arange(SWITCH,SWITCH+duration),new_first8=np.arange(SWITCH,SWITCH+min(8,duration)),
        new_first32=np.arange(SWITCH,SWITCH+min(32,duration)),new_last8=np.arange(SWITCH+duration-min(8,duration),SWITCH+duration),
        after32=np.arange(SWITCH+duration,min(T,SWITCH+duration+32)))
    rows,diffs,traces=[],[],[]
    for method in methods:
        for period,ix in periods.items():
            values=risks[method][:,ix].mean(axis=1)
            se=float(values.std(ddof=1)/np.sqrt(R)); mu=float(values.mean())
            rows.append(dict(duration=duration,method=method,period=period,metric='l2_mse',mean=mu,mc_se=se,low=mu-1.96*se,high=mu+1.96*se))
        for t in range(T):
            traces.append(dict(duration=duration,method=method,time=t,true_pair=int(truth[t]),l2_mse=float(risks[method][:,t].mean())))
    for method in base_methods:
        for period,ix in periods.items():
            for metric,arr in [('coverage',coverage[method]),('candidate_count',cardinality[method])]:
                vals=arr[:,ix].mean(axis=1); mu=float(vals.mean()); se=float(vals.std(ddof=1)/np.sqrt(R))
                rows.append(dict(duration=duration,method=method,period=period,metric=metric,mean=mu,mc_se=se,low=mu-1.96*se,high=mu+1.96*se))
    for candidate in ['mix_union_threshold','local_b16_top1_threshold','local_b64_top2_threshold']:
        for baseline in ['local_b16_top1_threshold','local_b64_top1_threshold','all_rotation_hmm']:
            if candidate==baseline:continue
            for period,ix in periods.items():
                vals=(risks[candidate][:,ix]-risks[baseline][:,ix]).mean(axis=1)
                mu=float(vals.mean()); se=float(vals.std(ddof=1)/np.sqrt(R))
                diffs.append(dict(duration=duration,candidate=candidate,baseline=baseline,period=period,
                                  difference=mu,mc_se=se,low=mu-1.96*se,high=mu+1.96*se))
    return rows,diffs,traces,law_rows


def main():
    before=time.time(); validate()
    summary,differences,traces,laws=[],[],[],[]
    for i,duration in enumerate([8,32,96]):
        a,b,c,d=simulate(duration,i)
        summary+=a; differences+=b; traces+=c; laws+=d
        print('completed migration new duration',duration,flush=True)
    save('summary.csv',summary); save('differences.csv',differences)
    save('time_curves.csv',traces); save('local_vote_laws.csv',laws)
    (OUT/'metadata.json').write_text(json.dumps(dict(seed=SEED,repetitions=R,users=N,pairs=M,
        timestamps=T,epsilon=EPS,w=W,old_duration=64,new_durations=[8,32,96],
        risk='selector Monte Carlo; current independent RR risk exactly integrated',seconds=time.time()-before,
        novelty_claim=False),indent=2),encoding='utf-8')


if __name__=='__main__':main()
