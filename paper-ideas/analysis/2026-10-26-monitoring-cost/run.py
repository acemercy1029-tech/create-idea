"""Final-release MSE study using an LPD-inspired scalar adaptation.

Read PROTOCOL.md first. This is neither full MTSP-LDP nor full LDP-IDS reproduction.
Oracle diagnostics are explicitly non-private and excluded from private ranking.
"""
from pathlib import Path
import csv
import importlib.util
import itertools
import json
import sys
import numpy as np
from scipy.stats import binom

OUT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('round8', OUT.parent/'2026-10-16-robustness'/'run.py')
r8 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r8)
base = r8.base
N, M, W, T, REPS, WARM = base.N, base.M, base.W, base.T, base.REPS, base.WARM
EPS, SEED, TARGET = 1., 2026102610, .01
PAIR_USERS = N//M
K = np.tanh(EPS/2)


def save(name, rows):
    with (OUT/name).open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def mean_ci(x):
    avg = float(x.mean())
    se = float(x.std(ddof=1)/np.sqrt(len(x)))
    return avg, avg-1.96*se, avg+1.96*se, se


def moment_checks():
    rows = []
    for r, rho, history in itertools.product([35,70,140], [0.,.6], [-.6,0.,.7]):
        count = np.arange(r+1)
        p = binom.pmf(count, r, (1+K*rho)/2)
        y = (2*count/r-1)/K
        variance = (K**-2-rho**2)/r
        estimated_variance = (K**-2-y*y)/(r-1)
        score = (y-history)**2-estimated_variance
        assert np.isclose(p.sum(), 1)
        assert np.isclose(p @ y, rho)
        assert np.isclose(p @ (y-rho)**2, variance)
        assert np.isclose(p @ estimated_variance, variance)
        assert np.isclose(p @ score, (rho-history)**2)
        rows.append(dict(reports=r,rho=rho,history=history,
                         expected_score=float(p @ score),true_reuse_loss=(rho-history)**2))
    return rows


def check_id_schedule(counts, capacity):
    # Concrete realization of one simulated count path; no value-based user selection.
    last = np.full(capacity, -W, dtype=int)
    for t, count in enumerate(counts):
        free = np.flatnonzero(t-last >= W)
        assert len(free) >= count
        selected = free[:count]
        assert np.all(t-last[selected] >= W)
        last[selected] = t


def draw_rr(r, rho, rng, shape):
    plus = rng.binomial(r, (1+K*rho)/2, size=shape)
    return (2*plus/r-1)/K


def run_lpd(rho, alpha, mode, rng):
    r_mon = int(PAIR_USERS*alpha)//W
    capacity = PAIR_USERS-r_mon*W
    shape = (REPS,M)
    ring = np.zeros(shape+(W,), dtype=int)
    history = np.zeros(shape)
    released = np.zeros((REPS,T,M))
    changed = np.zeros((T,M))
    report_means = np.zeros((T,M))
    one_path = np.zeros(T, dtype=int)
    max_active = 0
    for t in range(T):
        slot = t % W
        active_past = ring.sum(axis=-1)-ring[...,slot]
        candidate = (capacity-active_past)//2
        monitor = draw_rr(r_mon, rho[t], rng, shape)
        vhat = (K**-2-monitor**2)/(r_mon-1)
        if mode == 'corrected':
            score = (monitor-history)**2-vhat
        elif mode == 'uncorrected':
            score = (monitor-history)**2
        elif mode == 'oracle':
            score = (rho[t]-history)**2
        else:
            raise ValueError(mode)
        risk_bound = K**-2/np.maximum(candidate,1)
        update = (candidate >= 1) & (score > risk_bound)
        used = np.where(update,candidate,0)
        # The publication cohort has disjoint users; records are iid across time.
        # Draw for every candidate, but only selected reports enter the transcript.
        publication = draw_rr(np.maximum(candidate,1),rho[t],rng,shape)
        history = np.where(update,np.clip(publication,-1,1),history)
        released[:,t] = history
        ring[...,slot] = used
        window_used = ring.sum(axis=-1)
        assert np.all(window_used <= capacity)
        max_active = max(max_active,int(window_used.max()))
        changed[t] = update.mean(axis=0)
        report_means[t] = used.mean(axis=0)
        one_path[t] = used[0,0]
    check_id_schedule(one_path,capacity)
    diag = dict(alpha=alpha,mode=mode,is_private=mode!='oracle',
                monitor_reports_per_pair_round=r_mon,publication_pool_capacity=capacity,
                mean_publication_reports=float(report_means[WARM:].mean()),
                publication_fraction=float(changed[WARM:].mean()),
                max_window_publication_users=max_active,
                max_window_total_users=max_active+r_mon*W,
                mean_total_reports=float(report_means[WARM:].mean()+r_mon))
    return released,diag,changed,report_means


def main():
    checks = moment_checks()
    base.validate_model_and_privacy()
    cases = [(None,None)]+list(itertools.product([.2,.6],[2,8,'persistent']))
    rows, diagnostics, traces, differences, stationary = [],[],[],[],[]
    for si,(delta,duration) in enumerate(cases):
        rho = r8.truth(delta,duration)
        r8.validate_truth(rho)
        metrics = {}

        def record(name,est,is_private=True):
            est = np.clip(est,-1,1)
            sq = (est-rho[None])**2
            losses = r8.reduce_loss(sq,duration)
            item = dict(scenario='stationary' if delta is None else f'd{delta}_L{duration}',
                        delta='' if delta is None else delta,duration='' if duration is None else duration,
                        method=name,is_private=is_private,
                        max_pair_timeavg_mse=float(sq[:,WARM:].mean(axis=(0,1)).max()))
            for metric in ['overall','active','post10','return10']:
                stats = mean_ci(losses[metric]) if metric in losses else ('','','','')
                item.update(zip([metric+'_mse',metric+'_low',metric+'_high',metric+'_se'],stats))
            rows.append(item)
            metrics[name] = {key:mean_ci(val) for key,val in losses.items()}
            if delta is None:
                stationary.append(dict(method=name,is_private=is_private,
                    max_pair_timeavg_mse=item['max_pair_timeavg_mse'],
                    meets_point_target=item['max_pair_timeavg_mse']<=TARGET))
            elif delta == .6 and duration == 8 and name in ['rotating_h4','lpd_a50_corrected','lpd_a50_oracle','lpd_a50_uncorrected']:
                j=base.EVENTS[0][1]
                for t in range(60,120):
                    traces.append(dict(method=name,t=t,truth=float(rho[t,j]),
                        mean_estimate=float(est[:,t,j].mean()),mse=float(sq[:,t,j].mean())))

        rng=np.random.default_rng(np.random.SeedSequence([SEED,si,0]))
        y=draw_rr(PAIR_USERS//W,rho,rng,(REPS,T,M))
        for h in [1,4,5,20]:
            record(f'rotating_h{h}',base.rolling(y,h))
        recent,long=base.rolling(y,5),base.rolling(y,20)
        older=(20*long-5*recent)/15
        se=np.sqrt(K**-2/(PAIR_USERS//W)*(1/5+1/15))
        for z in [2,3]:
            short=abs(recent-older)>z*se
            short[:,:19]=False
            record(f'rotating_z{z}',np.where(short,recent,long))
        del y,recent,long,older

        for ai,alpha in enumerate([.25,.5]):
            rng=np.random.default_rng(np.random.SeedSequence([SEED,si,10+ai]))
            publication_reports=int(PAIR_USERS*(1-alpha))//W
            y=draw_rr(publication_reports,rho,rng,(REPS,T,M))
            record(f'reserved_a{int(100*alpha)}_h4',base.rolling(y,4))
            del y

        for mi,(alpha,mode) in enumerate([(.25,'corrected'),(.5,'corrected'),(.5,'uncorrected'),(.5,'oracle')]):
            rng=np.random.default_rng(np.random.SeedSequence([SEED,si,20+mi]))
            raw,diag,changed,reports=run_lpd(rho,alpha,mode,rng)
            name=f'lpd_a{int(100*alpha)}_{mode}'
            diagnostics.append(dict(scenario='stationary' if delta is None else f'd{delta}_L{duration}',**diag))
            record(name,raw,mode!='oracle')
            record(name+'_h4',base.rolling(raw,4),mode!='oracle')
            # Secondary fairness check added after first results; see protocol amendment.
            if mode != 'oracle':
                record(name+'_h20',base.rolling(raw,20),True)
            del raw

        # These sampling strategies use independent RNG streams from the full rotation baseline.
        # No comparison between the dependent raw/h4 pair is included here.
        for name in metrics:
            if name.startswith('rotating_'):
                continue
            for metric in metrics[name]:
                a,_,_,sa=metrics[name][metric]
                b,_,_,sb=metrics['rotating_h4'][metric]
                diff=a-b
                half=1.96*np.sqrt(sa*sa+sb*sb)
                differences.append(dict(scenario='stationary' if delta is None else f'd{delta}_L{duration}',
                    comparison=name+' minus rotating_h4',metric=metric,
                    mean_difference=diff,ci_low=diff-half,ci_high=diff+half))
        print('Finished',si,delta,duration,flush=True)

    save('summary.csv',rows)
    save('stationary_constraint.csv',stationary)
    save('resource_diagnostics.csv',diagnostics)
    save('independent_differences.csv',differences)
    save('moment_checks.csv',checks)
    save('pulse_trace.csv',traces)
    (OUT/'metadata.json').write_text(json.dumps(dict(seed=SEED,repeats=REPS,n=N,m=M,w=W,T=T,
        epsilon=EPS,stationary_target=TARGET,common_bits_per_user_round=1,
        per_user_participations_per_window=1,alphas=[.25,.5],amplitudes=[.2,.6],
        durations=[2,8,'persistent'],main_outputs_clipped=True,python=sys.version,
        numpy=np.__version__,full_paper_reproduction=False,
        secondary_after_first_results='fixed h20 on private LPD outputs; no changes to collection or main metrics',
        oracle_is_private=False,score_target='current population rho versus realized past release'),indent=2),encoding='utf-8')
    print('Stationary screening:')
    for row in stationary:
        print(row)
    print('Strong pulse:')
    for row in rows:
        if row['delta']==.6 and row['duration']==8:
            print(row['method'],row['overall_mse'],row['active_mse'],row['return10_mse'])


if __name__=='__main__':
    main()
