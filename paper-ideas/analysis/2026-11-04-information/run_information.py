"""Information loss audit. Oracle diagnostic != deployable streaming mechanism."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
from pathlib import Path
import sys, csv, json, time
import numpy as np
from scipy.signal import fftconvolve
from scipy.special import xlogy
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT.parent / '2026-11-03-allocation-grid'))
import run_grid as g
d = g.d
M, N, W, A, B = d.M, d.N, d.W, d.A, d.B
K, REPS, SEED = M - 1, 10000, 2026110419
KEEP = np.array([j for j in range(M) if j != A])
H = d.H
CHECKS = []

def save(name, rows):
    with (OUT / name).open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def check(name, error, tolerance, setting=''):
    assert error <= tolerance, (name, error, tolerance, setting)
    CHECKS.append(dict(test=name, error=error, tolerance=tolerance, setting=setting))

def nphase(b, t):
    return (N//2)//(W+b-1) + int(t % (W+b-1) < (N//2) % (W+b-1))

def laws(b, lam, t):
    times = np.arange(t-b+1, t+1)
    pa = tuple(np.where((times >= 96) & (times < 160), .8, .5))
    pb = tuple(np.where((times >= 160) & (times < 168), .8, .5))
    ca, cb = d.raw_counts(pa, lam), d.raw_counts(pb, lam)
    qa, qb = d.winners(ca, cb, lam)
    q0 = (1-qa-qb)/(M-2)
    return qa, qb, q0, cb

def conv(p, q):
    z = fftconvolve(p, q)
    assert z.min() > -1e-11
    return np.maximum(z, 0) / np.maximum(z, 0).sum()

def power(p, n):
    ans = np.ones(1)
    while n:
        if n & 1: ans = conv(ans, p)
        n //= 2
        if n: p = conv(p, p)
    return ans

def max_success(signal, null, candidates=K):
    assert len(signal) == len(null)
    cdf = np.minimum(1., np.cumsum(null)); cdf[-1] = 1.
    ans = np.zeros(len(null)); nz = null > 0
    q = np.minimum(1., null[nz]/cdf[nz])
    numerator = np.ones_like(q); inside = q < 1
    numerator[inside] = -np.expm1(candidates*np.log1p(-q[inside]))
    ans[nz] = cdf[nz]**(candidates-1)*numerator/(candidates*q)
    ans[~nz] = cdf[~nz]**(candidates-1)
    return float(signal @ ans)

def channel(qa, qb, q0, kappa):
    # Rows: hypothetical new coordinates; columns: (public row, received bit).
    means = q0*H.sum(axis=1)[None, :] + (qa-q0)*H[:, A][None, :] + (qb-q0)*H[:, KEEP].T
    pp = np.concatenate([1+kappa*means, 1-kappa*means], axis=1)/(2*M)
    # Row zero's impossible bit can be -2e-19 from roundoff; no model change.
    assert pp.min() > -1e-14
    pp = np.maximum(pp, 0)
    return pp/pp.sum(axis=1)[:,None]

def mutual(p):
    marginal = p.mean(axis=0)
    ratio = np.divide(p, marginal, out=np.ones_like(p), where=marginal > 0)
    return max(0., float(np.sum(xlogy(p, ratio))/len(p)))

def ml_rows(counts, qa, qb, q0, kappa):
    base = (qa-q0)*H[:, A]; delta = qb-q0
    # Public row 0 contributes only a coordinate-independent constant.
    plus = .5*np.log((1+kappa*(base+delta))/(1+kappa*(base-delta)))
    minus = .5*np.log((1-kappa*(base+delta))/(1-kappa*(base-delta)))
    return counts[:, :M]*plus + counts[:, M:]*minus

def pick(scores, rng, exclude_old=False):
    if exclude_old:
        return KEEP[d.iid.argmax_random(scores[:, KEEP], rng)]
    return d.iid.argmax_random(scores, rng)

def success_bound(info):
    return min(1., (info+np.log(2))/np.log(K), 1/K+np.sqrt(info/2))

def summary(rows, setting, method, values, metric):
    for period, vals in [('event8', values.mean(axis=1)), ('last', values[:, -1])]:
        avg, se = float(vals.mean()), float(vals.std(ddof=1)/np.sqrt(len(vals)))
        low,high,ci=avg-1.96*se,avg+1.96*se,'normal_MC'
        if metric=='support_success':
            if period=='last':
                z=1.96;den=1+z*z/len(vals)
                mid=(avg+z*z/(2*len(vals)))/den
                half=z*np.sqrt(avg*(1-avg)/len(vals)+z*z/(4*len(vals)**2))/den
                low,high,ci=max(0.,mid-half),min(1.,mid+half),'Wilson'
            elif avg==0 or avg==1:
                tail=1-.05**(1/len(vals))
                low,high,ci=max(0.,avg-tail),min(1.,avg+tail),'one_sided_extreme_bounded'
            else: low,high=max(0.,low),min(1.,high)
        rows.append(dict(**setting, method=method, metric=metric, period=period,
                         mean=avg, mc_se=se, low=low, high=high, interval=ci))

def validate():
    rng = np.random.default_rng(SEED+999)
    p = np.array([.2, .5, .3]); direct = np.ones(1)
    for i in range(1, 7):
        direct = np.convolve(direct, p)
        check('fft_power_direct', float(max(abs(direct-power(p, i)))), 1e-12, i)
    # Exact equal-distribution classification must be uniform, including ties.
    for k in [2, 3, 255]:
        check('null_exchangeability', abs(max_success(p, p, k)-1/k), 1e-12, k)
    for b in [4, 8, 16, 64]:
        stride = W+b-1
        touched = 0
        for phase in range(stride):
            ends = np.arange(-2*stride+phase, 3*stride+phase, stride)
            for start in range(stride):
                touched = max(touched, int(np.sum((ends >= start) & (ends-b+1 <= start+W-1))))
        check('changed_window_reports', touched, 1, b)
        check('phase_population', abs(sum(nphase(b, t) for t in range(stride))-N//2), 0, b)
        check('per_round_bits', max(nphase(b, t)+512 for t in range(stride)), 1024, b)
        for lam in [0., .6]:
            qa, qb, q0, _ = laws(b, lam, 167)
            for kap in [1., d.KAPPA]:
                pp = channel(qa, qb, q0, kap)
                check('channel_normalization', float(max(abs(pp.sum(axis=1)-1))), 1e-12, (b,lam,kap))
                counts = rng.multinomial(300, pp[list(KEEP).index(B)], size=3)
                fast = d.iid.fwht(ml_rows(counts, qa, qb, q0, kap))[:, KEEP]
                direct = counts @ np.log(np.maximum(pp, 1e-300)).T
                check('fast_ml_vs_direct', float(np.max(abs((fast-fast[:, :1])-(direct-direct[:, :1])))), 1e-9, (b,lam,kap))
    check('rr_epsilon', abs(np.log((1+d.KAPPA)/(1-d.KAPPA))-1), 1e-12)

def run():
    begin = time.monotonic(); validate()
    rows, curves, info_rows, raw_rows, vote_rows = [], [], [], [], []
    for li, lam in enumerate([0., .6]):
        for b in [4, 8, 16, 64]:
            rng = np.random.default_rng(np.random.SeedSequence([SEED,li,b]))
            setting = dict(lam=lam, b=b)
            oracle = {name: np.zeros((REPS,8), bool) for name in ['index_ml','onebit_ml','rr_ml']}
            actual = {h: np.zeros((REPS,8), bool) for h in [1,4,8]}
            raw_index_score = np.zeros((REPS,M))
            row_scores = {name: np.zeros((REPS,M)) for name in ['onebit_ml','rr_ml']}
            received = []
            cumul = dict(index=0., onebit=0., rr=0., any_index_ldp=0., noninteractive_index_ldp=0.)
            raw_null, raw_sig = np.ones(1), np.ones(1)
            total_reports = 0
            moments_null, moments_sig = np.zeros(2), np.zeros(2)
            for t in range(153,168):
                qa,qb,q0,pb = laws(b,lam,t); n = nphase(b,t)
                vote = np.full(M,q0); vote[A],vote[B] = qa,qb
                if t >= 160:
                    r=t-160
                    index_count = rng.multinomial(n,vote/vote.sum(),size=REPS)
                    raw_index_score += index_count*np.log(qb/q0)
                    oracle['index_ml'][:,r] = pick(raw_index_score,rng,True)==B
                    meanq=(1-qa)/K
                    i0=qb*np.log(qb/meanq)+(M-2)*q0*np.log(q0/meanq)
                    mi = {'index':max(0.,i0)}
                for name,kap in [('onebit_ml',1.),('rr_ml',d.KAPPA)]:
                    if t<160 and name=='onebit_ml': continue
                    pp = channel(qa,qb,q0,kap)
                    counts = rng.multinomial(n,pp[list(KEEP).index(B)]/pp[list(KEEP).index(B)].sum(),size=REPS)
                    if name=='rr_ml':
                        received.append(counts[:,:M]-counts[:,M:])
                    if t>=160:
                        row_scores[name] += ml_rows(counts,qa,qb,q0,kap)
                        oracle[name][:,r] = pick(d.iid.fwht(row_scores[name]),rng,True)==B
                        mi['onebit' if name=='onebit_ml' else 'rr'] = mutual(pp)
                if t<160: continue
                for h in actual:
                    aggregate=sum(received[-h:])
                    actual[h][:,r] = pick(d.iid.fwht(aggregate),rng)==B
                mi['any_index_ldp']=(1-1/K)*(np.exp(1.)-1)**2*(qb-q0)**2
                mi['noninteractive_index_ldp']=(qb-q0)**2*(np.exp(1.)-1)**2/(4*(1+(1-qa)*(np.exp(1.)-1)))
                check('mi_index_to_onebit', max(0.,mi['onebit']-mi['index']), 1e-12, (b,lam,t))
                check('mi_rr_processing', max(0.,mi['rr']-mi['onebit']), 1e-12, (b,lam,t))
                check('mi_rr_general_bound', max(0.,mi['rr']-mi['any_index_ldp']), 1e-12, (b,lam,t))
                check('mi_rr_noninteractive_bound', max(0.,mi['rr']-mi['noninteractive_index_ldp']), 1e-12, (b,lam,t))
                for key in cumul: cumul[key] += n*mi[key]
                for key in cumul:
                    inf=cumul[key] if key not in ['any_index_ldp','noninteractive_index_ldp'] else min(cumul[key],cumul['index'])
                    upper=success_bound(inf)
                    info_rows.append(dict(**setting,t=t,reports=total_reports+n,channel=key,
                        per_report_mi_or_kl_bound=mi[key],information_upper=inf,
                        success_upper=upper,bayes_mse_lower=.36*(1-upper)))
                # Raw count-vector diagnostic: convolution over exactly the same users.
                p0=d.raw_counts((.5,)*b,lam)
                raw_null=conv(raw_null,power(p0,n));raw_sig=conv(raw_sig,power(pb,n))
                total_reports += n
                k=np.arange(b+1)
                for pmf,mm in [(p0,moments_null),(pb,moments_sig)]:
                    mean=k@pmf;mm += n*np.array([mean,(k-mean)**2@pmf])
                for pmf,mm in [(raw_null,moments_null),(raw_sig,moments_sig)]:
                    k=np.arange(len(pmf));error=max(abs(k@pmf-mm[0]),abs((k-mm[0])**2@pmf-mm[1]))
                    check('raw_convolution_moments',float(error),2e-4,(b,lam,t))
                raw_rows.append(dict(**setting,t=t,reports=total_reports,
                                    success=max_success(raw_sig,raw_null)))
                vote_rows.append(dict(**setting,t=t,n=n,old_vote=qa,new_vote=qb,null_vote=q0,delta=qb-q0))
            for name,values in oracle.items():
                summary(rows,setting,name,values,'support_success')
                key={'index_ml':'index','onebit_ml':'onebit','rr_ml':'rr'}[name]
                bounds=[z['success_upper'] for z in info_rows if z['lam']==lam and z['b']==b and z['channel']==key]
                for r in range(8):
                    p=float(values[:,r].mean());se=np.sqrt(p*(1-p)/REPS)
                    check('mc_vs_information_upper',max(0.,p-bounds[r]),6*se+6/REPS,(b,lam,r,name))
                    curves.append(dict(**setting,t=160+r,method=name,success=p,mc_se=se))
            for h,values in actual.items():
                name='causal_g'+str(h)
                summary(rows,setting,name,values,'support_success')
                for th in [False,True]:
                    summary(rows,setting,name,g.single_risk(values,512,th),'mse_threshold' if th else 'mse_raw')
                for r in range(8):
                    p=float(values[:,r].mean());curves.append(dict(**setting,t=160+r,method=name,success=p,mc_se=np.sqrt(p*(1-p)/REPS)))
            save('results.csv',rows);save('curves.csv',curves);save('information_bounds.csv',info_rows)
            save('raw_diagnostic.csv',raw_rows);save('local_votes.csv',vote_rows);save('checks.csv',CHECKS)
            print('Done lambda={} b={} in {:.1f}s'.format(lam,b,time.monotonic()-begin),flush=True)
    references=[]
    for th in [False,True]:
        references.append(dict(threshold=th,known_location_mse=float(g.single_risk(True,512,th)),zero_mse=.36))
    save('known_location.csv',references)
    (OUT/'metadata.json').write_text(json.dumps(dict(seed=SEED,repeats=REPS,elapsed_seconds=time.monotonic()-begin,
        checks=len(CHECKS),note='15-round distinct-user report transcript; known-onset ML diagnostic distinct from causal fixed-window methods; publication noise integrated'),indent=2),encoding='utf-8')
    print(json.dumps(references),flush=True)

if __name__=='__main__': run()
