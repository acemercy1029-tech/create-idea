"""Classical Rao-Blackwell postprocessing, not a proposed new LDP method.

Read PROTOCOL.md. No truth parameter is used by rb_estimate.
"""
from pathlib import Path
import csv
import importlib.util
import itertools
import json
import sys
import time
import numpy as np
from scipy.special import gammaln
from scipy.stats import binom, hypergeom, norm

OUT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('round11', OUT.parent/'2026-10-27-feedback-reuse'/'run.py')
r11 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r11)
r10, r8, base = r11.r10, r11.r8, r11.base
K, M, T, W, WARM, REPS = r11.K, r11.M, r11.T, r11.W, r11.WARM, r11.REPS
SEED = 2026102812


def save(name, rows):
    with (OUT/name).open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def continuation(y, history, m, r):
    score = (y-history)**2-(K**-2-y*y)/(m-1)
    return (r > 0) & (score > K**-2/np.maximum(r, 1))


def rb_estimate(x, z, m, r, history):
    """Evaluate E[Y | N,S,H,r] via normalized log combinatorial weights.

    Inputs x,z are privatized positive counts; z is ignored when not continued.
    r and history must be predictable from the past, not from current x.
    """
    x, z, r, history = np.broadcast_arrays(x, z, r, history)
    shape = x.shape
    x, z, r, history = [v.reshape(-1) for v in (x,z,r,history)]
    y = (2*x/m-1)/K
    keep = continuation(y, history, m, r)
    out = y.copy()
    ii = np.flatnonzero(keep)
    # Chunking bounds temporary memory for full-stream arrays.
    grid = np.arange(m+1)
    first_log = gammaln(m+1)-gammaln(grid+1)-gammaln(m-grid+1)
    first_y = (2*grid/m-1)/K
    for start in range(0, len(ii), 2048):
        index = ii[start:start+2048]
        s = (x[index]+z[index])[:,None]
        rr = r[index,None]
        zz = s-grid
        allowed = continuation(first_y[None],history[index,None],m,rr)
        feasible = (zz >= 0) & (zz <= rr) & allowed
        safe_z = np.clip(zz, 0, rr)
        logw = first_log + gammaln(rr+1)-gammaln(safe_z+1)-gammaln(rr-safe_z+1)
        logw = np.where(feasible,logw,-np.inf)
        shift = logw.max(axis=1,keepdims=True)
        assert np.isfinite(shift).all()
        weights = np.exp(logw-shift)
        conditional_x = (weights*grid).sum(axis=1)/weights.sum(axis=1)
        out[index] = (2*conditional_x/m-1)/K
    return out.reshape(shape)


def exact_case(m, r, rho, history):
    x = np.arange(m+1)[:,None]
    z = np.arange(r+1)[None,:]
    y = np.broadcast_to((2*x/m-1)/K,(m+1,r+1))
    update = continuation(y,history,m,r)
    pooled = (2*(x+update*z)/(m+update*r)-1)/K
    rb = rb_estimate(x,z,m,r,history)
    p = (1+K*rho)/2
    prob = binom.pmf(x,m,p)*binom.pmf(z,r,p)
    assert abs(prob.sum()-1)<1e-12
    row = dict(m=m,r=r,rho=rho,history=history,
               update_probability=float((prob*update).sum()),
               expected_reports=float((prob*(m+update*r)).sum()))
    for name,est in [('monitor',y),('pooled',pooled),('rb',rb)]:
        mean = float((prob*est).sum())
        mse = float((prob*(est-rho)**2).sum())
        row.update({name+'_mean':mean,name+'_bias':mean-rho,name+'_mse':mse,
                    name+'_variance':mse-(mean-rho)**2,
                    name+'_clipped_bias':float((prob*np.clip(est,-1,1)).sum())-rho,
                    name+'_clipped_mse':float((prob*(np.clip(est,-1,1)-rho)**2).sum())})
    assert abs(row['rb_bias'])<1e-10
    assert row['rb_mse'] <= row['monitor_mse']+1e-10
    return row


def validate_rb():
    checks = []
    maximum = 0.
    # Independent hypergeometric PMF implementation, including boundary counts.
    for m,r,history in itertools.product([3,35,70],[1,35,140,700],[-1.,0.,.3,1.]):
        grid = np.arange(m+1)
        allowed = continuation((2*grid/m-1)/K,history,m,r)
        for s in np.unique(np.linspace(0,m+r,31,dtype=int)):
            feasible = allowed & (grid<=s) & (grid>=s-r)
            if not feasible.any():
                continue
            pmf = hypergeom.pmf(grid,m+r,s,m)*allowed
            expected = (2*(pmf@grid)/pmf.sum()/m-1)/K
            chosen = grid[feasible][0]
            actual = float(rb_estimate(chosen,s-chosen,m,r,history))
            maximum = max(maximum,abs(expected-actual))
    assert maximum<1e-10
    checks.append(dict(check='independent_hypergeometric_conditional_mean',passed=True,value=maximum))
    # No continuation reduces exactly to the original monitor.
    x = np.arange(71)
    assert np.array_equal(rb_estimate(x,0,70,0,0.),(2*x/70-1)/K)
    checks.append(dict(check='zero_available_reports_identity',passed=True,value=0.))
    return checks


def simulate(rho, rng, reps=REPS):
    m, capacity = 70, 1400
    shape = (reps,M)
    ring = np.zeros(shape+(W,),dtype=int)
    direct = np.zeros(shape)
    names = ['direct','monitor','pooled','rb']
    output = {name:np.empty((reps,T,M)) for name in names}
    used_total, rounds, maximum = 0,0,0
    one_path = np.zeros(T,dtype=int)
    for t in range(T):
        slot = t % W
        past = ring.sum(axis=-1)-ring[...,slot]
        candidate = (capacity-past)//2
        x = rng.binomial(m,(1+K*rho[t])/2,size=shape)
        y = (2*x/m-1)/K
        update = continuation(y,direct,m,candidate)
        used = np.where(update,candidate,0)
        # Matches prior simulation RNG exactly; unused counterfactual draws
        # enter neither statistics nor resource counts.
        z = rng.binomial(np.maximum(candidate,1),(1+K*rho[t])/2,size=shape)
        publication = (2*z/np.maximum(candidate,1)-1)/K
        corrected = rb_estimate(x,z,m,candidate,direct)
        pooled = (m*y+used*publication)/(m+used)
        direct = np.where(update,np.clip(publication,-1,1),direct)
        for name,value in [('direct',direct),('monitor',y),('pooled',pooled),('rb',corrected)]:
            output[name][:,t] = value
        ring[...,slot] = used
        assert np.all(ring.sum(axis=-1)<=capacity)
        maximum = max(maximum,int(ring.sum(axis=-1).max()))
        one_path[t] = used[0,0]
        if t>=WARM:
            used_total += int(used.sum())
            rounds += int(update.sum())
    r10.check_id_schedule(one_path,capacity)
    denom = reps*(T-WARM)*M
    return output,dict(monitor_reports=m,mean_total_reports=m+used_total/denom,
                       update_fraction=rounds/denom,max_window_total=maximum+m*W)


def main():
    started = time.time()
    checks = validate_rb()
    rows = [exact_case(70,140,rho,h) for rho,h in itertools.product([0.,.2,.6],[0.,.3,.6])]
    save('exact_selection.csv',rows)
    grid = [exact_case(m,r,float(rho),h) for m,r,h,rho in itertools.product(
        [35,70],[35,140,700],[-.6,0.,.6],np.linspace(-.9,.9,19))]
    save('exact_grid.csv',grid)
    # Regression against the previous independently computed ordinary estimator.
    with (OUT.parent/'2026-10-27-feedback-reuse'/'exact_selection.csv').open(encoding='utf-8') as f:
        previous = list(csv.DictReader(f))
    error = max(abs(new[key]-float(old[key])) for new,old in zip(rows,previous)
                for key in ['pooled_mean','pooled_mse'])
    assert error<1e-10
    checks.append(dict(check='round11_exact_enumeration_regression',passed=True,value=error))
    checks.append(dict(check='all_351_exact_biases_and_variance_domination',passed=True,
                       value=max(abs(row['rb_bias']) for row in rows+grid)))
    print('Exact cases complete:',len(rows)+len(grid),'max bias',checks[-1]['value'],flush=True)
    print('H=0 exact:',[(r['rho'],r['pooled_mse'],r['rb_mse']) for r in rows if r['history']==0],flush=True)
    # Same collection path and random numbers as the prior controller.
    rho = r8.truth(.6,8)
    old,_ = r11.simulate(rho,.5,'raw',np.random.default_rng(1201),reps=3)
    new,_ = simulate(rho,np.random.default_rng(1201),reps=3)
    assert np.array_equal(old['raw'],new['direct'])
    assert np.allclose(old['reuse_h4_terminal'],np.clip(base.rolling(new['pooled'],4),-1,1),atol=1e-12)
    checks.append(dict(check='round11_raw_collection_regression',passed=True,value=0.))

    summary,comparisons,diagnostics,bounds,traces = [],[],[],[],[]
    for si,(delta,duration) in enumerate([(None,None),(.2,8),(.6,8)]):
        scenario = 'stationary' if delta is None else f'd{delta}_L{duration}'
        rho = r8.truth(delta,duration)
        r8.validate_truth(rho)
        losses,cohorts = {},{}
        def record(name,est,cohort):
            est = np.clip(est,-1,1)
            sq = (est-rho[None])**2
            metrics = r8.reduce_loss(sq,duration)
            losses[name],cohorts[name] = metrics,cohort
            pair_loss = sq[:,WARM:].mean(axis=1)
            means = pair_loss.mean(axis=0)
            row = dict(scenario=scenario,method=name,max_pair_timeavg_mse=float(means.max()))
            for metric in ['overall','active','post10','return10']:
                values = r10.mean_ci(metrics[metric]) if metric in metrics else ('','','','')
                row.update(zip([metric+'_mse',metric+'_low',metric+'_high',metric+'_se'],values))
            summary.append(row)
            if scenario=='stationary':
                se = pair_loss.std(axis=0,ddof=1)/np.sqrt(REPS)
                zcrit = norm.ppf(1-.05/(2*M))
                bounds.append(dict(method=name,max_mean=float(means.max()),
                    simultaneous_low=float((means-zcrit*se).max()),
                    simultaneous_high=float((means+zcrit*se).max())))
            if delta==.6:
                j = base.EVENTS[0][1]
                for t in range(70,110):
                    traces.append(dict(method=name,t=t,truth=float(rho[t,j]),
                        mean_estimate=float(est[:,t,j].mean()),mse=float(sq[:,t,j].mean())))
        output,diag = simulate(rho,np.random.default_rng(np.random.SeedSequence([SEED,si,0])))
        diagnostics.append(dict(scenario=scenario,collection='lpd50_raw',**diag))
        for name in ['direct','monitor','pooled','rb']:
            record(name+'_h4',base.rolling(output[name],4),'lpd50_raw')
        for name in ['pooled','rb']:
            record(name+'_current',output[name],'lpd50_raw')
        del output
        rng = np.random.default_rng(np.random.SeedSequence([SEED,si,1]))
        y = r10.draw_rr(140,rho,rng,(REPS,T,M))
        record('rotating_h4',base.rolling(y,4),'rotation')
        diagnostics.append(dict(scenario=scenario,collection='rotation',monitor_reports=0,
            mean_total_reports=140,update_fraction=1.,max_window_total=2800))
        for a,b in [('rb_current','pooled_current'),('rb_h4','pooled_h4'),
                    ('rb_h4','monitor_h4'),('rb_h4','rotating_h4'),('pooled_h4','rotating_h4')]:
            paired = cohorts[a]==cohorts[b]
            for metric in losses[a]:
                av,bv = losses[a][metric],losses[b][metric]
                if paired:
                    difference,low,high,_ = r10.mean_ci(av-bv)
                else:
                    difference = float(av.mean()-bv.mean())
                    margin = 1.96*np.sqrt((av.var(ddof=1)+bv.var(ddof=1))/REPS)
                    low,high = difference-margin,difference+margin
                comparisons.append(dict(scenario=scenario,method=a,reference=b,metric=metric,
                    paired=paired,difference=difference,ci_low=low,ci_high=high))
        print('Finished',scenario,flush=True)
        for row in summary[-7:]:
            print(row['method'],row['max_pair_timeavg_mse'],row['active_mse'],row['return10_mse'],flush=True)
    save('summary.csv',summary)
    save('comparisons.csv',comparisons)
    save('resource_diagnostics.csv',diagnostics)
    save('stationary_intervals.csv',bounds)
    save('pulse_trace.csv',traces)
    save('validation.csv',checks)
    (OUT/'metadata.json').write_text(json.dumps(dict(seed=SEED,repeats=REPS,epsilon=1.,w=W,T=T,
        n=42000,attribute_pairs=M,exact_cases=351,stream_scenarios=3,methods=7,
        python=sys.version,numpy=np.__version__,elapsed_seconds=time.time()-started,
        rule='classical Rao-Blackwell conditional expectation, not a new method',
        target='current population correlation',rb_feedback=False,
        same_privacy_and_reports_within_lpd_collection=True,
        clipped_output_not_claimed_unbiased=True),indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
