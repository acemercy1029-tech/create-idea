"""Predeclared scalar feedback/reuse ablations. Read PROTOCOL.md.

No full-paper reproduction; no claim of adaptive grouping unbiasedness.
"""
from pathlib import Path
import csv
import importlib.util
import itertools
import json
import sys
import numpy as np

OUT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('round10', OUT.parent/'2026-10-26-monitoring-cost'/'run.py')
r10 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r10)
r8, base = r10.r8, r10.base
N, M, W, T, REPS, WARM = r10.N, r10.M, r10.W, r10.T, r10.REPS, r10.WARM
K, EPS, TARGET, PAIR_USERS = r10.K, r10.EPS, r10.TARGET, r10.PAIR_USERS
SEED = 2026102711


def save(name, rows):
    with (OUT/name).open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class Group:
    """Fresh estimates only; v is a proxy, not a selected-data risk guarantee."""
    def __init__(self, shape):
        self.count = np.zeros(shape, dtype=int)
        self.total = np.zeros(shape)
        self.var_sum = np.zeros(shape)

    def add(self, y, v, active):
        length = np.maximum(self.count, 1)
        old_mean = self.total/length
        q = (y-old_mean)**2-v-self.var_sum/length**2
        theta = ((length+1)**2*v-v-self.var_sum)/length**2
        merge = (self.count > 0) & (theta >= 0) & (q <= theta)
        reset = active & (self.count > 0) & ~merge
        self.total = np.where(active, np.where(merge, self.total+y, y), self.total)
        self.var_sum = np.where(active, np.where(merge, self.var_sum+v, v), self.var_sum)
        self.count = np.where(active, np.where(merge, self.count+1, 1), self.count)
        return np.clip(self.total/np.maximum(self.count, 1), -1, 1), reset


def group_series(y, reports):
    group = Group((y.shape[0], M))
    est = np.empty_like(y)
    resets = 0
    for t in range(T):
        est[:,t], reset = group.add(y[:,t], K**-2/reports, np.ones_like(group.count, dtype=bool))
        if t >= WARM:
            resets += int(reset.sum())
    return est, resets/((T-WARM)*y.shape[0]*M)


def simulate(rho, alpha, feedback, rng, reps=REPS):
    r_mon = int(PAIR_USERS*alpha)//W
    capacity = PAIR_USERS-r_mon*W
    shape = (reps, M)
    ring = np.zeros(shape+(W,), dtype=int)
    direct = np.zeros(shape)
    decision = np.zeros(shape)
    pub_group, pool_group = Group(shape), Group(shape)
    pool_four = np.zeros(shape+(4,))
    pool_sum = np.zeros(shape)
    all_active = np.ones(shape, dtype=bool)
    names = ['raw','group_terminal','reuse_h4_terminal','reuse_group_terminal'] if feedback == 'raw' else [feedback]
    output = {name:np.empty((reps,T,M)) for name in names}
    one_path = np.zeros(T, dtype=int)
    publication_users, publication_rounds = 0, 0
    group_resets, pool_resets, group_size, pool_size = 0, 0, 0, 0
    max_active = 0
    for t in range(T):
        slot = t % W
        active_past = ring.sum(axis=-1)-ring[...,slot]
        candidate = (capacity-active_past)//2
        monitor = r10.draw_rr(r_mon, rho[t], rng, shape)
        vhat = (K**-2-monitor**2)/(r_mon-1)
        score = (monitor-decision)**2-vhat
        risk_bound = K**-2/np.maximum(candidate,1)
        update = (candidate >= 1) & (score > risk_bound)
        used = np.where(update, candidate, 0)
        publication = r10.draw_rr(np.maximum(candidate,1), rho[t], rng, shape)
        direct = np.where(update, np.clip(publication,-1,1), direct)
        grouped, reset = pub_group.add(publication, risk_bound, update)

        # Selection depends on monitor; this ratio is NOT automatically unbiased.
        # Counterfactual publication draws with used=0 never enter any output.
        pooled = (r_mon*monitor+used*publication)/(r_mon+used)
        pooled_grouped, pool_reset = pool_group.add(pooled, K**-2/(r_mon+used), all_active)
        pool_slot = t % 4
        pool_sum += pooled-pool_four[...,pool_slot]
        pool_four[...,pool_slot] = pooled
        pooled_h4 = np.clip(pool_sum/min(t+1,4),-1,1)
        if feedback == 'raw':
            decision = direct
            output['raw'][:,t] = direct
            output['group_terminal'][:,t] = grouped
            output['reuse_h4_terminal'][:,t] = pooled_h4
            output['reuse_group_terminal'][:,t] = pooled_grouped
        elif feedback == 'group_feedback':
            decision = grouped
            output[feedback][:,t] = grouped
        elif feedback == 'reuse_h4_feedback':
            decision = pooled_h4
            output[feedback][:,t] = pooled_h4
        elif feedback == 'reuse_group_feedback':
            decision = pooled_grouped
            output[feedback][:,t] = pooled_grouped
        else:
            raise ValueError(feedback)
        ring[...,slot] = used
        window = ring.sum(axis=-1)
        assert np.all(window <= capacity)
        max_active = max(max_active, int(window.max()))
        one_path[t] = used[0,0]
        if t >= WARM:
            publication_users += int(used.sum())
            publication_rounds += int(update.sum())
            group_resets += int(reset.sum())
            pool_resets += int(pool_reset.sum())
            group_size += int(pub_group.count.sum())
            pool_size += int(pool_group.count.sum())
    r10.check_id_schedule(one_path, capacity)
    denom = (T-WARM)*reps*M
    diag = dict(alpha=alpha,feedback=feedback,monitor_reports=r_mon,
        mean_publication_reports=publication_users/denom,
        mean_total_reports=r_mon+publication_users/denom,
        publication_fraction=publication_rounds/denom,
        max_window_publication=max_active,max_window_total=max_active+r_mon*W,
        fresh_group_resets_per_round=group_resets/denom,
        pool_group_resets_per_round=pool_resets/denom,
        mean_fresh_group_size=group_size/denom,mean_pool_group_size=pool_size/denom)
    if feedback == 'raw':
        output['h4'] = np.clip(base.rolling(output['raw'],4),-1,1)
        output['h20'] = np.clip(base.rolling(output['raw'],20),-1,1)
    return output, diag


def validate():
    checks = []
    base.validate_model_and_privacy()
    # Fixed equal variance: compare the simplified original scalar decision.
    rng = np.random.default_rng(1911)
    group = Group((20,))
    group.count = rng.integers(1,30,20)
    group.total = rng.normal(0,.5,20)*group.count
    v = .04
    group.var_sum = group.count*v
    y = rng.normal(0,.5,20)
    expected_merge = (y-group.total/group.count)**2 <= 2*(1+1/group.count)*v
    expected_mean = np.where(expected_merge,(group.total+y)/(group.count+1),y)
    est, reset = group.add(y,v,np.ones(20,dtype=bool))
    assert np.array_equal(reset, ~expected_merge)
    assert np.allclose(est,np.clip(expected_mean,-1,1))
    checks.append(dict(check='equal_variance_group_identity',passed=True,detail='20 independent states'))
    previous = [x.copy() for x in [group.total,group.var_sum,group.count]]
    group.add(np.full(20,100.),100.,np.zeros(20,dtype=bool))
    assert all(np.array_equal(a,b) for a,b in zip(previous,[group.total,group.var_sum,group.count]))
    checks.append(dict(check='no_report_no_new_group_observation',passed=True,detail='state unchanged'))
    # Exact original collection regression; small repeat count, all 240 timestamps.
    old_reps = r10.REPS
    try:
        r10.REPS = 8
        rho = r8.truth(.6,8)
        for alpha in [.25,.5]:
            expected, _, _, _ = r10.run_lpd(rho,alpha,'corrected',np.random.default_rng(716))
            actual, _ = simulate(rho,alpha,'raw',np.random.default_rng(716),reps=8)
            assert np.array_equal(actual['raw'],expected)
            checks.append(dict(check='round10_exact_collection_alpha_'+str(alpha),passed=True,detail='8 repeats x 240 x 15'))
    finally:
        r10.REPS = old_reps
    return checks


def main():
    checks = validate()
    rows, diagnostics, differences, traces = [], [], [], []
    cases = [(None,None)]+list(itertools.product([.2,.6],[2,8,'persistent']))
    for si,(delta,duration) in enumerate(cases):
        scenario = 'stationary' if delta is None else f'd{delta}_L{duration}'
        rho = r8.truth(delta,duration)
        r8.validate_truth(rho)
        losses, cohorts = {}, {}

        def record(name, est, cohort):
            est = np.clip(est,-1,1)
            sq = (est-rho[None])**2
            loss = r8.reduce_loss(sq,duration)
            losses[name], cohorts[name] = loss, cohort
            item = dict(scenario=scenario,method=name,cohort=cohort,
                max_pair_timeavg_mse=float(sq[:,WARM:].mean(axis=(0,1)).max()))
            for metric in ['overall','active','post10','return10']:
                stats = r10.mean_ci(loss[metric]) if metric in loss else ('','','','')
                item.update(zip([metric+'_mse',metric+'_low',metric+'_high',metric+'_se'],stats))
            rows.append(item)
            if delta == .6 and duration == 8:
                j = base.EVENTS[0][1]
                for t in range(70,110):
                    traces.append(dict(method=name,t=t,truth=float(rho[t,j]),
                        mean_estimate=float(est[:,t,j].mean()),mse=float(sq[:,t,j].mean())))

        rng = np.random.default_rng(np.random.SeedSequence([SEED,si,0]))
        y = r10.draw_rr(PAIR_USERS//W,rho,rng,(REPS,T,M))
        for h in [1,4,5,20]:
            record(f'rotating_h{h}',base.rolling(y,h),'rotation')
        recent, long = base.rolling(y,5),base.rolling(y,20)
        older = (20*long-5*recent)/15
        se = np.sqrt(K**-2/(PAIR_USERS//W)*(1/5+1/15))
        for z in [2,3]:
            short = abs(recent-older)>z*se
            short[:,:19] = False
            record(f'rotating_z{z}',np.where(short,recent,long),'rotation')
        grouped, reset_rate = group_series(y,PAIR_USERS//W)
        record('rotating_group',grouped,'rotation')
        del y, recent, long, older, grouped
        for ai,alpha in enumerate([.25,.5]):
            for fi,feedback in enumerate(['raw','group_feedback','reuse_h4_feedback','reuse_group_feedback']):
                rng = np.random.default_rng(np.random.SeedSequence([SEED,si,10+ai*4+fi]))
                output, diag = simulate(rho,alpha,feedback,rng)
                prefix = f'lpd_a{int(100*alpha)}_'
                cohort = prefix+feedback
                diagnostics.append(dict(scenario=scenario,**diag))
                for name,est in output.items():
                    record(prefix+name,est,cohort)
                del output

        def compare(a,b):
            paired = cohorts[a] == cohorts[b]
            for metric in losses[a]:
                x, y = losses[a][metric], losses[b][metric]
                if paired:
                    diff,low,high,_ = r10.mean_ci(x-y)
                else:
                    diff = float(x.mean()-y.mean())
                    half = 1.96*np.sqrt(x.var(ddof=1)/REPS+y.var(ddof=1)/REPS)
                    low,high = diff-half,diff+half
                differences.append(dict(scenario=scenario,method=a,reference=b,metric=metric,
                    paired=paired,mean_difference=diff,ci_low=low,ci_high=high))
        for name in losses:
            if name != 'rotating_h4': compare(name,'rotating_h4')
        for alpha in [25,50]:
            prefix = f'lpd_a{alpha}_'
            for name in ['group_terminal','reuse_h4_terminal','reuse_group_terminal']:
                compare(prefix+name,prefix+'h4')
            for feedback,terminal in [('group_feedback','group_terminal'),
                                       ('reuse_h4_feedback','reuse_h4_terminal'),
                                       ('reuse_group_feedback','reuse_group_terminal')]:
                compare(prefix+feedback,prefix+terminal)
        print('Finished',scenario,flush=True)
    assert len(rows) == 175
    save('summary.csv',rows)
    save('resource_diagnostics.csv',diagnostics)
    save('comparisons.csv',differences)
    save('pulse_trace.csv',traces)
    save('validation.csv',checks)
    stable = [dict(method=r['method'],max_pair_timeavg_mse=r['max_pair_timeavg_mse'],
        meets_point_target=r['max_pair_timeavg_mse']<=TARGET) for r in rows if r['scenario']=='stationary']
    save('stationary_constraint.csv',stable)
    metadata = dict(seed=SEED,repeats=REPS,n=N,m=M,w=W,T=T,epsilon=EPS,
        target='current population correlation',stationary_working_target=TARGET,
        monitoring_alphas=[.25,.5],python=sys.version,numpy=np.__version__,
        collection_strategies_per_alpha=4,methods=25,scenarios=7,
        all_outputs_causal=True,full_paper_reproduction=False,
        grouping_proxy_not_conditional_variance_guarantee=True,
        adaptive_pooled_estimate_not_claimed_unbiased=True)
    (OUT/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print('Stationary:')
    for row in stable: print(row)
    print('Strong 8-round pulse:')
    for row in rows:
        if row['scenario']=='d0.6_L8':
            print(row['method'],row['overall_mse'],row['active_mse'],row['return10_mse'])


if __name__ == '__main__':
    main()
