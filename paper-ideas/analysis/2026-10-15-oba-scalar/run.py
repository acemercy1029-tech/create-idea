"""MTSP-LDP OBA *module* adaptation; see PROTOCOL.md for all differences."""
from pathlib import Path
import importlib.util
import csv
import json
import numpy as np

OUT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('round6', OUT.parent/'2026-10-15-multipair'/'run_experiment.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
N, M, W, T, REPS, WARM = base.N, base.M, base.W, base.T, base.REPS, base.WARM
EPS = 1.0
SEED = 2026101507
R = N//M


def variance(a):
    return (1/np.tanh(a/2)**2-1)/R


def choose_oba(scores, current, remaining):
    ordered = np.sort(scores, axis=-1)[..., ::-1]
    prefix = np.cumsum(ordered, axis=-1)
    total = prefix[..., -1:]
    ks = np.arange(1, W+1)
    costs = np.concatenate([total, total-prefix+ks*variance(EPS/(2*ks))], axis=-1)
    k = np.argmin(costs, axis=-1)
    # Current is the newest; it wins equal-score ties.
    rank = (scores > current[..., None]).sum(axis=-1)
    selected = (k > 0) & (rank < k)
    return np.where(selected, np.minimum(remaining, EPS/(2*np.maximum(k, 1))), 0)


def validate_selector():
    rng = np.random.default_rng(173)
    samples = rng.normal(size=(100, W))
    samples[0] = 0
    remain = rng.uniform(0, .5, size=100)
    actual = choose_oba(samples, samples[:, -1], remain)
    expected = []
    for row, left in zip(samples, remain):
        order = sorted(range(W), key=lambda j: (-row[j], -j))
        costs = [sum(row)]
        for k in range(1, W+1):
            costs.append(k*variance(EPS/(2*k))+sum(row[j] for j in order[k:]))
        best = int(np.argmin(costs))
        expected.append(min(left, EPS/(2*best)) if best and W-1 in order[:best] else 0)
    assert np.allclose(actual, expected)


def rr_from_counts(plus_raw, a, rng):
    prob_keep = (1+np.tanh(a/2))/2
    observed_plus = rng.binomial(plus_raw, prob_keep)+rng.binomial(R-plus_raw, 1-prob_keep)
    return (2*observed_plus/R-1)/np.tanh(a/2)


def run_oba(rho, rng):
    shape = (REPS, M)
    published = np.zeros((REPS, T, M))
    monitor_all = np.zeros_like(published)
    ring = np.zeros(shape+(W,))
    scores = np.zeros_like(ring)
    budget_trace = np.zeros_like(published)
    detect_a = EPS/(2*W)
    monitor_var = variance(detect_a)
    max_window, nonzero, capped, requested = 0., 0, 0, 0
    for t in range(T):
        plus_raw = rng.binomial(R, (1+rho[t])/2, size=shape)
        slot = t % W
        if t < W:
            published[:, t] = np.clip(rr_from_counts(plus_raw, EPS/W, rng), -1, 1)
            monitor_all[:, t] = published[:, t]
            ring[..., slot] = EPS/(2*W)
        else:
            monitor = rr_from_counts(plus_raw, detect_a, rng)
            monitor_all[:, t] = monitor
            prev = published[:, t-1]
            score = (monitor-prev)**2-monitor_var
            scores[..., slot] = score
            spent = ring.sum(axis=-1)-ring[..., slot]
            remaining = np.maximum(0, EPS/2-spent)
            wanted = choose_oba(scores, score, np.full(shape, EPS/2))
            a = choose_oba(scores, score, remaining)
            requested += int((wanted > 0).sum())
            capped += int((wanted > a+1e-10).sum())
            active = a > 1e-10
            a = np.where(active, a, 0)
            # Use the same latent records as monitoring; noise coins are fresh.
            result = prev.copy()
            if active.any():
                result[active] = rr_from_counts(plus_raw[active], a[active], rng)
            published[:, t] = np.clip(result, -1, 1)
            ring[..., slot] = a
            nonzero += int(active.sum())
        budget_trace[:, t] = ring[..., slot]
        max_window = max(max_window, float(ring.sum(axis=-1).max()+EPS/2))
        assert max_window <= EPS+1e-9
    diagnostics = dict(max_window_budget=max_window,
        publish_fraction=nonzero/((T-W)*REPS*M),
        capped_fraction_among_requests=capped/max(requested, 1),
        mean_publication_budget=float(budget_trace[:, WARM:].mean()),
        monitoring_budget_per_round=detect_a,
        monitoring_conditional_noise_variance=float(monitor_var))
    return published, monitor_all, diagnostics


def row_metrics(est, rho, scenario, name):
    sq = (est-rho[None])**2
    overall = sq[:, WARM:].mean(axis=(1,2))
    mean, lo, hi = base.mean_ci(overall)
    per_pair = sq[:, WARM:].mean(axis=(0,1))
    row = dict(scenario=scenario, method=name, overall_mse=mean, ci_low=lo, ci_high=hi,
               max_pair_timeavg_mse=float(per_pair.max()))
    if scenario != 'stationary':
        segments = [sq[:, t:t+10,j].mean(axis=1) for t,j,_ in base.EVENTS]
        pm, pl, ph = base.mean_ci(np.stack(segments,axis=1).mean(axis=1))
        row.update(post10_mse=pm, post10_low=pl, post10_high=ph)
        active = [sq[:,t:t+8,j].mean(axis=1) for t,j,_ in base.EVENTS]
        row['active8_mse'] = float(np.stack(active,axis=1).mean())
        row['return10_mse'] = (float(np.stack([sq[:,t+8:t+18,j].mean(axis=1)
                                  for t,j,_ in base.EVENTS],axis=1).mean()) if scenario == 'pulse8' else '')
    else:
        row.update(post10_mse='', post10_low='', post10_high='', active8_mse='', return10_mse='')
    return row


def save(name, rows):
    with (OUT/name).open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    validate_selector()
    base.validate_model_and_privacy()
    rows, diagnostics, traces = [], [], []
    for si, scenario in enumerate(['stationary','persistent','pulse8']):
        rho = base.truth(scenario) if scenario != 'stationary' else np.zeros((T,M))
        if scenario == 'stationary':
            for (_,j,_), value in zip(base.EVENTS, [.2,-.4,.6]):
                rho[:,j] = value
        rng = np.random.default_rng(np.random.SeedSequence([SEED,si,0]))
        raw, monitors, diag = run_oba(rho, rng)
        diagnostics.append(dict(scenario=scenario, **diag))
        estimates = {'oba_scalar':raw,
                     'oba_scalar_h5':np.clip(base.rolling(raw,5),-1,1),
                     'oba_scalar_h20':np.clip(base.rolling(raw,20),-1,1)}
        # Monitor-only control clarifies how informative the measurement is.
        estimates['monitor_only_h20'] = np.clip(base.rolling(monitors,20),-1,1)
        for schedule_id, schedule in enumerate(['rotating','uniform']):
            rng = np.random.default_rng(np.random.SeedSequence([SEED,si,schedule_id+1]))
            a = EPS if schedule == 'rotating' else EPS/W
            r = N//(W*M) if schedule == 'rotating' else N//M
            kappa = np.tanh(a/2)
            y = (2*rng.binomial(r,(1+kappa*rho)/2,size=(REPS,T,M))/r-1)/kappa
            hs = [1,5,20,40] if schedule == 'rotating' else [20]
            for h in hs:
                estimates[f'{schedule}_h{h}'] = np.clip(base.rolling(y,h),-1,1)
            if schedule == 'rotating':
                recent, long = base.rolling(y,5), base.rolling(y,20)
                older = (20*long-5*recent)/15
                for z in [2,3]:
                    choose_short = abs(recent-older)>z*np.sqrt(kappa**-2/r*(1/5+1/15))
                    choose_short[:,:19] = False
                    estimates[f'rotating_z{z}'] = np.clip(np.where(choose_short,recent,long),-1,1)
        estimates['constant_zero'] = np.zeros_like(raw)
        for name, est in estimates.items():
            rows.append(row_metrics(est,rho,scenario,name))
            if scenario != 'stationary' and name in ['oba_scalar','oba_scalar_h20','rotating_z3']:
                j=base.EVENTS[0][1]
                for t in range(60,115):
                    traces.append(dict(scenario=scenario,method=name,t=t,truth=float(rho[t,j]),
                                       mean_estimate=float(est[:,t,j].mean())))
        del estimates, raw, monitors
    save('summary.csv',rows); save('budget_diagnostics.csv',diagnostics); save('traces.csv',traces)
    stationary = {r['method']:r['max_pair_timeavg_mse'] for r in rows if r['scenario']=='stationary'}
    feasible = [dict(method=k,stationary_max_pair_mse=v,meets_point_target=v<=.01) for k,v in stationary.items()]
    save('stationary_constraint.csv',feasible)
    (OUT/'metadata.json').write_text(json.dumps(dict(seed=SEED,repeats=REPS,n=N,w=W,T=T,
       epsilon=EPS,stationary_target=.01,communication_cap_bits=2,
       source='https://arxiv.org/html/2512.24899v1',full_paper_reproduction=False,
       smallest_monitor_std=float(np.sqrt(variance(EPS/(2*W)))),
       python_dependencies={'numpy':np.__version__}),indent=2),encoding='utf-8')
    for r in rows:
        print(r['scenario'],r['method'],round(r['overall_mse'],5),
              round(r['max_pair_timeavg_mse'],5),r['post10_mse'])
    print(json.dumps(diagnostics,indent=2))


if __name__=='__main__':
    main()
