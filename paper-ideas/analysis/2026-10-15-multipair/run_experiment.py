"""Minimal baseline experiment; not a reproduction of MTSP-LDP or a new method.

Run: python run_experiment.py
Only numpy and matplotlib are needed. No private or external data are used.
Aggregated binomial sampling is exact for the documented independent-record model.
"""
from pathlib import Path
import csv
import itertools
import json
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
REPS, N, W, T, WARM = 1000, 42000, 20, 240, 40
PAIRS = list(itertools.combinations(range(6), 2))
M = len(PAIRS)
EPSILONS = [0.3, 1.0, 3.0]
HORIZONS = [1, 5, 20, 40]
SEED = 20261015
EVENTS = [(80, PAIRS.index((0, 1)), 0.6),
          (120, PAIRS.index((2, 3)), -0.6),
          (160, PAIRS.index((4, 5)), 0.6)]  # zero-based timestamps


def truth(scenario):
    rho = np.zeros((T, M))
    for start, j, value in EVENTS:
        stop = T if scenario == 'persistent' else start + 8
        rho[start:stop, j] = value
    return rho


def validate_model_and_privacy():
    # Feasible 6-dimensional joint distribution with three independent blocks.
    x = np.array(list(itertools.product([-1, 1], repeat=6)))
    for block_rho in ([0, 0, 0], [.6, -.6, .6]):
        p = np.ones(len(x)) / 64
        for b, r in enumerate(block_rho):
            p *= 1 + r * x[:, 2*b] * x[:, 2*b+1]
        assert np.all(p >= 0) and np.isclose(p.sum(), 1)
        assert np.allclose(p @ x, 0)
        cross = np.array([p @ (x[:, a]*x[:, b]) for a, b in PAIRS])
        expected = np.zeros(M)
        for b, r in enumerate(block_rho):
            expected[PAIRS.index((2*b, 2*b+1))] = r
        assert np.allclose(cross, expected)
    # Every public cohort phase has <= epsilon in every rolling W interval.
    assert N % (W*M) == 0
    for eps in EPSILONS:
        for phase in range(W):
            ledger = np.where(np.arange(T) % W == phase, eps, 0)
            assert np.max(np.convolve(ledger, np.ones(W), 'valid')) <= eps + 1e-12
        ledger = np.full(T, eps/W)
        assert np.max(np.convolve(ledger, np.ones(W), 'valid')) <= eps + 1e-12


def rolling(y, h):
    cs = np.concatenate([np.zeros_like(y[:, :1]), np.cumsum(y, axis=1)], axis=1)
    end = np.arange(1, T+1)
    begin = np.maximum(0, end-h)
    return (cs[:, end] - cs[:, begin]) / (end-begin)[None, :, None]


def mean_ci(per_rep):
    mean = float(per_rep.mean())
    half = float(1.96 * per_rep.std(ddof=1) / np.sqrt(len(per_rep)))
    return mean, mean-half, mean+half


def metrics(est, rho):
    sq = (est-rho[None])**2
    overall = sq[:, WARM:].mean(axis=(1, 2))
    post = np.stack([sq[:, t:t+10, j].mean(axis=1) for t, j, _ in EVENTS], axis=1).mean(axis=1)
    # E over repeats first, then max over attribute pairs, then time average.
    worst = float(sq[:, WARM:].mean(axis=0).max(axis=1).mean())
    return overall, post, worst


def save_csv(name, rows):
    with (OUT/name).open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    validate_model_and_privacy()
    summaries, differences, checks, traces, curves = [], [], [], [], []
    for scenario_id, scenario in enumerate(['persistent', 'pulse8']):
        rho = truth(scenario)
        for ei, eps in enumerate(EPSILONS):
            # Diagnostic only: no private reports, always predicts zero.
            # Added after the first run to audit global-MSE dilution; no tuning.
            overall, post, worst = metrics(np.zeros((REPS, T, M)), rho)
            om, ol, oh = mean_ci(overall)
            pm, pl, ph = mean_ci(post)
            summaries.append(dict(scenario=scenario, epsilon=eps, method='constant_zero',
                reports_per_pair_round=0, per_report_epsilon=0,
                overall_mse=om, overall_ci_low=ol, overall_ci_high=oh,
                changed_pair_post10_mse=pm, post10_ci_low=pl, post10_ci_high=ph,
                mean_worst_pair_mse=worst))
            for si, schedule in enumerate(['uniform', 'rotating']):
                reports = N//M if schedule == 'uniform' else N//(W*M)
                a = eps/W if schedule == 'uniform' else eps
                kappa = np.tanh(a/2)
                rng = np.random.default_rng(np.random.SeedSequence([SEED, scenario_id, ei, si]))
                plus = rng.binomial(reports, (1+kappa*rho)/2, size=(REPS, T, M))
                y = (2*plus/reports-1)/kappa
                del plus
                empirical = ((y-rho[None])**2)[:, WARM:].mean(axis=(1, 2))
                exact = ((kappa**-2-rho**2)/reports)[WARM:].mean()
                me, lo, hi = mean_ci(empirical)
                checks.append(dict(scenario=scenario, epsilon=eps, schedule=schedule,
                                   analytic_unclipped_h1_mse=float(exact), empirical_mse=me,
                                   ci_low=lo, ci_high=hi, relative_error=float(abs(me/exact-1))))
                estimates = {f'h{h}': rolling(y, h) for h in HORIZONS}
                if schedule == 'rotating':
                    # A conventional two-window threshold postprocessing control.
                    # Uses only noisy data through current t; fixed z values, no truth access.
                    recent = estimates['h5']
                    older = (20*estimates['h20']-5*recent)/15
                    se_upper = np.sqrt((kappa**-2/reports)*(1/5+1/15))
                    for z in [2, 3]:
                        changed = np.abs(recent-older) > z*se_upper
                        changed[:, :19] = False
                        estimates[f'threshold_z{z}'] = np.where(changed, recent, estimates['h20'])
                per_rep_post = {}
                for method, raw in estimates.items():
                    # Common valid-range postprocessing for ALL main comparisons.
                    est = np.clip(raw, -1, 1)
                    overall, post, worst = metrics(est, rho)
                    om, ol, oh = mean_ci(overall)
                    pm, pl, ph = mean_ci(post)
                    name = f'{schedule}_{method}'
                    summaries.append(dict(scenario=scenario, epsilon=eps, method=name,
                        reports_per_pair_round=reports, per_report_epsilon=a,
                        overall_mse=om, overall_ci_low=ol, overall_ci_high=oh,
                        changed_pair_post10_mse=pm, post10_ci_low=pl, post10_ci_high=ph,
                        mean_worst_pair_mse=worst))
                    per_rep_post[method] = post
                    if eps == 1 and schedule == 'rotating':
                        j = EVENTS[0][1]
                        for t in range(60, 115):
                            traces.append(dict(scenario=scenario, method=name, t=t,
                                truth=float(rho[t, j]), mean_estimate=float(est[:, t, j].mean()),
                                mse=float(((est[:, t, j]-rho[t, j])**2).mean())))
                        if method in ['h1', 'h5', 'h20', 'threshold_z2']:
                            curves.append((scenario, method, est[:, :, j].mean(axis=0)))
                if schedule == 'rotating':
                    for adaptive in ['threshold_z2', 'threshold_z3']:
                        for fixed in ['h1', 'h5', 'h20']:
                            diff = per_rep_post[adaptive]-per_rep_post[fixed]
                            dm, dl, dh = mean_ci(diff)
                            differences.append(dict(scenario=scenario, epsilon=eps,
                                comparison=f'{adaptive} minus {fixed}', mean_difference=dm,
                                ci_low=dl, ci_high=dh))
                del estimates, y
    assert max(x['relative_error'] for x in checks) < .02, 'Analytic sanity check failed'
    save_csv('summary.csv', summaries)
    save_csv('paired_differences.csv', differences)
    save_csv('analytic_checks.csv', checks)
    save_csv('first_pair_trace.csv', traces)
    metadata = dict(seed=SEED, repeats=REPS, n=N, w=W, T=T, warmup=WARM,
        d=6, pairs=PAIRS, epsilons=EPSILONS, events_zero_based=EVENTS,
        pulse_duration=8, post_metric_length=10, horizons=HORIZONS,
        python=sys.version, numpy=np.__version__, matplotlib=matplotlib.__version__,
        validates=['rolling window privacy ledger', 'feasible joint family', 'analytic unclipped MSE'],
        main_estimators_clipped=True, monte_carlo_ci='normal 95% interval over independent runs')
    (OUT/'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    colors = {'h1':'#a6a6a6', 'h5':'#167aa5', 'h20':'#da8623', 'threshold_z2':'#397c48'}
    for ax, scenario in zip(axes, ['persistent', 'pulse8']):
        ax.plot(np.arange(60, 115), truth(scenario)[60:115, EVENTS[0][1]],
                color='black', linewidth=2, linestyle='--', label='True current correlation')
        for sc, name, values in curves:
            if sc == scenario:
                ax.plot(np.arange(60,115), values[60:115], label=name, color=colors[name])
        ax.set_title('Persistent change' if scenario == 'persistent' else '8-round pulse')
        ax.set_xlabel('Time (zero based)')
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Mean estimate over 1,000 runs')
    axes[1].legend(fontsize=8)
    fig.suptitle('Rotating cohorts, epsilon=1, w=20, 15 pairs; synthetic baseline study')
    fig.tight_layout()
    fig.savefig(OUT/'tracking.png', dpi=170)
    fig.savefig(OUT/'tracking.svg')
    print(json.dumps({'rows':len(summaries), 'max_analytic_relative_error':max(x['relative_error'] for x in checks)}, indent=2))
    for row in summaries:
        if row['epsilon'] == 1:
            print(row['scenario'], row['method'], round(row['overall_mse'], 6),
                  round(row['changed_pair_post10_mse'], 6), round(row['mean_worst_pair_mse'], 6))


if __name__ == '__main__':
    main()
