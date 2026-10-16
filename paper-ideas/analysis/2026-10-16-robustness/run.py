"""Predeclared baseline robustness study; no new mechanism or paper reproduction."""
from pathlib import Path
import csv
import importlib.util
import itertools
import json
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('round6', OUT.parent/'2026-10-15-multipair'/'run_experiment.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
N, M, W, T, REPS, WARM = base.N, base.M, base.W, base.T, base.REPS, base.WARM
EPS, TARGET, SEED = 1., .01, 2026101608
R = N//(W*M)
KAPPA = np.tanh(EPS/2)
H_MIN = int(np.ceil(KAPPA**-2/(R*TARGET)))
HORIZONS = [1, H_MIN, 5, 20, 40]
AMPLITUDES = [.2, .4, .6]
DURATIONS = [2, 8, 20, 'persistent']


def save(name, rows):
    with (OUT/name).open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def truth(delta=None, duration=None):
    rho = np.zeros((T, M))
    for idx, (start, j, sign) in enumerate(base.EVENTS):
        if delta is None:
            rho[:, j] = [.2, -.4, .6][idx]
        else:
            stop = T if duration == 'persistent' else start+duration
            rho[start:stop, j] = np.sign(sign)*delta
    return rho


def validate_truth(rho):
    states = np.array(list(itertools.product([-1, 1], repeat=6)))
    for values in np.unique(rho, axis=0):
        p = np.ones(len(states))/64
        for b, (_, j, _) in enumerate(base.EVENTS):
            p *= 1+values[j]*states[:, 2*b]*states[:, 2*b+1]
        assert p.min() >= 0 and np.isclose(p.sum(), 1)
        assert np.allclose(p @ states, 0)
        cross = [p @ (states[:, a]*states[:, b]) for a, b in base.PAIRS]
        assert np.allclose(cross, values)


def noise_reports(rho, index):
    rng = np.random.default_rng(np.random.SeedSequence([SEED, index]))
    count = rng.binomial(R, (1+KAPPA*rho)/2, size=(REPS, T, M))
    return (2*count/R-1)/KAPPA


def estimates(y):
    for h in HORIZONS:
        yield f'rotating_h{h}', base.rolling(y, h)
    recent, long = base.rolling(y, 5), base.rolling(y, 20)
    older = (20*long-5*recent)/15
    se = np.sqrt(KAPPA**-2/R*(1/5+1/15))
    for z in [2, 3]:
        short = abs(recent-older) > z*se
        short[:, :19] = False
        yield f'rotating_z{z}', np.where(short, recent, long)
    yield 'constant_zero', np.zeros_like(y)


def reduce_loss(sq, duration):
    vals = {'overall': sq[:, WARM:].mean(axis=(1, 2))}
    if duration is None:
        return vals
    length = 20 if duration == 'persistent' else duration
    for name, offset, size in [('active', 0, length), ('post10', 0, 10)]:
        vals[name] = np.stack([sq[:, t+offset:t+offset+size, j].mean(axis=1)
                              for t, j, _ in base.EVENTS], axis=1).mean(axis=1)
    if duration != 'persistent':
        vals['return10'] = np.stack([sq[:, t+duration:t+duration+10, j].mean(axis=1)
                                    for t, j, _ in base.EVENTS], axis=1).mean(axis=1)
    return vals


def exact_loss(rho, h):
    mean = base.rolling(rho[None], h)[0]
    v = (KAPPA**-2-rho**2)/R
    length = np.minimum(np.arange(1, T+1), h)
    variance = base.rolling(v[None], h)[0]/length[:, None]
    return variance+(mean-rho)**2


def main():
    assert H_MIN == 4
    base.validate_model_and_privacy()
    summary, stationary, paired, checks = [], [], [], []
    cases = [(None, None)]+list(itertools.product(AMPLITUDES, DURATIONS))
    for idx, (delta, duration) in enumerate(cases):
        rho = truth(delta, duration)
        validate_truth(rho)
        y = noise_reports(rho, idx)
        losses_by_method = {}
        for name, raw in estimates(y):
            est = np.clip(raw, -1, 1)
            sq = (est-rho[None])**2
            losses = reduce_loss(sq, duration)
            if delta is None:
                stationary.append(dict(method=name,
                    max_pair_timeavg_mse=float(sq[:, WARM:].mean(axis=(0, 1)).max()),
                    overall_mse=float(losses['overall'].mean())))
            else:
                row = dict(delta=delta, duration=duration, method=name,
                           active_eval_length=20 if duration == 'persistent' else duration)
                for metric in ['overall', 'active', 'post10', 'return10']:
                    vals = base.mean_ci(losses[metric]) if metric in losses else ('', '', '')
                    row.update(zip([metric+'_mse', metric+'_low', metric+'_high'], vals))
                row['active_over_delta_squared'] = row['active_mse']/delta**2
                summary.append(row)
            losses_by_method[name] = losses
            if name.startswith('rotating_h'):
                h = int(name.split('_h')[1])
                empirical = reduce_loss((raw-rho[None])**2, duration)
                analytic = reduce_loss(exact_loss(rho, h)[None], duration)
                for metric, per_rep in empirical.items():
                    avg, lo, hi = base.mean_ci(per_rep)
                    expected = float(analytic[metric][0])
                    checks.append(dict(delta='stationary' if delta is None else delta,
                        duration='' if duration is None else duration, method=name, metric=metric,
                        analytic_unclipped_mse=expected, empirical_unclipped_mse=avg,
                        ci_low=lo, ci_high=hi, relative_error=abs(avg/expected-1)))
            del est, sq, raw
        if delta is not None:
            # Paired intervals, not differences between independently simulated methods.
            for left, right in itertools.combinations(['rotating_h4','rotating_h5',
                                                       'rotating_h20','rotating_z2','rotating_z3'], 2):
                for metric in ['active', 'post10', 'return10']:
                    if metric not in losses_by_method[left]:
                        continue
                    avg, lo, hi = base.mean_ci(losses_by_method[left][metric]-losses_by_method[right][metric])
                    paired.append(dict(delta=delta, duration=duration, comparison=left+' minus '+right,
                                       metric=metric, mean_difference=avg, ci_low=lo, ci_high=hi))
        del y, losses_by_method
    for row in stationary:
        row['meets_point_target'] = row['max_pair_timeavg_mse'] <= TARGET
    feasible = {r['method'] for r in stationary if r['meets_point_target']}
    comparisons = []
    for delta, duration in cases[1:]:
        selected = [r for r in summary if r['delta']==delta and r['duration']==duration and r['method'] in feasible]
        winner = min(selected, key=lambda row: row['active_mse'])
        h4 = next(r for r in selected if r['method']=='rotating_h4')
        h5 = next(r for r in selected if r['method']=='rotating_h5')
        comparisons.append(dict(delta=delta, duration=duration, lowest_observed_active_method=winner['method'],
            lowest_observed_active_mse=winner['active_mse'], h4_active_mse=h4['active_mse'],
            h5_active_mse=h5['active_mse'], h4_to_lowest_ratio=h4['active_mse']/winner['active_mse'],
            h5_to_lowest_ratio=h5['active_mse']/winner['active_mse']))
    assert max(r['relative_error'] for r in checks) < .05, 'Unclipped theory/MC mismatch >5%'
    save('summary.csv', summary)
    save('stationary_constraint.csv', stationary)
    save('paired_differences.csv', paired)
    save('analytic_checks.csv', checks)
    save('scenario_comparison.csv', comparisons)
    (OUT/'metadata.json').write_text(json.dumps(dict(seed=SEED, repeats=REPS, n=N, w=W,
        T=T, epsilon=EPS, d=6, m=M, reports_per_pair_round=R, communication_bits=1,
        stationary_target=TARGET, h_min_from_stationary_formula=H_MIN,
        theoretical_stationary_h4_upper=KAPPA**-2/(R*H_MIN), amplitudes=AMPLITUDES,
        durations=DURATIONS, horizons=HORIZONS, python=sys.version, numpy=np.__version__,
        matplotlib=matplotlib.__version__, max_analytic_relative_error=max(r['relative_error'] for r in checks),
        main_outputs_clipped=True, full_paper_reproduction=False), indent=2), encoding='utf-8')
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9), sharey=True)
    colors = {'rotating_h4':'#247ba0','rotating_h5':'#679436','rotating_h20':'#d78a25',
              'rotating_z2':'#8054a0','rotating_z3':'#c45a68'}
    for ax, delta in zip(axes, AMPLITUDES):
        for method, color in colors.items():
            vals = [next(r['active_over_delta_squared'] for r in summary
                         if r['delta']==delta and r['duration']==length and r['method']==method) for length in DURATIONS]
            ax.plot(range(4), vals, marker='o', label=method.replace('rotating_', ''), color=color)
        ax.axhline(1, color='gray', ls='--', lw=1, label='always zero (infeasible)')
        ax.set_xticks(range(4), ['2', '8', '20', 'persistent*'])
        ax.set_xlabel('Event duration (rounds)')
        ax.set_title(f'Change amplitude = {delta}')
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Active-period MSE / amplitude squared')
    axes[2].legend(fontsize=8)
    fig.suptitle('Private rotating cohorts; *persistent evaluated over first 20 rounds')
    fig.tight_layout()
    fig.savefig(OUT/'active_error.png', dpi=170)
    fig.savefig(OUT/'active_error.svg')
    svg = OUT/'active_error.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n', encoding='utf-8')
    print(json.dumps({'scenarios':len(cases), 'summary_rows':len(summary),
        'checks':len(checks), 'max_analytic_relative_error':max(r['relative_error'] for r in checks)}, indent=2))
    for row in stationary:
        print(row)
    for row in comparisons:
        print(row)


if __name__ == '__main__':
    main()
