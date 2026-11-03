from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
with (OUT / 'summary.csv').open(encoding='utf-8') as f:
    data = list(csv.DictReader(f))
with (OUT / 'effective_samples.csv').open(encoding='utf-8') as f:
    effective = list(csv.DictReader(f))
lambdas = sorted(set(float(r['lam']) for r in data))
plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(1, 3, figsize=(15.8, 4.7))
colors = {'local16': '#197a72', 'local64': '#c87918', 'server': '#6866ad'}
for ax, scene, period, title in zip(axes[:2], ['constant', 'migration'], ['stable', 'first96'],
                                   ['Steady association', 'First 96 rounds after A to B']):
    for method, label in [('local16', 'Local: 16 records'), ('local64', 'Local: 64 records')]:
        rows = [next(r for r in data if float(r['lam']) == lam and r['scene'] == scene
                     and r['period'] == period and r['threshold'] == 'True' and r['method'] == method) for lam in lambdas]
        ax.errorbar(lambdas, [float(r['mse']) for r in rows], yerr=[1.96 * float(r['mc_se']) for r in rows],
                    marker='o', lw=2, capsize=3, color=colors[method], label=label)
    best = [min(float(r['mse']) for r in data if float(r['lam']) == lam and r['scene'] == scene
                and r['period'] == period and r['threshold'] == 'True' and r['method'].startswith('server')) for lam in lambdas]
    ax.plot(lambdas, best, marker='s', lw=2, ls='--', color=colors['server'], label='Best fixed server window*')
    ax.axhline(.36, color='#444444', lw=1, ls=':', label='Always zero')
    ax.set(title=title, xlabel='Temporal correlation parameter', ylabel='Expected total squared error', ylim=(0, .39))
    ax.grid(alpha=.15)
for b in [16, 64]:
    rows = [next(r for r in effective if int(r['b']) == b and float(r['lam']) == lam) for lam in lambdas]
    axes[2].plot(lambdas, [float(r['effective_samples']) for r in rows], marker='o', lw=2,
                 color=colors['local' + str(b)], label=str(b) + ' raw records')
axes[2].set(title='Mean-variance effective sample size', xlabel='Temporal correlation parameter', ylabel='Equivalent independent sample count')
axes[2].grid(alpha=.15)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, fontsize=9, ncol=4, loc='lower center', bbox_to_anchor=(.5, .055), frameon=False)
axes[2].legend(fontsize=9)
fig.suptitle('Local history helps only in part of the temporal-dependence range', fontsize=16, y=.99)
fig.text(.5, .012, '*Post-hoc minimum over h=8/16/32/64/128; not an adaptive algorithm. Bars: Monte Carlo integration error.', ha='center', fontsize=9)
fig.tight_layout(rect=(0, .13, 1, .94))
fig.savefig(OUT / 'dependence-results.png', dpi=170)
fig.savefig(OUT / 'dependence-results.pdf')
