"""Render the two numerical comparisons used in the discussion."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

base = Path(__file__).resolve().parent
def read(name):
    with (base/name).open() as handle:
        return list(csv.DictReader(handle))

confirm = read('confirmation.csv')
budget = read('raw_bound_all_histories.csv')
extra = read('confirmation32.csv')
fig, (duration, epsilon) = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
for cut, family, color, label in [(0, 'top', '#2670a6', 'Local top1'), (.75, 'threshold', '#bc6534', 'All above threshold')]:
    rows = [r for r in confirm if float(r['cut'])==cut and r['b']=='16' and float(r['lam'])==0]
    updated = next(r for r in extra if r['family']==family)
    points = [updated if r['duration']=='32' else r for r in rows]
    duration.plot([int(r['duration']) for r in points],[float(r['risk']) for r in points],
                  'o-', color=color, label=label)
duration.axhline(.18,color='black',ls='--',label='Half-risk target')
duration.set(xlabel='Event duration (rounds)',ylabel='Event-average total MSE',
             title='Independent data, matched budgets',ylim=(.075,.38))
duration.legend(fontsize=8)
rows = [r for r in budget if r['duration']=='8']
epsilon.plot([float(r['epsilon']) for r in rows],[float(r['risk_lower']) for r in rows],'o-',color='#246652')
epsilon.axhline(.18,color='black',ls='--')
epsilon.set(xlabel='Privacy budget per window',ylabel='Lower bound on event-average MSE',
            title='8 rounds, any fixed block length')
for ax in (duration,epsilon):
    ax.grid(alpha=.2)
    ax.spines[['top','right']].set_visible(False)
fig.savefig(base/'duration-results.png',dpi=160)
