"""Publication-style diagnostic figure from the independently seeded check."""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

P = Path(__file__).resolve().parent
rows = list(csv.DictReader((P / 'stronger-check' / 'time_curves.csv').open()))
methods = [('local_b16_g4', 'Local history 16 + server 4', '#087e8b'),
           ('local_b64_g4', 'Local history 64 + server 4', '#d97924'),
           ('all_rotation_hmm', 'All-user rotation + HMM', '#7047a3'),
           ('server_h32', 'Server screening, window 32', '#7a8793')]
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
for ax, length in zip(axes, [32, 96]):
    ax.axvspan(96, 96 + length, color='#dde8ee', alpha=.65, label='Association active')
    for method, label, color in methods:
        selected = [x for x in rows if x['scene'] == 'amp06_len%d' % length and x['method'] == method and int(x['time']) >= 64]
        ax.plot([int(x['time']) for x in selected], [float(x['l2_mse']) for x in selected], label=label, color=color, lw=1.7)
    ax.set(title='Amplitude 0.6, duration %d rounds' % length, xlabel='Time', xlim=(64, 255), ylim=(-.01, .405))
    ax.grid(axis='y', alpha=.2)
axes[0].set_ylabel('Mean squared L2 error across 256 pairs')
handles, labels = axes[1].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(.5, .045), fontsize=9)
fig.suptitle('Causal stream estimation under w=20, epsilon=1', y=.99, fontsize=13)
fig.text(.5, .015, '500 repeated selector simulations; independent publication risk integrated exactly. No threshold post-processing.',
         ha='center', fontsize=8, color='#44515b')
fig.tight_layout(rect=(0, .22, 1, .96))
fig.savefig(P / 'stronger-comparison.png', dpi=180, bbox_inches='tight')
fig.savefig(P / 'stronger-comparison.pdf', bbox_inches='tight')
print(P / 'stronger-comparison.png')
