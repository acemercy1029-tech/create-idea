"""Static research figure; all 25 methods remain in summary.csv."""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
with (OUT/'summary.csv').open(encoding='utf-8') as f:
    rows = list(csv.DictReader(f))
lookup = {(r['scenario'],r['method']):r for r in rows}
methods = ['rotating_h4','lpd_a50_h4','lpd_a50_group_feedback',
           'lpd_a50_reuse_h4_terminal','lpd_a50_reuse_h4_feedback','lpd_a50_reuse_group_feedback']
labels = ['Rotation + h4','LPD + h4','LPD + group\nfeedback',
          'Reuse + h4\noutput only','Reuse + h4\nfeedback','Reuse + group\nfeedback']
colors = ['#315a8b','#9aa5b1','#e5a34b','#459e9a','#247b78','#af6b9b']
fig,axes = plt.subplots(1,2,figsize=(12,5.4))
for ax,scenario,metric,title in [(axes[0],'stationary','max_pair_timeavg_mse','Stationary: worst-pair time-average MSE'),
                                (axes[1],'d0.6_L8','active_mse','Strong 8-round event: active MSE')]:
    values = [float(lookup[scenario,m][metric]) for m in methods]
    bars = ax.bar(np.arange(len(methods)),values,color=colors,width=.72)
    ax.set_xticks(np.arange(len(methods)))
    ax.set_xticklabels(labels,rotation=33,ha='right',fontsize=9)
    ax.set_title(title,fontsize=12,pad=14)
    ax.set_ylim(0,max(values)*1.23)
    ax.set_ylabel('MSE (lower is better)')
    ax.grid(axis='y',alpha=.18)
    ax.set_axisbelow(True)
    ax.spines[['top','right']].set_visible(False)
    for bar,value in zip(bars,values):
        ax.text(bar.get_x()+bar.get_width()/2,value+max(values)*.025,f'{value:.5f}',ha='center',fontsize=9)
axes[0].axhline(.01,color='#b73e47',linestyle='--',linewidth=1.2,label='Working target: 0.01')
axes[0].legend(frameon=False,fontsize=9,loc='upper right')
fig.suptitle('Feedback and report reuse help, but do not remove the observed trade-off',fontsize=14,y=.99)
fig.text(.5,.015,'1000 repetitions; epsilon=1, w=20. LPD variants use 50% of users for monitoring.\n'
         'Scalar adaptations, not full-paper reproductions. The two h4 reuse results straddle 0.01 within Monte Carlo uncertainty.',
         ha='center',fontsize=9,color='#444444')
fig.tight_layout(rect=(0,.10,1,.95))
fig.savefig(OUT/'feedback-reuse.png',dpi=180)
fig.savefig(OUT/'feedback-reuse.svg')
plt.close(fig)
svg = OUT/'feedback-reuse.svg'
svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
