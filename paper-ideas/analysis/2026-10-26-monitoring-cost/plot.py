"""Render one illustrative tradeoff; the CSV contains every tested setting."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
rows=list(csv.DictReader((ROOT/'summary.csv').open(encoding='utf-8')))
methods=['rotating_h4','lpd_a50_corrected','lpd_a50_corrected_h20']
labels=['Full rotation\n4-round mean','50% monitor\nraw release','50% monitor\n20-round mean*']
colors=['#267f9f','#dc8440','#58926c']
fig,axes=plt.subplots(1,2,figsize=(9.5,4.4))
for ax,scenario,metric,title in zip(axes,['stationary','d0.6_L8'],
                                  ['max_pair_timeavg_mse','active_mse'],
                                  ['Stationary: worst pair over time','8-round pulse: active-period error']):
    values=[float(next(r[metric] for r in rows if r['scenario']==scenario and r['method']==m)) for m in methods]
    ax.bar(range(3),values,color=colors,width=.6)
    for i,v in enumerate(values):
        ax.text(i,v+max(values)*.025,f'{v:.5f}',ha='center',fontsize=10)
    ax.set_xticks(range(3),labels,fontsize=9)
    ax.set_ylim(0,max(values)*1.22)
    ax.set_ylabel('MSE')
    ax.set_title(title,fontsize=11)
    ax.grid(axis='y',alpha=.2)
    ax.set_axisbelow(True)
axes[0].axhline(.01,color='#ac3e45',ls='--',lw=1.2,label='Working target 0.01')
axes[0].legend(fontsize=8)
fig.suptitle('Same privacy caps: epsilon=1, w=20; pulse amplitude=0.6',fontsize=12)
fig.text(.5,.015,'*20-round mean added as a secondary check after the first results; not a tuned optimum.',ha='center',fontsize=8)
fig.tight_layout(rect=[0,.06,1,.94])
fig.savefig(ROOT/'tradeoff.png',dpi=170)
fig.savefig(ROOT/'tradeoff.svg')
p=ROOT/'tradeoff.svg'
p.write_text('\n'.join(line.rstrip() for line in p.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
