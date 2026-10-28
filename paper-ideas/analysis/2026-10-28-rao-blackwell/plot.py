"""Plot saved research results; reruns no simulation."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
with (OUT/'summary.csv').open() as f:
    rows = {(r['scenario'],r['method']):r for r in csv.DictReader(f)}
with (OUT/'stationary_intervals.csv').open() as f:
    stable = {r['method']:r for r in csv.DictReader(f)}
methods = ['direct_h4','pooled_h4','rb_h4','rotating_h4']
labels = ['Original\nrelease','Pooled\nreports','Rao–Blackwell','Rotation']
colors = ['#a0a7b4','#dc9b42','#4677a8','#3d9b79']
fig,axes = plt.subplots(1,3,figsize=(12.6,4.6))
for ax,scenario,title in zip(axes,['stationary','d0.2_L8','d0.6_L8'],
        ['Stationary: worst-pair mean MSE','Weak 8-round event: active MSE','Strong 8-round event: active MSE']):
    if scenario=='stationary':
        values = [float(stable[m]['max_mean']) for m in methods]
        low = [float(stable[m]['simultaneous_low']) for m in methods]
        high = [float(stable[m]['simultaneous_high']) for m in methods]
        ax.axhline(.01,color='#a44444',ls='--',lw=1,zorder=1)
    else:
        values = [float(rows[(scenario,m)]['active_mse']) for m in methods]
        low = [float(rows[(scenario,m)]['active_low']) for m in methods]
        high = [float(rows[(scenario,m)]['active_high']) for m in methods]
    ax.bar(range(4),values,color=colors,zorder=2)
    ax.errorbar(range(4),values,yerr=[np.array(values)-low,np.array(high)-values],
        fmt='none',color='#222222',capsize=3,zorder=3)
    for i,v in enumerate(values):
        ax.text(i,high[i]+max(high)*.025,f'{v:.5f}',ha='center',fontsize=9)
    ax.set_ylim(0,max(high)*1.22)
    ax.set_xticks(range(4),labels,fontsize=9)
    ax.set_title(title,fontsize=11,pad=12)
    ax.grid(axis='y',alpha=.2,zorder=0)
    ax.spines[['top','right']].set_visible(False)
axes[0].set_ylabel('MSE (lower is better)')
fig.suptitle('Removing selection bias does not guarantee lower release MSE',fontsize=14,y=.99)
fig.text(.5,.035,'All outputs: causal h4, clipped to [-1,1]. 1,000 repeats; ε=1, w=20.\n'
    'First 3 methods share reports; rotation has the same cap but uses more reports. Bars: 95% MC intervals; stationary intervals simultaneous across 15 pairs.',
    ha='center',fontsize=9)
fig.tight_layout(rect=(0,.13,1,.95))
fig.savefig(OUT/'rao-blackwell-mse.png',dpi=180)
fig.savefig(OUT/'rao-blackwell-mse.svg')
svg = OUT/'rao-blackwell-mse.svg'
svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
