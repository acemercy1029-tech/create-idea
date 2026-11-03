from pathlib import Path
import csv
import json
import numpy as np
import run_grid as g

def read(name):
    with (g.OUT/name).open(encoding='utf-8') as f:
        return list(csv.DictReader(f))

choices=read('frozen_choices.csv')
confirm=read('confirmation_summary.csv')
selected=[]
for choice in choices:
    period='stable' if choice['scene']=='constant' else 'first96'
    row=next(r for r in confirm if all(r[k]==choice[k] for k in ['lam','scene','threshold','method']) and r['period']==period)
    selected.append(dict(lam=choice['lam'],scene=choice['scene'],threshold=choice['threshold'],family=choice['family'],method=choice['method'],pilot_mse=choice['pilot_mse'],confirmation_mse=row['mse'],mc_se=row['mc_se'],low=row['low'],high=row['high']))
g.save('selected_confirmation.csv',selected)

global_choices=read('global_frozen_choices.csv')
global_rows=read('global_confirmation.csv')
overall=[]
for choice in global_choices:
    for period in ['first8','first32','first96','last8']:
        rows=[r for r in global_rows if r['threshold']==choice['threshold'] and r['method']==choice['method'] and r['period']==period]
        assert len(rows)==4
        mean=np.mean([float(r['mse']) for r in rows])
        se=np.sqrt(sum(float(r['mc_se'])**2 for r in rows))/4
        overall.append(dict(threshold=choice['threshold'],family=choice['family'],method=choice['method'],period=period,equal_weight_mse=mean,mc_se=se,low=mean-1.96*se,high=mean+1.96*se))
g.save('global_equal_weight.csv',overall)

# Export consistency and resource/accounting checks. No new selection on holdout.
pilot=read('pilot_summary.csv')
assert len(pilot)==2240 and len(choices)==48 and len(selected)==48
assert len(read('kernel_checks.csv'))==74
resources=read('resources.csv')
assert len(resources)==55
assert all(float(r['mean_uplink_bits'])<=1024+1e-10 for r in resources)
for row in pilot+confirm+global_rows:
    assert float(row['low'])<=float(row['mse'])<=float(row['high'])
    if row['method']=='zero':
        assert abs(float(row['mse'])-.36)<1e-12
for choice in choices:
    expected=min((r for r in pilot if r['lam']==choice['lam'] and r['scene']==choice['scene'] and r['threshold']==choice['threshold'] and r['family']==choice['family'] and r['period']==('stable' if choice['scene']=='constant' else 'first96')),key=lambda r:float(r['mse']))
    assert expected['method']==choice['method']
result=dict(pilot_rows=len(pilot),frozen_choices=len(choices),confirmation_rows=len(confirm),global_confirmation_rows=len(global_rows),kernel_checks=74,max_mean_bits=max(float(r['mean_uplink_bits']) for r in resources),checks='frozen selection matches pilot; resources within cap; zero baseline and output intervals valid')
(g.OUT/'output_checks.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
for scene in ['constant','migration']:
    print(scene,'threshold output:')
    for r in selected:
        if r['scene']==scene and r['threshold']=='True':
            print(r['lam'],r['family'],r['method'],round(float(r['confirmation_mse']),6))
print('Global fixed, threshold output:')
for r in overall:
    if r['threshold']=='True' and r['period']=='first96':
        print(r)

# Scientific plot of the independent confirmations.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(12,4.8),sharey=True)
colors={'local':'#197a72','server':'#6866ad','mix':'#c87918'}
labels={'local':'Local history','server':'Server history','mix':'Fixed equal-weight mixture'}
for family in ['local','server','mix']:
    sub=[r for r in selected if r['scene']=='migration' and r['threshold']=='True' and r['family']==family]
    sub.sort(key=lambda r:float(r['lam']))
    axes[0].errorbar([float(r['lam']) for r in sub],[float(r['confirmation_mse']) for r in sub],yerr=[1.96*float(r['mc_se']) for r in sub],color=colors[family],label=labels[family],marker='o',capsize=3,lw=2)
    name=next(r['method'] for r in global_choices if r['threshold']=='True' and r['family']==family)
    sub=[r for r in global_rows if r['threshold']=='True' and r['method']==name and r['period']=='first96']
    sub.sort(key=lambda r:float(r['lam']))
    axes[1].errorbar([float(r['lam']) for r in sub],[float(r['mse']) for r in sub],yerr=[1.96*float(r['mc_se']) for r in sub],color=colors[family],label=labels[family],marker='o',capsize=3,lw=2)
for ax in axes:
    ax.axhline(.36,color='#555555',ls=':',lw=1,label='Always zero')
    ax.set(xlabel='Temporal correlation parameter',ylim=(0,.39))
    ax.grid(alpha=.15)
axes[0].set(title='Configuration chosen for each environment*',ylabel='Mean total squared error, first 96 rounds')
axes[1].set(title='One frozen configuration for all environments')
handles,labels=axes[1].get_legend_handles_labels()
fig.legend(handles,labels,ncol=4,loc='lower center',bbox_to_anchor=(.5,.045),frameon=False,fontsize=9)
fig.suptitle('Independent validation of allocation tuning and ordinary mixtures',fontsize=15)
fig.text(.5,.012,'*Diagnostic reference, not an online policy. All choices frozen on pilot; error bars are Monte Carlo precision.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.13,1,.94))
fig.savefig(g.OUT/'allocation-results.png',dpi=170)
fig.savefig(g.OUT/'allocation-results.pdf')
