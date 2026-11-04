import csv
import json
import numpy as np
import run_calibration as r

def read(name):
    with (r.OUT/name).open(encoding='utf-8') as f:return list(csv.DictReader(f))

lookup=read('frozen_lookup.csv')
maps={(int(z['calibration_users']),z['threshold']=='True',float(z['training_lambda'])):z['policy'] for z in lookup}
baseline=json.loads((r.OUT/'frozen_baselines.json').read_text(encoding='utf-8'))
fixed=read('fixed_validation.csv')
fixed_index={(float(z['lam']),z['scene'],int(z['calibration_users']),z['threshold']=='True',z['policy'],z['period']):float(z['mse']) for z in fixed}
scenes=read('validation_scenes.csv')
max_error=0.
for z in scenes:
    lam,scene,th,method,period=float(z['lam']),z['scene'],z['threshold']=='True',z['method'],z['period']
    if method.startswith('private_cal'):
        c=int(method[len('private_cal'):]);p=r.calibration_probs(lam,scene,c)
        expected=sum(p[i]*fixed_index[(lam,scene,c,th,maps[(c,th,l)],period)] for i,l in enumerate(r.GRID))
    elif method=='fixed_full_population':expected=fixed_index[(lam,scene,0,th,baseline[str(th)],period)]
    elif method=='known_grid_no_cost':expected=fixed_index[(lam,scene,0,th,maps[(0,th,r.GRID[r.nearest(lam)])],period)]
    elif method.startswith('known_grid_cost'):
        c=int(method[len('known_grid_cost'):]);expected=fixed_index[(lam,scene,c,th,maps[(c,th,r.GRID[r.nearest(lam)])],period)]
    else:continue
    max_error=max(max_error,abs(expected-float(z['mse'])))
    assert abs(expected-float(z['mse']))<1e-12
summary=read('validation_summary.csv')
assert len(summary)==98
for z in summary:
    sub=[q for q in scenes if q['lam']==z['lam'] and q['threshold']==z['threshold'] and q['method']==z['method'] and q['period']=='full']
    assert len(sub)==4
    assert abs(np.mean([float(q['mse']) for q in sub])-float(z['mse']))<1e-12
    if z['method']=='zero':assert abs(float(z['mse'])-.18225)<1e-12
train=read('training_summary.csv')
for z in lookup:
    totals={}
    for policy in r.POLICIES:
        vals=[float(q['mse']) for q in train if q['lam']==z['training_lambda'] and q['calibration_users']==z['calibration_users'] and q['threshold']==z['threshold'] and q['policy']==policy and q['period']=='full']
        assert len(vals)==4;totals[policy]=np.mean(vals)
    assert min(totals,key=totals.get)==z['policy']

resource=[]
for c in r.COSTS:
    for policy,(kind,size,ns) in r.POLICIES.items():
        stride=r.d.W+size-1 if kind=='local' else r.d.W
        total,peak=0,0
        for t in range(r.d.T):
            available=t>=(2 if c else 0) and (kind=='server' or t>=size-1)
            bits=c if c and t==1 else 0
            if available:bits+=r.nphase(ns,stride,t)+r.nphase(r.d.N-c-ns,r.d.W,t)
            total+=bits;peak=max(peak,bits)
        resource.append(dict(calibration_users=c,policy=policy,screen_users=ns,publication_users=r.d.N-c-ns,production_first_round=max(2 if c else 0,size-1 if kind=='local' else 0),total_uplink_bits_320=total,maximum_per_round_bits=peak))
        assert peak<=1024
r.save('resource_accounting.csv',resource)
checks=dict(calibration_risk_reconstruction_max_error=max_error,summary_rows=len(summary),scene_rows=len(scenes),fixed_validation_rows=len(fixed),new_kernel_checks=len(read('kernel_checks.csv')),frozen_lookup_rows=len(lookup),checks='private-calibration mixture reconstructed from fixed-policy risks; four-scene totals; frozen training lookup; zero baseline; report resources')
(r.OUT/'output_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
print(json.dumps(checks,indent=2))
print('Frozen global baseline',baseline)
print('Lookup threshold=True')
for z in lookup:
    if z['threshold']=='True':print(z)
print('Aggregate threshold=True')
for z in read('aggregate.csv'):
    if z['threshold']=='True':print(z)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(12.4,4.8),sharey=True)
methods=[('fixed_full_population','Fixed, all users','#6866ad'),('private_cal256','Private calibration: 256 users','#197a72'),('private_cal1024','Private calibration: 1024 users','#c87918'),('known_grid_no_cost','Known grid, no calibration cost*','#777777')]
for ax,scope,lams in zip(axes,['Training-grid environments','Unseen environments'],[r.GRID,[.15,.45,.7]]):
    for name,label,color in methods:
        rows=[next(z for z in summary if z['threshold']=='True' and z['method']==name and float(z['lam'])==lam) for lam in lams]
        ax.errorbar(lams,[float(z['mse']) for z in rows],yerr=[1.96*float(z['mc_se']) for z in rows],label=label,color=color,marker='o',ls='--' if name.startswith('known') else '-',capsize=3,lw=2)
    ax.set(title=scope,xlabel='Temporal correlation parameter')
    ax.grid(alpha=.15)
axes[0].set_ylabel('Mean total squared error over all 320 rounds')
fig.suptitle('One-time private calibration, with users and startup cost included',fontsize=15)
handles,labels=axes[0].get_legend_handles_labels()
fig.legend(handles,labels,ncol=2,loc='lower center',bbox_to_anchor=(.5,.045),frameon=False,fontsize=9)
fig.text(.5,.012,'*Diagnostic reference using the nearest training grid. Four stream types equally weighted; MC integration error bars.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.18,1,.94))
fig.savefig(r.OUT/'calibration-results.png',dpi=170)
fig.savefig(r.OUT/'calibration-results.pdf')
