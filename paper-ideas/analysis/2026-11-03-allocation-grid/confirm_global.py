"""Freeze one configuration per family using pilot only, then fresh simulation."""
import csv
import time
import numpy as np
import run_grid as g

with (g.OUT/'pilot_summary.csv').open(encoding='utf-8') as f:
    pilot=list(csv.DictReader(f))
assert len(pilot)==2240, 'Full pilot must finish before global selection.'
choices=[]
for th in [False,True]:
    for family in ['local','server','mix']:
        candidates={}
        for name,config in g.CONFIGS.items():
            if config[0]!=family:
                continue
            values=[float(r['mse']) for r in pilot if r['scene']=='migration' and r['period']=='first96' and r['threshold']==str(th) and r['method']==name]
            assert len(values)==4
            candidates[name]=np.mean(values)
        best=min(candidates,key=candidates.get)
        choices.append(dict(threshold=th,family=family,method=best,pilot_equal_weight_mse=candidates[best]))
g.save('global_frozen_choices.csv',choices)
names=sorted(set(g.ANCHORS+[r['method'] for r in choices]))
rows,diffs=[],[]
started=time.monotonic()
for li,lam in enumerate(g.LAMBDAS):
    result,avg=g.run(lam,li,'migration',names,1500,2)
    rows.extend(result)
    g.save('global_confirmation.csv',rows)
    for th in [False,True]:
        picks={r['family']:r['method'] for r in choices if r['threshold']==th}
        pairs=[(picks['mix'],picks['server']),(picks['mix'],picks['local']),(picks['local'],picks['server'])]
        for family in picks:
            pairs.extend((picks[family],anchor) for anchor in g.ANCHORS)
        for left,right in pairs:
            for period in ['first8','first32','first96','last8']:
                x=avg[(th,left,period)]-avg[(th,right,period)]
                mu,se=float(x.mean()),float(x.std(ddof=1)/np.sqrt(len(x)))
                diffs.append(dict(lam=lam,period=period,threshold=th,method=left,reference=right,difference=mu,mc_se=se,low=mu-1.96*se,high=mu+1.96*se))
    g.save('global_differences.csv',diffs)
    print('Global fixed confirmation',lam,'elapsed',round(time.monotonic()-started,1),flush=True)
g.save('global_metadata.csv',[dict(seed=2026110319,repeats=1500,criterion='pilot equal-weight mean over four lambda values; frozen',elapsed_seconds=time.monotonic()-started)])
