"""Post-hoc pilot-only constant-weight diagnostic, not independent validation."""
import csv
import numpy as np
import run_grid as g

with (g.OUT/'pilot_summary.csv').open(encoding='utf-8') as f:
    rows=list(csv.DictReader(f))
index={(float(r['lam']),r['scene'],r['period'],r['threshold']=='True',r['method']):float(r['mse']) for r in rows}

def risk256_from_selector(lam,th,kind,size):
    # Pilot a1 selector has 5120 people, exactly the mixed subsystem selector.
    original=index[(lam,'migration','first96',th,kind+str(size)+'_a1')]
    correct768=float(g.single_risk(np.array([True]),768,th)[0])
    wrong768=float(g.single_risk(np.array([False]),768,th)[0])
    pi=(original-wrong768)/(correct768-wrong768)
    assert -1e-10<=pi<=1+1e-10
    return pi*float(g.single_risk(np.array([True]),256,th)[0])+(1-pi)*float(g.single_risk(np.array([False]),256,th)[0])

def minimum(rl,rs,cross):
    curvature=rl+rs-2*cross
    assert curvature>=-1e-10
    weight=np.clip((rs-cross)/curvature,0,1) if curvature>1e-12 else 0.
    value=weight**2*rl+(1-weight)**2*rs+2*weight*(1-weight)*cross
    return float(weight),float(value)

allrows=[]
for th in [False,True]:
    for b in g.SIZES:
        for h in g.SIZES:
            moments=[]
            for lam in g.LAMBDAS:
                rl=risk256_from_selector(lam,th,'local',b)
                rs=risk256_from_selector(lam,th,'server',h)
                half=index[(lam,'migration','first96',th,'mix_b'+str(b)+'_h'+str(h))]
                cross=2*half-.5*(rl+rs)
                v,risk=minimum(rl,rs,cross)
                allrows.append(dict(scope=str(lam),threshold=th,b=b,h=h,weight_on_local=v,pilot_optimized_mse=risk,local_subsystem_mse=rl,server_subsystem_mse=rs,cross_error=cross))
                moments.append([rl,rs,cross])
            rl,rs,cross=np.mean(moments,axis=0)
            v,risk=minimum(rl,rs,cross)
            allrows.append(dict(scope='equal_weight_four_lambda',threshold=th,b=b,h=h,weight_on_local=v,pilot_optimized_mse=risk,local_subsystem_mse=rl,server_subsystem_mse=rs,cross_error=cross))
g.save('pilot_weight_diagnostic_all.csv',allrows)
best=[]
for th in [False,True]:
    for scope in [str(lam) for lam in g.LAMBDAS]+['equal_weight_four_lambda']:
        best.append(min((r for r in allrows if r['threshold']==th and r['scope']==scope),key=lambda r:r['pilot_optimized_mse']))
g.save('pilot_weight_diagnostic_best.csv',best)
for r in best:
    print(r)
