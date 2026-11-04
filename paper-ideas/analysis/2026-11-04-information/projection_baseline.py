"""Ordinary sign-of-public-projection baseline, avoiding local top1 selection."""
from pathlib import Path
import sys, json, time
import numpy as np
import run_information as a
d,g=a.d,a.g
OUT=Path(__file__).resolve().parent
SEED=2026110420

def means(b,lam,t):
    times=np.arange(t-b+1,t+1)
    pa=d.raw_counts(tuple(np.where((times>=96)&(times<160),.8,.5)),lam)
    pb=d.raw_counts(tuple(np.where((times>=160)&(times<168),.8,.5)),lam)
    p0=d.raw_counts((.5,)*b,lam)
    null=a.power(p0,a.M-2)
    values={}
    for sa in [1,-1]:
        for sb in [1,-1]:
            pmf=a.conv(null,a.conv(pa if sa==1 else pa[::-1],pb if sb==1 else pb[::-1]))
            values[sa,sb]=float(np.sign(2*np.arange(len(pmf))-b*a.M)@pmf)
    a.check('projection_odd_symmetry',max(abs(values[1,1]+values[-1,-1]),abs(values[1,-1]+values[-1,1])),1e-10,(b,lam,t))
    ca=(values[1,1]+values[1,-1])/2
    cb=(values[1,1]-values[1,-1])/2
    mu=ca*a.H[:,a.A]+cb*a.H[:,a.B]
    a.check('projection_channel_range',max(0.,float(np.max(abs(mu)))-1),1e-10,(b,lam,t))
    return mu,ca,cb

def validate():
    # Direct count-vector Monte Carlo, independent of projection convolution.
    rng=np.random.default_rng(SEED+99)
    for b,lam,t in [(8,0.,167),(8,.6,164),(64,.6,167)]:
        times=np.arange(t-b+1,t+1)
        pa=d.raw_counts(tuple(np.where((times>=96)&(times<160),.8,.5)),lam)
        pb=d.raw_counts(tuple(np.where((times>=160)&(times<168),.8,.5)),lam)
        p0=d.raw_counts((.5,)*b,lam)
        mu,_,_=means(b,lam,t)
        row=int(np.flatnonzero((a.H[:,a.A]==1)&(a.H[:,a.B]==-1))[0])
        vals=[]
        for batch in range(20):
            counts=rng.choice(b+1,size=(2500,a.M),p=p0)
            counts[:,a.A]=rng.choice(b+1,2500,p=pa)
            counts[:,a.B]=rng.choice(b+1,2500,p=pb)
            vals.append(np.sign(((2*counts-b)*a.H[row]).sum(axis=1)))
        vals=np.concatenate(vals)
        a.check('projection_direct_counts',abs(float(vals.mean())-mu[row]),6*float(vals.std()/np.sqrt(len(vals))),(b,lam,t))

def run():
    start=time.monotonic();validate();rows=[];curves=[];coeff=[]
    for li,lam in enumerate([0.,.6]):
        for b in [4,8,16,64]:
            rng=np.random.default_rng(np.random.SeedSequence([SEED,li,b]))
            transcript=[];selected={h:np.zeros((a.REPS,8),bool) for h in [1,4,8]}
            for t in range(153,168):
                mu,ca,cb=means(b,lam,t)
                pp=np.r_[1+d.KAPPA*mu,1-d.KAPPA*mu]/(2*a.M)
                counts=rng.multinomial(a.nphase(b,t),pp/pp.sum(),size=a.REPS)
                transcript.append(counts[:,:a.M]-counts[:,a.M:])
                coeff.append(dict(lam=lam,b=b,t=t,old_coefficient=ca,new_coefficient=cb))
                if t<160:continue
                for h in selected:
                    score=d.iid.fwht(sum(transcript[-h:]))
                    selected[h][:,t-160]=d.iid.argmax_random(score,rng,absolute=True)==a.B
            for h,values in selected.items():
                method='projection_g'+str(h);setting=dict(lam=lam,b=b)
                a.summary(rows,setting,method,values,'support_success')
                for th in [False,True]:
                    a.summary(rows,setting,method,g.single_risk(values,512,th),'mse_threshold' if th else 'mse_raw')
                for r in range(8):
                    p=float(values[:,r].mean())
                    curves.append(dict(**setting,t=160+r,method=method,success=p,mc_se=np.sqrt(p*(1-p)/a.REPS)))
            print('Projection lambda={} b={} {:.1f}s'.format(lam,b,time.monotonic()-start),flush=True)
    a.save('projection_results.csv',rows);a.save('projection_curves.csv',curves)
    a.save('projection_coefficients.csv',coeff);a.save('projection_checks.csv',a.CHECKS)
    (OUT/'projection_metadata.json').write_text(json.dumps(dict(seed=SEED,repeats=a.REPS,checks=len(a.CHECKS),elapsed_seconds=time.monotonic()-start),indent=2),encoding='utf-8')

if __name__=='__main__':run()
