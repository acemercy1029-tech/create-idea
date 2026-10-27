"""Post-result boundary uncertainty and exact one-step selection diagnostics."""
import importlib.util
from pathlib import Path
import itertools
import numpy as np
from scipy.stats import norm, binom

OUT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('round11',OUT/'run.py')
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def stationary_uncertainty():
    rows, bounds = [], []
    rho = study.r8.truth()
    z = norm.ppf(1-.05/(2*study.M))
    for seed in [study.SEED,2026102712]:
        def record(name,est):
            losses = ((est-rho[None])**2)[:,study.WARM:].mean(axis=1)
            mean = losses.mean(axis=0)
            se = losses.std(axis=0,ddof=1)/np.sqrt(study.REPS)
            for j,(a,b) in enumerate(study.base.PAIRS):
                rows.append(dict(seed=seed,method=name,pair=f'{a}-{b}',mse=mean[j],se=se[j],
                    point_low=mean[j]-1.96*se[j],point_high=mean[j]+1.96*se[j],
                    simultaneous_low=mean[j]-z*se[j],simultaneous_high=mean[j]+z*se[j]))
            bounds.append(dict(seed=seed,method=name,max_pair_mean=float(mean.max()),
                max_mean_simultaneous_low=float((mean-z*se).max()),
                max_mean_simultaneous_high=float((mean+z*se).max())))
        rng = np.random.default_rng(np.random.SeedSequence([seed,0,0]))
        y = study.r10.draw_rr(study.PAIR_USERS//study.W,rho,rng,(study.REPS,study.T,study.M))
        record('rotating_h4',np.clip(study.base.rolling(y,4),-1,1))
        del y
        for fi,feedback in [(0,'raw'),(2,'reuse_h4_feedback')]:
            rng = np.random.default_rng(np.random.SeedSequence([seed,0,14+fi]))
            output,_ = study.simulate(rho,.5,feedback,rng)
            key = 'reuse_h4_terminal' if feedback=='raw' else feedback
            record('lpd_a50_'+key,output[key])
            del output
        print('Stationary uncertainty seed',seed,flush=True)
    study.save('stationary_pair_intervals.csv',rows)
    study.save('stationary_max_intervals.csv',bounds)
    for row in bounds: print(row)


def exact_selection():
    k, rm, rp = study.K, 70, 140
    count = np.arange(rm+1)
    y = (2*count/rm-1)/k
    vhat = (k**-2-y*y)/(rm-1)
    rows = []
    for rho,history in itertools.product([0.,.2,.6],[0.,.3,.6]):
        p = binom.pmf(count,rm,(1+k*rho)/2)
        selected = (y-history)**2-vhat > k**-2/rp
        wmon = rm/(rm+selected*rp)
        conditional_mean = wmon*y+(1-wmon)*rho
        conditional_var = (1-wmon)**2*(k**-2-rho**2)/rp
        mean = float(p @ conditional_mean)
        mse = float(p @ ((conditional_mean-rho)**2+conditional_var))
        rate = float(p @ selected)
        vmon = (k**-2-rho**2)/rm
        # Independent selection with the same rate has the same expected report count.
        independent_mse = (1-rate)*vmon+rate*(k**-2-rho**2)/(rm+rp)
        # Independently sum over both binomial report counts (71 x 141 cells).
        pub_count = np.arange(rp+1)
        pub_y = (2*pub_count/rp-1)/k
        pub_p = binom.pmf(pub_count,rp,(1+k*rho)/2)
        joint_p = p[:,None]*pub_p[None,:]
        all_pooled = wmon[:,None]*y[:,None]+(1-wmon[:,None])*pub_y[None,:]
        joint_mean = float((joint_p*all_pooled).sum())
        joint_mse = float((joint_p*(all_pooled-rho)**2).sum())
        assert abs(joint_mean-mean)<1e-12 and abs(joint_mse-mse)<1e-12
        rows.append(dict(rho=rho,history=history,monitor_reports=rm,candidate_reports=rp,
            update_probability=rate,expected_total_reports=rm+rate*rp,
            pooled_mean=mean,pooled_bias=mean-rho,pooled_mse=mse,
            pooled_variance=mse-(mean-rho)**2,
            mean_nominal_proxy=float(p @ (k**-2/(rm+selected*rp))),
            independent_selection_bias=0.,independent_selection_mse=independent_mse,
            monitor_only_mse=vmon,joint_enumeration_error=max(abs(joint_mean-mean),abs(joint_mse-mse))))
        assert np.isclose(p.sum(),1)
        assert mse >= (mean-rho)**2-1e-12
    study.save('exact_selection.csv',rows)
    for row in rows: print(row)


if __name__ == '__main__':
    stationary_uncertainty()
    exact_selection()
