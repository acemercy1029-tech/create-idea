"""Exact diagnostic risks for PUBLIC FIXED schedules, not an online proposal.

True scores are offline diagnostics only. See PROTOCOL.md.
"""
from pathlib import Path
import csv
import json
import sys
import numpy as np
from scipy.stats import binom

OUT = Path(__file__).resolve().parent
N,T,W,B,REPS,SEED = 2800,20,20,.5,100000,2026102813


def save(name,rows):
    with (OUT/name).open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def variance(a,rho):
    return (np.tanh(a/2)**-2-rho**2)/N


def exact_risk(rho,schedule):
    # Crucially, schedule is public and fixed, independent of all report values.
    mean,var=0.,0.
    cumulative,spent=0.,[]
    for t in range(T):
        a=schedule.get(t,0.)
        if a>0:
            mean,var=rho[t],variance(a,rho[t])
        cumulative+=(mean-rho[t])**2+var
        spent.append(a)
        assert sum(spent[max(0,t-W+1):t+1]) <= B+1e-12
    return float(cumulative)


def simulate(rho,schedule,seed):
    rng=np.random.default_rng(np.random.SeedSequence([SEED,seed]))
    output=np.zeros(REPS)
    loss=np.zeros(REPS)
    clipped=np.zeros(REPS)
    for t in range(T):
        a=schedule.get(t,0.)
        if a>0:
            k=np.tanh(a/2)
            x=rng.binomial(N,(1+k*rho[t])/2,size=REPS)
            output=(2*x/N-1)/k
        loss+=(output-rho[t])**2
        clipped+=(np.clip(output,-1,1)-rho[t])**2
    result={}
    for label,values in [('unclipped',loss),('clipped',clipped)]:
        mean=float(values.mean())
        se=float(values.std(ddof=1)/np.sqrt(REPS))
        result.update({label+'_mc_mean':mean,label+'_mc_se':se,
                       label+'_mc_low':mean-1.96*se,label+'_mc_high':mean+1.96*se})
    return result


def moment_checks():
    rows=[]
    x=np.arange(N+1)
    for a in [.025,.1,.25,.5]:
        k=np.tanh(a/2)
        y=(2*x/N-1)/k
        for rho in [.2,.6]:
            p=binom.pmf(x,N,(1+k*rho)/2)
            calculated=float(p@((y-rho)**2))
            expected=variance(a,rho)
            assert abs(p@y-rho)<1e-10
            assert abs(calculated-expected)<1e-10
            rows.append(dict(a=a,rho=rho,exact_rr_mse=calculated,formula_mse=expected,
                             absolute_error=abs(calculated-expected)))
    save('moment_checks.csv',rows)


def main():
    moment_checks()
    rho=np.full(T,.6)
    rows=[]
    for k in range(1,T+1):
        times=[int(j*T/k) for j in range(k)]
        schedule={t:B/k for t in times}
        assert len(schedule)==k
        proxy=k*variance(B/k,.6)+(T-k)*.6**2
        upper_proxy=k*variance(B/k,0.)+(T-k)*.6**2
        risk=exact_risk(rho,schedule)
        sim=simulate(rho,schedule,k)
        z=(sim['unclipped_mc_mean']-risk)/sim['unclipped_mc_se']
        assert abs(z)<6
        rows.append(dict(k=k,budget_each=B/k,publication_times=' '.join(map(str,times)),
            frozen_score_proxy=proxy,frozen_score_upper_proxy=upper_proxy,
            true_cumulative_mse=risk,standardized_mc_error=z,**sim))
    save('fixed_schedule.csv',rows)
    proxy_best=min(rows,key=lambda r:r['frozen_score_proxy'])['k']
    risk_best=min(rows,key=lambda r:r['true_cumulative_mse'])['k']
    print('Frozen proxy minimum:',proxy_best,'schedule risk minimum:',risk_best,flush=True)
    for row in rows:
        if row['k'] in [1,proxy_best]: print(row,flush=True)
    scenarios={'stable':np.full(T,.2),'jump_after_8':np.r_[np.full(8,.2),np.full(12,.6)]}
    policies={'frontload':{0:B},'reserve':{0:B/2,8:B/2}}
    future=[]
    for si,(name,path) in enumerate(scenarios.items()):
        for pi,(policy,schedule) in enumerate(policies.items()):
            risk=exact_risk(path,schedule)
            sim=simulate(path,schedule,100+si*2+pi)
            z=(sim['unclipped_mc_mean']-risk)/sim['unclipped_mc_se']
            assert abs(z)<6
            row=dict(scenario=name,policy=policy,true_cumulative_mse=risk,
                     standardized_mc_error=z,**sim)
            future.append(row)
            print(row,flush=True)
    save('same_prefix.csv',future)
    # Literal OUE variance from MTSP-LDP, not the RR population variance above.
    def oue(a):
        return 4*np.exp(a)/(N*np.expm1(a)**2)
    score=np.r_[np.zeros(W-1),.01]
    ordered=np.sort(score)[::-1]
    costs=[float(score.sum())]+[float(k*oue(B/k)+ordered[k:].sum()) for k in range(1,W+1)]
    chosen=int(np.argmin(costs))
    assert chosen>0 and .01>=ordered[chosen-1]
    save('cap_objective.csv',[dict(k=k,proxy_cost=value) for k,value in enumerate(costs)])
    cap=[]
    for remaining in [.5,.1,.05,.01]:
        wanted=B/chosen
        actual=min(wanted,remaining)
        row=dict(remaining=remaining,chosen_k=chosen,requested_budget=wanted,actual_budget=actual,
                 decision_publication_proxy=oue(wanted),executed_publication_proxy=oue(actual),
                 current_hold_score=.01,actual_proxy_exceeds_hold_score=oue(actual)>.01)
        cap.append(row)
        print(row,flush=True)
    save('cap_diagnostic.csv',cap)
    (OUT/'metadata.json').write_text(json.dumps(dict(seed=SEED,repeats=REPS,n=N,T=T,w=W,
        total_epsilon_cap=1.,publication_budget_used=B,monitoring_not_simulated=True,
        target='current population correlation',schedule='fixed public; never chosen by truth online',
        frozen_scores='full-information diagnostic, not deployable private input',
        full_paper_reproduction=False,policy_optimality_claim=False,
        python=sys.version,numpy=np.__version__),indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
