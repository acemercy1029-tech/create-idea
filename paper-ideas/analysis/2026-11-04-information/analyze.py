"""Archive numerical consistency checks, privacy-budget bounds, and a figure."""
from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import run_information as a
OUT=Path(__file__).resolve().parent

def read(name):
    return list(csv.DictReader((OUT/name).open(encoding='utf-8')))

def main():
    results=read('results.csv'); projection=read('projection_results.csv')
    bounds=read('information_bounds.csv');votes=read('local_votes.csv');raw=read('raw_diagnostic.csv')
    # All risk rows must reconstruct from independently integrated publication law.
    maxerr=0.
    for rows in [results,projection]:
        for r in rows:
            if not r['metric'].startswith('mse'):continue
            p=next(float(q['mean']) for q in rows if all(q[k]==r[k] for k in ['lam','b','method','period']) and q['metric']=='support_success')
            th=r['metric']=='mse_threshold'
            expected=p*float(a.g.single_risk(True,512,th))+(1-p)*float(a.g.single_risk(False,512,th))
            maxerr=max(maxerr,abs(expected-float(r['mean'])))
    assert maxerr<1e-12
    # Numeric audit of the analytic channel-range variance maximum.
    varianceerr=0.
    for qa in [0.,.2,.9,1.]:
        for eps in [.1,.5,1.,2.,4.]:
            e=np.exp(eps)-1
            s=e/(2+(1-qa)*e)
            expected=e*e/(4*(1+(1-qa)*e))
            actual=s*(e-s)/(1+(1-qa)*s)**2
            grid=np.linspace(0,e,10001)
            f=grid*(e-grid)/(1+(1-qa)*grid)**2
            varianceerr=max(varianceerr,abs(actual-expected))
            assert f.max()<=expected+1e-10
    budgets=[]
    for lam in [0.,.6]:
        for b in [4,8,16,64]:
            vv=[r for r in votes if float(r['lam'])==lam and int(r['b'])==b]
            indexcap=next(float(r['information_upper']) for r in bounds if float(r['lam'])==lam and int(r['b'])==b and r['t']=='167' and r['channel']=='index')
            for eps in [.1,.3,.5,1.,2.,4.]:
                e=np.exp(eps)-1
                info=min(indexcap,sum(int(r['n'])*float(r['delta'])**2*e**2/(4*(1+(1-float(r['old_vote']))*e)) for r in vv))
                upper=a.success_bound(info)
                budgets.append(dict(lam=lam,b=b,epsilon=eps,information_upper=info,success_upper=upper,mse_lower=.36*(1-upper)))
    a.save('budget_bounds.csv',budgets)
    get=lambda rows,lam,b,method,metric,period:next(float(r['mean']) for r in rows if float(r['lam'])==lam and int(r['b'])==b and r['method']==method and r['metric']==metric and r['period']==period)
    fig,axes=plt.subplots(2,2,figsize=(11.5,8.2),constrained_layout=True)
    x=np.arange(4);w=.36
    for b,off,color in [(8,-w/2,'#2277aa'),(64,w/2,'#cc7722')]:
        vals=[next(float(r['success']) for r in raw if float(r['lam'])==0 and int(r['b'])==b and r['t']=='167')]
        vals += [get(results,0,b,m,'support_success','last') for m in ['index_ml','onebit_ml','rr_ml']]
        axes[0,0].bar(x+off,np.array(vals)*100,w,label='b='+str(b),color=color)
    axes[0,0].set_yscale('log');axes[0,0].set_ylim(.1,160)
    axes[0,0].set_xticks(x);axes[0,0].set_xticklabels(['Full counts','Top1 index','1-bit sketch','1-bit + RR'])
    axes[0,0].set_ylabel('Final-round location success (%)');axes[0,0].set_title('A. Diagnostic with known onset / old location')
    axes[0,0].legend();axes[0,0].axhline(100/255,color='grey',ls=':',lw=1)
    bs=[4,8,16,64]
    for lam,color in [(0.,'#2277aa'),(.6,'#cc7722')]:
        vv=[next(float(r['success_upper'])*100 for r in bounds if float(r['lam'])==lam and int(r['b'])==b and r['t']=='167' and r['channel']=='noninteractive_index_ldp') for b in bs]
        axes[0,1].plot(range(4),vv,'o-',label='lambda='+str(lam),color=color)
    axes[0,1].set_xticks(range(4));axes[0,1].set_xticklabels(bs);axes[0,1].set_xlabel('Local history b')
    axes[0,1].set_ylabel('Upper bound on success (%)');axes[0,1].set_title('B. Any noninteractive LDP channel of top1')
    axes[0,1].legend();axes[0,1].set_ylim(0,32)
    for rows,method,color,label in [(results,'causal_g4','#2277aa','Top1 + RR'),(projection,'projection_g4','#cc7722','Raw projection + RR')]:
        vals=[get(rows,0,b,method,'mse_threshold','event8') for b in bs]
        axes[1,0].plot(range(4),vals,'o-',label=label,color=color)
    axes[1,0].axhline(.36,color='black',ls=':',label='Always zero')
    axes[1,0].set_xticks(range(4));axes[1,0].set_xticklabels(bs);axes[1,0].set_xlabel('Local history b')
    axes[1,0].set_ylabel('Event-average total MSE');axes[1,0].set_title('C. Causal baselines; g=4, lambda=0')
    axes[1,0].set_ylim(.350,.362);axes[1,0].legend(fontsize=8)
    for b,color in [(8,'#2277aa'),(64,'#cc7722')]:
        vv=[r for r in budgets if r['lam']==0 and r['b']==b]
        axes[1,1].plot([r['epsilon'] for r in vv],[r['success_upper']*100 for r in vv],'o-',label='b='+str(b),color=color)
    axes[1,1].set_ylim(0,105);axes[1,1].set_xlabel('Privacy epsilon');axes[1,1].set_ylabel('Upper bound on success (%)')
    axes[1,1].set_title('D. Necessary-information bound; not achievability');axes[1,1].legend()
    for ax in axes.flat:ax.grid(alpha=.18,axis='y');ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Short 8-round association: where information is lost\nN=20480, M=256, w=20; diagnosis is not a full-stream detection guarantee',fontsize=13)
    fig.savefig(OUT/'information-results.png',dpi=180);fig.savefig(OUT/'information-results.pdf');plt.close(fig)
    (OUT/'output_checks.json').write_text(json.dumps(dict(risk_reconstruction_max_error=maxerr,variance_bound_max_error=varianceerr,
        main_checks=len(read('checks.csv')),projection_checks=len(read('projection_checks.csv')),
        rows_main=len(results),rows_projection=len(projection),rows_bounds=len(bounds)),indent=2),encoding='utf-8')
    print((OUT/'output_checks.json').read_text())
    print('Budget-bound examples', [r for r in budgets if r['lam']==0 and r['b']==8])

if __name__=='__main__':main()
