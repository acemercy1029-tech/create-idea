"""Isolate reuse, predictable budgets, and selection; not a paper reproduction."""
from pathlib import Path
import csv
import json
import sys
import numpy as np
import scipy
from scipy.stats import binom

OUT = Path(__file__).resolve().parent
N, REPS, SEED, W = 2800, 200000, 2026101610, 20


def variance(a):
    return (1/np.tanh(np.asarray(a)/2)**2-1)/N


def rr(mu, a, rng, shape=(REPS,)):
    nplus = int(round(N*(1+mu)/2))
    assert np.isclose(2*nplus/N-1, mu)
    keep = (1+np.tanh(np.asarray(a)/2))/2
    count = rng.binomial(nplus, keep, size=shape)+rng.binomial(N-nplus, 1-keep, size=shape)
    return (2*count/N-1)/np.tanh(np.asarray(a)/2)


def zero_pmf(a):
    k = np.arange(N//2+1)
    keep = (1+np.tanh(a/2))/2
    p = np.convolve(binom.pmf(k, N//2, keep), binom.pmf(k, N//2, 1-keep))
    p /= p.sum()
    y = (2*np.arange(N+1)/N-1)/np.tanh(a/2)
    assert abs(p @ y) < 1e-10
    assert np.isclose(p @ (y*y), variance(a), rtol=1e-9)
    return y, p


def max_abs_second(a, k=4):
    y, p = zero_pmf(a)
    # Group exactly by integer distance from N/2, avoiding float abs ties.
    distance = abs(np.arange(N+1)-N//2)
    q = np.bincount(distance, weights=p)
    cdf = np.cumsum(q)
    probs = np.diff(np.r_[0., np.minimum(cdf, 1.)**k])
    square = (2*np.arange(N//2+1)/N/np.tanh(a/2))**2
    return float(probs @ square)


def ledger_max(values):
    ledger = np.r_[np.zeros(W), values, np.zeros(W)]
    return float(np.convolve(ledger, np.ones(W), 'valid').max())


def main():
    rng = np.random.default_rng(SEED)
    rows, budgets = [], []
    max_second = max_abs_second(.15)
    ys, ps = zero_pmf(.2)
    branch = abs(ys) > 1.5*np.sqrt(variance(.2))
    p_high = float(ps[branch].sum())
    e_var_now = p_high*variance(.5)+(1-p_high)*variance(.1)
    for name, spends in [('reuse',[.2,.4]), ('predictable_low',[.2,.1]),
                          ('predictable_high',[.2,.5]), ('selected',[.15]*4+[.4]),
                          ('independent_check',[.075+.075]*4+[.4])]:
        maximum = ledger_max(spends)
        assert maximum <= 1+1e-12
        budgets.append(dict(case=name, max_window_budget=maximum))

    def record(delta, case, correction, score, target, expected, expected_residual=0., check=True):
        residual = score-target
        mean, bias = float(score.mean()), float(residual.mean())
        score_se = float(score.std(ddof=1)/np.sqrt(len(score)))
        resid_se = float(residual.std(ddof=1)/np.sqrt(len(score)))
        if check:
            assert abs(mean-expected) <= 6*score_se+1e-12
            assert abs(bias-expected_residual) <= 6*resid_se+1e-12
        rows.append(dict(delta=delta, case=case, correction=correction, samples=len(score),
            target_mean=float(np.mean(target)), expected_score=float(expected),
            score_mean=mean, score_ci_low=mean-1.96*score_se, score_ci_high=mean+1.96*score_se,
            expected_bias=float(expected_residual), empirical_bias=bias,
            bias_ci_low=bias-1.96*resid_se, bias_ci_high=bias+1.96*resid_se,
            score_sd=float(score.std(ddof=1)), score_mse=float(np.mean(residual**2))))

    for delta in [0., .4]:
        target = delta**2
        old, now = rr(0., .2, rng), rr(delta, .4, rng)
        diff = (now-old)**2
        wrong_bias = .75*variance(.2)
        record(delta,'four_copies','assume_independent',diff-variance(.4)-variance(.2)/4,
               target,target+wrong_bias,wrong_bias)
        record(delta,'four_copies','account_for_covariance',diff-variance(.4)-variance(.2),target,target)

        old = rr(0., .2, rng)
        high = abs(old) > 1.5*np.sqrt(variance(.2))
        a = np.where(high, .5, .1)
        now = rr(delta, a, rng)
        diff = (now-old)**2
        wrong_bias = float(variance(.2)-e_var_now)
        record(delta,'predictable_budget','twice_current_variance',diff-2*variance(a),
               target,target+wrong_bias,wrong_bias)
        corrected = diff-variance(a)-variance(.2)
        record(delta,'predictable_budget','actual_variance_sum',corrected,target,target)
        for flag, label in [(True,'high'), (False,'low')]:
            select = high if flag else ~high
            exact_select = branch if flag else ~branch
            conditional_second = float(ps[exact_select] @ ys[exact_select]**2/ps[exact_select].sum())
            bias = conditional_second-variance(.2)
            record(delta,'predictable_budget_'+label,'actual_variance_sum',corrected[select],
                   target,target+bias,bias)

        old = rr(0., .15, rng, shape=(REPS,4))
        now = rr(delta, .4, rng)
        idx = np.argmax(abs(old), axis=1)
        selected = old[np.arange(REPS),idx]
        diff = (now-selected)**2
        selected_bias = max_second-variance(.15)
        record(delta,'selected_history','ordinary_variance_sum',diff-variance(.15)-variance(.4),
               target,target+selected_bias,selected_bias)
        record(delta,'fixed_first_history','ordinary_variance_sum',(now-old[:,0])**2-variance(.15)-variance(.4),
               target,target)
        # This is a different, random target: loss against the already realized release.
        record(delta,'selected_history_prediction_risk','current_variance_only',diff-variance(.4),
               (delta-selected)**2,target+max_second)

        a_hist = rr(0., .075, rng, shape=(REPS,4))
        b_hist = rr(0., .075, rng, shape=(REPS,4))
        idx = np.argmax(abs(a_hist), axis=1)
        selected_b = b_hist[np.arange(REPS),idx]
        now = rr(delta, .4, rng)
        score = (now-selected_b)**2-variance(.075)-variance(.4)
        record(delta,'independent_validation','actual_variance_sum',score,target,target)

    for filename, values in [('summary.csv',rows), ('privacy_ledger.csv',budgets)]:
        with (OUT/filename).open('w',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(values[0]))
            writer.writeheader(); writer.writerows(values)
    (OUT/'metadata.json').write_text(json.dumps(dict(seed=SEED,repeats=REPS,n=N,w=W,
        epsilon_cap=1,common_communication_cap_bits=2,estimates_clipped=False,
        fixed_records=True,history_candidates=4,probability_high_budget=p_high,
        exact_selected_second_moment=max_second,selected_history_bias=max_second-float(variance(.15)),
        python=sys.version,numpy=np.__version__,scipy=scipy.__version__,full_paper_reproduction=False),indent=2),encoding='utf-8')
    for row in rows:
        print(row['delta'],row['case'],row['correction'],
              'bias=',round(row['empirical_bias'],6),'MSE=',round(row['score_mse'],6))
    print('All analytic mean checks and window ledgers passed.')


if __name__=='__main__':
    main()
