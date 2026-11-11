"""IID raw-block information bound, for fixed periodic noninteractive one-bit sampling."""
from pathlib import Path
from itertools import product
import csv
import json
import math
import numpy as np
from scipy.optimize import brentq
from scipy.special import xlogy
from duration_sweep import ONSET, DURATIONS, previous, OUT, write_rows

K = 255
RHO = .6


def divergence(p, q):
    return xlogy(p, p/q) + xlogy(1-p, (1-p)/(1-q))


def success_ceiling(information, candidates=K):
    baseline = 1 / candidates
    if information <= 0:
        return baseline
    if information >= math.log(candidates):
        return 1.
    return brentq(lambda p: divergence(p, baseline)-information, baseline, 1.)


def constant(epsilon):
    lower = math.exp(-epsilon)
    c = (lower*math.log(lower)-lower+1)/(1-lower)**2
    return c*math.tanh(epsilon/2)**2


def risk_floor(b, duration, epsilon, tail=False):
    info = 0.
    missed = []
    for age in range(1, duration+1):
        users = math.ceil(10240/(19+b)) if tail else previous.nphase(b, ONSET+age-1)
        variance = math.expm1(min(age, b)*math.log1p(RHO**2))
        info += users*constant(epsilon)*variance/K
        missed.append(1-success_ceiling(info))
    return RHO**2*np.mean(missed)


def eight_round_budget_needed():
    """Invert the all-history risk bound, including the infinite tail of b."""
    def gap(epsilon):
        floors = [risk_floor(b, 8, epsilon) for b in range(1, 8)]
        floors.append(risk_floor(8, 8, epsilon, tail=True))
        return min(floors) - RHO**2/2

    epsilon = brentq(gap, 1., 1.25, xtol=1e-12)
    assert gap(epsilon - 1e-6) > 0 > gap(epsilon + 1e-6)
    return epsilon


def enumeration():
    # K=3, m=2. Integrate every raw path; use binary channels with unequal endpoints.
    rng = np.random.default_rng(2026111122)
    candidates, m = 3, 2
    data = np.array(list(product([-1,1],repeat=candidates*m))).reshape(-1,candidates,m)
    ratios = np.prod(1+RHO*data,axis=2)
    variance = (1+RHO**2)**m-1
    gram = (ratios-1).T @ (ratios-1)/len(data)
    np.testing.assert_allclose(gram, np.eye(candidates)*variance, atol=1e-12)
    best_ratio = 0.
    for eps in [.3,1.,2.]:
        kappa = math.tanh(eps/2)
        for i in range(100):
            # First sample randomized binary statistic, apply RR, then independent output map.
            feature = rng.integers(0,2,len(data)) if i%2 else (ratios[:,i%candidates]>1)
            f = (1-kappa)/2+kappa*feature
            post_low, post_high = sorted(rng.random(2))
            f = post_low+(post_high-post_low)*f
            base = np.mean(f)
            out = (ratios.T@f)/len(data)
            measured = np.mean(divergence(out,base))
            upper = constant(eps)*variance/candidates
            assert measured <= upper+1e-12
            best_ratio = max(best_ratio, float(measured/upper))
            # Exact single-bit MAP success under uniform prior vs inverse binary KL.
            mutual = np.mean(divergence(out, out.mean()))
            success = (max(out)+max(1-out))/candidates
            assert success <= success_ceiling(mutual,candidates)+1e-10
    return best_ratio


def main():
    ratio = enumeration()
    detail, envelopes = [], []
    for eps in [.5,.75,1.,1.25,1.5,2.]:
        for duration in DURATIONS:
            group = []
            for b in range(1,duration+1):
                tail = b==duration
                value = risk_floor(b,duration,eps,tail)
                row = dict(epsilon=eps,duration=duration,b=b,at_least_b=tail,risk_lower=value)
                detail.append(row)
                group.append(row)
            winner = min(group,key=lambda row:row['risk_lower'])
            envelopes.append(dict(winner))
    write_rows('raw_bound_detail.csv',detail)
    write_rows('raw_bound_all_histories.csv',envelopes)
    needed = eight_round_budget_needed()
    write_rows('budget_boundary.csv', [dict(duration=8, target=RHO**2/2,
                                            necessary_epsilon=needed)])
    # Independent brute-force audit for individual histories through b=512.
    base = [r for r in envelopes if r['epsilon']==1 and r['duration']==8][0]
    brute = [risk_floor(b,8,1) for b in range(1,513)]
    assert min(brute) >= base['risk_lower']-1e-12
    assert 0.192 < base['risk_lower'] < .193 and np.argmin(brute)+1==6
    (OUT/'raw_bound_checks.json').write_text(json.dumps(dict(exact_channel_checks=300,
        max_exact_to_upper_ratio=ratio, enumerated_histories=512, minimum_exact_schedule_bound=min(brute)),indent=2))
    print('eps=1 event8 all fixed histories',base,'brute',min(brute))

if __name__ == '__main__':
    main()
