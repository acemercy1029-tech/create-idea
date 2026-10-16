"""Isolated RR moment check, NOT a reproduction of GroupSmooth or its tree FO.

Fixed balanced records at two timestamps. Conditional mechanism variance is exact.
Written after the main robustness experiment; no main metrics/parameters changed.
"""
from pathlib import Path
import csv
import numpy as np

OUT = Path(__file__).resolve().parent
N, REPS, SEED = 2800, 200000, 2026101609


def variance(eps):
    return (1/np.tanh(eps/2)**2-1)/N


def release(eps, rng):
    p = (1+np.tanh(eps/2))/2
    plus = rng.binomial(N//2, p, size=REPS)+rng.binomial(N//2, 1-p, size=REPS)
    return (2*plus/N-1)/np.tanh(eps/2)


def main():
    rng = np.random.default_rng(SEED)
    rows = []
    for before, now in [(.1, .5), (.5, .1), (.3, .3)]:
        assert before+now <= 1
        y0, y1 = release(before, rng), release(now, rng)
        v0, v1 = variance(before), variance(now)
        for correction, value, expected in [('twice_current_variance', 2*v1, v0-v1),
                                             ('sum_actual_variances', v0+v1, 0.)]:
            scores = (y1-y0)**2-value
            mean = float(scores.mean())
            se = float(scores.std(ddof=1)/np.sqrt(REPS))
            assert abs(mean-expected) < 6*se
            rows.append(dict(previous_epsilon=before, current_epsilon=now,
                previous_variance=v0, current_variance=v1, true_squared_change=0.,
                correction=correction, theoretical_expected_score=expected,
                simulated_mean_score=mean, ci_low=mean-1.96*se, ci_high=mean+1.96*se,
                repeats=REPS, seed=SEED))
    with (OUT/'variance_audit.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)


if __name__ == '__main__':
    main()
