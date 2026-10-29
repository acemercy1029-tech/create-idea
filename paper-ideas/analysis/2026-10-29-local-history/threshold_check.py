"""Post-pilot, exact risk check of a public Hoeffding threshold.

No new Monte Carlo trials, no new private reports, no novel estimator claim.
"""
import csv
import numpy as np
from scipy.stats import binom
from run import OUT, KAPPA, N, W, save


def main():
    n, alpha = N // 2 // W, .01
    tau = np.sqrt(2 * np.log(2 / alpha) / (n * KAPPA ** 2))
    k = np.arange(n + 1)
    estimate = (2 * k / n - 1) / KAPPA
    output = np.where(np.abs(estimate) > tau, np.clip(estimate, -1, 1), 0.)
    risks, moments = {}, []
    for rho in [0., .2, .6]:
        prob = binom.pmf(k, n, (1 + KAPPA * rho) / 2)
        risk = float(prob @ ((output - rho) ** 2))
        risks[rho] = risk
        moments.append(dict(rho=rho, n=n, public_alpha=alpha, threshold=tau,
                            exact_conditional_mse=risk, probability_nonzero=float(prob @ (output != 0))))
    assert moments[0]['probability_nonzero'] < alpha
    save('threshold_moments.csv', moments)
    summary = list(csv.DictReader((OUT / 'stronger-check' / 'summary.csv').open()))
    rows = []
    for item in summary:
        if item['metric'] != 'support_accuracy' or not item['method'].startswith(('server_', 'local_')):
            continue
        scene, period = item['scene'], item['period']
        if period == 'evaluation' and scene.startswith('amp'):
            continue  # mixed-truth periods need time weighting; not presented here
        rho = .6 if scene == 'constant_06' or scene.startswith('amp06') else .2 if scene.startswith('amp02') else 0.
        if period == 'after32':
            rho = 0.
        acc = float(item['mean'])
        value = acc * risks[rho] + (1 - acc) * (risks[0.] + rho ** 2)
        rows.append(dict(scene=scene, method=item['method'] + '_threshold', period=period,
                         mean_l2_mse=value, selector_accuracy=acc,
                         origin='second-run selector MC; exact conditional release risk; post-hoc'))
    save('threshold_check.csv', rows)
    for row in moments:
        print(row)


if __name__ == '__main__':
    main()
