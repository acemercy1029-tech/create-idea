"""Additional exact accounting illustration and local-kernel table.

Run after the original pilot; no method or parameter tuning. Bounds are
elementary sufficient conditions, not new optimality or impossibility results.
"""
import math
import numpy as np
from run import OUT, EPS, KAPPA, winner_prob, save


def main():
    # Reports at 2 and 6 read [0,2] and [4,6]. Changed input window is [2,5].
    x = np.array([1, -1, 1, 1, 1, 1, 1])
    xp = np.array([1, -1, -1, 1, -1, -1, 1])
    assert np.array_equal(x[:2], xp[:2]) and x[6] == xp[6]
    means = [np.sign(x[t-2:t+1].sum()) for t in [2, 6]]
    means_p = [np.sign(xp[t-2:t+1].sum()) for t in [2, 6]]
    assert means == [1, 1] and means_p == [-1, -1]
    p, pp = ((1 + KAPPA) / 2) ** 2, ((1 - KAPPA) / 2) ** 2
    assert abs(math.log(p / pp) - 2 * EPS) < 1e-12
    save('history_privacy_counterexample.csv', [dict(w=4, history=3, report_stride=4,
        changed_start=2, changed_end=5, report_times='2;6', outputs='+1;+1',
        probability_x=p, probability_x_prime=pp, log_likelihood_ratio=math.log(p / pp),
        claimed_epsilon=EPS, exceeds_claim=True)])
    rows = []
    for dims in [16, 64, 256]:
        for b in [16, 64]:
            for amp in [.2, .6]:
                for active in sorted(set([8, b // 2, b])):
                    prob = winner_prob(b, active, amp, dims)
                    gap = prob - (1 - prob) / (dims - 1)
                    # A sufficient Hoeffding bound for independent one-bit
                    # Hadamard reports; deliberately conservative, not necessary.
                    bound = math.ceil(8 * math.log((dims - 1) / .05) / (KAPPA ** 2 * gap ** 2)) if gap > 0 else ''
                    rows.append(dict(dimensions=dims, batch=b, active_in_batch=active,
                        amplitude=amp, local_correct_probability=prob, private_vote_mean_gap=gap,
                        sufficient_reports_95pct_bound=bound))
    save('exact_local_frontier.csv', rows)
    print('counterexample log likelihood ratio:', math.log(p / pp))
    print('exact local frontier rows:', len(rows))


if __name__ == '__main__':
    main()
