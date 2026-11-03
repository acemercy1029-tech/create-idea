"""Ordinary thinning diagnostic; stationary only, no new algorithm claim."""
import numpy as np
from scipy.stats import binom
import run_dependence as d

rows = []
reps = 10000
for li, lam in enumerate(d.LAMBDAS):
    for ci, (label, b, step) in enumerate([('contiguous16', 16, 1), ('spaced16', 16, 4), ('contiguous64', 64, 1)]):
        rng = np.random.default_rng(np.random.SeedSequence([2026110316, li, ci]))
        span = 1 + (b - 1) * step
        stride = d.W + span - 1
        pa = d.raw_counts((.8,) * b, lam ** step)
        pb = d.raw_counts((.5,) * b, lam ** step)
        qa, qb = d.winners(pa, pb, lam ** step)
        vote = np.full(d.M, (1 - qa) / (d.M - 1))
        vote[d.A] = qa
        assert abs(qb - vote[d.B]) < 1e-10
        row_mean = d.H @ vote
        probs = np.r_[1 + d.KAPPA * row_mean, 1 - d.KAPPA * row_mean] / (2 * d.M)
        reporters = sum((d.N // 2) // stride + int(t % stride < (d.N // 2) % stride) for t in range(156, 160))
        correct = []
        for _ in range(reps // 500):
            counts = rng.multinomial(reporters, probs / probs.sum(), size=500)
            scores = d.iid.fwht(counts[:, :d.M] - counts[:, d.M:])
            correct.extend(d.iid.argmax_random(scores, rng) == d.A)
        correct = np.array(correct)
        for threshold in [False, True]:
            vals = np.where(correct, d.risk(.6, threshold), .36 + d.risk(0., threshold))
            se = vals.std(ddof=1) / np.sqrt(reps)
            rows.append(dict(lam=lam, method=label, records=b, spacing=step, span=span, report_stride=stride,
                             reporters_in_g4=reporters, mean_variance_effective_samples=d.effective_size(b, lam ** step),
                             local_correct_probability=qa, private_correct_probability=correct.mean(),
                             threshold=threshold, mse=vals.mean(), mc_se=se, low=vals.mean()-1.96*se, high=vals.mean()+1.96*se))
d.save('thinning_check.csv', rows)
print('Saved 18 stationary settings, 36 outputs; 10000 private snapshots per setting, shared by both output rules.')
