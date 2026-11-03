"""Exact Markov count kernels + private selection snapshot integration.

Snapshots estimate time-average expected risk, NOT full-path statistics.
Methods do not see the latent count laws. See PROTOCOL.md.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
from pathlib import Path
import sys
import csv
import json
import time
import itertools
from functools import lru_cache
import numpy as np
from scipy.stats import binom

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT.parent / '2026-10-29-local-history'))
import run as iid

N, M, W, EPS, R, SEED = 20480, 256, 20, 1., 1000, 2026103016
A, B, T = 37, 121, 320
LAMBDAS = [0., .3, .6, .8, .9, .95]
KAPPA = np.tanh(EPS / 2)
H = iid.H
HS = [8, 16, 32, 64, 128]


def save(name, rows):
    with (OUT / name).open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def trans(p0, p1, lam):
    q = lam * min(p0, p1) + (1 - lam) * p0 * p1
    a, b = (p1 - q) / (1 - p0), q / p0
    ans = np.array([[1 - a, a], [1 - b, b]])
    assert ans.min() >= -1e-13
    assert np.max(np.abs(np.array([1 - p0, p0]) @ ans - [1 - p1, p1])) < 1e-12
    return ans


def count_chain(p0, matrices, emission):
    """Distribution of number of emitted ones along a hidden two-state chain."""
    state = np.zeros((2, 2))
    prior = np.array([1 - p0, p0])
    state[:, 0] = prior * (1 - emission)
    state[:, 1] = prior * emission
    for matrix in matrices:
        predicted = matrix.T @ state
        state = np.zeros((2, predicted.shape[1] + 1))
        state[:, :-1] += predicted * (1 - emission[:, None])
        state[:, 1:] += predicted * emission[:, None]
    pmf = state.sum(axis=0)
    assert abs(pmf.sum() - 1) < 2e-11
    return pmf / pmf.sum()


@lru_cache(None)
def raw_counts(ps, lam):
    mats = [trans(ps[i - 1], ps[i], lam) for i in range(1, len(ps))]
    return count_chain(ps[0], mats, np.array([0., 1.]))


def folded(pmf):
    return iid.abs_pmf(pmf, len(pmf) - 1)


@lru_cache(None)
def null_weights(b, lam, dims=M):
    null = folded(raw_counts((.5,) * b, lam))
    cdf = np.cumsum(null)
    cdf[-1] = 1.
    w1, w2 = np.zeros(b + 1), np.zeros(b + 1)
    k = np.arange(dims - 1)
    for score in range(b + 1):
        if cdf[score] > 0:
            p = min(1., null[score] / cdf[score])
            weights = binom.pmf(k, dims - 2, p) * cdf[score] ** (dims - 2)
            w1[score] = np.sum(weights / (k + 1))
            w2[score] = np.sum(weights / (k + 2))
    return w1, w2


def winners(pa, pb, lam, dims=M):
    """Two exceptional coordinates, dims-2 iid nulls, uniform ties."""
    b = len(pa) - 1
    aa, bb = folded(pa), folded(pb)
    ca = np.r_[0., np.cumsum(aa)[:-1]]
    cb = np.r_[0., np.cumsum(bb)[:-1]]
    w1, w2 = null_weights(b, lam, dims)
    qa = float(aa @ (cb * w1 + bb * w2))
    qb = float(bb @ (ca * w1 + aa * w2))
    assert qa >= 0 and qb >= 0 and qa + qb <= 1 + 1e-12
    return qa, qb


class StreamLaw:
    def __init__(self, ps, lam):
        self.ps = np.asarray(ps)
        self.lam = lam
        self.step = [np.eye(2)] + [trans(ps[t - 1], ps[t], lam) for t in range(1, len(ps))]
        self.skip = [np.eye(2) for _ in ps]
        for t in range(W, len(ps)):
            mat = np.eye(2)
            for j in range(t - W + 1, t + 1):
                mat = mat @ self.step[j]
            self.skip[t] = mat

    def raw(self, end, b):
        return raw_counts(tuple(self.ps[end - b + 1:end + 1]), self.lam)

    @lru_cache(None)
    def reported_counts(self, start, end):
        """2 users per phase/coordinate, revisited every W times, fresh RR."""
        total = np.ones(1)
        emission = np.array([(1 - KAPPA) / 2, (1 + KAPPA) / 2])
        for phase in range(W):
            times = [t for t in range(start, end + 1) if t % W == phase]
            if not times:
                continue
            single = count_chain(self.ps[times[0]], [self.skip[t] for t in times[1:]], emission)
            total = np.convolve(total, np.convolve(single, single))
        assert len(total) == 2 * (end - start + 1) + 1
        assert abs(total.sum() - 1) < 1e-10
        return total / total.sum()


def risk(rho, threshold):
    n = N // 2 // W
    x = np.arange(n + 1)
    z = (2 * x / n - 1) / KAPPA
    tau = np.sqrt(2 * np.log(200) / (n * KAPPA ** 2)) if threshold else 0.
    estimate = np.where(abs(z) > tau, np.clip(z, -1, 1), 0.)
    return float(binom.pmf(x, n, (1 + KAPPA * rho) / 2) @ ((estimate - rho) ** 2))


def effective_size(b, lam):
    lag = np.arange(1, b)
    return b / (1 + 2 * np.sum((1 - lag / b) * lam ** lag))


def validate():
    rows = []
    rng = np.random.default_rng(SEED + 99)
    for ps, lam in [((.5, .5, .8, .8), .8), ((.8, .8, .5, .5), .95), ((.8,) * 5, .6)]:
        for emission in [np.array([0., 1.]), np.array([(1 - KAPPA) / 2, (1 + KAPPA) / 2])]:
            direct = np.zeros(len(ps) + 1)
            for path in itertools.product([0, 1], repeat=len(ps)):
                prob = ps[0] if path[0] else 1 - ps[0]
                for j in range(1, len(ps)):
                    prob *= trans(ps[j - 1], ps[j], lam)[path[j - 1], path[j]]
                emissions = np.ones(1)
                for z in path:
                    emissions = np.convolve(emissions, [1 - emission[z], emission[z]])
                direct += prob * emissions
            ans = count_chain(ps[0], [trans(ps[j - 1], ps[j], lam) for j in range(1, len(ps))], emission)
            error = float(np.max(np.abs(ans - direct)))
            assert error < 1e-12
            rows.append(dict(test='full_path_enumeration', parameter=str((ps, lam, emission.tolist())), error=error, tolerance=1e-12))
    for b in [16, 64]:
        for active in [0, 8, b]:
            pa = raw_counts((.5,) * (b - active) + (.8,) * active, 0.)
            pb = raw_counts((.5,) * b, 0.)
            actual = winners(pa, pb, 0.)[0]
            expected = iid.winner_prob(b, active, .6)
            error = abs(actual - expected)
            assert error < 1e-11
            rows.append(dict(test='iid_previous_winner', parameter=str((b, active)), error=error, tolerance=1e-11))
    sys.path.insert(0, str(OUT.parent / '2026-10-29-migration'))
    import run_migration as previous
    for b, ka, kb in [(16, 8, 8), (64, 56, 8), (64, 32, 32)]:
        qa, qb = winners(raw_counts((.8,) * ka + (.5,) * (b - ka), 0.),
                         raw_counts((.5,) * (b - kb) + (.8,) * kb, 0.), 0.)
        olda, oldb = previous.two_winners(b, ka, kb)
        error = max(abs(qa - olda), abs(qb - oldb))
        assert error < 1e-11
        rows.append(dict(test='iid_previous_two_winners', parameter=str((b, ka, kb)), error=error, tolerance=1e-11))
    # Direct simulation of local Markov records across ALL coordinates.
    for b, ka, kb, lam in [(16, 0, 16, .6), (64, 56, 8, .8), (64, 0, 64, .95)]:
        ps_a = np.r_[np.full(ka, .8), np.full(b - ka, .5)]
        ps_b = np.r_[np.full(b - kb, .5), np.full(kb, .8)]
        probs = np.full((b, M), .5)
        probs[:, A], probs[:, B] = ps_a, ps_b
        expected = winners(raw_counts(tuple(ps_a), lam), raw_counts(tuple(ps_b), lam), lam)
        reps, counts = 30000, np.zeros(2)
        for _ in range(reps // 500):
            z = rng.random((500, M)) < probs[0]
            accum = z.astype(int)
            for j in range(1, b):
                q = lam * np.minimum(probs[j - 1], probs[j]) + (1 - lam) * probs[j - 1] * probs[j]
                conditional = np.where(z, q / probs[j - 1], (probs[j] - q) / (1 - probs[j - 1]))
                z = rng.random(z.shape) < conditional
                accum += z
            chosen = iid.argmax_random(2 * accum - b, rng, absolute=True)
            counts += [np.sum(chosen == A), np.sum(chosen == B)]
        for k, name in enumerate(['A', 'B']):
            se = np.sqrt(expected[k] * (1 - expected[k]) / reps)
            error = abs(counts[k] / reps - expected[k])
            assert error < 6 * se + 1e-10
            rows.append(dict(test='raw_markov_selection_' + name, parameter=str((b, ka, kb, lam)), error=error, tolerance=6 * se + 1e-10))
    # Scalar observed-count kernel: repeat visits every 20 rounds after a change.
    for lam in [0., .8, .95]:
        ps = np.r_[np.full(30, .5), np.full(50, .8)]
        law = StreamLaw(ps, lam)
        times = [1, 21, 41, 61]
        emission = np.array([(1 - KAPPA) / 2, (1 + KAPPA) / 2])
        exact = count_chain(ps[1], [law.skip[t] for t in times[1:]], emission)
        reps = 100000
        z = rng.random(reps) < ps[0]
        counts = np.zeros(reps, dtype=int)
        for t in range(62):
            if t:
                z = rng.random(reps) < np.where(z, law.step[t][1, 1], law.step[t][0, 1])
            if t in times:
                counts += rng.random(reps) < np.where(z, emission[1], emission[0])
        empirical = np.bincount(counts, minlength=5) / reps
        tolerance = 6 * np.sqrt(exact * (1 - exact) / reps) + 1e-10
        assert np.all(abs(empirical - exact) < tolerance)
        rows.append(dict(test='raw_repeated_user_rr', parameter=str(lam), error=float(max(abs(empirical - exact))), tolerance=float(max(tolerance))))
    # Full server aggregate variance from autocovariance; nonstationary means
    # tested too. At lambda=0 the full count is the ordinary Poisson-binomial.
    for lam in LAMBDAS:
        for p in [.5, .8]:
            law = StreamLaw(np.full(T, p), lam)
            for h in [8, 32, 128]:
                pmf = law.reported_counts(160 - h, 159)
                x = np.arange(len(pmf))
                mean = 2 * h * (1 + KAPPA * (2 * p - 1)) / 2
                var = 2 * h * ((1 - (KAPPA * (2 * p - 1)) ** 2) / 4)
                for lag in range(W, h, W):
                    var += 4 * (h - lag) * KAPPA ** 2 * p * (1 - p) * lam ** lag
                error = max(abs(x @ pmf - mean), abs(((x - mean) ** 2) @ pmf - var))
                assert error < 1e-8
                rows.append(dict(test='server_count_moments', parameter=str((lam, p, h)), error=error, tolerance=1e-8))
            for b in [16, 64]:
                pmf = raw_counts((p,) * b, lam)
                v = ((np.arange(b + 1) - b * p) ** 2) @ pmf / b ** 2
                error = abs(v - p * (1 - p) / effective_size(b, lam))
                assert error < 1e-10
                rows.append(dict(test='effective_size_identity', parameter=str((lam, p, b)), error=float(error), tolerance=1e-10))
    # Privacy: arbitrary changed W-input interval may touch at most one block.
    for b in [16, 64]:
        stride = W + b - 1
        maximum = 0
        for phase in range(stride):
            ends = np.arange(-3 * stride + phase, 4 * stride + phase, stride)
            for start in range(stride):
                maximum = max(maximum, int(np.sum((ends >= start) & (ends - b + 1 <= start + W - 1))))
        assert maximum == 1
        rows.append(dict(test='affected_private_history_blocks', parameter=str((b, stride)), error=maximum, tolerance=1))
    save('kernel_checks.csv', rows)
    print('All {} kernel/accounting checks passed.'.format(len(rows)), flush=True)


def snapshots(lam, li, scene):
    p_a, p_b, p_0 = np.full(T, .5), np.full(T, .5), np.full(T, .5)
    if scene == 'constant':
        p_a[:] = .8
        times, target = [159], A
    else:
        p_a[96:160] = .8
        p_b[160:256] = .8
        times, target = list(range(160, 256)), B
    laws = [StreamLaw(ps, lam) for ps in [p_a, p_b, p_0]]
    rng = np.random.default_rng(np.random.SeedSequence([SEED, li, scene == 'migration']))
    methods = ['local16', 'local64'] + ['server' + str(h) for h in HS]
    correct = {method: np.zeros((R, len(times)), dtype=bool) for method in methods}
    vote_rows, curves = [], []
    for index, t in enumerate(times):
        for b in [16, 64]:
            stride = W + b - 1
            hist = np.zeros((R, M))
            for end in range(t - 3, t + 1):
                n = (N // 2) // stride + int(end % stride < (N // 2) % stride)
                qa, qb = winners(laws[0].raw(end, b), laws[1].raw(end, b), lam)
                vote = np.full(M, (1 - qa - qb) / (M - 2))
                vote[A], vote[B] = qa, qb
                row_mean = H @ vote
                probs = np.r_[1 + KAPPA * row_mean, 1 - KAPPA * row_mean] / (2 * M)
                assert abs(probs.sum() - 1) < 1e-11 and probs.min() >= 0
                counts = rng.multinomial(n, probs / probs.sum(), size=R)
                hist += counts[:, :M] - counts[:, M:]
                if end == t:
                    vote_rows.append(dict(lam=lam, scene=scene, time=t, b=b,
                                          local_vote_A=qa, local_vote_B=qb, reporters=n))
            picked = iid.argmax_random(iid.fwht(hist), rng)
            correct['local' + str(b)][:, index] = picked == target
        for h in HS:
            pmfs = [law.reported_counts(t - h + 1, t) for law in laws]
            counts = rng.choice(len(pmfs[2]), size=(R, M), p=pmfs[2])
            counts[:, A] = rng.choice(len(pmfs[0]), size=R, p=pmfs[0])
            counts[:, B] = rng.choice(len(pmfs[1]), size=R, p=pmfs[1])
            picked = iid.argmax_random(2 * counts - 2 * h, rng, absolute=True)
            correct['server' + str(h)][:, index] = picked == target
        for method in methods:
            x = correct[method][:, index]
            curves.append(dict(lam=lam, scene=scene, time=t, method=method,
                               selected_correct=float(x.mean()), mc_se=float(x.std(ddof=1) / np.sqrt(R))))
    periods = {'stable': np.ones(1, dtype=bool)} if scene == 'constant' else {
        'first8': np.arange(96) < 8, 'first32': np.arange(96) < 32,
        'first96': np.ones(96, dtype=bool), 'last8': np.arange(96) >= 88}
    summary, diff = [], []
    for th in [False, True]:
        r0, rs = risk(0., th), risk(.6, th)
        all_risks = {method: np.where(correct[method], rs, .36 + r0) for method in methods}
        all_risks['zero'] = np.full((R, len(times)), .36)
        for period, mask in periods.items():
            for method, vals in all_risks.items():
                avg = vals[:, mask].mean(axis=1)
                mu, se = float(avg.mean()), float(avg.std(ddof=1) / np.sqrt(R))
                summary.append(dict(lam=lam, scene=scene, period=period, threshold=th, method=method,
                                    mse=mu, mc_se=se, low=mu - 1.96 * se, high=mu + 1.96 * se))
            for local in ['local16', 'local64']:
                for h in HS:
                    reference = 'server' + str(h)
                    avg = (all_risks[local][:, mask] - all_risks[reference][:, mask]).mean(axis=1)
                    mu, se = float(avg.mean()), float(avg.std(ddof=1) / np.sqrt(R))
                    diff.append(dict(lam=lam, scene=scene, period=period, threshold=th, method=local,
                                     reference=reference, difference=mu, mc_se=se, low=mu - 1.96 * se, high=mu + 1.96 * se))
    return summary, diff, curves, vote_rows


def main():
    start = time.monotonic()
    validate()
    summary, differences, curves, votes = [], [], [], []
    for li, lam in enumerate(LAMBDAS):
        for scene in ['constant', 'migration']:
            a, b, c, d = snapshots(lam, li, scene)
            summary.extend(a)
            differences.extend(b)
            curves.extend(c)
            votes.extend(d)
            print('lambda={} {} done; {:.1f}s'.format(lam, scene, time.monotonic() - start), flush=True)
    save('summary.csv', summary)
    save('differences.csv', differences)
    save('time_curves.csv', curves)
    save('local_vote_laws.csv', votes)
    save('effective_samples.csv', [dict(lam=lam, b=b, effective_samples=effective_size(b, lam),
                                      variance_inflation=b / effective_size(b, lam))
                                   for lam in LAMBDAS for b in [16, 64]])
    resources = []
    for b in [16, 64]:
        resources.append(dict(method='local' + str(b), screen_users=N // 2, publication_users=N // 2,
                              screen_stride=W + b - 1, raw_history=b, average_bits_per_time=N / 2 / (W + b - 1) + N / 2 / W))
    for h in HS:
        resources.append(dict(method='server' + str(h), screen_users=N // 2, publication_users=N // 2,
                              screen_stride=W, raw_history=1, average_bits_per_time=N / W))
    save('resource_summary.csv', resources)
    metadata = dict(n=N, candidates=M, w=W, epsilon=EPS, trials_per_snapshot=R, seed=SEED,
                    lambdas=LAMBDAS, amplitude=.6, snapshots='independent across target times; not full trajectories',
                    elapsed_seconds=time.monotonic() - start,
                    null_risk=risk(0., False), signal_risk=risk(.6, False),
                    threshold_null_risk=risk(0., True), threshold_signal_risk=risk(.6, True))
    (OUT / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == '__main__':
    main()
