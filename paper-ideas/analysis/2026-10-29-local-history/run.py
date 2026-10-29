"""A feasibility probe, not a new method or a full paper reproduction.

Private selectors only see simulated randomized reports. Exact latent vote laws
are used for simulation efficiency, never exposed to selectors. See PROTOCOL.md.
"""
from pathlib import Path
import csv
import itertools
import json
import time
import numpy as np
from scipy.stats import binom
from scipy.linalg import hadamard

OUT = Path(__file__).resolve().parent
N, M, W, EPS, T, R, SEED = 20480, 256, 20, 1., 256, 300, 2026102914
START, WARM, EDGE = 96, 64, 37
KAPPA = np.tanh(EPS / 2)
H = hadamard(M).astype(float)
EXTRA = False


def save(name, rows):
    with (OUT / name).open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def fwht(x):
    y = np.array(x, dtype=float, copy=True)
    h = 1
    while h < y.shape[-1]:
        z = y.reshape(y.shape[:-1] + (-1, 2 * h))
        a, b = z[..., :h].copy(), z[..., h:].copy()
        z[..., :h], z[..., h:] = a + b, a - b
        h *= 2
    return y


def abs_pmf(pmf, b):
    ans = np.zeros(b + 1)
    np.add.at(ans, np.abs(2 * np.arange(b + 1) - b), pmf)
    return ans


def winner_prob(b, active, rho, dims=M):
    """Uniform tie breaking against dims-1 iid null absolute scores."""
    null = abs_pmf(binom.pmf(np.arange(b + 1), b, .5), b)
    sig = np.convolve(binom.pmf(np.arange(active + 1), active, (1 + rho) / 2),
                      binom.pmf(np.arange(b - active + 1), b - active, .5))
    sig = abs_pmf(sig, b)
    cdf = np.cumsum(null)
    cdf[-1] = 1.
    weights = np.zeros_like(null)
    valid = null > 0
    # Stable difference of close powers. Direct subtraction loses rare-tail
    # contributions when cdf and cdf-null both round to one (b=64).
    q = np.minimum(1., null[valid] / cdf[valid])
    numerator = np.ones_like(q)
    interior = q < 1
    numerator[interior] = -np.expm1(dims * np.log1p(-q[interior]))
    weights[valid] = cdf[valid] ** (dims - 1) * numerator / (dims * q)
    return float(sig @ weights)


def argmax_random(scores, rng, absolute=False):
    v = np.abs(scores) if absolute else scores
    # Scores are lattice-valued; this only randomizes exact ties at our scales.
    return np.argmax(v + rng.uniform(-1e-12, 1e-12, v.shape), axis=1)


def clipped_rr_risk(rho, n=N // 2 // W):
    x = np.arange(n + 1)
    z = np.clip((2 * x / n - 1) / KAPPA, -1, 1)
    pmf = binom.pmf(x, n, (1 + KAPPA * rho) / 2)
    return float(pmf @ ((z - rho) ** 2))


def moments(rows, method, scene, metric, period, vals):
    val = vals.mean(axis=1)
    se = float(val.std(ddof=1) / np.sqrt(R))
    mu = float(val.mean())
    rows.append(dict(scene=scene, method=method, period=period, metric=metric,
                     mean=mu, mc_se=se, low=mu - 1.96 * se, high=mu + 1.96 * se))


def validate():
    rows = []
    rng = np.random.default_rng(SEED + 100)
    assert np.max(np.abs(H @ H.T - M * np.eye(M))) == 0
    test = rng.normal(size=(3, M))
    assert np.max(np.abs(fwht(test) - test @ H.T)) < 1e-11
    channel_ratio = (1 + KAPPA) / (1 - KAPPA)
    assert abs(np.log(channel_ratio) - EPS) < 1e-12
    for b in [1, 2, 3, 4]:
        dims, rho, active = 3, .6, b
        pmfs = [binom.pmf(np.arange(b + 1), b, .8)] + [binom.pmf(np.arange(b + 1), b, .5)] * 2
        direct = 0.
        for counts in itertools.product(range(b + 1), repeat=dims):
            scores = np.abs(2 * np.array(counts) - b)
            if scores[0] == max(scores):
                direct += np.prod([pmfs[i][counts[i]] for i in range(dims)]) / sum(scores == scores[0])
        exact = winner_prob(b, active, rho, dims)
        assert abs(direct - exact) < 1e-12
        rows.append(dict(kind='small_complete_enumeration', b=b, active=active, rho=rho,
                         exact=exact, numerical=direct, mc_se=0.))
    for b in [16, 64]:
        for active in [0, 8, b]:
            rho, reps, chosen = .6, 50000, 0
            for _ in range(reps // 1000):
                counts = rng.binomial(b, .5, size=(1000, M))
                counts[:, EDGE] = rng.binomial(active, .8, 1000) + rng.binomial(b - active, .5, 1000)
                pick = argmax_random(np.abs(2 * counts - b), rng)
                chosen += np.sum(pick == EDGE)
            exact, mc = winner_prob(b, active, rho), chosen / reps
            se = float(np.sqrt(exact * (1 - exact) / reps))
            assert abs(exact - mc) < 6 * se + 1e-10
            rows.append(dict(kind='direct_local_count_simulation', b=b, active=active, rho=rho,
                             exact=exact, numerical=mc, mc_se=se))
    save('kernel_checks.csv', rows)
    privacy = []
    for b in [16, 64]:
        stride = W + b - 1
        maximum = 0
        # All phases and changed-window offsets; include future reports that
        # read changed old records even if sent after the changed window.
        for phase in range(stride):
            ends = np.arange(-4 * stride + phase, 5 * stride + phase, stride)
            for start in range(2 * stride):
                touched = int(np.sum((ends >= start) & (ends - b + 1 <= start + W - 1)))
                maximum = max(maximum, touched)
        assert maximum == 1
        privacy.append(dict(batch=b, stride=stride, max_affected_reports=maximum,
                            total_epsilon=maximum * EPS, private_bit_epsilon=np.log(channel_ratio)))
    save('privacy_checks.csv', privacy)


def simulate(scene, rho, duration, scene_id):
    rng = np.random.default_rng(np.random.SeedSequence([SEED, scene_id]))
    truth = np.zeros(T)
    if scene == 'constant_06':
        truth[:] = .6
    elif duration:
        truth[START:START + duration] = rho
    raw = rng.binomial(2, (1 + KAPPA * truth[None, :, None] * (np.arange(M) == EDGE)) / 2,
                       size=(R, T, M))
    signed = 2 * raw - 2
    selected = {}
    sums = np.concatenate([np.zeros((R, 1, M)), np.cumsum(signed, axis=1)], axis=1)
    for h in ([4, 8, 16, 32, 64, 128] if EXTRA else [4, 16, 64]):
        inds = np.arange(T)
        vals = (sums[:, inds + 1] - sums[:, np.maximum(0, inds + 1 - h)])
        picks = np.stack([argmax_random(vals[:, t], rng, True) for t in range(T)], axis=1)
        selected['server_h' + str(h)] = picks
    local_curves = []
    for b in [16, 64]:
        stride = W + b - 1
        history = np.zeros((R, T, M))
        ns = np.zeros(T, dtype=int)
        for t in range(b - 1, T):
            n = N // 2 // stride + int(t % stride < (N // 2) % stride)
            ns[t] = n
            window = truth[t - b + 1:t + 1]
            active = int(np.count_nonzero(window))
            amp = float(np.max(window))
            correct = winner_prob(b, active, amp)
            vote = np.full(M, (1 - correct) / (M - 1))
            vote[EDGE] = correct
            # Public row uniform on [M]. Each user sends one RR bit of H[row,vote].
            row_mean = H @ vote
            probs = np.concatenate([(1 + KAPPA * row_mean) / (2 * M),
                                    (1 - KAPPA * row_mean) / (2 * M)])
            probs /= probs.sum()
            counts = rng.multinomial(n, probs, size=R)
            history[:, t] = fwht(counts[:, :M] - counts[:, M:])
            local_curves.append(dict(scene=scene, batch=b, time=t, active_in_batch=active,
                                     raw_local_correct_probability=correct, reporters=n))
        accum = np.concatenate([np.zeros((R, 1, M)), np.cumsum(history, axis=1)], axis=1)
        for g in [1, 4]:
            picks = np.zeros((R, T), dtype=int)
            for t in range(T):
                vals = accum[:, t + 1] - accum[:, max(0, t + 1 - g)]
                picks[:, t] = argmax_random(vals, rng)
            selected['local_b%d_g%d' % (b, g)] = picks
    risk0 = clipped_rr_risk(0.)
    risk_signal = np.array([clipped_rr_risk(x) for x in truth])
    risks, active_risks, choices = {}, {}, {}
    for method, pick in selected.items():
        correct = pick == EDGE
        # Exact expectation over the independent second-stage RR release.
        risks[method] = np.where(correct, risk_signal, risk0 + truth ** 2)
        active_risks[method] = np.where(correct, risk_signal, truth ** 2)
        choices[method] = correct.astype(float)
    raw_all = rng.binomial(4, (1 + KAPPA * truth[None, :, None] * (np.arange(M) == EDGE)) / 2,
                           size=(R, T, M))
    if EXTRA:
        amplitudes = np.array([-.6, -.4, -.2, .2, .4, .6])
        logplus, logminus = np.log1p(KAPPA * amplitudes), np.log1p(-KAPPA * amplitudes)
        weights = np.full((R, M, len(amplitudes)), .5 / (M * len(amplitudes)))
        none = np.full(R, .5)
        risk, edge_risk, choice = [], [], []
        for t in range(T):
            if t:
                prev_none = none.copy()
                none = none * (1 - 1 / 64) + weights.sum(axis=(1, 2)) / 64
                weights = weights * (1 - 1 / 64) + prev_none[:, None, None] / (64 * M * len(amplitudes))
            x = raw_all[:, t, :, None]
            relative_likelihood = np.exp(x * logplus + (4 - x) * logminus)
            weights *= relative_likelihood
            norm = none + weights.sum(axis=(1, 2))
            weights /= norm[:, None, None]
            none /= norm
            estimate = np.sum(weights * amplitudes, axis=2)
            errors = estimate.copy()
            errors[:, EDGE] -= truth[t]
            risk.append(np.sum(errors ** 2, axis=1))
            edge_risk.append(errors[:, EDGE] ** 2)
            choice.append(argmax_random(np.sum(weights, axis=2), rng) == EDGE)
        risks['all_rotation_hmm'] = np.array(risk).T
        active_risks['all_rotation_hmm'] = np.array(edge_risk).T
        choices['all_rotation_hmm'] = np.array(choice, dtype=float).T
    sums_all = np.concatenate([np.zeros((R, 1, M)), np.cumsum(2 * raw_all - 4, axis=1)], axis=1)
    for h in [4, 16, 64]:
        method = 'all_rotation_h' + str(h)
        risk, active_risk, choice = [], [], []
        for t in range(T):
            vals = (sums_all[:, t + 1] - sums_all[:, max(0, t + 1 - h)]) / (4 * min(t + 1, h) * KAPPA)
            pick = argmax_random(vals, rng, True)
            value = np.clip(vals[np.arange(R), pick], -1, 1)
            correct = pick == EDGE
            risk.append(np.where(correct, (value - truth[t]) ** 2, value ** 2 + truth[t] ** 2))
            active_risk.append(np.where(correct, (value - truth[t]) ** 2, truth[t] ** 2))
            choice.append(correct)
        risks[method], active_risks[method], choices[method] = np.array(risk).T, np.array(active_risk).T, np.array(choice, dtype=float).T
    risks['zero'] = np.broadcast_to(truth ** 2, (R, T))
    active_risks['zero'] = risks['zero']
    choices['zero'] = np.zeros((R, T))
    periods = {'evaluation': np.arange(WARM, T)}
    if duration:
        periods.update(active=np.arange(START, START + duration), first8=np.arange(START, START + min(8, duration)),
                       last8=np.arange(START + duration - min(8, duration), START + duration),
                       after32=np.arange(START + duration, min(T, START + duration + 32)))
    rows, diffs = [], []
    for method in risks:
        for period, ix in periods.items():
            for name, values in [('l2_mse', risks[method]), ('edge_mse', active_risks[method]), ('support_accuracy', choices[method])]:
                moments(rows, method, scene, name, period, values[:, ix])
    # Predeclared primary comparisons of the b64/g4 probe against every server window.
    for period, ix in periods.items():
        baselines = ['server_h4', 'server_h16', 'server_h64', 'all_rotation_h4', 'all_rotation_h16', 'all_rotation_h64']
        if EXTRA:
            baselines += ['server_h8', 'server_h32', 'server_h128', 'all_rotation_hmm', 'zero']
        for candidate in (['local_b16_g4', 'local_b64_g4'] if EXTRA else ['local_b64_g4']):
            for baseline in baselines:
                vals = (risks[candidate][:, ix] - risks[baseline][:, ix]).mean(axis=1)
                se = float(vals.std(ddof=1) / np.sqrt(R))
                diffs.append(dict(scene=scene, period=period, candidate=candidate, baseline=baseline,
                                  difference=float(vals.mean()), mc_se=se, low=float(vals.mean()-1.96*se), high=float(vals.mean()+1.96*se)))
    traces = []
    for method in risks:
        for t in range(T):
            traces.append(dict(scene=scene, method=method, time=t, truth=truth[t],
                               l2_mse=float(risks[method][:, t].mean()),
                               edge_mse=float(active_risks[method][:, t].mean()),
                               support_accuracy=float(choices[method][:, t].mean())))
    return rows, diffs, traces, local_curves


def main():
    start = time.time()
    validate()
    scenarios = [('null', 0., 0), ('constant_06', .6, 0)] + [
        ('amp%s_len%d' % (str(a).replace('.', ''), d), a, d) for a in [.2, .6] for d in [8, 32, 96]]
    summaries, differences, traces, curves = [], [], [], []
    for i, (scene, rho, length) in enumerate(scenarios):
        a, b, c, d = simulate(scene, rho, length, i)
        summaries.extend(a); differences.extend(b); traces.extend(c); curves.extend(d)
        print('completed', scene, flush=True)
    save('summary.csv', summaries)
    save('differences.csv', differences)
    save('time_curves.csv', traces)
    save('local_vote_probabilities.csv', curves)
    (OUT / 'metadata.json').write_text(json.dumps(dict(seed=SEED, repetitions=R, users=N,
        pairs=M, epsilon=EPS, privacy_window=W, timestamps=T, warmup=WARM,
        scenes=len(scenarios), methods=15 if EXTRA else 11, stronger_baselines=EXTRA, seconds=time.time()-start,
        conditional_publication_risk='exact binomial; selector risk Monte Carlo',
        limitations=['star workload', 'independent times/users', 'at most one active coordinate',
                     'not a full paper implementation', 'no novelty claim']), indent=2), encoding='utf-8')


if __name__ == '__main__':
    import sys
    if '--stronger' in sys.argv:
        OUT = OUT / 'stronger-check'
        OUT.mkdir(exist_ok=True)
        SEED, R, EXTRA = 2026102915, 500, True
    main()
