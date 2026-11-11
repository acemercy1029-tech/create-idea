"""Duration sweep for threshold voting and top1 under identical report schedules."""
from pathlib import Path
from functools import lru_cache
import csv
import json
import sys
import numpy as np
from scipy.stats import binom

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT.parent / '2026-11-04-information'))
import run_information as previous

model = previous.d
M = model.M
A, B = model.A, model.B
H = model.H
DURATIONS = (8, 12, 16, 24, 32, 48, 64)
HISTORIES = (4, 8, 16, 32, 64)
WINDOWS = (4, 8, 16)
THRESHOLDS = (0.5, 0.75, 1.0)
LAMBDAS = (0., 0.6)
ONSET = 160
right_loss = float(previous.g.single_risk(True, 512, True))
wrong_loss = float(previous.g.single_risk(False, 512, True))


def write_rows(filename, rows):
    with (OUT / filename).open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


@lru_cache(None)
def laws(b, lam, end):
    times = np.arange(end - b + 1, end + 1)
    old = tuple(np.where((96 <= times) & (times < ONSET), .8, .5))
    new = tuple(np.where(times >= ONSET, .8, .5))
    return tuple(model.raw_counts(sequence, lam) for sequence in (old, new, (.5,) * b))


@lru_cache(None)
def votes(b, lam, end):
    old, new, _ = laws(b, lam, end)
    qa, qb = model.winners(old, new, lam)
    return qa, qb, (1 - qa - qb) / (M - 2)


@lru_cache(None)
def background_signs(positive, negative, prob):
    plus = binom.pmf(np.arange(positive + 1), positive, prob)
    minus = binom.pmf(np.arange(negative + 1), negative, prob)
    pmf = np.convolve(plus, minus[::-1])
    grid = np.arange(len(pmf)) - negative
    return {offset: float(np.sign(grid + offset) @ pmf) for offset in range(-2, 3)}


def threshold_means(b, lam, t, cut):
    counts = laws(b, lam, t)
    passed = abs(2 * np.arange(b + 1) - b) >= cut * b
    pa, pb, null = [float(np.clip(p[passed].sum(), 0, 1)) for p in counts]
    means = np.empty(M)
    for row_sum, sign_a, sign_b in ROW_GROUPS:
        mask = ((ROW_SUMS == row_sum) & (H[:, A] == sign_a) & (H[:, B] == sign_b))
        positive = (M + row_sum) // 2 - int(sign_a == 1) - int(sign_b == 1)
        expected = background_signs(positive, M - 2 - positive, null)
        means[mask] = ((1-pa)*(1-pb)*expected[0] + pa*(1-pb)*expected[sign_a]
                       + (1-pa)*pb*expected[sign_b] + pa*pb*expected[sign_a+sign_b])
    return means


ROW_SUMS = H.sum(axis=1).astype(int)
ROW_GROUPS = set(zip(ROW_SUMS, H[:, A].astype(int), H[:, B].astype(int)))


@lru_cache(None)
def report_law(b, lam, t, cut):
    if cut == 0:
        qa, qb, qnull = votes(b, lam, t)
        means = qnull*ROW_SUMS + (qa-qnull)*H[:, A] + (qb-qnull)*H[:, B]
    else:
        means = threshold_means(b, lam, t, cut)
    assert np.max(np.abs(means)) <= 1 + 1e-11
    probabilities = np.r_[1+model.KAPPA*means, 1-model.KAPPA*means] / (2*M)
    return probabilities / probabilities.sum()


def risk_curves(b, lam, cut, windows, repeats, seed, durations=DURATIONS):
    rng = np.random.default_rng(np.random.SeedSequence([seed, b, int(lam*10), int(100*cut)]))
    ring = []
    totals = {h: np.zeros(repeats) for h in windows}
    sums = {h: np.zeros((repeats, M), dtype=np.int32) for h in windows}
    rows = []
    for t in range(ONSET-max(windows)+1, ONSET+max(durations)):
        n = previous.nphase(b, t)
        counts = rng.multinomial(n, report_law(b, lam, t, cut), size=repeats)
        # Transform each day's row counts once. Linearity gives every g-window score.
        scores = model.iid.fwht(counts[:, :M] - counts[:, M:]).astype(np.int32)
        for h in windows:
            sums[h] += scores
            if len(ring) >= h:
                sums[h] -= ring[-h]
            if t < ONSET:
                continue
            # Integrate uniform ties exactly, without adding random tie noise.
            values = sums[h]
            at_max = values == values.max(axis=1, keepdims=True)
            success = at_max[:, B] / at_max.sum(axis=1)
            totals[h] += success
            duration = t-ONSET+1
            if duration in durations:
                prob = totals[h]/duration
                risks = wrong_loss+(right_loss-wrong_loss)*prob
                rows.append(dict(b=b, g=h, cut=cut, lam=lam, duration=duration,
                                 success=float(prob.mean()), risk=float(risks.mean()),
                                 mc_se=float(risks.std(ddof=1)/np.sqrt(repeats))))
        ring.append(scores)
        if len(ring) > max(windows):
            ring.pop(0)
    return rows


def information_bounds():
    rows = []
    excess = np.expm1(model.EPS)
    old_bounds = list(csv.DictReader((previous.OUT / 'information_bounds.csv').open()))
    for lam in LAMBDAS:
        for b in HISTORIES:
            cumulative = 0.
            lower_sum = 0.
            horizon = model.W + b - 1
            for duration in range(1, max(DURATIONS)+1):
                t = ONSET+duration-1
                qa, qb, null = votes(b, lam, t)
                term = (qb-null)**2 * excess**2 / (4*(1+(1-qa)*excess))
                mean = (1-qa)/(M-1)
                plain = max(0., qb*np.log(qb/mean) + (M-2)*null*np.log(null/mean))
                cumulative += previous.nphase(b, t)*min(term, plain)
                valid = lam == 0 or duration <= horizon
                bound = previous.success_bound(cumulative) if valid else 1.
                lower_sum += .36*(1-bound)
                if duration == 8 and b != 32:
                    prior = next(float(x['success_upper']) for x in old_bounds if float(x['lam'])==lam
                                 and int(x['b'])==b and x['t']=='167' and x['channel']=='noninteractive_index_ldp')
                    assert bound <= prior + 1e-10 and abs(bound-prior) < 1e-5
                if duration in DURATIONS:
                    rows.append(dict(b=b, lam=lam, duration=duration,
                                     event_risk_lower=lower_sum/duration, last_success_upper=bound,
                                     scope='all_screening' if lam==0 else 'event_transcript_only'))
    return rows


def ranking(rows, family, history=None):
    grouped = {}
    for row in rows:
        if (row['cut'] > 0) != (family=='threshold') or history not in (None, row['b']):
            continue
        key = row['b'], row['cut'], row['g']
        grouped.setdefault(key, []).append(row['risk'])
    return min(grouped, key=lambda key: (np.mean(grouped[key]), key))


def validate():
    # Direct local simulation for all 256 attributes, checking row symmetries and null interference.
    rng = np.random.default_rng(2026111199)
    errors = []
    for b, lam, t, cut in [(8, 0., 167, 1.), (16, .6, 166, .75), (32, 0., 180, .5)]:
        old, new, null = laws(b, lam, t)
        mu = threshold_means(b, lam, t, cut)
        row_ids = [int(np.flatnonzero((ROW_SUMS==s)&(H[:,A]==sa)&(H[:,B]==sb))[0])
                   for s,sa,sb in sorted(ROW_GROUPS)]
        observations = []
        for _ in range(32):
            count = rng.choice(b+1, size=(1024,M), p=null)
            count[:,A] = rng.choice(b+1, size=1024, p=old)
            count[:,B] = rng.choice(b+1, size=1024, p=new)
            active = abs(2*count-b) >= b*cut
            observations.append(np.stack([np.sign((active*H[r]).sum(axis=1)) for r in row_ids],axis=1))
        simulated = np.concatenate(observations)
        error = np.abs(simulated.mean(axis=0)-mu[row_ids])
        assert (error < 6/np.sqrt(len(simulated))).all()
        errors.append(float(error.max()))
    for b in HISTORIES:
        stride=model.W+b-1
        assert sum(previous.nphase(b, t) for t in range(stride))==model.N//2
        assert max(previous.nphase(b, t)+512 for t in range(stride)) <= 1024
        assert max(WINDOWS) <= stride
    return errors


def main():
    validation = validate()
    bounds = information_bounds()
    write_rows('bounds.csv', bounds)
    pilot = []
    for b in HISTORIES:
        for lam in LAMBDAS:
            for cut in (0.,)+THRESHOLDS:
                pilot.extend(risk_curves(b,lam,cut,WINDOWS,384,2026111120))
        print('Pilot history', b, flush=True)
    write_rows('pilot.csv', pilot)
    top = ranking(pilot, 'top')
    threshold = ranking(pilot, 'threshold')
    schedule = {top, threshold}
    for b in HISTORIES:
        cfg = ranking(pilot, 'threshold', b)
        schedule.update([cfg, (b,0.,cfg[2])])
    frozen = dict(top=top, threshold=threshold, confirm=sorted(schedule))
    (OUT/'frozen.json').write_text(json.dumps(frozen,indent=2))
    print('Frozen',frozen,flush=True)
    confirmed = []
    for b, cut in sorted({(b,c) for b,c,h in schedule}):
        windows = sorted(h for bb,c,h in schedule if bb==b and c==cut)
        for lam in LAMBDAS:
            confirmed.extend(risk_curves(b,lam,cut,windows,3072,2026111121))
        print('Confirmed',b,cut,flush=True)
    # Simultaneous two-sided Hoeffding bands across the independently frozen grid.
    margin=(wrong_loss-right_loss)*np.sqrt(np.log(2*len(confirmed)/.05)/(2*3072))
    for row in confirmed:
        row.update(low=max(right_loss,row['risk']-margin), high=min(wrong_loss,row['risk']+margin))
    write_rows('confirmation.csv',confirmed)
    (OUT/'checks.json').write_text(json.dumps(dict(validation_errors=validation,
        simultaneous_margin=margin, right_loss=right_loss, wrong_loss=wrong_loss,
        pilot_rows=len(pilot), confirmed_rows=len(confirmed)),indent=2))


if __name__ == '__main__':
    main()
