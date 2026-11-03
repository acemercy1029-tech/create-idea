"""Check exported risks against independently stored selection probabilities."""
import csv
import json
import numpy as np
import run_dependence as d

def read(name):
    with (d.OUT / name).open(encoding='utf-8') as f:
        return list(csv.DictReader(f))

summary, curves = read('summary.csv'), read('time_curves.csv')
assert len(summary) == 480 and len(curves) == 4074
assert len(read('differences.csv')) == 600
assert len(read('local_vote_laws.csv')) == 1164
assert len(read('kernel_checks.csv')) == 86
max_error = 0.
for row in summary:
    method, scene, period = row['method'], row['scene'], row['period']
    if method == 'zero':
        expected = .36
    else:
        selected = [r for r in curves if r['lam'] == row['lam'] and r['scene'] == scene and r['method'] == method]
        if period == 'first8':
            selected = [r for r in selected if int(r['time']) < 168]
        elif period == 'first32':
            selected = [r for r in selected if int(r['time']) < 192]
        elif period == 'last8':
            selected = [r for r in selected if int(r['time']) >= 248]
        assert len(selected) == {'stable': 1, 'first8': 8, 'first32': 32, 'first96': 96, 'last8': 8}[period]
        p = np.mean([float(r['selected_correct']) for r in selected])
        th = row['threshold'] == 'True'
        expected = p * d.risk(.6, th) + (1 - p) * (.36 + d.risk(0., th))
    error = abs(expected - float(row['mse']))
    max_error = max(max_error, error)
    assert error < 1e-12
    assert float(row['low']) <= float(row['mse']) <= float(row['high'])

thin = read('thinning_check.csv')
assert len(thin) == 36
for row in thin:
    lam = float(row['lam'])
    spacing = int(row['spacing'])
    for p in [.5, .8]:
        assert np.max(abs(np.linalg.matrix_power(d.trans(p, p, lam), spacing) - d.trans(p, p, lam ** spacing))) < 1e-12
    assert int(row['span']) == 1 + (int(row['records']) - 1) * spacing
    assert int(row['report_stride']) == d.W + int(row['span']) - 1
    p = float(row['private_correct_probability'])
    th = row['threshold'] == 'True'
    expected = p * d.risk(.6, th) + (1 - p) * (.36 + d.risk(0., th))
    assert abs(expected - float(row['mse'])) < 1e-12

result = dict(summary_rows=len(summary), curve_rows=len(curves), thinning_output_rows=len(thin),
              kernel_checks=86, max_risk_aggregation_error=max_error,
              checks='risk aggregation, zero baseline, output dimensions, thinning transition powers and accounting')
(d.OUT / 'output_checks.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2))
