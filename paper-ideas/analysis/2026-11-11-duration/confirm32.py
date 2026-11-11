"""Fresh confirmation of the two globally selected schemes at D=32, lambda=0."""
import math
import json
import numpy as np
from duration_sweep import OUT, risk_curves, write_rows, right_loss, wrong_loss

def main():
    frozen = json.loads((OUT/'frozen.json').read_text())
    rows = []
    batch_size, batches = 1024, 40
    repeats = batch_size*batches
    half = (wrong_loss-right_loss)*math.sqrt(math.log(2*2/.05)/(2*repeats))
    for family in ('top','threshold'):
        b,cut,g = frozen[family]
        means = []
        for batch in range(batches):
            result = risk_curves(b,0.,cut,[g],batch_size,2026112000+batch,durations=(32,))
            means.append(result[0]['risk'])
        risk=float(np.mean(means))
        rows.append(dict(family=family,b=b,g=g,cut=cut,lam=0,duration=32,risk=risk,
                         low=risk-half,high=risk+half,repeats=repeats))
        print(rows[-1], flush=True)
    write_rows('confirmation32.csv',rows)

if __name__=='__main__':
    main()
