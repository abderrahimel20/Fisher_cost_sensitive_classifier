"""
Clean execution-time measurement. For every dataset and method, the configuration
selected most often in the cross-validation is refitted on 80% of the data and
scored on the remaining 20%, alone on the machine and single-threaded, `reps`
times; the median fit and prediction times are reported.

    python time_methods.py results_compact
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

import fcsc_pipeline as F

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, sys.argv[1] if len(sys.argv) > 1 else "results")
GRID = sys.argv[2] if len(sys.argv) > 2 else "compact"
REPS = 3
rows = []
for ds in ["australian", "german", "lendingclub"]:
    folds = pd.read_csv(os.path.join(RES, f"{ds}_folds.csv"))
    df, y, num, nom, sc = F.load_lendingclub() if ds == "lendingclub" else F.LOADERS[ds]()
    idx = np.random.RandomState(0).permutation(len(y))
    tr, te = idx[:int(0.8 * len(y))], idx[int(0.8 * len(y)):]
    pp = F.Preprocessor(num, nom, sc).fit(df.iloc[tr], y[tr])
    X, Xt, yy = pp.transform(df.iloc[tr]), pp.transform(df.iloc[te]), y[tr]
    cfp, cfn = F.sklearn_costs(yy)
    M = F.build_methods(cfp, cfn, F.GRIDS[GRID])
    st = F.fisher_stats(X, yy)
    for name, m in M.items():
        sel = folds[folds.method == name]
        if sel.empty:
            continue
        p = json.loads(sel.params.mode().iloc[0])
        fit_t, pred_t = [], []
        for _ in range(REPS):
            ctx = {"st": st}                       # fresh: no cached solution
            t = time.perf_counter()
            model = m["fit"](X, yy, p, ctx)
            fit_t.append(time.perf_counter() - t)
            t = time.perf_counter()
            model.score(Xt)
            pred_t.append(time.perf_counter() - t)
        rows.append(dict(dataset=ds, method=name, params=json.dumps(p), n_train=len(yy),
                         fit_seconds=float(np.median(fit_t)),
                         predict_ms_per_1000=1e6 * float(np.median(pred_t)) / len(te)))
        print(f"{ds:<12} {name:<20} fit {rows[-1]['fit_seconds']:8.3f} s   "
              f"predict {rows[-1]['predict_ms_per_1000']:7.3f} ms/1000", flush=True)
pd.DataFrame(rows).to_csv(os.path.join(RES, "timing_clean.csv"), index=False)
print("wrote", os.path.join(RES, "timing_clean.csv"))
