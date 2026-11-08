"""Estimate the run time of a grid profile: time a random sample of configurations
of every method on an inner-fold-sized training set and extrapolate to the full
nested protocol.   python bench_grid.py --dataset lendingclub --grid wide"""
import argparse
import os
import time

import numpy as np

import fcsc_pipeline as F

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="lendingclub")
ap.add_argument("--grid", default="wide")
ap.add_argument("--sample", type=int, default=6)
ap.add_argument("--inner", type=int, default=5)
ap.add_argument("--outer", type=int, default=25)
ap.add_argument("--jobs", type=int, default=18)
a = ap.parse_args()

df, y, num, nom, sc = F.LOADERS[a.dataset]() if a.dataset != "lendingclub" else F.load_lendingclub()
n_tr = int(len(y) * 0.8 * (a.inner - 1) / a.inner)            # inner-training size
idx = np.random.RandomState(0).permutation(len(y))[:n_tr]
pp = F.Preprocessor(num, nom, sc).fit(df.iloc[idx], y[idx])
X, yy = pp.transform(df.iloc[idx]), y[idx]
cfp, cfn = F.sklearn_costs(yy)
M = F.build_methods(cfp, cfn, F.GRIDS[a.grid])
rng = np.random.RandomState(1)
total = 0.0
print(f"{a.dataset}, grid '{a.grid}', inner-train n={n_tr}")
for name, m in M.items():
    ctx = {"st": F.fisher_stats(X, yy)}
    g = m["grid"]
    pick = [g[i] for i in rng.choice(len(g), min(a.sample, len(g)), replace=False)]
    t0 = time.time()
    for p in pick:
        m["fit"](X, yy, p, ctx)
    per = (time.time() - t0) / len(pick)
    fold = per * len(g) * a.inner + per * 2           # inner CV + refit (+ diagnostics)
    if name == "FCS-C":
        fold += per * 8                               # J0 and alpha curve
    total += fold
    print(f"  {name:<20} {len(g):>4} configs  {per:6.2f} s/fit  ->  {fold / 60:6.1f} min per outer fold")
wall = total * np.ceil(a.outer / a.jobs)
print(f"per outer fold: {total / 60:.1f} min;  {a.outer} folds on {a.jobs} workers: ~{wall / 60:.0f} min")
