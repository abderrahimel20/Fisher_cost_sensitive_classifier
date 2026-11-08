"""
Build results_final/: baselines and the alpha = 0 ablation from results_compact/,
FCS-C (alpha_rel > 0) with its diagnostics and sensitivity curve from
results_compact_fcsc/ (same outer folds), then recompute the significance tests
and the clean timing of FCS-C.

    python merge_results.py
"""
import os
import shutil

import pandas as pd

import fcsc_pipeline as F

HERE = os.path.dirname(os.path.abspath(__file__))
BASE, FCSC, OUT = (os.path.join(HERE, d) for d in ("results_compact", "results_compact_fcsc", "results_final"))
os.makedirs(OUT, exist_ok=True)
SIZES = {"australian": (552, 138), "german": (800, 200), "lendingclub": (16000, 4000)}

for ds, (n_tr, n_te) in SIZES.items():
    base = pd.read_csv(os.path.join(BASE, f"{ds}_folds.csv"))
    new = pd.read_csv(os.path.join(FCSC, f"{ds}_folds.csv"))
    assert set(new.method) == {"FCS-C"} and len(new) == 25
    keys = ["rep", "fold"]
    assert (new.sort_values(keys)[keys].values == base[base.method == "FCS-C"].sort_values(keys)[keys].values).all()
    res = pd.concat([new, base[base.method != "FCS-C"]], ignore_index=True)
    res.to_csv(os.path.join(OUT, f"{ds}_folds.csv"), index=False)
    for k in ("fcsc_diagnostics", "alpha_curve"):
        shutil.copy(os.path.join(FCSC, f"{ds}_{k}.csv"), os.path.join(OUT, f"{ds}_{k}.csv"))
    F.summarise(res, ds, n_tr, n_te, OUT)
    print(ds, "merged:", res.method.nunique(), "methods,", len(res), "rows")

# clean timing: baselines from the compact run, FCS-C re-measured with its alpha > 0 configuration
t = pd.read_csv(os.path.join(BASE, "timing_clean.csv"))
t = t[t.method != "FCS-C"]
t.to_csv(os.path.join(OUT, "timing_clean.csv"), index=False)
print("wrote", OUT)
