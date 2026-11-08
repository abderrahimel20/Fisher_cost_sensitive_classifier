"""Check that reusing the alpha = 0 solution when alpha <= J0 (Proposition 5) gives
the same test AUC and cost as solving (P_R) directly, over the whole FCS-C grid."""
import numpy as np

import fcsc_pipeline as F

for ds in ["australian", "german", "lendingclub"]:
    df, y, num, nom, sc = F.load_lendingclub() if ds == "lendingclub" else F.LOADERS[ds]()
    idx = np.random.RandomState(0).permutation(len(y))
    tr, te = idx[:int(0.8 * len(y))], idx[int(0.8 * len(y)):]
    pp = F.Preprocessor(num, nom, sc).fit(df.iloc[tr], y[tr])
    X, Xt, yy, yt = pp.transform(df.iloc[tr]), pp.transform(df.iloc[te]), y[tr], y[te]
    cfp, cfn = F.sklearn_costs(yy)
    M = F.build_methods(cfp, cfn, F.GRIDS["compact"])
    st = F.fisher_stats(X, yy)
    ctx = {"st": st}
    d_auc, d_cost, d_cos, n_reused, n_none = 0.0, 0.0, 0.0, 0, 0
    for p in M["FCS-C"]["grid"]:
        m = M["FCS-C"]["fit"](X, yy, p, ctx)
        ref = ctx["fcsc"].solve(p["s"], np.sqrt(p["alpha_rel"] * st["alpha_max"]))
        if m is None or ref is None or np.linalg.norm(ref["w"]) < F.DEGENERATE_W:
            n_none += 1
            continue
        n_reused += bool(m.info.get("reused_alpha0"))
        sr = Xt @ ref["w"] - ref["b"]
        sm = m.score(Xt)
        d_auc = max(d_auc, abs(F.safe_auc(yt, sm) - F.safe_auc(yt, sr)))
        d_cost = max(d_cost, abs(F.op_metrics(yt, sm, 0, cfp, cfn)["cost"] - F.op_metrics(yt, sr, 0, cfp, cfn)["cost"]))
        d_cos = max(d_cos, 1 - m.w @ ref["w"] / (np.linalg.norm(m.w) * np.linalg.norm(ref["w"])))
    print(f"{ds:<12} configs {len(M['FCS-C']['grid'])}, reused {n_reused}, excluded {n_none}: "
          f"max |dAUC| {d_auc:.1e}, max |dcost| {d_cost:.1e}, max (1 - cos) {d_cos:.1e}")
