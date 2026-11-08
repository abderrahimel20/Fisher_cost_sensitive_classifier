"""
Numerical check that the implementation of FCS-C satisfies the model and the
propositions of Section 3 of the paper, on a training split of each dataset.

    python verify_theory.py
"""
import numpy as np
import cvxpy as cp

import fcsc_pipeline as F

TOL = 1e-5
rows = []


def check(ds, name, ok, detail=""):
    rows.append((ds, name, bool(ok), detail))


for ds in ["australian", "german", "lendingclub"]:
    df, y, num, nom, sc = F.load_lendingclub() if ds == "lendingclub" else F.LOADERS[ds]()
    idx = np.random.RandomState(0).permutation(len(y))[:int(0.8 * len(y))]
    pp = F.Preprocessor(num, nom, sc).fit(df.iloc[idx], y[idx])
    X, yy = pp.transform(df.iloc[idx]), y[idx]
    n = len(yy)
    st = F.fisher_stats(X, yy)
    d, S, L, amax = st["d"], st["Sigma"], st["L"], st["alpha_max"]
    cfp, cfn = F.sklearn_costs(yy)
    c = F.cost_weights(yy, cfp, cfn)
    ys = 2 * yy - 1
    P = F.HyperplaneProblem(X, yy, L, c, st=st, cone=True)
    w_lda = np.linalg.solve(S, d)

    def solve(s, a_rel, rho=1.0):
        sol = P.solve(s, np.sqrt(a_rel * amax), rho)
        w, b = sol["w"], sol["b"]
        eps = np.maximum(0.0, 1 - ys * (X @ w - b))          # optimal deviations for (w, b)
        return w, b, eps, sol

    # ---- the programme (P_R) itself
    s, a_rel, rho = 10.0, 0.5, 1.0
    w, b, eps, sol = solve(s, a_rel, rho)
    epsv = P.eps.value
    cone_res = d @ w - np.sqrt(a_rel * amax) * np.sqrt(w @ S @ w)
    check(ds, "(C1)-(C2) deviation constraints", np.min(ys * (X @ w - b) - 1 + epsv) > -TOL)
    check(ds, "(C4) eps >= 0", epsv.min() > -TOL)
    check(ds, "(C3') Fisher cone", cone_res > -TOL, f"residual {cone_res:.2e}")
    CB, CG = (s / n) * rho * cfp, (s / n) * cfn
    obj = CB * epsv[yy == 0].sum() + CG * epsv[yy == 1].sum() + w @ S @ w
    check(ds, "objective = C_B sum eps_B + C_G sum eps_G + w'Sw", abs(obj - P.prob.value) < 1e-6 * max(1, obj),
          f"{obj:.6f} vs {P.prob.value:.6f}")
    check(ds, "C_k = (s/n) rho_k c_k with balanced costs n/(2n_k)",
          np.isclose(cfp, n / (2 * (yy == 0).sum())) and np.isclose(cfn, n / (2 * (yy == 1).sum())))
    check(ds, "||L'w|| = ||Sigma^1/2 w||", np.isclose(np.linalg.norm(L.T @ w), np.sqrt(w @ S @ w)))

    # ---- Proposition 3: J(w) <= alpha_max, and J >= alpha on the feasible set
    J = F.fisher_ratio(w, st)
    check(ds, "Prop 3: J(w) <= alpha_max", J <= amax * (1 + 1e-8), f"J={J:.4f}, amax={amax:.4f}")
    check(ds, "Prop 3: J(w) >= alpha (solution feasible)", J >= a_rel * amax * (1 - 1e-5),
          f"J/amax={J / amax:.4f} >= {a_rel}")

    # ---- Proposition 4: implicit Fisher bound
    eG, eB = epsv[yy == 1].mean(), epsv[yy == 0].mean()
    check(ds, "Prop 4: d'w >= 2 - mean eps_G - mean eps_B", d @ w >= 2 - eG - eB - 1e-6,
          f"{d @ w:.4f} >= {2 - eG - eB:.4f}")
    if eG + eB < 2:
        check(ds, "Prop 4: J >= (2-eG-eB)^2 / w'Sw", J >= (2 - eG - eB) ** 2 / (w @ S @ w) - 1e-6)

    # ---- Proposition 5: bound active iff alpha > J0 (same s, rho)
    w0, *_ = solve(s, 0.0, rho)
    J0 = F.fisher_ratio(w0, st)
    ok5 = True
    msg = []
    for ar in (0.1, 0.3, 0.5, 0.7, 0.9, 0.99):
        wa, *_ = solve(s, ar, rho)
        res = (d @ wa - np.sqrt(ar * amax) * np.sqrt(wa @ S @ wa)) / max(np.linalg.norm(wa), 1e-12)
        active = res < 1e-4
        predicted = ar * amax > J0
        same_as_0 = np.allclose(wa, w0, atol=1e-4 * max(1, np.linalg.norm(w0)))
        ok5 &= (active == predicted) and (predicted or same_as_0)
        msg.append(f"{ar}:{'A' if active else '-'}{'A' if predicted else '-'}")
    check(ds, "Prop 5: active iff alpha > J0 (and w = w0 otherwise)", ok5,
          f"J0/amax={J0 / amax:.3f}  " + " ".join(msg))

    # ---- Proposition 6: cone angle
    u, dh = np.linalg.cholesky(S).T @ w, np.linalg.solve(np.linalg.cholesky(S), d)
    cosang = (u @ dh) / (np.linalg.norm(u) * np.linalg.norm(dh))
    check(ds, "Prop 6: cos angle(Sigma^1/2 w, LDA) >= sqrt(alpha_rel)", cosang >= np.sqrt(a_rel) - 1e-6,
          f"cos={cosang:.4f} >= {np.sqrt(a_rel):.4f}")

    # ---- Proposition 7 (i): alpha -> alpha_max gives the LDA direction
    w1, *_ = solve(s, 0.999999, rho)
    cos1 = (w1 @ w_lda) / (np.linalg.norm(w1) * np.linalg.norm(w_lda))
    check(ds, "Prop 7(i): alpha_rel -> 1 gives the LDA direction", cos1 > 1 - 1e-4, f"cos={cos1:.6f}")

    # ---- Proposition 7 (ii): class-mean constraints give 2 Sigma^-1 d / alpha_max
    wv, bv = cp.Variable(X.shape[1]), cp.Variable()
    mG, mB = X[yy == 1].mean(0), X[yy == 0].mean(0)
    cp.Problem(cp.Minimize(cp.quad_form(wv, cp.psd_wrap(S))),
               [mG @ wv >= bv + 1, mB @ wv <= bv - 1]).solve(solver=cp.CLARABEL)
    check(ds, "Prop 7(ii): mean constraints -> w = 2 Sigma^-1 d / alpha_max",
          np.allclose(wv.value, 2 * w_lda / amax, rtol=1e-4, atol=1e-6))

    # ---- Proposition 2: unique optimal w (second solver gives the same w)
    P2 = F.HyperplaneProblem(X, yy, L, c, st=st, cone=True)
    P2.sB.value, P2.sG.value, P2.sa.value = s * rho, s, np.sqrt(a_rel * amax)
    P2.prob.solve(solver=cp.SCS, eps=1e-9, max_iters=200_000)
    rel = np.linalg.norm(P2.w.value - w) / np.linalg.norm(w)
    check(ds, "Prop 2: unique w (Clarabel vs SCS)", rel < 1e-3, f"relative difference {rel:.1e}")

    # ---- Proposition (non-degeneracy): balanced weights => w* != 0, for every s
    wB = sum(c[yy == 0]) * rho, sum(c[yy == 1])
    norms = [np.linalg.norm(solve(sv, 0.0, 1.0)[0]) for sv in (1e-4, 1e-2, 1.0, 1e2, 1e4)]
    check(ds, "Non-degeneracy: balanced weights => ||w|| > 0 for all s",
          np.isclose(wB[0], wB[1]) and min(norms) > 1e-6,
          f"class weight totals {wB[0]:.1f}/{wB[1]:.1f}; min ||w|| = {min(norms):.2e}")
    # informational: unbalanced weights (rho = 2) can make w = 0 optimal
    w2 = solve(s, 0.0, 2.0)[0]
    rows.append((ds, "  (info) rho = 2, unbalanced weights", True, f"||w|| = {np.linalg.norm(w2):.2e}"))

w = max(len(r[1]) for r in rows)
fails = 0
for ds, name, ok, det in rows:
    fails += not ok
    print(f"{'PASS' if ok else 'FAIL'}  {ds:<12} {name:<{w}}  {det}")
print(f"\n{len(rows) - fails}/{len(rows)} checks passed")
