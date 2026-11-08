"""
FCS-C revision pipeline -- Fisher Cost-Sensitive Classifier.

Implements the revised model (P_R) of the paper:

    min  sum_i C_i eps_i + w' Sigma w
    s.t. y_i (w'x_i - b) >= 1 - eps_i,   eps_i >= 0        (C1)-(C2), (C4)
         d'w >= sqrt(alpha) ||Sigma^{1/2} w||_2             (C3')

with C_i = (s/n) * c_k, where c_k is the misclassification cost of class k given by
the scikit-learn balanced class-weight formula c_k = n / (2 n_k), computed on each
training fold (c_FP = n/(2 n_Bad), c_FN = n/(2 n_Good)); s = tuned scale;
Good = +1 = positive class.

Evaluation: repeated stratified K-fold (outer) with hyper-parameters tuned by
inner stratified K-fold on the outer training part only. Every method sees the
same folds and the same dataset cost matrix. Reported per outer fold: AUC,
realized cost per applicant and threshold metrics, at the native operating
point and at a cost-optimal threshold chosen on inner out-of-fold scores.

Usage:
    python fcsc_pipeline.py --dataset australian
    python fcsc_pipeline.py --dataset lendingclub --repeats 2 --inner 3
"""

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "RAYON_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")      # one thread per worker process
import argparse
import itertools
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
import cvxpy as cp
from joblib import Parallel, delayed
from scipy import stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
SEED = 42

def _data_path(env, filename, fallback):
    """Dataset location: environment variable `env` if set, else data/<filename>
    next to this script if it exists, else the relative path data/<filename>."""
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", filename)
    return os.environ.get(env) or (here if os.path.exists(here) else fallback)


DATA_PATHS = {
    "australian":  _data_path("FCSC_AUSTRALIAN", "australian.dat",
                              "data/australian.dat"),
    "german":      _data_path("FCSC_GERMAN", "german.data",
                              "data/german.data"),
    "lendingclub": _data_path("FCSC_LENDINGCLUB", "accepted_2007_to_2018Q4.csv",
                              "data/accepted_2007_to_2018Q4.csv"),
}
def sklearn_costs(y):
    """Misclassification costs from the scikit-learn balanced class-weight formula
    c_k = n / (2 n_k):  c_FP = cost of accepting a Bad case (class 0),
    c_FN = cost of rejecting a Good case (class 1)."""
    from sklearn.utils.class_weight import compute_class_weight
    w = compute_class_weight("balanced", classes=np.array([0, 1]), y=y)
    return float(w[0]), float(w[1])
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_cache")


# =============================================================================
# 1. DATA
# =============================================================================
def load_australian():
    df = pd.read_csv(DATA_PATHS["australian"], sep=r"\s+", header=None)
    df.columns = [f"A{i}" for i in range(1, 15)] + ["target"]
    y = df.pop("target").to_numpy().astype(int)          # 1 = Good, 0 = Bad
    nominal = ["A4", "A5", "A6", "A12"]                  # multi-level categorical
    numeric = [c for c in df.columns if c not in nominal]
    return df, y, numeric, nominal, None


def load_german():
    cols = ["checking_acc", "duration", "credit_history", "purpose", "credit_amount",
            "savings_acc", "employment", "installment_rate", "personal_status",
            "other_debtors", "residence_since", "property", "age", "other_installment",
            "housing", "existing_credits", "job", "num_dependents", "own_telephone",
            "foreign_worker", "class"]
    df = pd.read_csv(DATA_PATHS["german"], sep=" ", header=None, names=cols)
    y = (df.pop("class") == 1).astype(int).to_numpy()   # UCI 1 = Good, 2 = Bad
    nominal = [c for c in df.columns if df[c].dtype == object]
    numeric = [c for c in df.columns if c not in nominal]
    return df, y, numeric, nominal, None


def load_lendingclub(n_sample=20_000):
    """Accepted-loans file only, polars pipeline of the original notebook
    (column/row pushdown, FICO mid-range, issue year, drop_nulls), followed by
    a stratified random sample of n_sample loans. Fixes w.r.t. the notebook:
    emp_length -> numeric years (0 for '< 1 year', 10 for '10+ years') instead
    of alphabetical label codes; purpose and state are encoded fold-safely in
    Preprocessor (one-hot / default-rate tertiles)."""
    import polars as pl
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, f"lendingclub_{n_sample}.parquet")
    if not os.path.exists(cache):
        raw = ["loan_amnt", "issue_d", "purpose", "fico_range_low", "fico_range_high",
               "dti", "emp_length", "addr_state", "loan_status"]
        full = (
            pl.scan_csv(DATA_PATHS["lendingclub"], infer_schema_length=10_000,
                        ignore_errors=True)
              .select(raw)                                                   # column pushdown
              .filter(pl.col("loan_status").is_in(["Fully Paid", "Charged Off"]))  # row pushdown
              .with_columns([
                  (0.5 * pl.col("fico_range_low") + 0.5 * pl.col("fico_range_high")).alias("fico"),
                  (pl.col("loan_status") == "Fully Paid").cast(pl.Int64).alias("target"),
                  pl.col("issue_d").str.extract(r"-(\d{4})", 1).cast(pl.Int64).alias("issue_year"),
                  pl.when(pl.col("emp_length").str.starts_with("<")).then(pl.lit("0"))
                    .otherwise(pl.col("emp_length").str.extract(r"(\d+)", 1))
                    .cast(pl.Float64).alias("emp_years"),
              ])
              .drop(["fico_range_low", "fico_range_high", "loan_status", "issue_d", "emp_length"])
              .drop_nulls()
              .collect()
        )
        print(f"   LendingClub accepted after filtering: {full.shape}", flush=True)
        # stratified random sample (class proportions preserved)
        parts = []
        for cls, frac in full["target"].value_counts(normalize=True).iter_rows():
            k = int(round(n_sample * frac))
            parts.append(full.filter(pl.col("target") == cls).sample(n=k, seed=SEED))
        pl.concat(parts).sample(fraction=1.0, shuffle=True, seed=SEED).write_parquet(cache)
    df = pl.read_parquet(cache).to_pandas()
    y = df.pop("target").to_numpy().astype(int)        # 1 = Fully Paid (Good)
    numeric = ["loan_amnt", "fico", "dti", "emp_years"]
    nominal = ["purpose"]
    return df[numeric + nominal + ["addr_state"]], y, numeric, nominal, "addr_state"


LOADERS = {"australian": load_australian, "german": load_german,
           "lendingclub": load_lendingclub}


class Preprocessor:
    """Fold-safe preprocessing: one-hot (drop first) for nominal variables,
    default-rate tertile dummies for the state column (two dummies, lowest
    tertile as reference), standardisation of all resulting columns."""

    def __init__(self, numeric, nominal, state_col=None):
        self.numeric, self.nominal, self.state_col = numeric, nominal, state_col

    def fit(self, df, y):
        self.ohe = None
        if self.nominal:
            self.ohe = OneHotEncoder(drop="first", handle_unknown="ignore",
                                     min_frequency=0.01, sparse_output=False)
            self.ohe.fit(df[self.nominal].astype(str))
        if self.state_col:
            rate = (1 - pd.Series(y, index=df.index)).groupby(df[self.state_col]).mean()
            order = rate.sort_values().index.to_list()      # low -> high default rate
            k = len(order)
            self.state_group = {s: (0 if i < k // 3 else 1 if i < 2 * k // 3 else 2)
                                for i, s in enumerate(order)}
        self.scaler = StandardScaler().fit(self._raw(df))
        return self

    def _raw(self, df):
        parts = [df[self.numeric].to_numpy(float)]
        if self.ohe is not None:
            parts.append(self.ohe.transform(df[self.nominal].astype(str)))
        if self.state_col:
            g = df[self.state_col].map(self.state_group).fillna(1).to_numpy()
            parts.append(np.column_stack([(g == 1), (g == 2)]).astype(float))
        return np.hstack(parts)

    def transform(self, df):
        return self.scaler.transform(self._raw(df))

    def feature_names(self):
        names = list(self.numeric)
        if self.ohe is not None:
            names += list(self.ohe.get_feature_names_out(self.nominal))
        if self.state_col:
            names += ["state_mid_default", "state_high_default"]
        return names


# =============================================================================
# 2. FISHER STATISTICS AND HYPERPLANE SOLVERS
# =============================================================================
def fisher_stats(X, y, delta_rel=1e-6):
    """Class means, pooled within-class covariance (n-2 denominator),
    ridge Sigma_delta = Sigma + delta I with delta = delta_rel * tr(Sigma)/r."""
    G, B = X[y == 1], X[y == 0]
    n, r = X.shape
    mu_G, mu_B = G.mean(0), B.mean(0)
    S = ((G - mu_G).T @ (G - mu_G) + (B - mu_B).T @ (B - mu_B)) / (n - 2)
    delta = delta_rel * np.trace(S) / r
    S_d = S + delta * np.eye(r)
    ev_raw, ev = np.linalg.eigvalsh(S), np.linalg.eigvalsh(S_d)
    d = mu_G - mu_B
    return dict(d=d, Sigma=S_d, L=np.linalg.cholesky(S_d),
                alpha_max=float(d @ np.linalg.solve(S_d, d)), delta=delta,
                kappa_raw=float(ev_raw.max() / max(ev_raw.min(), 1e-300)),
                kappa=float(ev.max() / ev.min()))


def fisher_ratio(w, st):
    v = float(w @ st["Sigma"] @ w)
    return float((st["d"] @ w) ** 2 / v) if v > 1e-14 else 0.0


class HyperplaneProblem:
    """min  (s/n) * sum_i rho_i c_i eps_i + ||L' w||^2   (rho_i = rho for Bad, 1 for Good)
       s.t. y_i (w'x_i - b) >= 1 - eps_i, eps >= 0
            [optional] d'w >= sqrt(alpha) ||L_F' w||   (Fisher cone)
    Compiled once per training set; s and sqrt(alpha) are CVXPY parameters.
      FCS-C          : L = chol(Sigma), c = class costs, cone on
      FCS-C (a=0)    : same, cone with alpha = 0 (i.e. d'w >= 0)
      CS-SVM         : L = I, c = class costs, no cone
      SVM/LDA        : L = chol(lambda Sigma + I), c = 1, no cone
    """

    def __init__(self, X, y, L, cvec, st=None, cone=False):
        n, r = X.shape
        ys = 2.0 * y - 1.0
        self.w, self.b = cp.Variable(r), cp.Variable()
        self.eps = cp.Variable(n, nonneg=True)
        self.sB = cp.Parameter(nonneg=True, value=1.0)     # s * rho  (Bad cases)
        self.sG = cp.Parameter(nonneg=True, value=1.0)     # s        (Good cases)
        self.sa = cp.Parameter(nonneg=True, value=0.0)
        cons = [cp.multiply(ys, X @ self.w - self.b) >= 1 - self.eps]
        self.cone = None
        if cone:
            self.cone = st["d"] @ self.w >= self.sa * cp.norm(st["L"].T @ self.w, 2)
            cons.append(self.cone)
        # loss averaged over n so that the solution (and the score scale) does
        # not depend on the training-set size: inner folds and refit agree
        c = np.asarray(cvec, float) / n
        obj = cp.Minimize(self.sB * (c * (y == 0)) @ self.eps
                          + self.sG * (c * (y == 1)) @ self.eps
                          + cp.sum_squares(L.T @ self.w))
        self.prob = cp.Problem(obj, cons)

    def solve(self, s, sqrt_alpha=0.0, rho=1.0):
        self.sB.value, self.sG.value, self.sa.value = s * rho, s, sqrt_alpha
        try:
            self.prob.solve(solver=cp.CLARABEL)
            solver = "CLARABEL"
        except cp.SolverError:
            self.prob.solve(solver=cp.SCS, max_iters=50_000, eps=1e-7)
            solver = "SCS"
        if self.w.value is None:
            return None
        dual = None
        if self.cone is not None and self.cone.dual_value is not None:
            dual = float(np.atleast_1d(self.cone.dual_value)[0])
        return dict(w=self.w.value.copy(), b=float(self.b.value),
                    status=self.prob.status, solver=solver, cone_dual=dual)


def cost_weights(y, c_fp, c_fn):
    """Per-case weights C_i / s = c_k (c_FP for Bad cases, c_FN for Good cases)."""
    return np.where(y == 0, c_fp, c_fn).astype(float)


DEGENERATE_W = 1e-6     # ||w|| below this: constant classifier (w = 0), treated as infeasible


class LinearScorer:
    def __init__(self, w, b, info=None):
        self.w, self.b, self.info = w, b, info or {}

    def score(self, X):
        return X @ self.w - self.b


# =============================================================================
# 3. METHODS
#    Each method: grid (list of dicts) and fit(X, y, params, ctx) -> model,
#    where model.score(X) is higher for Good and ctx carries costs and caches.
#    native threshold: decision rule of the method as published.
# =============================================================================
class SkProb:
    """Wrap a sklearn classifier: score = P(Good)."""
    def __init__(self, clf):
        self.clf = clf

    def score(self, X):
        return self.clf.predict_proba(X)[:, 1]


class MetaCost:
    """MetaCost (Domingos 1999): relabel by minimum expected cost under bagged
    class-probability estimates, then refit the base learner."""
    def __init__(self, make_base, cost, n_bags=20, seed=SEED):
        self.make_base, self.cost, self.n_bags, self.seed = make_base, cost, n_bags, seed

    def fit(self, X, y):
        rng = np.random.RandomState(self.seed)
        P = np.zeros((len(X), 2))
        for _ in range(self.n_bags):
            idx = rng.choice(len(X), len(X), replace=True)
            m = self.make_base().fit(X[idx], y[idx])
            P[:, m.classes_.astype(int)] += m.predict_proba(X)
        P /= self.n_bags
        # cost[true][pred]; expected cost of predicting j = sum_i P(i) cost[i][j]
        y_new = np.argmin(P @ self.cost, axis=1)
        if len(np.unique(y_new)) < 2:
            y_new = y
        self.final = self.make_base().fit(X, y_new)
        return self

    def score(self, X):
        return self.final.predict_proba(X)[:, 1]


class CSAdaBoost:
    """Cost-sensitive AdaBoost with cost-proportional initial weights
    (Ting 2000): AdaBoost (SAMME) on decision trees, started from
    sample weights proportional to each case's misclassification cost.
    Replaces AdaCost (Fan et al. 1999), whose adjustment function
    beta+(c) = -0.5c + 0.5 makes the boosting edge negative under a 5:1
    cost ratio (all alpha_t < 0), which produced one-class predictions."""
    def __init__(self, c_fp, c_fn, n_estimators=100, depth=1, learning_rate=0.5, seed=SEED):
        self.c_fp, self.c_fn = c_fp, c_fn
        self.n_estimators, self.depth, self.seed = n_estimators, depth, seed
        self.learning_rate = learning_rate

    def fit(self, X, y):
        from sklearn.ensemble import AdaBoostClassifier
        sw = np.where(y == 0, self.c_fp, self.c_fn).astype(float)
        self.clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=self.depth),
                                      n_estimators=self.n_estimators, learning_rate=self.learning_rate,
                                      random_state=self.seed)
        self.clf.fit(X, y, sample_weight=sw / sw.mean())
        return self

    def score(self, X):
        return self.clf.decision_function(X)


GRIDS = {
    # grid profile used in the first balanced-cost run
    "base": dict(
        s=[1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0], rho=[1.0], alpha=[0.1, 0.3, 0.5, 0.7, 0.9],
        lam=[0.1, 1.0, 10.0], shrink=["auto", 0.0, 0.1, 0.3, 0.6], C=[0.01, 0.1, 1.0, 10.0],
        cs_rho=[1.0], depth=[3, 5, 8, None], leaf=[1, 10, 30], tree_rho=[1.0],
        mc_leaf=[5, 20, 50], mc_bags=[20], ada_depth=[1, 2], ada_n=[50, 200], ada_lr=[0.5],
        lgb_leaves=[7, 31], lgb_child=[20, 100], lgb_lr=[0.05], lgb_rho=[1.0]),
    # wider grid
    "wide": dict(
        s=[1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4], rho=[0.25, 0.5, 1.0, 2.0, 4.0, 8.0],
        alpha=[0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99],
        lam=[0.01, 0.1, 1.0, 10.0, 100.0], shrink=["auto", 0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8],
        C=[1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 1e2, 1e3, 1e4], cs_rho=[0.25, 0.5, 1.0, 2.0, 4.0, 8.0],
        depth=[2, 3, 4, 5, 6, 8, 10, 15, None], leaf=[1, 5, 10, 20, 30, 50], tree_rho=[0.5, 1.0, 2.0],
        mc_leaf=[1, 5, 10, 20, 50, 100], mc_bags=[10, 20, 50],
        ada_depth=[1, 2, 3], ada_n=[50, 100, 200, 500], ada_lr=[0.1, 0.5, 1.0],
        lgb_leaves=[7, 15, 31, 63], lgb_child=[10, 20, 50, 100], lgb_lr=[0.1, 0.05, 0.025],
        lgb_rho=[0.5, 1.0, 2.0]),
}
# wide grid trimmed to run all three datasets in about one hour on 18 workers
# FCS-C, its ablation and CS-SVM keep balanced weights (rho = 1): with balanced
# weights the solution cannot degenerate to w = 0 (Proposition "non-degeneracy");
# rho != 1 breaks this guarantee. The freed budget goes to the widest s / alpha grids.
GRIDS["compact"] = dict(
    s=[1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0], rho=[1.0], alpha=[0.1, 0.3, 0.5, 0.7, 0.9, 0.99],
    lam=[0.1, 1.0, 10.0], shrink=["auto", 0.0, 0.2, 0.5], C=[1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0],
    cs_rho=[0.5, 1.0, 2.0], depth=[3, 5, 8, None], leaf=[1, 10, 30], tree_rho=[0.5, 1.0, 2.0],
    mc_leaf=[5, 20, 50], mc_bags=[20], ada_depth=[1, 2], ada_n=[50, 200], ada_lr=[0.5],
    lgb_leaves=[7, 31], lgb_child=[20, 100], lgb_lr=[0.05], lgb_rho=[0.5, 1.0, 2.0],
    svmlda_s=[1e-2, 1.0, 100.0])
GRIDS["wide60"] = dict(GRIDS["wide"], rho=[1.0],
                       s=[1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5],
                       alpha=[0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99],
                       ada_n=[50, 100, 200], lgb_leaves=[7, 15, 31], lgb_child=[10, 50, 100],
                       lgb_lr=[0.1, 0.05])


def build_methods(c_fp, c_fn, G):
    """Methods and their grids. G: grid profile (see GRIDS). rho multiplies the
    Bad-class weight relative to the costs (training weights only; the
    evaluation cost is always c_FP, c_FN)."""
    bayes_t = c_fp / (c_fp + c_fn)          # accept if P(Good) >= c_FP/(c_FP+c_FN)
    cw = lambda rho=1.0: {0: rho * c_fp, 1: c_fn}
    P = itertools.product

    def fcsc_fit(X, y, p, ctx, cone_alpha=None):
        """Solve (P_R). By Proposition 5(i), whenever alpha <= J0 (the Fisher ratio of
        the alpha = 0 solution, unique by Proposition 2) the solution equals the
        alpha = 0 solution, so it is reused instead of solving again."""
        st = ctx["st"]
        if "fcsc" not in ctx:
            ctx["fcsc"] = HyperplaneProblem(X, y, st["L"], cost_weights(y, c_fp, c_fn),
                                            st=st, cone=True)
            ctx["fcsc_cache"] = {}
        cache = ctx["fcsc_cache"]
        s_, rho = p["s"], p.get("rho", 1.0)
        a_rel = p["alpha_rel"] if cone_alpha is None else cone_alpha

        def solve(a):
            key = (s_, rho, a)
            if key not in cache:
                sol = ctx["fcsc"].solve(s_, np.sqrt(a * st["alpha_max"]), rho)
                if sol is not None:
                    sol["J"] = fisher_ratio(sol["w"], st)
                cache[key] = sol
            return cache[key]

        sol0 = solve(0.0)
        if sol0 is not None and a_rel * st["alpha_max"] <= sol0["J"] * (1 - 1e-9):
            sol = dict(sol0, cone_dual=0.0, reused_alpha0=True)      # Proposition 5(i)
        else:
            sol = solve(a_rel)
        if sol is None or np.linalg.norm(sol["w"]) < DEGENERATE_W:
            return None
        return LinearScorer(sol["w"], sol["b"], sol)

    def cssvm_fit(X, y, p, ctx):
        if "cssvm" not in ctx:
            ctx["cssvm"] = HyperplaneProblem(X, y, np.eye(X.shape[1]), cost_weights(y, c_fp, c_fn))
        sol = ctx["cssvm"].solve(p["s"], rho=p.get("rho", 1.0))
        if sol is None or np.linalg.norm(sol["w"]) < DEGENERATE_W:
            return None
        return LinearScorer(sol["w"], sol["b"], sol)

    def svmlda_fit(X, y, p, ctx):
        key = ("svmlda", p["lam"])
        if key not in ctx:
            M = p["lam"] * ctx["st"]["Sigma"] + np.eye(X.shape[1])
            ctx[key] = HyperplaneProblem(X, y, np.linalg.cholesky(M), np.ones(len(y)))
        sol = ctx[key].solve(p["s"])
        if sol is None or np.linalg.norm(sol["w"]) < DEGENERATE_W:
            return None
        return LinearScorer(sol["w"], sol["b"], sol)

    def lda_fit(X, y, p, ctx):
        return SkProb(LinearDiscriminantAnalysis(solver="lsqr", shrinkage=p["shrinkage"]).fit(X, y))

    def lr_fit(X, y, p, ctx, weighted):
        return SkProb(LogisticRegression(C=p["C"], max_iter=5000,
                                         class_weight=cw(p.get("rho", 1.0)) if weighted else None).fit(X, y))

    def tree_fit(X, y, p, ctx, crit, weighted):
        q = {k: v for k, v in p.items() if k != "rho"}
        return SkProb(DecisionTreeClassifier(criterion=crit, random_state=SEED,
                                             class_weight=cw(p.get("rho", 1.0)) if weighted else None,
                                             **q).fit(X, y))

    def metacost_fit(X, y, p, ctx):
        cost = np.array([[0.0, c_fp], [c_fn, 0.0]])     # rows: true Bad, Good
        mk = lambda: DecisionTreeClassifier(criterion="gini", random_state=SEED,
                                            min_samples_leaf=p["min_samples_leaf"])
        return MetaCost(mk, cost, n_bags=p["n_bags"]).fit(X, y)

    def adaboost_fit(X, y, p, ctx):
        return CSAdaBoost(c_fp, c_fn, **p).fit(X, y)

    def lgbm_fit(X, y, p, ctx):
        from lightgbm import LGBMClassifier
        return SkProb(LGBMClassifier(n_estimators=int(round(15 / p["learning_rate"])),
                                     learning_rate=p["learning_rate"], num_leaves=p["num_leaves"],
                                     min_child_samples=p["min_child_samples"],
                                     subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                     class_weight=cw(p["rho"]), random_state=SEED, n_jobs=1,
                                     verbose=-1).fit(X, y))

    tree_g = [dict(max_depth=d, min_samples_leaf=m) for d, m in P(G["depth"], G["leaf"])]
    cs_tree_g = [dict(t, rho=r) for t in tree_g for r in G["tree_rho"]]
    return {
        "FCS-C": dict(native=0.0, fit=fcsc_fit,
                      grid=[dict(s=s_, rho=r, alpha_rel=a) for s_, r, a in P(G["s"], G["rho"], G["alpha"])]),
        "FCS-C (alpha=0)": dict(native=0.0,
                                fit=lambda X, y, p, ctx: fcsc_fit(X, y, p, ctx, cone_alpha=0.0),
                                grid=[dict(s=s_, rho=r) for s_, r in P(G["s"], G["rho"])]),
        "CS-SVM": dict(native=0.0, fit=cssvm_fit,
                       grid=[dict(s=s_, rho=r) for s_, r in P(G["s"], G["rho"])]),
        "SVM/LDA": dict(native=0.0, fit=svmlda_fit,
                        grid=[dict(s=s_, lam=l) for s_, l in P(G.get("svmlda_s", G["s"]), G["lam"])]),
        "R-LDA": dict(native=0.5, fit=lda_fit, grid=[dict(shrinkage=g) for g in G["shrink"]]),
        "R-LDA (Bayes cost)": dict(native=bayes_t, fit=lda_fit,
                                   grid=[dict(shrinkage=g) for g in G["shrink"]]),
        "LR": dict(native=0.5, fit=lambda X, y, p, ctx: lr_fit(X, y, p, ctx, False),
                   grid=[dict(C=c) for c in G["C"]]),
        "CS-LR": dict(native=0.5, fit=lambda X, y, p, ctx: lr_fit(X, y, p, ctx, True),
                      grid=[dict(C=c, rho=r) for c, r in P(G["C"], G["cs_rho"])]),
        "C4.5": dict(native=0.5, grid=tree_g,
                     fit=lambda X, y, p, ctx: tree_fit(X, y, p, ctx, "entropy", False)),
        "CART": dict(native=0.5, grid=tree_g,
                     fit=lambda X, y, p, ctx: tree_fit(X, y, p, ctx, "gini", False)),
        "CS-C4.5": dict(native=0.5, grid=cs_tree_g,
                        fit=lambda X, y, p, ctx: tree_fit(X, y, p, ctx, "entropy", True)),
        "CS-CART": dict(native=0.5, grid=cs_tree_g,
                        fit=lambda X, y, p, ctx: tree_fit(X, y, p, ctx, "gini", True)),
        "MetaCost": dict(native=0.5, fit=metacost_fit,
                         grid=[dict(min_samples_leaf=m, n_bags=b) for m, b in P(G["mc_leaf"], G["mc_bags"])]),
        "CS-AdaBoost": dict(native=0.0, fit=adaboost_fit,
                            grid=[dict(depth=d, n_estimators=n, learning_rate=l)
                                  for d, n, l in P(G["ada_depth"], G["ada_n"], G["ada_lr"])]),
        "LightGBM": dict(native=0.5, fit=lgbm_fit,
                         grid=[dict(num_leaves=a_, min_child_samples=m, learning_rate=l, rho=r)
                               for a_, m, l, r in P(G["lgb_leaves"], G["lgb_child"], G["lgb_lr"], G["lgb_rho"])]),
    }


# =============================================================================
# 4. METRICS
# =============================================================================
def op_metrics(y, score, t, c_fp, c_fn):
    pred = (score >= t).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    rec_g = tp / max(tp + fn, 1); rec_b = tn / max(tn + fp, 1)
    prec_g = tp / max(tp + fp, 1); prec_b = tn / max(tn + fn, 1)
    f1 = lambda p, r: 2 * p * r / (p + r) if p + r > 0 else 0.0
    return dict(cost=(c_fp * fp + c_fn * fn) / len(y), gmean=np.sqrt(rec_g * rec_b),
                rec_good=rec_g, rec_bad=rec_b, acc=(tp + tn) / len(y),
                f1_good=f1(prec_g, rec_g), f1_bad=f1(prec_b, rec_b),
                TP=tp, TN=tn, FP=fp, FN=fn)


def best_threshold(y, score, c_fp, c_fn):
    """Threshold minimising realized cost (accept if score >= t), exact search
    over all cut points in O(n log n). Returns (threshold, cost per applicant)."""
    o = np.argsort(-score)
    s, yy = score[o], y[o]
    # accepting the top k cases: FP = bad among top k, FN = good outside top k
    fp = np.concatenate([[0], np.cumsum(yy == 0)])
    fn = (yy == 1).sum() - np.concatenate([[0], np.cumsum(yy == 1)])
    cost = (c_fp * fp + c_fn * fn) / len(y)
    valid = np.concatenate([[True], s[:-1] > s[1:], [True]])   # cut only between distinct scores
    cost = np.where(valid, cost, np.inf)
    k = int(np.argmin(cost))
    if k == 0:
        t = s[0] + 1e-9
    elif k == len(s):
        t = s[-1] - 1e-9
    else:
        t = 0.5 * (s[k - 1] + s[k])
    return float(t), float(cost[k])


def safe_auc(y, s):
    try:
        return float(roc_auc_score(y, s))
    except ValueError:
        return 0.5


# =============================================================================
# 5. NESTED EVALUATION OF ONE OUTER FOLD
# =============================================================================
def run_outer_fold(ds, df, y, cols, tr, te, rep, fold, cfg):
    numeric, nominal, state_col = cols
    c_fp, c_fn = sklearn_costs(y[tr])          # balanced costs from the outer-training part
    methods = build_methods(c_fp, c_fn, GRIDS[cfg["grid"]])
    if cfg["methods"]:
        methods = {k: v for k, v in methods.items() if k in cfg["methods"]}
    rows, diag, curve = [], [], []

    df_tr, y_tr = df.iloc[tr], y[tr]
    df_te, y_te = df.iloc[te], y[te]

    # inner folds: preprocessing fitted on each inner-train part
    inner = []
    skf = StratifiedKFold(cfg["inner"], shuffle=True, random_state=SEED + 1000 * rep + fold)
    for itr, iva in skf.split(np.zeros(len(y_tr)), y_tr):
        pp = Preprocessor(numeric, nominal, state_col).fit(df_tr.iloc[itr], y_tr[itr])
        Xi, Xv = pp.transform(df_tr.iloc[itr]), pp.transform(df_tr.iloc[iva])
        inner.append(dict(X=Xi, y=y_tr[itr], Xv=Xv, yv=y_tr[iva], st=fisher_stats(Xi, y_tr[itr])))

    pp = Preprocessor(numeric, nominal, state_col).fit(df_tr, y_tr)
    X_tr, X_te = pp.transform(df_tr), pp.transform(df_te)
    st_out = fisher_stats(X_tr, y_tr)

    for name, M in methods.items():
        for f in inner:
            f["ctx"] = {"st": f["st"]}             # fresh solver context per method
        ctx_out = {"st": st_out}
        t0 = time.time()
        # ---- inner CV: select params by realized cost at the cost-optimal
        #      threshold on pooled out-of-fold scores (tie-break: mean AUC)
        cands = []
        for p in M["grid"]:
            costs, aucs, oof_s, oof_y = [], [], [], []
            for f in inner:
                m = M["fit"](f["X"], f["y"], p, f["ctx"])
                if m is None:
                    costs = None
                    break
                sv = m.score(f["Xv"])
                costs.append(op_metrics(f["yv"], sv, M["native"], c_fp, c_fn)["cost"])
                aucs.append(safe_auc(f["yv"], sv))
                # scale-free score: quantile among the inner model's training scores
                str_ = np.sort(m.score(f["X"]))
                oof_s.append(np.searchsorted(str_, sv, side="right") / len(str_))
                oof_y.append(f["yv"])
            if costs is None:
                continue
            oof_s, oof_y = np.concatenate(oof_s), np.concatenate(oof_y)
            t_p, c_p = best_threshold(oof_y, oof_s, c_fp, c_fn)
            cands.append(((round(c_p, 6), -np.mean(aucs)), len(cands), p, t_p))
        # ---- refit the best configuration on the whole outer-training part
        #      (next best if its refit is degenerate), evaluate on the test fold
        t_tune = time.time() - t0
        t1 = time.time()
        model = None
        for _, _, p_best, q_opt in sorted(cands, key=lambda z: (z[0], z[1])):
            model = M["fit"](X_tr, y_tr, p_best, ctx_out)   # q_opt: cut-off as training-score quantile
            if model is not None:
                break
        if model is None:
            print(f"   [{ds} r{rep} f{fold}] {name}: no non-degenerate configuration", flush=True)
            continue
        t_fit = time.time() - t1
        t2 = time.perf_counter()
        s_te = model.score(X_te)
        t_pred = time.perf_counter() - t2
        s_trs = np.sort(model.score(X_tr))
        q_te = np.searchsorted(s_trs, s_te, side="right") / len(s_trs)
        row = dict(dataset=ds, method=name, rep=rep, fold=fold,
                   params=json.dumps(p_best), auc=safe_auc(y_te, s_te),
                   c_fp=c_fp, c_fn=c_fn, seconds=time.time() - t0,
                   tune_seconds=t_tune, fit_seconds=t_fit,
                   predict_ms_per_1000=1e6 * t_pred / len(y_te),
                   n_configs=len(M["grid"]))
        for tag, sc, t in (("native", s_te, M["native"]), ("opt", q_te, q_opt)):
            for k, v in op_metrics(y_te, sc, t, c_fp, c_fn).items():
                row[f"{k}_{tag}"] = v
        rows.append(row)

        # ---- FCS-C diagnostics (Prop. 5) and alpha curve at the selected s
        if name == "FCS-C":
            st = ctx_out["st"]
            m0 = M["fit"](X_tr, y_tr, dict(p_best, alpha_rel=0.0), ctx_out)
            J0 = m0.info["J"] if m0 is not None else np.nan
            info = model.info
            diag.append(dict(dataset=ds, rep=rep, fold=fold, s=p_best["s"], c_fp=c_fp, c_fn=c_fn,
                             alpha_rel=p_best["alpha_rel"], alpha_max=st["alpha_max"],
                             alpha_used=p_best["alpha_rel"] * st["alpha_max"],
                             J_achieved=info["J"], J0=J0,
                             J0_rel=J0 / st["alpha_max"],
                             active=bool(p_best["alpha_rel"] * st["alpha_max"] > J0 * (1 + 1e-4)),
                             cone_dual=info["cone_dual"], status=info["status"],
                             solver=info["solver"], kappa_raw=st["kappa_raw"],
                             kappa=st["kappa"], delta=st["delta"], r=X_tr.shape[1],
                             w=json.dumps(list(np.round(info["w"], 6))),
                             features=json.dumps(pp.feature_names())))
            for a in cfg["curve"]:
                mc = M["fit"](X_tr, y_tr, dict(p_best, alpha_rel=a), ctx_out)
                if mc is None:
                    continue
                sc = mc.score(X_te)
                om = op_metrics(y_te, sc, 0.0, c_fp, c_fn)
                str_c = np.sort(mc.score(X_tr))
                oq = op_metrics(y_te, np.searchsorted(str_c, sc, side="right") / len(str_c),
                                q_opt, c_fp, c_fn)
                curve.append(dict(dataset=ds, rep=rep, fold=fold, alpha_rel=a,
                                  auc=safe_auc(y_te, sc), cost_native=om["cost"],
                                  gmean_native=om["gmean"], cost_opt=oq["cost"],
                                  gmean_opt=oq["gmean"], J=mc.info["J"],
                                  J_rel=mc.info["J"] / st["alpha_max"],
                                  active=bool(not mc.info.get("reused_alpha0", False) and a > 0)))
        print(f"   [{ds} r{rep} f{fold}] {name:<20} AUC={row['auc']:.4f} "
              f"cost={row['cost_native']:.4f}/{row['cost_opt']:.4f} "
              f"({row['seconds']:.1f}s) {p_best}", flush=True)
    return rows, diag, curve


# =============================================================================
# 6. SIGNIFICANCE: corrected resampled t-test (Nadeau & Bengio, 2003)
# =============================================================================
def corrected_ttest(diff, n_train, n_test):
    diff = np.asarray(diff, float)
    J = len(diff)
    var = diff.var(ddof=1)
    if var == 0:
        return 0.0, 1.0
    t = diff.mean() / np.sqrt((1 / J + n_test / n_train) * var)
    return float(t), float(2 * stats.t.sf(abs(t), J - 1))


def summarise(res, ds, n_train, n_test, out_dir):
    metrics = ["auc", "cost_native", "cost_opt", "gmean_native", "gmean_opt",
               "rec_good_native", "rec_bad_native", "acc_native", "f1_good_native",
               "f1_bad_native"]
    g = res.groupby("method")[metrics]
    summ = g.mean().round(4).astype(str) + " ± " + g.std().round(4).astype(str)
    order = res.groupby("method")["cost_opt"].mean().sort_values().index
    summ = summ.loc[order]
    summ.to_csv(os.path.join(out_dir, f"{ds}_summary.csv"))

    ref = res[res.method == "FCS-C"].sort_values(["rep", "fold"])
    tests = []
    for m in res.method.unique():
        if m == "FCS-C":
            continue
        oth = res[res.method == m].sort_values(["rep", "fold"])
        if len(oth) != len(ref):
            continue
        row = dict(method=m)
        for k in ("auc", "cost_native", "cost_opt"):
            d = ref[k].to_numpy() - oth[k].to_numpy()
            t, p = corrected_ttest(d, n_train, n_test)
            try:
                pw = stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0
            except ValueError:
                pw = 1.0
            row.update({f"{k}_diff": round(d.mean(), 4), f"{k}_p": round(p, 4),
                        f"{k}_p_wilcoxon": round(float(pw), 4)})
        tests.append(row)
    tests = pd.DataFrame(tests)
    tests.to_csv(os.path.join(out_dir, f"{ds}_tests_vs_fcsc.csv"), index=False)
    return summ, tests


# =============================================================================
# 7. MAIN
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(LOADERS) + ["all"])
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--inner", type=int, default=5)
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--n-lc", type=int, default=20_000)
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--max-outer", type=int, default=None, help="debug: limit outer folds")
    ap.add_argument("--grid", default="base", choices=list(GRIDS))
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    cfg = dict(inner=a.inner, grid=a.grid, methods=a.methods,
               curve=[round(0.05 * k, 2) for k in range(20)] + [0.99])
    print("grid sizes:", {k: len(v["grid"]) for k, v in build_methods(1.0, 1.0, GRIDS[a.grid]).items()})
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), a.out)
    os.makedirs(out_dir, exist_ok=True)

    names = ["lendingclub", "german", "australian"] if a.dataset == "all" else [a.dataset]
    data, tasks = {}, []
    for ds in names:                        # slowest dataset first: its folds are dispatched first
        df, y, numeric, nominal, state_col = (LOADERS[ds](a.n_lc) if ds == "lendingclub"
                                              else LOADERS[ds]())
        print(f"{ds}: n={len(y)}  Good={y.sum()}  Bad={(1 - y).sum()}  "
              "costs(c_FP,c_FN)=(%.4f, %.4f) [sklearn balanced, full data; "
              "recomputed on each training fold]" % sklearn_costs(y), flush=True)
        rskf = RepeatedStratifiedKFold(n_splits=a.folds, n_repeats=a.repeats, random_state=SEED)
        splits = list(rskf.split(np.zeros(len(y)), y))[:a.max_outer]
        data[ds] = (len(splits[0][0]), len(splits[0][1]), len(splits))
        tasks += [(ds, df, y, (numeric, nominal, state_col), tr, te, k // a.folds, k % a.folds)
                  for k, (tr, te) in enumerate(splits)]

    t0 = time.time()
    out = Parallel(n_jobs=a.jobs, verbose=0, batch_size=1)(
        delayed(run_outer_fold)(*t, cfg) for t in tasks)
    print(f"\nall outer folds done in {time.time() - t0:.0f}s", flush=True)

    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    for ds in names:
        sel = [o for t, o in zip(tasks, out) if t[0] == ds]
        res = pd.DataFrame([r for o in sel for r in o[0]])
        diag = pd.DataFrame([r for o in sel for r in o[1]])
        curve = pd.DataFrame([r for o in sel for r in o[2]])
        res.to_csv(os.path.join(out_dir, f"{ds}_folds.csv"), index=False)
        diag.to_csv(os.path.join(out_dir, f"{ds}_fcsc_diagnostics.csv"), index=False)
        curve.to_csv(os.path.join(out_dir, f"{ds}_alpha_curve.csv"), index=False)
        n_tr, n_te, n_out = data[ds]
        summ, tests = summarise(res, ds, n_tr, n_te, out_dir)
        print(f"\n==== {ds}: {n_out} outer folds ====")
        print(summ[["auc", "cost_native", "cost_opt", "gmean_native", "gmean_opt"]].to_string())
        print("\n-- FCS-C minus method (corrected resampled t-test p) --")
        print(tests.to_string(index=False))
        if len(diag):
            print("\n-- FCS-C diagnostics (means) --")
            print(diag[["alpha_rel", "alpha_max", "alpha_used", "J_achieved", "J0",
                        "kappa_raw", "kappa"]].mean().round(4).to_string())
            print(f"active fraction: {diag.active.mean():.2f}   "
                  f"solvers: {diag.solver.value_counts().to_dict()}")
        if len(curve):
            print("\n-- alpha curve (means) --")
            print(curve.groupby("alpha_rel")[["auc", "cost_native", "gmean_native", "J_rel"]]
                  .mean().round(4).to_string())


if __name__ == "__main__":
    main()
