"""
Write a standalone LaTeX report of the FCS-C experiments from the CSV files in
results/ (produced by fcsc_pipeline.py or FCS-C_experiments.ipynb).

    python make_results_tex.py            ->  results/report/FCS-C_results.tex (+ figures)
"""
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
import sys
RES = os.path.join(HERE, sys.argv[1] if len(sys.argv) > 1 else "results")   # results folder
OUT = os.path.join(RES, "report")
os.makedirs(OUT, exist_ok=True)

DS = ["australian", "german", "lendingclub"]
DS_NAME = {"australian": "Australian Credit", "german": "German Credit",
           "lendingclub": "LendingClub"}
DS_INFO = {"australian": dict(n=690), "german": dict(n=1000), "lendingclub": dict(n=20000)}
GOOD_SHARE = {"australian": 307 / 690, "german": 700 / 1000, "lendingclub": 16094 / 20000}


def costs(d):
    """Mean fold costs (scikit-learn balanced formula c_k = n/(2 n_k)); falls back to
    the full-data value when the fold file has no cost columns."""
    r = res[d]
    if "c_fp" in r:
        return float(r.c_fp.mean()), float(r.c_fn.mean())
    return 1 / (2 * (1 - GOOD_SHARE[d])), 1 / (2 * GOOD_SHARE[d])
ABL = "FCS-C (alpha=0)"                 # ablation: reported in its own section
METHOD_ORDER = ["FCS-C", "CS-SVM", "SVM/LDA", "R-LDA", "R-LDA (Bayes cost)",
                "LR", "CS-LR", "C4.5", "CART", "CS-C4.5", "CS-CART", "MetaCost",
                "CS-AdaBoost", "LightGBM"]
TEX_NAME = {"FCS-C": r"\textbf{FCS-C}", "FCS-C (alpha=0)": r"FCS-C ($\alpha=0$)",
            "R-LDA (Bayes cost)": "R-LDA (Bayes cost)"}
DEGENERATE_THR = 1e-3        # ||w|| below this: constant classifier

res = {d: pd.read_csv(os.path.join(RES, f"{d}_folds.csv")) for d in DS}
diag = {d: pd.read_csv(os.path.join(RES, f"{d}_fcsc_diagnostics.csv")) for d in DS}
curve = {d: pd.read_csv(os.path.join(RES, f"{d}_alpha_curve.csv")) for d in DS}
tests = {d: pd.read_csv(os.path.join(RES, f"{d}_tests_vs_fcsc.csv")).set_index("method") for d in DS}
for d in DS:
    diag[d]["wnorm"] = diag[d].w.map(lambda s: float(np.linalg.norm(json.loads(s))))

# hinge-type models whose solution is degenerate (w = 0) on a dataset: flagged with a dagger
DEGEN = {d: set() for d in DS}
for d in DS:
    if (diag[d].wnorm < DEGENERATE_THR).mean() > 0.5:
        DEGEN[d] |= {"FCS-C", "FCS-C (alpha=0)", "CS-SVM", "SVM/LDA"}


def name(m, d=None):
    s = TEX_NAME.get(m, m.replace("&", r"\&"))
    if d is not None and m in DEGEN[d]:
        s += r"$^{\dagger}$"
    return s


def fmt(mean, sd, best=False, star=False, digits=3):
    s = f"{mean:.{digits}f} ({sd:.{digits}f})"
    if best:
        s = r"\textbf{" + s + "}"
    if star:
        s += r"$^{*}$"
    return s


def method_table(d, cols, lower_better, caption, label, op="opt", show_tests=True):
    r = res[d]
    g_mean = r.groupby("method")[cols].mean()
    g_sd = r.groupby("method")[cols].std()
    methods = [m for m in METHOD_ORDER if m in g_mean.index]
    valid = [m for m in methods if m not in DEGEN[d]]
    best = {c: (g_mean.loc[valid, c].idxmin() if c in lower_better else g_mean.loc[valid, c].idxmax())
            for c in cols}
    head = {"auc": "AUC", f"cost_{op}": "Cost", f"gmean_{op}": "G-mean",
            f"rec_good_{op}": r"Rec$_G$", f"rec_bad_{op}": r"Rec$_B$", f"acc_{op}": "Acc.",
            f"f1_good_{op}": r"F1$_G$", f"f1_bad_{op}": r"F1$_B$"}
    pcol = {"auc": "auc_p", f"cost_{op}": f"cost_{op}_p"}
    lines = [r"\begin{table}[htbp]", r"\centering", r"\caption{" + caption + "}", r"\label{" + label + "}",
             r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\toprule",
             "Method & " + " & ".join(head[c] for c in cols) + r" \\", r"\midrule"]
    for m in methods:
        cells = []
        for c in cols:
            star = False
            if show_tests and m != "FCS-C" and c in pcol and m in tests[d].index:
                star = tests[d].loc[m, pcol[c]] < 0.05
            cells.append(fmt(g_mean.loc[m, c], g_sd.loc[m, c], best=(m == best[c]), star=star))
        lines.append(name(m, d) + " & " + " & ".join(cells) + r" \\")
        if m == "FCS-C":
            lines.append(r"\midrule")
    for m in [m for m in METHOD_ORDER if m not in g_mean.index]:
        lines.append(name(m, d) + r" & \multicolumn{" + str(len(cols)) +
                     r"}{c}{degenerate ($w=0$, constant classifier) in every outer fold} \\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(lines)


def tests_table(d):
    t = tests[d]
    methods = [m for m in METHOD_ORDER if m in t.index]
    lines = [r"\begin{table}[htbp]", r"\centering",
             r"\caption{" + DS_NAME[d] + r": paired comparison FCS-C minus method over the outer folds. "
             r"$\Delta$ = mean difference; $p$ = corrected resampled $t$-test \cite{NadeauBengio2003}; "
             r"$p_W$ = Wilcoxon signed-rank test. For AUC a positive $\Delta$ favours FCS-C; "
             r"for cost a negative $\Delta$ favours FCS-C.}",
             r"\label{tab:tests_" + d + "}", r"\small",
             r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r" & \multicolumn{3}{c}{AUC} & \multicolumn{3}{c}{Cost (cost-optimal cut-off)} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
             r"Method & $\Delta$ & $p$ & $p_W$ & $\Delta$ & $p$ & $p_W$ \\", r"\midrule"]
    for m in methods:
        row = t.loc[m]
        pv = lambda x: (r"\textbf{" + f"{x:.3f}" + "}") if x < 0.05 else f"{x:.3f}"
        lines.append(name(m, d) + f" & {row.auc_diff:+.4f} & {pv(row.auc_p)} & {pv(row.auc_p_wilcoxon)}"
                     f" & {row.cost_opt_diff:+.4f} & {pv(row.cost_opt_p)} & {pv(row.cost_opt_p_wilcoxon)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ----------------------------------------------------------------------------- figures
FCSC_COLOR, OTHER_COLOR, EDGE = "#c0392b", "#b8b0a2", "#4d4d4d"     # no blue
DS_COLOR = {"australian": "#e66101", "german": "#1b9e77", "lendingclub": "#c51b7d"}
plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12})


def fig_overview(d):
    """Vertical bars (mean +- sd over outer folds) of AUC and realized cost;
    FCS-C highlighted, competitors in neutral grey."""
    r = res[d]
    order = [m for m in METHOD_ORDER if m in r.method.unique()]
    labels = [m.replace(" (Bayes cost)", " (Bayes)") for m in order]
    x = np.arange(len(order))
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.2))
    for a, (k, lab) in zip(ax, [("auc", "AUC"), ("cost_opt", "Realized cost per applicant")]):
        g = r.groupby("method")[k].agg(["mean", "std"]).reindex(order)
        cols = [FCSC_COLOR if m == "FCS-C" else OTHER_COLOR for m in order]
        bars = a.bar(x, g["mean"], yerr=g["std"], color=cols, edgecolor=EDGE, linewidth=0.6,
                     capsize=3, error_kw=dict(ecolor=EDGE, lw=1))
        bars[order.index("FCS-C")].set_linewidth(1.6)
        a.set_xticks(x); a.set_xticklabels(labels, rotation=55, ha="right")
        a.set_ylabel(lab); a.grid(axis="y", alpha=.35); a.set_axisbelow(True)
        lo, hi = (g["mean"] - g["std"]).min(), (g["mean"] + g["std"]).max()
        a.set_ylim(lo - 0.08 * (hi - lo), hi + 0.08 * (hi - lo))
        a.spines[["top", "right"]].set_visible(False)
    missing = [m for m in METHOD_ORDER if m not in r.method.unique()]
    title = DS_NAME[d] + (f"   ({', '.join(missing)}: degenerate in every fold)" if missing else "")
    fig.suptitle(title, fontweight="bold")
    plt.tight_layout()
    fn = f"fig_{d}_overview.pdf"
    plt.savefig(os.path.join(OUT, fn), bbox_inches="tight"); plt.close()
    return fn


def fig_alpha():
    """Sensitivity to alpha_rel: one colour per dataset, thick curves, s.e. band,
    shaded region where the Fisher bound is active."""
    fig, ax = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    for j, d in enumerate(DS):
        c = curve[d].groupby("alpha_rel")
        n = c.size().iloc[0]
        col = DS_COLOR[d]
        j0 = diag[d].J0_rel.mean()
        for i, (k, lab) in enumerate([("cost_opt", "Realized cost"), ("auc", "AUC")]):
            a = ax[i, j]
            m_, se = c[k].mean(), c[k].std() / np.sqrt(n)
            a.axvspan(j0, 1.0, color="#bdbdbd", alpha=.35, lw=0,
                      label="bound active" if (i == 0 and j == 0) else None)
            a.fill_between(m_.index, m_ - se, m_ + se, color=col, alpha=.10, lw=0)
            a.plot(m_.index, m_ - se, color=col, lw=0.8, alpha=.5)
            a.plot(m_.index, m_ + se, color=col, lw=0.8, alpha=.5)
            a.plot(m_.index, m_, "-o", color=col, lw=3.4, ms=7.5, mec="white", mew=1.3, zorder=5)
            a.axvline(j0, ls="--", color="#333333", lw=1.6,
                      label=r"$J_0/\alpha_{\max}$" if (i == 0 and j == 0) else None)
            a.set_ylabel(lab); a.grid(alpha=.35); a.set_xlim(-0.02, 1.0)
            a.spines[["top", "right"]].set_visible(False)
            if i == 0:
                a.set_title(DS_NAME[d], color=col, fontweight="bold")
            if i == 1:
                a.set_xlabel(r"$\alpha_{rel}=\alpha/\alpha_{\max}$")
    ax[0, 0].legend(loc="lower left", frameon=False)
    plt.tight_layout()
    fn = "fig_alpha_curve.pdf"
    plt.savefig(os.path.join(OUT, fn), bbox_inches="tight"); plt.close()
    return fn


figs = {d: fig_overview(d) for d in DS}
fig_a = fig_alpha()

# ----------------------------------------------------------------------------- tables
# cost of trivial rules: accept all = c_FP * bad share ; reject all = c_FN * good share
triv = {}
for d in DS:
    cfp, cfn = costs(d)
    triv[d] = (cfp * (1 - GOOD_SHARE[d]), cfn * GOOD_SHARE[d])
    DS_INFO[d]["cost"] = f"{cfp:.3f} : {cfn:.3f}"

setup_tab = "\n".join([
    r"\begin{table}[htbp]", r"\centering",
    r"\caption{Datasets and protocol. $r$ = number of features after encoding; "
    r"the last two columns give the realized cost per applicant of the two trivial rules.}",
    r"\label{tab:setup}", r"\small",
    r"\begin{tabular}{lrrrcccc}", r"\toprule",
    r"Dataset & $n$ & $r$ & Bad & $c_{FP}:c_{FN}$ & Outer folds & Accept all & Reject all \\", r"\midrule",
    *[f"{DS_NAME[d]} & {DS_INFO[d]['n']:,} & {int(diag[d].r.median())} & {100 * (1 - GOOD_SHARE[d]):.1f}\\% & "
      f"{DS_INFO[d]['cost']} & {res[d].rep.nunique()}$\\times${res[d].fold.nunique()} & "
      f"{triv[d][0]:.4f} & {triv[d][1]:.4f} \\\\" for d in DS],
    r"\bottomrule", r"\end{tabular}", r"\end{table}"])

fisher_rows = []
for d in DS:
    g = diag[d]
    fisher_rows.append(
        f"{DS_NAME[d]}{'$^{\\dagger}$' if DEGEN[d] else ''} & {g.alpha_max.mean():.4f} & {np.sqrt(g.alpha_max).mean():.4f} & "
        f"{g.kappa_raw.mean():.2e} & {g.kappa.mean():.1f} & {g.alpha_rel.mean():.2f} & "
        f"{g.J0_rel.mean():.3f} & {(g.J_achieved / g.alpha_max).mean():.3f} & "
        f"{100 * g.active.mean():.0f}\\% & {g.wnorm.median():.2e} & {int((g.wnorm < DEGENERATE_THR).sum())}/{len(g)} \\\\")
fisher_tab = "\n".join([
    r"\begin{table}[htbp]", r"\centering",
    r"\caption{Fisher diagnostics of FCS-C, averaged over the outer folds. $\kappa$: condition number of $\Sigma$ "
    r"before and after the ridge $\delta=10^{-6}\,\mathrm{tr}(\Sigma)/r$; $\alpha_{rel}$: selected value; "
    r"$J_0$: Fisher ratio of the $\alpha=0$ solution (Proposition~5: the bound is active iff $\alpha>J_0$); "
    r"$J$: achieved Fisher ratio; last column: folds with a degenerate solution $\|w\|<10^{-3}$.}",
    r"\label{tab:fisher}",
    r"\resizebox{\textwidth}{!}{%",
    r"\begin{tabular}{lcccccccccc}", r"\toprule",
    r"Dataset & $\alpha_{\max}$ & Mahal. & $\kappa$ (raw) & $\kappa$ (ridge) & $\alpha_{rel}$ & "
    r"$J_0/\alpha_{\max}$ & $J/\alpha_{\max}$ & Bound active & median $\|w\|$ & Degenerate \\", r"\midrule",
    *fisher_rows, r"\bottomrule", r"\end{tabular}}", r"\end{table}"])

# sensitivity of FCS-C to alpha_rel (rows) for every dataset (columns)
alphas = sorted(curve[DS[0]].alpha_rel.unique())
sens_rows = []
for a_ in alphas:
    cells = []
    for d in DS:
        c = curve[d][curve[d].alpha_rel == a_]
        cells += [f"{c.cost_opt.mean():.4f}", f"{c.auc.mean():.4f}", f"{c.J_rel.mean():.3f}",
                  f"{100 * c.active.mean():.0f}"]
    sens_rows.append(f"{a_:g} & " + " & ".join(cells) + r" \\")
sens_tab = "\n".join([
    r"\begin{table}[htbp]", r"\centering",
    r"\caption{Sensitivity of FCS-C to the Fisher tightness $\alpha_{rel}$: in every outer fold FCS-C is refitted "
    r"for each $\alpha_{rel}$ with the other hyper-parameters at their selected values and evaluated on the test fold "
    r"at the selected acceptance rate (mean over the 25 outer folds). Act.: share of folds (\%) in which the "
    r"Fisher bound is active ($\alpha > J_0$, Proposition~5); $\alpha_{rel}=0$ is the ablation.}",
    r"\label{tab:sensitivity}", r"\scriptsize\setlength{\tabcolsep}{3pt}",
    r"\begin{tabular}{l" + "rrrr" * len(DS) + "}", r"\toprule",
    r" & " + " & ".join(r"\multicolumn{4}{c}{" + DS_NAME[d] + "}" for d in DS) + r" \\",
    "".join(rf"\cmidrule(lr){{{2 + 4 * i}-{5 + 4 * i}}}" for i in range(len(DS))),
    r"$\alpha_{rel}$ & " + " & ".join([r"Cost & AUC & $J/\alpha_{\max}$ & Act."] * len(DS)) + r" \\",
    r"\midrule", *sens_rows, r"\bottomrule", r"\end{tabular}", r"\end{table}"])

# ablation: FCS-C (alpha_rel > 0, tuned) versus FCS-C with alpha = 0 (tuned s)
abl_rows = []
for d in DS:
    r = res[d]
    f_, a_ = r[r.method == "FCS-C"], r[r.method == ABL]
    t = tests[d].loc[ABL]
    pv = lambda x: (r"\textbf{" + f"{x:.3f}" + "}") if x < 0.05 else f"{x:.3f}"
    abl_rows.append(
        f"{DS_NAME[d]} & {f_.cost_opt.mean():.4f} ({f_.cost_opt.std():.3f}) & {a_.cost_opt.mean():.4f} ({a_.cost_opt.std():.3f}) & "
        f"{t.cost_opt_diff:+.4f} & {pv(t.cost_opt_p)} & {pv(t.cost_opt_p_wilcoxon)} & "
        f"{f_.auc.mean():.4f} & {a_.auc.mean():.4f} & {t.auc_diff:+.4f} & {pv(t.auc_p)} & "
        f"{100 * diag[d].active.mean():.0f}\\% \\\\")
abl_tab = "\n".join([
    r"\begin{table}[htbp]", r"\centering",
    r"\caption{Ablation of the Fisher bound: FCS-C ($\alpha_{rel}>0$ tuned) against the same model with "
    r"$\alpha=0$ (Fisher bound reduced to $d^\top w\ge 0$, scale $s$ tuned on the same grid), over the same 25 "
    r"outer folds. $\Delta$: FCS-C minus ablation (negative cost difference favours FCS-C); $p$: corrected "
    r"resampled $t$-test; $p_W$: Wilcoxon. Last column: share of folds in which the bound selected by FCS-C is active.}",
    r"\label{tab:ablation}", r"\scriptsize\setlength{\tabcolsep}{3pt}",
    r"\begin{tabular}{lcccccccccc}", r"\toprule",
    r" & \multicolumn{5}{c}{Realized cost} & \multicolumn{4}{c}{AUC} & \\",
    r"\cmidrule(lr){2-6}\cmidrule(lr){7-10}",
    r"Dataset & FCS-C & $\alpha=0$ & $\Delta$ & $p$ & $p_W$ & FCS-C & $\alpha=0$ & $\Delta$ & $p$ & Active \\",
    r"\midrule", *abl_rows, r"\bottomrule", r"\end{tabular}", r"\end{table}"])

# cross-dataset summary with average ranks (cost)
ranks = pd.concat([res[d].groupby("method").cost_opt.mean().reindex(METHOD_ORDER).rank() for d in DS], axis=1)
ranks.columns = DS
ranks = ranks.fillna(len(METHOD_ORDER))       # a degenerate method takes the last rank
ranks["avg"] = ranks.mean(axis=1)
cross_rows = []
for m in ranks.sort_values("avg").index:
    cells = []
    for d in DS:
        g = res[d][res[d].method == m]
        if g.empty:
            cells += [r"degen.$^{\ddagger}$", "--"]
            continue
        st = "" if m == "FCS-C" or m not in tests[d].index or tests[d].loc[m, "cost_opt_p"] >= 0.05 else "$^{*}$"
        cells.append(f"{g.cost_opt.mean():.4f}{st}")
        st = "" if m == "FCS-C" or m not in tests[d].index or tests[d].loc[m, "auc_p"] >= 0.05 else "$^{*}$"
        cells.append(f"{g.auc.mean():.4f}{st}")
    cross_rows.append(name(m) + " & " + " & ".join(cells) + f" & {ranks.loc[m, 'avg']:.2f} \\\\")
cross_tab = "\n".join([
    r"\begin{table}[htbp]", r"\centering",
    r"\caption{Cross-dataset summary: mean realized cost (cost-optimal cut-off) and mean AUC; "
    r"methods sorted by average rank of cost. $^{*}$: significantly different from FCS-C "
    r"(corrected resampled $t$-test, $p<0.05$). $^{\ddagger}$: degenerate solution ($w=0$) in every fold, "
    r"counted as last rank.}",
    r"\label{tab:cross}",
    r"\resizebox{\textwidth}{!}{%",
    r"\begin{tabular}{lccccccc}", r"\toprule",
    " & " + " & ".join(r"\multicolumn{2}{c}{" + DS_NAME[d].replace(" Credit", "")
                        + (r"$^{\dagger}$" if DEGEN[d] else "") + "}" for d in DS) + r" & \\",
    r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
    r"Method & Cost & AUC & Cost & AUC & Cost & AUC & Avg.\ rank (cost) \\", r"\midrule",
    *cross_rows, r"\bottomrule", r"\end{tabular}}", r"\end{table}"])

# selected FCS-C hyper-parameters
hp_rows = []
for d in DS:
    g = diag[d]
    hp_rows.append(f"{DS_NAME[d]} & {g.s.median():g} & "
                   f"{g.alpha_rel.mean():.2f} ({g.alpha_rel.min():g}--{g.alpha_rel.max():g}) & "
                   f"{g.solver.value_counts().idxmax()} ({(g.solver == g.solver.value_counts().idxmax()).mean():.0%}) \\\\".replace('%)', r'\%)'))
hp_tab = "\n".join([
    r"\begin{table}[htbp]", r"\centering",
    r"\caption{FCS-C hyper-parameters selected by the inner cross-validation over the outer folds: "
    r"median $s$ and mean (range) of $\alpha_{rel}$.}",
    r"\label{tab:hp}", r"\small",
    r"\begin{tabular}{lccc}", r"\toprule",
    r"Dataset & median $s$ & $\alpha_{rel}$ & solver \\", r"\midrule",
    *hp_rows, r"\bottomrule", r"\end{tabular}", r"\end{table}"])

opt_cols = ["auc", "cost_opt", "gmean_opt", "rec_good_opt", "rec_bad_opt", "acc_opt", "f1_good_opt", "f1_bad_opt"]
nat_cols = ["cost_native", "gmean_native", "rec_good_native", "rec_bad_native", "acc_native"]

sections = []
for d in DS:
    sections.append(r"\subsection{" + DS_NAME[d] + r" ($c_{FP}:c_{FN}$ = " + DS_INFO[d]["cost"] + ")}")
    if DEGEN[d]:
        sections.append(
            r"\paragraph{Degenerate hinge-loss solutions.} On this dataset the selected solutions of FCS-C, "
            r"FCS-C ($\alpha=0$), CS-SVM and SVM/LDA have $\|w\|\approx 0$ in "
            f"{int((diag[d].wnorm < DEGENERATE_THR).sum())} of {len(diag[d])} outer folds: "
            r"the cost-weighted hinge loss is minimised by a constant classifier for every value of $s$ "
            r"(SVM degeneracy \cite{Rifkin1999}), and the Fisher cone does not exclude $w=0$ "
            r"(Proposition~3). The ranking these models produce comes from numerically negligible weight "
            r"vectors; their rows are marked $^{\dagger}$ and are not valid estimates of the methods' performance.")
    sections.append(method_table(d, opt_cols, {"cost_opt"},
                                 DS_NAME[d] + r": test performance at the cost-optimal cut-off (acceptance rate selected on "
                                 r"inner out-of-fold scores). Mean (sd) over " + f"{len(diag[d])}" + r" outer folds. "
                                 r"Bold: best valid method per column; $^{*}$: significantly different from FCS-C "
                                 r"(corrected resampled $t$-test, $p<0.05$; AUC and cost only).",
                                 "tab:opt_" + d))
    sections.append(method_table(d, nat_cols, {"cost_native"},
                                 DS_NAME[d] + r": test performance at each method's native decision rule "
                                 r"(hyperplane models: $w^\top x\ge b$; probabilistic models: $P(\text{Good})\ge 0.5$, "
                                 r"or $\ge c_{FP}/(c_{FP}+c_{FN})$ for R-LDA (Bayes cost)). Mean (sd).",
                                 "tab:nat_" + d, op="native", show_tests=False))
    sections.append(tests_table(d))
    sections.append(r"\begin{figure}[htbp]\centering\includegraphics[width=\textwidth]{" + figs[d] +
                    r"}\caption{" + DS_NAME[d] + r": mean test AUC and realized cost ($\pm$ sd over outer folds). "
                    + (r"Grey bars ($\dagger$): degenerate solutions." if DEGEN[d] else "")
                    + r"}\label{fig:" + d + r"}\end{figure}")
    sections.append(r"\clearpage")

# ---- hyper-parameter grids actually used (read from the pipeline's grid profile)
import fcsc_pipeline as FP
_b = os.path.basename(RES)
PROFILE = "wide60" if "wide60" in _b else ("compact" if "compact" in _b else "base")
G = FP.GRIDS[PROFILE]
sizes = {k: len(v["grid"]) for k, v in FP.build_methods(1.0, 1.0, G).items()}
fmtl = lambda v: ", ".join("\\infty" if x is None else ("\\text{LW}" if x == "auto" else f"{x:g}") for x in v)
GRID_DESC = {
    "FCS-C": rf"$s\in\{{{fmtl(G['s'])}\}}$; $\rho\in\{{{fmtl(G['rho'])}\}}$; $\alpha_{{rel}}\in\{{{fmtl(G['alpha'])}\}}$",
    "FCS-C (alpha=0)": rf"$s$, $\rho$ as FCS-C",
    "CS-SVM": rf"$s$, $\rho$ as FCS-C",
    "SVM/LDA": rf"$s$ as FCS-C; $\lambda\in\{{{fmtl(G['lam'])}\}}$",
    "R-LDA": rf"shrinkage $\in\{{{fmtl(G['shrink'])}\}}$",
    "R-LDA (Bayes cost)": "as R-LDA",
    "LR": rf"$C\in\{{{fmtl(G['C'])}\}}$",
    "CS-LR": rf"$C$ as LR; $\rho\in\{{{fmtl(G['cs_rho'])}\}}$",
    "C4.5": rf"depth $\in\{{{fmtl(G['depth'])}\}}$; min leaf $\in\{{{fmtl(G['leaf'])}\}}$",
    "CART": "as C4.5",
    "CS-C4.5": rf"as C4.5; $\rho\in\{{{fmtl(G['tree_rho'])}\}}$",
    "CS-CART": "as CS-C4.5",
    "MetaCost": rf"min leaf $\in\{{{fmtl(G['mc_leaf'])}\}}$; bags $\in\{{{fmtl(G['mc_bags'])}\}}$",
    "CS-AdaBoost": rf"depth $\in\{{{fmtl(G['ada_depth'])}\}}$; rounds $\in\{{{fmtl(G['ada_n'])}\}}$; learning rate $\in\{{{fmtl(G['ada_lr'])}\}}$",
    "LightGBM": rf"leaves $\in\{{{fmtl(G['lgb_leaves'])}\}}$; min child $\in\{{{fmtl(G['lgb_child'])}\}}$; learning rate $\in\{{{fmtl(G['lgb_lr'])}\}}$; $\rho\in\{{{fmtl(G['lgb_rho'])}\}}$",
}
grid_tab = "\n".join([
    r"\begin{table}[htbp]", r"\centering",
    r"\caption{Hyper-parameter grids (profile \texttt{" + PROFILE + r"}), tuned by the inner cross-validation. "
    r"$\rho$ multiplies the Bad-class training weight relative to the costs (the evaluation cost is unchanged); "
    r"FCS-C, its ablation and CS-SVM keep balanced weights ($\rho=1$), which guarantees a non-degenerate solution. "
    r"LW: Ledoit--Wolf shrinkage.}",
    r"\label{tab:grids}", r"\footnotesize",
    r"\begin{tabular}{lrp{11.4cm}}", r"\toprule",
    r"Method & Configs & Grid \\", r"\midrule",
    *[f"{name(m)} & {sizes[m]} & {GRID_DESC[m]} \\\\" for m in METHOD_ORDER],
    r"\midrule", f"Total & {sum(sizes[m] for m in METHOD_ORDER)} & \\\\",
    r"\bottomrule", r"\end{tabular}", r"\end{table}"])

# ---- execution time
def time_tables():
    out = []
    if "fit_seconds" in res[DS[0]]:
        cols = " & ".join([r"\multicolumn{3}{c}{" + DS_NAME[d] + "}" for d in DS])
        L = [r"\begin{table}[htbp]", r"\centering",
             r"\caption{Execution time within the cross-validation (mean over the outer folds, wall-clock with "
             r"one worker per physical core): \emph{tune} = inner cross-validation over the whole grid (s), "
             r"\emph{fit} = refit of the selected configuration on the outer-training part (s), "
             r"\emph{pred} = prediction time (ms per 1\,000 applicants). Configs: grid size.}",
             r"\label{tab:time_cv}", r"\scriptsize\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{lr" + "rrr" * len(DS) + "}", r"\toprule",
             r" & & " + cols + r" \\",
             "".join(rf"\cmidrule(lr){{{3 + 3 * i}-{5 + 3 * i}}}" for i in range(len(DS))),
             r"Method & Configs & " + " & ".join(["tune & fit & pred"] * len(DS)) + r" \\", r"\midrule"]
        for m in METHOD_ORDER:
            cells = []
            for d in DS:
                g = res[d][res[d].method == m]
                cells += ([f"{g.tune_seconds.mean():.1f}", f"{g.fit_seconds.mean():.2f}",
                           f"{g.predict_ms_per_1000.mean():.2f}"] if len(g) else ["--"] * 3)
            nc = int(pd.concat([res[d][res[d].method == m] for d in DS]).n_configs.iloc[0])
            L.append(name(m) + f" & {nc} & " + " & ".join(cells) + r" \\")
        L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        out.append("\n".join(L))
    clean = os.path.join(RES, "timing_clean.csv")
    if os.path.exists(clean):
        t = pd.read_csv(clean)
        cols = " & ".join([r"\multicolumn{2}{c}{" + DS_NAME[d] + "}" for d in DS])
        L = [r"\begin{table}[htbp]", r"\centering",
             r"\caption{Execution time measured alone on the machine, single-threaded: the configuration "
             r"selected most often by the cross-validation is fitted on 80\% of the data and scored on the "
             r"remaining 20\% (median of three runs). \emph{fit} in seconds, \emph{pred} in ms per 1\,000 "
             r"applicants. For FCS-C the fit includes the $\alpha=0$ solve used by Proposition~5.}",
             r"\label{tab:time_clean}", r"\small",
             r"\begin{tabular}{l" + "rr" * len(DS) + "}", r"\toprule",
             r" & " + cols + r" \\",
             "".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(DS))),
             r"Method & " + " & ".join(["fit & pred"] * len(DS)) + r" \\", r"\midrule"]
        for m in METHOD_ORDER:
            cells = []
            for d in DS:
                g = t[(t.dataset == d) & (t.method == m)]
                cells += ([f"{g.fit_seconds.iloc[0]:.3f}", f"{g.predict_ms_per_1000.iloc[0]:.3f}"]
                          if len(g) else ["--", "--"])
            L.append(name(m) + " & " + " & ".join(cells) + r" \\")
        L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        out.append("\n".join(L))
    return "\n\n".join(out)


time_tex = time_tables()

tex = r"""\documentclass[11pt,a4paper]{article}
\usepackage[margin=2cm]{geometry}
\usepackage{booktabs,graphicx,amsmath,amssymb,float}
\usepackage[hidelinks]{hyperref}
\title{FCS-C: Experimental Results\\\large Revision of AODS-D-26-00749}
\date{\today}
\begin{document}
\maketitle

\section{Protocol}
All methods are evaluated by repeated stratified $5$-fold cross-validation ($5$ repeats, $25$ outer folds) with
nested stratified $5$-fold tuning on the outer-training part only. Preprocessing (one-hot encoding of nominal
variables, default-rate tertile dummies for the LendingClub state, standardisation) is fitted on each training part.
Misclassification costs follow the scikit-learn balanced class-weight formula, $c_k=n/(2n_k)$, computed on each
training fold: $c_{FP}=n/(2n_{\text{Bad}})$ for a Bad applicant accepted and $c_{FN}=n/(2n_{\text{Good}})$ for a Good
applicant rejected (Good is the positive class). Every method uses these costs and has its own hyper-parameter grid. Hyper-parameters are selected by the realized cost
at the cost-optimal cut-off on pooled inner out-of-fold scores, expressed as training-score quantiles (an acceptance
rate); the same acceptance rate is applied to the test fold. Realized cost is $(c_{FP}\,FP+c_{FN}\,FN)/n_{\text{test}}$.
FCS-C solves the restricted SOCP (P$_R$) with the Clarabel interior-point solver,
\[
  \min_{w,b,\varepsilon}\ \frac{s}{n}\sum_i c_{k(i)}\,\varepsilon_i + w^\top\Sigma w
  \quad\text{s.t.}\quad y_i(w^\top x_i-b)\ge 1-\varepsilon_i,\ \varepsilon_i\ge0,\
  d^\top w\ge\sqrt{\alpha}\,\|\Sigma^{1/2}w\|_2,
\]
with $s$ and $\alpha_{rel}=\alpha/\alpha_{\max}>0$ tuned (Table~\ref{tab:grids}); the case $\alpha=0$ is
reported separately as an ablation (Table~\ref{tab:ablation}). Baselines: CS-SVM ($\Sigma$ replaced by $I$, no cone), SVM/LDA
\cite{Xiong2005}, regularised LDA (shrinkage tuned) at the default and at the Bayes cost threshold, LR, CS-LR,
C4.5, CART and their cost-sensitive versions (depth and leaf size tuned), MetaCost \cite{Domingos1999},
cost-sensitive AdaBoost with cost-proportional initial weights \cite{Ting2000}, and LightGBM with cost class weights.

""" + setup_tab + "\n\n" + grid_tab + "\n\n" + hp_tab + r"""

\section{Results by dataset}
""" + "\n\n".join(sections) + r"""

\section{Cross-dataset summary}
""" + cross_tab + r"""

\section{Execution time}
""" + time_tex + r"""

\section{Ablation and sensitivity to the Fisher bound}
In the benchmark, FCS-C is the model with an active Fisher bound ($\alpha_{rel}>0$). This section isolates the
bound: Table~\ref{tab:ablation} compares FCS-C with the same model at $\alpha=0$, and Table~\ref{tab:sensitivity}
and Figure~\ref{fig:alpha} trace the test performance over $\alpha_{rel}\in[0,0.99]$. By Proposition~5 the
solution does not depend on $\alpha$ as long as $\alpha\le J_0$; the curves can only change to the right of the
dashed line $J_0/\alpha_{\max}$.

""" + abl_tab + "\n\n" + sens_tab + r"""

\begin{figure}[htbp]\centering\includegraphics[width=\textwidth]{""" + fig_a + r"""}
\caption{Sensitivity of FCS-C to $\alpha_{rel}$: realized cost (top) and AUC (bottom), mean $\pm$ s.e. over the
25 outer folds. Dashed line and shaded area: mean $J_0/\alpha_{\max}$ and the region where the Fisher bound is
active.}\label{fig:alpha}\end{figure}

""" + fisher_tab + r"""

\begin{thebibliography}{9}
\bibitem{Domingos1999} P.~Domingos. MetaCost: a general method for making classifiers cost-sensitive.
  \emph{Proc.\ 5th ACM SIGKDD}, 155--164, 1999.
\bibitem{NadeauBengio2003} C.~Nadeau and Y.~Bengio. Inference for the generalization error.
  \emph{Machine Learning}, 52(3):239--281, 2003.
\bibitem{Rifkin1999} R.~Rifkin, M.~Pontil and A.~Verri. A note on support vector machine degeneracy.
  \emph{Proc.\ Algorithmic Learning Theory (ALT)}, LNCS 1720, 252--263, 1999.
\bibitem{Ting2000} K.~M.~Ting. A comparative study of cost-sensitive boosting algorithms.
  \emph{Proc.\ 17th ICML}, 983--990, 2000.
\bibitem{Xiong2005} T.~Xiong and V.~Cherkassky. A combined SVM and LDA approach for classification.
  \emph{Proc.\ IJCNN}, 1455--1459, 2005.
\end{thebibliography}
\end{document}
"""
path = os.path.join(OUT, "FCS-C_results.tex")
with open(path, "w", encoding="utf8") as f:
    f.write(tex)
print("wrote", path)
