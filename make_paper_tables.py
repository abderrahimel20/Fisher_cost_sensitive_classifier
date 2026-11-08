"""
Write the result tables of Sections 4-5 of sn-article.tex (blue, paper style)
from a results folder, and copy the figures next to the paper.

    python make_paper_tables.py results_final   ->  results_final/paper_tables.tex, ../fig_*.pdf

The benchmark tables report FCS-C with an active Fisher bound (alpha_rel > 0);
the alpha = 0 variant appears only in the ablation and sensitivity tables.
"""
import os
import shutil
import sys

import numpy as np
import pandas as pd

import fcsc_pipeline as FP

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, sys.argv[1] if len(sys.argv) > 1 else "results_final")
GRID = sys.argv[2] if len(sys.argv) > 2 else "compact"
PAPER_DIR = os.path.dirname(HERE)
DS = ["australian", "german", "lendingclub"]
NAME = {"australian": "Australian Credit", "german": "German Credit", "lendingclub": "LendingClub"}
SHORT = {"australian": "aus", "german": "ger", "lendingclub": "lc"}
ABL = "FCS-C (alpha=0)"
ORDER = ["FCS-C", "CS-SVM", "SVM/LDA", "R-LDA", "R-LDA (Bayes cost)", "LR", "CS-LR", "C4.5", "CART",
         "CS-C4.5", "CS-CART", "MetaCost", "CS-AdaBoost", "LightGBM"]
TEX = {"FCS-C": r"\textbf{FCS-C}", ABL: r"FCS-C ($\alpha=0$)", "R-LDA (Bayes cost)": "R-LDA (Bayes)"}
DEGEN = r"degenerate ($w=0$) in every fold"

res = {d: pd.read_csv(os.path.join(RES, f"{d}_folds.csv")) for d in DS}
diag = {d: pd.read_csv(os.path.join(RES, f"{d}_fcsc_diagnostics.csv")) for d in DS}
curve = {d: pd.read_csv(os.path.join(RES, f"{d}_alpha_curve.csv")) for d in DS}
tests = {d: pd.read_csv(os.path.join(RES, f"{d}_tests_vs_fcsc.csv")).set_index("method") for d in DS}
timing = pd.read_csv(os.path.join(RES, "timing_clean.csv"))
out = []


def blue_table(label, caption, colspec, header, rows, size=r"\small", sep="4pt", place="htbp"):
    return "\n".join([
        rf"\begin{{table}}[{place}]", r"\centering",
        r"\caption{\textcolor{blue}{" + caption + "}}", rf"\label{{{label}}}",
        rf"{{\color{{blue}}{size}\setlength{{\tabcolsep}}{{{sep}}}", rf"\begin{{tabular}}{{{colspec}}}",
        r"\toprule", *header, r"\midrule", *rows, r"\bottomrule", r"\end{tabular}}", r"\end{table}"])


pv = lambda x: (r"\textbf{" + f"{x:.3f}" + "}") if x < 0.05 else f"{x:.3f}"

# ---- datasets
rows = []
counts = {"australian": (307, 383), "german": (700, 300), "lendingclub": (16094, 3906)}
for d in DS:
    g, b = counts[d]
    n = g + b
    rows.append((f"{NAME[d]} & {n:,} & {int(diag[d].r.median())} & {g:,} ({100 * g / n:.1f}\\%) & "
                 f"{b:,} ({100 * b / n:.1f}\\%) & {max(g, b) / min(g, b):.2f}:1 & {n / (2 * b):.3f} & "
                 f"{n / (2 * g):.3f} \\\\").replace(",", r"\,"))
out.append("% ---- datasets\n" + blue_table(
    "tab:datasets",
    r"The three evaluation datasets. $r$: number of features after encoding; IR: imbalance ratio; "
    r"$c_{FP}=n/(2n_B)$ and $c_{FN}=n/(2n_G)$: balanced misclassification costs on the full data (recomputed on "
    r"each training fold). Australian and German: UCI Statlog \cite{UCI2024}; LendingClub: stratified sample of "
    r"the 1\,266\,782 completed accepted loans (2007--2018).",
    "lrrrrccc",
    [r"\textbf{Dataset} & $n$ & $r$ & \textbf{Good (\%)} & \textbf{Bad (\%)} & \textbf{IR} & $c_{FP}$ & $c_{FN}$ \\"],
    rows))

# ---- hyper-parameter grids
G = FP.GRIDS[GRID]
sizes = {k: len(v["grid"]) for k, v in FP.build_methods(1.0, 1.0, G).items()}
fl = lambda v: ", ".join(r"\infty" if x is None else (r"\text{LW}" if x == "auto" else f"{x:g}") for x in v)
desc = {
    "FCS-C": rf"$s\in\{{{fl(G['s'])}\}}$, $\alpha_{{\text{{rel}}}}\in\{{{fl(G['alpha'])}\}}$",
    "CS-SVM": r"$s$ as FCS-C",
    "SVM/LDA": rf"$s\in\{{{fl(G.get('svmlda_s', G['s']))}\}}$, $\lambda\in\{{{fl(G['lam'])}\}}$",
    "R-LDA": rf"shrinkage $\in\{{{fl(G['shrink'])}\}}$ (LW: Ledoit--Wolf)",
    "R-LDA (Bayes cost)": "as R-LDA",
    "LR": rf"$C\in\{{{fl(G['C'])}\}}$",
    "CS-LR": rf"$C$ as LR, $\rho\in\{{{fl(G['cs_rho'])}\}}$",
    "C4.5": rf"depth $\in\{{{fl(G['depth'])}\}}$, min.\ leaf $\in\{{{fl(G['leaf'])}\}}$",
    "CART": "as C4.5",
    "CS-C4.5": rf"as C4.5, $\rho\in\{{{fl(G['tree_rho'])}\}}$",
    "CS-CART": "as CS-C4.5",
    "MetaCost": rf"min.\ leaf $\in\{{{fl(G['mc_leaf'])}\}}$, {G['mc_bags'][0]} bags",
    "CS-AdaBoost": rf"depth $\in\{{{fl(G['ada_depth'])}\}}$, rounds $\in\{{{fl(G['ada_n'])}\}}$",
    "LightGBM": rf"leaves $\in\{{{fl(G['lgb_leaves'])}\}}$, min.\ child $\in\{{{fl(G['lgb_child'])}\}}$, "
                rf"$\rho\in\{{{fl(G['lgb_rho'])}\}}$",
}
rows = [f"{TEX.get(m, m)} & {sizes[m]} & {desc[m]} \\\\" for m in ORDER]
rows += [r"\midrule", rf"{TEX[ABL]} (ablation) & {sizes[ABL]} & $s$ as FCS-C, $\alpha=0$ \\"]
out.append("% ---- grids\n" + blue_table(
    "tab:grids",
    r"Hyper-parameter grids, tuned by the inner cross-validation. $\rho$ multiplies the Bad-class training weight "
    r"relative to the costs (the evaluation cost is unchanged). FCS-C and CS-SVM keep balanced weights ($\rho=1$), "
    r"which guarantees a non-degenerate solution (Proposition~\ref{prop:nondegenerate}).",
    "lrp{8.3cm}", [r"\textbf{Method} & \textbf{Configs} & \textbf{Grid} \\"], rows, size=r"\footnotesize", sep="3pt"))

# ---- selected FCS-C hyper-parameters
rows = []
for d in DS:
    g = diag[d]
    cnt = g.alpha_rel.value_counts()
    rows.append(f"{NAME[d]} & {g.s.median():g} & {g.alpha_rel.median():g} & " +
                ", ".join(f"{a:g} ({cnt[a]})" for a in sorted(cnt.index)) + r" \\")
out.append("% ---- selected hyper-parameters\n" + blue_table(
    "tab:hyperparams",
    r"FCS-C hyper-parameters selected by the inner cross-validation: median over the 25 outer folds, and number of "
    r"folds selecting each value of $\alpha_{\text{rel}}$.",
    "lccl", [r"\textbf{Dataset} & $s$ & $\alpha_{\text{rel}}$ & \textbf{folds per} $\alpha_{\text{rel}}$ \\"], rows))

# ---- Fisher diagnostics
rows = []
for d in DS:
    g = diag[d]
    rows.append(f"{NAME[d]} & {g.alpha_max.mean():.3f} & {np.sqrt(g.alpha_max).mean():.3f} & {g.kappa.mean():.1f} & "
                f"{g.J0.mean():.3f} ({g.J0_rel.mean():.3f}) & {g.J_achieved.mean():.3f} ({(g.J_achieved / g.alpha_max).mean():.3f}) & "
                f"{100 * g.active.mean():.0f}\\% \\\\")
out.append("% ---- Fisher pre-analysis\n" + blue_table(
    "tab:fisher_pre",
    r"Fisher diagnostics of FCS-C, mean over the 25 outer folds. $\kappa$: condition number of $\Sigma_\delta$; "
    r"$J_0$: Fisher ratio of the $\alpha=0$ solution and $J$: Fisher ratio achieved by FCS-C (in brackets, divided "
    r"by $\alpha_{\max}$); last column: share of folds in which the selected Fisher bound is active ($\alpha>J_0$, "
    r"Proposition~\ref{prop:active}).",
    "lcccccc", [r"\textbf{Dataset} & $\alpha_{\max}$ & \textbf{Mahal.} & $\kappa(\Sigma_\delta)$ & $J_0$ & $J$ & \textbf{active} \\"],
    rows, size=r"\footnotesize", sep="3pt"))


# ---- per-dataset benchmark
def results_table(d):
    cols = [("auc", "AUC", max), ("cost_opt", "Cost", min), ("gmean_opt", "G-Mean", max),
            ("rec_good_opt", r"Rec\textsubscript{G}", max), ("rec_bad_opt", r"Rec\textsubscript{B}", max)]
    r = res[d][res[d].method != ABL]
    m, s = r.groupby("method").mean(numeric_only=True), r.groupby("method").std(numeric_only=True)
    best = {c: (m[c].idxmax() if f is max else m[c].idxmin()) for c, _, f in cols}
    rows = []
    for meth in ORDER:
        if meth not in m.index:
            rows.append(TEX.get(meth, meth) + rf" & \multicolumn{{{len(cols)}}}{{c}}{{{DEGEN}}} \\")
            continue
        cells = []
        for c, _, _ in cols:
            dg = 4 if c == "cost_opt" else 3
            v = f"{m.loc[meth, c]:.{dg}f} ({s.loc[meth, c]:.3f})".replace("(0.", "(.")
            if meth == best[c]:
                v = r"\textbf{" + v + "}"
            pk = {"auc": "auc_p", "cost_opt": "cost_opt_p"}.get(c)
            if pk and meth != "FCS-C" and tests[d].loc[meth, pk] < 0.05:
                v += r"$^{*}$"
            cells.append(v)
        rows.append(TEX.get(meth, meth) + " & " + " & ".join(cells) + r" \\")
        if meth == "FCS-C":
            rows.append(r"\midrule")
    cfp, cfn = r.c_fp.mean(), r.c_fn.mean()
    return blue_table(
        "tab:results_" + SHORT[d],
        NAME[d] + r": test performance over 25 outer folds ($5\times5$ repeated stratified cross-validation), "
        r"mean (sd), at the cost-optimal cut-off. Costs $c_{FP}=" + f"{cfp:.3f}" + r"$, $c_{FN}=" + f"{cfn:.3f}" +
        r"$. Bold: best per column. $^{*}$: significantly different from FCS-C (corrected resampled $t$-test, "
        r"$p<0.05$; AUC and cost).",
        "l" + "c" * len(cols), [r"\textbf{Method} & " + " & ".join(r"\textbf{" + h + "}" for _, h, _ in cols) + r" \\"],
        rows, size=r"\scriptsize", sep="2pt")


for d in DS:
    out.append(f"% ---- results {d}\n" + results_table(d))

# ---- cross-dataset summary (degenerate method = last rank)
ranks = pd.concat([res[d][res[d].method != ABL].groupby("method").cost_opt.mean().reindex(ORDER).rank()
                   for d in DS], axis=1).fillna(len(ORDER)).mean(axis=1)
rows = []
for meth in ranks.sort_values().index:
    cells = []
    for d in DS:
        g = res[d][res[d].method == meth]
        if g.empty:
            cells += [r"degen.$^{\ddagger}$", "--"]
            continue
        for k, pk in (("cost_opt", "cost_opt_p"), ("auc", "auc_p")):
            st = r"$^{*}$" if meth != "FCS-C" and tests[d].loc[meth, pk] < 0.05 else ""
            cells.append(f"{g[k].mean():.4f}{st}")
    rows.append(TEX.get(meth, meth) + " & " + " & ".join(cells) + f" & {ranks[meth]:.2f} \\\\")
out.append("% ---- cross-dataset\n" + blue_table(
    "tab:cross_dataset",
    r"Cross-dataset summary: mean realized cost at the cost-optimal cut-off and mean AUC; methods sorted by their "
    r"average rank on cost. $^{*}$: significantly different from FCS-C (corrected resampled $t$-test, $p<0.05$). "
    r"$^{\ddagger}$: degenerate solution ($w=0$) in every fold, counted as last rank. The table lists FCS-C and the thirteen benchmark methods; regularised LDA appears twice, at its posterior and Bayes cost thresholds.",
    "lccccccc",
    [r" & \multicolumn{2}{c}{\textbf{Australian}} & \multicolumn{2}{c}{\textbf{German}} & \multicolumn{2}{c}{\textbf{LendingClub}} & \\",
     r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
     r"\textbf{Method} & Cost & AUC & Cost & AUC & Cost & AUC & \textbf{Avg.\ rank} \\"],
    rows, sep="4pt"))

# ---- execution time
rows = []
for meth in ORDER:
    cells = []
    for d in DS:
        g = res[d][res[d].method == meth]
        t = timing[(timing.dataset == d) & (timing.method == meth)]
        if g.empty or t.empty:
            cells += ["--"] * 3
        else:
            cells += [f"{g.tune_seconds.mean():.1f}", f"{t.fit_seconds.iloc[0]:.3f}",
                      f"{t.predict_ms_per_1000.iloc[0]:.2f}"]
    rows.append(TEX.get(meth, meth) + " & " + " & ".join(cells) + r" \\")
out.append("% ---- execution time\n" + blue_table(
    "tab:time",
    r"Execution time. \emph{Tune}: inner cross-validation over the whole grid, mean wall-clock seconds per outer "
    r"fold (one worker per physical core). \emph{Fit}: fit of the configuration selected most often, on 80\% of the "
    r"data, alone on the machine and single-threaded (seconds, median of three runs). \emph{Pred}: prediction time "
    r"(ms per 1\,000 applicants). For FCS-C the fit includes the $\alpha=0$ solve used by "
    r"Proposition~\ref{prop:active}.",
    "l" + "rrr" * len(DS),
    [" & " + " & ".join(r"\multicolumn{3}{c}{\textbf{" + NAME[d] + "}}" for d in DS) + r" \\",
     "".join(rf"\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(DS))),
     r"\textbf{Method} & " + " & ".join(["Tune & Fit & Pred"] * len(DS)) + r" \\"],
    rows, size=r"\scriptsize", sep="3pt"))

# ---- ablation
rows = []
for d in DS:
    f_, a_ = res[d][res[d].method == "FCS-C"], res[d][res[d].method == ABL]
    t = tests[d].loc[ABL]
    rows.append(f"{NAME[d]} & {f_.cost_opt.mean():.4f} & {a_.cost_opt.mean():.4f} & {t.cost_opt_diff:+.4f} & "
                f"{pv(t.cost_opt_p)} & {pv(t.cost_opt_p_wilcoxon)} & {f_.auc.mean():.4f} & {a_.auc.mean():.4f} & "
                f"{t.auc_diff:+.4f} & {pv(t.auc_p)} & {100 * diag[d].active.mean():.0f}\\% \\\\")
out.append("% ---- ablation\n" + blue_table(
    "tab:ablation",
    r"Ablation of the Fisher bound: FCS-C ($\alpha_{\text{rel}}>0$ tuned) against the same model with $\alpha=0$ "
    r"(the bound reduced to $d^\top w\geq 0$, scale $s$ tuned on the same grid), on the same 25 outer folds. "
    r"$\Delta$: FCS-C minus ablation; $p$: corrected resampled $t$-test; $p_W$: Wilcoxon; last column: share of "
    r"folds in which the bound selected by FCS-C is active.",
    "lcccccccccc",
    [r" & \multicolumn{5}{c}{\textbf{Realized cost}} & \multicolumn{4}{c}{\textbf{AUC}} & \\",
     r"\cmidrule(lr){2-6}\cmidrule(lr){7-10}",
     r"\textbf{Dataset} & FCS-C & $\alpha=0$ & $\Delta$ & $p$ & $p_W$ & FCS-C & $\alpha=0$ & $\Delta$ & $p$ & \textbf{active} \\"],
    rows, size=r"\scriptsize", sep="1.5pt"))

# ---- sensitivity to alpha_rel
alphas = [a for a in sorted(curve[DS[0]].alpha_rel.unique())
          if a in (0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99)]
rows = []
for a in alphas:
    cells = []
    for d in DS:
        c = curve[d][curve[d].alpha_rel == a]
        cells += [f"{c.cost_opt.mean():.4f}", f"{c.auc.mean():.4f}", f"{c.J_rel.mean():.3f}", f"{100 * c.active.mean():.0f}"]
    rows.append(f"{a:g} & " + " & ".join(cells) + r" \\")
out.append("% ---- sensitivity\n" + blue_table(
    "tab:sensitivity",
    r"Sensitivity of FCS-C to $\alpha_{\text{rel}}$: in every outer fold FCS-C is refitted for each value with the "
    r"other hyper-parameters at their selected values and evaluated on the test fold at the selected acceptance rate "
    r"(mean over the 25 outer folds). Act.: share of folds (\%) in which the bound is active. The full curve "
    r"($\alpha_{\text{rel}}=0, 0.05, \dots, 0.95, 0.99$) is shown in Figure~\ref{fig:alpha_curve}.",
    "l" + "rrrr" * len(DS),
    [" & " + " & ".join(r"\multicolumn{4}{c}{\textbf{" + NAME[d] + "}}" for d in DS) + r" \\",
     "".join(rf"\cmidrule(lr){{{2 + 4 * i}-{5 + 4 * i}}}" for i in range(len(DS))),
     r"$\alpha_{\text{rel}}$ & " + " & ".join([r"Cost & AUC & $J/\alpha_{\max}$ & Act."] * len(DS)) + r" \\"],
    rows, size=r"\scriptsize", sep="2pt"))

with open(os.path.join(RES, "paper_tables.tex"), "w", encoding="utf8") as f:
    f.write("\n\n".join(out) + "\n")
for fn in ["fig_australian_overview.pdf", "fig_german_overview.pdf", "fig_lendingclub_overview.pdf",
           "fig_alpha_curve.pdf"]:
    shutil.copy(os.path.join(RES, "report", fn), os.path.join(PAPER_DIR, fn))
print("wrote", os.path.join(RES, "paper_tables.tex"), "and copied figures to", PAPER_DIR)
