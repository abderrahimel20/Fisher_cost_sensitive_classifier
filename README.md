# FCS-C: A Fisher Cost-Sensitive Classifier for Imbalanced Credit Scoring

Code and results for the paper of the same title, *Annals of Data Science* (manuscript AODS-D-26-00749).

This repository contains the code and the fold-level experimental results that support
Sections 4 and 5 of the paper. Every table and figure of these sections can be regenerated
from the files provided here.

## Contents

| Path | Description |
|---|---|
| `code/fcsc_pipeline.py` | Complete experimental pipeline: data loading and fold-safe pre-processing, the FCS-C solver (restricted SOCP (P_R), Clarabel interior-point solver), the thirteen benchmark methods, the repeated nested cross-validation, the cost-optimal cut-off, the significance tests, the Fisher diagnostics and the sensitivity study. |
| `code/FCS-C_experiments.ipynb` | The same pipeline as a single self-contained Jupyter notebook, with tables and figures for each dataset. |
| `code/verify_theory.py` | Numerical verification of the model and of Propositions 2–8 of the paper on the three datasets (51 checks). |
| `code/check_cache.py` | Checks that reusing the alpha = 0 solution when alpha <= J0 (Proposition 6) gives the same solution as a direct solve. |
| `code/time_methods.py` | Clean execution-time measurement (Table 10 of the paper). |
| `code/bench_grid.py` | Run-time estimate of a hyper-parameter grid before running it. |
| `code/merge_results.py` | Script used to assemble `results_final/` (see "Provenance of the reported results"). |
| `code/make_paper_tables.py`, `code/make_results_tex.py` | Generate the LaTeX tables and figures of the paper and of the results report from the CSV files. |
| `code/requirements.txt` | Python package versions used. |
| `code/results_final/` | Fold-level results of all experiments reported in the paper (CSV). |

### Result files (`code/results_final/`)

| File | Content |
|---|---|
| `<dataset>_folds.csv` | One row per method and outer fold (25 folds): selected hyper-parameters, AUC, realized cost, G-Mean, recalls, F1, accuracy and confusion counts at the cost-optimal and at the native operating point, costs, tuning/fit/prediction times. |
| `<dataset>_summary.csv` | Mean ± sd over the outer folds. |
| `<dataset>_tests_vs_fcsc.csv` | FCS-C minus each method: mean difference, corrected resampled t-test and Wilcoxon p-values (AUC and cost). |
| `<dataset>_fcsc_diagnostics.csv` | Per fold: selected s and alpha_rel, alpha_max, J0, achieved Fisher ratio, bound activity, condition numbers, weight vector. |
| `<dataset>_alpha_curve.csv` | Sensitivity study: test AUC, realized cost and J/alpha_max for alpha_rel = 0, 0.05, …, 0.95, 0.99, per fold. |
| `timing_clean.csv` | Execution times of Table 10. |

`<dataset>` is `australian`, `german` or `lendingclub`.

## Data

The data are not redistributed; they are publicly available:

- **Australian Credit** (UCI Statlog): https://archive.ics.uci.edu/dataset/143/statlog+australian+credit+approval — file `australian.dat`
- **German Credit** (UCI Statlog): https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data — file `german.data`
- **LendingClub accepted loans 2007–2018**: https://www.kaggle.com/datasets/wordsforthewise/lending-club — file `accepted_2007_to_2018Q4.csv` (uncompressed)

Place the three files in `code/data/`, or set the environment variables `FCSC_AUSTRALIAN`, `FCSC_GERMAN` and
`FCSC_LENDINGCLUB` to their locations. The LendingClub sample of 20,000 loans is drawn reproducibly
(stratified, seed 42) from the 1,266,782 completed loans and cached in `code/data_cache/` on first use.

## Requirements

Python 3.12 with the packages listed in `code/requirements.txt`:

```
pip install -r code/requirements.txt
```

## Reproducing the results

All experiments (three datasets, 25 outer folds each, all methods, ablation and sensitivity study):

```
cd code
python fcsc_pipeline.py --dataset all --repeats 5 --folds 5 --inner 5 --grid compact --jobs 14 --out results
python time_methods.py results compact
python make_results_tex.py results
python make_paper_tables.py results
```

To rebuild the tables and figures from the saved results without rerunning the experiments:

```
cd code
python make_results_tex.py results_final
python make_paper_tables.py results_final
```

`--jobs` is the number of parallel outer folds; we used one per physical core (14), for which the
experiments take about 25 minutes. A single dataset can be run with `--dataset australian`, `german` or
`lendingclub`. All random seeds are fixed (seed 42), so the outer and inner folds are reproducible.

The theoretical results can be checked numerically with

```
python verify_theory.py
python check_cache.py
```

Alternatively, open `FCS-C_experiments.ipynb`: setting `RUN_EXPERIMENTS = False` rebuilds every table and
figure from the saved CSV files, and `QUICK = True` runs a short check.

## Provenance of the reported results

The results reported in the paper were obtained in two runs on the same 75 outer folds (identical seeds):
a run of all methods with the `compact` grid, and a second run of FCS-C alone with the Fisher tightness
restricted to alpha_rel > 0, as described in Section 4.4 of the paper. `merge_results.py` combines the
benchmark methods and the alpha = 0 ablation of the first run with FCS-C from the second run and
recomputes the significance tests. The `compact` grid profile in `fcsc_pipeline.py` now encodes the final
configuration, so the single command above reproduces the reported setting directly.

## Protocol summary

- 5 × 5 repeated stratified cross-validation (25 outer folds), nested 5-fold tuning on the outer-training part only, same folds for every method.
- Misclassification costs: scikit-learn balanced formula c_FP = n / (2 n_Bad), c_FN = n / (2 n_Good), recomputed on each training fold.
- Hyper-parameters selected by realized cost at the cost-optimal acceptance rate on pooled inner out-of-fold scores; the selected acceptance rate is applied to the test fold.
- Significance: corrected resampled t-test (Nadeau and Bengio, 2003) and Wilcoxon signed-rank test on the 25 paired fold-level differences.
