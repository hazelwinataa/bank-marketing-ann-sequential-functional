# Experiment evidence & methodological notes

## Original material

- Private provenance backup: `PRIVATE-BACKUP-original_main.ipynb` (included in the download ZIP outside the GitHub repository folder) — exact byte copy of the user-provided `main.ipynb`, which also contains answers to other Deep Learning coursework questions.
- Working notebook: `notebooks/bank_marketing_ann.ipynb` — preserved original Bank Marketing code cells, markdown, and saved outputs from the original notebook's Bank Marketing section (starting at original cell 15), with a short scope preface. Original training logic, evaluation code and stored metrics are preserved; only one incomplete optimizer import is corrected to `Adam` in the working notebook.
- Charts in `figures/` are extracted from actual embedded `image/png` notebook outputs and converted to compressed WebP without changing the plotted values.
- **No code was executed to retrain the ANN models during preparation of this repository.** Syntax validation is not equivalent to a successful end-to-end rerun.

## Saved held-out evaluation

| Field | Sequential | Functional |
| --- | ---: | ---: |
| Accuracy | 0.840755 | 0.838101 |
| Precision | 0.411035 | 0.408188 |
| Recall | 0.836066 | 0.854981 |
| F1 | 0.551122 | 0.552567 |
| ROC AUC | 0.915079 | 0.918852 |
| Test loss | 0.348217 | 0.344654 |

The numeric values are from the original saved output of `results_df = pd.DataFrame([seq_results, func_results])`. The notebook has a repeated Sequential evaluation cell; this does not add a separate model/run. Train/validation/test sizes are 31,647 / 6,782 / 6,782.

## Material limitations

1. `duration` is observed during or after telephone contact, preventing prospective pre-call application. Source: UCI documentation for dataset 222.
2. A random stratified split was used, not a chronological holdout, despite the UCI source data being ordered by date.
3. Class weights prioritize sensitivity and may result in many false positive subscriber predictions. Metrics apply to the notebook's 0.5 threshold.
4. Sequential and Functional use the same layer topology; changes in reported metrics are not evidence of intrinsic API performance differences.
5. Notebook history includes prior output/checkpoint artifacts and saved execution order; simply restarting and running top-to-bottom may yield different numeric results. Do not present saved outputs as evidence of a newly executed reproduction.
6. The unmodified original has extra coursework sections (garbage image classification theory/questions) unrelated to the actual Bank Marketing training portion. The focused notebook extracts the relevant section for readability apart from fixing the incomplete optimizer import (`Adam`).
7. The dataset and fitted model binaries are not stored in the repository.

## Potential next steps

- Remove `duration` and repeat experiments using only pre-call observable features.
- Compare models over repeat seeds or paired cross-validation and quantify uncertainty.
- Evaluate on a later chronological period and tune classification thresholds using validation only.
- Include PR-AUC and business-oriented false-positive cost / call capacity analysis.
