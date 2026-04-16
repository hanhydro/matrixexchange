# Reproducibility Scripts

Scripts for reproducing all figures, tables, and numerical results in the
manuscript and supporting information.

## Prerequisites

- Python 3.10+
- NumPy, SciPy, pandas, matplotlib
- `cartopy` (for `plotting/generate_site_map.py` only)

Input data paths (currently hard-coded to local paths) must be updated to
point to the archived data directory. The two primary data sources are:

| File | Description |
|------|-------------|
| `runs.csv` | Main parameter survey (14,245 accepted realizations) |
| `benchmark_data/` | Field discharge records and event summaries |

## Directory structure

```
scripts/
  Model_lowDa.py   # Core dual-porosity ODE model
  build_package.py       # Main orchestrator (plot functions)
  build_si_tables.py                      # SI table generator (11 tables)
  benchmark_block_cv.py                   # Block cross-validation table
  regen_hi_envelope_classifier.py         # Four-state classifier pipeline
  regen_main_benchmark_summary.py         # Table 2 (main) + SI full audit
  regen_s15_envelope_table.py             # Per-site envelope comparison
  regen_lowR_envelope_robustness.py       # Low-R envelope robustness
  regen_lowR_quantile_alt_classifier.py   # Quantile-alternative classifier
  regen_matched_cutoff_classifier.py      # Matched-cutoff classifier rerun
  regen_sampled_pipeline_classifier.py    # Sampled-pipeline classifier rerun
  regen_sg_derivative_classifier.py       # SG-derivative classifier rerun
  regen_decision_rule_sensitivity.py      # Decision-rule threshold sweep
```

### Main-text table

| Script | Output |
|--------|--------|
| `regen_main_benchmark_summary.py` | `benchmark_summary_main.tex` (Table 2) |

### SI tables

| Script | Output |
|--------|--------|
| `build_si_tables.py` | 11 tables (centroid, convergence, forcing, provenance, resolution, metrics, CV, derivatives, outflow, forcing families, rescaling) |
| `regen_main_benchmark_summary.py` | `benchmark_summary_si_full.tex` |
| `benchmark_block_cv.py` | `block_cv.tex` |
| `regen_s15_envelope_table.py` | `s15_envelope_comparison.tex` |
| `regen_lowR_envelope_robustness.py` | `lowR_envelope_robustness.tex` |
| `regen_matched_cutoff_classifier.py` | `matched_cutoff_regime_comparison.tex` |
| `regen_sg_derivative_classifier.py` | `sg_derivative_classifier.tex` |
| `regen_decision_rule_sensitivity.py` | `decision_rule_sensitivity.tex` |
| `regen_sampled_pipeline_classifier.py` | `sampled_pipeline_regime_comparison.tex` |
| `regen_lowR_quantile_alt_classifier.py` | `lowR_quantile_alt_classifier.tex` |

## Reproduction order

```bash
# 1. Core classifier tables (required by most downstream scripts)
python scripts/regen_hi_envelope_classifier.py

# 2. SI tables
python scripts/build_si_tables.py
python scripts/benchmark_block_cv.py
python scripts/regen_s15_envelope_table.py
python scripts/regen_lowR_envelope_robustness.py
python scripts/regen_matched_cutoff_classifier.py
python scripts/regen_sg_derivative_classifier.py
python scripts/regen_decision_rule_sensitivity.py
python scripts/regen_sampled_pipeline_classifier.py
python scripts/regen_lowR_quantile_alt_classifier.py

# 3. Main-text table
python scripts/regen_main_benchmark_summary.py


```

