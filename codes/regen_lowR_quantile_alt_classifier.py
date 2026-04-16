#!/usr/bin/env python3
"""Low-R off-shoulder reference envelope: quantile-alternative classifier
sensitivity.
Envelope alternatives tested:
  [min, max]           (production; baseline for comparison)
  [q05, q95]           (conservative quantile; drops extreme 10% of pool)
  [q02, q98]           (tighter than production; 4%-trimmed)
  [q10, q90]           (matches ridge/floor bounds; most symmetric)

All other classifier parts (classifier intervals, ridge and floor envelopes,
decision rule thresholds, tie-break rule) are held at production values.
"""
from pathlib import Path

import numpy as np
import pandas as pd

PKG = Path(__file__).resolve().parent.parent
TABLES = PKG / "tables"
OUT_TEX = PKG / "manuscript" / "tables_tex" / "lowR_quantile_alt_classifier.tex"
OUT_CSV = TABLES / "lowR_quantile_alt_classifier.csv"

HI_ENV = TABLES / "benchmark_hi_envelope.csv"
REG_ENV = TABLES / "benchmark_regime_envelopes.csv"
# Full runs table needed to recompute low-R envelope at alternative quantiles
RUNS_BASE = Path("/Volumes/Backup/Projects/AquiferMemory/hysteresis_outputs/run_20260317_123445/runs.csv")

SITES = [
    ("USGS-07014500", "Meramec"),
    ("USGS-07067500", "Big Spring"),
    ("USGS-02322500", "Ichetucknee"),
    ("USGS-08155500", "Barton"),
    ("USGS-08169000", "Comal"),
    ("USGS-08171000", "Blanco"),
    ("11148900", "Arroyo Seco"),
    ("1013500", "Fish River"),
]

REGIME_DISPLAY = {
    "ridge": "R",
    "floor": "F",
    "simple_off_ridge": "S",
    "unclassified": "U",
}

# Envelope recipes: (label, lo_quantile or 'min', hi_quantile or 'max')
RECIPES = [
    (r"$[\min, \max]$", "min", "max"),
    (r"$[q_{02}, q_{98}]$", 0.02, 0.98),
    (r"$[q_{05}, q_{95}]$", 0.05, 0.95),
    (r"$[q_{10}, q_{90}]$", 0.10, 0.90),
]

# Off-shoulder definition (same as production)
LOWR_R_LO, LOWR_R_HI = 1.0, 6.0
OFF_SHOULDER_MIN_DIST = 0.75


def interval_overlap(lo1, hi1, lo2, hi2):
    w = max(0.0, min(hi1, hi2) - max(lo1, lo2))
    obs_w = hi1 - lo1
    return float(w / obs_w) if obs_w > 0 else 0.0


def classify(overlaps, med_dists, thresh_unclass=0.05, thresh_tie=0.50):
    overlap_first = max(overlaps, key=lambda k: overlaps[k] if np.isfinite(overlaps[k]) else -1)
    full_cover = [k for k, v in overlaps.items() if np.isfinite(v) and v >= 0.999]
    substantial = [k for k, v in overlaps.items() if np.isfinite(v) and v >= thresh_tie]
    if full_cover and len(substantial) >= 2:
        preferred = min(substantial,
                        key=lambda k: med_dists[k] if np.isfinite(med_dists[k]) else np.inf)
    elif overlaps[overlap_first] < thresh_unclass:
        preferred = "unclassified"
    else:
        preferred = overlap_first
    return preferred


# Compute the low-R off-shoulder pool from baseline runs
runs = pd.read_csv(RUNS_BASE)
runs = runs[np.isfinite(runs["Hysteresis"]) & np.isfinite(runs["Da"]) & (runs["Da"] > 0)].copy()
runs["log10Da"] = np.log10(runs["Da"])
# Load baseline centroids
centroids = pd.read_csv(TABLES / "centroid_ridge_main.csv")
runs = runs.merge(
    centroids[["Contrast", "Da_centroid_f095"]].rename(columns={"Da_centroid_f095": "Dac"}),
    on="Contrast", how="left")
runs["dist_log10_from_centroid"] = np.abs(runs["log10Da"] - np.log10(runs["Dac"]))
pool = runs[(runs["Contrast"] >= LOWR_R_LO)
            & (runs["Contrast"] <= LOWR_R_HI)
            & (runs["dist_log10_from_centroid"] > OFF_SHOULDER_MIN_DIST)]
pool_hi = pool["Hysteresis"].to_numpy(dtype=float)

# Per-recipe envelope bounds
recipe_bounds = {}
for label, lo_q, hi_q in RECIPES:
    lo_val = float(np.min(pool_hi)) if lo_q == "min" else float(np.quantile(pool_hi, lo_q))
    hi_val = float(np.max(pool_hi)) if hi_q == "max" else float(np.quantile(pool_hi, hi_q))
    med_val = float(np.median(pool_hi))
    recipe_bounds[label] = (lo_val, hi_val, med_val)

# Build per-site overlaps under each recipe
hi_env = pd.read_csv(HI_ENV)
reg_env = pd.read_csv(REG_ENV)

per_site = {}
for sid, label in SITES:
    row = hi_env[hi_env["system_id"].astype(str) == sid]
    if row.empty:
        continue
    obs_lo = float(row["HI_obs_median_ci_lo"].iloc[0])
    obs_hi = float(row["HI_obs_median_ci_hi"].iloc[0])
    obs_med = float(row["HI_obs_median"].iloc[0])
    envs = reg_env[reg_env["system_id"].astype(str) == sid]
    def _env(reg_id):
        r = envs[envs["regime_id"] == reg_id]
        if r.empty:
            return np.nan, np.nan, np.nan
        return (float(r["HI_lo"].iloc[0]), float(r["HI_hi"].iloc[0]), float(r["HI_median"].iloc[0]))
    r_lo, r_hi, r_med = _env("ridge")
    f_lo, f_hi, f_med = _env("floor")
    per_site[sid] = (label, obs_lo, obs_hi, obs_med, r_lo, r_hi, r_med, f_lo, f_hi, f_med)

# Run classifier under each recipe
rows = []
for recipe_label, _, _ in RECIPES:
    s_lo, s_hi, s_med = recipe_bounds[recipe_label]
    rec_row = {"recipe": recipe_label, "ref_lo": s_lo, "ref_hi": s_hi, "ref_med": s_med}
    for sid, label in SITES:
        if sid not in per_site:
            rec_row[label] = ""
            continue
        _, obs_lo, obs_hi, obs_med, r_lo, r_hi, r_med, f_lo, f_hi, f_med = per_site[sid]
        ov_r = interval_overlap(obs_lo, obs_hi, r_lo, r_hi)
        ov_f = interval_overlap(obs_lo, obs_hi, f_lo, f_hi)
        ov_s = interval_overlap(obs_lo, obs_hi, s_lo, s_hi)
        def _md(m): return abs(obs_med - m) if np.isfinite(m) else np.nan
        pref = classify({"ridge": ov_r, "floor": ov_f, "simple_off_ridge": ov_s},
                        {"ridge": _md(r_med), "floor": _md(f_med), "simple_off_ridge": _md(s_med)})
        rec_row[label] = REGIME_DISPLAY.get(pref, pref)
    rows.append(rec_row)

df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False)

# LaTeX table
site_labels = [label for _, label in SITES]
body = []
for _, r in df.iterrows():
    # Recipe strings are already wrapped in $...$ (math mode) so LaTeX
    # does not interpret the leading bracket as an optional argument
    # to \\ after \midrule.
    recipe_cell = r["recipe"]
    if r["recipe"] == r"$[\min, \max]$":
        recipe_cell = r["recipe"] + r" \textbf{(production)}"
    cells = [recipe_cell, f"[{r['ref_lo']:.3f}, {r['ref_hi']:.3f}]",
             f"{r['ref_med']:.3f}"] + [r[s] for s in site_labels]
    body.append(" & ".join(cells) + r" \\")
header = " & ".join(["Recipe", "Ref.\\ bounds", "Ref.\\ median"]
                    + [f"\\rotatebox{{60}}{{{s}}}" for s in site_labels]) + r" \\"

tex = (
    "% Auto-generated by regen_lowR_quantile_alt_classifier.py — do not edit.\n"
    "\\begin{table}[ht]\n"
    "\\caption{Low-$R$ off-shoulder reference envelope: classifier "
    "sensitivity to envelope-bound recipe. The production convention "
    "is $[\\min, \\max]$ of the pooled off-shoulder sample at "
    "$R \\in [1.0, 6.0]$; here we retest the classifier under three "
    "alternative quantile-bounded recipes while holding the ridge and "
    "floor envelopes at the production $[q_{10}, q_{90}]$ convention "
    "and all other classifier parts (classifier intervals, decision-rule "
    "thresholds, tie-break rule) at their production values. Cell entries "
    "encode the preferred regime: \\textbf{R}~=~ridge, \\textbf{F}~=~low-$\\mathrm{Da}$ "
    "floor, \\textbf{S}~=~low-$R$ off-shoulder reference, "
    "\\textbf{U}~=~unclassified. Under all four recipes, Meramec remains "
    "ridge-preferred and Fish River remains unclassified; label changes "
    "concentrate at sites whose classifier interval sits at or above "
    "the reference-envelope upper quantile, where tightening from "
    "$[\\min, \\max]$ to $[q_{05}, q_{95}]$ or $[q_{10}, q_{90}]$ "
    "removes extreme-value-anchored overlap coverage and shifts those "
    "sites toward unclassified.}\n"
    "\\label{tab:lowR_quantile_alt_classifier}\n"
    "\\centering\n"
    "\\small\n"
    "\\begin{tabular}{lcc" + "c" * len(site_labels) + "}\n"
    "\\toprule\n"
    + header + "\n"
    "\\midrule\n"
    + "\n".join(body)
    + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
)
OUT_TEX.write_text(tex)
print(f"Wrote {OUT_CSV}")
print(f"Wrote {OUT_TEX}")
print()
print("Classifier under alternative low-R reference-envelope recipes:")
print(df.to_string(index=False))
