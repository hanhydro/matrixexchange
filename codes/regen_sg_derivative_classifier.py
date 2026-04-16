#!/usr/bin/env python3
"""Item 2 (staged, SI-only) — Savitzky-Golay derivative classifier rerun.

Scores each benchmark site's Savitzky-Golay-derivative HI distribution
(HI_savgol column, already cached in benchmark_event_summary.csv)
against the \textbf{baseline} synthetic envelopes. Outputs a per-site
classifier-interval / overlap / preferred-regime table.

Scientific framing: the synthetic envelopes are built from the
\emph{analytical} $-dQ/dt$ computed from the ODE right-hand side, not
from a numerical derivative operator applied to a sampled time series.
Switching the observed-HI pipeline from central differences (CD) to
Savitzky-Golay (SG) therefore applies a derivative operator on one side
of the comparison (the observed side) without applying a matched
operator on the synthetic side. SI Section S10 documents that SG
inflates median HI by 0.25-0.31 units relative to CD at each primary
control site, so the observed distribution is systematically shifted up
relative to the synthetic envelopes. This classifier rerun quantifies
that shift at the site-level rule-based output.

"""
from pathlib import Path

import numpy as np
import pandas as pd

PKG = Path(__file__).resolve().parent.parent
TABLES = PKG / "tables"
OUT_TEX = PKG / "manuscript" / "tables_tex"

EVENTS = TABLES / "benchmark_event_summary.csv"
BASELINE_REGIME_ENV = TABLES / "benchmark_regime_envelopes.csv"
BASELINE_HI_ENV = TABLES / "benchmark_hi_envelope.csv"

OUT_CLASS = TABLES / "benchmark_regime_consistency_sg.csv"

BOOTSTRAP_B = 2000
BOOTSTRAP_SEED = 42
CI_LO_Q = 0.10
CI_HI_Q = 0.90


def bootstrap_median_ci(hi, b=BOOTSTRAP_B, seed=BOOTSTRAP_SEED):
    n = len(hi)
    if n < 3:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    bs = np.array([np.median(rng.choice(hi, size=n, replace=True)) for _ in range(b)])
    return float(np.quantile(bs, CI_LO_Q)), float(np.quantile(bs, CI_HI_Q))


def interval_overlap(lo1, hi1, lo2, hi2):
    if not (np.isfinite(lo1) and np.isfinite(hi1) and np.isfinite(lo2) and np.isfinite(hi2)):
        return np.nan
    w = max(0.0, min(hi1, hi2) - max(lo1, lo2))
    obs_w = hi1 - lo1
    return float(w / obs_w) if obs_w > 0 else 0.0


events = pd.read_csv(EVENTS)
envelopes = pd.read_csv(BASELINE_REGIME_ENV)
hi_env = pd.read_csv(BASELINE_HI_ENV)

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
    "ridge": "ridge",
    "floor": "low-$\\mathrm{Da}$ floor",
    "simple_off_ridge": "low-$R$ off-shoulder ref.",
    "unclassified": "unclassified",
}

rows = []
summary_rows = []

for sid, label in SITES:
    site_events = events[
        (events["system_id"].astype(str) == sid)
        & (events["filter_id"] == "baseline")
        & (events["data_resolution"] == "instantaneous")
        & events["HI_savgol"].notna()
    ]
    hi_sg = site_events["HI_savgol"].to_numpy(dtype=float)
    n = int(hi_sg.size)
    if n < 3:
        continue
    med_sg = float(np.median(hi_sg))
    ci_lo, ci_hi = bootstrap_median_ci(hi_sg)

    # Baseline envelopes at this site
    site_envs = envelopes[envelopes["system_id"].astype(str) == sid]

    def _env(reg_id):
        r = site_envs[site_envs["regime_id"] == reg_id]
        if r.empty:
            return np.nan, np.nan, np.nan
        return float(r["HI_lo"].iloc[0]), float(r["HI_hi"].iloc[0]), float(r["HI_median"].iloc[0])

    r_lo, r_hi, r_med = _env("ridge")
    f_lo, f_hi, f_med = _env("floor")
    s_lo, s_hi, s_med = _env("simple_off_ridge")

    ov_ridge = interval_overlap(ci_lo, ci_hi, r_lo, r_hi)
    ov_floor = interval_overlap(ci_lo, ci_hi, f_lo, f_hi)
    ov_ref = interval_overlap(ci_lo, ci_hi, s_lo, s_hi)

    md_ridge = abs(med_sg - r_med) if np.isfinite(r_med) else np.nan
    md_floor = abs(med_sg - f_med) if np.isfinite(f_med) else np.nan
    md_ref = abs(med_sg - s_med) if np.isfinite(s_med) else np.nan

    overlaps = {"ridge": ov_ridge, "floor": ov_floor, "simple_off_ridge": ov_ref}
    med_dists = {"ridge": md_ridge, "floor": md_floor, "simple_off_ridge": md_ref}
    overlap_first = max(overlaps, key=lambda k: overlaps[k] if np.isfinite(overlaps[k]) else -1)
    full_cover = [k for k, v in overlaps.items() if np.isfinite(v) and v >= 0.999]
    substantial = [k for k, v in overlaps.items() if np.isfinite(v) and v >= 0.5]

    if full_cover and len(substantial) >= 2:
        preferred = min(substantial, key=lambda k: med_dists[k] if np.isfinite(med_dists[k]) else np.inf)
        rule = "distance_when_overlap_nondiscriminating"
    elif overlaps[overlap_first] < 0.05:
        preferred = "unclassified"
        rule = "no_overlap"
    else:
        preferred = overlap_first
        rule = "overlap_first"

    # Baseline preferred label for comparison
    hi_row = hi_env[hi_env["system_id"].astype(str) == sid]
    base_med = float(hi_row["HI_obs_median"].iloc[0]) if not hi_row.empty else np.nan
    base_class_lo = float(hi_row["HI_obs_median_ci_lo"].iloc[0]) if not hi_row.empty else np.nan
    base_class_hi = float(hi_row["HI_obs_median_ci_hi"].iloc[0]) if not hi_row.empty else np.nan

    summary_rows.append({
        "system_id": sid, "label": label, "n_events": n,
        "HI_CD_median": base_med,
        "HI_CD_ci": f"[{base_class_lo:.3f}, {base_class_hi:.3f}]",
        "HI_SG_median": med_sg,
        "HI_SG_ci": f"[{ci_lo:.3f}, {ci_hi:.3f}]",
        "ridge_ov_SG": ov_ridge,
        "floor_ov_SG": ov_floor,
        "ref_ov_SG": ov_ref,
        "preferred_SG": preferred,
        "rule_SG": rule,
    })

    for reg_id, ov, mdd in [("ridge", ov_ridge, md_ridge),
                             ("floor", ov_floor, md_floor),
                             ("simple_off_ridge", ov_ref, md_ref)]:
        rows.append({
            "system_id": sid, "regime_id": reg_id,
            "HI_obs_median_SG": med_sg,
            "HI_obs_median_ci_lo_SG": ci_lo, "HI_obs_median_ci_hi_SG": ci_hi,
            "overlap_fraction_obs_SG": ov,
            "distance_obs_median_to_regime_median_SG": mdd,
            "preferred_regime_by_rule_SG": preferred,
            "preference_rule_applied_SG": rule,
        })

out = pd.DataFrame(rows)
out.to_csv(OUT_CLASS, index=False)
print(f"Wrote {OUT_CLASS}")
print()

summary = pd.DataFrame(summary_rows)
print("Per-site SG-derivative classifier (baseline envelopes held fixed):")
print(summary.to_string(index=False))
print()
n_unclass = int((summary["preferred_SG"] == "unclassified").sum())
print(f"{n_unclass}/{len(summary)} sites are unclassified under SG derivative (baseline envelopes).")

# SI LaTeX table
lines = []
for _, r in summary.iterrows():
    disp = REGIME_DISPLAY.get(r["preferred_SG"], r["preferred_SG"]).replace("_", r"\_")
    lines.append(" & ".join([
        r["label"],
        f"{r['n_events']}",
        f"{r['HI_CD_median']:.3f}",
        f"{r['HI_SG_median']:.3f}",
        f"{r['HI_SG_ci']}",
        f"{r['ridge_ov_SG']:.4f}",
        f"{r['ref_ov_SG']:.4f}",
        disp,
    ]) + r" \\")

tex = (
    "% Auto-generated by regen_sg_derivative_classifier.py — do not edit.\n"
    "\\begin{table}[ht]\n"
    "\\caption{Savitzky-Golay-derivative rerun of the rule-based "
    "four-state classifier against the \\textit{baseline} synthetic "
    "envelopes (ridge and floor $[q_{10}, q_{90}]$; low-$R$ "
    "off-shoulder reference $[\\min, \\max]$). $\\mathrm{HI}_{\\mathrm{CD}}$ "
    "median is the central-difference median from main-text Table~2. "
    "$\\mathrm{HI}_{\\mathrm{SG}}$ median is computed from the same events "
    "using a Savitzky-Golay derivative (polynomial order 2, window 7). "
    "SG inflates every site median by 0.25--0.31 units relative to CD "
    "(SI Section~S10), which pushes the classifier interval above every "
    "baseline-envelope upper bound; \\textit{all eight sites become "
    "unclassified under the SG pipeline with the baseline envelopes "
    "held fixed}. This is not a finding about the sites; it is a "
    "finding that the classifier is derivative-operator conditional "
    "and that matching the observed and synthetic derivative operators "
    "would require a full SG-derivative synthetic sweep rerun, which "
    "is outside the current scope.}\n"
    "\\label{tab:sg_derivative_classifier}\n"
    "\\centering\n"
    "\\small\n"
    "\\begin{tabular}{lrrrcrrl}\n"
    "\\toprule\n"
    "Site & $n$ & $\\mathrm{HI}_{\\mathrm{CD}}$ med. & "
    "$\\mathrm{HI}_{\\mathrm{SG}}$ med. & SG CI & Ridge ov. & Ref. ov. & "
    "Preferred (SG) \\\\\n"
    "\\midrule\n"
    + "\n".join(lines) +
    "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
)
(OUT_TEX / "sg_derivative_classifier.tex").write_text(tex)
print(f"Wrote {OUT_TEX / 'sg_derivative_classifier.tex'}")
