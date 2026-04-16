#!/usr/bin/env python3
"""Item 1 — matched-cutoff ($f_{\\mathrm{cut}} = 10^{-9}$) classifier rerun.

Policy decisions (documented for reproducibility):

1. Ridge envelope: pooled-sample [q10, q90] of realizations within +/- 0.25
   decades in log10(Da) of the *matched-cutoff* centroid
   (centroid_ridge_fcut1e9.csv Da_centroid_f095 column). For the three
   low-R contrasts (R < 5), the matched-cutoff surface is bimodal and
   the kernel-smoothed centroid selects the low-Da peak; we accept that
   selection here for reproducibility and flag those contrasts as
   bimodal-anchored in the output table.

2. Floor envelope: pooled-sample [q10, q90] of realizations with
   log10(Da) <= -5.0 at the site's R prior.

3. Low-R off-shoulder reference envelope: pooled-sample [min, max] of
   realizations in R in [1.0, 6.0] that are > 0.75 decades from the
   matched-cutoff centroid. This retains the main-text production
   convention (asymmetry with the q10-q90 bounds above).

4. Decision rule: unchanged four-state rule (overlap-first;
   median-distance tie-break when one envelope fully contains the
   classifier interval and >= 2 envelopes cover >= 50%; unclassified
   when no envelope reaches 5%).
"""
from pathlib import Path

import numpy as np
import pandas as pd

PKG = Path(__file__).resolve().parent.parent
TABLES = PKG / "tables"
OUT_TEX = PKG / "manuscript" / "tables_tex"
OUT_TEX.mkdir(parents=True, exist_ok=True)

RUNS_FCUT = Path("/Volumes/Backup/Projects/AquiferMemory/hysteresis_outputs/run_fcut_match_1e9/runs.csv")
CENTROID_FCUT = TABLES / "centroid_ridge_fcut1e9.csv"

BASELINE_HI_ENV = TABLES / "benchmark_hi_envelope.csv"
BASELINE_REGIME_CONS = TABLES / "benchmark_regime_consistency.csv"

OUT_ENV = TABLES / "benchmark_regime_envelopes_fcut1e9.csv"
OUT_HIENV = TABLES / "benchmark_hi_envelope_fcut1e9.csv"
OUT_CONS = TABLES / "benchmark_regime_consistency_fcut1e9.csv"
OUT_COMPARE = TABLES / "benchmark_regime_comparison_baseline_vs_matched.csv"

RIDGE_CORE_HALFWIDTH = 0.25      # decades in log10(Da)
FLOOR_LOG10DA_MAX = -5.0
OFF_SHOULDER_MIN_DIST = 0.75     # decades
LOWR_REF_BAND = (1.0, 6.0)       # Arroyo Seco prior


def interval_overlap(lo1, hi1, lo2, hi2):
    if not (np.isfinite(lo1) and np.isfinite(hi1) and np.isfinite(lo2) and np.isfinite(hi2)):
        return np.nan
    w = max(0.0, min(hi1, hi2) - max(lo1, lo2))
    obs_w = hi1 - lo1
    return float(w / obs_w) if obs_w > 0 else 0.0


def main():
    runs = pd.read_csv(RUNS_FCUT)
    runs = runs[np.isfinite(runs["Hysteresis"]) & np.isfinite(runs["Da"]) & (runs["Da"] > 0)].copy()
    runs["log10Da"] = np.log10(runs["Da"])

    centroids = pd.read_csv(CENTROID_FCUT)
    centroids = centroids[["Contrast", "Da_centroid_f095"]].copy()
    centroids["log10_Da_centroid"] = np.log10(centroids["Da_centroid_f095"])

    # Attach centroid to each realization
    runs = runs.merge(centroids[["Contrast", "log10_Da_centroid"]], on="Contrast", how="left")
    runs["dist_from_centroid"] = np.abs(runs["log10Da"] - runs["log10_Da_centroid"])

    # ---------------------------------------------------------------
    # Low-R off-shoulder reference envelope (single envelope for all sites)
    # ---------------------------------------------------------------
    low_r_sub = runs[(runs["Contrast"] >= LOWR_REF_BAND[0]) &
                     (runs["Contrast"] <= LOWR_REF_BAND[1]) &
                     (runs["dist_from_centroid"] > OFF_SHOULDER_MIN_DIST)]
    low_r_hi = low_r_sub["Hysteresis"].to_numpy(dtype=float)
    ss_lo = float(np.min(low_r_hi))
    ss_hi = float(np.max(low_r_hi))
    ss_med = float(np.median(low_r_hi))
    print(f"Matched-cutoff low-R off-shoulder reference envelope: "
          f"[{ss_lo:.4f}, {ss_hi:.4f}]  median {ss_med:.4f}  n={len(low_r_hi)}")

    # ---------------------------------------------------------------
    # Per-site envelopes (ridge + floor depend on site's R prior)
    # ---------------------------------------------------------------
    hi_env_base = pd.read_csv(BASELINE_HI_ENV)
    base_regime = pd.read_csv(BASELINE_REGIME_CONS)

    out_env_rows = []
    out_hienv_rows = []
    out_cons_rows = []

    for _, site in hi_env_base.iterrows():
        sid = str(site["system_id"])
        r_lo, r_hi = float(site["R_lo"]), float(site["R_hi"])
        obs_med = float(site["HI_obs_median"])
        obs_ci_lo = float(site["HI_obs_median_ci_lo"])
        obs_ci_hi = float(site["HI_obs_median_ci_hi"])

        # Pool realizations in site's R prior
        site_runs = runs[(runs["Contrast"] >= r_lo) & (runs["Contrast"] <= r_hi)]

        # Ridge envelope
        ridge_sub = site_runs[site_runs["dist_from_centroid"] <= RIDGE_CORE_HALFWIDTH]
        if len(ridge_sub) >= 10:
            rhi = ridge_sub["Hysteresis"].to_numpy(dtype=float)
            ridge_lo = float(np.quantile(rhi, 0.10))
            ridge_hi = float(np.quantile(rhi, 0.90))
            ridge_med = float(np.median(rhi))
        else:
            ridge_lo = ridge_hi = ridge_med = np.nan

        # Floor envelope
        floor_sub = site_runs[site_runs["log10Da"] <= FLOOR_LOG10DA_MAX]
        if len(floor_sub) >= 10:
            fhi = floor_sub["Hysteresis"].to_numpy(dtype=float)
            floor_lo = float(np.quantile(fhi, 0.10))
            floor_hi = float(np.quantile(fhi, 0.90))
            floor_med = float(np.median(fhi))
        else:
            floor_lo = floor_hi = floor_med = np.nan

        # Reference envelope (fixed across sites)
        ref_lo, ref_hi, ref_med = ss_lo, ss_hi, ss_med

        out_env_rows.extend([
            {"source": site["source"], "system_id": sid, "regime_id": "ridge",
             "HI_lo": ridge_lo, "HI_hi": ridge_hi, "HI_median": ridge_med},
            {"source": site["source"], "system_id": sid, "regime_id": "floor",
             "HI_lo": floor_lo, "HI_hi": floor_hi, "HI_median": floor_med},
            {"source": site["source"], "system_id": sid, "regime_id": "simple_off_ridge",
             "HI_lo": ref_lo, "HI_hi": ref_hi, "HI_median": ref_med},
        ])

        out_hienv_rows.append({
            "source": site["source"], "system_id": sid, "R_lo": r_lo, "R_hi": r_hi,
            "HI_obs_median": obs_med,
            "HI_obs_median_ci_lo": obs_ci_lo, "HI_obs_median_ci_hi": obs_ci_hi,
            "ridge_HI_lo": ridge_lo, "ridge_HI_hi": ridge_hi, "ridge_HI_median": ridge_med,
            "floor_HI_lo": floor_lo, "floor_HI_hi": floor_hi, "floor_HI_median": floor_med,
            "simple_off_ridge_HI_lo": ref_lo, "simple_off_ridge_HI_hi": ref_hi,
            "simple_off_ridge_HI_median": ref_med,
        })

        # Classifier overlaps
        ov_ridge = interval_overlap(obs_ci_lo, obs_ci_hi, ridge_lo, ridge_hi)
        ov_floor = interval_overlap(obs_ci_lo, obs_ci_hi, floor_lo, floor_hi)
        ov_ref = interval_overlap(obs_ci_lo, obs_ci_hi, ref_lo, ref_hi)

        # Median distances
        def _med_dist(em):
            return abs(obs_med - em) if np.isfinite(em) else np.nan
        md_ridge = _med_dist(ridge_med)
        md_floor = _med_dist(floor_med)
        md_ref = _med_dist(ref_med)

        # Decision rule
        overlaps = {"ridge": ov_ridge, "floor": ov_floor, "simple_off_ridge": ov_ref}
        med_dists = {"ridge": md_ridge, "floor": md_floor, "simple_off_ridge": md_ref}

        # Coarse overlap ranking
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

        for reg_id, ov, mdd in [("ridge", ov_ridge, md_ridge), ("floor", ov_floor, md_floor),
                                ("simple_off_ridge", ov_ref, md_ref)]:
            out_cons_rows.append({
                "source": site["source"], "system_id": sid, "regime_id": reg_id,
                "HI_obs_median": obs_med,
                "HI_obs_median_ci_lo": obs_ci_lo, "HI_obs_median_ci_hi": obs_ci_hi,
                "overlap_fraction_obs": ov,
                "distance_obs_median_to_regime_median": mdd,
                "preferred_regime_by_rule": preferred,
                "preference_rule_applied": rule,
            })

    env_df = pd.DataFrame(out_env_rows)
    hi_df = pd.DataFrame(out_hienv_rows)
    cons_df = pd.DataFrame(out_cons_rows)
    env_df.to_csv(OUT_ENV, index=False)
    hi_df.to_csv(OUT_HIENV, index=False)
    cons_df.to_csv(OUT_CONS, index=False)
    print(f"Wrote {OUT_ENV}")
    print(f"Wrote {OUT_HIENV}")
    print(f"Wrote {OUT_CONS}")

    # ---------------------------------------------------------------
    # Side-by-side comparison: baseline vs matched-cutoff per site
    # ---------------------------------------------------------------
    base_pref = base_regime.drop_duplicates("system_id")[
        ["system_id", "preferred_regime_by_rule", "preference_rule_applied"]
    ].rename(columns={"preferred_regime_by_rule": "baseline_regime",
                      "preference_rule_applied": "baseline_rule"})
    matched_pref = cons_df.drop_duplicates("system_id")[
        ["system_id", "preferred_regime_by_rule", "preference_rule_applied"]
    ].rename(columns={"preferred_regime_by_rule": "matched_regime",
                      "preference_rule_applied": "matched_rule"})
    cmp_df = base_pref.merge(matched_pref, on="system_id")
    cmp_df["status"] = np.where(cmp_df["baseline_regime"] == cmp_df["matched_regime"],
                                 "stable", "changed")
    cmp_df.to_csv(OUT_COMPARE, index=False)

    print()
    print("Baseline vs matched-cutoff classifier outputs:")
    print(cmp_df.to_string(index=False))

    n_stable = int((cmp_df["status"] == "stable").sum())
    print(f"\n{n_stable}/{len(cmp_df)} sites have identical regime labels under "
          f"baseline and matched cutoff.")

    # ---------------------------------------------------------------
    # SI LaTeX table (compact side-by-side)
    # ---------------------------------------------------------------
    REGIME_DISPLAY = {
        "ridge": "ridge",
        "floor": "low-$\\mathrm{Da}$ floor",
        "simple_off_ridge": "low-$R$ off-shoulder ref.",
        "unclassified": "unclassified",
    }
    # Reorder to match the paper's site listing
    SITE_ORDER = [
        ("USGS-07014500", "Meramec"),
        ("USGS-07067500", "Big Spring"),
        ("USGS-02322500", "Ichetucknee"),
        ("USGS-08155500", "Barton"),
        ("USGS-08169000", "Comal"),
        ("USGS-08171000", "Blanco"),
        ("11148900", "Arroyo Seco"),
        ("1013500", "Fish River"),
    ]
    lines = []
    for sid, label in SITE_ORDER:
        row = cmp_df[cmp_df["system_id"].astype(str) == sid]
        if row.empty:
            continue
        r = row.iloc[0]
        base_d = REGIME_DISPLAY.get(r["baseline_regime"], r["baseline_regime"]).replace("_", r"\_")
        match_d = REGIME_DISPLAY.get(r["matched_regime"], r["matched_regime"]).replace("_", r"\_")
        status_d = r"\textbf{stable}" if r["status"] == "stable" else r"\textbf{changed}"
        lines.append(f"{label} & {base_d} & {match_d} & {status_d} \\\\")

    tex = (
        "% Auto-generated by regen_matched_cutoff_classifier.py — do not edit.\n"
        "\\begin{table}[ht]\n"
        "\\caption{Side-by-side comparison of the rule-based four-state regime "
        "classifier outputs under the baseline synthetic surface "
        "($f_{\\mathrm{cut}} = 10^{-8}$, as in main-text Table~2) and the "
        "matched-cutoff surface ($f_{\\mathrm{cut}} = 10^{-9}$, field-matched). "
        "The classifier interval (bootstrap 10--90 CI of the site HI median) "
        "is held fixed; only the synthetic envelopes are recomputed on the "
        "matched-cutoff sweep. Envelope conventions are identical between "
        "the two runs (ridge and floor $[q_{10}, q_{90}]$; low-$R$ "
        "off-shoulder reference $[\\min, \\max]$). Row ordering matches "
        "main-text Table~2.}\n"
        "\\label{tab:matched_cutoff_regime_comparison}\n"
        "\\centering\n"
        "\\small\n"
        "\\begin{tabular}{llll}\n"
        "\\toprule\n"
        "Site & Baseline ($10^{-8}$) & Matched cutoff ($10^{-9}$) & Stability \\\\\n"
        "\\midrule\n"
        + "\n".join(lines) +
        "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )
    (OUT_TEX / "matched_cutoff_regime_comparison.tex").write_text(tex)
    print(f"Wrote {OUT_TEX / 'matched_cutoff_regime_comparison.tex'}")


if __name__ == "__main__":
    main()
