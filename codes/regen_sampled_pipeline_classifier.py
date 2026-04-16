#!/usr/bin/env python3
"""Rebuild envelopes + four-state classifier on the sampled-pipeline
synthetic surface (run_sampled_pipeline_1e9/runs.csv).

Outputs:
  tables/benchmark_hi_envelope_sampled1e9.csv
  tables/benchmark_regime_envelopes_sampled1e9.csv
  tables/benchmark_regime_consistency_sampled1e9.csv
  tables/benchmark_regime_comparison_baseline_vs_sampled.csv
  manuscript/tables_tex/sampled_pipeline_regime_comparison.tex
"""
from pathlib import Path

import numpy as np
import pandas as pd

PKG = Path(__file__).resolve().parent.parent
TABLES = PKG / "tables"
OUT_TEX = PKG / "manuscript" / "tables_tex"
OUT_TEX.mkdir(parents=True, exist_ok=True)

RUNS_SP = Path("/Volumes/Backup/Projects/AquiferMemory/hysteresis_outputs/run_sampled_pipeline_1e9/runs.csv")
BASELINE_HI_ENV = TABLES / "benchmark_hi_envelope.csv"
BASELINE_REGIME_CONS = TABLES / "benchmark_regime_consistency.csv"

OUT_ENV = TABLES / "benchmark_regime_envelopes_sampled1e9.csv"
OUT_HIENV = TABLES / "benchmark_hi_envelope_sampled1e9.csv"
OUT_CONS = TABLES / "benchmark_regime_consistency_sampled1e9.csv"
OUT_COMPARE = TABLES / "benchmark_regime_comparison_baseline_vs_sampled.csv"

RIDGE_CORE_HALFWIDTH = 0.25
FLOOR_LOG10DA_MAX = -5.0
OFF_SHOULDER_MIN_DIST = 0.75
LOWR_REF_BAND = (1.0, 6.0)


def interval_overlap(lo1, hi1, lo2, hi2):
    if not (np.isfinite(lo1) and np.isfinite(hi1) and np.isfinite(lo2) and np.isfinite(hi2)):
        return np.nan
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
        rule = "distance_when_overlap_nondiscriminating"
    elif overlaps[overlap_first] < thresh_unclass:
        preferred = "unclassified"
        rule = "no_overlap"
    else:
        preferred = overlap_first
        rule = "overlap_first"
    return preferred, rule


def kernel_centroid(log10_Da, HI, bw=0.25, n_grid=200):
    m = np.isfinite(log10_Da) & np.isfinite(HI) & (HI > 0)
    if m.sum() < 10:
        return np.nan
    x, h = log10_Da[m], HI[m]
    grid = np.linspace(x.min(), x.max(), n_grid)
    dx = grid[:, None] - x[None, :]
    w = np.exp(-0.5 * (dx / bw) ** 2)
    hi_smooth = (w * h[None, :]).sum(axis=1) / (w.sum(axis=1) + 1e-30)
    return float(grid[int(np.argmax(hi_smooth))])


def main():
    # HI column selector: use HI_sampled from the sampled-pipeline sweep
    runs = pd.read_csv(RUNS_SP)
    hi_col = "HI_sampled"
    assert hi_col in runs.columns
    runs = runs[np.isfinite(runs[hi_col]) & np.isfinite(runs["Da"]) & (runs["Da"] > 0)].copy()
    runs["log10Da"] = np.log10(runs["Da"])

    # Compute per-contrast centroid on SAMPLED HI
    contrasts = sorted(runs["contrast"].unique())
    cent_rows = []
    for c in contrasts:
        sub = runs[runs["contrast"] == c]
        lx = sub["log10Da"].to_numpy(); h = sub[hi_col].to_numpy()
        cen = kernel_centroid(lx, h)
        cent_rows.append({"Contrast": c, "Da_centroid_f095": 10.0 ** cen, "log10_Da_centroid": cen})
    centroids = pd.DataFrame(cent_rows)

    runs = runs.merge(centroids[["Contrast", "log10_Da_centroid"]].rename(columns={"Contrast": "contrast"}),
                      on="contrast", how="left")
    runs["dist_from_centroid"] = np.abs(runs["log10Da"] - runs["log10_Da_centroid"])

    # Low-R off-shoulder reference envelope (fixed across sites)
    lowr = runs[(runs["contrast"] >= LOWR_REF_BAND[0])
                & (runs["contrast"] <= LOWR_REF_BAND[1])
                & (runs["dist_from_centroid"] > OFF_SHOULDER_MIN_DIST)]
    ss_hi = lowr[hi_col].to_numpy(dtype=float)
    ss_lo, ss_hi_, ss_med = float(np.min(ss_hi)), float(np.max(ss_hi)), float(np.median(ss_hi))
    print(f"Sampled-pipeline low-R off-shoulder reference envelope: "
          f"[{ss_lo:.4f}, {ss_hi_:.4f}]  median {ss_med:.4f}  n={len(ss_hi)}")

    # Load baseline benchmark provenance (same classifier intervals!)
    hi_env_base = pd.read_csv(BASELINE_HI_ENV)
    base_regime = pd.read_csv(BASELINE_REGIME_CONS)

    out_env_rows, out_hi_rows, out_cons_rows = [], [], []

    for _, site in hi_env_base.iterrows():
        sid = str(site["system_id"])
        r_lo, r_hi = float(site["R_lo"]), float(site["R_hi"])
        obs_med = float(site["HI_obs_median"])
        obs_ci_lo = float(site["HI_obs_median_ci_lo"])
        obs_ci_hi = float(site["HI_obs_median_ci_hi"])

        site_runs = runs[(runs["contrast"] >= r_lo) & (runs["contrast"] <= r_hi)]

        ridge_sub = site_runs[site_runs["dist_from_centroid"] <= RIDGE_CORE_HALFWIDTH]
        if len(ridge_sub) >= 10:
            r_arr = ridge_sub[hi_col].to_numpy(dtype=float)
            r_lo_hi, r_hi_hi = float(np.quantile(r_arr, 0.1)), float(np.quantile(r_arr, 0.9))
            r_med = float(np.median(r_arr))
        else:
            r_lo_hi = r_hi_hi = r_med = np.nan

        floor_sub = site_runs[site_runs["log10Da"] <= FLOOR_LOG10DA_MAX]
        if len(floor_sub) >= 10:
            f_arr = floor_sub[hi_col].to_numpy(dtype=float)
            f_lo_hi, f_hi_hi = float(np.quantile(f_arr, 0.1)), float(np.quantile(f_arr, 0.9))
            f_med = float(np.median(f_arr))
        else:
            f_lo_hi = f_hi_hi = f_med = np.nan

        out_env_rows.extend([
            {"source": site["source"], "system_id": sid, "regime_id": "ridge",
             "HI_lo": r_lo_hi, "HI_hi": r_hi_hi, "HI_median": r_med},
            {"source": site["source"], "system_id": sid, "regime_id": "floor",
             "HI_lo": f_lo_hi, "HI_hi": f_hi_hi, "HI_median": f_med},
            {"source": site["source"], "system_id": sid, "regime_id": "simple_off_ridge",
             "HI_lo": ss_lo, "HI_hi": ss_hi_, "HI_median": ss_med},
        ])

        out_hi_rows.append({
            "source": site["source"], "system_id": sid, "R_lo": r_lo, "R_hi": r_hi,
            "HI_obs_median": obs_med,
            "HI_obs_median_ci_lo": obs_ci_lo, "HI_obs_median_ci_hi": obs_ci_hi,
            "ridge_HI_lo": r_lo_hi, "ridge_HI_hi": r_hi_hi, "ridge_HI_median": r_med,
            "floor_HI_lo": f_lo_hi, "floor_HI_hi": f_hi_hi, "floor_HI_median": f_med,
            "simple_off_ridge_HI_lo": ss_lo, "simple_off_ridge_HI_hi": ss_hi_,
            "simple_off_ridge_HI_median": ss_med,
        })

        ov_r = interval_overlap(obs_ci_lo, obs_ci_hi, r_lo_hi, r_hi_hi)
        ov_f = interval_overlap(obs_ci_lo, obs_ci_hi, f_lo_hi, f_hi_hi)
        ov_s = interval_overlap(obs_ci_lo, obs_ci_hi, ss_lo, ss_hi_)

        def _md(m): return abs(obs_med - m) if np.isfinite(m) else np.nan
        overlaps = {"ridge": ov_r, "floor": ov_f, "simple_off_ridge": ov_s}
        med_dists = {"ridge": _md(r_med), "floor": _md(f_med), "simple_off_ridge": _md(ss_med)}
        preferred, rule = classify(overlaps, med_dists)

        for reg_id, ov, mdd in [("ridge", ov_r, med_dists["ridge"]),
                                 ("floor", ov_f, med_dists["floor"]),
                                 ("simple_off_ridge", ov_s, med_dists["simple_off_ridge"])]:
            out_cons_rows.append({
                "source": site["source"], "system_id": sid, "regime_id": reg_id,
                "HI_obs_median": obs_med,
                "HI_obs_median_ci_lo": obs_ci_lo, "HI_obs_median_ci_hi": obs_ci_hi,
                "overlap_fraction_obs": ov,
                "distance_obs_median_to_regime_median": mdd,
                "preferred_regime_by_rule": preferred,
                "preference_rule_applied": rule,
            })

    env_df = pd.DataFrame(out_env_rows); env_df.to_csv(OUT_ENV, index=False)
    hi_df = pd.DataFrame(out_hi_rows); hi_df.to_csv(OUT_HIENV, index=False)
    cons_df = pd.DataFrame(out_cons_rows); cons_df.to_csv(OUT_CONS, index=False)

    # Comparison table
    base_pref = base_regime.drop_duplicates("system_id")[
        ["system_id", "preferred_regime_by_rule"]
    ].rename(columns={"preferred_regime_by_rule": "baseline_regime"})
    samp_pref = cons_df.drop_duplicates("system_id")[
        ["system_id", "preferred_regime_by_rule"]
    ].rename(columns={"preferred_regime_by_rule": "sampled_regime"})
    cmp_df = base_pref.merge(samp_pref, on="system_id")
    cmp_df["status"] = np.where(cmp_df["baseline_regime"] == cmp_df["sampled_regime"],
                                 "stable", "changed")
    cmp_df.to_csv(OUT_COMPARE, index=False)

    print()
    print("Baseline (analytical, 1e-8) vs sampled-pipeline (CD, 1e-9) classifier:")
    print(cmp_df.to_string(index=False))
    n_stable = int((cmp_df["status"] == "stable").sum())
    print(f"\n{n_stable}/{len(cmp_df)} sites have identical labels.")

    # LaTeX table
    REGIME_DISPLAY = {
        "ridge": "ridge",
        "floor": "low-$\\mathrm{Da}$ floor",
        "simple_off_ridge": "low-$R$ off-shoulder ref.",
        "unclassified": "unclassified",
    }
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
        if row.empty: continue
        r = row.iloc[0]
        b = REGIME_DISPLAY.get(r["baseline_regime"], r["baseline_regime"]).replace("_", r"\_")
        s = REGIME_DISPLAY.get(r["sampled_regime"], r["sampled_regime"]).replace("_", r"\_")
        status_d = r"\textbf{stable}" if r["status"] == "stable" else r"\textit{changed}"
        lines.append(f"{label} & {b} & {s} & {status_d} \\\\")

    tex = (
        "% Auto-generated by regen_sampled_pipeline_classifier.py — do not edit.\n"
        "\\begin{table}[ht]\n"
        "\\caption{Sampled-pipeline classifier vs baseline-pipeline classifier. "
        "Baseline uses the analytical $-dQ/dt$ from the ODE right-hand side and "
        "synthetic-side $f_{\\mathrm{cut}} = 10^{-8}$. Sampled pipeline rebuilds "
        "the synthetic envelopes from \\texttt{run\\_sampled\\_pipeline\\_1e9} "
        "(CHTC sweep, 14,245 realizations) using the \\textit{same} central-"
        "difference derivative on rolling-median-smoothed $Q_{\\mathrm{out}}$ "
        "and the \\textit{same} $f_{\\mathrm{cut}} = 10^{-9}$ extraction cutoff "
        "that the field side uses. Classifier intervals (bootstrap 10--90 CI "
        "of each site's field-HI median) are held fixed. Envelope conventions "
        "are identical between the two runs (ridge and floor $[q_{10}, q_{90}]$; "
        "low-$R$ off-shoulder reference $[\\min, \\max]$).}\n"
        "\\label{tab:sampled_pipeline_regime_comparison}\n"
        "\\centering\n"
        "\\small\n"
        "\\begin{tabular}{llll}\n"
        "\\toprule\n"
        "Site & Baseline (analytical, $10^{-8}$) & Sampled pipeline (CD, $10^{-9}$) & Stability \\\\\n"
        "\\midrule\n"
        + "\n".join(lines) +
        "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )
    (OUT_TEX / "sampled_pipeline_regime_comparison.tex").write_text(tex)
    print(f"\nWrote {OUT_TEX / 'sampled_pipeline_regime_comparison.tex'}")


if __name__ == "__main__":
    main()
