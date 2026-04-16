#!/usr/bin/env python3
"""Leave-year-out block cross-validation for benchmark HI estimates."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def run_block_cv(package: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    events = pd.read_csv(package / "tables" / "benchmark_event_summary.csv")
    events = events[events["filter_id"] == "baseline"].copy()
    events["HI_obs"] = pd.to_numeric(events["HI_obs"], errors="coerce")
    events = events[np.isfinite(events["HI_obs"])].copy()

    # Extract year from peak_date
    events["year"] = pd.to_datetime(events["peak_date"], errors="coerce").dt.year
    events = events[events["year"].notna()].copy()
    events["year"] = events["year"].astype(int)

    cv_rows: list[dict] = []
    summary_rows: list[dict] = []

    for system_id, grp in events.groupby("system_id"):
        years = sorted(grp["year"].unique())
        if len(years) < 3:
            continue

        full_hi = grp["HI_obs"].values
        full_median = float(np.median(full_hi))
        full_q10 = float(np.quantile(full_hi, 0.10))
        full_q90 = float(np.quantile(full_hi, 0.90))
        full_n = len(full_hi)

        fold_medians: list[float] = []
        for yr in years:
            remaining = grp[grp["year"] != yr]["HI_obs"].values
            if len(remaining) < 3:
                continue
            med = float(np.median(remaining))
            q10 = float(np.quantile(remaining, 0.10))
            q90 = float(np.quantile(remaining, 0.90))
            fold_medians.append(med)
            cv_rows.append({
                "system_id": system_id,
                "held_out_year": yr,
                "n_events_held_out": int(np.sum(grp["year"] == yr)),
                "n_events_remaining": len(remaining),
                "HI_median_cv": med,
                "HI_q10_cv": q10,
                "HI_q90_cv": q90,
                "HI_median_full": full_median,
                "HI_q10_full": full_q10,
                "HI_q90_full": full_q90,
            })

        if fold_medians:
            arr = np.asarray(fold_medians)
            summary_rows.append({
                "system_id": system_id,
                "n_years": len(years),
                "n_events_total": full_n,
                "HI_median_full": full_median,
                "max_abs_deviation_median": float(np.max(np.abs(arr - full_median))),
                "cv_of_median": float(np.std(arr) / np.abs(np.mean(arr))) if np.mean(arr) != 0 else np.nan,
                "median_range": float(np.ptp(arr)),
            })

    cv_df = pd.DataFrame(cv_rows)
    summary_df = pd.DataFrame(summary_rows)
    return cv_df, summary_df


def plot_block_cv(cv_df: pd.DataFrame, out_path: Path):
    sites = cv_df["system_id"].unique()
    n = len(sites)
    if n == 0:
        return
    ncols = min(n, 3)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    for i, sid in enumerate(sites):
        ax = axes[i // ncols, i % ncols]
        sub = cv_df[cv_df["system_id"] == sid].sort_values("held_out_year")
        years = sub["held_out_year"].values
        meds = sub["HI_median_cv"].values
        q10s = sub["HI_q10_cv"].values
        q90s = sub["HI_q90_cv"].values
        full_med = sub["HI_median_full"].iloc[0]

        ax.errorbar(years, meds, yerr=[meds - q10s, q90s - meds],
                     fmt="o", color="#1f3552", capsize=3, markersize=5)
        ax.axhline(full_med, color="#b03a2e", lw=1.5, ls="--", label=f"full median={full_med:.3f}")
        _cv_names = {
            "USGS-08155500": "Barton Springs",
            "USGS-07014500": "Meramec River near Sullivan",
            "USGS-08169000": "Comal Springs",
            "USGS-08171000": "Blanco River",
            "11148900": "Arroyo Seco near Soledad",
        }
        short = _cv_names.get(str(sid), str(sid).replace("USGS-", ""))
        _panel_label = chr(ord('a') + i)
        ax.text(0.02, 0.95, f"({_panel_label}) {short}", transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")
        ax.set_xlabel("Held-out year")
        ax.set_ylabel("HI median (remaining)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.25)
        ax.tick_params(axis="x", rotation=45)

    # Hide empty panels
    for j in range(n, nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    args = parser.parse_args()

    cv_df, summary_df = run_block_cv(args.package)
    cv_df.to_csv(args.package / "tables" / "benchmark_block_cv.csv", index=False)
    summary_df.to_csv(args.package / "tables" / "benchmark_block_cv_summary.csv", index=False)
    plot_block_cv(cv_df, args.package / "figures" / "benchmark_block_cv.png")

    print(f"[BlockCV] {len(cv_df)} folds across {len(summary_df)} sites")
    if not summary_df.empty:
        for _, row in summary_df.iterrows():
            print(f"  {row['system_id']}: max_abs_dev={row['max_abs_deviation_median']:.4f}, "
                  f"cv={row['cv_of_median']:.4f}, range={row['median_range']:.4f}")


if __name__ == "__main__":
    main()
