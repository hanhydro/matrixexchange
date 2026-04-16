#!/usr/bin/env python3
"""
"""
from pathlib import Path
import numpy as np
import pandas as pd

PKG = Path(__file__).resolve().parent.parent
TABLES = PKG / "tables"

BOOTSTRAP_B = 2000
BOOTSTRAP_SEED = 42
CI_LO_Q = 0.10
CI_HI_Q = 0.90


def bootstrap_median_ci(hi, b=BOOTSTRAP_B, seed=BOOTSTRAP_SEED):
    rng = np.random.default_rng(seed)
    n = len(hi)
    if n < 3:
        return np.nan, np.nan
    bs_meds = np.array(
        [np.median(rng.choice(hi, size=n, replace=True)) for _ in range(b)]
    )
    return float(np.quantile(bs_meds, CI_LO_Q)), float(np.quantile(bs_meds, CI_HI_Q))


def interval_overlap(lo1, hi1, lo2, hi2):
    if not (np.isfinite(lo1) and np.isfinite(hi1) and np.isfinite(lo2) and np.isfinite(hi2)):
        return np.nan, np.nan
    ov_lo = max(lo1, lo2)
    ov_hi = min(hi1, hi2)
    w = max(0.0, ov_hi - ov_lo)
    obs_w = hi1 - lo1
    frac = w / obs_w if obs_w > 0 else 0.0
    return float(w), float(frac)


events = pd.read_csv(TABLES / "benchmark_event_summary.csv")
prov = pd.read_csv(TABLES / "benchmark_provenance.csv")
envelopes = pd.read_csv(TABLES / "benchmark_regime_envelopes.csv")

env_rows = []
for _, r in prov.iterrows():
    sid = str(r["system_id"])
    pref = r["preferred_data_resolution"]
    sub = events[
        (events["system_id"].astype(str) == sid)
        & (events["data_resolution"] == pref)
        & (events["filter_id"] == "baseline")
        & events["HI_obs"].notna()
    ]
    hi = sub["HI_obs"].to_numpy(dtype=float)
    n = int(hi.size)
    q10, q90 = (float(np.quantile(hi, 0.10)), float(np.quantile(hi, 0.90))) if n >= 3 else (np.nan, np.nan)
    med = float(np.median(hi)) if n > 0 else np.nan
    ci_lo, ci_hi = bootstrap_median_ci(hi)
    env_rows.append(
        {
            "source": str(r["source"]),
            "system_id": sid,
            "R_lo": float(r["R_lo"]),
            "R_hi": float(r["R_hi"]),
            "n_events_baseline_preferred": n,
            "HI_obs_event_q10": q10,
            "HI_obs_event_q90": q90,
            "HI_obs_median": med,
            "HI_obs_median_ci_lo": ci_lo,
            "HI_obs_median_ci_hi": ci_hi,
            # classifier interval (explicit alias; lo/hi = median_ci_lo/hi)
            "HI_obs_lo": ci_lo,
            "HI_obs_hi": ci_hi,
        }
    )
hi_envelope = pd.DataFrame(env_rows)

# Attach envelope bounds (ridge, floor, simple_off_ridge) per site
env_wide = envelopes.pivot_table(
    index=["source", "system_id"],
    columns="regime_id",
    values=["HI_lo", "HI_hi", "HI_median"],
    aggfunc="first",
)
env_wide.columns = [f"{reg}_{stat}" for stat, reg in env_wide.columns]
env_wide = env_wide.reset_index()
env_wide["source"] = env_wide["source"].astype(str)
env_wide["system_id"] = env_wide["system_id"].astype(str)

hi_envelope = hi_envelope.merge(env_wide, on=["source", "system_id"], how="left")

# Rebuild regime_consistency from the corrected hi_envelope
reg_rows = []
best_rows = []
for _, row in hi_envelope.iterrows():
    sid = str(row["system_id"])
    obs_lo = float(row["HI_obs_lo"])
    obs_hi = float(row["HI_obs_hi"])
    obs_med = float(row["HI_obs_median"])
    site_rows = []
    for reg in ["ridge", "floor", "simple_off_ridge"]:
        lo = float(row[f"{reg}_HI_lo"])
        hi = float(row[f"{reg}_HI_hi"])
        med = float(row[f"{reg}_HI_median"])
        ov_w, ov_f = interval_overlap(obs_lo, obs_hi, lo, hi)
        med_dist = abs(obs_med - med) if np.isfinite(obs_med) and np.isfinite(med) else np.nan
        entry = {
            "source": str(row["source"]),
            "system_id": sid,
            "regime_id": reg,
            "n_events_baseline_preferred": int(row["n_events_baseline_preferred"]),
            "HI_obs_median": obs_med,
            "HI_obs_median_ci_lo": obs_lo,
            "HI_obs_median_ci_hi": obs_hi,
            "HI_obs_event_q10": float(row["HI_obs_event_q10"]),
            "HI_obs_event_q90": float(row["HI_obs_event_q90"]),
            "obs_interval_width": obs_hi - obs_lo,
            "envelope_HI_lo": lo,
            "envelope_HI_hi": hi,
            "envelope_HI_median": med,
            "overlap_width": ov_w,
            "overlap_fraction_obs": ov_f,
            "distance_obs_median_to_regime_median": med_dist,
        }
        reg_rows.append(entry)
        site_rows.append(entry)

    # Apply the same decision rule as before:
    # - overlap_first if overlap is discriminating;
    # - distance_when_overlap_nondiscriminating if >=1 regime fully contains obs
    #   AND >=2 regimes have >=50% overlap;
    # - unclassified if no regime has >=5% overlap.
    sub = pd.DataFrame(site_rows)
    overlap_first = sub.sort_values(
        ["overlap_fraction_obs", "overlap_width", "distance_obs_median_to_regime_median"],
        ascending=[False, False, True],
    ).iloc[0]
    full_cover = sub[sub["overlap_fraction_obs"] >= 0.999]
    substantial = sub[sub["overlap_fraction_obs"] >= 0.5]
    if (not full_cover.empty) and (len(substantial) >= 2):
        preferred = substantial.sort_values(
            ["distance_obs_median_to_regime_median", "overlap_fraction_obs", "overlap_width"],
            ascending=[True, False, False],
        ).iloc[0]
        rule = "distance_when_overlap_nondiscriminating"
        reason = (
            "At least one regime fully contains the observed interval and multiple "
            "regimes cover at least half of it, so distance-to-median breaks the tie."
        )
    elif float(overlap_first["overlap_fraction_obs"]) < 0.05:
        preferred = overlap_first.copy()
        preferred["regime_id"] = "unclassified"
        rule = "no_overlap"
        reason = (
            "No regime achieves at least 5% overlap with the observed interval; "
            "site is classified as unclassified rather than assigned by distance alone."
        )
    else:
        preferred = overlap_first
        rule = "overlap_first"
        reason = "Overlap fraction is discriminating; regimes ranked by overlap first."
    best_rows.append(
        {
            "source": str(row["source"]),
            "system_id": sid,
            "best_regime_id": str(preferred["regime_id"]),
            "best_overlap_fraction_obs": float(preferred["overlap_fraction_obs"]),
            "best_distance_obs_median_to_regime_median": float(preferred["distance_obs_median_to_regime_median"]),
            "best_regime_id_overlap_first": str(overlap_first["regime_id"]),
            "preferred_regime_by_rule": str(preferred["regime_id"]),
            "preference_rule_applied": rule,
            "preference_reason": reason,
        }
    )

reg_df = pd.DataFrame(reg_rows)
best_df = pd.DataFrame(best_rows)
reg_out = reg_df.merge(best_df, on=["source", "system_id"], how="left")

# Save
hi_envelope.to_csv(TABLES / "benchmark_hi_envelope.csv", index=False)
reg_out.to_csv(TABLES / "benchmark_regime_consistency.csv", index=False)

print(f"Wrote {TABLES / 'benchmark_hi_envelope.csv'} ({len(hi_envelope)} sites)")
print(f"Wrote {TABLES / 'benchmark_regime_consistency.csv'} ({len(reg_out)} regime rows across {reg_out['system_id'].nunique()} sites)")
print()
print("Site-level preferred-regime summary:")
for _, b in best_df.iterrows():
    print(f"  {b['system_id']:>14}: {b['preferred_regime_by_rule']:>20}  "
          f"overlap={b['best_overlap_fraction_obs']:.3f}  "
          f"med_dist={b['best_distance_obs_median_to_regime_median']:.3f}  "
          f"rule={b['preference_rule_applied']}")
