#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dual-porosity hysteresis model.

Configuration:
  - Triangular forcing (HYETO = "tri")
  - IVP replacement via independent Radau solver
  - Axis-wise HI normalization: Q_CUTOFF_FRAC = 1e-8, SPAN_MIN_DECADES = 0.05
  - Latin-hypercube sweep over (K, Da x R) space

The script contains the model-side HI implementation (hysteresis_index_Qspace),
the numerical workflow for the parameter sweep, and the solver-replacement
gating logic used to generate the manuscript tables and figures.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import json
import zipfile
from datetime import datetime
from scipy.integrate import solve_ivp
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.signal import find_peaks
from joblib import Parallel, delayed
# Numba JIT for performance-critical functions
try:
    from numba import njit
    HAS_NUMBA = True
except Exception as e:
    HAS_NUMBA = False
    try:
        print(f"[Numba] disabled ({e})")
    except Exception:
        pass
    # Fallback: no-op decorator
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if args and callable(args[0]) else decorator

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# --- Sampling Mode ---
# RK4 stability controls
SUBSTEP_TARGET = 1.0      # target dimensionless stiffness per output step
NSUB_MAX = 2000            # max substeps per output step (tradeoff speed/stability)
MIN_NSUB_EVENT = 16       # accuracy floor: minimum substeps per output step during storm + early recession
# Exchange stiffness handling
USE_EXCHANGE_EXACT = True    # Strang-split RK4 with exact exchange update (removes alpha stiffness)
EXCHANGE_SPLIT = "strang"    # "strang" (recommended) or "lie"
EXACT_EXCHANGE_SUBSTEP_FACTOR = 0.0  # include a fraction of exchange rate in substep selection (controls splitting error)
H_MAX = 1e6               # head clamp threshold; above this treat as unstable
# Selective fallback for rare suspect cases (keeps charts accurate without slowing the full sweep)
FALLBACK_SOLVEIVP = True      # run solve_ivp only for a tiny subset of stiff/suspect RK4 cases
FALLBACK_METHOD = "Radau"     # reference integrator
FALLBACK_NSUB = 500           # hard trigger when RK4 needed >= this many substeps in any output interval
FULL_IVP_MODE = False         # when True, skip RK4 entirely and always use solve_ivp(Radau) for all realizations

# Soft fallback for bimodality verification / split-error guard (high-R, moderate Da, unexpectedly large HI)
# Motivation: some cases can show large HI bias vs Radau even when nsub_max is modest (operator splitting artifacts).
# We validate these cases by running solve_ivp and (optionally) replacing the RK4 result.
FALLBACK_SOFT_BIMODAL = True
FALLBACK_SOFT_R_MIN = 2.4  # expanded to cover R~2.4–3.5 where RK4 mid-Da hump artifacts can appear          # apply only to large-contrast systems
FALLBACK_SOFT_DA_MIN = 1e-2         # focus on moderate-to-large Da where spurious loops can appear
FALLBACK_SOFT_DA_MAX = 1.2          # avoid spending ivp on extreme tail
FALLBACK_SOFT_HI_MIN = 0.03         # only if RK4 HI is large enough to matter for peaks/ridges
FALLBACK_SOFT_NSUB_MAX = 9999        # only when RK4 did NOT need many substeps (not already caught by hard fallback)
FALLBACK_SOFT_ABS_DHI_TOL = 0.02    # if ivp differs from RK4 by > this, replace (always store both)
FALLBACK_SOFT_ALWAYS_REPLACE = True # if True, always replace with ivp when soft-triggered (recommended for verification)


USE_DAR_SAMPLING = True          # LHC over (K, DaR_target) instead of (K, alpha)
AUTO_BRACKET = False              # NOTE: rerun/expansion logic not implemented in this file              # Expand bounds when peak hits boundary
REFINE_DAR_ONCE = True           # 1-step refinement to hit target DaR
USE_PARALLEL = True
PAR_BACKEND = "threading"   # important: avoids pickling, and numba releases the GIL
CHUNK_SIZE = 200            # each job handles 200 ODEs (tune 100–500)
N_JOBS = -1
P_VEC_GLOBAL = None
def _set_globals_for_workers(P_vec):
    global P_VEC_GLOBAL
    P_VEC_GLOBAL = P_vec



# --- Bootstrap Settings ---
BOOTSTRAP = True
N_BOOT = 2000
BOOT_SEED = 123
BOOT_FRAC = 1.0
RIDGE_CI_Q = (0.16, 0.84)        # ~68% CI

# --- Ridge / field-guide exports ---
RIDGE_CI_LOGSPACE = True          # compute CIs in log10-space, then exponentiate
PEAK_WIDTH_FRAC = 0.98            # primary bandwidth threshold: HI >= frac * HI_max
PEAK_WIDTH_FRACS = (0.98, 0.95, 0.90)   # compute widths at multiple thresholds (decades)
PEAK_WIDTH_INTERP = True          # interpolate threshold crossings in log10(Da)
PEAK_CENTROID_FRACS = (0.95, 0.90)   # compute centroid in top band (often more stable than argmax)
PEAK_CENTROID_WEIGHT = "excess"      # "excess" weights use (y - thr); "raw" uses y

# --- Bimodality / multi-lobe peak diagnostics ---
BIMODALITY_DETECT = True
BIMODAL_PROM_FRAC = 0.05         # find_peaks prominence threshold as fraction of (segment) max HI
BIMODAL_MIN_SEP_DECADES = 1.0    # require peaks separated by >= this in log10(Da) decades
BIMODAL_SECONDARY_FRAC = 0.50    # secondary peak must be >= this fraction of global peak HI to call bimodal
DA_LOBE_SPLIT = 1e-2             # split between 'low-Da' and 'high-Da' lobes for reporting
# To mitigate solver-policy "kinks" right at the low/high lobe split, the publication-domain
# replacement gate can be slightly wider than DA_LOBE_SPLIT. DA_LOBE_SPLIT remains the
# reporting split used in diagnostics; the gate band is controlled by LOWDA_GATE_BUFFER_DECADES.
LOWDA_GATE_BUFFER_DECADES = 0.5  # 0.5 decades => factor ~3.16 above DA_LOBE_SPLIT

# --- Additional publication-domain gate for LOW-Da, HIGH-contrast regime ---
# RK4 can be inaccurate here even when nsub_max is modest (so hard fallback won't catch it).
FALLBACK_SOFT_LOWDA = True
FALLBACK_SOFT_LOWDA_R_MIN = 20.0              # start where low-Da RK4 errors become material
FALLBACK_SOFT_LOWDA_DA_MIN = 1e-10            # basically the whole low lobe
FALLBACK_SOFT_LOWDA_DA_MAX = DA_LOBE_SPLIT * (10.0 ** LOWDA_GATE_BUFFER_DECADES)  # widened band for publication-domain continuity
FALLBACK_SOFT_LOWDA_NSUB_MAX = 9999           # optional (won't block anything unless you change it)
FALLBACK_SOFT_LOWDA_ALWAYS_REPLACE = True     # recommended: always replace with IVP in this regime

# --- Mid-Da 'hump band' diagnostics (explicit band, independent of DA_LOBE_SPLIT) ---
DA_MID_MIN = 5e-2
DA_MID_MAX = 1.2
MID_DA_RATIO_THRESH = 0.50   # hump deemed 'material' if HI_mid / HI_low >= this

# Ridge overlay choice: we always compute both, but can choose one to plot as the main ridge
RIDGE_OVERLAY_MODE = "global"   # "global" or "primary_lowDa"
PLOT_BOTH_RIDGES = True          # overlay both global and primary_lowDa ridges in Panel A

# --- Bimodality verification (by source) ---
BIMODALITY_SOURCE_COMPARE = True
BIMODALITY_SOURCE_MIN_N = 10   # minimum n in the ivp-only subset to attempt a peak/bimodality check
WRITE_RUN_OUTPUTS = True          # write ridge.csv + compact progress summary
OUTDIR_BASE = "hysteresis_outputs"
SAVE_RUNS_CSV = True              # runs.csv can be large; set False if you only want ridge.csv
MAKE_CORE_ZIP = True             # create a timestamped ZIP with core outputs for easy upload/sharing
ZIP_INCLUDE_RUNS_CSV = True        # include runs.csv inside the ZIP (often useful; still compressed)

# --- Ridge-centered publication-domain gate (activated after ridge-core audit failure) ---
# Uses a previously audited centroid ridge as a reference band. This is only a verification
# policy, not a new scientific ridge fit: cases whose achieved Da land inside the publish
# domain around the reference centroid are always replaced by IVP truth in production.
FALLBACK_SOFT_RIDGE = True
FALLBACK_SOFT_RIDGE_BAND_DECADES = 0.75
FALLBACK_SOFT_RIDGE_MIN_BAND_DECADES = 0.75
FALLBACK_SOFT_RIDGE_WIDTH_FRAC = 0.50
FALLBACK_SOFT_RIDGE_WIDTH_BUFFER_DECADES = 0.15
FALLBACK_SOFT_RIDGE_NSUB_MAX = 9999
FALLBACK_SOFT_RIDGE_ALWAYS_REPLACE = True
FALLBACK_SOFT_RIDGE_REFERENCE_CSV = os.path.join("hysteresis_outputs", "run_20260312_121717", "ridge.csv")
FALLBACK_SOFT_RIDGE_REFERENCE_CSV_FALLBACKS = [
    os.path.join("hysteresis_outputs", "run_20260312_121717", "ridge.csv"),
    os.path.join("specialist_paper_package_rigor_20260312_121717_v6", "inputs", "ridge.csv"),
    os.path.join("specialist_paper_package_rigor_20260312_121717_v6", "tables", "centroid_ridge_main.csv"),
    os.path.join("hysteresis_outputs", "run_20260311_051944", "ridge.csv"),
    os.path.join("specialist_paper_package_20260311_051944", "tables", "centroid_ridge_main.csv"),
    os.path.join("specialist_paper_package_20260311_051944", "inputs", "ridge.csv"),
]
FALLBACK_SOFT_RIDGE_REFERENCE_COL = "Da_centroid_f095"
FALLBACK_SOFT_RIDGE_WIDTH_COL = "width_decades_f095"
FALLBACK_SOFT_RIDGE_REFERENCE_CSV_RESOLVED = ""

# --- Shoulder publication-domain gate (donut around ridge: between inner and outer radius) ---
# Covers shoulder zone that is NOT caught by the ridge gate (which only covers ±0.75 dec)
# or the bimodal gate (which requires R >= 2.4). Shoulder error is worst at LOW R.
FALLBACK_SOFT_SHOULDER = True
FALLBACK_SOFT_SHOULDER_INNER_DEC = 0.25     # inner radius (core already covered by ridge gate)
FALLBACK_SOFT_SHOULDER_OUTER_DEC = 1.25     # outer radius (matches shoulder-rescue pilot range)
FALLBACK_SOFT_SHOULDER_NSUB_MAX = 9999
FALLBACK_SOFT_SHOULDER_ALWAYS_REPLACE = True

# --- Parameter Bounds ---
LOG10_DAR_MIN, LOG10_DAR_MAX = -6.0, 3.0
LOG10K_MIN, LOG10K_MAX = -2.0, 2.5
LOG10A_MIN, LOG10A_MAX = -3.0, 1.0
LOG10K_MAX_EXT = 3.5             # Extended for auto-bracketing
LOG10A_MIN_EXT = -5.0            # Extended for auto-bracketing
H_REF = 0.2                      # Reference head for DaR -> alpha

# --- Peak Detection ---
N_BINS_PEAK = 40
SMOOTH_SIGMA = 1.2
# Smoothing robustness option: when n_bins changes (e.g., ridge sensitivity audit),
# scale the Gaussian kernel in *bin units* so the effective smoothing width stays roughly
# constant in log10(x) decades.
SMOOTH_SIGMA_SCALE_WITH_NBINS = True

# Publication-grade peak extraction: reduce sensitivity to bin edges by resampling the
# supported, smoothed binned curve onto a fine log10(x) grid before extracting peak,
# top-band centroid(s), and resonance width(s).
PEAK_REFINE = True
PEAK_REFINE_N_FINE = 400
PEAK_REFINE_SMOOTH_DECADES = 0.00   # extra smoothing on refined grid (log10-decades); 0 disables

Q_LOW, Q_HIGH = 0.05, 0.95
NMIN_PER_BIN = 2
MIN_VALID_BINS_FOR_PEAK = 10

# --- Hysteresis metric guards ---
# Cut the post-storm recession tail at a fixed fraction of the peak signal.
# This prevents ultra-late-time, ultra-low-Q behavior from dominating the loop metric.
Q_CUTOFF_FRAC = 1e-8          # keep times where Q_out >= Q_out_peak * Q_CUTOFF_FRAC
RQ_CUTOFF_FRAC = 1e-8         # keep times where (-dQ/dt) >= rQ_peak * RQ_CUTOFF_FRAC

# Minimum dynamic range (in decades) required to compute a stable normalized area.
SPAN_MIN_DECADES = 0.05       # if log-span in Q or rQ is below this, treat as unsupported

# Physical sanity: normalized polygon area in a bounding box should be O(0..1).
HI_CLIP_MAX = 1.0

# --- Mechanism Diagnostics ---
THETA_DH = 0.2                   # Persistence threshold: ΔH > θ·ΔH_max
POST_ONLY = True                 # Diagnostics on t > t_storm_end only

# --- Hyetograph ---
HYETO = "tri"                    # defended manuscript configuration; "box" and "gamma" branches below are archived exploratory utilities not used for manuscript outputs
TRI_PEAK_FRAC = 0.5
GAMMA_SHAPE = 3.0
GAMMA_EPS = 1e-12

# --- Discharge Law: Q = Q_BASE + K_LIN*H + K*H² ---
Q_BASE = 0.0
K_LIN = 0.0

# --- Simulation Parameters ---
SEED = 42
n_lhc_per_contrast = 600
T_END = 100.0
N_T = 800
RTOL, ATOL = 1e-6, 1e-9

# --- Scenarios ---
Sy_f = 0.05
#Contrast_range = [1.2, 1.5, 2, 12, 22, 32, 42, 52, 62, 82, 102]
Contrast_range = [1.2, 1.4, 1.7, 2.0, 2.4, 2.9, 3.5, 4.2, 5.1, 6.2, 7.6, 9.2, 11.2, 13.6, 16.5, 20.1, 24.4, 29.7, 36.1, 43.9, 53.3, 64.7, 78.6, 95.4, 115.8]
P_mag, P_dur = 40.0, 8.0
t0_storm = 1.0
t_storm_end = t0_storm + P_dur
t_eval = np.linspace(0.0, T_END, N_T)

USE_RK4 = True          # if True: use numba RK4 fixed-step integrator (fastest)
DT = float(t_eval[1] - t_eval[0])
N_STEPS = int(len(t_eval))


# --- Validation / run modes ---
# Runs a small RK4 vs solve_ivp spot-check before the full LHC to confirm fidelity.
RUN_SPOTCHECK = True
DO_FULL_LHC = True
SPOTCHECK_N = 20
SPOTCHECK_R_LEVELS = (1.2, 11.2, 64.7, 115.8)
SPOTCHECK_SEED = 12345

RUN_SAT_SPOTCHECK = True
SAT_SPOTCHECK_N_SAT = 20
SAT_SPOTCHECK_N_UNSAT = 20
SAT_SPOTCHECK_SEED = 24680

# When HI is tiny, relative errors can look enormous; report a floored relative error too.
HI_REL_FLOOR = 0.01  # spotcheck: rel = dHI / max(|HI_ref|, HI_REL_FLOOR)


# --- Ridge extraction robustness (lightweight) ---
RUN_RIDGE_SENSITIVITY = True
SENS_NBINS_LIST = (30, 40, 50)
SENS_SIGMA_LIST = (0.8, 1.2, 1.6)
SENS_NMIN_PER_BIN = 5
SENS_MIN_VALID_BINS = 10



# ==============================================================================
# SAMPLING FUNCTIONS
# ==============================================================================

def _latin_hypercube(n: int, d: int, seed: int = 0) -> np.ndarray:
    """Lightweight LHC fallback. Returns n×d array in [0,1)."""
    rng = np.random.default_rng(seed)
    u = np.zeros((n, d))
    for j in range(d):
        cut = (np.arange(n) + rng.random(n)) / n
        rng.shuffle(cut)
        u[:, j] = cut
    return u


def sample_lhc_log_targets(n: int, seed: int = 0):
    """LHC in 2D over (log10K, log10DaR_target)."""
    try:
        from scipy.stats import qmc
        u = qmc.LatinHypercube(d=2, seed=seed).random(n=n)
    except ImportError:
        u = _latin_hypercube(n=n, d=2, seed=seed)

    logK = LOG10K_MIN + u[:, 0] * (LOG10K_MAX - LOG10K_MIN)
    logDaR = LOG10_DAR_MIN + u[:, 1] * (LOG10_DAR_MAX - LOG10_DAR_MIN)
    return 10.0 ** logK, 10.0 ** logDaR, logK, logDaR


def sample_lhc_log_params(n: int, seed: int = 0):
    """LHC in 2D over (log10K, log10alpha)."""
    try:
        from scipy.stats import qmc
        u = qmc.LatinHypercube(d=2, seed=seed).random(n=n)
    except ImportError:
        u = _latin_hypercube(n=n, d=2, seed=seed)

    logK = LOG10K_MIN + u[:, 0] * (LOG10K_MAX - LOG10K_MIN)
    logA = LOG10A_MIN + u[:, 1] * (LOG10A_MAX - LOG10A_MIN)
    return 10.0 ** logK, 10.0 ** logA, logK, logA


def sample_lhc_log_params_bounds(n: int, seed: int, logK_min: float, logK_max: float,
                                  logA_min: float, logA_max: float):
    """LHC with custom bounds (for auto-bracketing)."""
    try:
        from scipy.stats import qmc
        u = qmc.LatinHypercube(d=2, seed=seed).random(n=n)
    except ImportError:
        u = _latin_hypercube(n=n, d=2, seed=seed)

    logK = logK_min + u[:, 0] * (logK_max - logK_min)
    logA = logA_min + u[:, 1] * (logA_max - logA_min)
    return 10.0 ** logK, 10.0 ** logA, logK, logA


def _load_ridge_reference_bundle(
    path: str,
    da_col: str = "Da_centroid_f095",
    width_col: str = "width_decades_f095",
):
    """Load an audited ridge table used to define the ridge-centered IVP gate."""
    global FALLBACK_SOFT_RIDGE_REFERENCE_CSV_RESOLVED
    if (not path) or (not bool(FALLBACK_SOFT_RIDGE)):
        return {}, {}, pd.DataFrame()
    candidates = [str(path)]
    for extra in FALLBACK_SOFT_RIDGE_REFERENCE_CSV_FALLBACKS:
        extra = str(extra)
        if extra and (extra not in candidates):
            candidates.append(extra)
    last_err = None
    for candidate in candidates:
        try:
            ref = pd.read_csv(candidate)
            if ("Contrast" not in ref.columns) or (da_col not in ref.columns):
                print(f"[RidgeGate] skip {candidate}: missing Contrast/{da_col}")
                continue
            da_map = {}
            width_map = {}
            rows = []
            for _, row in ref.iterrows():
                try:
                    R = float(row["Contrast"])
                    da = float(row[da_col])
                except Exception:
                    continue
                if np.isfinite(R) and np.isfinite(da) and (da > 0):
                    key = round(R, 6)
                    da_map[key] = da
                    width = np.nan
                    try:
                        width = float(row.get(width_col, np.nan))
                    except Exception:
                        width = np.nan
                    if np.isfinite(width) and (width > 0):
                        width_map[key] = width
                    rows.append(
                        {
                            "Contrast": float(R),
                            "Da_reference": float(da),
                            "width_decades_f095_reference": float(width) if np.isfinite(width) else np.nan,
                        }
                    )
            if len(da_map) == 0:
                print(f"[RidgeGate] skip {candidate}: no finite ridge references")
                continue
            FALLBACK_SOFT_RIDGE_REFERENCE_CSV_RESOLVED = candidate
            print(f"[RidgeGate] loaded {len(da_map)} reference centroids from {candidate}")
            return da_map, width_map, pd.DataFrame(rows).sort_values("Contrast").reset_index(drop=True)
        except Exception as e:
            last_err = e
    if last_err is not None:
        print(f"[RidgeGate] disabled: could not load any reference table ({last_err})")
    return {}, {}, pd.DataFrame()


FALLBACK_SOFT_RIDGE_MAP, FALLBACK_SOFT_RIDGE_WIDTH_MAP, FALLBACK_SOFT_RIDGE_REFERENCE_TABLE = _load_ridge_reference_bundle(
    FALLBACK_SOFT_RIDGE_REFERENCE_CSV,
    da_col=FALLBACK_SOFT_RIDGE_REFERENCE_COL,
    width_col=FALLBACK_SOFT_RIDGE_WIDTH_COL,
)


def _lookup_ridge_reference_da(contrast_val: float) -> float:
    """Return the reference centroid Da for a contrast, using a small tolerance on R."""
    if not FALLBACK_SOFT_RIDGE_MAP:
        return np.nan
    key = round(float(contrast_val), 6)
    if key in FALLBACK_SOFT_RIDGE_MAP:
        return float(FALLBACK_SOFT_RIDGE_MAP[key])
    keys = np.asarray(list(FALLBACK_SOFT_RIDGE_MAP.keys()), float)
    if keys.size == 0:
        return np.nan
    i = int(np.argmin(np.abs(keys - float(contrast_val))))
    if abs(float(keys[i]) - float(contrast_val)) <= 1e-3:
        return float(FALLBACK_SOFT_RIDGE_MAP[float(keys[i])])
    return np.nan


def _lookup_ridge_reference_width(contrast_val: float) -> float:
    """Return the reference f095 width (decades in log10(Da)) for a contrast."""
    if not FALLBACK_SOFT_RIDGE_WIDTH_MAP:
        return np.nan
    key = round(float(contrast_val), 6)
    if key in FALLBACK_SOFT_RIDGE_WIDTH_MAP:
        return float(FALLBACK_SOFT_RIDGE_WIDTH_MAP[key])
    keys = np.asarray(list(FALLBACK_SOFT_RIDGE_WIDTH_MAP.keys()), float)
    if keys.size == 0:
        return np.nan
    i = int(np.argmin(np.abs(keys - float(contrast_val))))
    if abs(float(keys[i]) - float(contrast_val)) <= 1e-3:
        return float(FALLBACK_SOFT_RIDGE_WIDTH_MAP[float(keys[i])])
    return np.nan


def _ridge_reference_half_band_decades(contrast_val: float, band_decades: float = None) -> float:
    """Return the publication-domain half-band around the audited ridge centroid."""
    if band_decades is not None and np.isfinite(band_decades) and (band_decades > 0):
        return float(abs(band_decades))
    width = _lookup_ridge_reference_width(float(contrast_val))
    dyn = np.nan
    if np.isfinite(width) and (width > 0):
        dyn = float(FALLBACK_SOFT_RIDGE_WIDTH_FRAC) * float(width) + float(FALLBACK_SOFT_RIDGE_WIDTH_BUFFER_DECADES)
    base = float(FALLBACK_SOFT_RIDGE_MIN_BAND_DECADES)
    if np.isfinite(dyn) and (dyn > 0):
        return float(max(base, dyn))
    return float(base)


def _ridge_reference_band_hit(contrast_val: float, da_val: float, band_decades: float = None) -> bool:
    """Check whether achieved Da lies inside the audited centroid-ridge band."""
    if (not bool(FALLBACK_SOFT_RIDGE)) or (not np.isfinite(da_val)) or (da_val <= 0):
        return False
    da_ref = _lookup_ridge_reference_da(float(contrast_val))
    if (not np.isfinite(da_ref)) or (da_ref <= 0):
        return False
    band_decades = _ridge_reference_half_band_decades(float(contrast_val), band_decades=band_decades)
    dist = abs(np.log10(float(da_val)) - np.log10(float(da_ref)))
    return bool(np.isfinite(dist) and (dist <= float(abs(band_decades))))


def _ridge_reference_distance(contrast_val: float, da_val: float) -> float:
    """Return log10-distance from Da to the audited centroid ridge. NaN if unavailable."""
    if (not np.isfinite(da_val)) or (da_val <= 0):
        return np.nan
    da_ref = _lookup_ridge_reference_da(float(contrast_val))
    if (not np.isfinite(da_ref)) or (da_ref <= 0):
        return np.nan
    return float(abs(np.log10(float(da_val)) - np.log10(float(da_ref))))


def _shoulder_band_hit(contrast_val: float, da_val: float) -> bool:
    """Check whether achieved Da lies in the shoulder zone (donut between inner and outer radius)."""
    if not bool(FALLBACK_SOFT_SHOULDER):
        return False
    dist = _ridge_reference_distance(float(contrast_val), float(da_val))
    if not np.isfinite(dist):
        return False
    return bool(dist > float(FALLBACK_SOFT_SHOULDER_INNER_DEC) and dist <= float(FALLBACK_SOFT_SHOULDER_OUTER_DEC))


def _build_publish_domain_gate_summary() -> pd.DataFrame:
    """Summarize the contrast-specific IVP publication-domain gate used for the rerun."""
    if FALLBACK_SOFT_RIDGE_REFERENCE_TABLE is None or FALLBACK_SOFT_RIDGE_REFERENCE_TABLE.empty:
        return pd.DataFrame()
    rows = []
    for _, row in FALLBACK_SOFT_RIDGE_REFERENCE_TABLE.iterrows():
        try:
            R = float(row["Contrast"])
            da_ref = float(row["Da_reference"])
        except Exception:
            continue
        if (not np.isfinite(R)) or (not np.isfinite(da_ref)) or (da_ref <= 0):
            continue
        width_ref = float(row.get("width_decades_f095_reference", np.nan))
        half_band = _ridge_reference_half_band_decades(R)
        fac = float(10.0 ** half_band)
        rows.append(
            {
                "Contrast": float(R),
                "Da_reference": float(da_ref),
                "width_decades_f095_reference": float(width_ref) if np.isfinite(width_ref) else np.nan,
                "half_band_decades_used": float(half_band),
                "Da_band_lo": float(da_ref / fac),
                "Da_band_hi": float(da_ref * fac),
                "band_formula": "max(min_band, width_frac*width_decades_f095 + buffer)",
                "min_band_decades": float(FALLBACK_SOFT_RIDGE_MIN_BAND_DECADES),
                "width_frac": float(FALLBACK_SOFT_RIDGE_WIDTH_FRAC),
                "buffer_decades": float(FALLBACK_SOFT_RIDGE_WIDTH_BUFFER_DECADES),
                "reference_csv": str(FALLBACK_SOFT_RIDGE_REFERENCE_CSV_RESOLVED or FALLBACK_SOFT_RIDGE_REFERENCE_CSV),
            }
        )
    return pd.DataFrame(rows).sort_values("Contrast").reset_index(drop=True)


# ==============================================================================
# PHYSICS (Numba JIT-compiled for 10-50x speedup)
# ==============================================================================
@njit(cache=True, fastmath=True)
def _rhs_jit(t, Hf, Hm, K, alpha, Sy_f, Sy_m, P_mag, P_dur, t0_storm, peak_frac, Q_BASE, K_LIN, is_tri):
    # identical physics as _universal_model_jit, but returns tuple (dHf,dHm)
    if Hf < 0.0: Hf = 0.0
    if Hm < 0.0: Hm = 0.0

    if is_tri:
        P = _precip_scalar_tri_jit(t, P_mag, P_dur, t0_storm, peak_frac)
    else:
        P = _precip_scalar_box_jit(t, P_mag, P_dur, t0_storm)

    Q_out = Q_BASE + K_LIN * Hf + K * Hf * Hf
    Q_ex  = alpha * (Hf - Hm)

    dHf = (P - Q_out - Q_ex) / Sy_f
    dHm = Q_ex / Sy_m

    if Hf <= 0.0 and dHf < 0.0: dHf = 0.0
    if Hm <= 0.0 and dHm < 0.0: dHm = 0.0
    return dHf, dHm


@njit(cache=True, fastmath=True)
def _rhs_noex_jit(t, Hf, Hm, K, Sy_f, P_mag, P_dur, t0_storm, peak_frac, Q_BASE, K_LIN, is_tri):
    """RHS for the non-exchange subsystem: dHf/dt = (P - Qout)/Sy_f ; dHm/dt = 0.
    Exchange is handled separately via an exact update (operator splitting).
    """
    if is_tri:
        P = _precip_scalar_tri_jit(t, P_mag, P_dur, t0_storm, peak_frac)
    else:
        P = _precip_scalar_box_jit(t, P_mag, P_dur, t0_storm)

    Q_out = Q_BASE + K_LIN * Hf + K * Hf * Hf
    dHf = (P - Q_out) / Sy_f
    if Hf <= 0.0 and dHf < 0.0:
        dHf = 0.0
    return dHf


@njit(cache=True, fastmath=True)
def exchange_exact_step_jit(hf, hm, alpha, Sy_f, Sy_m, dt):
    """Exact update for linear fracture–matrix exchange over dt.

    Exchange-only subsystem:
      dHf/dt = -alpha*(Hf-Hm)/Sy_f
      dHm/dt = +alpha*(Hf-Hm)/Sy_m

    Let Δ = Hf-Hm. Then:
      dΔ/dt = -alpha*(1/Sy_f + 1/Sy_m)*Δ  =>  Δ(t+dt) = Δ(t)*exp(-λ dt)
    and the storage-weighted mean M = Sy_f*Hf + Sy_m*Hm is conserved.

    Returns updated (hf, hm), with nonnegativity clamps.
    """
    if alpha <= 0.0 or dt <= 0.0:
        return hf, hm

    lam = alpha * (1.0 / Sy_f + 1.0 / Sy_m)
    # underflow to 0 is fine for very stiff cases
    decay = np.exp(-lam * dt)

    d = hf - hm
    d_new = d * decay

    M = Sy_f * hf + Sy_m * hm
    denom = Sy_f + Sy_m

    hf_new = (M + Sy_m * d_new) / denom
    hm_new = hf_new - d_new

    if hf_new < 0.0:
        hf_new = 0.0
    if hm_new < 0.0:
        hm_new = 0.0

    return hf_new, hm_new


from math import isfinite  # at top (optional; we also use np.isfinite)

@njit(cache=True, fastmath=True)
def rk4_integrate_arrays_sub(K, alpha, Sy_f, Sy_m, P_mag, P_dur, t0_storm, peak_frac,
                             Q_BASE, K_LIN, is_tri,
                             dt, n_steps, n_sub,
                             Hf0=0.1, Hm0=0.1,
                             H_MAX=1e6):
    """
    Fixed output grid RK4 with per-run substepping.
    - dt is the base output-step (t_eval spacing)
    - n_sub is number of substeps per output step
    - returns (Hf,Hm, ok_flag)
    """
    Hf = np.empty(n_steps, dtype=np.float64)
    Hm = np.empty(n_steps, dtype=np.float64)

    hf = Hf0
    hm = Hm0
    t = 0.0

    # guard n_sub
    if n_sub < 1:
        n_sub = 1
    if n_sub > 2000:
        n_sub = 2000  # hard cap to prevent runaway

    dt_sub = dt / n_sub

    ok = True
    for i in range(n_steps):
        Hf[i] = hf
        Hm[i] = hm

        # do not integrate beyond final stored time
        if i == n_steps - 1:
            break

        # integrate within this output interval using n_sub small steps
        for _ in range(n_sub):
            # RK4 stages at substep scale
            k1f, k1m = _rhs_jit(t, hf, hm, K, alpha, Sy_f, Sy_m, P_mag, P_dur,
                                t0_storm, peak_frac, Q_BASE, K_LIN, is_tri)

            k2f, k2m = _rhs_jit(t + 0.5*dt_sub,
                                hf + 0.5*dt_sub*k1f, hm + 0.5*dt_sub*k1m,
                                K, alpha, Sy_f, Sy_m, P_mag, P_dur,
                                t0_storm, peak_frac, Q_BASE, K_LIN, is_tri)

            k3f, k3m = _rhs_jit(t + 0.5*dt_sub,
                                hf + 0.5*dt_sub*k2f, hm + 0.5*dt_sub*k2m,
                                K, alpha, Sy_f, Sy_m, P_mag, P_dur,
                                t0_storm, peak_frac, Q_BASE, K_LIN, is_tri)

            k4f, k4m = _rhs_jit(t + dt_sub,
                                hf + dt_sub*k3f, hm + dt_sub*k3m,
                                K, alpha, Sy_f, Sy_m, P_mag, P_dur,
                                t0_storm, peak_frac, Q_BASE, K_LIN, is_tri)

            hf = hf + (dt_sub/6.0)*(k1f + 2.0*k2f + 2.0*k3f + k4f)
            hm = hm + (dt_sub/6.0)*(k1m + 2.0*k2m + 2.0*k3m + k4m)

            # nonnegativity
            if hf < 0.0:
                hf = 0.0
            if hm < 0.0:
                hm = 0.0

            # stability checks
            # (numba supports np.isfinite)
            if (not np.isfinite(hf)) or (not np.isfinite(hm)) or (hf > H_MAX) or (hm > H_MAX):
                ok = False
                break

            t += dt_sub

        if not ok:
            # fill remainder with nan to make downstream rejection easy
            for j in range(i, n_steps):
                Hf[j] = np.nan
                Hm[j] = np.nan
            break

    return Hf, Hm, ok



@njit(cache=True, fastmath=True)
def rk4_integrate_arrays_adaptive(K, alpha, Sy_f, Sy_m, P_mag, P_dur, t0_storm, peak_frac,
                                  Q_BASE, K_LIN, is_tri,
                                  dt, n_steps,
                                  substep_target, nsub_max,
                                  Hf0=0.1, Hm0=0.1,
                                  H_MAX=1e6):
    """
    Fixed output grid RK4 with *adaptive* substepping per output interval.

    Key idea:
      - Explicit RK4 stability/accuracy is controlled by an estimate of the local Jacobian magnitude.
      - The dual-porosity system is stiff not only from exchange (alpha) but also from nonlinear outflow:
            dQout/dH ~ (K_LIN + 2*K*H)
      - We therefore choose n_sub each output interval so that:
            (rate_est * dt_sub) ~ O(substep_target)
        where rate_est combines exchange and outflow contributions.

    Returns (Hf, Hm, ok_flag, max_n_sub_used)
    """
    Hf = np.empty(n_steps, dtype=np.float64)
    Hm = np.empty(n_steps, dtype=np.float64)

    hf = Hf0
    hm = Hm0
    t = 0.0

    ok = True
    max_n_sub = 1

    # guard controls
    if substep_target <= 0.0:
        substep_target = 1.0
    if nsub_max < 1:
        nsub_max = 1
    if nsub_max > 20000:
        nsub_max = 20000  # safety hard cap

    # precompute exchange stiffness term (constant in time)
    # Jacobian magnitude contribution from exchange ~ alpha*(1/Sy_f + 1/Sy_m)
    rate_ex = alpha * (1.0 / Sy_f + 1.0 / Sy_m)

    for i in range(n_steps):
        Hf[i] = hf
        Hm[i] = hm

        if i == n_steps - 1:
            break

        # Estimate local precipitation within this output interval to anticipate storm-rise stiffness.
        # Use max(P(t), P(t+dt/2), P(t+dt)).
        if is_tri:
            p0 = _precip_scalar_tri_jit(t, P_mag, P_dur, t0_storm, peak_frac)
            p1 = _precip_scalar_tri_jit(t + 0.5*dt, P_mag, P_dur, t0_storm, peak_frac)
            p2 = _precip_scalar_tri_jit(t + dt, P_mag, P_dur, t0_storm, peak_frac)
        else:
            p0 = _precip_scalar_box_jit(t, P_mag, P_dur, t0_storm)
            p1 = _precip_scalar_box_jit(t + 0.5*dt, P_mag, P_dur, t0_storm)
            p2 = _precip_scalar_box_jit(t + dt, P_mag, P_dur, t0_storm)

        Pmax = p0
        if p1 > Pmax: Pmax = p1
        if p2 > Pmax: Pmax = p2

        # Equilibrium head scale at this interval's peak forcing (for Q=K*H^2):
        #   P ≈ K H^2  =>  H_eq ≈ sqrt(P/K).
        H_eq = 0.0
        if Pmax > 0.0 and K > 0.0:
            H_eq = np.sqrt(Pmax / (K + 1e-30))

        H_est = hf
        if H_eq > H_est:
            H_est = H_eq
        if H_est < 0.0:
            H_est = 0.0

        # Outflow Jacobian contribution on Hf:
        dQdH = K_LIN + 2.0 * K * H_est
        rate_out = dQdH / Sy_f

        rate = rate_ex + rate_out
        stiff = rate * dt

        n_sub = 1 + int(stiff / substep_target)
        if n_sub < 1:
            n_sub = 1
        if n_sub > nsub_max:
            n_sub = nsub_max

        # accuracy floor during storm + early recession: avoid extremely coarse resolution
        # that can bias recession-loop geometry even when the solution is numerically stable.
        if t < (t0_storm + P_dur + 2.0*dt):
            if n_sub < MIN_NSUB_EVENT:
                n_sub = MIN_NSUB_EVENT
                if n_sub > nsub_max:
                    n_sub = nsub_max

        if n_sub > max_n_sub:
            max_n_sub = n_sub

        dt_sub = dt / n_sub

        for _ in range(n_sub):
            k1f, k1m = _rhs_jit(t, hf, hm, K, alpha, Sy_f, Sy_m, P_mag, P_dur,
                                t0_storm, peak_frac, Q_BASE, K_LIN, is_tri)

            k2f, k2m = _rhs_jit(t + 0.5*dt_sub,
                                hf + 0.5*dt_sub*k1f, hm + 0.5*dt_sub*k1m,
                                K, alpha, Sy_f, Sy_m, P_mag, P_dur,
                                t0_storm, peak_frac, Q_BASE, K_LIN, is_tri)

            k3f, k3m = _rhs_jit(t + 0.5*dt_sub,
                                hf + 0.5*dt_sub*k2f, hm + 0.5*dt_sub*k2m,
                                K, alpha, Sy_f, Sy_m, P_mag, P_dur,
                                t0_storm, peak_frac, Q_BASE, K_LIN, is_tri)

            k4f, k4m = _rhs_jit(t + dt_sub,
                                hf + dt_sub*k3f, hm + dt_sub*k3m,
                                K, alpha, Sy_f, Sy_m, P_mag, P_dur,
                                t0_storm, peak_frac, Q_BASE, K_LIN, is_tri)

            hf = hf + (dt_sub/6.0)*(k1f + 2.0*k2f + 2.0*k3f + k4f)
            hm = hm + (dt_sub/6.0)*(k1m + 2.0*k2m + 2.0*k3m + k4m)

            if hf < 0.0:
                hf = 0.0
            if hm < 0.0:
                hm = 0.0

            if (not np.isfinite(hf)) or (not np.isfinite(hm)) or (hf > H_MAX) or (hm > H_MAX):
                ok = False
                break

            t += dt_sub

        if not ok:
            for j in range(i, n_steps):
                Hf[j] = np.nan
                Hm[j] = np.nan
            break

    return Hf, Hm, ok, max_n_sub


@njit(cache=True, fastmath=True)
def rk4_integrate_arrays_adaptive_exchange_exact(K, alpha, Sy_f, Sy_m, P_mag, P_dur, t0_storm, peak_frac,
                                                 Q_BASE, K_LIN, is_tri,
                                                 dt, n_steps,
                                                 substep_target, nsub_max,
                                                 Hf0=0.1, Hm0=0.1,
                                                 H_MAX=1e6,
                                                 use_strang=1):
    """
    Fixed output grid RK4 with *adaptive* substepping, but with stiff exchange handled by an
    exact update (operator splitting). This removes alpha-driven stiffness from the explicit RK4.

    Splitting options:
      - use_strang=1: Strang splitting (half exchange, full forcing/outflow RK4, half exchange)
      - use_strang=0: Lie splitting (forcing/outflow RK4 then full exchange)

    Substepping is chosen using ONLY the nonlinear outflow stiffness estimate:
        rate_out ~ (dQout/dH) / Sy_f
    since exchange is integrated exactly.

    Returns (Hf, Hm, ok_flag, max_n_sub_used)
    """
    Hf = np.empty(n_steps, dtype=np.float64)
    Hm = np.empty(n_steps, dtype=np.float64)

    hf = Hf0
    hm = Hm0
    t = 0.0

    ok = True
    max_n_sub = 1

    if substep_target <= 0.0:
        substep_target = 1.0
    if nsub_max < 1:
        nsub_max = 1
    if nsub_max > 20000:
        nsub_max = 20000

    # exchange rate scale (used only to control splitting accuracy when exchange is handled exactly)
    rate_ex = alpha * (1.0 / Sy_f + 1.0 / Sy_m)

    for i in range(n_steps):
        Hf[i] = hf
        Hm[i] = hm

        if i == n_steps - 1:
            break

        # integrate this output interval, but split exactly at storm breakpoints (triangle kink times)
        t_target = t + dt
        t_peak = t0_storm + peak_frac * P_dur
        t_endstorm = t0_storm + P_dur
        
        while t < (t_target - 1e-15):
            seg_end = t_target
            if (t < t_peak) and (t_peak < seg_end):
                seg_end = t_peak
            if (t < t_endstorm) and (t_endstorm < seg_end):
                seg_end = t_endstorm
            seg_dt = seg_end - t
            if seg_dt <= 0.0:
                t = seg_end
                continue
            # If we are extremely close to the segment end (floating roundoff), snap to seg_end
            if seg_dt < 1e-12:
                t = seg_end
                continue
        
            # anticipate stiffness within this sub-interval (seg_dt)
            if is_tri:
                p0 = _precip_scalar_tri_jit(t, P_mag, P_dur, t0_storm, peak_frac)
                p1 = _precip_scalar_tri_jit(t + 0.5*seg_dt, P_mag, P_dur, t0_storm, peak_frac)
                p2 = _precip_scalar_tri_jit(t + seg_dt, P_mag, P_dur, t0_storm, peak_frac)
            else:
                p0 = _precip_scalar_box_jit(t, P_mag, P_dur, t0_storm)
                p1 = _precip_scalar_box_jit(t + 0.5*seg_dt, P_mag, P_dur, t0_storm)
                p2 = _precip_scalar_box_jit(t + seg_dt, P_mag, P_dur, t0_storm)
        
            Pmax = p0
            if p1 > Pmax: Pmax = p1
            if p2 > Pmax: Pmax = p2
        
            # head scale to estimate outflow Jacobian
            H_eq = 0.0
            if Pmax > 0.0 and K > 0.0:
                H_eq = np.sqrt(Pmax / (K + 1e-30))
        
            H_est = hf
            if H_eq > H_est:
                H_est = H_eq
            if H_est < 0.0:
                H_est = 0.0
        
            dQdH = K_LIN + 2.0 * K * H_est
            rate_out = dQdH / Sy_f
        
            rate_eff = rate_out + EXACT_EXCHANGE_SUBSTEP_FACTOR * rate_ex
            stiff = rate_eff * seg_dt
            n_sub = 1 + int(stiff / substep_target)
            if n_sub < 1:
                n_sub = 1
            if n_sub > nsub_max:
                n_sub = nsub_max
        
            # accuracy floor during storm + early recession (prevents coarse resolution around kinks)
            if t < (t0_storm + P_dur + 2.0*dt):
                if n_sub < MIN_NSUB_EVENT:
                    n_sub = MIN_NSUB_EVENT
                    if n_sub > nsub_max:
                        n_sub = nsub_max
        
            if n_sub > max_n_sub:
                max_n_sub = n_sub
        
            dt_sub = seg_dt / n_sub
            # Guard: if dt_sub is too small to advance time at the current magnitude, snap to seg_end
            if (t + dt_sub) == t:
                t = seg_end
                continue

            tt = t
            for _ in range(n_sub):
                if use_strang == 1:
                    hf, hm = exchange_exact_step_jit(hf, hm, alpha, Sy_f, Sy_m, 0.5*dt_sub)

                # forcing/outflow RK4 (exchange excluded)
                k1f = _rhs_noex_jit(tt, hf, hm, K, Sy_f, P_mag, P_dur, t0_storm, peak_frac, Q_BASE, K_LIN, is_tri)
                k2f = _rhs_noex_jit(tt + 0.5*dt_sub, hf + 0.5*dt_sub*k1f, hm, K, Sy_f, P_mag, P_dur, t0_storm, peak_frac, Q_BASE, K_LIN, is_tri)
                k3f = _rhs_noex_jit(tt + 0.5*dt_sub, hf + 0.5*dt_sub*k2f, hm, K, Sy_f, P_mag, P_dur, t0_storm, peak_frac, Q_BASE, K_LIN, is_tri)
                k4f = _rhs_noex_jit(tt + dt_sub, hf + dt_sub*k3f, hm, K, Sy_f, P_mag, P_dur, t0_storm, peak_frac, Q_BASE, K_LIN, is_tri)

                hf = hf + (dt_sub/6.0) * (k1f + 2.0*k2f + 2.0*k3f + k4f)
                if hf < 0.0:
                    hf = 0.0

                if use_strang == 1:
                    hf, hm = exchange_exact_step_jit(hf, hm, alpha, Sy_f, Sy_m, 0.5*dt_sub)
                else:
                    hf, hm = exchange_exact_step_jit(hf, hm, alpha, Sy_f, Sy_m, dt_sub)

                if (not np.isfinite(hf)) or (not np.isfinite(hm)) or (hf > H_MAX) or (hm > H_MAX):
                    ok = False
                    break

                tt = tt + dt_sub

            # Snap the time to the exact segment end to prevent roundoff from creating micro-segments
            t = seg_end

            if not ok:
                break
        if not ok:
            for j in range(i, n_steps):
                Hf[j] = np.nan
                Hm[j] = np.nan
            break

    return Hf, Hm, ok, max_n_sub




# @njit(cache=True, fastmath=True)
# def rk4_integrate_arrays(K, alpha, Sy_f, Sy_m, P_mag, P_dur, t0_storm, peak_frac,
#                          Q_BASE, K_LIN, is_tri, dt, n_steps, Hf0=0.1, Hm0=0.1):
#     """
#     Fixed-step RK4 over uniform dt, returns arrays Hf,Hm (length n_steps).
#     """
#     Hf = np.empty(n_steps, dtype=np.float64)
#     Hm = np.empty(n_steps, dtype=np.float64)

#     hf = Hf0
#     hm = Hm0
#     t = 0.0

#     for i in range(n_steps):
#         Hf[i] = hf
#         Hm[i] = hm

#         # RK4 stages
#         k1f, k1m = _rhs_jit(t, hf, hm, K, alpha, Sy_f, Sy_m, P_mag, P_dur, t0_storm, peak_frac, Q_BASE, K_LIN, is_tri)
#         k2f, k2m = _rhs_jit(t + 0.5*dt, hf + 0.5*dt*k1f, hm + 0.5*dt*k1m, K, alpha, Sy_f, Sy_m, P_mag, P_dur, t0_storm, peak_frac, Q_BASE, K_LIN, is_tri)
#         k3f, k3m = _rhs_jit(t + 0.5*dt, hf + 0.5*dt*k2f, hm + 0.5*dt*k2m, K, alpha, Sy_f, Sy_m, P_mag, P_dur, t0_storm, peak_frac, Q_BASE, K_LIN, is_tri)
#         k4f, k4m = _rhs_jit(t + dt, hf + dt*k3f, hm + dt*k3m, K, alpha, Sy_f, Sy_m, P_mag, P_dur, t0_storm, peak_frac, Q_BASE, K_LIN, is_tri)

#         hf = hf + (dt/6.0)*(k1f + 2.0*k2f + 2.0*k3f + k4f)
#         hm = hm + (dt/6.0)*(k1m + 2.0*k2m + 2.0*k3m + k4m)

#         # clamp small negatives
#         if hf < 0.0: hf = 0.0
#         if hm < 0.0: hm = 0.0

#         t += dt

#     return Hf, Hm

# JIT-compiled triangular hyetograph (most common case)
@njit(cache=True, fastmath=True)
def _precip_scalar_tri_jit(t: float, P_mag: float, P_dur: float, t0: float,
                           peak_frac: float) -> float:
    """JIT-compiled triangular precipitation intensity."""
    tp = t0 + peak_frac * P_dur
    t1 = t0 + P_dur
    P_peak = 2.0 * P_mag

    if t <= t0 or t >= t1:
        return 0.0
    if t <= tp:
        return P_peak * (t - t0) / (tp - t0 + 1e-12)
    else:
        return P_peak * (t1 - t) / (t1 - tp + 1e-12)


@njit(cache=True, fastmath=True)
def _precip_scalar_box_jit(t: float, P_mag: float, P_dur: float, t0: float) -> float:
    """JIT-compiled box precipitation intensity."""
    if t0 < t < (t0 + P_dur):
        return P_mag
    return 0.0


def _precip_scalar_gamma(t: float, P_mag: float, P_dur: float, t0: float,
                          gamma_k: float, gamma_norm: float) -> float:
    """Gamma-shaped precipitation intensity (non-JIT, for solve_ivp path).

    P(t) = gamma_norm * ((t-t0)/tau)^(k-1) * exp(-(t-t0)/tau)
    where tau = P_dur * (k-1)/k (mode at peak_frac = (k-1)/(k) of duration)
    and gamma_norm is chosen so that integral over [t0, t0+P_dur] = P_mag * P_dur.
    """
    if t <= t0 or t >= (t0 + P_dur):
        return 0.0
    s = (t - t0)
    tau = P_dur * max((gamma_k - 1.0), 0.1) / max(gamma_k, 1.0)
    if tau <= 0:
        tau = P_dur * 0.1
    x = s / tau
    # Gamma PDF shape: x^(k-1) * exp(-x) / tau
    import math
    val = gamma_norm * (x ** (gamma_k - 1.0)) * math.exp(-x) / tau
    return float(val)


def _precompute_gamma_norm(P_mag: float, P_dur: float, gamma_k: float, n_quad: int = 500) -> float:
    """Precompute normalization constant for gamma hyetograph so integral = P_mag * P_dur."""
    import math
    tau = P_dur * max((gamma_k - 1.0), 0.1) / max(gamma_k, 1.0)
    if tau <= 0:
        tau = P_dur * 0.1
    # Simpson's rule on [0, P_dur]
    ds = P_dur / n_quad
    integral = 0.0
    for i in range(n_quad + 1):
        s = i * ds
        x = s / tau
        f = (x ** (gamma_k - 1.0)) * math.exp(-x) / tau
        w = 1.0 if (i == 0 or i == n_quad) else (4.0 if i % 2 == 1 else 2.0)
        integral += w * f
    integral *= ds / 3.0
    target = P_mag * P_dur
    return target / max(integral, 1e-30)


# JIT-compiled ODE RHS (the performance-critical bottleneck)
@njit(cache=True, fastmath=True)
def _universal_model_jit(t: float, Hf: float, Hm: float, K: float, alpha: float,
                         Sy_f: float, Sy_m: float, P_mag: float, P_dur: float,
                         t0_storm: float, peak_frac: float,
                         Q_BASE: float, K_LIN: float, hyeto_is_tri: bool) -> tuple:
    """JIT-compiled ODE RHS for dual-porosity system."""
    # Enforce non-negativity
    if Hf < 0.0:
        Hf = 0.0
    if Hm < 0.0:
        Hm = 0.0

    # Precipitation (inline for speed)
    if hyeto_is_tri:
        P = _precip_scalar_tri_jit(t, P_mag, P_dur, t0_storm, peak_frac)
    else:
        P = _precip_scalar_box_jit(t, P_mag, P_dur, t0_storm)

    # Fluxes
    Q_out = Q_BASE + K_LIN * Hf + K * Hf * Hf
    Q_ex = alpha * (Hf - Hm)

    # State derivatives
    dHf = (P - Q_out - Q_ex) / Sy_f
    dHm = Q_ex / Sy_m

    # Prevent negative states
    if Hf <= 0.0 and dHf < 0.0:
        dHf = 0.0
    if Hm <= 0.0 and dHm < 0.0:
        dHm = 0.0

    return (dHf, dHm)


def precip_vec(t: np.ndarray, P_mag: float, P_dur: float, t0: float = 1.0) -> np.ndarray:
    """Vectorized precipitation intensity (box, triangular, or gamma)."""
    t = np.asarray(t, float)
    P = np.zeros_like(t)

    if HYETO == "box":
        P[(t > t0) & (t < (t0 + P_dur))] = P_mag

    elif HYETO == "tri":
        tp, t1 = t0 + TRI_PEAK_FRAC * P_dur, t0 + P_dur
        P_peak = 2.0 * P_mag
        m1 = (t > t0) & (t <= tp)
        m2 = (t > tp) & (t < t1)
        P[m1] = P_peak * (t[m1] - t0) / max(tp - t0, 1e-12)
        P[m2] = P_peak * (t1 - t[m2]) / max(t1 - tp, 1e-12)

    elif HYETO == "gamma":
        tau = t - t0
        m = (tau > 0) & (tau < P_dur)
        tau_m = tau[m]
        scale = P_dur / max(GAMMA_SHAPE, 1e-6)
        kern = (tau_m + GAMMA_EPS) ** (GAMMA_SHAPE - 1) * np.exp(-(tau_m + GAMMA_EPS) / scale)
        A = (P_mag * P_dur) / (np.trapz(kern, tau_m) + 1e-30)
        P[m] = A * kern
    else:
        raise ValueError(f"Unknown HYETO={HYETO}")

    return P


_GAMMA_NORM_CACHE: dict[tuple, float] = {}

def precip_scalar(t: float, P_mag: float, P_dur: float, t0: float = 1.0,
                  hyeto_override: str = None, gamma_k_override: float = None) -> float:
    """Scalar precipitation for ODE RHS (uses JIT when available)."""
    hyeto = hyeto_override or HYETO
    if hyeto == "tri":
        return _precip_scalar_tri_jit(t, P_mag, P_dur, t0, TRI_PEAK_FRAC)
    elif hyeto == "box":
        return _precip_scalar_box_jit(t, P_mag, P_dur, t0)
    elif hyeto == "gamma":
        gk = gamma_k_override if gamma_k_override is not None else GAMMA_SHAPE
        cache_key = (P_mag, P_dur, gk)
        if cache_key not in _GAMMA_NORM_CACHE:
            _GAMMA_NORM_CACHE[cache_key] = _precompute_gamma_norm(P_mag, P_dur, gk)
        return _precip_scalar_gamma(t, P_mag, P_dur, t0, gk, _GAMMA_NORM_CACHE[cache_key])
    else:
        return float(precip_vec(np.array([t]), P_mag, P_dur, t0=t0)[0])


def universal_model(t, y, K, alpha, Sy_f, Sy_m, P_mag, P_dur):
    """
    ODE RHS wrapper (kept for fallback/debug).
    Correctly distinguishes tri vs box.
    """
    is_tri = (HYETO == "tri")
    if HYETO == "gamma":
        # Use non-JIT gamma path
        return universal_model_configurable(t, y, K, alpha, Sy_f, Sy_m, P_mag, P_dur,
                                             hyeto="gamma", gamma_k=GAMMA_SHAPE)
    return _universal_model_jit(
        t, y[0], y[1], K, alpha, Sy_f, Sy_m, P_mag, P_dur,
        t0_storm, TRI_PEAK_FRAC, Q_BASE, K_LIN, is_tri
    )


def universal_model_configurable(t, y, K, alpha, Sy_f, Sy_m, P_mag, P_dur,
                                  hyeto: str = "tri", peak_frac: float = None,
                                  gamma_k: float = None):
    """Configurable ODE RHS for solve_ivp with arbitrary hyetograph.

    Supports tri, box, and gamma without modifying global HYETO.
    Used by forcing-admissibility sweeps.
    """
    Hf = max(y[0], 0.0)
    Hm = max(y[1], 0.0)

    pf = peak_frac if peak_frac is not None else TRI_PEAK_FRAC
    P = precip_scalar(t, P_mag, P_dur, t0_storm,
                      hyeto_override=hyeto, gamma_k_override=gamma_k)

    Q_out = Q_BASE + K_LIN * Hf + K * Hf * Hf
    Q_ex = alpha * (Hf - Hm)

    dHf = (P - Q_out - Q_ex) / Sy_f
    dHm = Q_ex / Sy_m

    if Hf <= 0.0 and dHf < 0.0:
        dHf = 0.0
    if Hm <= 0.0 and dHm < 0.0:
        dHm = 0.0

    return (dHf, dHm)



# ==============================================================================
# HYSTERESIS & DIAGNOSTICS
# ==============================================================================

def shoelace_area_loglog(x: np.ndarray, y: np.ndarray) -> float:
    """Shoelace area on log10(x), log10(y). Expects x>0, y>0."""
    lx, ly = np.log10(x), np.log10(y)
    return 0.5 * np.abs(np.dot(lx, np.roll(ly, 1)) - np.dot(ly, np.roll(lx, 1)))


def hysteresis_index_Qspace(t, Hf, Hm, K, alpha, Sy_f, Sy_m, P_vec, t_storm_end, min_pts=30):
    """Compute normalized loop area for Q_out vs (-dQ_out/dt).

    Guard strategy:
      - Use a post-storm window (t > t_storm_end) and require Q_out>0, rQ>0 for log10 safety.
      - Truncate only the ultra-late tail at a small fraction of peak signal (Q_CUTOFF_FRAC, RQ_CUTOFF_FRAC).
      - Floor the log-span normalization by SPAN_MIN_DECADES so near-degenerate loops cannot explode.
      - Clip physically impossible normalized areas (HI > HI_CLIP_MAX) to NaN (dropped upstream).
    """
    Hf, Hm = np.maximum(Hf, 0.0), np.maximum(Hm, 0.0)

    Q_out = Q_BASE + K_LIN * Hf + K * Hf * Hf
    Q_ex = alpha * (Hf - Hm)
    dHf = (P_vec - Q_out - Q_ex) / Sy_f
    rQ = (K_LIN + 2.0 * K * Hf) * (-dHf)

    # Recession window (post-storm), keep only positive signals for log-space area
    Qmax, rQmax = float(np.max(Q_out)), float(np.max(rQ))
    epsQ = max(Qmax * Q_CUTOFF_FRAC, 1e-18)
    epsR = max(rQmax * RQ_CUTOFF_FRAC, 1e-18)
    m = (t > t_storm_end) & (Q_out > epsQ) & (rQ > epsR)

    if np.count_nonzero(m) < min_pts:
        return np.nan, 0.0, np.nan, np.nan, Q_out, rQ

    x, y = Q_out[m], rQ[m]
    raw = shoelace_area_loglog(x, y)

    x_span = float(np.ptp(np.log10(x)))
    y_span = float(np.ptp(np.log10(y)))

    # Floor spans to prevent normalization blow-up in near-degenerate cases
    x_eff = max(x_span, SPAN_MIN_DECADES)
    y_eff = max(y_span, SPAN_MIN_DECADES)

    HI = raw / (x_eff * y_eff)

    if (not np.isfinite(HI)) or (HI < 0.0) or (HI > HI_CLIP_MAX * (1.0 + 1e-6)):
        return np.nan, float(raw), x_span, y_span, Q_out, rQ

    return float(HI), float(raw), x_span, y_span, Q_out, rQ



def mechanism_diagnostics(t, Hf, Hm, K, alpha, Sy_f, Sy_m, P_vec, t_storm_end, theta=THETA_DH):
    """Compute per-run mechanism diagnostics (post-storm)."""
    Hf, Hm = np.maximum(Hf, 0.0), np.maximum(Hm, 0.0)

    Q_out = Q_BASE + K_LIN * Hf + K * Hf * Hf
    dH = Hf - Hm
    Q_ex = alpha * dH
    dHf = (P_vec - Q_out - Q_ex) / Sy_f
    rQ = (K_LIN + 2.0 * K * Hf) * (-dHf)

    m = (t > t_storm_end) if POST_ONLY else np.ones_like(t, dtype=bool)
    tt = t[m]
    if tt.size < 5:
        return {}

    dH_m, Qout_m, Qex_m = dH[m], Q_out[m], Q_ex[m]

    # Integrals
    V_out = float(np.trapz(Qout_m, tt))
    V_ex_pos = float(np.trapz(np.maximum(Qex_m, 0.0), tt))
    V_ex_neg = float(np.trapz(np.maximum(-Qex_m, 0.0), tt))
    V_ex_abs = float(np.trapz(np.abs(Qex_m), tt))

    # Persistence
    dH_max = float(np.max(dH_m)) if dH_m.size else np.nan
    if np.isfinite(dH_max) and dH_max > 0:
        thr = theta * dH_max
        mask_persist = dH_m > thr
        T_persist = float(np.sum(mask_persist) * np.median(np.diff(tt)))
        frac_persist = float(np.mean(mask_persist))
        t_peakgrad = float(tt[np.argmax(dH_m)])
    else:
        T_persist, frac_persist, t_peakgrad, thr = 0.0, 0.0, np.nan, np.nan

    denom = V_out + V_ex_pos + 1e-30
    return {
        "dH_max_post": dH_max, "dH_thr": float(thr) if np.isfinite(thr) else np.nan,
        "T_persist": T_persist, "frac_persist": frac_persist, "t_peakgrad": t_peakgrad,
        "V_out": V_out, "V_ex_pos": V_ex_pos, "V_ex_neg": V_ex_neg, "V_ex_abs": V_ex_abs,
        "frac_ex_pos": float(V_ex_pos / denom), "frac_out": float(V_out / denom),
    }


# ==============================================================================
# PEAK DETECTION
# ==============================================================================

def _binned_curve_quantile_logx(
    df_sub,
    xcol: str = "Da",
    ycol: str = "Hysteresis",
    n_bins: int = N_BINS_PEAK,
    smooth_sigma: float = SMOOTH_SIGMA,
    q_low: float = Q_LOW,
    q_high: float = Q_HIGH,
    nmin: int = NMIN_PER_BIN,
    min_valid_bins: int = MIN_VALID_BINS_FOR_PEAK,
    smooth_sigma_decades: float = None,
    sigma_scale_with_nbins: bool = False,
):
    """Build a support-aware binned+smoothed curve in log10(x) using quantile (equal-count) bins.

    Notes on smoothing (for ridge stability):
      - If smooth_sigma_decades is provided, it is interpreted in *log10(x) decades* and is
        converted (per contiguous support segment) to a sigma in bin-index units. This reduces
        sensitivity when n_bins changes, because the smoothing acts on a consistent log-scale.
      - Else if sigma_scale_with_nbins is True, smooth_sigma is scaled by (n_bins_eff / N_BINS_PEAK),
        which approximates holding the smoothing width constant in log10(x) decades.

    Returns:
        centers (linear x), y_smooth, count, is_valid
    """
    if df_sub is None or df_sub.empty:
        return None

    x = df_sub[xcol].to_numpy()
    y = df_sub[ycol].to_numpy()
    m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y >= 0)
    x, y = x[m], y[m]

    if x.size < 100:
        return None

    lx = np.log10(x)
    lxlo, lxhi = np.quantile(lx, q_low), np.quantile(lx, q_high)
    if not (np.isfinite(lxlo) and np.isfinite(lxhi) and lxhi > lxlo):
        return None

    mt = (lx >= lxlo) & (lx <= lxhi)
    lx, y = lx[mt], y[mt]
    if lx.size < 80:
        return None

    qs = np.linspace(0.0, 1.0, int(n_bins) + 1)
    edges = np.quantile(lx, qs)
    edges = np.unique(edges)

    n_bins_eff = int(edges.size - 1)
    if n_bins_eff < int(min_valid_bins):
        return None

    centers = 10.0 ** (0.5 * (edges[:-1] + edges[1:]))
    b = np.clip(np.digitize(lx, edges) - 1, 0, n_bins_eff - 1)

    count = np.bincount(b, minlength=n_bins_eff).astype(int)
    y_sum = np.bincount(b, weights=y, minlength=n_bins_eff)

    y_mean = np.full(n_bins_eff, np.nan)
    has = count > 0
    y_mean[has] = y_sum[has] / count[has]

    is_valid = (count >= int(nmin)) & np.isfinite(y_mean)
    idx_valid = np.where(is_valid)[0]
    if idx_valid.size < int(min_valid_bins):
        return None

    # Count-weighted smoothing, separately on each contiguous support segment.
    y_smooth = np.full(n_bins_eff, np.nan)

    breaks = np.where(np.diff(idx_valid) > 1)[0]
    seg_starts = np.r_[0, breaks + 1]
    seg_ends = np.r_[breaks, idx_valid.size - 1]

    for s0, s1 in zip(seg_starts, seg_ends):
        seg_idx = idx_valid[s0:s1 + 1]
        ys = y_mean[seg_idx].astype(float)
        ws = count[seg_idx].astype(float)

        if seg_idx.size >= 4:
            sigma_bins = float(smooth_sigma)

            # Option A: explicitly specify smoothing width in log10(x) decades.
            if smooth_sigma_decades is not None:
                try:
                    lx_seg = np.log10(np.maximum(centers[seg_idx], 1e-300))
                    if lx_seg.size >= 2:
                        dlog = float(np.nanmedian(np.diff(lx_seg)))
                        if np.isfinite(dlog) and (dlog > 0):
                            sigma_bins = float(smooth_sigma_decades) / dlog
                except Exception:
                    sigma_bins = float(smooth_sigma)

            # Option B: heuristic scaling with effective number of bins.
            elif bool(sigma_scale_with_nbins):
                try:
                    sigma_bins = float(smooth_sigma) * (float(n_bins_eff) / float(N_BINS_PEAK))
                except Exception:
                    sigma_bins = float(smooth_sigma)

            # Guardrails
            if (not np.isfinite(sigma_bins)) or (sigma_bins <= 0):
                y_smooth[seg_idx] = ys
            else:
                num = gaussian_filter1d(ys * ws, sigma=sigma_bins)
                den = gaussian_filter1d(ws, sigma=sigma_bins)
                y_smooth[seg_idx] = num / np.maximum(den, 1e-12)
        else:
            y_smooth[seg_idx] = ys

    return centers, y_smooth, count, is_valid



# ------------------------------------------------------------------------------
# Peak extraction helper: refined peak/centroid/width on a fine log10(x) grid
# ------------------------------------------------------------------------------

def _refine_segment_from_curve(
    centers: np.ndarray,
    y_smooth: np.ndarray,
    valid_mask: np.ndarray,
    k_peak: int,
    n_fine: int = PEAK_REFINE_N_FINE,
    smooth_decades: float = PEAK_REFINE_SMOOTH_DECADES,
):
    """Resample the contiguous valid segment containing k_peak onto a fine log10(x) grid.

    Returns:
        lx_f (log10 x), y_f (smoothed y)  OR (None, None) if refinement is not possible.
    """
    try:
        centers = np.asarray(centers, float)
        y_smooth = np.asarray(y_smooth, float)
        valid_mask = np.asarray(valid_mask, bool)
        n = int(centers.size)
        k = int(k_peak)
        if n < 4 or (k < 0) or (k >= n):
            return None, None

        # contiguous segment around coarse peak
        left = k
        while left > 0 and bool(valid_mask[left - 1]):
            left -= 1
        right = k
        while right < n - 1 and bool(valid_mask[right + 1]):
            right += 1

        x_seg = centers[left:right + 1]
        y_seg = y_smooth[left:right + 1]

        m = np.isfinite(x_seg) & (x_seg > 0) & np.isfinite(y_seg)
        x_seg = x_seg[m]
        y_seg = y_seg[m]
        if x_seg.size < 4:
            return None, None

        lx_seg = np.log10(x_seg)
        order = np.argsort(lx_seg)
        lx_seg = lx_seg[order]
        y_seg = y_seg[order]

        if (not np.isfinite(lx_seg[0])) or (not np.isfinite(lx_seg[-1])) or (lx_seg[-1] <= lx_seg[0]):
            return None, None

        n_fine = int(max(50, n_fine))
        lx_f = np.linspace(float(lx_seg[0]), float(lx_seg[-1]), n_fine)
        y_f = np.interp(lx_f, lx_seg, y_seg)

        # Optional very-light smoothing in *decades* (log10-units), applied on the fine grid
        if smooth_decades is not None:
            sd = float(smooth_decades)
            if sd > 0:
                dlog = float((lx_f[-1] - lx_f[0]) / max(1, (n_fine - 1)))
                if np.isfinite(dlog) and (dlog > 0):
                    sigma_pts = sd / dlog
                    if np.isfinite(sigma_pts) and (sigma_pts > 0):
                        y_f = gaussian_filter1d(y_f, sigma=float(sigma_pts))

        return lx_f, y_f
    except Exception:
        return None, None


def _band_edges_logx(lx: np.ndarray, y: np.ndarray, k: int, thr: float, interp: bool = True):
    """Find contiguous above-threshold band edges around k, optionally interpolating crossings."""
    lx = np.asarray(lx, float)
    y = np.asarray(y, float)
    n = int(lx.size)
    if n == 0:
        return np.nan, np.nan, 0, 0

    k = int(np.clip(int(k), 0, n - 1))

    # Left edge
    left = k
    while left > 0 and (y[left - 1] >= thr):
        left -= 1
    lx_left = float(lx[left])
    if interp and (left > 0):
        y0 = float(y[left - 1])
        y1 = float(y[left])
        if np.isfinite(y0) and np.isfinite(y1) and (y0 < thr) and (y1 >= thr) and (y1 != y0):
            t = float((thr - y0) / (y1 - y0))
            t = float(np.clip(t, 0.0, 1.0))
            lx_left = float(lx[left - 1] + t * (lx[left] - lx[left - 1]))

    # Right edge
    right = k
    while right < n - 1 and (y[right + 1] >= thr):
        right += 1
    lx_right = float(lx[right])
    if interp and (right < n - 1):
        y0 = float(y[right])
        y1 = float(y[right + 1])
        if np.isfinite(y0) and np.isfinite(y1) and (y0 >= thr) and (y1 < thr) and (y1 != y0):
            t = float((thr - y0) / (y1 - y0))
            t = float(np.clip(t, 0.0, 1.0))
            lx_right = float(lx[right] + t * (lx[right + 1] - lx[right]))

    return lx_left, lx_right, int(left), int(right)


def _width_metrics_logx(lx: np.ndarray, y: np.ndarray, frac: float, interp: bool = True):
    """Width metrics in log10(x) decades at a given fraction of the peak."""
    lx = np.asarray(lx, float)
    y = np.asarray(y, float)
    if lx.size == 0:
        return np.nan, np.nan, np.nan, (np.nan, np.nan)

    k = int(np.nanargmax(y))
    ypk = float(y[k])
    if (not np.isfinite(ypk)) or (ypk <= 0):
        return np.nan, np.nan, np.nan, (np.nan, np.nan)

    thr = float(frac) * ypk
    lx_left, lx_right, _, _ = _band_edges_logx(lx, y, k, thr, interp=interp)
    if (not np.isfinite(lx_left)) or (not np.isfinite(lx_right)):
        return np.nan, np.nan, np.nan, (lx_left, lx_right)

    lx_pk = float(lx[k])
    width_dec = max(0.0, lx_right - lx_left)
    width_left = max(0.0, lx_pk - lx_left)
    width_right = max(0.0, lx_right - lx_pk)
    return float(width_dec), float(width_left), float(width_right), (float(lx_left), float(lx_right))


def _centroid_logx(lx: np.ndarray, y: np.ndarray, frac: float, weight_mode: str = None):
    """Top-band centroid in log10(x) around the peak, on arrays already in log10(x)."""
    if weight_mode is None:
        weight_mode = PEAK_CENTROID_WEIGHT

    lx = np.asarray(lx, float)
    y = np.asarray(y, float)
    if lx.size == 0:
        return np.nan, np.nan

    k = int(np.nanargmax(y))
    ypk = float(y[k])
    if (not np.isfinite(ypk)) or (ypk <= 0):
        return np.nan, np.nan

    thr = float(frac) * ypk

    # band indices (contiguous around peak)
    left = k
    while left > 0 and (y[left - 1] >= thr):
        left -= 1
    right = k
    n = int(lx.size)
    while right < n - 1 and (y[right + 1] >= thr):
        right += 1

    band = np.arange(int(left), int(right) + 1)
    band = band[np.isfinite(y[band]) & (y[band] >= thr)]
    if band.size == 0:
        return np.nan, np.nan

    if str(weight_mode).lower().startswith("raw"):
        w = np.asarray(y[band], float)
    else:
        w = np.asarray(y[band] - thr, float)

    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    sw = float(np.sum(w))
    if sw <= 0:
        return np.nan, np.nan

    lx_c = float(np.sum(lx[band] * w) / sw)
    return float(10.0 ** lx_c), float(lx_c)


def _peak_metrics_from_binned_curve(
    centers: np.ndarray,
    y_smooth: np.ndarray,
    is_valid: np.ndarray,
    width_fracs = PEAK_WIDTH_FRACS,
    centroid_fracs = PEAK_CENTROID_FRACS,
    frac_main: float = PEAK_WIDTH_FRAC,
    interp: bool = PEAK_WIDTH_INTERP,
):
    """Compute peak/width/centroid metrics from a binned+smoothed curve, with optional refinement."""
    centers = np.asarray(centers, float)
    y_smooth = np.asarray(y_smooth, float)
    is_valid = np.asarray(is_valid, bool)

    valid = is_valid & np.isfinite(y_smooth) & (centers > 0)
    idx = np.where(valid)[0]
    if idx.size == 0:
        return {
            "x_peak": np.nan, "y_peak": np.nan,
            "hit_left": True, "hit_right": True,
            "width_decades": np.nan, "width_left_decades": np.nan, "width_right_decades": np.nan,
            "extra_widths": {}, "extra_centroids": {},
            "refine_used": False,
            "x_peak_bin": np.nan, "y_peak_bin": np.nan,
        }

    k = int(idx[np.argmax(y_smooth[idx])])
    x_peak_bin = float(centers[k])
    y_peak_bin = float(y_smooth[k])

    hit_left = bool(k <= idx[0] + 1)
    hit_right = bool(k >= idx[-1] - 1)

    # Defaults (fallback to coarse-bin metrics)
    x_peak = x_peak_bin
    y_peak = y_peak_bin

    extra_widths = {}
    extra_centroids = {}

    # Optionally refine on a fine log10 grid to reduce bin-edge sensitivity.
    refine_used = False
    lx_f = None
    y_f = None
    if bool(PEAK_REFINE):
        lx_f, y_f = _refine_segment_from_curve(
            centers, y_smooth, valid, k,
            n_fine=PEAK_REFINE_N_FINE,
            smooth_decades=PEAK_REFINE_SMOOTH_DECADES,
        )
        if (lx_f is not None) and (y_f is not None) and (len(lx_f) >= 20) and np.isfinite(np.nanmax(y_f)):
            refine_used = True

    if refine_used:
        kf = int(np.nanargmax(y_f))
        x_peak = float(10.0 ** float(lx_f[kf]))
        y_peak = float(y_f[kf])

        # Main width (for quick access)
        wdec, wL, wR, _ = _width_metrics_logx(lx_f, y_f, frac=float(frac_main), interp=bool(interp))
        width_decades = wdec
        width_left_decades = wL
        width_right_decades = wR

        # Extra widths at multiple thresholds
        for f in width_fracs:
            wd, wl, wr, _ = _width_metrics_logx(lx_f, y_f, frac=float(f), interp=bool(interp))
            tag = int(round(float(f) * 100.0))
            extra_widths[f"width_decades_f{tag:03d}"] = float(wd) if np.isfinite(wd) else np.nan
            extra_widths[f"width_left_decades_f{tag:03d}"] = float(wl) if np.isfinite(wl) else np.nan
            extra_widths[f"width_right_decades_f{tag:03d}"] = float(wr) if np.isfinite(wr) else np.nan

        # Centroids on refined curve
        for f in centroid_fracs:
            xc, lxc = _centroid_logx(lx_f, y_f, frac=float(f), weight_mode=PEAK_CENTROID_WEIGHT)
            tag = int(round(float(f) * 100.0))
            extra_centroids[f"centroid_x_f{tag:03d}"] = float(xc) if np.isfinite(xc) else np.nan
            extra_centroids[f"centroid_log10x_f{tag:03d}"] = float(lxc) if np.isfinite(lxc) else np.nan

    else:
        # Coarse-bin metrics (legacy path)
        x_peak, y_peak, hit_left, hit_right, width_decades, width_left_decades, width_right_decades = _peak_and_width_from_curve(
            centers, y_smooth, is_valid, frac=float(frac_main), interp=bool(interp)
        )
        for f in width_fracs:
            try:
                _, _, _, _, wd, wl, wr = _peak_and_width_from_curve(
                    centers, y_smooth, is_valid, frac=float(f), interp=bool(interp)
                )
            except Exception:
                wd = wl = wr = np.nan
            tag = int(round(float(f) * 100.0))
            extra_widths[f"width_decades_f{tag:03d}"] = float(wd) if np.isfinite(wd) else np.nan
            extra_widths[f"width_left_decades_f{tag:03d}"] = float(wl) if np.isfinite(wl) else np.nan
            extra_widths[f"width_right_decades_f{tag:03d}"] = float(wr) if np.isfinite(wr) else np.nan

        for f in centroid_fracs:
            try:
                xc, lxc = _centroid_from_curve(centers, y_smooth, is_valid, frac=float(f), weight_mode=PEAK_CENTROID_WEIGHT)
            except Exception:
                xc, lxc = (np.nan, np.nan)
            tag = int(round(float(f) * 100.0))
            extra_centroids[f"centroid_x_f{tag:03d}"] = float(xc) if np.isfinite(xc) else np.nan
            extra_centroids[f"centroid_log10x_f{tag:03d}"] = float(lxc) if np.isfinite(lxc) else np.nan

    return {
        "x_peak": float(x_peak) if np.isfinite(x_peak) else np.nan,
        "y_peak": float(y_peak) if np.isfinite(y_peak) else np.nan,
        "hit_left": bool(hit_left),
        "hit_right": bool(hit_right),
        "width_decades": float(width_decades) if np.isfinite(width_decades) else np.nan,
        "width_left_decades": float(width_left_decades) if np.isfinite(width_left_decades) else np.nan,
        "width_right_decades": float(width_right_decades) if np.isfinite(width_right_decades) else np.nan,
        "extra_widths": extra_widths,
        "extra_centroids": extra_centroids,
        "refine_used": bool(refine_used),
        "x_peak_bin": float(x_peak_bin) if np.isfinite(x_peak_bin) else np.nan,
        "y_peak_bin": float(y_peak_bin) if np.isfinite(y_peak_bin) else np.nan,
    }


def _peak_and_width_from_curve(centers, y_smooth, is_valid, frac=PEAK_WIDTH_FRAC, interp=PEAK_WIDTH_INTERP):
    """Pick peak on a smoothed curve and compute resonance width in decades.

    Width is defined as the contiguous region around the peak where:
        y >= frac * y_max.

    The interval edges are optionally refined by linear interpolation in log10(x)
    between the first above-threshold bin and its first below-threshold neighbor.
    This avoids artificial zero-width results when the threshold falls between bins.

    Returns:
        x_peak, y_peak, hit_left, hit_right,
        width_decades, width_left_decades, width_right_decades
    """
    if centers is None or y_smooth is None or is_valid is None:
        return np.nan, np.nan, True, True, np.nan, np.nan, np.nan

    centers = np.asarray(centers, float)
    y_smooth = np.asarray(y_smooth, float)
    is_valid = np.asarray(is_valid, bool)

    valid = is_valid & np.isfinite(y_smooth) & (centers > 0)
    idx = np.where(valid)[0]
    if idx.size == 0:
        return np.nan, np.nan, True, True, np.nan, np.nan, np.nan

    k = idx[np.argmax(y_smooth[idx])]
    x_peak = float(centers[k])
    y_peak = float(y_smooth[k])

    hit_left = bool(k <= idx[0] + 1)
    hit_right = bool(k >= idx[-1] - 1)

    if (not np.isfinite(y_peak)) or (y_peak <= 0.0) or (not np.isfinite(x_peak)) or (x_peak <= 0.0):
        return np.nan, np.nan, hit_left, hit_right, np.nan, np.nan, np.nan

    thr = float(frac) * y_peak
    lx = np.log10(np.maximum(centers, 1e-300))
    lx_peak = float(lx[k])

    # walk to the left edge of the above-threshold plateau
    left = int(k)
    while left > 0 and valid[left - 1] and (y_smooth[left - 1] >= thr):
        left -= 1

    lx_left = float(lx[left])
    if interp and left > 0 and valid[left - 1]:
        y0 = float(y_smooth[left - 1])
        y1 = float(y_smooth[left])
        if np.isfinite(y0) and np.isfinite(y1) and (y0 < thr) and (y1 >= thr) and (y1 != y0):
            t = (thr - y0) / (y1 - y0)
            t = float(np.clip(t, 0.0, 1.0))
            lx_left = float(lx[left - 1] + t * (lx[left] - lx[left - 1]))

    # walk to the right edge of the above-threshold plateau
    right = int(k)
    n = int(centers.size)
    while right < n - 1 and valid[right + 1] and (y_smooth[right + 1] >= thr):
        right += 1

    lx_right = float(lx[right])
    if interp and right < n - 1 and valid[right + 1]:
        y0 = float(y_smooth[right])
        y1 = float(y_smooth[right + 1])
        if np.isfinite(y0) and np.isfinite(y1) and (y0 >= thr) and (y1 < thr) and (y1 != y0):
            t = (thr - y0) / (y1 - y0)
            t = float(np.clip(t, 0.0, 1.0))
            lx_right = float(lx[right] + t * (lx[right + 1] - lx[right]))

    width_dec = max(0.0, lx_right - lx_left)
    width_left = max(0.0, lx_peak - lx_left)
    width_right = max(0.0, lx_right - lx_peak)

    return x_peak, y_peak, hit_left, hit_right, float(width_dec), float(width_left), float(width_right)



def _centroid_from_curve(centers, y_smooth, is_valid, frac=0.95, weight_mode=None):
    """Compute a log-space centroid of the top-band around the peak.

    Band definition:
        y >= frac * y_max

    The centroid is computed in log10(x) over the *contiguous* above-threshold
    region that contains the peak (to avoid disjoint islands).

    Args:
        centers: x bin centers (must be > 0)
        y_smooth: smoothed y values
        is_valid: boolean support mask
        frac: threshold fraction of y_max
        weight_mode: "excess" (default) uses weights=(y-thr), "raw" uses weights=y

    Returns:
        x_centroid, log10x_centroid
    """
    if centers is None or y_smooth is None or is_valid is None:
        return np.nan, np.nan

    if weight_mode is None:
        weight_mode = PEAK_CENTROID_WEIGHT

    centers = np.asarray(centers, float)
    y_smooth = np.asarray(y_smooth, float)
    is_valid = np.asarray(is_valid, bool)

    valid = is_valid & np.isfinite(y_smooth) & (centers > 0)
    idx = np.where(valid)[0]
    if idx.size == 0:
        return np.nan, np.nan

    k = idx[np.argmax(y_smooth[idx])]
    y_peak = float(y_smooth[k])
    if (not np.isfinite(y_peak)) or (y_peak <= 0.0):
        return np.nan, np.nan

    thr = float(frac) * y_peak
    lx = np.log10(np.maximum(centers, 1e-300))

    # contiguous band around peak
    left = int(k)
    while left > 0 and valid[left - 1] and (y_smooth[left - 1] >= thr):
        left -= 1
    right = int(k)
    n = int(centers.size)
    while right < n - 1 and valid[right + 1] and (y_smooth[right + 1] >= thr):
        right += 1

    band = np.arange(left, right + 1)
    band = band[valid[band] & (y_smooth[band] >= thr)]
    if band.size == 0:
        return np.nan, np.nan

    if str(weight_mode).lower().startswith("raw"):
        w = np.asarray(y_smooth[band], float)
    else:
        w = np.asarray(y_smooth[band] - thr, float)

    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    sw = float(np.sum(w))
    if sw <= 0.0:
        return np.nan, np.nan

    lx_c = float(np.sum(lx[band] * w) / sw)
    return float(10.0 ** lx_c), float(lx_c)



def _valid_segments(mask: np.ndarray):
    """Yield (i0,i1) index pairs for contiguous True segments in mask."""
    n = int(mask.size)
    i = 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i + 1
        while j < n and mask[j]:
            j += 1
        yield i, j
        i = j


def _local_peaks_from_curve(centers, y_smooth, is_valid,
                           prom_frac=BIMODAL_PROM_FRAC,
                           min_sep_decades=BIMODAL_MIN_SEP_DECADES):
    """Find local maxima on each contiguous valid segment of a smoothed curve.

    Returns a list of peak dicts sorted by descending y.
    Each dict has: idx (global index), x, y, prominence.
    """
    peaks = []
    if centers is None or y_smooth is None or is_valid is None:
        return peaks

    centers = np.asarray(centers, dtype=float)
    y_smooth = np.asarray(y_smooth, dtype=float)
    is_valid = np.asarray(is_valid, dtype=bool)

    for i0, i1 in _valid_segments(is_valid):
        xs = centers[i0:i1]
        ys = y_smooth[i0:i1]
        if xs.size < 3:
            continue
        y_max = float(np.nanmax(ys))
        if not np.isfinite(y_max) or y_max <= 0.0:
            continue
        prom = float(prom_frac) * y_max
        try:
            idxs, props = find_peaks(ys, prominence=prom)
        except Exception:
            idxs = np.array([], dtype=int)
            props = {}

        for k, loc in enumerate(idxs):
            gi = i0 + int(loc)
            x = float(centers[gi])
            y = float(y_smooth[gi])
            pr = np.nan
            try:
                pr = float(props.get("prominences", [np.nan]*len(idxs))[k])
            except Exception:
                pr = np.nan
            if np.isfinite(x) and np.isfinite(y):
                peaks.append({"idx": gi, "x": x, "y": y, "prominence": pr})

    # If no local maxima found, fall back to global max over valid bins
    if len(peaks) == 0:
        try:
            vv = np.where(is_valid)[0]
            if vv.size >= 1:
                gi = int(vv[np.nanargmax(y_smooth[vv])])
                peaks.append({"idx": gi, "x": float(centers[gi]), "y": float(y_smooth[gi]), "prominence": np.nan})
        except Exception:
            pass

    # sort by height
    peaks.sort(key=lambda d: d.get("y", -np.inf), reverse=True)

    # enforce minimum separation in log10(x) decades (keep highest peaks)
    kept = []
    for p in peaks:
        ok = True
        for q in kept:
            if (np.isfinite(p["x"]) and np.isfinite(q["x"]) and (p["x"] > 0) and (q["x"] > 0)):
                if abs(np.log10(p["x"]) - np.log10(q["x"])) < float(min_sep_decades):
                    ok = False
                    break
        if ok:
            kept.append(p)
    return kept


def _classify_bimodality(peaks, secondary_frac=BIMODAL_SECONDARY_FRAC):
    """Return (is_bimodal, n_peaks, sep_decades, sec_ratio)."""
    if peaks is None or len(peaks) == 0:
        return False, 0, np.nan, np.nan
    n = len(peaks)
    if n < 2:
        return False, n, np.nan, np.nan
    p0, p1 = peaks[0], peaks[1]
    y0, y1 = float(p0["y"]), float(p1["y"])
    sec_ratio = (y1 / y0) if (np.isfinite(y0) and y0 > 0 and np.isfinite(y1)) else np.nan
    sep = np.nan
    if (p0["x"] > 0) and (p1["x"] > 0):
        sep = abs(np.log10(p0["x"]) - np.log10(p1["x"]))
    is_bi = bool(np.isfinite(sec_ratio) and (sec_ratio >= float(secondary_frac)) and np.isfinite(sep) and (sep >= float(BIMODAL_MIN_SEP_DECADES)))
    return is_bi, n, sep, sec_ratio

def _curve_peak_from_bins(
    df_sub,
    xcol: str = "Da",
    ycol: str = "Hysteresis",
    n_bins: int = N_BINS_PEAK,
    smooth_sigma: float = SMOOTH_SIGMA,
    q_low: float = Q_LOW,
    q_high: float = Q_HIGH,
    nmin: int = NMIN_PER_BIN,
    min_valid_bins: int = MIN_VALID_BINS_FOR_PEAK,
    smooth_sigma_decades: float = None,
    sigma_scale_with_nbins: bool = SMOOTH_SIGMA_SCALE_WITH_NBINS,
):
    """Support-aware peak detection from a binned+smoothed curve.

    Uses quantile bins in log10(x), then count-weighted smoothing on supported segments.

    Enhancements for publication-grade stability:
      - Optional smoothing specified in *log10(x) decades* (smooth_sigma_decades).
      - Optional sigma scaling when n_bins changes (sigma_scale_with_nbins).
      - Optional fine-grid peak refinement (PEAK_REFINE) to reduce bin-edge sensitivity.

    Returns:
        x_peak, y_peak, hit_left, hit_right, n_valid_bins, info_dict
    """
    built = _binned_curve_quantile_logx(
        df_sub,
        xcol=xcol,
        ycol=ycol,
        n_bins=int(n_bins),
        smooth_sigma=float(smooth_sigma),
        q_low=float(q_low),
        q_high=float(q_high),
        nmin=int(nmin),
        min_valid_bins=int(min_valid_bins),
        smooth_sigma_decades=smooth_sigma_decades,
        sigma_scale_with_nbins=bool(sigma_scale_with_nbins),
    )
    if built is None:
        return np.nan, np.nan, True, True, 0, None

    centers, y_smooth, count, is_valid = built

    # Peak / width / centroid metrics (refined if PEAK_REFINE=True)
    metrics = _peak_metrics_from_binned_curve(
        centers, y_smooth, is_valid,
        width_fracs=PEAK_WIDTH_FRACS,
        centroid_fracs=PEAK_CENTROID_FRACS,
        frac_main=float(PEAK_WIDTH_FRAC),
        interp=bool(PEAK_WIDTH_INTERP),
    )

    x_peak = metrics.get("x_peak", np.nan)
    y_peak = metrics.get("y_peak", np.nan)
    hit_left = metrics.get("hit_left", True)
    hit_right = metrics.get("hit_right", True)
    wdec = metrics.get("width_decades", np.nan)
    wL = metrics.get("width_left_decades", np.nan)
    wR = metrics.get("width_right_decades", np.nan)

    extra_widths = metrics.get("extra_widths", {}) or {}
    extra_centroids = metrics.get("extra_centroids", {}) or {}

    n_valid = int(np.count_nonzero(is_valid))

    # multi-peak / bimodality detection on the *coarse* smoothed curve (support-aware)
    n_peaks = 0
    multi_peak = False
    is_bi = False
    sep_dec = np.nan
    sec_ratio = np.nan

    # track second-strongest peak location (coarse), and primary low-Da peak for cross-lobe consistency
    peak2_x = np.nan
    primary_x = np.nan
    primary_y = np.nan

    try:
        if centers is not None and y_smooth is not None and is_valid is not None:
            centers_arr = np.asarray(centers, dtype=float)
            y_arr = np.asarray(y_smooth, dtype=float)
            vmask = np.asarray(is_valid, dtype=bool) & np.isfinite(centers_arr) & (centers_arr > 0) & np.isfinite(y_arr)

            # Use find_peaks on a NaN-safe vector
            yy = np.where(vmask, y_arr, np.nan)
            yy2 = np.where(np.isfinite(yy), yy, -np.inf)

            peaks, _ = find_peaks(yy2)
            peaks = [int(p) for p in peaks if vmask[int(p)]]

            n_peaks = int(len(peaks))
            multi_peak = bool(n_peaks >= 2)

            if multi_peak:
                # sort by peak height
                peaks_sorted = sorted(peaks, key=lambda k: float(yy2[int(k)]), reverse=True)
                k1 = int(peaks_sorted[0])
                k2 = int(peaks_sorted[1])

                x1 = float(centers_arr[k1])
                y1 = float(yy2[k1])
                x2 = float(centers_arr[k2])
                y2 = float(yy2[k2])

                peak2_x = float(x2) if np.isfinite(x2) else np.nan

                if (x1 > 0) and (x2 > 0) and np.isfinite(y1) and np.isfinite(y2) and (y1 > 0):
                    sep_dec = float(abs(np.log10(x2) - np.log10(x1)))
                    sec_ratio = float(y2 / y1)

                    # Bimodality requires substantial separation + substantial secondary amplitude
                    is_bi = bool((sep_dec >= float(BIMODAL_MIN_SEP_DECADES)) and (sec_ratio >= float(BIMODAL_SECONDARY_FRAC)))

                # Primary low-Da peak (consistent track left of DA_LOBE_SPLIT)
                cand = [p for p in peaks_sorted if (np.isfinite(centers_arr[int(p)]) and (centers_arr[int(p)] > 0) and (centers_arr[int(p)] <= float(DA_LOBE_SPLIT)))]
                if len(cand) > 0:
                    kL = int(cand[0])
                    primary_x = float(centers_arr[kL])
                    primary_y = float(yy2[kL])

    except Exception:
        pass

    # Cross-lobe maxima (low vs high) for xcol (typically Da), on coarse curve
    low_lobe_x = low_lobe_y = np.nan
    high_lobe_x = high_lobe_y = np.nan
    cross_bimodal = False
    cross_ratio = np.nan
    cross_sep_decades = np.nan
    try:
        if centers is not None and y_smooth is not None and is_valid is not None:
            centers_arr = np.asarray(centers, dtype=float)
            y_arr = np.asarray(y_smooth, dtype=float)
            vmask = np.asarray(is_valid, dtype=bool)

            # low lobe: x <= DA_LOBE_SPLIT
            mL = vmask & (centers_arr > 0) & (centers_arr <= float(DA_LOBE_SPLIT))
            if np.any(mL):
                idxL = np.where(mL)[0]
                jL = idxL[int(np.nanargmax(y_arr[idxL]))]
                low_lobe_x = float(centers_arr[jL])
                low_lobe_y = float(y_arr[jL])

            # high lobe: x > DA_LOBE_SPLIT
            mH = vmask & (centers_arr > float(DA_LOBE_SPLIT))
            if np.any(mH):
                idxH = np.where(mH)[0]
                jH = idxH[int(np.nanargmax(y_arr[idxH]))]
                high_lobe_x = float(centers_arr[jH])
                high_lobe_y = float(y_arr[jH])

            if np.isfinite(low_lobe_x) and np.isfinite(high_lobe_x) and (low_lobe_x > 0) and (high_lobe_x > 0):
                cross_sep_decades = float(abs(np.log10(high_lobe_x) - np.log10(low_lobe_x)))
                if np.isfinite(low_lobe_y) and np.isfinite(high_lobe_y):
                    yhi = float(max(low_lobe_y, high_lobe_y))
                    ylo = float(min(low_lobe_y, high_lobe_y))
                    if yhi > 0:
                        cross_ratio = float(ylo / yhi)
                        cross_bimodal = bool((cross_sep_decades >= float(BIMODAL_MIN_SEP_DECADES)) and (cross_ratio >= float(BIMODAL_SECONDARY_FRAC)))
    except Exception:
        pass

    info = {
        "centers": centers,
        "y_smooth": y_smooth,
        "count": count,
        "is_valid": is_valid,

        # peak refinement (publication stability)
        "peak_refine_used": int(metrics.get("refine_used", False)),
        "peak_x_bin": float(metrics.get("x_peak_bin", np.nan)),
        "peak_y_bin": float(metrics.get("y_peak_bin", np.nan)),
        "peak_refine_n": int(PEAK_REFINE_N_FINE),
        "peak_refine_smooth_decades": float(PEAK_REFINE_SMOOTH_DECADES),

        # peak width + flatness diagnostics (in decades)
        "width_decades": float(wdec) if np.isfinite(wdec) else np.nan,
        "width_left_decades": float(wL) if np.isfinite(wL) else np.nan,
        "width_right_decades": float(wR) if np.isfinite(wR) else np.nan,
        "width_frac": float(PEAK_WIDTH_FRAC),
        "width_interp": bool(PEAK_WIDTH_INTERP),

        # smoothing metadata
        "smooth_sigma": float(smooth_sigma),
        "smooth_sigma_decades": float(smooth_sigma_decades) if (smooth_sigma_decades is not None) else np.nan,
        "sigma_scale_with_nbins": int(bool(sigma_scale_with_nbins)),
        "n_bins": int(n_bins),

        **extra_widths,
        **extra_centroids,

        # cross-lobe maxima diagnostics
        "low_lobe_x": float(low_lobe_x) if np.isfinite(low_lobe_x) else np.nan,
        "low_lobe_y": float(low_lobe_y) if np.isfinite(low_lobe_y) else np.nan,
        "high_lobe_x": float(high_lobe_x) if np.isfinite(high_lobe_x) else np.nan,
        "high_lobe_y": float(high_lobe_y) if np.isfinite(high_lobe_y) else np.nan,
        "cross_bimodal": bool(cross_bimodal),
        "cross_ratio": float(cross_ratio) if np.isfinite(cross_ratio) else np.nan,
        "cross_sep_decades": float(cross_sep_decades) if np.isfinite(cross_sep_decades) else np.nan,

        # multi-peak / bimodality info
        "n_peaks": int(n_peaks),
        "multi_peak": bool(multi_peak),
        "bimodal": bool(is_bi),
        "peak_sep_decades": float(sep_dec) if np.isfinite(sep_dec) else np.nan,
        "secondary_ratio": float(sec_ratio) if np.isfinite(sec_ratio) else np.nan,
        "primary_lowDa_x": float(primary_x) if np.isfinite(primary_x) else np.nan,
        "primary_lowDa_y": float(primary_y) if np.isfinite(primary_y) else np.nan,
        "peak2_x": float(peak2_x) if np.isfinite(peak2_x) else np.nan,
    }

    return float(x_peak) if np.isfinite(x_peak) else np.nan,            float(y_peak) if np.isfinite(y_peak) else np.nan,            bool(hit_left), bool(hit_right), int(n_valid), info




def _print_peak_table(df):
    """Print peak summary table per contrast."""
    rows = []
    for R in sorted(df["Contrast"].unique()):
        sub = df[df["Contrast"] == R]
        da_pk, hi_pk, hitL, hitR, nvb, _ = _curve_peak_from_bins(sub, xcol="Da")
        dar_pk, _, hitL2, hitR2, nvb2, info_dar = _curve_peak_from_bins(sub, xcol="DaR")
        rows.append({
            "Contrast": float(R), "n": len(sub),
            "Da_peak_curve": da_pk, "DaR_peak_curve": dar_pk, "HI_at_Da_peak": hi_pk,
            "hitL_Da": hitL, "hitR_Da": hitR, "nBins_Da": nvb,
            "hitL_DaR": hitL2, "hitR_DaR": hitR2, "nBins_DaR": nvb2,
        })

    out = pd.DataFrame(rows)
    print("\n=== PEAK FROM BINNED+SMOOTHED CURVES ===")
    print(out.to_string(index=False, formatters={
        "Contrast": lambda v: f"{v:>5.1f}", "n": lambda v: f"{int(v):>5d}",
        "Da_peak_curve": lambda v: f"{v:.3e}" if np.isfinite(v) else "  nan ",
        "DaR_peak_curve": lambda v: f"{v:.3e}" if np.isfinite(v) else "  nan ",
    }))
    return out


def ridge_table_with_bootstrap(df, n_boot=N_BOOT, seed=BOOT_SEED, ci_q=RIDGE_CI_Q):
    """Ridge table with bootstrap CIs.

    Updates vs earlier versions:
      - CIs computed in log10-space for Da_peak and DaR_peak (RIDGE_CI_LOGSPACE=True),
        then exponentiated back (more interpretable across decades).
      - Adds peak-width / flatness diagnostic in decades:
            width_decades_f098 = log10(Da_hi) - log10(Da_lo) where HI >= 0.98*HI_max.
      - Boundary-hit replicates are treated as censored (axis-specific) and excluded from that axis CI.
    """
    rng = np.random.default_rng(seed)
    rows = []

    qlo, qhi = ci_q

    for R in sorted(df["Contrast"].unique()):
        sub = df[df["Contrast"] == R].copy()
        n = len(sub)
        if n < 150:
            continue

        da_pk, hi_pk, hitL, hitR, nvb, info_da = _curve_peak_from_bins(sub, xcol="Da")
        dar_pk, _, hitL2, hitR2, nvb2, info_dar = _curve_peak_from_bins(sub, xcol="DaR")

        wdec = info_da["width_decades"] if (info_da is not None) else np.nan
        wL = info_da["width_left_decades"] if (info_da is not None) else np.nan
        wR = info_da["width_right_decades"] if (info_da is not None) else np.nan


        # centroid (top-band f095) from the same smoothed curve (often more stable than argmax)
        daC_pk = float(info_da.get("centroid_x_f095", np.nan)) if (info_da is not None) else np.nan
        darC_pk = float(info_dar.get("centroid_x_f095", np.nan)) if (info_dar is not None) else np.nan

        # primary low-Da peak (if bimodal): track the low-Da lobe consistently
        daP_pk = float(info_da.get("primary_lowDa_x", np.nan)) if (info_da is not None) else np.nan
        darP_pk = float(info_dar.get("primary_lowDa_x", np.nan)) if (info_dar is not None) else np.nan

        # bimodality flags from the Da curve
        bimodal_flag = bool(info_da.get("bimodal", False)) if (info_da is not None) else False
        multi_peak_flag = bool(info_da.get("multi_peak", False)) if (info_da is not None) else False
        n_peaks = int(info_da.get("n_peaks", 0)) if (info_da is not None) else 0
        peak_sep_decades = float(info_da.get("peak_sep_decades", np.nan)) if (info_da is not None) else np.nan
        secondary_ratio = float(info_da.get("secondary_ratio", np.nan)) if (info_da is not None) else np.nan
        da_peak2 = float(info_da.get("peak2_x", np.nan)) if (info_da is not None) else np.nan

        # cross-lobe (low vs high) bimodality diagnostics
        cross_bimodal_flag = bool(info_da.get("cross_bimodal", False)) if (info_da is not None) else False
        cross_ratio = float(info_da.get("cross_ratio", np.nan)) if (info_da is not None) else np.nan
        cross_sep_decades = float(info_da.get("cross_sep_decades", np.nan)) if (info_da is not None) else np.nan
        da_low = float(info_da.get("low_lobe_x", np.nan)) if (info_da is not None) else np.nan
        hi_low = float(info_da.get("low_lobe_y", np.nan)) if (info_da is not None) else np.nan
        da_high = float(info_da.get("high_lobe_x", np.nan)) if (info_da is not None) else np.nan
        hi_high = float(info_da.get("high_lobe_y", np.nan)) if (info_da is not None) else np.nan

        da_samp = []
        dar_samp = []
        hi_samp = []
        daC_samp = []  # centroid in top-band (f095)
        darC_samp = []
        daP_samp = []  # primary low-Da peak
        darP_samp = []

        hitL_da = hitR_da = hitL_dar = hitR_dar = 0
        finite_da = finite_dar = 0

        if BOOTSTRAP:
            for _ in range(n_boot):
                sb = sub.iloc[rng.choice(n, size=n, replace=True)]

                da_b, hi_b, hL_b, hR_b, _, info_da_b = _curve_peak_from_bins(sb, xcol="Da")
                dar_b, _, hL2_b, hR2_b, _, info_dar_b = _curve_peak_from_bins(sb, xcol="DaR")

                hitL_da += int(hL_b)
                hitR_da += int(hR_b)
                hitL_dar += int(hL2_b)
                hitR_dar += int(hR2_b)

                if np.isfinite(da_b) and (da_b > 0.0):
                    finite_da += 1
                    if (not hL_b) and (not hR_b):
                        da_samp.append(np.log10(da_b) if RIDGE_CI_LOGSPACE else float(da_b))

                if np.isfinite(dar_b) and (dar_b > 0.0):
                    finite_dar += 1
                    if (not hL2_b) and (not hR2_b):
                        dar_samp.append(np.log10(dar_b) if RIDGE_CI_LOGSPACE else float(dar_b))


                # centroid samples (top-band f095), excluded if peak is boundary-censored
                daC_b = np.nan
                if info_da_b is not None:
                    daC_b = info_da_b.get("centroid_x_f095", np.nan)
                if np.isfinite(daC_b) and (daC_b > 0.0) and (not hL_b) and (not hR_b):
                    daC_samp.append(np.log10(daC_b) if RIDGE_CI_LOGSPACE else float(daC_b))

                darC_b = np.nan
                if info_dar_b is not None:
                    darC_b = info_dar_b.get("centroid_x_f095", np.nan)
                if np.isfinite(darC_b) and (darC_b > 0.0) and (not hL2_b) and (not hR2_b):
                    darC_samp.append(np.log10(darC_b) if RIDGE_CI_LOGSPACE else float(darC_b))


                # primary low-Da peaks from this bootstrap replicate (may be NaN if no low-Da lobe exists)
                daP_b = np.nan
                if info_da_b is not None:
                    daP_b = info_da_b.get("primary_lowDa_x", np.nan)
                darP_b = np.nan
                if info_dar_b is not None:
                    darP_b = info_dar_b.get("primary_lowDa_x", np.nan)

                daP_samp.append(np.log10(daP_b) if (RIDGE_CI_LOGSPACE and np.isfinite(daP_b) and daP_b > 0) else float(daP_b) if np.isfinite(daP_b) else np.nan)
                darP_samp.append(np.log10(darP_b) if (RIDGE_CI_LOGSPACE and np.isfinite(darP_b) and darP_b > 0) else float(darP_b) if np.isfinite(darP_b) else np.nan)

                if np.isfinite(hi_b) and (hi_b >= 0.0) and (not hL_b) and (not hR_b):
                    hi_samp.append(float(hi_b))

        def _ci(arr_list):
            """Quantile CI helper that ignores NaNs."""
            if len(arr_list) == 0:
                return np.nan, np.nan
            arr = np.asarray(arr_list, float)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                return np.nan, np.nan
            lo = float(np.quantile(arr, qlo))
            hi = float(np.quantile(arr, qhi))
            if RIDGE_CI_LOGSPACE:
                return float(10.0 ** lo), float(10.0 ** hi)
            return lo, hi

        Da_lo, Da_hi = _ci(da_samp)
        DaR_lo, DaR_hi = _ci(dar_samp)
        DaP_lo, DaP_hi = _ci(daP_samp)
        DaRP_lo, DaRP_hi = _ci(darP_samp)
        DaC_lo, DaC_hi = _ci(daC_samp)
        DaRC_lo, DaRC_hi = _ci(darC_samp)

        HI_lo = float(np.quantile(np.asarray(hi_samp), qlo)) if len(hi_samp) else np.nan
        HI_hi = float(np.quantile(np.asarray(hi_samp), qhi)) if len(hi_samp) else np.nan

        logDa_pk = float(np.log10(da_pk)) if (np.isfinite(da_pk) and da_pk > 0) else np.nan
        logDaR_pk = float(np.log10(dar_pk)) if (np.isfinite(dar_pk) and dar_pk > 0) else np.nan
        logDaC_pk = float(np.log10(daC_pk)) if (np.isfinite(daC_pk) and daC_pk > 0) else np.nan
        logDaRC_pk = float(np.log10(darC_pk)) if (np.isfinite(darC_pk) and darC_pk > 0) else np.nan
        dlogDaC_minus_peak = float(logDaC_pk - logDa_pk) if (np.isfinite(logDaC_pk) and np.isfinite(logDa_pk)) else np.nan
        dlogDaRC_minus_peak = float(logDaRC_pk - logDaR_pk) if (np.isfinite(logDaRC_pk) and np.isfinite(logDaR_pk)) else np.nan

        rows.append({
            "Contrast": float(R), "n": int(n),

            "Da_peak": float(da_pk), "Da_peak_lo": float(Da_lo), "Da_peak_hi": float(Da_hi),
            "Da_peak_primary_lowDa": float(daP_pk), "Da_peak_primary_lowDa_lo": float(DaP_lo), "Da_peak_primary_lowDa_hi": float(DaP_hi),
            "Da_centroid_f095": float(daC_pk), "Da_centroid_f095_lo": float(DaC_lo), "Da_centroid_f095_hi": float(DaC_hi),
            "dlog10Da_centroid_f095_minus_peak": float(dlogDaC_minus_peak),
            "DaR_peak": float(dar_pk), "DaR_peak_lo": float(DaR_lo), "DaR_peak_hi": float(DaR_hi),
            "DaR_peak_primary_lowDa": float(darP_pk), "DaR_peak_primary_lowDa_lo": float(DaRP_lo), "DaR_peak_primary_lowDa_hi": float(DaRP_hi),
            "DaR_centroid_f095": float(darC_pk), "DaR_centroid_f095_lo": float(DaRC_lo), "DaR_centroid_f095_hi": float(DaRC_hi),
            "dlog10DaR_centroid_f095_minus_peak": float(dlogDaRC_minus_peak),

            "log10Da_peak": logDa_pk, "log10DaR_peak": logDaR_pk,

            "HI_peak_curve": float(hi_pk),

            "bimodal_flag": bool(bimodal_flag), "multi_peak_flag": bool(multi_peak_flag), "n_peaks": int(n_peaks), "peak_sep_decades": float(peak_sep_decades) if np.isfinite(peak_sep_decades) else np.nan, "secondary_ratio": float(secondary_ratio) if np.isfinite(secondary_ratio) else np.nan, "Da_peak2": float(da_peak2) if np.isfinite(da_peak2) else np.nan,
            "cross_bimodal": bool(cross_bimodal_flag), "cross_ratio": float(cross_ratio) if np.isfinite(cross_ratio) else np.nan, "cross_sep_decades": float(cross_sep_decades) if np.isfinite(cross_sep_decades) else np.nan,
            "Da_low_lobe": float(da_low) if np.isfinite(da_low) else np.nan, "HI_low_lobe": float(hi_low) if np.isfinite(hi_low) else np.nan,
            "Da_high_lobe": float(da_high) if np.isfinite(da_high) else np.nan, "HI_high_lobe": float(hi_high) if np.isfinite(hi_high) else np.nan,
            "HI_peak_lo": float(HI_lo), "HI_peak_hi": float(HI_hi),
            # peak-width / flatness diagnostics (decades in log10(Da))
            "width_decades_f098": float(info_da.get("width_decades_f098", np.nan)) if (info_da is not None) else (float(wdec) if np.isfinite(wdec) else np.nan),
            "width_left_decades_f098": float(info_da.get("width_left_decades_f098", np.nan)) if (info_da is not None) else (float(wL) if np.isfinite(wL) else np.nan),
            "width_right_decades_f098": float(info_da.get("width_right_decades_f098", np.nan)) if (info_da is not None) else (float(wR) if np.isfinite(wR) else np.nan),

            "width_decades_f095": float(info_da.get("width_decades_f095", np.nan)) if (info_da is not None) else np.nan,
            "width_left_decades_f095": float(info_da.get("width_left_decades_f095", np.nan)) if (info_da is not None) else np.nan,
            "width_right_decades_f095": float(info_da.get("width_right_decades_f095", np.nan)) if (info_da is not None) else np.nan,

            "width_decades_f090": float(info_da.get("width_decades_f090", np.nan)) if (info_da is not None) else np.nan,
            "width_left_decades_f090": float(info_da.get("width_left_decades_f090", np.nan)) if (info_da is not None) else np.nan,
            "width_right_decades_f090": float(info_da.get("width_right_decades_f090", np.nan)) if (info_da is not None) else np.nan,

            "width_frac": float(PEAK_WIDTH_FRAC),
            "width_interp": bool(PEAK_WIDTH_INTERP),

            "hitL_Da": bool(hitL), "hitR_Da": bool(hitR),
            "hitL_DaR": bool(hitL2), "hitR_DaR": bool(hitR2),
            "nBins_Da": int(nvb), "nBins_DaR": int(nvb2),

            "n_boot": int(n_boot) if BOOTSTRAP else 0,
            "boot_kept_Da": int(len(da_samp)),
            "boot_kept_DaR": int(len(dar_samp)),
            "boot_finite_Da": int(finite_da),
            "boot_finite_DaR": int(finite_dar),

            "boot_hitL_Da_frac": (hitL_da / n_boot) if BOOTSTRAP else np.nan,
            "boot_hitR_Da_frac": (hitR_da / n_boot) if BOOTSTRAP else np.nan,
            "boot_hitL_DaR_frac": (hitL_dar / n_boot) if BOOTSTRAP else np.nan,
            "boot_hitR_DaR_frac": (hitR_dar / n_boot) if BOOTSTRAP else np.nan,
        })

    out = pd.DataFrame(rows)

    print("\n=== RIDGE TABLE + BOOTSTRAP CI (log-space) ===")
    cols = ["Contrast", "n",
            "Da_peak", "Da_peak_lo", "Da_peak_hi", "Da_peak_primary_lowDa", "Da_peak_primary_lowDa_lo", "Da_peak_primary_lowDa_hi", "Da_centroid_f095",
            "HI_peak_curve",
            "bimodal_flag", "multi_peak_flag", "n_peaks", "peak_sep_decades", "secondary_ratio", "cross_bimodal", "cross_ratio", "Da_low_lobe", "Da_high_lobe", "width_decades_f098",
            "DaR_peak", "DaR_peak_lo", "DaR_peak_hi", "DaR_centroid_f095",
            "boot_kept_Da", "boot_kept_DaR",
            "boot_hitL_Da_frac", "boot_hitR_Da_frac"]
    cols = [c for c in cols if c in out.columns]
    with pd.option_context("display.max_rows", 200, "display.max_columns", 200):
        print(out[cols].to_string(index=False))
    return out

# ==============================================================================
# RIDGE EXTRACTION ROBUSTNESS (SENSITIVITY)
# ==============================================================================

def _curve_diag_points(
    sub,
    da_col="Da",
    hi_col="Hysteresis",
    da_split=1e-2,
    top_frac=0.05,
    min_per_lobe=10,
    cross_ratio_thresh=0.5,
):
    """
    Point-based cross-lobe diagnostic for small subsets (e.g., IVP-only).
    Uses top-fraction statistics to reduce sensitivity to single-point outliers.

    Returns:
        da_pk:  log-centroid Da of global top-fraction points
        daL:    log-centroid Da of top-fraction points in low lobe (Da <= da_split)
        hiL:    mean HI of top-fraction points in low lobe
        daH:    log-centroid Da of top-fraction points in high lobe (Da > da_split)
        hiH:    mean HI of top-fraction points in high lobe
        cross_bimodal: True if both lobes are present and cross_ratio >= threshold
        cross_ratio: min(hiL,hiH)/max(hiL,hiH) (NaN if not computable)
    """

    # --- sanitize ---
    if sub is None or getattr(sub, "empty", True):
        return (np.nan, np.nan, np.nan, np.nan, np.nan, 0, np.nan)

    # resolve expected column names (defensive against earlier naming)
    if da_col not in sub.columns:
        if "Da" in sub.columns:
            da_col = "Da"
        elif "da" in sub.columns:
            da_col = "da"
        else:
            return (np.nan, np.nan, np.nan, np.nan, np.nan, 0, np.nan)

    if hi_col not in sub.columns:
        if "Hysteresis" in sub.columns:
            hi_col = "Hysteresis"
        elif "HI" in sub.columns:
            hi_col = "HI"
        else:
            return (np.nan, np.nan, np.nan, np.nan, np.nan, 0, np.nan)

    x = sub[[da_col, hi_col]].copy()
    x = x.replace([np.inf, -np.inf], np.nan).dropna()
    x = x[(x[da_col] > 0) & (x[hi_col].notna())]

    if x.empty:
        return (np.nan, np.nan, np.nan, np.nan, np.nan, 0, np.nan)

    def _top_stats(df):
        """Return (da_centroid, hi_mean, n_total) for a lobe/global df."""
        n = int(df.shape[0])
        if n <= 0:
            return (np.nan, np.nan, 0)

        # top_frac as at least 1 point
        k = max(1, int(np.ceil(top_frac * n)))

        # pick top-k by HI
        top = df.nlargest(k, hi_col)

        # mean HI among top points
        hi_mean = float(top[hi_col].mean())

        # log-centroid Da among top points (geometric mean)
        da_vals = top[da_col].to_numpy(dtype=float)
        da_centroid = float(10.0 ** np.mean(np.log10(da_vals)))

        return (da_centroid, hi_mean, n)

    # --- low / high lobes ---
    low = x[x[da_col] <= da_split]
    high = x[x[da_col] > da_split]

    daL, hiL, nL = _top_stats(low)
    daH, hiH, nH = _top_stats(high)

    # --- global "peak" proxy (top-fraction over all points) ---
    da_pk, _, _ = _top_stats(x)

    # --- cross-ratio ---
    if np.isfinite(hiL) and np.isfinite(hiH) and (hiL > 0) and (hiH > 0):
        cross_ratio = float(min(hiL, hiH) / max(hiL, hiH))
    else:
        cross_ratio = np.nan

    # --- cross-bimodal flag ---
    cross_bimodal = int(
        (nL >= min_per_lobe)
        and (nH >= min_per_lobe)
        and np.isfinite(cross_ratio)
        and (cross_ratio >= cross_ratio_thresh)
    )

    return (da_pk, daL, hiL, daH, hiH, cross_bimodal, cross_ratio)

def bimodality_source_compare(df: pd.DataFrame, out_csv: str = None) -> pd.DataFrame:
    """Compare cross-lobe (low vs high) structure for:
        - ALL runs (post-fallback dataset; 'best available' truth),
        - IVP-validated subset (fallback_used==1),
        - RK4-only subset (fallback_used==0).

    This is meant to answer: does the high-Da lobe exist in IVP-supported data, or only in RK4-only points?

    Notes on robustness:
      * ALL and RK4 subsets are typically large enough for binned+smoothed curve diagnostics.
      * IVP-only subsets can be small; we fall back to a point-based diagnostic (top-fraction stats).
      * We also report per-lobe sample counts so "ivp-only=0" can be interpreted correctly.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    def _resolve_hi_col(sub: pd.DataFrame, preferred: str = "Hysteresis") -> str:
        if preferred in sub.columns:
            return preferred
        for alt in ("HI", "Hysteresis", "hysteresis", "hi"):
            if alt in sub.columns:
                return alt
        return None

    def _lobe_counts(sub: pd.DataFrame,
                     da_split: float = DA_LOBE_SPLIT,
                     da_col: str = "Da",
                     hi_col_preferred: str = "Hysteresis"):
        """Return (n_valid, n_low, n_high) after finite filtering."""
        if sub is None or sub.empty or (da_col not in sub.columns):
            return (0, 0, 0)
        hi_col = _resolve_hi_col(sub, preferred=hi_col_preferred)
        if hi_col is None:
            return (0, 0, 0)

        x = sub[[da_col, hi_col]].replace([np.inf, -np.inf], np.nan).dropna()
        if x.empty:
            return (0, 0, 0)
        da = x[da_col].to_numpy(dtype=float)
        hi = x[hi_col].to_numpy(dtype=float)
        msk = np.isfinite(da) & np.isfinite(hi) & (da > 0)
        if not np.any(msk):
            return (0, 0, 0)
        da = da[msk]
        n_valid = int(da.size)
        n_low = int(np.sum(da <= float(da_split)))
        n_high = int(n_valid - n_low)
        return (n_valid, n_low, n_high)

    def _mid_band_max(sub: pd.DataFrame,
                      da_min: float = DA_MID_MIN,
                      da_max: float = DA_MID_MAX,
                      da_col: str = "Da",
                      hi_col_preferred: str = "Hysteresis",
                      top_frac: float = 0.05,
                      min_n: int = 10):
        """Return (n_mid, da_mid, hi_mid) in explicit mid-Da band [da_min, da_max].

        Robust summary (outlier-resistant):
          - n_mid: number of valid points in the band
          - da_mid: geometric-mean Da of the top-fraction HI points (log-centroid)
          - hi_mid: mean HI of those top-fraction points

        This is intentionally *not* a raw max, to avoid single-point outliers
        creating a false "hump" verdict.
        """
        if sub is None or sub.empty or (da_col not in sub.columns):
            return (0, np.nan, np.nan)
        hi_col = _resolve_hi_col(sub, preferred=hi_col_preferred)
        if hi_col is None:
            return (0, np.nan, np.nan)

        x = sub[[da_col, hi_col]].replace([np.inf, -np.inf], np.nan).dropna()
        if x.empty:
            return (0, np.nan, np.nan)

        da = x[da_col].to_numpy(dtype=float)
        hi = x[hi_col].to_numpy(dtype=float)

        msk = (np.isfinite(da) & np.isfinite(hi) &
               (da > 0) &
               (da >= float(da_min)) & (da <= float(da_max)))
        if not np.any(msk):
            return (0, np.nan, np.nan)

        da_m = da[msk]
        hi_m = hi[msk]
        n_mid = int(hi_m.size)

        if n_mid < int(min_n):
            # underpowered: return count only
            return (n_mid, np.nan, np.nan)

        k = max(1, int(np.ceil(float(top_frac) * n_mid)))
        # top-k by HI
        idx_top = np.argsort(hi_m)[-k:]
        da_top = da_m[idx_top]
        hi_top = hi_m[idx_top]

        # mean HI among top points
        hi_mid = float(np.mean(hi_top))

        # log-centroid Da (geometric mean) among top points
        da_mid = float(10.0 ** np.mean(np.log10(da_top)))

        return (n_mid, da_mid, hi_mid)

    def _curve_diag(sub: pd.DataFrame):
        if sub is None or sub.empty:
            return (np.nan, np.nan, np.nan, np.nan, np.nan, 0, np.nan)
        da_pk, _, _, _, _, info = _curve_peak_from_bins(sub, xcol="Da")
        if info is None:
            return _curve_diag_points(
                sub,
                da_col="Da",
                hi_col="Hysteresis",
                da_split=DA_LOBE_SPLIT,
                top_frac=0.05,
                min_per_lobe=10,
                cross_ratio_thresh=float(BIMODAL_SECONDARY_FRAC),
            )
        daL = float(info.get("low_lobe_x", np.nan))
        hiL = float(info.get("low_lobe_y", np.nan))
        daH = float(info.get("high_lobe_x", np.nan))
        hiH = float(info.get("high_lobe_y", np.nan))
        cb = int(bool(info.get("cross_bimodal", False)))
        cr = float(info.get("cross_ratio", np.nan))
        return (da_pk, daL, hiL, daH, hiH, cb, cr)

    rows = []
    min_n = int(BIMODALITY_SOURCE_MIN_N)

    for R in sorted(df["Contrast"].unique()):
        sub_all = df[df["Contrast"] == R].copy()
        if len(sub_all) < 30:
            continue

        # Per-lobe counts (valid points) for interpretation
        nV_all, nL_all, nH_all = _lobe_counts(sub_all)

        # ALL (post-fallback)
        da_pk_all, daL_all, hiL_all, daH_all, hiH_all, cb_all, cr_all = _curve_diag(sub_all)
        # Mid-Da hump band (explicit): Da in [DA_MID_MIN, DA_MID_MAX]
        nmid_all, da_mid_all, hi_mid_all = _mid_band_max(sub_all)
        mid_ratio_all = (hi_mid_all / hiL_all) if (np.isfinite(hi_mid_all) and np.isfinite(hiL_all) and (hiL_all > 0)) else np.nan
        mid_flag_all = int(bool(np.isfinite(mid_ratio_all) and (mid_ratio_all >= float(MID_DA_RATIO_THRESH))))

        # IVP-only (fallback)
        sub_ivp = sub_all[sub_all.get("fallback_used", 0).astype(int) == 1].copy()
        n_ivp = int(len(sub_ivp))
        nV_ivp, nL_ivp, nH_ivp = _lobe_counts(sub_ivp)
        if n_ivp >= min_n:
            da_pk_ivp, daL_ivp, hiL_ivp, daH_ivp, hiH_ivp, cb_ivp, cr_ivp = _curve_diag(sub_ivp)
        else:
            da_pk_ivp = daL_ivp = hiL_ivp = daH_ivp = hiH_ivp = np.nan
            cb_ivp = 0
            cr_ivp = np.nan

        # Mid-Da hump band stats for IVP subset (point-based)
        nmid_ivp, da_mid_ivp, hi_mid_ivp = _mid_band_max(sub_ivp)
        mid_ratio_ivp = (hi_mid_ivp / hiL_ivp) if (np.isfinite(hi_mid_ivp) and np.isfinite(hiL_ivp) and (hiL_ivp > 0)) else np.nan
        mid_flag_ivp = int(bool(np.isfinite(mid_ratio_ivp) and (mid_ratio_ivp >= float(MID_DA_RATIO_THRESH))))

        # RK4-only (non-fallback)
        sub_rk4 = sub_all[sub_all.get("fallback_used", 0).astype(int) == 0].copy()
        n_rk4 = int(len(sub_rk4))
        nV_rk4, nL_rk4, nH_rk4 = _lobe_counts(sub_rk4)
        if n_rk4 >= min_n:
            da_pk_rk4, daL_rk4, hiL_rk4, daH_rk4, hiH_rk4, cb_rk4, cr_rk4 = _curve_diag(sub_rk4)
        else:
            da_pk_rk4 = daL_rk4 = hiL_rk4 = daH_rk4 = hiH_rk4 = np.nan
            cb_rk4 = 0
            cr_rk4 = np.nan

        # Mid-Da hump band stats for RK4-only subset (point-based)
        nmid_rk4, da_mid_rk4, hi_mid_rk4 = _mid_band_max(sub_rk4)
        mid_ratio_rk4 = (hi_mid_rk4 / hiL_rk4) if (np.isfinite(hi_mid_rk4) and np.isfinite(hiL_rk4) and (hiL_rk4 > 0)) else np.nan
        mid_flag_rk4 = int(bool(np.isfinite(mid_ratio_rk4) and (mid_ratio_rk4 >= float(MID_DA_RATIO_THRESH))))

        rows.append({
            "Contrast": float(R),
            "n_all": int(len(sub_all)),
            "n_ivp": n_ivp,
            "n_rk4": n_rk4,

            "n_valid_all": nV_all,
            "n_low_all": nL_all,
            "n_high_all": nH_all,

            "n_valid_ivp": nV_ivp,
            "n_low_ivp": nL_ivp,
            "n_high_ivp": nH_ivp,

            "n_valid_rk4": nV_rk4,
            "n_low_rk4": nL_rk4,
            "n_high_rk4": nH_rk4,

            "Da_peak_all": float(da_pk_all) if np.isfinite(da_pk_all) else np.nan,
            "Da_low_all": float(daL_all) if np.isfinite(daL_all) else np.nan,
            "HI_low_all": float(hiL_all) if np.isfinite(hiL_all) else np.nan,
            "Da_high_all": float(daH_all) if np.isfinite(daH_all) else np.nan,
            "HI_high_all": float(hiH_all) if np.isfinite(hiH_all) else np.nan,
            "cross_bimodal_all": int(cb_all),
            "cross_ratio_all": float(cr_all) if np.isfinite(cr_all) else np.nan,

            "n_mid_all": int(nmid_all),
            "Da_mid_all": float(da_mid_all) if np.isfinite(da_mid_all) else np.nan,
            "HI_mid_all": float(hi_mid_all) if np.isfinite(hi_mid_all) else np.nan,
            "mid_ratio_all": float(mid_ratio_all) if np.isfinite(mid_ratio_all) else np.nan,
            "mid_hump_all": int(mid_flag_all),

            "Da_peak_ivp": float(da_pk_ivp) if np.isfinite(da_pk_ivp) else np.nan,
            "Da_low_ivp": float(daL_ivp) if np.isfinite(daL_ivp) else np.nan,
            "HI_low_ivp": float(hiL_ivp) if np.isfinite(hiL_ivp) else np.nan,
            "Da_high_ivp": float(daH_ivp) if np.isfinite(daH_ivp) else np.nan,
            "HI_high_ivp": float(hiH_ivp) if np.isfinite(hiH_ivp) else np.nan,
            "cross_bimodal_ivp": int(cb_ivp),
            "cross_ratio_ivp": float(cr_ivp) if np.isfinite(cr_ivp) else np.nan,

            "n_mid_ivp": int(nmid_ivp),
            "Da_mid_ivp": float(da_mid_ivp) if np.isfinite(da_mid_ivp) else np.nan,
            "HI_mid_ivp": float(hi_mid_ivp) if np.isfinite(hi_mid_ivp) else np.nan,
            "mid_ratio_ivp": float(mid_ratio_ivp) if np.isfinite(mid_ratio_ivp) else np.nan,
            "mid_hump_ivp": int(mid_flag_ivp),

            "Da_peak_rk4": float(da_pk_rk4) if np.isfinite(da_pk_rk4) else np.nan,
            "Da_low_rk4": float(daL_rk4) if np.isfinite(daL_rk4) else np.nan,
            "HI_low_rk4": float(hiL_rk4) if np.isfinite(hiL_rk4) else np.nan,
            "Da_high_rk4": float(daH_rk4) if np.isfinite(daH_rk4) else np.nan,
            "HI_high_rk4": float(hiH_rk4) if np.isfinite(hiH_rk4) else np.nan,
            "cross_bimodal_rk4": int(cb_rk4),
            "cross_ratio_rk4": float(cr_rk4) if np.isfinite(cr_rk4) else np.nan,
            "n_mid_rk4": int(nmid_rk4),
            "Da_mid_rk4": float(da_mid_rk4) if np.isfinite(da_mid_rk4) else np.nan,
            "HI_mid_rk4": float(hi_mid_rk4) if np.isfinite(hi_mid_rk4) else np.nan,
            "mid_ratio_rk4": float(mid_ratio_rk4) if np.isfinite(mid_ratio_rk4) else np.nan,
            "mid_hump_rk4": int(mid_flag_rk4),
        })

    out = pd.DataFrame(rows)
    if out_csv:
        try:
            out.to_csv(out_csv, index=False)
        except Exception:
            pass
    return out

def ridge_sensitivity(df: pd.DataFrame, out_csv: str = None):
    """Robustness check for ridge location vs. binning/smoothing choices.

    Computes Da_peak(R) under a small grid of (n_bins, smooth_sigma) values and
    reports deviations in log10(Da_peak) relative to the baseline (N_BINS_PEAK, SMOOTH_SIGMA).

    Also computes an alternative ridge location:
        Da_centroid_f095 = log-space centroid of the band HI >= 0.95*HI_max
    which is often more stable than argmax when the resonance peak is flat.

    Returns:
        sens_df (per-R per-setting), sens_info (compact summary), sens_byR (per-R range summary).
    """
    if df is None or df.empty:
        return None, None, None

    Rs = sorted(df["Contrast"].unique())
    if len(Rs) == 0:
        return None, None, None

    # --- baseline (default extraction parameters) ---
    base_peak = {}
    base_cent = {}
    for R in Rs:
        sub = df[df["Contrast"] == R]
        da_pk, _, _, _, _, info = _curve_peak_from_bins(
            sub, xcol="Da",
            n_bins=N_BINS_PEAK, smooth_sigma=SMOOTH_SIGMA,
            nmin=SENS_NMIN_PER_BIN, min_valid_bins=SENS_MIN_VALID_BINS
        )
        base_peak[R] = float(np.log10(da_pk)) if (np.isfinite(da_pk) and da_pk > 0) else np.nan

        daC = np.nan
        if info is not None:
            daC = info.get("centroid_x_f095", np.nan)
        base_cent[R] = float(np.log10(daC)) if (np.isfinite(daC) and daC > 0) else np.nan

    settings = [(int(nb), float(sig)) for nb in SENS_NBINS_LIST for sig in SENS_SIGMA_LIST]

    # --- evaluate grid ---
    rows = []
    for nb, sig in settings:
        for R in Rs:
            sub = df[df["Contrast"] == R]
            da_pk, hi_pk, hitL, hitR, nvb, info = _curve_peak_from_bins(
                sub, xcol="Da",
                n_bins=nb, smooth_sigma=sig,
                nmin=SENS_NMIN_PER_BIN, min_valid_bins=SENS_MIN_VALID_BINS
            )
            logDa = float(np.log10(da_pk)) if (np.isfinite(da_pk) and da_pk > 0) else np.nan

            daC = np.nan
            logDaC = np.nan
            if info is not None:
                daC = info.get("centroid_x_f095", np.nan)
                logDaC = float(np.log10(daC)) if (np.isfinite(daC) and daC > 0) else np.nan

            base0 = base_peak.get(R, np.nan)
            baseC0 = base_cent.get(R, np.nan)

            dlog_peak = (logDa - base0) if (np.isfinite(logDa) and np.isfinite(base0)) else np.nan
            dlog_cent = (logDaC - baseC0) if (np.isfinite(logDaC) and np.isfinite(baseC0)) else np.nan

            rows.append({
                "Contrast": float(R),
                "n_bins": int(nb),
                "sigma": float(sig),
                "log10Da_peak": float(logDa) if np.isfinite(logDa) else np.nan,
                "dlog10Da_peak_vs_base": float(dlog_peak) if np.isfinite(dlog_peak) else np.nan,
                "log10Da_centroid_f095": float(logDaC) if np.isfinite(logDaC) else np.nan,
                "dlog10Da_centroid_f095_vs_base": float(dlog_cent) if np.isfinite(dlog_cent) else np.nan,
                "HI_peak": float(hi_pk) if np.isfinite(hi_pk) else np.nan,
                "hitL": bool(hitL),
                "hitR": bool(hitR),
                "nBins_valid": int(nvb),
            })

    out = pd.DataFrame(rows)

    # --- summarize by setting ---
    summ_rows = []
    for nb, sig in settings:
        sub = out[(out["n_bins"] == nb) & (out["sigma"] == sig)]
        d1 = sub["dlog10Da_peak_vs_base"].to_numpy()
        d1 = d1[np.isfinite(d1)]
        d2 = sub["dlog10Da_centroid_f095_vs_base"].to_numpy()
        d2 = d2[np.isfinite(d2)]
        if d1.size == 0 and d2.size == 0:
            continue
        summ_rows.append({
            "n_bins": nb,
            "sigma": sig,
            "max_abs_dlog10Da_peak": float(np.max(np.abs(d1))) if d1.size else np.nan,
            "median_abs_dlog10Da_peak": float(np.median(np.abs(d1))) if d1.size else np.nan,
            "max_abs_dlog10Da_centroid_f095": float(np.max(np.abs(d2))) if d2.size else np.nan,
            "median_abs_dlog10Da_centroid_f095": float(np.median(np.abs(d2))) if d2.size else np.nan,
        })
    summ = pd.DataFrame(summ_rows)

    # --- summarize by R (range across settings) ---
    byR_rows = []
    for R in Rs:
        sub = out[out["Contrast"] == float(R)]
        a = sub["log10Da_peak"].to_numpy()
        a = a[np.isfinite(a)]
        r_peak = float(np.max(a) - np.min(a)) if a.size else np.nan

        b = sub["log10Da_centroid_f095"].to_numpy()
        b = b[np.isfinite(b)]
        r_cent = float(np.max(b) - np.min(b)) if b.size else np.nan

        byR_rows.append({"Contrast": float(R), "range_log10Da_peak": r_peak, "range_log10Da_centroid_f095": r_cent})
    byR = pd.DataFrame(byR_rows)

    # --- pack sens_info ---
    sens_info = None
    if not summ.empty:
        worst_peak = summ.dropna(subset=["max_abs_dlog10Da_peak"]).sort_values("max_abs_dlog10Da_peak", ascending=False).head(1)
        worst_cent = summ.dropna(subset=["max_abs_dlog10Da_centroid_f095"]).sort_values("max_abs_dlog10Da_centroid_f095", ascending=False).head(1)

        sens_info = {}

        if len(worst_peak):
            wp = worst_peak.iloc[0]
            sens_info.update({
                "overall_max_abs_dlog10Da_peak": float(wp["max_abs_dlog10Da_peak"]),
                "overall_median_abs_dlog10Da_peak": float(wp["median_abs_dlog10Da_peak"]) if np.isfinite(wp["median_abs_dlog10Da_peak"]) else np.nan,
                "worst_n_bins_peak": int(wp["n_bins"]),
                "worst_sigma_peak": float(wp["sigma"]),
            })

        if len(worst_cent):
            wc = worst_cent.iloc[0]
            sens_info.update({
                "overall_max_abs_dlog10Da_centroid_f095": float(wc["max_abs_dlog10Da_centroid_f095"]),
                "overall_median_abs_dlog10Da_centroid_f095": float(wc["median_abs_dlog10Da_centroid_f095"]) if np.isfinite(wc["median_abs_dlog10Da_centroid_f095"]) else np.nan,
                "worst_n_bins_centroid": int(wc["n_bins"]),
                "worst_sigma_centroid": float(wc["sigma"]),
            })

        # Typical (across settings): median of per-setting maxima
        try:
            v1 = summ["max_abs_dlog10Da_peak"].to_numpy()
            v1 = v1[np.isfinite(v1)]
            v2 = summ["max_abs_dlog10Da_centroid_f095"].to_numpy()
            v2 = v2[np.isfinite(v2)]
            sens_info["median_over_settings_max_abs_peak"] = float(np.median(v1)) if v1.size else np.nan
            sens_info["median_over_settings_max_abs_centroid"] = float(np.median(v2)) if v2.size else np.nan
        except Exception:
            pass

        # Top-R ranges (largest sensitivity across settings)
        try:
            topP = byR.dropna(subset=["range_log10Da_peak"]).sort_values("range_log10Da_peak", ascending=False).head(5)
            topC = byR.dropna(subset=["range_log10Da_centroid_f095"]).sort_values("range_log10Da_centroid_f095", ascending=False).head(5)
            sens_info["topR_range_peak"] = [(float(r), float(v)) for r, v in zip(topP["Contrast"], topP["range_log10Da_peak"])]
            sens_info["topR_range_centroid"] = [(float(r), float(v)) for r, v in zip(topC["Contrast"], topC["range_log10Da_centroid_f095"])]
            sens_info["median_range_peak"] = float(np.nanmedian(byR["range_log10Da_peak"].to_numpy()))
            sens_info["median_range_centroid"] = float(np.nanmedian(byR["range_log10Da_centroid_f095"].to_numpy()))
        except Exception:
            pass

    # --- optional writes ---
    if out_csv:
        try:
            out.to_csv(out_csv, index=False)
        except Exception as e:
            print(f"[Sensitivity] could not write {out_csv} ({e})")
        try:
            base = out_csv.replace(".csv", "_summary.csv")
            summ.to_csv(base, index=False)
        except Exception:
            pass
        try:
            base = out_csv.replace(".csv", "_byR.csv")
            byR.to_csv(base, index=False)
        except Exception:
            pass

    return out, sens_info, byR




# ==============================================================================
# NAN-AWARE SMOOTHING
# ==============================================================================

def nan_gaussian_smooth(Z, sigma=0.8, min_weight=0.05):
    """Smooth Z while respecting NaNs."""
    Z = np.asarray(Z, float)
    W = np.isfinite(Z).astype(float)
    Z0 = np.nan_to_num(Z, nan=0.0)
    Zs = gaussian_filter(Z0, sigma=sigma)
    Ws = gaussian_filter(W, sigma=sigma)
    out = Zs / np.maximum(Ws, 1e-12)
    out[Ws < min_weight] = np.nan
    return out


# ==============================================================================
# PARALLEL WORKER FUNCTION
# ==============================================================================

# def _run_single_ode(args):
#     """Worker function for parallel ODE solving."""
#     contrast_val, i, K, alpha, logK, logA, DaR_target, Sy_m, P_vec = args
    
#     # Solve ODE
#     is_tri = (HYETO == "tri")  # RK4 supports tri/box directly; gamma should fallback to solve_ivp
#     if USE_RK4 and HYETO in ("tri", "box"):
#         Hf, Hm = rk4_integrate_arrays(K, alpha, Sy_f, Sy_m, P_mag, P_dur, t0_storm, TRI_PEAK_FRAC,
#                                       Q_BASE, K_LIN, is_tri, DT, N_STEPS, 0.1, 0.1)
#     else:
#         # fallback
#         sol = solve_ivp(universal_model, (0.0, T_END), (0.1, 0.1), t_eval=t_eval,
#                         args=(K, alpha, Sy_f, Sy_m, P_mag, P_dur), rtol=RTOL, atol=ATOL)
#         if sol.status < 0 or sol.y.shape[1] != t_eval.size:
#             return None
#         Hf, Hm = sol.y[0], sol.y[1]

#     if sol.status < 0 or sol.y.shape[1] != t_eval.size:
#         return None

#     Hf, Hm = sol.y[0], sol.y[1]
#     H_peak = float(np.max(Hf))
#     if not np.isfinite(H_peak) or H_peak <= 0.0:
#         return None

#     # Optional DaR refinement
#     if USE_DAR_SAMPLING and REFINE_DAR_ONCE:
#         Qout_pk = Q_BASE + K_LIN * H_peak + K * H_peak * H_peak
#         if Qout_pk > 0:
#             DaR_ach = (alpha * H_peak) / (Qout_pk + 1e-30)
#             scale = np.clip(DaR_target / (DaR_ach + 1e-30), 1e-3, 1e3)
#             alpha2 = alpha * scale
#             if USE_RK4 and HYETO in ("tri", "box"):
#                 Hf, Hm = rk4_integrate_arrays(K, alpha, Sy_f, Sy_m, P_mag, P_dur, t0_storm, TRI_PEAK_FRAC,
#                                               Q_BASE, K_LIN, is_tri, DT, N_STEPS, 0.1, 0.1)
#             else:
#                 # fallback
#                 sol2 = solve_ivp(universal_model, (0.0, T_END), (0.1, 0.1), t_eval=t_eval,
#                                 args=(K, alpha2, Sy_f, Sy_m, P_mag, P_dur), rtol=RTOL, atol=ATOL)
#                 if sol2.status < 0 or sol2.y.shape[1] != t_eval.size:
#                     return None
#                 Hf, Hm = sol2.y[0], sol2.y[1]

#             if sol2.status >= 0 and sol2.y.shape[1] == t_eval.size:
#                 H_peak2 = float(np.max(sol2.y[0]))
#                 if np.isfinite(H_peak2) and H_peak2 > 0.0:
#                     alpha, logA = alpha2, float(np.log10(alpha2))
#                     Hf, Hm, H_peak = sol2.y[0], sol2.y[1], H_peak2

#     # Compute dimensionless groups
#     Qout_peak = Q_BASE + K_LIN * H_peak + K * H_peak * H_peak
#     if Qout_peak <= 0:
#         return None
#     DaR = (alpha * H_peak) / (Qout_peak + 1e-30)
#     Da = DaR / (Sy_m / Sy_f)

#     # Hysteresis + diagnostics
#     HI, rawA, xspan, yspan, _, _ = hysteresis_index_Qspace(
#         t_eval, Hf, Hm, K, alpha, Sy_f, Sy_m, P_vec, t_storm_end)
#     if HI <= 0.0 or not np.isfinite(HI):
#         return None

#     mech = mechanism_diagnostics(t_eval, Hf, Hm, K, alpha, Sy_f, Sy_m, P_vec, t_storm_end)
#     return {
#         "Contrast": float(contrast_val), "Da": float(Da), "DaR": float(DaR),
#         "Hysteresis": float(HI), "H_peak": float(H_peak),
#         "log10K": logK, "log10alpha": logA,
#         "area_raw": float(rawA), "xspan_logQ": xspan, "yspan_logrQ": yspan,
#         **mech
#     }


def rk4_integrate_driver(K, alpha, Sy_f, Sy_m, is_tri,
                         dt=DT, n_steps=N_STEPS,
                         Hf0=0.1, Hm0=0.1):
    """Dispatch to the configured RK4 engine.

    - If USE_EXCHANGE_EXACT: adaptive substeps based on outflow stiffness + exact exchange (Strang/Lie split)
    - Else: legacy adaptive RK4 with combined exchange+outflow stiffness estimate
    """
    if USE_EXCHANGE_EXACT:
        use_strang = 1 if str(EXCHANGE_SPLIT).lower().startswith("strang") else 0
        return rk4_integrate_arrays_adaptive_exchange_exact(
            K, alpha, Sy_f, Sy_m, P_mag, P_dur, t0_storm, TRI_PEAK_FRAC,
            Q_BASE, K_LIN, is_tri,
            dt, n_steps, SUBSTEP_TARGET, NSUB_MAX,
            Hf0, Hm0, H_MAX, use_strang
        )
    else:
        return rk4_integrate_arrays_adaptive(
            K, alpha, Sy_f, Sy_m, P_mag, P_dur, t0_storm, TRI_PEAK_FRAC,
            Q_BASE, K_LIN, is_tri,
            dt, n_steps, SUBSTEP_TARGET, NSUB_MAX,
            Hf0, Hm0, H_MAX
        )


def _run_chunk(chunk_args):
    """
    chunk_args: list of tuples (contrast_val, K, alpha, logK, logA, DaR_target, Sy_m)
    Returns list of result dicts (no None).
    Uses global P_VEC_GLOBAL.
    """
    out = []
    P_vec = P_VEC_GLOBAL
    is_tri = (HYETO == "tri")

    for (contrast_val, K, alpha, logK, logA, DaR_target, Sy_m) in chunk_args:
        # --- integrate ---
        nsub_max = np.nan

        if USE_RK4 and (not FULL_IVP_MODE) and HYETO in ("tri", "box"):
            Hf, Hm, ok, nsub_max_i = rk4_integrate_driver(K, alpha, Sy_f, Sy_m, is_tri, DT, N_STEPS, 0.1, 0.1)
            if not ok:
                continue
            nsub_max = float(nsub_max_i)
        else:
            sol = solve_ivp(universal_model, (0.0, T_END), (0.1, 0.1), t_eval=t_eval,
                            args=(K, alpha, Sy_f, Sy_m, P_mag, P_dur),
                            rtol=RTOL, atol=ATOL)
            if sol.status < 0 or sol.y.shape[1] != t_eval.size:
                continue
            Hf, Hm = sol.y[0], sol.y[1]

        H_peak = float(np.max(Hf))
        if not np.isfinite(H_peak) or H_peak <= 0.0:
            continue

        # --- optional 1-step DaR refinement ---
        if USE_DAR_SAMPLING and REFINE_DAR_ONCE and np.isfinite(DaR_target):
            Qout_pk = Q_BASE + K_LIN * H_peak + K * H_peak * H_peak
            if Qout_pk > 0.0:
                DaR_ach = (alpha * H_peak) / (Qout_pk + 1e-30)
                scale = np.clip(DaR_target / (DaR_ach + 1e-30), 1e-3, 1e3)
                alpha2 = alpha * scale

                if USE_RK4 and (not FULL_IVP_MODE) and HYETO in ("tri", "box"):
                    Hf2, Hm2, ok2, nsub_max2 = rk4_integrate_driver(K, alpha2, Sy_f, Sy_m, is_tri, DT, N_STEPS, 0.1, 0.1)
                    if not ok2:
                        Hf2, Hm2 = None, None
                else:
                    sol2 = solve_ivp(universal_model, (0.0, T_END), (0.1, 0.1), t_eval=t_eval,
                                     args=(K, alpha2, Sy_f, Sy_m, P_mag, P_dur),
                                     rtol=RTOL, atol=ATOL)
                    if sol2.status < 0 or sol2.y.shape[1] != t_eval.size:
                        Hf2, Hm2 = None, None
                    else:
                        Hf2, Hm2 = sol2.y[0], sol2.y[1]
                        nsub_max2 = np.nan

                if Hf2 is not None:
                    H_peak2 = float(np.max(Hf2))
                    if np.isfinite(H_peak2) and H_peak2 > 0.0:
                        alpha = alpha2
                        logA = float(np.log10(alpha2))
                        Hf, Hm, H_peak = Hf2, Hm2, H_peak2
                        if np.isfinite(nsub_max2):
                            nsub_max = float(nsub_max2)

        # --- selective fallback to solve_ivp (two-stage) ---
        # Stage 1 (hard): extremely stiff outflow cases (large nsub_max) -> replace with Radau.
        # Stage 2 (soft): high-R, moderate Da candidates where operator splitting can create spurious hysteresis -> validate with Radau.
        fallback_used = False
        fallback_reason = ""
        HI_rk4_pre = np.nan
        HI_ivp_check = np.nan
        abs_dHI_check = np.nan

        # quick achieved Da at peak (used for soft-trigger gating)
        Qout_pk = Q_BASE + K_LIN * H_peak + K * H_peak * H_peak
        DaR_ach_pk = (alpha * H_peak) / (Qout_pk + 1e-30) if (Qout_pk > 0.0) else np.nan
        Da_ach_pk = DaR_ach_pk / (float(contrast_val) + 1e-30) if np.isfinite(DaR_ach_pk) else np.nan
        hard_trigger = bool(USE_RK4 and (not FULL_IVP_MODE) and FALLBACK_SOLVEIVP and np.isfinite(nsub_max) and (nsub_max >= FALLBACK_NSUB))

        soft_trigger = False  # existing mid-Da hump gate
        if USE_RK4 and (not FULL_IVP_MODE) and FALLBACK_SOLVEIVP and FALLBACK_SOFT_BIMODAL:
            if (float(contrast_val) >= float(FALLBACK_SOFT_R_MIN)
                and np.isfinite(Da_ach_pk)
                and (Da_ach_pk >= float(FALLBACK_SOFT_DA_MIN)) and (Da_ach_pk <= float(FALLBACK_SOFT_DA_MAX))
                and np.isfinite(nsub_max) and (nsub_max <= float(FALLBACK_SOFT_NSUB_MAX))):
                soft_trigger = True
        
        soft_trigger_lowDa = False  # NEW low-Da/high-R gate
        if USE_RK4 and (not FULL_IVP_MODE) and FALLBACK_SOLVEIVP and FALLBACK_SOFT_LOWDA:
            if (float(contrast_val) >= float(FALLBACK_SOFT_LOWDA_R_MIN)
                and np.isfinite(Da_ach_pk)
                and (Da_ach_pk >= float(FALLBACK_SOFT_LOWDA_DA_MIN)) and (Da_ach_pk <= float(FALLBACK_SOFT_LOWDA_DA_MAX))
                and np.isfinite(nsub_max) and (nsub_max <= float(FALLBACK_SOFT_LOWDA_NSUB_MAX))):
                soft_trigger_lowDa = True

        soft_trigger_ridge = False
        if USE_RK4 and (not FULL_IVP_MODE) and FALLBACK_SOLVEIVP and FALLBACK_SOFT_RIDGE:
            if (_ridge_reference_band_hit(float(contrast_val), float(Da_ach_pk), FALLBACK_SOFT_RIDGE_BAND_DECADES)
                and np.isfinite(nsub_max) and (nsub_max <= float(FALLBACK_SOFT_RIDGE_NSUB_MAX))):
                soft_trigger_ridge = True

        soft_trigger_shoulder = False
        if USE_RK4 and (not FULL_IVP_MODE) and FALLBACK_SOLVEIVP and FALLBACK_SOFT_SHOULDER:
            if (_shoulder_band_hit(float(contrast_val), float(Da_ach_pk))
                and np.isfinite(nsub_max) and (nsub_max <= float(FALLBACK_SOFT_SHOULDER_NSUB_MAX))):
                soft_trigger_shoulder = True

        # compute RK4 HI pre-value if we might use it (needed for mid-Da HI threshold only)
        if USE_RK4 and (not FULL_IVP_MODE) and FALLBACK_SOLVEIVP and (hard_trigger or soft_trigger or soft_trigger_lowDa or soft_trigger_ridge or soft_trigger_shoulder):
            try:
                HI_rk4_pre, _, _, _, _, _ = hysteresis_index_Qspace(
                    t_eval, Hf, Hm, K, alpha, Sy_f, Sy_m, P_vec, t_storm_end
                )
            except Exception:
                HI_rk4_pre = np.nan
        
        # apply HI threshold ONLY to the mid-Da hump gate (do NOT apply to low-Da gate)
        if soft_trigger:
            if (not np.isfinite(HI_rk4_pre)) or (HI_rk4_pre < float(FALLBACK_SOFT_HI_MIN)):
                soft_trigger = False

        # hard_trigger = bool(USE_RK4 and FALLBACK_SOLVEIVP and np.isfinite(nsub_max) and (nsub_max >= FALLBACK_NSUB))

        # soft_trigger = False
        # if USE_RK4 and FALLBACK_SOLVEIVP and FALLBACK_SOFT_BIMODAL:
        #     if (float(contrast_val) >= float(FALLBACK_SOFT_R_MIN)
        #         and np.isfinite(Da_ach_pk)
        #         and (Da_ach_pk >= float(FALLBACK_SOFT_DA_MIN)) and (Da_ach_pk <= float(FALLBACK_SOFT_DA_MAX))
        #         and np.isfinite(nsub_max) and (nsub_max <= float(FALLBACK_SOFT_NSUB_MAX))):
        #         soft_trigger = True

        # # compute RK4 HI pre-value if we might use it (for gating / logging)
        # if USE_RK4 and FALLBACK_SOLVEIVP and (hard_trigger or soft_trigger):
        #     try:
        #         HI_rk4_pre, _, _, _, _, _ = hysteresis_index_Qspace(
        #             t_eval, Hf, Hm, K, alpha, Sy_f, Sy_m, P_vec, t_storm_end
        #         )
        #     except Exception:
        #         HI_rk4_pre = np.nan

        # # soft-trigger is only meaningful if RK4 HI is "large enough to matter"
        # if soft_trigger:
        #     if (not np.isfinite(HI_rk4_pre)) or (HI_rk4_pre < float(FALLBACK_SOFT_HI_MIN)):
        #         soft_trigger = False

        # Run Radau if any trigger fires
        if USE_RK4 and (not FULL_IVP_MODE) and FALLBACK_SOLVEIVP and (hard_trigger or soft_trigger or soft_trigger_lowDa or soft_trigger_ridge or soft_trigger_shoulder):
            sol_fb = solve_ivp(
                universal_model, (0.0, T_END), (0.1, 0.1), t_eval=t_eval,
                args=(K, alpha, Sy_f, Sy_m, P_mag, P_dur),
                rtol=RTOL, atol=ATOL, method=FALLBACK_METHOD
            )
            if (sol_fb.status >= 0) and (sol_fb.y.shape[1] == t_eval.size):
                Hf_fb, Hm_fb = sol_fb.y[0], sol_fb.y[1]

                # compute HI on ivp for soft-trigger decision / logging
                try:
                    HI_ivp_check, _, _, _, _, _ = hysteresis_index_Qspace(
                        t_eval, Hf_fb, Hm_fb, K, alpha, Sy_f, Sy_m, P_vec, t_storm_end
                    )
                    if np.isfinite(HI_ivp_check) and np.isfinite(HI_rk4_pre):
                        abs_dHI_check = float(abs(HI_rk4_pre - HI_ivp_check))
                except Exception:
                    HI_ivp_check = np.nan
                    abs_dHI_check = np.nan

                # Hard trigger: always replace
                if hard_trigger:
                    Hf, Hm = Hf_fb, Hm_fb
                    H_peak = float(np.max(Hf))
                    fallback_used = bool(np.isfinite(H_peak) and H_peak > 0.0)
                    fallback_reason = "hard_nsub" if fallback_used else ""

                # Soft trigger: verify / optionally replace
                elif soft_trigger:
                    do_replace = bool(FALLBACK_SOFT_ALWAYS_REPLACE)
                    if (not do_replace) and np.isfinite(abs_dHI_check):
                        do_replace = bool(abs_dHI_check >= float(FALLBACK_SOFT_ABS_DHI_TOL))
                    if do_replace:
                        Hf, Hm = Hf_fb, Hm_fb
                        H_peak = float(np.max(Hf))
                        fallback_used = bool(np.isfinite(H_peak) and H_peak > 0.0)
                        fallback_reason = "soft_bimodal" if fallback_used else ""

                # Soft trigger (low-Da / high-R): verify / optionally replace
                elif soft_trigger_lowDa:
                    do_replace = bool(FALLBACK_SOFT_LOWDA_ALWAYS_REPLACE)
                    if (not do_replace) and np.isfinite(abs_dHI_check):
                        do_replace = bool(abs_dHI_check >= float(FALLBACK_SOFT_ABS_DHI_TOL))
                    if do_replace:
                        Hf, Hm = Hf_fb, Hm_fb
                        H_peak = float(np.max(Hf))
                        fallback_used = bool(np.isfinite(H_peak) and H_peak > 0.0)
                        fallback_reason = "soft_lowDa" if fallback_used else ""

                # Soft trigger (ridge-centered): always verify published ridge neighborhood with IVP truth
                elif soft_trigger_ridge:
                    do_replace = bool(FALLBACK_SOFT_RIDGE_ALWAYS_REPLACE)
                    if (not do_replace) and np.isfinite(abs_dHI_check):
                        do_replace = bool(abs_dHI_check >= float(FALLBACK_SOFT_ABS_DHI_TOL))
                    if do_replace:
                        Hf, Hm = Hf_fb, Hm_fb
                        H_peak = float(np.max(Hf))
                        fallback_used = bool(np.isfinite(H_peak) and H_peak > 0.0)
                        fallback_reason = "soft_ridge" if fallback_used else ""

                # Soft trigger (shoulder donut): verify shoulder zone with IVP truth
                elif soft_trigger_shoulder:
                    do_replace = bool(FALLBACK_SOFT_SHOULDER_ALWAYS_REPLACE)
                    if (not do_replace) and np.isfinite(abs_dHI_check):
                        do_replace = bool(abs_dHI_check >= float(FALLBACK_SOFT_ABS_DHI_TOL))
                    if do_replace:
                        Hf, Hm = Hf_fb, Hm_fb
                        H_peak = float(np.max(Hf))
                        fallback_used = bool(np.isfinite(H_peak) and H_peak > 0.0)
                        fallback_reason = "soft_shoulder" if fallback_used else ""
        # --- groups ---
        Qout_peak = Q_BASE + K_LIN * H_peak + K * H_peak * H_peak
        if Qout_peak <= 0.0:
            continue
        DaR = (alpha * H_peak) / (Qout_peak + 1e-30)
        Da  = DaR / (Sy_m / Sy_f)

        # --- metric + diagnostics ---
        HI, rawA, xspan, yspan, _, _ = hysteresis_index_Qspace(
            t_eval, Hf, Hm, K, alpha, Sy_f, Sy_m, P_vec, t_storm_end
        )
        if not np.isfinite(HI):
            continue

        mech = mechanism_diagnostics(t_eval, Hf, Hm, K, alpha, Sy_f, Sy_m, P_vec, t_storm_end)

        out.append({
            "Contrast": float(contrast_val), "Da": float(Da), "DaR": float(DaR),
            "DaR_target": float(DaR_target) if np.isfinite(DaR_target) else np.nan,
            "Hysteresis": float(HI), "H_peak": float(H_peak),
            "log10K": float(logK), "log10alpha": float(logA),
            "K": float(K), "alpha": float(alpha),
            "nsub_max": float(nsub_max) if np.isfinite(nsub_max) else np.nan,
            "fallback_used": int(fallback_used),
            "fallback_reason": str(fallback_reason) if (fallback_reason is not None) else "",
            "HI_ivp_check": float(HI_ivp_check) if np.isfinite(HI_ivp_check) else np.nan,
            "abs_dHI_check": float(abs_dHI_check) if np.isfinite(abs_dHI_check) else np.nan,
            "Da_ach_pk": float(Da_ach_pk) if np.isfinite(Da_ach_pk) else np.nan,
            "DaR_ach_pk": float(DaR_ach_pk) if np.isfinite(DaR_ach_pk) else np.nan,
            "HI_rk4_pre": float(HI_rk4_pre) if np.isfinite(HI_rk4_pre) else np.nan,
            "area_raw": float(rawA), "xspan_logQ": float(xspan), "yspan_logrQ": float(yspan),
            **mech
        })

    return out

# ==============================================================================
# VALIDATION: RK4 (substepped) vs solve_ivp spot-check
# ==============================================================================

def spotcheck_rk4_vs_solveivp(n=SPOTCHECK_N, seed=SPOTCHECK_SEED, R_levels=SPOTCHECK_R_LEVELS, out_csv=None):
    """
    Quick numerical fidelity check.
    Samples (K, DaR_target) with LHC, converts to alpha using H_REF, then compares:
      - RK4 + per-run substepping (production path) vs solve_ivp (reference)

    Returns a DataFrame of per-case comparisons and writes a CSV in the working directory.
    """
    print("\n=== SPOTCHECK: RK4 vs solve_ivp (subset) ===")
    K_samps, DaR_t_samps, logK_samps, logDaR_samps = sample_lhc_log_targets(n, seed=int(seed))

    P_vec = precip_vec(t_eval, P_mag, P_dur, t0=t0_storm)
    # (threading backend shares memory; this is for consistency with the main run)
    _set_globals_for_workers(P_vec)

    rows = []
    is_tri = (HYETO == "tri")

    for R in R_levels:
        Sy_m = Sy_f * float(R)

        for i in range(n):
            K = float(K_samps[i])
            DaR_target = float(DaR_t_samps[i])

            # sampling transform (same as main)
            Qout_ref = Q_BASE + K_LIN * H_REF + K * H_REF * H_REF
            alpha = DaR_target * Qout_ref / (H_REF + 1e-30)

            # --- RK4 (adaptive substepping: exchange + nonlinear outflow stiffness) ---
            Hf_rk4, Hm_rk4, ok, n_sub = rk4_integrate_driver(K, alpha, Sy_f, Sy_m, is_tri, DT, N_STEPS, 0.1, 0.1)

            if not ok:
                rows.append({
                    "Contrast": float(R), "log10K": float(np.log10(K)), "log10DaR_target": float(np.log10(DaR_target)),
                    "status": "rk4_failed"
                })
                continue

            HI_rk4, _, _, _, _, _ = hysteresis_index_Qspace(
                t_eval, Hf_rk4, Hm_rk4, K, alpha, Sy_f, Sy_m, P_vec, t_storm_end
            )
            Hpk_rk4 = float(np.nanmax(Hf_rk4))
            Qpk_rk4 = Q_BASE + K_LIN * Hpk_rk4 + K * Hpk_rk4 * Hpk_rk4
            DaR_rk4 = (alpha * Hpk_rk4) / (Qpk_rk4 + 1e-30) if (Qpk_rk4 > 0 and np.isfinite(Hpk_rk4)) else np.nan
            Da_rk4 = DaR_rk4 / (Sy_m / Sy_f) if np.isfinite(DaR_rk4) else np.nan

            # --- solve_ivp (reference) ---
            sol = solve_ivp(
                universal_model, (0.0, T_END), (0.1, 0.1), t_eval=t_eval,
                args=(K, alpha, Sy_f, Sy_m, P_mag, P_dur),
                rtol=RTOL, atol=ATOL, method="Radau"
            )
            if sol.status < 0 or sol.y.shape[1] != t_eval.size:
                rows.append({
                    "Contrast": float(R), "log10K": float(np.log10(K)), "log10DaR_target": float(np.log10(DaR_target)),
                    "status": "ivp_failed"
                })
                continue

            Hf_ivp, Hm_ivp = sol.y[0], sol.y[1]
            HI_ivp, _, _, _, _, _ = hysteresis_index_Qspace(
                t_eval, Hf_ivp, Hm_ivp, K, alpha, Sy_f, Sy_m, P_vec, t_storm_end
            )
            Hpk_ivp = float(np.nanmax(Hf_ivp))
            Qpk_ivp = Q_BASE + K_LIN * Hpk_ivp + K * Hpk_ivp * Hpk_ivp
            DaR_ivp = (alpha * Hpk_ivp) / (Qpk_ivp + 1e-30) if (Qpk_ivp > 0 and np.isfinite(Hpk_ivp)) else np.nan
            Da_ivp = DaR_ivp / (Sy_m / Sy_f) if np.isfinite(DaR_ivp) else np.nan

            # --- differences ---
            dHI = (HI_rk4 - HI_ivp) if (np.isfinite(HI_rk4) and np.isfinite(HI_ivp)) else np.nan
            abs_dHI = float(np.abs(dHI)) if np.isfinite(dHI) else np.nan
            denom = max(abs(HI_ivp), HI_REL_FLOOR) if np.isfinite(HI_ivp) else np.nan
            rel_HI = (dHI / denom) if (np.isfinite(dHI) and np.isfinite(denom) and denom > 0) else np.nan
            rel_HI_raw = (dHI / (HI_ivp + 1e-30)) if (np.isfinite(dHI) and np.isfinite(HI_ivp)) else np.nan
            rel_Hpk = (Hpk_rk4 - Hpk_ivp) / (Hpk_ivp + 1e-30) if np.isfinite(Hpk_ivp) else np.nan

            # Would this case be replaced by IVP (soft fallback) in production?
            soft_region = bool(
                (float(R) >= float(FALLBACK_SOFT_R_MIN))
                and np.isfinite(Da_rk4)
                and (Da_rk4 >= float(FALLBACK_SOFT_DA_MIN)) and (Da_rk4 <= float(FALLBACK_SOFT_DA_MAX))
                and (int(n_sub) <= int(FALLBACK_SOFT_NSUB_MAX))
            ) if bool(FALLBACK_SOFT_BIMODAL) else False
            would_soft_trigger = bool(
                soft_region
                and np.isfinite(HI_rk4) and (HI_rk4 >= float(FALLBACK_SOFT_HI_MIN))
            ) if bool(FALLBACK_SOFT_BIMODAL) else False

            lowda_region = bool(
                bool(FALLBACK_SOFT_LOWDA)
                and (float(R) >= float(FALLBACK_SOFT_LOWDA_R_MIN))
                and np.isfinite(Da_rk4)
                and (Da_rk4 >= float(FALLBACK_SOFT_LOWDA_DA_MIN)) and (Da_rk4 <= float(FALLBACK_SOFT_LOWDA_DA_MAX))
                and (int(n_sub) <= int(FALLBACK_SOFT_LOWDA_NSUB_MAX))
            )
            would_lowda_trigger = bool(lowda_region)  # no HI threshold for low-Da gate

            ridge_region = bool(
                bool(FALLBACK_SOFT_RIDGE)
                and _ridge_reference_band_hit(float(R), float(Da_rk4), FALLBACK_SOFT_RIDGE_BAND_DECADES)
                and (int(n_sub) <= int(FALLBACK_SOFT_RIDGE_NSUB_MAX))
            )
            would_ridge_trigger = bool(ridge_region)

            shoulder_region = bool(
                bool(FALLBACK_SOFT_SHOULDER)
                and _shoulder_band_hit(float(R), float(Da_rk4))
                and (int(n_sub) <= int(FALLBACK_SOFT_SHOULDER_NSUB_MAX))
            )
            would_shoulder_trigger = bool(shoulder_region)

            # Production-equivalent decision: hard fallback if n_sub >= FALLBACK_NSUB; soft replace if any soft gate triggers
            hard_trigger = bool(int(n_sub) >= int(FALLBACK_NSUB))

            soft_replace_mid = bool(
                would_soft_trigger
                and (bool(FALLBACK_SOFT_ALWAYS_REPLACE) or (np.isfinite(abs_dHI) and (abs_dHI >= float(FALLBACK_SOFT_ABS_DHI_TOL))))
            )
            soft_replace_lowDa = bool(would_lowda_trigger and bool(FALLBACK_SOFT_LOWDA_ALWAYS_REPLACE))
            soft_replace_ridge = bool(would_ridge_trigger and bool(FALLBACK_SOFT_RIDGE_ALWAYS_REPLACE))
            soft_replace_shoulder = bool(would_shoulder_trigger and bool(FALLBACK_SOFT_SHOULDER_ALWAYS_REPLACE))
            soft_replace = bool(soft_replace_mid or soft_replace_lowDa or soft_replace_ridge or soft_replace_shoulder)

            HI_prod = float(HI_ivp) if (hard_trigger or soft_replace) else float(HI_rk4)
            dHI_prod = (HI_prod - HI_ivp) if (np.isfinite(HI_prod) and np.isfinite(HI_ivp)) else np.nan
            abs_dHI_prod = float(np.abs(dHI_prod)) if np.isfinite(dHI_prod) else np.nan
            denom_prod = max(abs(HI_ivp), HI_REL_FLOOR) if np.isfinite(HI_ivp) else np.nan
            rel_HI_prod = (dHI_prod / denom_prod) if (np.isfinite(dHI_prod) and np.isfinite(denom_prod) and denom_prod > 0) else np.nan

            rows.append({
                "Contrast": float(R),
                "log10K": float(np.log10(K)),
                "log10DaR_target": float(np.log10(DaR_target)),
                "alpha": float(alpha),
                "n_sub": int(n_sub),
                "HI_rk4": float(HI_rk4),
                "HI_ivp": float(HI_ivp),
                "HI_prod": float(HI_prod) if np.isfinite(HI_prod) else np.nan,
                "abs_dHI_prod": float(abs_dHI_prod) if np.isfinite(abs_dHI_prod) else np.nan,
                "rel_HI_prod": float(rel_HI_prod) if np.isfinite(rel_HI_prod) else np.nan,
                "hard_trigger": int(hard_trigger),
                "soft_replace": int(soft_replace),
                "dHI": float(dHI) if np.isfinite(dHI) else np.nan,
                "abs_dHI": float(abs_dHI) if np.isfinite(abs_dHI) else np.nan,
                "rel_HI_raw": float(rel_HI_raw) if np.isfinite(rel_HI_raw) else np.nan,
                "rel_HI": float(rel_HI) if np.isfinite(rel_HI) else np.nan,
                "Hpk_rk4": float(Hpk_rk4),
                "Hpk_ivp": float(Hpk_ivp),
                "rel_Hpk": float(rel_Hpk) if np.isfinite(rel_Hpk) else np.nan,
                "Da_rk4": float(Da_rk4) if np.isfinite(Da_rk4) else np.nan,
                "Da_ivp": float(Da_ivp) if np.isfinite(Da_ivp) else np.nan,
                "soft_region": int(soft_region),
                "would_soft_trigger": int(would_soft_trigger),
                "lowda_region": int(lowda_region),
                "would_lowda_trigger": int(would_lowda_trigger),
                "ridge_region": int(ridge_region),
                "would_ridge_trigger": int(would_ridge_trigger),
                "shoulder_region": int(shoulder_region),
                "would_shoulder_trigger": int(would_shoulder_trigger),
                "soft_replace_mid": int(soft_replace_mid),
                "soft_replace_lowDa": int(soft_replace_lowDa),
                "soft_replace_ridge": int(soft_replace_ridge),
                "soft_replace_shoulder": int(soft_replace_shoulder),
                "status": "ok"
            })

    dfv = pd.DataFrame(rows)

    ok = dfv["status"] == "ok"
    if ok.any():
        med_abs_dHI = float(np.nanmedian(dfv.loc[ok, "abs_dHI"].to_numpy())) if "abs_dHI" in dfv.columns else np.nan
        p95_abs_dHI = float(np.nanquantile(dfv.loc[ok, "abs_dHI"].to_numpy(), 0.95)) if "abs_dHI" in dfv.columns else np.nan
        med_abs_rel_HI = float(np.nanmedian(np.abs(dfv.loc[ok, "rel_HI"].to_numpy())))
        p95_abs_rel_HI = float(np.nanquantile(np.abs(dfv.loc[ok, "rel_HI"].to_numpy()), 0.95))
        med_abs_rel_Hpk = float(np.nanmedian(np.abs(dfv.loc[ok, "rel_Hpk"].to_numpy())))
        # Production-equivalent errors (after publication-domain replacement decisions)
        med_abs_dHI_prod = float(np.nanmedian(dfv.loc[ok, "abs_dHI_prod"].to_numpy())) if "abs_dHI_prod" in dfv.columns else np.nan
        p95_abs_dHI_prod = float(np.nanquantile(dfv.loc[ok, "abs_dHI_prod"].to_numpy(), 0.95)) if "abs_dHI_prod" in dfv.columns else np.nan
        med_abs_rel_HI_prod = float(np.nanmedian(np.abs(dfv.loc[ok, "rel_HI_prod"].to_numpy()))) if "rel_HI_prod" in dfv.columns else np.nan
        p95_abs_rel_HI_prod = float(np.nanquantile(np.abs(dfv.loc[ok, "rel_HI_prod"].to_numpy()), 0.95)) if "rel_HI_prod" in dfv.columns else np.nan
        print(f"[Spotcheck] ok={ok.sum()}/{len(dfv)} | RK4 med|dHI|={med_abs_dHI:.3e} p95|dHI|={p95_abs_dHI:.3e} | RK4 med|rel_HI(floor={HI_REL_FLOOR:g})|={med_abs_rel_HI:.3e} p95|rel_HI|={p95_abs_rel_HI:.3e} | med|rel_Hpk|={med_abs_rel_Hpk:.3e} | PROD med|dHI|={med_abs_dHI_prod:.3e} p95|dHI|={p95_abs_dHI_prod:.3e} | PROD med|rel_HI|={med_abs_rel_HI_prod:.3e} p95|rel_HI|={p95_abs_rel_HI_prod:.3e}")
        # Worst-case diagnostics (helps interpret whether p95 errors come from a few outliers)
        try:
            sub_ok = dfv.loc[ok].copy()
            sub_ok["abs_rel_HI"] = np.abs(sub_ok["rel_HI"].to_numpy())
            worst = sub_ok.sort_values("abs_dHI", ascending=False).head(5)
            cols = ["Contrast","log10K","log10alpha","DaR_target","Da","DaR","log10DaR_target","Da_ivp","Da_rk4","HI_ivp","HI_rk4","HI_prod","abs_dHI","abs_dHI_prod","rel_HI","rel_HI_prod","hard_trigger","soft_replace","soft_region","would_soft_trigger","n_sub"]
            cols = [c for c in cols if c in worst.columns]
            print("[Spotcheck worst |dHI|] (top 5)")
            print(worst[cols].to_string(index=False))
        except Exception as e:
            print(f"[Spotcheck] could not summarize worst cases ({e})")

    else:
        print("[Spotcheck] No successful comparisons (check integrator stability or solve_ivp tolerances).")

    if out_csv is None:
        out_csv = "spotcheck_rk4_vs_solveivp.csv"
    try:
        dfv.to_csv(out_csv, index=False)
        print(f"[Spotcheck] wrote {out_csv}")
    except Exception as e:
        print(f"[Spotcheck] could not write CSV ({e})")

    return dfv


def spotcheck_saturated_vs_unsat(
    df_runs: pd.DataFrame,
    n_sat: int = SAT_SPOTCHECK_N_SAT,
    n_unsat: int = SAT_SPOTCHECK_N_UNSAT,
    seed: int = SAT_SPOTCHECK_SEED,
    out_csv: str = None
):
    """Targeted numerical check on the hardest cases: substep-saturated vs non-saturated runs.

    Upgrade (publication rigor):
      - Reports BOTH RK4-vs-IVP errors and PROD-vs-IVP errors, where "PROD" means the
        publication-domain solver policy outcome (i.e., cases that would be replaced by solve_ivp
        in production are treated as IVP-backed).
    """
    if df_runs is None or df_runs.empty:
        return None

    rng = np.random.default_rng(seed)

    # Identify saturated vs unsaturated (based on recorded nsub_max from production RK4 attempt)
    if "nsub_max" not in df_runs.columns:
        print("[SatSpotcheck] missing nsub_max column; skipping.")
        return None

    sat_mask = np.isfinite(df_runs["nsub_max"].to_numpy(dtype=float)) & (df_runs["nsub_max"].to_numpy(dtype=float) >= float(NSUB_MAX))
    df_sat = df_runs[sat_mask].copy()
    df_unsat = df_runs[~sat_mask].copy()

    if df_sat.empty or df_unsat.empty:
        print(f"[SatSpotcheck] insufficient groups: sat={len(df_sat)}, unsat={len(df_unsat)}; skipping.")
        return None

    # Sample
    df_sat_s = df_sat.sample(n=min(int(n_sat), len(df_sat)), random_state=int(seed))
    df_uns_s = df_unsat.sample(n=min(int(n_unsat), len(df_unsat)), random_state=int(seed + 1))

    P_vec = precip_vec(t_eval, P_mag, P_dur, t0=t0_storm)
    is_tri = (HYETO == "tri")

    rows = []
    for group, subdf in [("sat", df_sat_s), ("unsat", df_uns_s)]:
        for _, row in subdf.iterrows():
            try:
                R = float(row["Contrast"])
                K = float(row["K"])
                alpha = float(row["alpha"])
                logK = float(row.get("log10K", np.log10(K)))
                logA = float(row.get("log10alpha", np.log10(alpha)))
                n_sub_prod = int(row.get("nsub_max", np.nan)) if np.isfinite(row.get("nsub_max", np.nan)) else -1

                Sy_m = Sy_f * R

                # --- RK4 (production fast integrator) ---
                Hf, Hm, ok_rk4, n_sub = rk4_integrate_driver(
                    K, alpha, Sy_f, Sy_m, is_tri,
                    DT, N_STEPS, 0.1, 0.1
                )
                if not ok_rk4:
                    rows.append({
                        "group": group,
                        "Contrast": R,
                        "log10K": logK,
                        "log10alpha": logA,
                        "alpha": float(alpha),
                        "status": "rk4_failed",
                    })
                    continue
                HI_rk4, _, _, _, _, _ = hysteresis_index_Qspace(
                    t_eval, Hf, Hm, K, alpha, Sy_f, Sy_m, P_vec, t_storm_end
                )
                Hpk_rk4 = float(np.nanmax(Hf))

                Qpk_rk4 = Q_BASE + K_LIN * Hpk_rk4 + K * Hpk_rk4 * Hpk_rk4
                DaR_rk4 = (alpha * Hpk_rk4) / (Qpk_rk4 + 1e-30) if (Qpk_rk4 > 0 and np.isfinite(Hpk_rk4)) else np.nan
                Da_rk4 = DaR_rk4 / (R + 1e-30) if np.isfinite(DaR_rk4) else np.nan

                # --- IVP truth ---
                sol = solve_ivp(
                    universal_model,
                    (0.0, T_END),
                    (0.1, 0.1),
                    t_eval=t_eval,
                    args=(K, alpha, Sy_f, Sy_m, P_mag, P_dur),
                    method="Radau",
                    rtol=RTOL,
                    atol=ATOL,
                )
                if sol.status < 0 or sol.y.shape[1] != t_eval.size:
                    rows.append({
                        "group": group,
                        "Contrast": R,
                        "log10K": logK,
                        "log10alpha": logA,
                        "status": "ivp_failed",
                    })
                    continue

                Hf_ivp, Hm_ivp = sol.y[0], sol.y[1]
                HI_ivp, _, _, _, _, _ = hysteresis_index_Qspace(
                    t_eval, Hf_ivp, Hm_ivp, K, alpha, Sy_f, Sy_m, P_vec, t_storm_end
                )
                Hpk_ivp = float(np.nanmax(Hf_ivp))

                # --- RK4 vs IVP errors ---
                dHI = (HI_rk4 - HI_ivp) if (np.isfinite(HI_rk4) and np.isfinite(HI_ivp)) else np.nan
                abs_dHI = float(np.abs(dHI)) if np.isfinite(dHI) else np.nan
                denom = max(abs(HI_ivp), HI_REL_FLOOR) if np.isfinite(HI_ivp) else np.nan
                rel_HI = (dHI / denom) if (np.isfinite(dHI) and np.isfinite(denom) and denom > 0) else np.nan
                rel_Hpk = (Hpk_rk4 - Hpk_ivp) / (Hpk_ivp + 1e-30) if np.isfinite(Hpk_ivp) else np.nan

                # --- Production-equivalent (publication-domain) decision ---
                mid_region = bool(
                    bool(FALLBACK_SOFT_BIMODAL)
                    and (float(R) >= float(FALLBACK_SOFT_R_MIN))
                    and np.isfinite(Da_rk4)
                    and (Da_rk4 >= float(FALLBACK_SOFT_DA_MIN)) and (Da_rk4 <= float(FALLBACK_SOFT_DA_MAX))
                    and (int(n_sub) <= int(FALLBACK_SOFT_NSUB_MAX))
                )
                would_soft_trigger = bool(
                    mid_region
                    and np.isfinite(HI_rk4) and (HI_rk4 >= float(FALLBACK_SOFT_HI_MIN))
                ) if bool(FALLBACK_SOFT_BIMODAL) else False

                lowda_region = bool(
                    bool(FALLBACK_SOFT_LOWDA)
                    and (float(R) >= float(FALLBACK_SOFT_LOWDA_R_MIN))
                    and np.isfinite(Da_rk4)
                    and (Da_rk4 >= float(FALLBACK_SOFT_LOWDA_DA_MIN)) and (Da_rk4 <= float(FALLBACK_SOFT_LOWDA_DA_MAX))
                    and (int(n_sub) <= int(FALLBACK_SOFT_LOWDA_NSUB_MAX))
                )
                would_lowda_trigger = bool(lowda_region)

                ridge_region = bool(
                    bool(FALLBACK_SOFT_RIDGE)
                    and _ridge_reference_band_hit(float(R), float(Da_rk4), FALLBACK_SOFT_RIDGE_BAND_DECADES)
                    and (int(n_sub) <= int(FALLBACK_SOFT_RIDGE_NSUB_MAX))
                )
                would_ridge_trigger = bool(ridge_region)

                shoulder_region = bool(
                    bool(FALLBACK_SOFT_SHOULDER)
                    and _shoulder_band_hit(float(R), float(Da_rk4))
                    and (int(n_sub) <= int(FALLBACK_SOFT_SHOULDER_NSUB_MAX))
                )
                would_shoulder_trigger = bool(shoulder_region)

                hard_trigger = bool(int(n_sub) >= int(FALLBACK_NSUB))

                soft_replace_mid = bool(
                    would_soft_trigger
                    and (bool(FALLBACK_SOFT_ALWAYS_REPLACE) or (np.isfinite(abs_dHI) and (abs_dHI >= float(FALLBACK_SOFT_ABS_DHI_TOL))))
                )
                soft_replace_lowDa = bool(would_lowda_trigger and bool(FALLBACK_SOFT_LOWDA_ALWAYS_REPLACE))
                soft_replace_ridge = bool(would_ridge_trigger and bool(FALLBACK_SOFT_RIDGE_ALWAYS_REPLACE))
                soft_replace_shoulder = bool(would_shoulder_trigger and bool(FALLBACK_SOFT_SHOULDER_ALWAYS_REPLACE))
                soft_replace = bool(soft_replace_mid or soft_replace_lowDa or soft_replace_ridge or soft_replace_shoulder)

                HI_prod = float(HI_ivp) if (hard_trigger or soft_replace) else float(HI_rk4)
                dHI_prod = (HI_prod - HI_ivp) if (np.isfinite(HI_prod) and np.isfinite(HI_ivp)) else np.nan
                abs_dHI_prod = float(np.abs(dHI_prod)) if np.isfinite(dHI_prod) else np.nan
                denom_prod = max(abs(HI_ivp), HI_REL_FLOOR) if np.isfinite(HI_ivp) else np.nan
                rel_HI_prod = (dHI_prod / denom_prod) if (np.isfinite(dHI_prod) and np.isfinite(denom_prod) and denom_prod > 0) else np.nan

                Hpk_prod = float(Hpk_ivp) if (hard_trigger or soft_replace) else float(Hpk_rk4)
                rel_Hpk_prod = (Hpk_prod - Hpk_ivp) / (Hpk_ivp + 1e-30) if np.isfinite(Hpk_ivp) else np.nan

                rows.append({
                    "group": group,
                    "Contrast": float(R),
                    "log10K": float(logK),
                    "log10alpha": float(logA),
                    "alpha": float(alpha),
                    "n_sub": int(n_sub),
                    "nsub_max_prod": int(n_sub_prod),

                    "Da_rk4": float(Da_rk4) if np.isfinite(Da_rk4) else np.nan,

                    "HI_rk4": float(HI_rk4) if np.isfinite(HI_rk4) else np.nan,
                    "HI_ivp": float(HI_ivp) if np.isfinite(HI_ivp) else np.nan,
                    "HI_prod": float(HI_prod) if np.isfinite(HI_prod) else np.nan,

                    "abs_dHI": float(abs_dHI) if np.isfinite(abs_dHI) else np.nan,
                    "rel_HI": float(rel_HI) if np.isfinite(rel_HI) else np.nan,

                    "abs_dHI_prod": float(abs_dHI_prod) if np.isfinite(abs_dHI_prod) else np.nan,
                    "rel_HI_prod": float(rel_HI_prod) if np.isfinite(rel_HI_prod) else np.nan,

                    "Hpk_rk4": float(Hpk_rk4) if np.isfinite(Hpk_rk4) else np.nan,
                    "Hpk_ivp": float(Hpk_ivp) if np.isfinite(Hpk_ivp) else np.nan,
                    "Hpk_prod": float(Hpk_prod) if np.isfinite(Hpk_prod) else np.nan,

                    "rel_Hpk": float(rel_Hpk) if np.isfinite(rel_Hpk) else np.nan,
                    "rel_Hpk_prod": float(rel_Hpk_prod) if np.isfinite(rel_Hpk_prod) else np.nan,

                    "hard_trigger": int(hard_trigger),
                    "mid_region": int(mid_region),
                    "lowda_region": int(lowda_region),
                    "ridge_region": int(ridge_region),
                    "shoulder_region": int(shoulder_region),
                    "soft_trigger_mid": int(would_soft_trigger),
                    "soft_replace_mid": int(soft_replace_mid),
                    "soft_replace_lowDa": int(soft_replace_lowDa),
                    "soft_replace_ridge": int(soft_replace_ridge),
                    "soft_replace_shoulder": int(soft_replace_shoulder),
                    "prod_replaced": int(bool(hard_trigger or soft_replace)),
                    "status": "ok",
                })

            except Exception as e:
                rows.append({
                    "group": group,
                    "Contrast": float(row.get("Contrast", np.nan)),
                    "log10K": float(row.get("log10K", np.nan)),
                    "log10alpha": float(row.get("log10alpha", np.nan)),
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "status": "error",
                })

    out = pd.DataFrame(rows)

    # Print concise summary (RK4 vs IVP AND PROD vs IVP)
    try:
        for g in ["sat", "unsat"]:
            sub = out[out["group"] == g]
            if sub.empty:
                continue

            med_abs = float(np.nanmedian(sub["abs_dHI"].to_numpy(dtype=float)))
            p95_abs = float(np.nanpercentile(sub["abs_dHI"].to_numpy(dtype=float), 95))
            med_abs_prod = float(np.nanmedian(sub["abs_dHI_prod"].to_numpy(dtype=float)))
            p95_abs_prod = float(np.nanpercentile(sub["abs_dHI_prod"].to_numpy(dtype=float), 95))

            med_rhpk = float(np.nanmedian(sub["rel_Hpk"].to_numpy(dtype=float)))
            p95_rhpk = float(np.nanpercentile(sub["rel_Hpk"].to_numpy(dtype=float), 95))
            med_rhpk_prod = float(np.nanmedian(sub["rel_Hpk_prod"].to_numpy(dtype=float)))
            p95_rhpk_prod = float(np.nanpercentile(sub["rel_Hpk_prod"].to_numpy(dtype=float), 95))

            print(f"[SatSpotcheck:{g}] RK4-vs-IVP: med|dHI|={med_abs:.3e}, p95|dHI|={p95_abs:.3e} | "
                  f"med rel(Hpk)={med_rhpk:.3e}, p95 rel(Hpk)={p95_rhpk:.3e}")
            print(f"[SatSpotcheck:{g}] PROD-vs-IVP: med|dHI|={med_abs_prod:.3e}, p95|dHI|={p95_abs_prod:.3e} | "
                  f"med rel(Hpk)={med_rhpk_prod:.3e}, p95 rel(Hpk)={p95_rhpk_prod:.3e}")
    except Exception:
        pass

    if out_csv:
        try:
            out.to_csv(out_csv, index=False)
        except Exception:
            pass

    return out




def _build_run_config_dict(run_tag: str, out_dir: str):
    """Collect key run settings into a JSON-serializable dict."""
    return {
        "run_tag": run_tag,
        "out_dir": out_dir,
        "timestamp": datetime.now().isoformat(),
        "n_lhc_per_contrast": int(n_lhc_per_contrast),
        "n_contrasts": int(len(Contrast_range)),
        "Contrast_range": [float(x) for x in Contrast_range],
        "Sy_f": float(Sy_f),
        "HYETO": str(HYETO),
        "P_mag": float(P_mag),
        "P_dur": float(P_dur),
        "t0_storm": float(t0_storm),
        "t_storm_end": float(t_storm_end),
        "T_END": float(T_END),
        "N_T": int(N_T),
        "DT": float(DT),
        "USE_DAR_SAMPLING": bool(USE_DAR_SAMPLING),
        "REFINE_DAR_ONCE": bool(REFINE_DAR_ONCE),
        "H_REF": float(H_REF),
        "bounds": {
            "LOG10K_MIN": float(LOG10K_MIN),
            "LOG10K_MAX": float(LOG10K_MAX),
            "LOG10_DAR_MIN": float(LOG10_DAR_MIN),
            "LOG10_DAR_MAX": float(LOG10_DAR_MAX),
        },
        "integrator": {
            "USE_RK4": bool(USE_RK4),
            "SUBSTEP_TARGET": float(SUBSTEP_TARGET),
            "NSUB_MAX": int(NSUB_MAX),
            "H_MAX": float(H_MAX),
            "RTOL": float(RTOL),
            "ATOL": float(ATOL),
        },
        "HI_guards": {
            "Q_CUTOFF_FRAC": float(Q_CUTOFF_FRAC),
            "RQ_CUTOFF_FRAC": float(RQ_CUTOFF_FRAC),
            "SPAN_MIN_DECADES": float(SPAN_MIN_DECADES),
            "HI_CLIP_MAX": float(HI_CLIP_MAX),
        },
        "peak_detection": {
            "N_BINS_PEAK": int(N_BINS_PEAK),
            "SMOOTH_SIGMA": float(SMOOTH_SIGMA),
            "SMOOTH_SIGMA_SCALE_WITH_NBINS": bool(SMOOTH_SIGMA_SCALE_WITH_NBINS),
            "Q_LOW": float(Q_LOW),
            "Q_HIGH": float(Q_HIGH),
            "NMIN_PER_BIN": int(NMIN_PER_BIN),
            "MIN_VALID_BINS_FOR_PEAK": int(MIN_VALID_BINS_FOR_PEAK),
            "PEAK_REFINE": bool(PEAK_REFINE),
            "PEAK_REFINE_N_FINE": int(PEAK_REFINE_N_FINE),
            "PEAK_REFINE_SMOOTH_DECADES": float(PEAK_REFINE_SMOOTH_DECADES),
            "PEAK_WIDTH_FRAC": float(PEAK_WIDTH_FRAC),
            "PEAK_WIDTH_FRACS": [float(x) for x in PEAK_WIDTH_FRACS],
            "PEAK_WIDTH_INTERP": bool(PEAK_WIDTH_INTERP),
            "RUN_RIDGE_SENSITIVITY": bool(RUN_RIDGE_SENSITIVITY),
            "SENS_NBINS_LIST": [int(x) for x in SENS_NBINS_LIST],
            "SENS_SIGMA_LIST": [float(x) for x in SENS_SIGMA_LIST],
        },
        "fallback_policy": {
            "FALLBACK_SOLVEIVP": bool(FALLBACK_SOLVEIVP),
            "FALLBACK_METHOD": str(FALLBACK_METHOD),
            "FALLBACK_NSUB": int(FALLBACK_NSUB),
            "FALLBACK_SOFT_BIMODAL": bool(FALLBACK_SOFT_BIMODAL),
            "FALLBACK_SOFT_R_MIN": float(FALLBACK_SOFT_R_MIN),
            "FALLBACK_SOFT_DA_MIN": float(FALLBACK_SOFT_DA_MIN),
            "FALLBACK_SOFT_DA_MAX": float(FALLBACK_SOFT_DA_MAX),
            "FALLBACK_SOFT_HI_MIN": float(FALLBACK_SOFT_HI_MIN),
            "FALLBACK_SOFT_NSUB_MAX": int(FALLBACK_SOFT_NSUB_MAX),
            "FALLBACK_SOFT_ABS_DHI_TOL": float(FALLBACK_SOFT_ABS_DHI_TOL),
            "FALLBACK_SOFT_ALWAYS_REPLACE": bool(FALLBACK_SOFT_ALWAYS_REPLACE),
            "FALLBACK_SOFT_LOWDA": bool(FALLBACK_SOFT_LOWDA),
            "FALLBACK_SOFT_LOWDA_R_MIN": float(FALLBACK_SOFT_LOWDA_R_MIN),
            "FALLBACK_SOFT_LOWDA_DA_MIN": float(FALLBACK_SOFT_LOWDA_DA_MIN),
            "FALLBACK_SOFT_LOWDA_DA_MAX": float(FALLBACK_SOFT_LOWDA_DA_MAX),
            "FALLBACK_SOFT_LOWDA_NSUB_MAX": int(FALLBACK_SOFT_LOWDA_NSUB_MAX),
            "FALLBACK_SOFT_LOWDA_ALWAYS_REPLACE": bool(FALLBACK_SOFT_LOWDA_ALWAYS_REPLACE),
            "FALLBACK_SOFT_RIDGE": bool(FALLBACK_SOFT_RIDGE),
            "FALLBACK_SOFT_RIDGE_BAND_DECADES": float(FALLBACK_SOFT_RIDGE_BAND_DECADES),
            "FALLBACK_SOFT_RIDGE_MIN_BAND_DECADES": float(FALLBACK_SOFT_RIDGE_MIN_BAND_DECADES),
            "FALLBACK_SOFT_RIDGE_WIDTH_FRAC": float(FALLBACK_SOFT_RIDGE_WIDTH_FRAC),
            "FALLBACK_SOFT_RIDGE_WIDTH_BUFFER_DECADES": float(FALLBACK_SOFT_RIDGE_WIDTH_BUFFER_DECADES),
            "FALLBACK_SOFT_RIDGE_NSUB_MAX": int(FALLBACK_SOFT_RIDGE_NSUB_MAX),
            "FALLBACK_SOFT_RIDGE_ALWAYS_REPLACE": bool(FALLBACK_SOFT_RIDGE_ALWAYS_REPLACE),
            "FALLBACK_SOFT_RIDGE_REFERENCE_CSV": str(FALLBACK_SOFT_RIDGE_REFERENCE_CSV),
            "FALLBACK_SOFT_RIDGE_REFERENCE_CSV_FALLBACKS": [str(x) for x in FALLBACK_SOFT_RIDGE_REFERENCE_CSV_FALLBACKS],
            "FALLBACK_SOFT_RIDGE_REFERENCE_CSV_RESOLVED": str(FALLBACK_SOFT_RIDGE_REFERENCE_CSV_RESOLVED),
            "FALLBACK_SOFT_RIDGE_REFERENCE_COL": str(FALLBACK_SOFT_RIDGE_REFERENCE_COL),
            "FALLBACK_SOFT_RIDGE_WIDTH_COL": str(FALLBACK_SOFT_RIDGE_WIDTH_COL),
            "DA_LOBE_SPLIT": float(DA_LOBE_SPLIT),
            "LOWDA_GATE_BUFFER_DECADES": float(LOWDA_GATE_BUFFER_DECADES),
            "DA_MID_MIN": float(DA_MID_MIN),
            "DA_MID_MAX": float(DA_MID_MAX),
            "MID_DA_RATIO_THRESH": float(MID_DA_RATIO_THRESH),
        },
        "bootstrap": {
            "BOOTSTRAP": bool(BOOTSTRAP),
            "N_BOOT": int(N_BOOT),
            "BOOT_SEED": int(BOOT_SEED),
            "RIDGE_CI_Q": list(RIDGE_CI_Q),
            "RIDGE_CI_LOGSPACE": bool(RIDGE_CI_LOGSPACE),
        },
        "parallel": {
            "USE_PARALLEL": bool(USE_PARALLEL),
            "PAR_BACKEND": str(PAR_BACKEND),
            "CHUNK_SIZE": int(CHUNK_SIZE),
            "N_JOBS": int(N_JOBS),
        }
    }


def _fmt_sci(x, prec=3):
    if x is None:
        return "nan"
    try:
        x = float(x)
    except Exception:
        return "nan"
    return f"{x:.{prec}e}" if np.isfinite(x) else "nan"


def _compact_ridge_preview(ridge: pd.DataFrame, max_rows: int = 9) -> str:
    if ridge is None or ridge.empty:
        return "(ridge is empty)"

    r = ridge.sort_values("Contrast").reset_index(drop=True)

    n = len(r)
    if n <= max_rows:
        pick = list(range(n))
    else:
        mid = n // 2
        pick = sorted(set([0, 1, 2, mid - 1, mid, mid + 1, n - 3, n - 2, n - 1]))
        pick = [i for i in pick if 0 <= i < n]

    cols = [
        "Contrast",
        "Da_peak", "Da_peak_lo", "Da_peak_hi", "Da_peak_primary_lowDa", "Da_peak_primary_lowDa_lo", "Da_peak_primary_lowDa_hi", "Da_centroid_f095",
        "HI_peak_curve",
        "Da_low_lobe", "HI_low_lobe", "Da_high_lobe", "HI_high_lobe",
        "cross_bimodal", "cross_ratio", "cross_sep_decades",
        "bimodal_flag", "multi_peak_flag", "n_peaks", "peak_sep_decades", "secondary_ratio",
        "width_decades_f098", "width_decades_f095",
        "DaR_peak", "DaR_peak_lo", "DaR_peak_hi", "DaR_centroid_f095",
        "boot_kept_Da", "boot_kept_DaR",
        "boot_hitL_Da_frac", "boot_hitR_Da_frac",
    ]
    cols = [c for c in cols if c in r.columns]
    rp = r.loc[pick, cols].copy()

    # formatted printing
    fmt = {}
    for c in rp.columns:
        if c in ("Contrast",):
            fmt[c] = lambda v: f"{float(v):.1f}"
        elif c.startswith("boot_") and c.endswith("_frac"):
            fmt[c] = lambda v: f"{float(v):.2f}" if np.isfinite(v) else "nan"
        elif c in ("boot_kept_Da", "boot_kept_DaR"):
            fmt[c] = lambda v: f"{int(v)}"
        elif c in ("HI_peak_curve", "width_decades_f098", "width_decades_f095"):
            fmt[c] = lambda v: f"{float(v):.3f}" if np.isfinite(v) else "nan"
        else:
            fmt[c] = lambda v: _fmt_sci(v, prec=3)

    return rp.to_string(index=False, formatters=fmt)


def _build_progress_summary(df: pd.DataFrame, ridge: pd.DataFrame, attempted: int,
                           spot_df: pd.DataFrame, sat_spot_df: pd.DataFrame,
                           out_dir: str, sens_info: dict = None) -> str:
    lines = []
    acc = int(len(df)) if df is not None else 0
    attempted = int(attempted)

    lines.append(f"attempted={attempted} accepted={acc} acceptance={acc/(attempted+1e-30):.3f}")
    # integrator mode (compact)
    if FULL_IVP_MODE:
        lines.append("integrator=FULL_IVP_MODE (solve_ivp Radau for all realizations)")
    elif USE_RK4:
        if USE_EXCHANGE_EXACT:
            lines.append(f"integrator=rk4_exchange_exact({EXCHANGE_SPLIT}) dt={DT:.4g} n_steps={N_STEPS} substep_target={SUBSTEP_TARGET:g} nsub_max={NSUB_MAX}")
        else:
            lines.append(f"integrator=rk4_classic dt={DT:.4g} n_steps={N_STEPS} substep_target={SUBSTEP_TARGET:g} nsub_max={NSUB_MAX}")
    else:
        lines.append("integrator=solve_ivp")
    if df is not None and not df.empty:
        da = df["Da"].to_numpy()
        dar = df["DaR"].to_numpy()
        hi = df["Hysteresis"].to_numpy()
        lines.append(f"Da  min={_fmt_sci(np.nanmin(da))}  q01={_fmt_sci(np.nanquantile(da,0.01))}  q99={_fmt_sci(np.nanquantile(da,0.99))}  max={_fmt_sci(np.nanmax(da))}")
        lines.append(f"DaR min={_fmt_sci(np.nanmin(dar))} q01={_fmt_sci(np.nanquantile(dar,0.01))} q99={_fmt_sci(np.nanquantile(dar,0.99))} max={_fmt_sci(np.nanmax(dar))}")
        lines.append(f"HI  median={float(np.nanmedian(hi)):.4f}  p90={float(np.nanquantile(hi,0.90)):.4f}  p99={float(np.nanquantile(hi,0.99)):.4f}")

        if "nsub_max" in df.columns:
            ns = df["nsub_max"].to_numpy()
            if np.any(np.isfinite(ns)):
                lines.append(f"nsub_max median={float(np.nanmedian(ns)):.1f} p95={float(np.nanquantile(ns,0.95)):.1f} max={float(np.nanmax(ns)):.1f}")
                sat = float(np.mean(ns >= (NSUB_MAX - 1e-9)))
                lines.append(f"substep_sat frac={sat:.3f} (nsub_max=={NSUB_MAX})")
                try:
                    sat_by_R = df.assign(_sat=df["nsub_max"] >= (NSUB_MAX - 1e-9)).groupby("Contrast")["_sat"].mean()
                    lines.append(f"substep_sat per-R: min={float(sat_by_R.min()):.3f} median={float(sat_by_R.median()):.3f} max={float(sat_by_R.max()):.3f}")
                except Exception:
                    pass


        # fallback stats (if enabled)
        if "fallback_used" in df.columns:
            fb = df["fallback_used"].to_numpy().astype(bool)
            nfb = int(np.sum(fb))
            lines.append(f"fallback solve_ivp: n={nfb} frac={nfb/max(len(df),1):.3f} (hard_nsub>={FALLBACK_NSUB}, soft_bimodal={bool(FALLBACK_SOFT_BIMODAL)}, soft_lowDa={bool(FALLBACK_SOFT_LOWDA)}, soft_ridge={bool(FALLBACK_SOFT_RIDGE)}, soft_shoulder={bool(FALLBACK_SOFT_SHOULDER)})")
            if nfb > 0:
                # breakdown by reason (if available)
                if "fallback_reason" in df.columns:
                    try:
                        r = df.loc[fb, "fallback_reason"].astype(str)
                        n_hard = int(np.sum(r == "hard_nsub"))
                        n_soft = int(np.sum(r == "soft_bimodal"))
                        n_low = int(np.sum(r == "soft_lowDa"))
                        n_ridge = int(np.sum(r == "soft_ridge"))
                        n_shoulder = int(np.sum(r == "soft_shoulder"))
                        lines.append(f"fallback reasons: hard_nsub={n_hard} soft_bimodal={n_soft} soft_lowDa={n_low} soft_ridge={n_ridge} soft_shoulder={n_shoulder}")
                        
                    except Exception:
                        pass
                # how much fallback changed HI (vs the pre-fallback RK4 value)
                if "HI_rk4_pre" in df.columns:
                    try:
                        d = np.abs(df.loc[fb, "Hysteresis"].to_numpy() - df.loc[fb, "HI_rk4_pre"].to_numpy())
                        lines.append(f"fallback ΔHI: med={float(np.nanmedian(d)):.3e} p95={float(np.nanquantile(d,0.95)):.3e}")
                    except Exception:
                        pass
                # diagnostic: absolute HI disagreement computed during the check (if stored)
                if "abs_dHI_check" in df.columns:
                    try:
                        dd = df.loc[fb, "abs_dHI_check"].to_numpy()
                        dd = dd[np.isfinite(dd)]
                        if dd.size > 0:
                            lines.append(f"fallback check |dHI|: med={float(np.nanmedian(dd)):.3e} p95={float(np.nanquantile(dd,0.95)):.3e}")
                    except Exception:
                        pass

        # per-R acceptance distribution
        g = df.groupby("Contrast").size()
        lines.append(f"per-R accepted: min={int(g.min())} median={int(g.median())} max={int(g.max())} over nR={g.size}")

    if ridge is not None and not ridge.empty:
        rr = ridge.sort_values("Contrast")
        # slope of log10(Da_peak) vs log10(R) as a compact diagnostic of leftward shift
        m = np.isfinite(rr["Contrast"]) & np.isfinite(rr["Da_peak"]) & (rr["Contrast"] > 0) & (rr["Da_peak"] > 0)
        if np.count_nonzero(m) >= 3:
            x = np.log10(rr.loc[m, "Contrast"].to_numpy())
            y = np.log10(rr.loc[m, "Da_peak"].to_numpy())
            slope, intercept = np.polyfit(x, y, 1)
            lines.append(f"ridge trend: log10(Da_peak) = {slope:.3f} * log10(R) {intercept:+.3f}")
            # also trend for primary low-Da ridge if present
            if "Da_peak_primary_lowDa" in rr.columns:
                try:
                    m2 = np.isfinite(rr["Da_peak_primary_lowDa"].to_numpy()) & (rr["Da_peak_primary_lowDa"].to_numpy() > 0)
                    if np.count_nonzero(m2) >= 3:
                        x2 = np.log10(rr.loc[m2, "Contrast"].to_numpy())
                        y2 = np.log10(rr.loc[m2, "Da_peak_primary_lowDa"].to_numpy())
                        slope2, intercept2 = np.polyfit(x2, y2, 1)
                        lines.append(f"ridge trend (primary_lowDa): log10(Da_pk_low) = {slope2:.3f} * log10(R) {intercept2:+.3f}")
                except Exception:
                    pass

            # bimodality / multi-peak summary
            try:
                nb = None
                if "bimodal_flag" in rr.columns:
                    # 'bimodal_flag' indicates a separated two-peak structure on the global curve;
                    # those peaks do not necessarily straddle DA_LOBE_SPLIT.
                    nb = int(np.sum(rr["bimodal_flag"].to_numpy().astype(bool)))
                    lines.append(f"separated two-peak rows: {nb}/{len(rr)} (sec>= {BIMODAL_SECONDARY_FRAC:.2f}, sep>= {BIMODAL_MIN_SEP_DECADES:.2f} dec)")

                if "multi_peak_flag" in rr.columns:
                    # generic multi-peak structure (can include within-lobe wiggles)
                    nm = int(np.sum(rr["multi_peak_flag"].to_numpy().astype(bool)))
                    lines.append(f"within-lobe multi-peak flagged: {nm}/{len(rr)} (sec>= {BIMODAL_SECONDARY_FRAC:.2f}, sep>= {BIMODAL_MIN_SEP_DECADES:.2f} dec)")

                if "cross_bimodal" in rr.columns:
                    # low-lobe vs high-lobe max comparison (more physically meaningful than within-lobe wiggles)
                    ncb = int(np.sum(rr["cross_bimodal"].to_numpy().astype(bool)))
                    lines.append(f"cross-lobe (low/high) max flagged: {ncb}/{len(rr)} (split={DA_LOBE_SPLIT:g}, sec>={BIMODAL_SECONDARY_FRAC:.2f})")

                if (nb is not None) and (nb > 0) and ("Da_peak" in rr.columns):
                    high_lobe = int(np.sum((rr["Da_peak"].to_numpy() > DA_LOBE_SPLIT) & rr["bimodal_flag"].to_numpy().astype(bool)))
                    lines.append(f"global ridge on high-Da lobe for {high_lobe}/{nb} separated-two-peak rows (Da>{DA_LOBE_SPLIT:g})")
            except Exception:
                pass

# boundary-hit count
        if "hitL_Da" in rr.columns and "hitR_Da" in rr.columns:
            bh = np.count_nonzero(rr["hitL_Da"].to_numpy() | rr["hitR_Da"].to_numpy())
            lines.append(f"ridge boundary-hit (point estimate): {int(bh)}/{len(rr)}")
            # ridge width diagnostics (flatness)
            for _col in ("width_decades_f098", "width_decades_f095", "width_decades_f090"):
                if _col in rr.columns:
                    w = rr[_col].to_numpy()
                    w = w[np.isfinite(w)]
                    if w.size:
                        zf = float(np.mean(w <= 1e-6))
                        lines.append(
                            f"{_col}: median={float(np.median(w)):.3f} "
                            f"p10={float(np.quantile(w,0.10)):.3f} "
                            f"p90={float(np.quantile(w,0.90)):.3f} "
                            f"zero_frac={zf:.2f}"
                        )


            # peak vs centroid (top-band f095) discrepancy: large values indicate a flat/noisy peak
            if "dlog10Da_centroid_f095_minus_peak" in rr.columns:
                dd = rr["dlog10Da_centroid_f095_minus_peak"].to_numpy()
                dd = dd[np.isfinite(dd)]
                if dd.size:
                    ad = np.abs(dd)
                    lines.append(
                        f"peak vs centroid(f095): median|Δlog10Da|={float(np.median(ad)):.3f} "
                        f"p90={float(np.quantile(ad,0.90)):.3f}"
                    )
            if ("boot_hitL_Da_frac" in rr.columns) and ("boot_hitR_Da_frac" in rr.columns):
                cL = rr["boot_hitL_Da_frac"].to_numpy()
                cR = rr["boot_hitR_Da_frac"].to_numpy()
                if np.any(np.isfinite(cL)) and np.any(np.isfinite(cR)):
                    lines.append(
                        f"ridge bootstrap censor(Da): "
                        f"med_hitL={float(np.nanmedian(cL)):.2f} "
                        f"med_hitR={float(np.nanmedian(cR)):.2f} "
                        f"max_hitL={float(np.nanmax(cL)):.2f} "
                        f"max_hitR={float(np.nanmax(cR)):.2f}"
                    )

    if spot_df is not None and not spot_df.empty:
        ok = (spot_df["status"] == "ok")
        if ok.any():
            med_abs_dHI = float(np.nanmedian(spot_df.loc[ok, "abs_dHI"].to_numpy())) if "abs_dHI" in spot_df.columns else np.nan
            p95_abs_dHI = float(np.nanquantile(spot_df.loc[ok, "abs_dHI"].to_numpy(), 0.95)) if "abs_dHI" in spot_df.columns else np.nan
            med_abs_rel_HI = float(np.nanmedian(np.abs(spot_df.loc[ok, "rel_HI"].to_numpy())))
            p95_abs_rel_HI = float(np.nanquantile(np.abs(spot_df.loc[ok, "rel_HI"].to_numpy()), 0.95))
            med_abs_rel_Hpk = float(np.nanmedian(np.abs(spot_df.loc[ok, "rel_Hpk"].to_numpy())))
            # If available, also report production-equivalent errors after publication-domain replacement
            if ("abs_dHI_prod" in spot_df.columns) and ("rel_HI_prod" in spot_df.columns):
                med_abs_dHI_prod = float(np.nanmedian(spot_df.loc[ok, "abs_dHI_prod"].to_numpy()))
                p95_abs_dHI_prod = float(np.nanquantile(spot_df.loc[ok, "abs_dHI_prod"].to_numpy(), 0.95))
                med_abs_rel_HI_prod = float(np.nanmedian(np.abs(spot_df.loc[ok, "rel_HI_prod"].to_numpy())))
                p95_abs_rel_HI_prod = float(np.nanquantile(np.abs(spot_df.loc[ok, "rel_HI_prod"].to_numpy()), 0.95))
                lines.append(f"spotcheck: ok={int(ok.sum())}/{len(spot_df)} RK4 med|dHI|={med_abs_dHI:.3e} p95|dHI|={p95_abs_dHI:.3e} med|rel_HI(floor={HI_REL_FLOOR:g})|={med_abs_rel_HI:.3e} p95|rel_HI|={p95_abs_rel_HI:.3e} med|rel_Hpk|={med_abs_rel_Hpk:.3e} | PROD med|dHI|={med_abs_dHI_prod:.3e} p95|dHI|={p95_abs_dHI_prod:.3e} med|rel_HI|={med_abs_rel_HI_prod:.3e} p95|rel_HI|={p95_abs_rel_HI_prod:.3e}")
            else:
                lines.append(f"spotcheck: ok={int(ok.sum())}/{len(spot_df)} med|dHI|={med_abs_dHI:.3e} p95|dHI|={p95_abs_dHI:.3e} med|rel_HI(floor={HI_REL_FLOOR:g})|={med_abs_rel_HI:.3e} p95|rel_HI|={p95_abs_rel_HI:.3e} med|rel_Hpk|={med_abs_rel_Hpk:.3e}")

    if sat_spot_df is not None and not sat_spot_df.empty:
        try:
            parts = []
            for grp in ("sat", "unsat"):
                sub = sat_spot_df[(sat_spot_df.get("group") == grp) & (sat_spot_df.get("status") == "ok")]
                if len(sub) == 0:
                    continue
                med_abs_dHI = float(np.nanmedian(sub["abs_dHI"].to_numpy())) if "abs_dHI" in sub.columns else np.nan
                p95_abs_dHI = float(np.nanquantile(sub["abs_dHI"].to_numpy(), 0.95)) if "abs_dHI" in sub.columns else np.nan
                med_abs_rel_HI = float(np.nanmedian(np.abs(sub["rel_HI"].to_numpy())))
                p95_abs_rel_HI = float(np.nanquantile(np.abs(sub["rel_HI"].to_numpy()), 0.95))
                parts.append(f"{grp}:ok={len(sub)} med|dHI|={med_abs_dHI:.3e} p95|dHI|={p95_abs_dHI:.3e} med|rel_HI(floor={HI_REL_FLOOR:g})|={med_abs_rel_HI:.3e} p95|rel_HI|={p95_abs_rel_HI:.3e}")
            if parts:
                lines.append("sat spotcheck: " + " | ".join(parts))
        except Exception:
            pass


    if sens_info is not None:
        try:
            # peak-based sensitivity (argmax)
            mxp = float(sens_info.get("overall_max_abs_dlog10Da_peak", np.nan))
            nbp = sens_info.get("worst_n_bins_peak", None)
            sgp = sens_info.get("worst_sigma_peak", None)
            typ_p = float(sens_info.get("median_over_settings_max_abs_peak", np.nan))
            if np.isfinite(mxp):
                line = f"ridge sensitivity (peak): worst max|Δlog10Da_peak|={mxp:.3f} (n_bins={nbp}, sigma={sgp})"
                if np.isfinite(typ_p):
                    line += f"; median_over_settings_max|Δ|={typ_p:.3f}"
                lines.append(line)

            # centroid-based sensitivity (top-band centroid f095)
            mxc = float(sens_info.get("overall_max_abs_dlog10Da_centroid_f095", np.nan))
            nbc = sens_info.get("worst_n_bins_centroid", None)
            sgc = sens_info.get("worst_sigma_centroid", None)
            typ_c = float(sens_info.get("median_over_settings_max_abs_centroid", np.nan))
            if np.isfinite(mxc):
                line = f"ridge sensitivity (centroid f095): worst max|Δlog10Da|={mxc:.3f} (n_bins={nbc}, sigma={sgc})"
                if np.isfinite(typ_c):
                    line += f"; median_over_settings_max|Δ|={typ_c:.3f}"
                lines.append(line)

            # top-R ranges across settings (compact)
            topP = sens_info.get("topR_range_peak", None)
            topC = sens_info.get("topR_range_centroid", None)
            if isinstance(topP, (list, tuple)) and len(topP):
                s = ", ".join([f"R={r:g}:{v:.3f}" for r, v in list(topP)[:3]])
                lines.append(f"ridge sens by-R (peak): {s}")
            if isinstance(topC, (list, tuple)) and len(topC):
                s = ", ".join([f"R={r:g}:{v:.3f}" for r, v in list(topC)[:3]])
                lines.append(f"ridge sens by-R (centroid): {s}")
        except Exception:
            pass

    if out_dir:
        lines.append(f"out_dir={out_dir}")

    return "\n".join(lines)


# ------------------------------------------------------------------------------
# SI audit table + packaging helper (core outputs ZIP)
# ------------------------------------------------------------------------------

def build_si_audit_table(
    df: pd.DataFrame,
    ridge: pd.DataFrame = None,
    bimod_src_df: pd.DataFrame = None,
    sens_byR: pd.DataFrame = None,
    out_csv: str = None,
    ridge_neighborhood_decades: float = 1.0,
):
    """Build a single, publication-friendly audit table for the SI.

    The table summarizes (by storage contrast R):
      - IVP coverage (overall and in key Da bands)
      - IVP replacement reasons (hard_nsub / mid-Da artifact band / low-Da gate)
      - Artifact status (cross-lobe bimodality + mid-Da hump) if diagnostics are available
      - Ridge sensitivity range (vs binning/smoothing) if ridge_sensitivity() was run

    Returns:
        audit_df (pd.DataFrame)
    """
    if df is None or df.empty:
        return None

    # Ridge location (prefer centroid f095, which is more stable on flat ridges)
    ridge_map = {}
    if ridge is not None and (not ridge.empty) and ("Contrast" in ridge.columns):
        for _, rrow in ridge.iterrows():
            try:
                R = float(rrow["Contrast"])
            except Exception:
                continue
            da_ridge = np.nan
            if "Da_centroid_f095" in ridge.columns:
                da_ridge = float(rrow.get("Da_centroid_f095", np.nan))
            if (not np.isfinite(da_ridge)) and ("Da_peak" in ridge.columns):
                da_ridge = float(rrow.get("Da_peak", np.nan))
            ridge_map[R] = da_ridge

    # Diagnostics maps (optional)
    bimod_map = {}
    if bimod_src_df is not None and (not bimod_src_df.empty) and ("Contrast" in bimod_src_df.columns):
        for _, brow in bimod_src_df.iterrows():
            try:
                R = float(brow["Contrast"])
            except Exception:
                continue
            bimod_map[R] = brow.to_dict()

    sens_map = {}
    if sens_byR is not None and (not sens_byR.empty) and ("Contrast" in sens_byR.columns):
        for _, srow in sens_byR.iterrows():
            try:
                R = float(srow["Contrast"])
            except Exception:
                continue
            sens_map[R] = srow.to_dict()

    rows = []
    for R in sorted(df["Contrast"].unique()):
        sub = df[df["Contrast"] == R].copy()
        if sub.empty:
            continue

        n = int(len(sub))

        fb = sub.get("fallback_used", 0)
        if isinstance(fb, pd.Series):
            fb = fb.to_numpy()
        fb = np.asarray(fb).astype(int)
        n_ivp = int(np.sum(fb == 1))
        n_rk4 = int(np.sum(fb == 0))
        frac_ivp = float(n_ivp / max(1, n))

        # Reasons (string match to be robust to minor naming tweaks)
        reason = sub.get("fallback_reason", "")
        if isinstance(reason, pd.Series):
            reason = reason.fillna("").astype(str).to_numpy()
        reason = np.asarray(reason, dtype=str)

        n_hard = int(np.sum(np.char.find(reason.astype(str), "hard_nsub") >= 0))
        n_soft_mid = int(np.sum(np.char.find(reason.astype(str), "soft_bimodal") >= 0))
        n_soft_low = int(np.sum(np.char.find(reason.astype(str), "soft_lowDa") >= 0))
        n_soft_ridge = int(np.sum(np.char.find(reason.astype(str), "soft_ridge") >= 0))
        n_soft_shoulder = int(np.sum(np.char.find(reason.astype(str), "soft_shoulder") >= 0))

        # Error magnitude where we *did* run an IVP check (abs_dHI_check is stored from the RK4 pre-pass)
        abs_chk = sub.get("abs_dHI_check", np.nan)
        if isinstance(abs_chk, pd.Series):
            abs_chk = abs_chk.to_numpy(dtype=float)
        abs_chk = np.asarray(abs_chk, float)
        m_chk = np.isfinite(abs_chk)
        med_abs_chk = float(np.nanmedian(abs_chk[m_chk])) if np.any(m_chk) else np.nan
        p95_abs_chk = float(np.nanpercentile(abs_chk[m_chk], 95)) if np.sum(m_chk) >= 5 else np.nan
        max_abs_chk = float(np.nanmax(abs_chk[m_chk])) if np.any(m_chk) else np.nan

        # Da bands
        Da = sub.get("Da", np.nan)
        if isinstance(Da, pd.Series):
            Da = Da.to_numpy(dtype=float)
        Da = np.asarray(Da, float)

        # Mid-Da "artifact band" (hump falsification band)
        mid_mask = np.isfinite(Da) & (Da >= float(DA_MID_MIN)) & (Da <= float(DA_MID_MAX))
        n_mid = int(np.sum(mid_mask))
        frac_ivp_mid = float(np.mean(fb[mid_mask] == 1)) if n_mid > 0 else np.nan

        # Low-Da publication-domain band (includes buffer above DA_LOBE_SPLIT)
        low_mask = np.isfinite(Da) & (Da >= float(FALLBACK_SOFT_LOWDA_DA_MIN)) & (Da <= float(FALLBACK_SOFT_LOWDA_DA_MAX))
        n_low = int(np.sum(low_mask))
        frac_ivp_low = float(np.mean(fb[low_mask] == 1)) if n_low > 0 else np.nan

        # "Transition" just above reporting split (useful to audit policy kinks)
        trans_lo = float(DA_LOBE_SPLIT)
        trans_hi = float(FALLBACK_SOFT_LOWDA_DA_MAX)
        trans_mask = np.isfinite(Da) & (Da >= trans_lo) & (Da <= trans_hi)
        n_trans = int(np.sum(trans_mask))
        frac_ivp_trans = float(np.mean(fb[trans_mask] == 1)) if n_trans > 0 else np.nan

        # Ridge-neighborhood IVP coverage (± ridge_neighborhood_decades around ridge)
        da_ridge = ridge_map.get(float(R), np.nan)
        n_ridge_band = 0
        frac_ivp_ridge_band = np.nan
        if np.isfinite(da_ridge) and (da_ridge > 0):
            fac = float(10.0 ** float(abs(ridge_neighborhood_decades)))
            lo = float(da_ridge / fac)
            hi = float(da_ridge * fac)
            ridge_band = np.isfinite(Da) & (Da >= lo) & (Da <= hi)
            n_ridge_band = int(np.sum(ridge_band))
            frac_ivp_ridge_band = float(np.mean(fb[ridge_band] == 1)) if n_ridge_band > 0 else np.nan

        # Artifact flags (optional)
        cross_bimodal_all = mid_hump_all = np.nan
        cross_ratio_all = mid_ratio_all = np.nan
        if float(R) in bimod_map:
            d = bimod_map[float(R)]
            cross_bimodal_all = float(d.get("cross_bimodal_all", np.nan))
            cross_ratio_all = float(d.get("cross_ratio_all", np.nan))
            mid_hump_all = float(d.get("mid_hump_all", np.nan))
            mid_ratio_all = float(d.get("mid_ratio_all", np.nan))

        # Ridge sensitivity ranges (optional)
        range_logDa_peak = np.nan
        range_logDa_cent = np.nan
        if float(R) in sens_map:
            d = sens_map[float(R)]
            range_logDa_peak = float(d.get("range_log10Da_peak", np.nan))
            range_logDa_cent = float(d.get("range_log10Da_centroid_f095", np.nan))

        rows.append({
            "Contrast_R": float(R),
            "n_accept": int(n),

            "n_ivp": int(n_ivp),
            "frac_ivp": float(frac_ivp),

            "n_rk4": int(n_rk4),

            "n_hard_nsub": int(n_hard),
            "n_soft_midDa": int(n_soft_mid),
            "n_soft_lowDa": int(n_soft_low),
            "n_soft_ridge": int(n_soft_ridge),
            "n_soft_shoulder": int(n_soft_shoulder),

            "frac_ivp_midDaBand": float(frac_ivp_mid) if np.isfinite(frac_ivp_mid) else np.nan,
            "n_midDaBand": int(n_mid),

            "frac_ivp_lowDaBand": float(frac_ivp_low) if np.isfinite(frac_ivp_low) else np.nan,
            "n_lowDaBand": int(n_low),

            "frac_ivp_transitionBand": float(frac_ivp_trans) if np.isfinite(frac_ivp_trans) else np.nan,
            "n_transitionBand": int(n_trans),

            "Da_ridge_centroid_f095": float(da_ridge) if np.isfinite(da_ridge) else np.nan,
            "frac_ivp_ridgeBand_pmDec": float(frac_ivp_ridge_band) if np.isfinite(frac_ivp_ridge_band) else np.nan,
            "n_ridgeBand_pmDec": int(n_ridge_band),

            "med_abs_dHI_check": float(med_abs_chk) if np.isfinite(med_abs_chk) else np.nan,
            "p95_abs_dHI_check": float(p95_abs_chk) if np.isfinite(p95_abs_chk) else np.nan,
            "max_abs_dHI_check": float(max_abs_chk) if np.isfinite(max_abs_chk) else np.nan,

            # Artifact status (should be 0/False after publication-domain replacement is correct)
            "cross_bimodal_all": float(cross_bimodal_all) if np.isfinite(cross_bimodal_all) else np.nan,
            "cross_ratio_all": float(cross_ratio_all) if np.isfinite(cross_ratio_all) else np.nan,
            "mid_hump_all": float(mid_hump_all) if np.isfinite(mid_hump_all) else np.nan,
            "mid_ratio_all": float(mid_ratio_all) if np.isfinite(mid_ratio_all) else np.nan,

            # Ridge sensitivity ranges (decades)
            "range_log10Da_peak": float(range_logDa_peak) if np.isfinite(range_logDa_peak) else np.nan,
            "range_log10Da_centroid_f095": float(range_logDa_cent) if np.isfinite(range_logDa_cent) else np.nan,
        })

    audit = pd.DataFrame(rows)

    if out_csv:
        try:
            audit.to_csv(out_csv, index=False)
        except Exception:
            pass

    return audit


def package_core_outputs_zip(
    out_dir: str,
    run_tag: str = None,
    zip_prefix: str = "core_outputs",
    include_runs_csv: bool = True,
    extra_files: list = None,
):
    """Create a timestamped ZIP with core outputs for sharing / SI review."""
    if not out_dir:
        return None
    if not os.path.isdir(out_dir):
        return None

    if run_tag is None:
        run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")

    zip_name = f"{zip_prefix}_{run_tag}.zip"
    zip_path = os.path.join(out_dir, zip_name)

    core_files = [
        "run_config.json",
        "progress_summary.txt",
        "publish_domain_gate_summary.csv",

        "ridge.csv",
        "peak_curve.csv",
        "runs.csv" if include_runs_csv else None,

        "spotcheck_rk4_vs_solveivp.csv",
        "spotcheck.csv",

        "spotcheck_sat_vs_unsat.csv",
        "sat_spotcheck.csv",

        "bimodality_source_compare.csv",

        "ridge_sensitivity.csv",
        "ridge_sensitivity_byR.csv",

        "si_audit_table.csv",
    ]
    core_files = [f for f in core_files if f is not None]

    if extra_files:
        for f in extra_files:
            if f and isinstance(f, str):
                core_files.append(f)

    # Only include what exists
    file_paths = []
    for f in core_files:
        p = os.path.join(out_dir, f)
        if os.path.isfile(p):
            file_paths.append((p, f))

    if len(file_paths) == 0:
        return None

    try:
        with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            for p, arc in file_paths:
                zf.write(p, arcname=arc)
        print(f"[ZIP] wrote core outputs: {zip_path}")
        return zip_path
    except Exception as e:
        print(f"[ZIP] failed: {e}")
        return None


def _write_run_outputs(out_dir: str, df: pd.DataFrame, ridge: pd.DataFrame, peak_curve: pd.DataFrame,
                       attempted: int, spot_df: pd.DataFrame, sat_spot_df: pd.DataFrame = None,
                       sens_df: pd.DataFrame = None, sens_byR: pd.DataFrame = None, sens_info: dict = None):
    if not out_dir:
        return

    try:
        if peak_curve is not None and not peak_curve.empty:
            peak_curve.to_csv(os.path.join(out_dir, "peak_curve.csv"), index=False)
    except Exception as e:
        print(f"[Output] could not write peak_curve.csv ({e})")

    try:
        if ridge is not None and not ridge.empty:
            ridge.to_csv(os.path.join(out_dir, "ridge.csv"), index=False)
    except Exception as e:
        print(f"[Output] could not write ridge.csv ({e})")

    try:
        if SAVE_RUNS_CSV and df is not None and not df.empty:
            df.to_csv(os.path.join(out_dir, "runs.csv"), index=False)
    except Exception as e:
        print(f"[Output] could not write runs.csv ({e})")

    try:
        if spot_df is not None and not spot_df.empty:
            spot_df.to_csv(os.path.join(out_dir, "spotcheck.csv"), index=False)
    except Exception as e:
        print(f"[Output] could not write spotcheck.csv ({e})")

    try:
        if sat_spot_df is not None and not sat_spot_df.empty:
            sat_spot_df.to_csv(os.path.join(out_dir, "spotcheck_sat_vs_unsat.csv"), index=False)
    except Exception as e:
        print(f"[Output] could not write spotcheck_sat_vs_unsat.csv ({e})")

    try:
        if sens_df is not None and not sens_df.empty:
            sens_df.to_csv(os.path.join(out_dir, "ridge_sensitivity.csv"), index=False)
    except Exception as e:
        print(f"[Output] could not write ridge_sensitivity.csv ({e})")

    try:
        if sens_byR is not None and not sens_byR.empty:
            sens_byR.to_csv(os.path.join(out_dir, "ridge_sensitivity_byR.csv"), index=False)
    except Exception as e:
        print(f"[Output] could not write ridge_sensitivity_byR.csv ({e})")

    # summary txt + ridge preview
    try:
        summary = _build_progress_summary(df, ridge, attempted=attempted, spot_df=spot_df, sat_spot_df=sat_spot_df, out_dir=out_dir, sens_info=sens_info)
        preview = _compact_ridge_preview(ridge)
        with open(os.path.join(out_dir, "progress_summary.txt"), "w") as f:
            f.write(summary + "\n\n")
            f.write("----- RIDGE PREVIEW -----\n")
            f.write(preview + "\n")
        print("\n----- BEGIN MODEL PROGRESS SUMMARY -----")
        print(summary)
        print("----- RIDGE PREVIEW -----")
        print(preview)
        print("----- END MODEL PROGRESS SUMMARY -----\n")
    except Exception as e:
        print(f"[Output] could not write/print summary ({e})")

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

# Parallelization settings
N_JOBS = -1  # -1 = use all cores, or set to specific number

def main():
    # Output directory (compact progress reporting)
    out_dir = None
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    if WRITE_RUN_OUTPUTS:
        out_dir = os.path.join(OUTDIR_BASE, f"run_{run_tag}")
        os.makedirs(out_dir, exist_ok=True)
        print(f"[Output] out_dir={out_dir}")
        try:
            cfg = _build_run_config_dict(run_tag, out_dir)
            with open(os.path.join(out_dir, "run_config.json"), "w") as f:
                json.dump(cfg, f, indent=2)
        except Exception as e:
            print(f"[Output] could not write run_config.json ({e})")

    # Optional: numerical fidelity check (RK4 vs solve_ivp)
    spot_df = None
    if RUN_SPOTCHECK:
        out_csv = os.path.join(out_dir, "spotcheck_rk4_vs_solveivp.csv") if out_dir else None
        spot_df = spotcheck_rk4_vs_solveivp(
            n=SPOTCHECK_N, seed=SPOTCHECK_SEED, R_levels=SPOTCHECK_R_LEVELS,
            out_csv=out_csv
        )
    if out_dir and FALLBACK_SOFT_RIDGE:
        try:
            gate_df = _build_publish_domain_gate_summary()
            if gate_df is not None and (not gate_df.empty):
                gate_df.to_csv(os.path.join(out_dir, "publish_domain_gate_summary.csv"), index=False)
        except Exception as e:
            print(f"[Output] could not write publish_domain_gate_summary.csv ({e})")

    if not DO_FULL_LHC:
        print("[RunMode] DO_FULL_LHC=False: spotcheck only; skipping full LHC run.")
        return pd.DataFrame(), pd.DataFrame()

    print(f"Running Level 19-LHC: n={n_lhc_per_contrast} samples/contrast "
          f"({n_lhc_per_contrast * len(Contrast_range)} total ODE runs)")
    

    # Import joblib (optional dependency)
    try:
        from joblib import Parallel, delayed
        import multiprocessing
        n_cores = multiprocessing.cpu_count() if N_JOBS == -1 else N_JOBS
        USE_PARALLEL = True
        print(f"[Parallel] Using {n_cores} cores")
    except Exception as e:
        USE_PARALLEL = False
        Parallel, delayed = None, None
        print(f"[Parallel] Failed ({e}), running sequentially")

    # Sampling
    if USE_DAR_SAMPLING:
        K_samps, DaR_t_samps, logK_samps, logDaR_samps = sample_lhc_log_targets(n_lhc_per_contrast, seed=SEED)
        print(f"[DaR target] range: {DaR_t_samps.min():.3e} – {DaR_t_samps.max():.3e}")
    else:
        K_samps, A_samps, logK_samps, logA_samps = sample_lhc_log_params(n_lhc_per_contrast, seed=SEED)

    P_vec = precip_vec(t_eval, P_mag, P_dur, t0=t0_storm)
    _set_globals_for_workers(P_vec)

    # Build argument list for all ODE runs
    # all_args = []
    # for contrast_val in Contrast_range:
    #     Sy_m = Sy_f * contrast_val
    #     for i in range(n_lhc_per_contrast):
    #         if USE_DAR_SAMPLING:
    #             K = float(K_samps[i])
    #             DaR_target = float(DaR_t_samps[i])
    #             logK = float(logK_samps[i])
    #             Qout_ref = Q_BASE + K_LIN * H_REF + K * H_REF * H_REF
    #             alpha = DaR_target * Qout_ref / (H_REF + 1e-30)
    #             logA = float(np.log10(alpha))
    #         else:
    #             K, alpha = float(K_samps[i]), float(A_samps[i])
    #             logK, logA = float(logK_samps[i]), float(logA_samps[i])
    #             DaR_target = np.nan
            
    #         all_args.append((contrast_val, i, K, alpha, logK, logA, DaR_target, Sy_m, P_vec))
    # Build flat list of run tuples (WITHOUT P_vec)
    runs = []
    for contrast_val in Contrast_range:
        Sy_m = Sy_f * contrast_val
        for i in range(n_lhc_per_contrast):
            if USE_DAR_SAMPLING:
                K = float(K_samps[i])
                DaR_target = float(DaR_t_samps[i])
                logK = float(logK_samps[i])
                Qout_ref = Q_BASE + K_LIN * H_REF + K * H_REF * H_REF
                alpha = DaR_target * Qout_ref / (H_REF + 1e-30)
                logA = float(np.log10(alpha))
            else:
                K = float(K_samps[i])
                alpha = float(A_samps[i])
                logK = float(logK_samps[i])
                logA = float(logA_samps[i])
                DaR_target = np.nan
    
            runs.append((contrast_val, K, alpha, logK, logA, DaR_target, Sy_m))
    
    # Chunk it
    chunks = [runs[i:i+CHUNK_SIZE] for i in range(0, len(runs), CHUNK_SIZE)]
    print(f"[Chunking] {len(runs)} runs -> {len(chunks)} chunks of ~{CHUNK_SIZE}")

    # Run in parallel or sequential
    if USE_PARALLEL:
        chunk_results = Parallel(n_jobs=N_JOBS, backend=PAR_BACKEND, verbose=10)(
            delayed(_run_chunk)(ch) for ch in chunks
            )
    else:
        chunk_results = [_run_chunk(ch) for ch in chunks]
    
    # Flatten list of lists
    data = [row for block in chunk_results for row in block]
    
    # Summary per contrast
    for contrast_val in Contrast_range:
        contrast_data = [d for d in data if d["Contrast"] == contrast_val]
        if contrast_data:
            Da_vals = np.array([d["Da"] for d in contrast_data])
            print(f"[R={contrast_val:>4}] accepted={len(contrast_data):>4} | Da: {Da_vals.min():.3e} – {Da_vals.max():.3e}")

    df = pd.DataFrame(data).replace([np.inf, -np.inf], np.nan).dropna(subset=["Contrast", "Da", "DaR", "Hysteresis"])
    print(f"\n=== RUN SUMMARY ===\nAttempted: {len(runs)} | Accepted: {len(df)}")

    if df.empty:
        raise RuntimeError("No valid points.")

    peak_curve = _print_peak_table(df)

    # Optional: targeted check on substep-saturated runs (hard cases)
    sat_spot_df = None
    if RUN_SAT_SPOTCHECK:
        sat_out_csv = os.path.join(out_dir, "spotcheck_sat_vs_unsat.csv") if out_dir else None
        sat_spot_df = spotcheck_saturated_vs_unsat(df, out_csv=sat_out_csv)

    # Auto-bracketing (if needed)
    if AUTO_BRACKET:
        peak_ok = np.isfinite(peak_curve["Da_peak_curve"]) & np.isfinite(peak_curve["DaR_peak_curve"])
        support_ok = (peak_curve["nBins_Da"] >= MIN_VALID_BINS_FOR_PEAK)
        needs_expand = peak_curve[peak_ok & support_ok & peak_curve["hitL_Da"]].copy()

        if not needs_expand.empty:
            print(f"\n[AutoBracket] Peak hit left boundary for: {needs_expand['Contrast'].tolist()}")
            print("[AutoBracket] NOTE: rerun/expansion is not implemented in this file. Expand LOG10_DAR_MIN/MAX (and/or n_lhc_per_contrast) and rerun if you truly need more domain coverage.")

    # Ridge table
    ridge = ridge_table_with_bootstrap(df)

    # Bimodality verification: compare ALL vs IVP-only subset
    bimod_src_df = None
    if BIMODALITY_SOURCE_COMPARE and out_dir:
        bimod_src_csv = os.path.join(out_dir, "bimodality_source_compare.csv")
        bimod_src_df = bimodality_source_compare(df, out_csv=bimod_src_csv)
        if bimod_src_df is not None and (not bimod_src_df.empty):
            try:
                ncb_all = int(np.sum(bimod_src_df.get("cross_bimodal_all", 0).to_numpy().astype(int) == 1)) if ("cross_bimodal_all" in bimod_src_df.columns) else 0
                ncb_ivp = int(np.sum(bimod_src_df.get("cross_bimodal_ivp", 0).to_numpy().astype(int) == 1)) if ("cross_bimodal_ivp" in bimod_src_df.columns) else 0
                ncb_rk4 = int(np.sum(bimod_src_df.get("cross_bimodal_rk4", 0).to_numpy().astype(int) == 1)) if ("cross_bimodal_rk4" in bimod_src_df.columns) else 0
                print(f"[Bimodality] cross-lobe flagged: all={ncb_all}/{len(bimod_src_df)} | fallback-subset={ncb_ivp}/{len(bimod_src_df)} | rk4-only={ncb_rk4}/{len(bimod_src_df)}")
                # Mid-Da hump verdict (explicit band Da in [DA_MID_MIN, DA_MID_MAX])
                if ('mid_hump_all' in bimod_src_df.columns) and ('mid_ratio_all' in bimod_src_df.columns):
                    nR = int(len(bimod_src_df))
                    mh_all = int(np.sum(bimod_src_df['mid_hump_all'].fillna(0).astype(int).to_numpy() == 1))
                    mh_ivp = int(np.sum(bimod_src_df.get('mid_hump_ivp', 0).fillna(0).astype(int).to_numpy() == 1)) if ('mid_hump_ivp' in bimod_src_df.columns) else 0
                    mh_rk4 = int(np.sum(bimod_src_df.get('mid_hump_rk4', 0).fillna(0).astype(int).to_numpy() == 1)) if ('mid_hump_rk4' in bimod_src_df.columns) else 0
                    def _nanmed(arr):
                        try:
                            v = np.nanmedian(arr.astype(float))
                            return float(v) if np.isfinite(v) else np.nan
                        except Exception:
                            return np.nan
                    mr_all = _nanmed(bimod_src_df.get('mid_ratio_all', np.nan).to_numpy())
                    mr_ivp = _nanmed(bimod_src_df.get('mid_ratio_ivp', np.nan).to_numpy()) if ('mid_ratio_ivp' in bimod_src_df.columns) else np.nan
                    mr_rk4 = _nanmed(bimod_src_df.get('mid_ratio_rk4', np.nan).to_numpy()) if ('mid_ratio_rk4' in bimod_src_df.columns) else np.nan
                    print(f"[MidDaHump] band=[{DA_MID_MIN:g},{DA_MID_MAX:g}] ratio>={MID_DA_RATIO_THRESH:.2f} flagged: all={mh_all}/{nR} | fallback-subset={mh_ivp}/{nR} | rk4-only={mh_rk4}/{nR} | med_ratio(all/ivp/rk4)={mr_all:.3f}/{mr_ivp:.3f}/{mr_rk4:.3f}")
                    # If flagged in ALL, print which contrasts and whether they fall outside publication-domain gating.
                    if mh_all > 0:
                        try:
                            flagged = bimod_src_df[bimod_src_df["mid_hump_all"].fillna(0).astype(int) == 1].copy()
                            cols_show = ["Contrast","mid_ratio_all","Da_mid_all","HI_mid_all","HI_low_all","n_mid_all","n_ivp","n_high_ivp","n_high_rk4"]
                            cols_show = [c for c in cols_show if c in flagged.columns]
                            if len(flagged) > 0:
                                print("[MidDaHump] flagged contrasts (ALL):")
                                with pd.option_context("display.max_rows", 200, "display.max_columns", 200):
                                    print(flagged[cols_show].to_string(index=False))
                                # Suggest widening publication-domain region if flags occur below current soft-R gate
                                try:
                                    if np.any(flagged["Contrast"].to_numpy(dtype=float) < float(FALLBACK_SOFT_R_MIN)):
                                        print(f"[MidDaHump] NOTE: some flagged contrasts are below FALLBACK_SOFT_R_MIN={FALLBACK_SOFT_R_MIN:g}. Consider lowering FALLBACK_SOFT_R_MIN or tightening MID_DA_RATIO_THRESH if you want ALL curves fully IVP-verified in the mid-Da band.")
                                except Exception:
                                    pass
                        except Exception:
                            pass
                # Coverage sanity: if ALL indicates cross-bimodal but IVP has too few samples, ivp-only result is non-informative.
                if ("cross_bimodal_all" in bimod_src_df.columns) and ("n_ivp" in bimod_src_df.columns):
                    mask_all = (bimod_src_df["cross_bimodal_all"].astype(int) == 1)
                    if bool(mask_all.any()):
                        n_mask = int(mask_all.sum())
                        n_under = int(np.sum(mask_all & (bimod_src_df["n_ivp"].astype(int) < int(BIMODALITY_SOURCE_MIN_N))))
                        n_ivp_vals = bimod_src_df.loc[mask_all, "n_ivp"].astype(int).to_numpy()
                        try:
                            n_min = int(np.min(n_ivp_vals))
                            n_med = int(np.median(n_ivp_vals))
                            n_max = int(np.max(n_ivp_vals))
                        except Exception:
                            n_min = n_med = n_max = -1
                        print(f"[Bimodality] ivp coverage for cross-bimodal ALL rows: n={n_mask} | underpowered(n_ivp<{int(BIMODALITY_SOURCE_MIN_N)})={n_under} | n_ivp min/med/max={n_min}/{n_med}/{n_max}")
                        # Also report lobe-wise IVP support (Da<=split vs Da>split) if available
                        if ('n_high_ivp' in bimod_src_df.columns) and ('n_low_ivp' in bimod_src_df.columns):
                            n_hi = bimod_src_df.loc[mask_all, 'n_high_ivp'].astype(int).to_numpy()
                            n_lo = bimod_src_df.loc[mask_all, 'n_low_ivp'].astype(int).to_numpy()
                            try:
                                hi_min, hi_med, hi_max = int(n_hi.min()), int(np.median(n_hi)), int(n_hi.max())
                                lo_min, lo_med, lo_max = int(n_lo.min()), int(np.median(n_lo)), int(n_lo.max())
                            except Exception:
                                hi_min = hi_med = hi_max = -1
                                lo_min = lo_med = lo_max = -1
                            under_hi = int(np.sum(n_hi < 10))
                            print(f"[Bimodality] ivp lobe support (split={DA_LOBE_SPLIT:g}): n_high min/med/max={hi_min}/{hi_med}/{hi_max} | n_low min/med/max={lo_min}/{lo_med}/{lo_max} | underpowered(n_high<10)={under_hi}/{n_mask}")
                        if n_under > 0:
                            print(f"[Bimodality] NOTE: soft fallback currently uses R>={FALLBACK_SOFT_R_MIN:g}, Da in [{FALLBACK_SOFT_DA_MIN:g},{FALLBACK_SOFT_DA_MAX:g}]. Lower FALLBACK_SOFT_R_MIN and/or raise FALLBACK_SOFT_DA_MAX to verify the high-Da lobe where it occurs.")
            except Exception:
                pass


    # Ridge sensitivity (robustness to binning/smoothing)
    sens_df = None
    sens_info = None
    sens_byR = None
    if RUN_RIDGE_SENSITIVITY:
        sens_out_csv = os.path.join(out_dir, "ridge_sensitivity.csv") if out_dir else None
        sens_df, sens_info, sens_byR = ridge_sensitivity(df, out_csv=sens_out_csv)

    # --- SI audit table + outputs ---
    si_audit_df = None
    if WRITE_RUN_OUTPUTS and out_dir:
        try:
            si_audit_csv = os.path.join(out_dir, "si_audit_table.csv")
            si_audit_df = build_si_audit_table(
                df,
                ridge=ridge,
                bimod_src_df=bimod_src_df,
                sens_byR=sens_byR,
                out_csv=si_audit_csv,
            )
        except Exception:
            si_audit_df = None

        _write_run_outputs(
            out_dir,
            df,
            ridge,
            peak_curve,
            attempted=len(runs),
            spot_df=spot_df,
            sat_spot_df=sat_spot_df,
            sens_df=sens_df,
            sens_byR=sens_byR,
            sens_info=sens_info,
        )

        if bool(MAKE_CORE_ZIP):
            try:
                package_core_outputs_zip(
                    out_dir,
                    run_tag=run_tag,
                    include_runs_csv=bool(ZIP_INCLUDE_RUNS_CSV),
                )
            except Exception:
                pass

    # Visualization
    _plot_results(df, ridge)

    return df, ridge

def _col_to_1d(sub: pd.DataFrame, colname: str) -> np.ndarray:
    """
    Robustly extract a 1-D float array even if columns are duplicated and sub[colname]
    returns a DataFrame instead of a Series.
    """
    v = sub[colname]
    if isinstance(v, pd.DataFrame):
        v = v.iloc[:, 0]
    a = np.asarray(v, dtype=float).reshape(-1)
    return a


def _resample_logx(x: np.ndarray, y: np.ndarray, n: int = 200, sigma: float = 0.8):
    """
    Resample (x,y) onto a fine log10(x) grid and optionally smooth slightly.
    Handles NaNs in y by restricting to finite points.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & (x > 0) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size < 4:
        return x, y

    lx = np.log10(x)
    # ensure monotone x for interp
    order = np.argsort(lx)
    lx = lx[order]
    y = y[order]

    lx_f = np.linspace(lx.min(), lx.max(), n)
    y_f = np.interp(lx_f, lx, y)
    if sigma and sigma > 0:
        y_f = gaussian_filter1d(y_f, sigma=sigma)

    return 10.0 ** lx_f, y_f


def design_curve_binned(
    sub: pd.DataFrame,
    n_bins: int = 40,
    q_low: float = 0.02,
    q_high: float = 0.98,
    min_count: int = 5,
    smooth_sigma: float = 1.6,   # <-- bump this slightly (1.6–2.2 works well)
):
    """
    Count-weighted binning + smoothing for a single contrast slice.

    Returns (centers, y_smooth, count, w_smooth).
    y_smooth has NaNs where support is too weak.
    """
    da = _col_to_1d(sub, "Da")
    hi = _col_to_1d(sub, "Hysteresis")

    m = np.isfinite(da) & (da > 0) & np.isfinite(hi) & (hi >= 0)
    da = da[m]
    hi = hi[m]
    if da.size < 50:
        return None

    lx = np.log10(da)
    lx_lo = float(np.quantile(lx, q_low))
    lx_hi = float(np.quantile(lx, q_high))
    if (not np.isfinite(lx_lo)) or (not np.isfinite(lx_hi)) or (lx_hi <= lx_lo):
        return None

    da_lo = max(10.0 ** lx_lo, 1e-12)
    da_hi = 10.0 ** lx_hi
    if (not np.isfinite(da_lo)) or (not np.isfinite(da_hi)) or (da_hi <= da_lo):
        return None

    edges = np.logspace(np.log10(da_lo), np.log10(da_hi), n_bins + 1)  # 41 edges -> 40 bins
    centers = np.sqrt(edges[:-1] * edges[1:])

    b = np.digitize(da, edges) - 1
    b = np.clip(b, 0, n_bins - 1)

    count = np.bincount(b, minlength=n_bins).astype(int)
    s = np.bincount(b, weights=hi, minlength=n_bins)

    mean = np.full(n_bins, np.nan, dtype=float)
    has = count > 0
    mean[has] = s[has] / count[has]

    # --- count-weighted smoothing (prevents low-count bins from causing wiggles) ---
    y0 = np.nan_to_num(mean, nan=0.0)
    w0 = count.astype(float)

    num = gaussian_filter1d(y0 * w0, sigma=smooth_sigma)
    den = gaussian_filter1d(w0, sigma=smooth_sigma)
    y_smooth = num / np.maximum(den, 1e-12)

    # support mask in *smoothed-weight* space
    y_smooth[den < float(min_count)] = np.nan

    # require at least a few supported bins
    if np.count_nonzero(np.isfinite(y_smooth)) < 6:
        return None

    return centers, y_smooth, count, den


def _plot_results(df, ridge):
    """Generate heatmap (Panel A) and design curves (Panel B)."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # ==========================
    # PANEL A: Binned (logDa, R) heatmap + support outline + ridge overlay
    # ==========================
    ax = axes[0]

    # ---- discrete R levels (rows) ----
    R_levels = np.array(sorted(df["Contrast"].unique()), dtype=float)
    nR = len(R_levels)
    
    # ---- Da bins (columns) based on data ----
    dfA = df.dropna(subset=["Da", "Contrast", "Hysteresis"]).copy()
    dfA = dfA[(dfA["Da"] > 0) & np.isfinite(dfA["Da"])]
    
    logDa = np.log10(dfA["Da"].to_numpy())

    # Use quantile-based bin edges in log10(Da) to reduce sparse "striping"
    nx = 80
    q_edges = np.linspace(0.001, 0.999, nx + 1)
    x_edges = np.quantile(logDa, q_edges)

    # enforce strictly increasing edges (pcolormesh requirement)
    for k in range(1, len(x_edges)):
        if x_edges[k] <= x_edges[k-1]:
            x_edges[k] = x_edges[k-1] + 1e-6

    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    Da_centers = 10.0 ** x_centers
    Da_edges = 10.0 ** x_edges
    
    # ---- build Z matrix: rows=R_levels, cols=Da bins ----
    Z = np.full((nR, nx), np.nan, dtype=float)
    C = np.zeros((nR, nx), dtype=int)  # counts
    
    # map each record into (row, col)
    R_map = {float(r): i for i, r in enumerate(R_levels)}
    ix = np.clip(np.digitize(logDa, x_edges) - 1, 0, nx - 1)
    rows = np.array([R_map[float(r)] for r in dfA["Contrast"].to_numpy()], dtype=int)
    
    flat = rows * nx + ix
    Cflat = np.bincount(flat, minlength=nR * nx).reshape(nR, nx)
    Sflat = np.bincount(flat, weights=dfA["Hysteresis"].to_numpy(), minlength=nR * nx).reshape(nR, nx)
    
    C[:] = Cflat
    with np.errstate(invalid="ignore", divide="ignore"):
        Z = Sflat / np.maximum(C, 1)
    
    # support mask
    N_MIN = 2
    Z[C < N_MIN] = np.nan
    
    # count-weighted smoothing along Da only (keeps rows discrete; avoids "inventing" between R values)
    Z0 = np.nan_to_num(Z, nan=0.0)
    W0 = np.where(np.isfinite(Z), C.astype(float), 0.0)
    Z_num = gaussian_filter1d(Z0 * W0, sigma=1.2, axis=1)
    W_den = gaussian_filter1d(W0, sigma=1.2, axis=1)
    Zs = Z_num / np.maximum(W_den, 1e-12)
    Zs[W_den < N_MIN] = np.nan
    
    # colormap
    base_cmap = plt.cm.nipy_spectral
    carr = base_cmap(np.linspace(0, 1, 256))
    carr[:10] = [1, 1, 1, 1]
    cmap = mcolors.LinearSegmentedColormap.from_list("WhiteSpectral", carr)
    cmap.set_bad("white")
    
    vmax = np.nanquantile(Zs, 0.99) if np.any(np.isfinite(Zs)) else 1.0
    
    # pcolormesh: need y-edges; make midpoints between discrete R levels
    y_edges = np.empty(nR + 1, dtype=float)
    y_edges[1:-1] = 0.5 * (R_levels[:-1] + R_levels[1:])
    y_edges[0] = R_levels[0] - (y_edges[1] - R_levels[0])
    y_edges[-1] = R_levels[-1] + (R_levels[-1] - y_edges[-2])
    
    M = ax.pcolormesh(Da_edges, y_edges, Zs, shading="auto", cmap=cmap, vmin=0.0, vmax=vmax)
    
    ax.set_xscale("log")
    ax.set_xlabel(r"Damköhler Number ($Da$)")
    ax.set_ylabel(r"Storage Contrast ($S_m/S_f$)")
    ax.set_title("A. Universal Resonance Landscape", fontweight="bold")
    plt.colorbar(M, ax=ax, label="Normalized Hysteresis Index")
    
    # ridge overlay(s): global peak ridge and (optionally) primary low-Da ridge
    if ridge is not None and ("Contrast" in ridge.columns):
        # choose which ridge to emphasize
        if (RIDGE_OVERLAY_MODE == "primary_lowDa") and ("Da_peak_primary_lowDa" in ridge.columns):
            main_col = "Da_peak_primary_lowDa"
        else:
            main_col = "Da_peak"

        # main ridge
        if main_col in ridge.columns:
            rr = ridge.dropna(subset=["Contrast", main_col]).sort_values("Contrast")
            rr = rr[(rr[main_col] > 0) & np.isfinite(rr[main_col])]
            if len(rr) >= 3:
                ax.plot(rr[main_col].to_numpy(), rr["Contrast"].to_numpy(), color="k", lw=2.2)

        # secondary ridge (diagnostic): the other lobe
        if PLOT_BOTH_RIDGES and ("Da_peak" in ridge.columns) and ("Da_peak_primary_lowDa" in ridge.columns):
            other_col = "Da_peak_primary_lowDa" if (main_col == "Da_peak") else "Da_peak"
            rr2 = ridge.dropna(subset=["Contrast", other_col]).sort_values("Contrast")
            rr2 = rr2[(rr2[other_col] > 0) & np.isfinite(rr2[other_col])]
            if len(rr2) >= 3:
                ax.plot(rr2[other_col].to_numpy(), rr2["Contrast"].to_numpy(),
                        color="k", lw=1.6, ls="--", alpha=0.8)


    # ==========================
    # PANEL B: Design curves (per-slice bins, single loop)
    # ==========================
    ax = axes[1]
    #plot_slices = [1.2, 2, 12, 32, 52, 62]
    plot_slices = [1.2, 1.4, 1.7, 2.0, 2.4, 2.9, 3.5, 4.2, 5.1, 6.2, 7.6, 9.2, 11.2, 13.6, 16.5, 20.1, 24.4, 29.7, 36.1, 43.9, 53.3, 64.7, 78.6, 95.4, 115.8]

    # Color map keyed by slice value (stable even if we change plotting order)
    colors = plt.cm.jet(np.linspace(0, 1, len(plot_slices)))
    colors[0] = [0.2, 0.2, 0.2, 1.0]  # dark gray for smallest R slice
    color_map = {float(s): colors[i] for i, s in enumerate(plot_slices)}

    # Emphasize extreme cases: draw all mid cases first (faint), then smallest and largest on top
    R_min = float(min(plot_slices))
    R_max = float(max(plot_slices))
    mid_slices = [float(s) for s in plot_slices if float(s) not in (R_min, R_max)]
    plot_order = mid_slices + [R_min, R_max]

    def _fmt_R_label(sval: float) -> str:
        sval = float(sval)
        # avoid duplicate labels (e.g., 2.0, 2.4, 2.9 all becoming "R = 2")
        if abs(sval - round(sval)) < 1e-9:
            return f"R = {int(round(sval))}"
        return f"R = {sval:.1f}"

    N_BINS_DESIGN = 40
    DESIGN_QLOW, DESIGN_QHIGH = 0.02, 0.98
    DESIGN_MIN_COUNT = 5
    DESIGN_SMOOTH_SIGMA = 1.8      # try 1.6–2.2
    RESAMPLE_N = 240               # finer x-grid
    RESAMPLE_SIGMA = 0.6           # tiny extra polish after resampling
    
    for s in plot_order:
        sub = df[df["Contrast"] == float(s)].copy()
        if sub.empty:
            continue
        
        res = design_curve_binned(
            sub,
            n_bins=N_BINS_DESIGN,
            q_low=DESIGN_QLOW,
            q_high=DESIGN_QHIGH,
            min_count=DESIGN_MIN_COUNT,
            smooth_sigma=DESIGN_SMOOTH_SIGMA,
        )
        if res is None:
            continue
        
        da_axis, y_smooth, count, w_smooth = res
        
        # optional: print counts for low/high slices
        if float(s) in (1.2, 115.8):
            print(f"[DesignCurve count] R={float(s):g} | min={int(count.min())} | mean={count.mean():.2f} | max={int(count.max())}")
        
        # resample for a visually smooth curve (removes polyline look)
        x_f, y_f = _resample_logx(da_axis, y_smooth, n=RESAMPLE_N, sigma=RESAMPLE_SIGMA)
        if x_f.size < 4:
            continue
        lbl = _fmt_R_label(float(s))

        is_extreme = (float(s) == R_min) or (float(s) == R_max)
        line_alpha = 1.0 if is_extreme else 0.7
        line_lw = 4.0 if is_extreme else 3.0
        line_z = 5 if is_extreme else 2
        col = color_map.get(float(s), colors[0])

        ax.plot(x_f, y_f, color=col, lw=line_lw, alpha=line_alpha, zorder=line_z, label=lbl)

    ax.set_xscale("log")
    ax.set_xlabel(r"Damköhler Number ($Da$)")
    ax.set_ylabel("Normalized Hysteresis Index")
    ax.set_title("B. Design Curves", fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Use a data-driven x-range for diagnostics; switch back to fixed later if you want
    da_lo_plot = max(df["Da"].quantile(0.001), 1e-12)
    da_hi_plot = df["Da"].quantile(0.999)
    ax.set_xlim(da_lo_plot, da_hi_plot)

    # Legend ordering: smallest R and largest R on top, then the remaining slices below
    handles, labels = ax.get_legend_handles_labels()
    handle_map = {}
    for h, l in zip(handles, labels):
        if l not in handle_map:
            handle_map[l] = h

    legend_order = [_fmt_R_label(R_min), _fmt_R_label(R_max)] + [_fmt_R_label(s) for s in mid_slices]
    ordered_labels = [l for l in legend_order if l in handle_map]
    ordered_handles = [handle_map[l] for l in ordered_labels]

    ax.legend(ordered_handles, ordered_labels, loc="upper right")

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    df, ridge = main()
