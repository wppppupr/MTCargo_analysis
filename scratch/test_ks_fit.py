#!/usr/bin/env python3
"""
Test KS-minimization for x_min and N(X>=x) >= 5 for x_max.
"""
import sys
from pathlib import Path
current_dir = Path(__file__).resolve().parent.parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

import glob
import numpy as np
import pandas as pd
from scipy.stats import linregress

from libs import displacement as dpm

root = Path('/mnt/NAS-Ebanaru/sasaki/MTsingleBeads')
beads = [
    {"name": "beads06um", "diameter_um": 0.63, "marker": "^"},
    {"name": "beads1um",  "diameter_um": 1.18, "marker": "o"},
    {"name": "beads3um",  "diameter_um": 3.37, "marker": "d"},
    {"name": "beads5um",  "diameter_um": 5.00, "marker": "p"},
    {"name": "beads7um",  "diameter_um": 7.24, "marker": "h"},
    {"name": "beads20um", "diameter_um": 20.0, "marker": "s"},
]

def fit_tail_ks_min(x_vals, min_n_tail=10):
    """
    x_max: N(X >= x) >= 5 (i.e. x_vals[-5])
    x_min: Selected by Kolmogorov-Smirnov distance minimization.
    """
    sorted_x = np.sort(x_vals[np.isfinite(x_vals) & (x_vals > 0)])
    N = len(sorted_x)
    if N < min_n_tail + 5:
        return None

    # x_max: 累積カウント >= 5 件残っている最大値
    idx_max = N - 5
    x_max = sorted_x[idx_max]

    # x_min candidate search
    # candidates from median up to index idx_max - min_n_tail
    idx_min_start = int(N * 0.3)
    idx_min_end = idx_max - min_n_tail
    if idx_min_start >= idx_min_end:
        idx_min_start = max(0, idx_min_end - 50)

    best_D = np.inf
    best_xmin = None
    best_alpha = None
    best_idx = None

    # Step through unique candidate values to speed up
    cand_indices = np.linspace(idx_min_start, idx_min_end, min(200, idx_min_end - idx_min_start + 1), dtype=int)
    cand_indices = np.unique(cand_indices)

    for idx in cand_indices:
        x_min_cand = sorted_x[idx]
        sub = sorted_x[(sorted_x >= x_min_cand) & (sorted_x <= x_max)]
        n = len(sub)
        if n < min_n_tail:
            continue

        # MLE for power-law exponent alpha: P(x) ~ x^(-alpha)
        # alpha = 1 + n / sum(ln(x_i / x_min))
        log_ratios = np.log(sub / x_min_cand)
        sum_log = np.sum(log_ratios)
        if sum_log <= 0:
            continue
        alpha = 1.0 + n / sum_log

        # Empirical CDF on [x_min, x_max]
        emp_cdf = np.arange(1, n + 1) / float(n)
        # Theoretical model CDF: 1 - (x / x_min)^(-(alpha - 1))
        theo_cdf = 1.0 - (sub / x_min_cand) ** (-(alpha - 1.0))
        theo_cdf = np.clip(theo_cdf, 0.0, 1.0)

        # KS distance
        D = np.max(np.abs(emp_cdf - theo_cdf))
        if D < best_D:
            best_D = D
            best_xmin = x_min_cand
            best_alpha = alpha
            best_idx = idx

    if best_xmin is None:
        return None

    # Now calculate CCDF and linear regression on [best_xmin, x_max]
    # Full empirical CCDF: P(X >= x) = (N - rank + 1) / N
    # For linear fit:
    sub_mask = (sorted_x >= best_xmin) & (sorted_x <= x_max)
    x_fit_pts = sorted_x[sub_mask]
    ranks = np.where(sub_mask)[0]
    y_fit_pts = (N - ranks) / float(N)

    log_x = np.log10(x_fit_pts)
    log_y = np.log10(y_fit_pts)

    res = linregress(log_x, log_y)
    slope = float(res.slope)
    mu = float(-res.slope)
    r2 = float(res.rvalue**2)
    stderr = float(res.stderr) if res.stderr is not None else 0.0

    return {
        'x_min': best_xmin,
        'x_max': x_max,
        'slope': slope,
        'mu': mu,
        'stderr': stderr,
        'r2': r2,
        'mle_alpha': best_alpha,
        'ks_D': best_D,
        'n_tail': len(x_fit_pts),
        'n_total': N,
        'intercept': float(res.intercept)
    }

for tau in [1, 25]:
    print(f"\n=== Tau = {tau} (dt = {tau * 4.0}s) ===")
    for b in beads:
        files = sorted(glob.glob(str(root / b["name"] / "*" / "*" / "beads_tracks.csv")))
        all_disps = []
        for f in files:
            df = pd.read_csv(f)
            disp = dpm.calc_displacement_magnitudes(df, tau=tau, scale=0.11, component='norm')
            all_disps.extend(disp)
        all_disps = np.array(all_disps)
        res = fit_tail_ks_min(all_disps)
        if res:
            print(f"{b['name']} ({b['diameter_um']} um): slope -mu = {res['slope']:.3f} (mu = {res['mu']:.3f} +/- {res['stderr']:.3f}, R^2={res['r2']:.3f}), KS D = {res['ks_D']:.4f}, range = [{res['x_min']:.2f}, {res['x_max']:.2f}] um (N_tail={res['n_tail']}/{res['n_total']})")
