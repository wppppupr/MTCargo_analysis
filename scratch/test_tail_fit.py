#!/usr/bin/env python3
"""
Test CCDF calculation and tail power-law fitting for cargo displacements.
"""
import sys
from pathlib import Path
current_dir = Path(__file__).resolve().parent.parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

from libs import displacement as dpm

root = Path('/mnt/NAS-Ebanaru/sasaki/MTsingleBeads')
beads = [
    {"name": "beads06um", "diameter_um": 0.63, "marker": "^", "color": "#1f77b4"},
    {"name": "beads1um",  "diameter_um": 1.18, "marker": "o", "color": "#ff7f0e"},
    {"name": "beads3um",  "diameter_um": 3.37, "marker": "d", "color": "#2ca02c"},
    {"name": "beads5um",  "diameter_um": 5.00, "marker": "p", "color": "#d62728"},
    {"name": "beads7um",  "diameter_um": 7.24, "marker": "h", "color": "#9467bd"},
    {"name": "beads20um", "diameter_um": 20.0, "marker": "s", "color": "#8c564b"},
]

def calc_ccdf(data):
    sorted_data = np.sort(data)
    n = len(sorted_data)
    # Complementary CDF: P(X >= x)
    ccdf = (n - np.arange(n)) / float(n)
    return sorted_data, ccdf

def fit_powerlaw_tail(x_vals, y_vals, p_max=0.15, p_min=1e-3, min_points=5):
    mask = (y_vals <= p_max) & (y_vals >= p_min) & (x_vals > 0)
    if np.sum(mask) < min_points:
        # fallback to top 20%
        idx_start = int(len(x_vals) * 0.8)
        mask = np.zeros(len(x_vals), dtype=bool)
        mask[idx_start:] = True
        
    x_sub = x_vals[mask]
    y_sub = y_vals[mask]
    
    log_x = np.log10(x_sub)
    log_y = np.log10(y_sub)
    
    res = linregress(log_x, log_y)
    # slope = -mu
    mu = -res.slope
    slope = res.slope
    intercept = res.intercept
    r2 = res.rvalue**2
    
    # fit curve
    x_fit = np.logspace(np.log10(x_sub[0]), np.log10(x_sub[-1]), 100)
    y_fit = 10**(slope * np.log10(x_fit) + intercept)
    
    return {
        'mu': mu,
        'slope': slope,
        'intercept': intercept,
        'r2': r2,
        'p_value': res.pvalue,
        'stderr': res.stderr,
        'x_min': x_sub[0],
        'x_max': x_sub[-1],
        'fit_x': x_fit,
        'fit_y': y_fit,
        'n_points': len(x_sub)
    }

print("Calculating CCDFs and tail fits for tau=1 (4s):")
for b in beads:
    files = sorted(glob.glob(str(root / b["name"] / "*" / "*" / "beads_tracks.csv")))
    all_disps = []
    for f in files:
        df = pd.read_csv(f)
        disp = dpm.calc_displacement_magnitudes(df, tau=1, scale=0.11, component='norm')
        all_disps.extend(disp)
    all_disps = np.array(all_disps)
    all_disps = all_disps[all_disps > 0]
    
    x_ccdf, y_ccdf = calc_ccdf(all_disps)
    tail_fit = fit_powerlaw_tail(x_ccdf, y_ccdf, p_max=0.10, p_min=1e-3)
    print(f"{b['name']} ({b['diameter_um']} um): Tail slope -mu = {tail_fit['slope']:.3f} (mu = {tail_fit['mu']:.3f} +/- {tail_fit['stderr']:.3f}), R^2 = {tail_fit['r2']:.3f}, range = [{tail_fit['x_min']:.2f}, {tail_fit['x_max']:.2f}] um (N_tail={tail_fit['n_points']})")
