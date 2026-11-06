#!/usr/bin/env python3
"""
scratch/analyze_xi_robustness.py

相関長 xi の定義の頑健性（指数フィット、C=1/e の距離、積分相関長）および時間変化の可視化と集計。
補足資料用プロットおよび集計表を出力する。
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

OUT_DIR = ROOT / "figure" / "xi_robustness"
OUT_DIR.mkdir(parents=True, exist_ok=True)

try:
    plt.style.use(str(ROOT / 'libs' / 'my_style.mplstyle'))
except Exception:
    pass

ORDER = ["beads06um", "beads1um", "beads3um", "beads5um", "beads7um", "beads20um"]
COLORS = {
    'control': 'black',
    'beads06um': '#882255',
    'beads1um': '#CC6677',
    'beads3um': '#DDCC77',
    'beads5um': '#999933',
    'beads7um': '#117733',
    'beads20um': '#44AA99',
}
MARKERS = {
    'control': 'x',
    'beads06um': '^',
    'beads1um': 'o',
    'beads3um': 'd',
    'beads5um': 'p',
    'beads7um': 'h',
    'beads20um': 's',
}
NAMES = {
    'control': 'w/o Cargo',
    'beads06um': '0.63 µm',
    'beads1um': '1.18 µm',
    'beads3um': '3.37 µm',
    'beads5um': '5.00 µm',
    'beads7um': '7.24 µm',
    'beads20um': '20.0 µm',
}


def analyze_definitions():
    pts = pd.read_csv(ROOT / 'figure' / 'bg_angular_correlation' / 'bg_angular_correlation_points.csv')
    per_exp = pd.read_csv(ROOT / 'figure' / 'bg_angular_correlation' / 'bg_angular_correlation_length_per_experiment.csv')
    
    rows = []
    c_target = 1.0 / np.e
    
    for exp, g in pts.groupby('exp_dir', sort=False):
        g = g.sort_values('distance_um')
        r = g['distance_um'].to_numpy(dtype=float)
        c = g['mean_c'].to_numpy(dtype=float)
        bead = g['bead_name'].iloc[0]
        diam = g['diameter_um'].iloc[0]
        
        # 1. Exponential fit xi
        xi_fit = float(per_exp[per_exp['exp_dir'] == exp]['xi_um'].iloc[0])
        xi_fit_err = float(per_exp[per_exp['exp_dir'] == exp]['xi_err_um'].iloc[0])
        r2_fit = float(per_exp[per_exp['exp_dir'] == exp]['xi_r2_log'].iloc[0])
        
        # 2. C(r) = 1/e distance
        if c[0] >= c_target and np.any(c <= c_target):
            idx = np.where(c <= c_target)[0][0]
            r0, r1 = r[idx-1], r[idx]
            c0, c1 = c[idx-1], c[idx]
            r_1oe = float(r0 + (c_target - c0) * (r1 - r0) / (c1 - c0))
        else:
            r_1oe = np.nan
            
        # 3. Integral correlation length (up to 20 um)
        sub20 = r <= 20.0
        xi_int_20 = float(trapezoid(c[sub20], r[sub20]))
        
        # 4. Integral up to first zero crossing
        zero_cross = np.where(c <= 0)[0]
        if len(zero_cross) > 0:
            i_cut = zero_cross[0]
            r_sub = list(r[:i_cut])
            c_sub = list(c[:i_cut])
            r_zero = r[i_cut-1] + (0.0 - c[i_cut-1]) * (r[i_cut] - r[i_cut-1]) / (c[i_cut] - c[i_cut-1])
            r_sub.append(r_zero)
            c_sub.append(0.0)
            xi_int_zero = float(trapezoid(c_sub, r_sub))
        else:
            xi_int_zero = float(trapezoid(c, r))
            
        Rc = (diam / 2.0) if np.isfinite(diam) else np.nan
        rows.append({
            'bead_name': bead,
            'display_name': NAMES.get(bead, bead),
            'diameter_um': diam,
            'radius_um': Rc,
            'exp_dir': exp,
            'xi_fit_um': xi_fit,
            'xi_fit_err_um': xi_fit_err,
            'xi_fit_r2_log': r2_fit,
            'r_1oe_um': r_1oe,
            'xi_int_20_um': xi_int_20,
            'xi_int_zero_um': xi_int_zero,
            'Rc_over_xi_fit': Rc / xi_fit if np.isfinite(Rc) else np.nan,
            'Rc_over_xi_1oe': Rc / r_1oe if np.isfinite(Rc) else np.nan,
            'Rc_over_xi_int20': Rc / xi_int_20 if np.isfinite(Rc) else np.nan,
            'Rc_over_xi_intzero': Rc / xi_int_zero if np.isfinite(Rc) else np.nan,
        })
        
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "xi_definitions_per_experiment.csv", index=False)
    
    # Condition-level summary
    summary_rows = []
    for bead in ["control"] + ORDER:
        sub = df[df.bead_name == bead]
        if sub.empty: continue
        diam = sub['diameter_um'].iloc[0]
        Rc = sub['radius_um'].iloc[0]
        n_exp = len(sub)
        summary_rows.append({
            'bead_name': bead,
            'display_name': NAMES.get(bead, bead),
            'diameter_um': diam,
            'radius_um': Rc,
            'n_experiments': n_exp,
            'xi_fit_mean': sub['xi_fit_um'].mean(),
            'xi_fit_sem': sub['xi_fit_um'].std(ddof=1) / np.sqrt(n_exp) if n_exp > 1 else 0.0,
            'r_1oe_mean': sub['r_1oe_um'].mean(),
            'r_1oe_sem': sub['r_1oe_um'].std(ddof=1) / np.sqrt(n_exp) if n_exp > 1 else 0.0,
            'xi_int_20_mean': sub['xi_int_20_um'].mean(),
            'xi_int_20_sem': sub['xi_int_20_um'].std(ddof=1) / np.sqrt(n_exp) if n_exp > 1 else 0.0,
            'xi_int_zero_mean': sub['xi_int_zero_um'].mean(),
            'xi_int_zero_sem': sub['xi_int_zero_um'].std(ddof=1) / np.sqrt(n_exp) if n_exp > 1 else 0.0,
            'Rc_over_xi_fit_mean': sub['Rc_over_xi_fit'].mean(),
            'Rc_over_xi_fit_sem': sub['Rc_over_xi_fit'].std(ddof=1) / np.sqrt(n_exp) if n_exp > 1 else 0.0,
            'Rc_over_xi_1oe_mean': sub['Rc_over_xi_1oe'].mean(),
            'Rc_over_xi_1oe_sem': sub['Rc_over_xi_1oe'].std(ddof=1) / np.sqrt(n_exp) if n_exp > 1 else 0.0,
            'Rc_over_xi_int20_mean': sub['Rc_over_xi_int20'].mean(),
            'Rc_over_xi_int20_sem': sub['Rc_over_xi_int20'].std(ddof=1) / np.sqrt(n_exp) if n_exp > 1 else 0.0,
        })
    sum_df = pd.DataFrame(summary_rows)
    sum_df.to_csv(OUT_DIR / "xi_definitions_summary.csv", index=False)
    
    # 1. Comparison Scatter Plot (xi_fit vs r_1oe vs xi_int)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharex=False, sharey=False)
    
    # Panel A: r_1oe vs xi_fit
    ax = axes[0]
    for bead in ["control"] + ORDER:
        sub = df[df.bead_name == bead]
        ax.scatter(sub['xi_fit_um'], sub['r_1oe_um'], color=COLORS[bead], marker=MARKERS[bead],
                   s=65, label=NAMES[bead], alpha=0.9, zorder=4)
    lims = [2.0, 14.0]
    ax.plot(lims, lims, 'k--', lw=1.2, alpha=0.6, label='Identity (1:1)')
    r_pearson = df[['xi_fit_um', 'r_1oe_um']].corr().iloc[0, 1]
    r_spearman = df[['xi_fit_um', 'r_1oe_um']].corr(method='spearman').iloc[0, 1]
    ax.text(0.05, 0.92, f"Pearson $r = {r_pearson:.3f}$\nSpearman $\\rho = {r_spearman:.3f}$",
            transform=ax.transAxes, fontsize=10.5, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))
    ax.set_xlabel(r"Exponential Fit $\xi_{\mathrm{fit}}$ [$\mu$m]", fontsize=12)
    ax.set_ylabel(r"$C(r) = 1/e$ Crossing Distance $r_{1/e}$ [$\mu$m]", fontsize=12)
    ax.set_title(r"(a) $r_{1/e}$ vs $\xi_{\mathrm{fit}}$ across 25 movies", fontsize=12.5, fontweight='bold')
    ax.grid(True, ls='--', alpha=0.4)
    ax.legend(fontsize=9, loc='lower right', framealpha=0.9)
    
    # Panel B: Rc / xi comparison across definitions
    ax = axes[1]
    cargo_sum = sum_df[sum_df.bead_name != "control"].copy()
    x = np.arange(len(cargo_sum))
    w = 0.25
    ax.bar(x - w, cargo_sum['Rc_over_xi_fit_mean'], yerr=cargo_sum['Rc_over_xi_fit_sem'], width=w,
           label=r"$\xi_{\mathrm{fit}}$ (Exponential fit)", color='#4477AA', capsize=3, alpha=0.85)
    ax.bar(x, cargo_sum['Rc_over_xi_1oe_mean'], yerr=cargo_sum['Rc_over_xi_1oe_sem'], width=w,
           label=r"$r_{1/e}$ ($C=1/e$ crossing)", color='#EE6677', capsize=3, alpha=0.85)
    ax.bar(x + w, cargo_sum['Rc_over_xi_int20_mean'], yerr=cargo_sum['Rc_over_xi_int20_sem'], width=w,
           label=r"$\xi_{\mathrm{int}}$ (Integral $\int_0^{20} C dr$)", color='#228833', capsize=3, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(cargo_sum['display_name'], fontsize=11)
    ax.set_xlabel("Cargo Diameter", fontsize=12)
    ax.set_ylabel(r"Dimensionless Ratio $R_c / \xi$", fontsize=12)
    ax.set_title(r"(b) Robustness of $R_c/\xi$ across $\xi$ definitions", fontsize=12.5, fontweight='bold')
    ax.grid(True, ls='--', alpha=0.4, axis='y')
    ax.legend(fontsize=9.5, loc='upper left', framealpha=0.9)
    
    # Note on 5um vs 7um inversion
    ax.annotate("5 µm > 7 µm\nin all definitions",
                xy=(3.0, cargo_sum.loc[cargo_sum.bead_name=='beads5um', 'Rc_over_xi_fit_mean'].iloc[0]),
                xytext=(2.7, 1.2),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.0),
                fontsize=9.5, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.2, edgecolor='orange'))
    
    fig.tight_layout()
    fig.savefig(OUT_DIR / "xi_definitions_comparison.png", dpi=300)
    fig.savefig(OUT_DIR / "xi_definitions_comparison.svg")
    plt.close(fig)
    print("Saved xi_definitions_comparison figure")


def analyze_time_window():
    win_file = ROOT / "figure" / "xi_time_window" / "xi_t_windows_w200.csv"
    if not win_file.exists():
        print(f"Warning: {win_file} does not exist.")
        return
    win = pd.read_csv(win_file)
    
    # 2. Time trajectory plot for each condition
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
    axes = axes.flatten()
    
    for idx, bead in enumerate(ORDER):
        ax = axes[idx]
        sub = win[win.bead_name == bead]
        ax.set_title(f"{NAMES[bead]} ($2R_c = {sub['diameter_um'].iloc[0]:.2f}\,\mu$m)", fontsize=11.5, fontweight='bold')
        
        for exp, g in sub.groupby('exp_dir'):
            g = g.sort_values('win_start')
            t_mid = (g['win_start'] + g['win_end']) / 2.0
            
            # Good fits vs poor fits
            good = g['r2_log'] >= 0.90
            poor = ~good
            
            # Line connecting all points
            ax.plot(t_mid, g['xi_um'], color=COLORS[bead], alpha=0.45, lw=1.2, ls='-')
            
            # Scatter for good fits
            if good.any():
                ax.scatter(t_mid[good], g.loc[good, 'xi_um'], color=COLORS[bead], marker=MARKERS[bead],
                           s=50, edgecolors='k', linewidth=0.6, alpha=0.9, label=exp if not poor.any() else None)
            # Scatter for poor fits (r2_log < 0.9)
            if poor.any():
                ax.scatter(t_mid[poor], g.loc[poor, 'xi_um'], color='gray', marker='x',
                           s=60, linewidth=1.5, alpha=0.8, label=f"{exp} ($R^2<0.9$)")
                
            # Movie average reference
            ax.axhline(g['xi_movie_um'].iloc[0], color=COLORS[bead], ls=':', alpha=0.4, lw=1.0)
            
        ax.set_ylabel(r"$\xi(t)$ [$\mu$m]", fontsize=11)
        ax.set_xlabel("Frame number", fontsize=11)
        ax.set_ylim(0, 20)
        ax.grid(True, ls='--', alpha=0.35)
        # Legend with unique labels
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if by_label:
            ax.legend(by_label.values(), by_label.keys(), fontsize=7.5, loc='upper right', framealpha=0.8)
            
    fig.suptitle(r"Temporal Evolution of Background Correlation Length $\xi(t)$ (200-frame windows)",
                 fontsize=14, fontweight='bold', y=0.99)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "xi_t_trajectories_by_condition.png", dpi=300)
    fig.savefig(OUT_DIR / "xi_t_trajectories_by_condition.svg")
    plt.close(fig)
    print("Saved xi_t_trajectories_by_condition figure")


if __name__ == '__main__':
    analyze_definitions()
    analyze_time_window()
