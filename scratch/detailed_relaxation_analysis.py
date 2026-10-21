#!/usr/bin/env python3
"""
detailed_relaxation_analysis.py

各手法のMSDフィッティング、局所傾き alpha(t) の比較、および
緩和時間 tau vs 粒子径 2R_c の可視化・詳細評価スクリプト。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
from pathlib import Path

# libs スタイル
style_path = Path('libs/my_style.mplstyle')
if style_path.exists():
    plt.style.use(str(style_path))

try:
    style_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
except Exception:
    style_colors = ['#882255', '#CC6677', '#DDCC77', '#999933', '#117733', '#44AA99', '#88CCEE', '#332288']

BEADS_INFO = [
    {"name": "beads06um", "diameter_um": 0.63, "marker": "^", "color": style_colors[0]},
    {"name": "beads1um",  "diameter_um": 1.18, "marker": "o", "color": style_colors[1]},
    {"name": "beads3um",  "diameter_um": 3.37, "marker": "d", "color": style_colors[2]},
    {"name": "beads5um",  "diameter_um": 5.00, "marker": "p", "color": style_colors[3]},
    {"name": "beads7um",  "diameter_um": 7.24, "marker": "h", "color": style_colors[4]},
    {"name": "beads20um", "diameter_um": 20.0, "marker": "s", "color": style_colors[5]},
]

def log_pade_msd(log_t, log_v0_sq, log_tau):
    t = 10.0 ** log_t
    tau = 10.0 ** log_tau
    v0_sq = 10.0 ** log_v0_sq
    msd = v0_sq * (t ** 2) / (1.0 + t / tau)
    return np.log10(np.maximum(msd, 1e-12))

def alpha_pade(t, tau):
    return (2.0 * tau + t) / (tau + t)

def log_abp_msd(log_t, log_Deff, log_tau):
    t = 10.0 ** log_t
    tau = 10.0 ** log_tau
    Deff = 10.0 ** log_Deff
    u = t / tau
    small_u = u < 1e-4
    active_term = np.zeros_like(u)
    active_term[small_u] = tau * (0.5 * u[small_u]**2 - (u[small_u]**3)/6.0)
    active_term[~small_u] = t[~small_u] - tau * (1.0 - np.exp(-u[~small_u]))
    msd = 4.0 * Deff * active_term
    return np.log10(np.maximum(msd, 1e-12))

def alpha_abp(t, tau):
    u = t / tau
    # t * (1 - exp(-u)) / [t - tau * (1 - exp(-u))] = u * (1 - exp(-u)) / [u - (1 - exp(-u))]
    small = u < 1e-4
    res = np.zeros_like(u)
    res[small] = 2.0 - u[small] / 3.0
    u_l = u[~small]
    exp_u = np.exp(-u_l)
    res[~small] = u_l * (1.0 - exp_u) / (u_l - 1.0 + exp_u)
    return res

def main():
    df = pd.read_csv('figure/msd/msd_pooled_mean_iqr_summary.csv')
    out_dir = Path('figure/msd')
    out_dir.mkdir(parents=True, exist_ok=True)
    
    fit_summary = []
    
    # 図 1: 6パネル MSD とフィッティング (Padé vs ABP)
    fig_msd, axes_msd = plt.subplots(2, 3, figsize=(15, 9), sharex=True, sharey=True)
    axes_msd = axes_msd.flatten()

    # 図 2: 6パネル 局所傾き alpha(t) の比較 (Raw gradient vs Padé vs ABP vs Spline)
    fig_alpha, axes_alpha = plt.subplots(2, 3, figsize=(15, 9), sharex=True, sharey=True)
    axes_alpha = axes_alpha.flatten()

    for idx, b_info in enumerate(BEADS_INFO):
        b = b_info["name"]
        d_um = b_info["diameter_um"]
        color = b_info["color"]
        marker = b_info["marker"]
        
        sub = df[df['bead_name'] == b].sort_values('lag_time_s')
        # 1000s以内で粒子数2以上
        sub = sub[(sub['lag_time_s'] <= 1000) & (sub['n_active_particles'] >= 2) & (sub['mean_msd_um2'] > 0)]
        t = sub['lag_time_s'].values
        msd = sub['mean_msd_um2'].values
        log_t = np.log10(t)
        log_msd = np.log10(msd)
        
        # 生微分
        alpha_raw = np.gradient(log_msd, log_t)
        
        # Padé
        p0_pade = [np.log10(msd[0] / (t[0]**2)), np.log10(50.0)]
        popt_p, pcov_p = curve_fit(log_pade_msd, log_t, log_msd, p0=p0_pade, maxfev=5000)
        tau_p = 10.0 ** popt_p[1]
        tau_p_err = np.log(10.0) * tau_p * np.sqrt(pcov_p[1, 1])
        v0_sq_p = 10.0 ** popt_p[0]
        r2_p = 1.0 - np.sum((log_msd - log_pade_msd(log_t, *popt_p))**2) / np.sum((log_msd - np.mean(log_msd))**2)

        # ABP
        p0_abp = [np.log10(msd[-1] / (4.0 * t[-1])), np.log10(30.0)]
        popt_a, pcov_a = curve_fit(log_abp_msd, log_t, log_msd, p0=p0_abp, maxfev=5000)
        tau_a = 10.0 ** popt_a[1]
        tau_a_err = np.log(10.0) * tau_a * np.sqrt(pcov_a[1, 1])
        Deff_a = 10.0 ** popt_a[0]
        r2_a = 1.0 - np.sum((log_msd - log_abp_msd(log_t, *popt_a))**2) / np.sum((log_msd - np.mean(log_msd))**2)
        tau_a_cross = 2.14914 * tau_a

        # 平滑化スプライン (UnivariateSpline)
        spline = UnivariateSpline(log_t, log_msd, s=0.003 * len(log_t), k=3)
        alpha_spline = spline.derivative()(log_t)

        fit_summary.append({
            'bead_name': b,
            'diameter_um': d_um,
            'tau_pade_s': tau_p,
            'tau_pade_err_s': tau_p_err,
            'v0_sq_pade': v0_sq_p,
            'r2_pade': r2_p,
            'tau_abp_s': tau_a,
            'tau_abp_err_s': tau_a_err,
            'tau_abp_cross_s': tau_a_cross,
            'Deff_abp_um2_s': Deff_a,
            'r2_abp': r2_a,
        })

        # --- プロット 1: MSD ---
        ax_m = axes_msd[idx]
        # 生データ点 (IQR バンド付き)
        if 'q25_msd_um2' in sub.columns and 'q75_msd_um2' in sub.columns:
            ax_m.fill_between(t, sub['q25_msd_um2'], sub['q75_msd_um2'], color=color, alpha=0.15, edgecolor='none')
        ax_m.plot(t, msd, 'o', color=color, markersize=5, alpha=0.7, label='Exp. MSD')
        
        t_dense = np.logspace(np.log10(t[0]), np.log10(t[-1]), 200)
        msd_p_dense = 10.0 ** log_pade_msd(np.log10(t_dense), *popt_p)
        msd_a_dense = 10.0 ** log_abp_msd(np.log10(t_dense), *popt_a)
        
        ax_m.plot(t_dense, msd_p_dense, '-', color='#111111', lw=1.8, label=rf'Padé ($\tau={tau_p:.1f}\,\mathrm{{s}}$, $R^2={r2_p:.3f}$)')
        ax_m.plot(t_dense, msd_a_dense, '--', color='#d62728', lw=1.5, label=rf'ABP/OU ($\tau={tau_a:.1f}\,\mathrm{{s}}$, $R^2={r2_a:.3f}$)')
        
        # 緩和時間 tau_p の位置に縦線
        ax_m.axvline(tau_p, color='#111111', linestyle=':', alpha=0.6)
        
        ax_m.set_xscale('log')
        ax_m.set_yscale('log')
        ax_m.set_title(rf'{d_um:.2f} $\mu\mathrm{{m}}$ ({b})', fontsize=11, fontweight='bold')
        ax_m.set_xlabel(r'Lag time $\Delta t$ [s]', fontsize=10)
        ax_m.set_ylabel(r'MSD [$\mu\mathrm{m}^2$]', fontsize=10)
        ax_m.grid(True, which='both', ls='--', alpha=0.3)
        ax_m.legend(fontsize=8, loc='lower right', framealpha=0.9)

        # --- プロット 2: 局所傾き alpha(t) ---
        ax_a = axes_alpha[idx]
        ax_a.plot(t, alpha_raw, color='tab:blue', alpha=0.35, lw=1.0, label='Raw gradient (Noisy)')
        ax_a.plot(t, alpha_spline, color='tab:purple', linestyle='-.', lw=1.5, alpha=0.75, label='Smoothing Spline')
        
        alpha_p_dense = alpha_pade(t_dense, tau_p)
        alpha_a_dense = alpha_abp(t_dense, tau_a)
        
        ax_a.plot(t_dense, alpha_p_dense, '-', color='#111111', lw=2.0, label=rf'Padé model ($\tau={tau_p:.1f}\,\mathrm{{s}}$)')
        ax_a.plot(t_dense, alpha_a_dense, '--', color='#d62728', lw=1.8, label=rf'ABP model ($\tau_{{cross}}={tau_a_cross:.1f}\,\mathrm{{s}}$)')
        
        # 基準線
        ax_a.axhline(2.0, color='gray', linestyle='--', alpha=0.5)
        ax_a.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
        ax_a.axhline(1.5, color='orange', linestyle=':', lw=1.5, alpha=0.8)
        ax_a.axvline(tau_p, color='#111111', linestyle=':', alpha=0.6)
        
        ax_a.set_xscale('log')
        ax_a.set_ylim(0.4, 2.3)
        ax_a.set_title(rf'{d_um:.2f} $\mu\mathrm{{m}}$: Local Exponent $\alpha(t)$', fontsize=11, fontweight='bold')
        ax_a.set_xlabel(r'Lag time $\Delta t$ [s]', fontsize=10)
        ax_a.set_ylabel(r'$\alpha(t) = d\ln\mathrm{MSD}/d\ln\Delta t$', fontsize=10)
        ax_a.grid(True, which='both', ls='--', alpha=0.3)
        ax_a.legend(fontsize=7.5, loc='lower left', framealpha=0.9)

    fig_msd.suptitle(r'MSD Crossover Fitting: Ballistic ($\Delta t^2$) to Diffusive ($\Delta t^1$)', fontsize=14, fontweight='bold', y=0.99)
    fig_msd.tight_layout()
    fig_msd.savefig(out_dir / 'msd_crossover_fit_6panel.png', dpi=200, bbox_inches='tight')
    fig_msd.savefig(out_dir / 'msd_crossover_fit_6panel.svg', bbox_inches='tight')
    plt.close(fig_msd)

    fig_alpha.suptitle(r'Local Scaling Exponent $\alpha(t)$: Raw Gradient vs Smooth Theoretical Models', fontsize=14, fontweight='bold', y=0.99)
    fig_alpha.tight_layout()
    fig_alpha.savefig(out_dir / 'local_alpha_crossover_comparison_6panel.png', dpi=200, bbox_inches='tight')
    fig_alpha.savefig(out_dir / 'local_alpha_crossover_comparison_6panel.svg', bbox_inches='tight')
    plt.close(fig_alpha)

    # --- 図 3: 緩和時間 tau vs 粒子径 2R_c ---
    df_fit = pd.DataFrame(fit_summary)
    df_fit.to_csv(out_dir / 'msd_relaxation_time_summary.csv', index=False)
    
    fig_tau, ax_t = plt.subplots(figsize=(7.5, 5.5))
    d_arr = df_fit['diameter_um'].values
    tau_p_arr = df_fit['tau_pade_s'].values
    tau_p_err_arr = df_fit['tau_pade_err_s'].values
    tau_a_cross_arr = df_fit['tau_abp_cross_s'].values
    
    # Padé tau
    ax_t.errorbar(
        d_arr, tau_p_arr, yerr=tau_p_err_arr,
        fmt='o-', color='#1f77b4', ecolor='#1f77b4', elinewidth=1.8,
        capsize=4.5, capthick=1.2, markersize=8.0, lw=2.0,
        label=r'Padé Model $\tau_{\mathrm{Pade}}$ ($\alpha = 1.5$ exact)',
        zorder=4
    )
    
    # ABP crossover tau
    ax_t.plot(
        d_arr, tau_a_cross_arr, 's--', color='#d62728', markersize=7.5, lw=1.8,
        label=r'ABP/OU Model $t_{\mathrm{cross}} \approx 2.15\,\tau_{\mathrm{ABP}}$',
        zorder=3
    )

    ax_t.set_xscale('log')
    ax_t.set_yscale('log')
    ax_t.set_xlim(0.4, 28.0)
    ax_t.xaxis.set_major_locator(ticker.FixedLocator([0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0]))
    ax_t.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
    ax_t.set_xlabel(r'Cargo Diameter $2R_c$ [$\mu\mathrm{m}$]', fontsize=12, fontweight='bold')
    ax_t.set_ylabel(r'Relaxation / Crossover Time $\tau$ [s]', fontsize=12, fontweight='bold')
    ax_t.set_title(r'Ballistic-to-Diffusive Crossover Time $\tau$ vs Cargo Diameter', fontsize=12, fontweight='bold', pad=10)
    ax_t.grid(True, which='both', ls='--', alpha=0.35)
    ax_t.legend(fontsize=9.5, loc='upper right', framealpha=0.92)
    
    fig_tau.tight_layout()
    fig_tau.savefig(out_dir / 'relaxation_time_vs_diameter.png', dpi=200, bbox_inches='tight')
    fig_tau.savefig(out_dir / 'relaxation_time_vs_diameter.svg', bbox_inches='tight')
    plt.close(fig_tau)
    
    print("All plots and CSV saved successfully in figure/msd/!")
    print(df_fit[['bead_name', 'diameter_um', 'tau_pade_s', 'tau_pade_err_s', 'r2_pade', 'tau_abp_s', 'tau_abp_cross_s', 'r2_abp']].to_string(index=False))

if __name__ == '__main__':
    main()
