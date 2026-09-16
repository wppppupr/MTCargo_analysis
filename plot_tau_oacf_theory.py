#!/usr/bin/env python3
"""
plot_tau_oacf_theory.py

tau_OACF の粒子径依存性 (log-log プロット) および理論モデルの可視化スクリプト。
理論式:
    tau_OACF(D_c) = tau_0 * exp(-2 * D_c / (3 * R_0))
    1 / tau_OACF(D_c) = (1 / tau_0) * exp(2 * D_c / (3 * R_0))

プロット内容:
  - tau_OACF 実測値 (0.63, 1.18, 3.37 um: 積分相関時間 tau_int, 指数フィッティング時間 tau_fit)
  - 理論曲線 (tau_0 * exp(-2 * D_c / (3 * R_0)))
  - べき乗則スケーリング補助線: D_c^-1 および D_c^-2
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.optimize import curve_fit
import pandas as pd

CURRENT_DIR = Path(__file__).parent.resolve()
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

# スタイルの適用
style_path = CURRENT_DIR / 'libs' / 'my_style.mplstyle'
if style_path.exists():
    try:
        plt.style.use(str(style_path))
    except Exception:
        pass


def load_parameters_and_data():
    # 1. Run 速度フィッティングパラメータ (R_0, v_0) の取得
    vel_path = CURRENT_DIR / "figure" / "velocity" / "run_velocity_summary.csv"
    if vel_path.exists():
        df_vel = pd.read_csv(vel_path)
        d_vel = df_vel['diameter_um'].values
        v_run_g = df_vel['v_run_geom_um_s'].values
        
        # Run速度の幾何平均フィットに基づくパラメータ (ln(v) = -1/(3*R0) * d + B)
        p_g = np.polyfit(d_vel, np.log(v_run_g), 1)
        three_R0 = -1.0 / p_g[0]  # three_R0 = 8.3321 um
        R0 = three_R0 / 3.0       # R0 = 2.7774 um
        v0 = np.exp(p_g[1])
    else:
        R0 = 2.7774
        three_R0 = 8.3321
        v0 = 0.2335

    # 2. OACF 実測データの取得
    acf_path = CURRENT_DIR / "figure" / "hmm_1d" / "hmm_autocorrelation_summary_k2.csv"
    df_acf = pd.read_csv(acf_path) if acf_path.exists() else pd.DataFrame()
    
    if not df_acf.empty:
        sub_oacf = df_acf[(df_acf['state'] == 1) & (df_acf['mode'] == 'oacf')].sort_values('diameter_um')
    else:
        sub_oacf = pd.DataFrame()

    return {
        "R0": R0, "three_R0": three_R0, "v0": v0,
        "df_oacf": sub_oacf
    }


def main():
    params = load_parameters_and_data()
    R0 = params["R0"]
    three_R0 = params["three_R0"]
    v0 = params["v0"]
    df_oacf = params["df_oacf"]

    out_dirs = [
        CURRENT_DIR / "figure" / "hmm_1d",
        CURRENT_DIR / "figure" / "msd",
        Path("/Volumes/data/Sasaki/MTsingleBeads/figure/msd"),
        Path("/Volumes/data/Sasaki/MTsingleBeads/figure/hmm_1d"),
    ]
    for d in out_dirs:
        try:
            if d.exists() or d.parent.exists():
                d.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    # 5, 7, 20 um を除外して 0.63, 1.18, 3.37 um のみ抽出 (Dc <= 3.5 um)
    if not df_oacf.empty:
        sub_3beads = df_oacf[df_oacf['diameter_um'] <= 3.5].copy()
        d_meas = sub_3beads['diameter_um'].values
        tau_int_meas = sub_3beads['tau_int_zero_s'].values
        tau_fit_meas = sub_3beads['tau_corr_s'].values
    else:
        d_meas = np.array([0.63, 1.18, 3.37])
        tau_int_meas = np.array([17.91, 9.24, 4.78])
        tau_fit_meas = np.array([19.2, 10.1, 5.0])

    d_dense = np.linspace(0.0, 25.0, 300)

    # 理論式: tau_p(Rc) = tau_0 * exp(-4 * Rc / (3 * xi)) = tau_0 * exp(-2 * (2Rc) / (3 * xi))
    # xi は plot_run_velocity.py でフィッティングして求めた値 (xi = 2.7774 um, 3xi/2 = 4.166 um) で固定
    # 3点の実測値から ln(y) 空間で tau_0 をフィッティング
    valid_int = np.isfinite(tau_int_meas) & (tau_int_meas > 0)
    if np.any(valid_int):
        ln_t0_vals = np.log(tau_int_meas[valid_int]) + (2.0 / (3.0 * R0)) * d_meas[valid_int]
        ln_t0_fitted = float(np.mean(ln_t0_vals))
        t0_fitted = float(np.exp(ln_t0_fitted))
    else:
        t0_fitted = 14.00

    tau_fit_curve = t0_fitted * np.exp(-2.0 * d_dense / (3.0 * R0))
    inv_tau_fit_curve = (1.0 / t0_fitted) * np.exp(2.0 * d_dense / (3.0 * R0))

    # =========================================================================
    # 図1: 2パネル比較 (左: 緩和速度 1/tau_p vs 2Rc (linear), 右: 緩和時間 tau_p vs 2Rc (semilog-y, x in [0, 25]))
    # =========================================================================
    fig_2p, (ax_rate, ax_time) = plt.subplots(1, 2, figsize=(16, 6.2))

    # --- 左パネル: 緩和速度 1/tau_p [s^-1] (Linear Scale, x in [0, 25]) ---
    ax_rate.plot(
        d_dense, inv_tau_fit_curve,
        color='#1f78b4', linestyle='-', linewidth=2.4,
        label=rf'Theory Fit: $\frac{{1}}{{\tau_0}} \exp\left(\frac{{4 R_c}}{{3 \xi}}\right)$' + '\n' + rf'  ($\tau_0 = {t0_fitted:.2f}\,\mathrm{{s}},\ \xi = {R0:.2f}\,\mu\mathrm{{m}}$)'
    )

    if len(d_meas) > 0:
        ax_rate.plot(
            d_meas[valid_int], 1.0 / tau_int_meas[valid_int],
            marker='o', color='#2b83ba', linestyle='none', markersize=9.0,
            label=r'Measured $1/\tau_{\mathrm{p}}^{\mathrm{int}}$', zorder=5
        )
        valid_fit = np.isfinite(tau_fit_meas) & (tau_fit_meas > 0)
        if np.any(valid_fit):
            ax_rate.plot(
                d_meas[valid_fit], 1.0 / tau_fit_meas[valid_fit],
                marker='^', color='#984ea3', linestyle='none', markersize=8.0,
                label=r'Measured $1/\tau_{\mathrm{p}}^{\mathrm{fit}}$', zorder=5
            )

    ax_rate.set_xlim(0, 25.0)
    ax_rate.set_ylim(0.0, 1.5)
    ax_rate.xaxis.set_major_locator(ticker.MultipleLocator(5.0))
    ax_rate.xaxis.set_minor_locator(ticker.MultipleLocator(1.0))
    ax_rate.set_xlabel(r'Cargo Diameter $2R_c$ [$\mu\mathrm{m}$]', fontsize=12, fontweight='bold')
    ax_rate.set_ylabel(r'Orientation Relaxation Rate $\tau_{\mathrm{p}}^{-1}$ [$\mathrm{s}^{-1}$]', fontsize=12, fontweight='bold')
    ax_rate.set_title(r'(a) Relaxation Rate $\frac{1}{\tau_{\mathrm{p}}} = \frac{1}{\tau_0} \exp\left(\frac{4 R_c}{3 \xi}\right)$', fontsize=13, fontweight='bold')
    ax_rate.grid(True, which='both', linestyle='--', alpha=0.4)
    ax_rate.legend(fontsize=9.0, loc='upper left', frameon=True, framealpha=0.92)

    # --- 右パネル: 緩和時間 tau_p [s] (Semilog-y, x in [0, 25]) ---
    ax_time.plot(
        d_dense, tau_fit_curve,
        color='#1f78b4', linestyle='-', linewidth=2.4,
        label=rf'Theory Fit: $\tau_0 \exp\left(-\frac{{4 R_c}}{{3 \xi}}\right)$' + '\n' + rf'  ($\tau_0 = {t0_fitted:.2f}\,\mathrm{{s}},\ \xi = {R0:.2f}\,\mu\mathrm{{m}}$)',
        zorder=3
    )

    # 実測データ点
    if len(d_meas) > 0:
        ax_time.plot(
            d_meas[valid_int], tau_int_meas[valid_int],
            marker='o', color='#2b83ba', linestyle='none', markersize=9.0,
            label=r'Measured $\tau_{\mathrm{p}}^{\mathrm{int}}$', zorder=5
        )
        if np.any(valid_fit):
            ax_time.plot(
                d_meas[valid_fit], tau_fit_meas[valid_fit],
                marker='^', color='#984ea3', linestyle='none', markersize=8.0,
                label=r'Measured $\tau_{\mathrm{p}}^{\mathrm{fit}}$', zorder=5
            )

    param_info = (
        f"Parameters from Run Velocity:\n"
        f"  $\\xi = {R0:.2f}\\,\\mu\\mathrm{{m}}$\n"
        f"  $v_0 = {v0:.3f}\\,\\mu\\mathrm{{m/s}}$\n"
        f"Fit in $\\ln(y)$ space:\n"
        f"  $\\tau_0 = {t0_fitted:.2f}\\,\\mathrm{{s}}$"
    )
    ax_time.text(
        0.04, 0.04, param_info,
        transform=ax_time.transAxes, verticalalignment='bottom',
        fontsize=9.0, bbox=dict(boxstyle='round,pad=0.35', facecolor='white', alpha=0.88, edgecolor='#cccccc')
    )

    ax_time.set_yscale('log')
    ax_time.set_xlim(0, 25.0)
    ax_time.set_ylim(0.01, 50.0)
    ax_time.xaxis.set_major_locator(ticker.MultipleLocator(5.0))
    ax_time.xaxis.set_minor_locator(ticker.MultipleLocator(1.0))
    ax_time.yaxis.set_major_locator(ticker.FixedLocator([0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 20, 50]))
    ax_time.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}"))
    ax_time.set_xlabel(r'Cargo Diameter $2R_c$ [$\mu\mathrm{m}$]', fontsize=12, fontweight='bold')
    ax_time.set_ylabel(r'Orientation Persistence Time $\tau_{\mathrm{p}}$ [s]', fontsize=12, fontweight='bold')
    ax_time.set_title(r'(b) Orientation Persistence Time $\tau_{\mathrm{p}}$ vs $2R_c$ (Log Scale)', fontsize=13, fontweight='bold')
    ax_time.grid(True, which='both', linestyle='--', alpha=0.4)
    ax_time.legend(fontsize=8.5, loc='upper right', frameon=True, framealpha=0.92)

    fig_2p.suptitle(
        r'Orientation Relaxation Model: $\tau_{\mathrm{p}}(R_c) = \tau_0 \exp\left(-\frac{4 R_c}{3 \xi}\right)$ ($x \in [0, 25]\,\mu\mathrm{m}$)',
        fontsize=15, fontweight='bold', y=0.98
    )
    fig_2p.tight_layout()

    for d in out_dirs:
        try:
            p_png = d / "tau_p_theory_2panel.png"
            p_svg = d / "tau_p_theory_2panel.svg"
            fig_2p.savefig(p_png, dpi=300, bbox_inches='tight')
            fig_2p.savefig(p_svg, bbox_inches='tight')
            # 互換用にも保存
            fig_2p.savefig(d / "tau_oacf_theory_2panel.png", dpi=300, bbox_inches='tight')
            fig_2p.savefig(d / "tau_oacf_theory_2panel.svg", bbox_inches='tight')
            print(f"Saved: {p_png}")
        except Exception:
            pass

    # =========================================================================
    # 図2: tau_p_vs_diameter 単体図 (Log y, x in [0, 25])
    # =========================================================================
    fig_single, ax_s = plt.subplots(figsize=(8.0, 5.8))
    ax_s.plot(
        d_dense, tau_fit_curve,
        color='#1f78b4', linestyle='-', linewidth=2.4,
        label=rf'Theory: $\tau_{{\mathrm{{p}}}}(R_c) = \tau_0 \exp\left(-\frac{{4 R_c}}{{3 \xi}}\right)$' + '\n' + rf'  ($\tau_0 = {t0_fitted:.2f}\,\mathrm{{s}},\ \xi = {R0:.2f}\,\mu\mathrm{{m}}$)',
        zorder=3
    )

    if len(d_meas) > 0:
        ax_s.plot(
            d_meas[valid_int], tau_int_meas[valid_int],
            marker='o', color='#2b83ba', linestyle='none', markersize=9.0,
            label=r'Measured $\tau_{\mathrm{p}}^{\mathrm{int}}$ (Orientation int. time)', zorder=5
        )
        for d_val, t_o in zip(d_meas[valid_int], tau_int_meas[valid_int]):
            ax_s.annotate(
                f"{t_o:.2f}s",
                (d_val, t_o),
                textcoords="offset points",
                xytext=(0, 9),
                ha='center',
                fontsize=9.0,
                fontweight='bold',
                color='#2b83ba',
                bbox=dict(boxstyle='round,pad=0.18', facecolor='white', edgecolor='#2b83ba', alpha=0.9)
            )

    ax_s.set_yscale('log')
    ax_s.set_xlim(0, 25.0)
    ax_s.set_ylim(0.01, 50.0)
    ax_s.xaxis.set_major_locator(ticker.MultipleLocator(5.0))
    ax_s.xaxis.set_minor_locator(ticker.MultipleLocator(1.0))
    ax_s.yaxis.set_major_locator(ticker.FixedLocator([0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 20, 50]))
    ax_s.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}"))
    ax_s.set_xlabel(r'Cargo Diameter $2R_c$ [$\mu\mathrm{m}$]', fontsize=12, fontweight='bold')
    ax_s.set_ylabel(r'Orientation Persistence Time $\tau_{\mathrm{p}}$ [s]', fontsize=12, fontweight='bold')
    ax_s.set_title(r'Orientation Persistence Time $\tau_{\mathrm{p}}$ vs Cargo Diameter' + '\n' + rf'($\tau_0 = {t0_fitted:.2f}\,\mathrm{{s}},\ \xi = {R0:.2f}\,\mu\mathrm{{m}},\ x \in [0, 25]\,\mu\mathrm{{m}}$)', fontsize=12, fontweight='bold', pad=10)
    ax_s.grid(True, which='both', linestyle='--', alpha=0.4)
    ax_s.legend(fontsize=9.0, loc='upper right', frameon=True, framealpha=0.92)

    fig_single.tight_layout()
    for d in out_dirs:
        try:
            p_png = d / "tau_p_vs_diameter.png"
            p_svg = d / "tau_p_vs_diameter.svg"
            fig_single.savefig(p_png, dpi=300, bbox_inches='tight')
            fig_single.savefig(p_svg, bbox_inches='tight')
            fig_single.savefig(d / "tau_oacf_vs_diameter.png", dpi=300, bbox_inches='tight')
            fig_single.savefig(d / "tau_oacf_vs_diameter.svg", bbox_inches='tight')
            print(f"Saved: {p_png}")
        except Exception:
            pass

    plt.close('all')
    print("\n[Done] Successfully generated all tau_p theoretical plots!")


if __name__ == "__main__":
    main()
