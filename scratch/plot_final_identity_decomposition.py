#!/usr/bin/env python3
"""
scratch/plot_final_identity_decomposition.py

恒等式に基づく速度分解プロット (Figure 9 改訂版):
<v_parallel> / <|u|>_{R_c} = [ <v_parallel> / <|<u|_{R_c}>| ] * [ <|<u|_{R_c}>| / <|u|>_{R_c} ]
                         = eta (力学的伝達効率) * A (足元幾何学的相殺)

分母に足元の局所ピクセルスカラー流速 <|u|>_{R_c} を用い、
粒子単位のブートストラップ (N_boot=2000) で誤差を付与。
さらに、2次元等方性乱流場の相関関数 C(r)=exp(-r/xi) から導かれる
幾何学的相殺の理論曲線 Phi(R_c/xi) を重ね描きする。
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import quad

CURRENT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(CURRENT_DIR))
import verify_cos_decomposition as vcd

ROOT_DIR = vcd.ROOT_DIR
SCALE = 0.11
DT = 4.0
V_UNIT = SCALE / DT

OUT_FIG_PNG = CURRENT_DIR / "figure" / "contrast_analysis" / "contrast_fig9_velocity_decomposition_identity.png"
OUT_FIG_SVG = CURRENT_DIR / "figure" / "contrast_analysis" / "contrast_fig9_velocity_decomposition_identity.svg"
OUT_FIG2_PNG = CURRENT_DIR / "figure" / "contrast_analysis" / "contrast_fig9b_velocity_decomposition_2panels.png"
OUT_FIG2_SVG = CURRENT_DIR / "figure" / "contrast_analysis" / "contrast_fig9b_velocity_decomposition_2panels.svg"
OUT_SUMMARY_PKL = CURRENT_DIR / "scratch" / "velocity_decomposition_identity_summary.pkl"


def calc_disk_polar_order(rc_over_xi):
    """
    円盤領域 (半径 Rc) 内で相関関数 C(r) = exp(-r/xi) を持つ
    2次元等方性ベクトル場を平均したときの局所ポーラーオーダー比 Phi(Rc/xi).
    Phi^2(y) = (2/pi) * int_0^2 x * [2*arccos(x/2) - x*sqrt(4 - x^2)] * exp(-y*x) dx
    """
    def integrand(x, y):
        if x >= 2.0:
            return 0.0
        k = 2.0 * np.arccos(x / 2.0) - x * np.sqrt(4.0 - x**2)
        return (2.0 / np.pi) * x * k * np.exp(-y * x)

    if np.isscalar(rc_over_xi):
        val, _ = quad(integrand, 0.0, 2.0, args=(rc_over_xi,))
        return np.sqrt(max(0.0, val))
    else:
        res = []
        for y in rc_over_xi:
            val, _ = quad(integrand, 0.0, 2.0, args=(y,))
            res.append(np.sqrt(max(0.0, val)))
        return np.array(res)


def main():
    vcd.setup_matplotlib()

    # 1. Load data
    inst_cache = CURRENT_DIR / "scratch" / "contrast_inst_cache.pkl"
    local_spd_cache = CURRENT_DIR / "scratch" / "particle_local_scalar_speed_summary.pkl"

    if not inst_cache.exists() or not local_spd_cache.exists():
        print(f"Error: Required caches not found ({inst_cache.name}, {local_spd_cache.name})")
        sys.exit(1)

    df_inst = pd.read_pickle(inst_cache)
    df_inst['unique_pid'] = df_inst['exp_dir'] + '_' + df_inst['particle'].astype(str)

    df_p_speeds = pd.read_pickle(local_spd_cache)
    # Convert local scalar speed from px/frame to um/s
    df_p_speeds['local_scalar_spd_um'] = df_p_speeds['local_scalar_spd_px'] * V_UNIT

    # Merge instantaneous records with particle-level local scalar speed
    df_merged = pd.merge(df_inst, df_p_speeds[['unique_pid', 'local_scalar_spd_um']], on='unique_pid', how='inner')

    # Instantaneous components
    flow_norm_px = np.hypot(df_merged['flow_u'], df_merged['flow_v'])
    valid = flow_norm_px > 0
    df_merged = df_merged[valid].copy()
    flow_norm_px = flow_norm_px[valid]

    u_hat_x = df_merged['flow_u'] / flow_norm_px
    u_hat_y = df_merged['flow_v'] / flow_norm_px

    # v_parallel in um/s: cargo velocity along local footprint direction
    df_merged['v_parallel'] = (df_merged['cargo_dx'] * u_hat_x + df_merged['cargo_dy'] * u_hat_y) * V_UNIT
    # u_fp in um/s: magnitude of footprint-averaged velocity vector
    df_merged['u_fp'] = flow_norm_px * V_UNIT
    # u_local_scalar in um/s: scalar speed of pixels within footprint
    df_merged['u_local_scalar'] = df_merged['local_scalar_spd_um']

    np.random.seed(42)
    N_BOOT = 2000

    results = []

    print("=== PARTICLE-LEVEL BOOTSTRAP IDENTITY DECOMPOSITION ===")
    print(f"{'Bead':10s} {'d (um)':8s} {'Rc/xi':8s} {'N_p':5s} | {'A_local':18s} | {'eta':18s} | {'Total':18s} | {'eta * A':10s}")
    print("-" * 105)

    for b_item in vcd.BEADS_INFO:
        bname = b_item['name']
        d_um = b_item['diameter_um']
        r_um = b_item['radius_um']
        sub = df_merged[df_merged['bead_name'] == bname]
        if len(sub) == 0:
            continue

        rc_xi = float(sub['rc_over_xi'].mean())
        xi_mean = float((r_um / sub['rc_over_xi']).mean())

        # Particle-level means
        p_means = sub.groupby('unique_pid')[['v_parallel', 'u_fp', 'u_local_scalar']].mean().reset_index()
        n_p = len(p_means)

        # Unweighted particle point estimates
        X_pt = float(p_means['v_parallel'].mean())
        Y_pt = float(p_means['u_fp'].mean())
        Z_pt = float(p_means['u_local_scalar'].mean())

        eta_pt = X_pt / Y_pt
        A_pt = Y_pt / Z_pt
        tot_pt = X_pt / Z_pt

        # Bootstrap over particles
        boot_eta = []
        boot_A = []
        boot_tot = []
        p_idx = np.arange(n_p)
        for _ in range(N_BOOT):
            idx = np.random.choice(p_idx, size=n_p, replace=True)
            samp = p_means.iloc[idx]
            xm = samp['v_parallel'].mean()
            ym = samp['u_fp'].mean()
            zm = samp['u_local_scalar'].mean()
            boot_eta.append(xm / ym)
            boot_A.append(ym / zm)
            boot_tot.append(xm / zm)

        eta_std = float(np.std(boot_eta))
        eta_ci = (float(np.percentile(boot_eta, 2.5)), float(np.percentile(boot_eta, 97.5)))
        A_std = float(np.std(boot_A))
        A_ci = (float(np.percentile(boot_A, 2.5)), float(np.percentile(boot_A, 97.5)))
        tot_std = float(np.std(boot_tot))
        tot_ci = (float(np.percentile(boot_tot, 2.5)), float(np.percentile(boot_tot, 97.5)))

        # Theoretical geometric cancellation Phi(Rc/xi)
        phi_theory = float(calc_disk_polar_order(rc_xi))

        row = {
            'bead_name': bname,
            'diameter_um': d_um,
            'radius_um': r_um,
            'rc_over_xi': rc_xi,
            'xi_um': xi_mean,
            'n_particles': n_p,
            'A_local': A_pt,
            'A_local_std': A_std,
            'A_local_ci': A_ci,
            'eta': eta_pt,
            'eta_std': eta_std,
            'eta_ci': eta_ci,
            'total': tot_pt,
            'total_std': tot_std,
            'total_ci': tot_ci,
            'phi_theory': phi_theory,
            'color': b_item['color'],
            'marker': b_item['marker']
        }
        results.append(row)

        print(f"{bname:10s} {d_um:6.2f} um {rc_xi:8.3f} {n_p:5d} | "
              f"{A_pt:6.3f} +/- {A_std:5.3f}   | "
              f"{eta_pt:6.3f} +/- {eta_std:5.3f}   | "
              f"{tot_pt:6.3f} +/- {tot_std:5.3f}   | "
              f"{eta_pt * A_pt:6.3f}")

    df_res = pd.DataFrame(results)
    df_res.to_pickle(OUT_SUMMARY_PKL)
    print(f"\nSaved summary to {OUT_SUMMARY_PKL}")

    # ==========================================
    # FIGURE 9: Single Panel Version
    # ==========================================
    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    # Theoretical curve Phi(Rc/xi)
    y_fine = np.logspace(np.log10(0.025), np.log10(1.8), 200)
    phi_fine = calc_disk_polar_order(y_fine)
    ax.plot(y_fine, phi_fine, color='#2CA02C', linestyle='--', linewidth=2.0,
            label=r'Theoretical spatial cancellation $\Phi(R_c/\xi)$', zorder=2)

    # Plot data points
    # Separate normal points from 20um (artifact)
    norm_res = df_res[df_res['diameter_um'] < 15.0]
    art_res = df_res[df_res['diameter_um'] >= 15.0]

    # 1. A_local (Geometric cancellation)
    ax.errorbar(norm_res['rc_over_xi'], norm_res['A_local'], yerr=norm_res['A_local_std'],
                fmt='s', color='#2CA02C', markersize=8, capsize=4, elinewidth=1.5,
                markeredgecolor='black', markeredgewidth=1.0,
                label=r'$A = \frac{\langle|\langle\mathbf{u}\rangle_{R_c}|\rangle}{\langle|\mathbf{u}|\rangle_{R_c}}$ (Footprint cancellation)', zorder=4)
    if len(art_res) > 0:
        ax.errorbar(art_res['rc_over_xi'], art_res['A_local'], yerr=art_res['A_local_std'],
                    fmt='s', color='white', markersize=8, capsize=4, elinewidth=1.5,
                    markeredgecolor='#2CA02C', markeredgewidth=2.0, zorder=4)

    # 2. eta (Dynamical transfer efficiency)
    ax.errorbar(norm_res['rc_over_xi'], norm_res['eta'], yerr=norm_res['eta_std'],
                fmt='o', color='#D62728', markersize=8, capsize=4, elinewidth=1.5,
                markeredgecolor='black', markeredgewidth=1.0,
                label=r'$\eta = \frac{\langle v_\parallel\rangle}{\langle|\langle\mathbf{u}\rangle_{R_c}|\rangle}$ (Dynamical transfer efficiency)', zorder=5)
    if len(art_res) > 0:
        ax.errorbar(art_res['rc_over_xi'], art_res['eta'], yerr=art_res['eta_std'],
                    fmt='o', color='white', markersize=8, capsize=4, elinewidth=1.5,
                    markeredgecolor='#D62728', markeredgewidth=2.0, zorder=5)

    # 3. Total transfer ratio
    ax.errorbar(norm_res['rc_over_xi'], norm_res['total'], yerr=norm_res['total_std'],
                fmt='^', color='#1F77B4', markersize=8, capsize=4, elinewidth=1.5,
                markeredgecolor='black', markeredgewidth=1.0,
                label=r'$\mathrm{Total} = \frac{\langle v_\parallel\rangle}{\langle|\mathbf{u}|\rangle_{R_c}} = \eta \times A$', zorder=3)
    if len(art_res) > 0:
        ax.errorbar(art_res['rc_over_xi'], art_res['total'], yerr=art_res['total_std'],
                    fmt='^', color='white', markersize=8, capsize=4, elinewidth=1.5,
                    markeredgecolor='#1F77B4', markeredgewidth=2.0, zorder=3)

    # Annotate 20um open marker
    if len(art_res) > 0:
        x_20 = art_res['rc_over_xi'].iloc[0]
        y_20_a = art_res['A_local'].iloc[0]
        ax.annotate(r'$20\,\mu\mathrm{m}$ (Shadow artifact)',
                    xy=(x_20, y_20_a), xytext=(x_20 * 0.45, y_20_a + 0.08),
                    arrowprops=dict(arrowstyle="->", color='#555555', lw=1.0),
                    fontsize=9, color='#333333', fontweight='semibold')

    # Annotate cargo sizes
    for _, r in norm_res.iterrows():
        ax.annotate(f"{r['diameter_um']:.2f} $\\mu$m",
                    xy=(r['rc_over_xi'], r['eta']),
                    xytext=(0, 10), textcoords='offset points',
                    ha='center', fontsize=8, color='#333333')

    ax.set_xscale('log')
    ax.set_xlabel(r'Normalized Cargo Size $R_c / \xi$', fontsize=12, labelpad=8)
    ax.set_ylabel(r'Velocity Decomposition Ratio', fontsize=12, labelpad=8)
    ax.set_ylim(-0.05, 1.15)
    ax.set_xlim(0.025, 2.0)
    ax.axhline(0, color='gray', linestyle=':', lw=0.8)
    ax.axhline(1, color='gray', linestyle=':', lw=0.8)

    ax.grid(True, which='both', linestyle='--', alpha=0.3)
    ax.legend(frameon=True, fontsize=9.5, loc='center left', bbox_to_anchor=(0.02, 0.45), framealpha=0.92)

    title_text = "Exact Velocity Decomposition Identity: Transfer Efficiency vs Footprint Cancellation\n" \
                 r"$\frac{\langle v_\parallel\rangle}{\langle|\mathbf{u}|\rangle_{R_c}} = \eta \times A$ (Local Footprint Baseline, Bootstrap SE)"
    ax.set_title(title_text, fontsize=11, fontweight='bold', pad=12)

    plt.tight_layout()
    fig.savefig(OUT_FIG_PNG, dpi=300)
    fig.savefig(OUT_FIG_SVG)
    plt.close(fig)
    print(f"Saved Figure 9 to {OUT_FIG_PNG}")

    # ==========================================
    # FIGURE 9B: Two Panel Version
    # ==========================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

    # Panel A: Components A and eta
    ax1.plot(y_fine, phi_fine, color='#2CA02C', linestyle='--', linewidth=1.8,
             label=r'Theory $\Phi(R_c/\xi)$', zorder=2)
    ax1.errorbar(norm_res['rc_over_xi'], norm_res['A_local'], yerr=norm_res['A_local_std'],
                 fmt='s', color='#2CA02C', markersize=7.5, capsize=4, elinewidth=1.5,
                 markeredgecolor='black', label=r'Footprint Cancellation $A = \frac{\langle|\langle\mathbf{u}\rangle_{R_c}|\rangle}{\langle|\mathbf{u}|\rangle_{R_c}}$', zorder=4)
    ax1.errorbar(norm_res['rc_over_xi'], norm_res['eta'], yerr=norm_res['eta_std'],
                 fmt='o', color='#D62728', markersize=7.5, capsize=4, elinewidth=1.5,
                 markeredgecolor='black', label=r'Transfer Efficiency $\eta = \frac{\langle v_\parallel\rangle}{\langle|\langle\mathbf{u}\rangle_{R_c}|\rangle}$', zorder=5)

    if len(art_res) > 0:
        ax1.errorbar(art_res['rc_over_xi'], art_res['A_local'], yerr=art_res['A_local_std'],
                     fmt='s', color='white', markersize=7.5, capsize=4, elinewidth=1.5,
                     markeredgecolor='#2CA02C', markeredgewidth=2.0, zorder=4)
        ax1.errorbar(art_res['rc_over_xi'], art_res['eta'], yerr=art_res['eta_std'],
                     fmt='o', color='white', markersize=7.5, capsize=4, elinewidth=1.5,
                     markeredgecolor='#D62728', markeredgewidth=2.0, zorder=5)

    ax1.set_xscale('log')
    ax1.set_xlabel(r'Cargo Size $R_c / \xi$', fontsize=11)
    ax1.set_ylabel(r'Ratio Component', fontsize=11)
    ax1.set_ylim(-0.05, 1.15)
    ax1.set_xlim(0.025, 2.0)
    ax1.grid(True, which='both', linestyle='--', alpha=0.3)
    ax1.legend(frameon=True, fontsize=9, loc='center left')
    ax1.set_title(r'(a) Decomposition Components $\eta$ and $A$', fontsize=11, fontweight='bold')

    # Panel B: Total transfer ratio
    ax2.errorbar(norm_res['rc_over_xi'], norm_res['total'], yerr=norm_res['total_std'],
                 fmt='^', color='#1F77B4', markersize=8, capsize=4, elinewidth=1.5,
                 markeredgecolor='black', label=r'Total Transfer Ratio $\frac{\langle v_\parallel\rangle}{\langle|\mathbf{u}|\rangle_{R_c}} = \eta \times A$', zorder=3)
    if len(art_res) > 0:
        ax2.errorbar(art_res['rc_over_xi'], art_res['total'], yerr=art_res['total_std'],
                     fmt='^', color='white', markersize=8, capsize=4, elinewidth=1.5,
                     markeredgecolor='#1F77B4', markeredgewidth=2.0, zorder=3)

    for _, r in norm_res.iterrows():
        ax2.annotate(f"{r['diameter_um']:.2f} $\\mu$m",
                     xy=(r['rc_over_xi'], r['total']),
                     xytext=(0, 10), textcoords='offset points',
                     ha='center', fontsize=8, color='#333333')

    ax2.set_xscale('log')
    ax2.set_xlabel(r'Cargo Size $R_c / \xi$', fontsize=11)
    ax2.set_ylabel(r'Total Transfer Ratio $\langle v_\parallel\rangle / \langle|\mathbf{u}|\rangle_{R_c}$', fontsize=11)
    ax2.set_ylim(-0.05, 0.65)
    ax2.grid(True, which='both', linestyle='--', alpha=0.3)
    ax2.legend(frameon=True, fontsize=9, loc='upper right')
    ax2.set_title(r'(b) Total Transfer Ratio $\eta \times A$', fontsize=11, fontweight='bold')

    plt.tight_layout()
    fig.savefig(OUT_FIG2_PNG, dpi=300)
    fig.savefig(OUT_FIG2_SVG)
    plt.close(fig)
    print(f"Saved Figure 9b to {OUT_FIG2_PNG}")


if __name__ == '__main__':
    main()

