#!/usr/bin/env python3
"""
plot_velocity_decomposition.py

恒等式分解に基づく速度伝達効率・幾何平均化係数の解析スクリプト:
  <v_parallel> / <|u|> = [ <v_parallel> / <|<u_Rc>|> ] x [ <|<u_Rc>|> / <|u|> ]
                       = eta (伝達効率: 平均化以外) x A (幾何平均化による足元の流れの減少)

- v_parallel = v_c . u_hat_fp (貨物速度の足元の流れ方向成分: ノイズで偏らない量)
- <|u|>: ピクセル単位の流れの速さ (velocities_mean.csv より取得)
- フィッティング一切不要 (平均値の比のみ)
- 誤差は粒子単位のブートストラップ (Bootstrap N=2000)
- Rc/xi に対する A と eta (および eta*A) を 1 枚の高品位図に描画
- 20 um は別記号 (オープンマーカー) で明示
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from typing import Dict, List, Tuple
from scipy.integrate import quad

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = Path("/mnt/NAS-Ebanaru/sasaki/MTsingleBeads")
if not ROOT_DIR.exists():
    ROOT_DIR = CURRENT_DIR

sys.path.append(str(CURRENT_DIR))
import verify_cos_decomposition as vcd

OUT_DIRS = [
    CURRENT_DIR / "figure" / "contrast_analysis",
    ROOT_DIR / "figure" / "contrast_analysis"
]

SCALE = 0.11  # um/px
DT = 4.0      # s/frame
V_UNIT = SCALE / DT  # um/s per (px/frame)
N_BOOT = 2000
SEED = 42


def setup_matplotlib():
    plt.rcParams.update({
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.family': 'sans-serif',
        'mathtext.fontset': 'dejavusans',
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 15,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
        'figure.titlesize': 16,
        'axes.linewidth': 1.2,
        'lines.linewidth': 2.0,
        'grid.linewidth': 0.6,
        'grid.alpha': 0.4,
        'grid.linestyle': '--',
    })


def save_fig(fig, base_name):
    for out_dir in OUT_DIRS:
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f"{base_name}.png", dpi=300, bbox_inches='tight')
        fig.savefig(out_dir / f"{base_name}.svg", format='svg', bbox_inches='tight')
    print(f"Saved figure: {base_name}.png and .svg")


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


def load_and_preprocess_data(use_local_scalar=True):
    df_inst = pd.read_pickle(CURRENT_DIR / "scratch" / "contrast_inst_cache.pkl")
    df_inst['unique_pid'] = df_inst['exp_dir'] + '_' + df_inst['particle'].astype(str)

    local_spd_file = CURRENT_DIR / "scratch" / "particle_local_scalar_speed_summary.pkl"
    has_local = False
    if use_local_scalar and local_spd_file.exists():
        df_p_speeds = pd.read_pickle(local_spd_file)
        df_p_speeds['local_scalar_spd_um'] = df_p_speeds['local_scalar_spd_px'] * V_UNIT
        df_inst = pd.merge(df_inst, df_p_speeds[['unique_pid', 'local_scalar_spd_um']], on='unique_pid', how='inner')
        has_local = True
        print(f"Loaded particle-level local scalar speeds from {local_spd_file.name} (N={len(df_p_speeds)})")

    # Load velocities_mean.csv
    vm_list = []
    for b_item in vcd.BEADS_INFO:
        bname = b_item['name']
        exp_dirs = vcd.find_experiment_dirs(ROOT_DIR, bname)
        for ed in exp_dirs:
            vm_file = ed / "velocities_mean.csv"
            if vm_file.exists():
                df_vm = pd.read_csv(vm_file)
                df_vm['exp_dir'] = ed.name
                vm_list.append(df_vm[['exp_dir', 'frame', 'mean_velocity']])

    df_vm_all = pd.concat(vm_list, ignore_index=True)
    df_merged = pd.merge(df_inst, df_vm_all, on=['exp_dir', 'frame'], how='inner')

    flow_norm_px = np.hypot(df_merged['flow_u'], df_merged['flow_v'])
    valid = flow_norm_px > 0
    df_merged = df_merged[valid].copy()

    u_hat_x = df_merged['flow_u'] / flow_norm_px
    u_hat_y = df_merged['flow_v'] / flow_norm_px

    # Variables in physical units [um/s]
    df_merged['v_parallel'] = (df_merged['cargo_dx'] * u_hat_x + df_merged['cargo_dy'] * u_hat_y) * V_UNIT
    df_merged['u_fp'] = flow_norm_px * V_UNIT
    if has_local:
        df_merged['u_px'] = df_merged['local_scalar_spd_um']
    else:
        df_merged['u_px'] = df_merged['mean_velocity']

    return df_merged


def compute_decomposition_statistics(df_merged: pd.DataFrame):
    np.random.seed(SEED)
    results = []

    for b_item in vcd.BEADS_INFO:
        bname = b_item['name']
        d_um = b_item['diameter_um']
        r_um = b_item['radius_um']
        sub = df_merged[df_merged['bead_name'] == bname]
        rc_xi = float(sub['rc_over_xi'].mean())
        xi_val = float(sub['xi'].mean())

        p_ids = sub['unique_pid'].unique()
        n_p = len(p_ids)
        n_frames = len(sub)

        # -------------------------------------------------------------
        # Method 1: Particle-level time means -> Bootstrap over particles
        # (Equal weight to each particle)
        # -------------------------------------------------------------
        p_means = sub.groupby('unique_pid')[['v_parallel', 'u_fp', 'u_px']].mean().reset_index()
        m1_v_par = p_means['v_parallel'].mean()
        m1_u_fp = p_means['u_fp'].mean()
        m1_u_px = p_means['u_px'].mean()

        m1_A_pt = m1_u_fp / m1_u_px
        m1_eta_pt = m1_v_par / m1_u_fp
        m1_tot_pt = m1_v_par / m1_u_px

        boot_m1_A = []
        boot_m1_eta = []
        boot_m1_tot = []
        p_indices = np.arange(n_p)
        for _ in range(N_BOOT):
            idx = np.random.choice(p_indices, size=n_p, replace=True)
            samp = p_means.iloc[idx]
            xm = samp['v_parallel'].mean()
            ym = samp['u_fp'].mean()
            zm = samp['u_px'].mean()
            boot_m1_A.append(ym / zm)
            boot_m1_eta.append(xm / ym)
            boot_m1_tot.append(xm / zm)

        # -------------------------------------------------------------
        # Method 2: Cluster Bootstrap with frame pooling
        # (Cluster bootstrap: frames aggregated per sampled particle)
        # -------------------------------------------------------------
        m2_v_par = sub['v_parallel'].mean()
        m2_u_fp = sub['u_fp'].mean()
        m2_u_px = sub['u_pixel' if 'u_pixel' in sub.columns else 'u_px'].mean()

        m2_A_pt = m2_u_fp / m2_u_px
        m2_eta_pt = m2_v_par / m2_u_fp
        m2_tot_pt = m2_v_par / m2_u_px

        p_dict = {pid: grp[['v_parallel', 'u_fp', 'u_px']].values for pid, grp in sub.groupby('unique_pid')}
        boot_m2_A = []
        boot_m2_eta = []
        boot_m2_tot = []
        for _ in range(N_BOOT):
            samp_pids = np.random.choice(p_ids, size=n_p, replace=True)
            arrs = [p_dict[pid] for pid in samp_pids]
            cat = np.concatenate(arrs, axis=0)
            xm = np.mean(cat[:, 0])
            ym = np.mean(cat[:, 1])
            zm = np.mean(cat[:, 2])
            boot_m2_A.append(ym / zm)
            boot_m2_eta.append(xm / ym)
            boot_m2_tot.append(xm / zm)

        rec = {
            'bead_name': bname,
            'diameter_um': d_um,
            'radius_um': r_um,
            'xi_um': xi_val,
            'rc_over_xi': rc_xi,
            'n_particles': n_p,
            'n_frames': n_frames,
            # Method 1
            'm1_v_par': m1_v_par,
            'm1_u_fp': m1_u_fp,
            'm1_u_px': m1_u_px,
            'm1_A': m1_A_pt,
            'm1_A_sem': float(np.std(boot_m1_A)),
            'm1_A_ci95': (float(np.percentile(boot_m1_A, 2.5)), float(np.percentile(boot_m1_A, 97.5))),
            'm1_eta': m1_eta_pt,
            'm1_eta_sem': float(np.std(boot_m1_eta)),
            'm1_eta_ci95': (float(np.percentile(boot_m1_eta, 2.5)), float(np.percentile(boot_m1_eta, 97.5))),
            'm1_tot': m1_tot_pt,
            'm1_tot_sem': float(np.std(boot_m1_tot)),
            'm1_tot_ci95': (float(np.percentile(boot_m1_tot, 2.5)), float(np.percentile(boot_m1_tot, 97.5))),
            # Method 2
            'm2_v_par': m2_v_par,
            'm2_u_fp': m2_u_fp,
            'm2_u_px': m2_u_px,
            'm2_A': m2_A_pt,
            'm2_A_sem': float(np.std(boot_m2_A)),
            'm2_A_ci95': (float(np.percentile(boot_m2_A, 2.5)), float(np.percentile(boot_m2_A, 97.5))),
            'm2_eta': m2_eta_pt,
            'm2_eta_sem': float(np.std(boot_m2_eta)),
            'm2_eta_ci95': (float(np.percentile(boot_m2_eta, 2.5)), float(np.percentile(boot_m2_eta, 97.5))),
            'm2_tot': m2_tot_pt,
            'm2_tot_sem': float(np.std(boot_m2_tot)),
            'm2_tot_ci95': (float(np.percentile(boot_m2_tot, 2.5)), float(np.percentile(boot_m2_tot, 97.5))),
        }
        results.append(rec)

    df_res = pd.DataFrame(results).sort_values('rc_over_xi').reset_index(drop=True)
    return df_res


def plot_identity_decomposition(df_res: pd.DataFrame):
    setup_matplotlib()

    # =========================================================================
    # Figure 9: Master Single-Panel Decomposition Plot
    # Shows A (Geometric Averaging Factor), eta (Transfer Efficiency), and total
    # vs Rc / xi on log-linear and log-log scale.
    # =========================================================================
    fig, ax = plt.subplots(figsize=(8.0, 6.2))

    # Colors and markers
    c_eta = '#D95F02'     # Red-orange: Mechanical coupling efficiency
    c_A   = '#1B9E77'     # Green-teal: Footprint geometric averaging
    c_tot = '#7570B3'     # Purple: Overall ratio eta * A

    # Sort by rc_over_xi
    df_sorted = df_res.sort_values('rc_over_xi')

    # Points for standard beads (Rc/xi < 1.0) and 20um (Rc/xi > 1.0)
    std_mask = df_sorted['diameter_um'] < 15.0
    sp_mask  = df_sorted['diameter_um'] >= 15.0

    rc_std = df_sorted.loc[std_mask, 'rc_over_xi'].values
    rc_sp  = df_sorted.loc[sp_mask,  'rc_over_xi'].values

    # Theoretical curve Phi(Rc/xi)
    y_fine = np.logspace(np.log10(0.025), np.log10(1.8), 200)
    phi_fine = calc_disk_polar_order(y_fine)
    ax.plot(y_fine, phi_fine, color=c_A, linestyle='--', linewidth=2.0, alpha=0.8,
            label=r'Theoretical spatial cancellation $\Phi(R_c/\xi)$', zorder=2)

    # Plot A: Geometric Averaging Factor <|<u_Rc>|> / <|u|>
    # Standard points
    ax.errorbar(
        rc_std, df_sorted.loc[std_mask, 'm2_A'],
        yerr=[df_sorted.loc[std_mask, 'm2_A'] - df_sorted.loc[std_mask, 'm2_A_ci95'].apply(lambda x: x[0]),
              df_sorted.loc[std_mask, 'm2_A_ci95'].apply(lambda x: x[1]) - df_sorted.loc[std_mask, 'm2_A']],
        fmt='s-', color=c_A, ecolor=c_A, elinewidth=1.6, capsize=4.5, capthick=1.4,
        markersize=9, linewidth=2.2, label=r'$A = \frac{\langle|\langle\mathbf{u}\rangle_{R_c}|\rangle}{\langle|\mathbf{u}|\rangle_{R_c}}$ (Footprint Averaging / Polar Order)',
        zorder=4
    )
    # 20 um point (distinct open marker)
    ax.errorbar(
        rc_sp, df_sorted.loc[sp_mask, 'm2_A'],
        yerr=[df_sorted.loc[sp_mask, 'm2_A'] - df_sorted.loc[sp_mask, 'm2_A_ci95'].apply(lambda x: x[0]),
              df_sorted.loc[sp_mask, 'm2_A_ci95'].apply(lambda x: x[1]) - df_sorted.loc[sp_mask, 'm2_A']],
        fmt='s', mfc='white', mec=c_A, mew=2.5, color=c_A, ecolor=c_A, elinewidth=1.6, capsize=4.5, capthick=1.4,
        markersize=10, zorder=5
    )

    # Plot eta: Coupling Efficiency <v_parallel> / <|<u_Rc>|>
    ax.errorbar(
        rc_std, df_sorted.loc[std_mask, 'm2_eta'],
        yerr=[df_sorted.loc[std_mask, 'm2_eta'] - df_sorted.loc[std_mask, 'm2_eta_ci95'].apply(lambda x: x[0]),
              df_sorted.loc[std_mask, 'm2_eta_ci95'].apply(lambda x: x[1]) - df_sorted.loc[std_mask, 'm2_eta']],
        fmt='o-', color=c_eta, ecolor=c_eta, elinewidth=1.6, capsize=4.5, capthick=1.4,
        markersize=9, linewidth=2.2, label=r'$\eta = \frac{\langle v_\parallel\rangle}{\langle|\langle\mathbf{u}\rangle_{R_c}|\rangle}$ (Transfer Efficiency: Non-averaging)',
        zorder=6
    )
    ax.errorbar(
        rc_sp, df_sorted.loc[sp_mask, 'm2_eta'],
        yerr=[df_sorted.loc[sp_mask, 'm2_eta'] - df_sorted.loc[sp_mask, 'm2_eta_ci95'].apply(lambda x: x[0]),
              df_sorted.loc[sp_mask, 'm2_eta_ci95'].apply(lambda x: x[1]) - df_sorted.loc[sp_mask, 'm2_eta']],
        fmt='o', mfc='white', mec=c_eta, mew=2.5, color=c_eta, ecolor=c_eta, elinewidth=1.6, capsize=4.5, capthick=1.4,
        markersize=10, zorder=7
    )

    # Plot Overall Ratio: <v_parallel> / <|u|> = eta * A
    ax.errorbar(
        rc_std, df_sorted.loc[std_mask, 'm2_tot'],
        yerr=[df_sorted.loc[std_mask, 'm2_tot'] - df_sorted.loc[std_mask, 'm2_tot_ci95'].apply(lambda x: x[0]),
              df_sorted.loc[std_mask, 'm2_tot_ci95'].apply(lambda x: x[1]) - df_sorted.loc[std_mask, 'm2_tot']],
        fmt='d--', color=c_tot, ecolor=c_tot, elinewidth=1.4, capsize=4.0, capthick=1.2,
        markersize=8, linewidth=1.8, label=r'$\frac{\langle v_\parallel\rangle}{\langle|\mathbf{u}|\rangle} = \eta \times A$ (Overall Transport Ratio)',
        alpha=0.85, zorder=3
    )
    ax.errorbar(
        rc_sp, df_sorted.loc[sp_mask, 'm2_tot'],
        yerr=[df_sorted.loc[sp_mask, 'm2_tot'] - df_sorted.loc[sp_mask, 'm2_tot_ci95'].apply(lambda x: x[0]),
              df_sorted.loc[sp_mask, 'm2_tot_ci95'].apply(lambda x: x[1]) - df_sorted.loc[sp_mask, 'm2_tot']],
        fmt='d', mfc='white', mec=c_tot, mew=2.5, color=c_tot, ecolor=c_tot, elinewidth=1.4, capsize=4.0, capthick=1.2,
        markersize=9, zorder=4
    )

    # Reference lines
    ax.axhline(1.0, color='gray', linestyle=':', linewidth=1.2, alpha=0.7, label='Ideal Perfect Coupling ($y = 1$)')
    ax.axhline(0.0, color='black', linestyle='-', linewidth=1.0, alpha=0.5)

    # Annotate 20 um bead
    ax.annotate(
        r'20 $\mu\mathrm{m}$' + f'\n($R_c/\\xi=2.56$, $N=3$)\n' + r'$\eta = 0.02 \approx 0$',
        xy=(rc_sp[0], df_sorted.loc[sp_mask, 'm2_eta'].values[0]),
        xytext=(rc_sp[0] * 0.45, 0.22),
        arrowprops=dict(arrowstyle='->', lw=1.3, color='#333333', connectionstyle='arc3,rad=-0.15'),
        fontsize=10.5, fontweight='semibold',
        bbox=dict(boxstyle='round,pad=0.35', fc='#FFF8EE', ec='#D95F02', alpha=0.9, lw=1.2)
    )

    # Annotate A vs eta contrast
    ax.text(
        0.05, 0.92,
        'Exact Identity (No Fitting Required):\n'
        r'$\frac{\langle v_\parallel\rangle}{\langle|\mathbf{u}|\rangle_{R_c}} = \eta \times A$' + '\n\n'
        r'$\bullet\ A \to 1$ at small $R_c/\xi$, follows theory $\Phi$' + '\n'
        r'$\bullet\ \eta \to 0$ as $R_c/\xi > 1$ (Decoupling)',
        transform=ax.transAxes,
        fontsize=10.5, verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.5', fc='#F4F6F9', ec='#334155', alpha=0.95, lw=1.2)
    )

    # Labels and scales
    ax.set_xscale('log')
    ax.set_xlim(0.04, 4.0)
    ax.set_ylim(-0.10, 1.45)
    ax.set_xlabel(r'Normalized Cargo Size $R_c / \xi$', fontsize=14, labelpad=8)
    ax.set_ylabel('Velocity Ratio [-]', fontsize=14, labelpad=8)
    ax.set_title(r'Exact Velocity Decomposition: $\frac{\langle v_\parallel\rangle}{\langle|\mathbf{u}|\rangle_{R_c}} = \eta \times A$',
                 fontsize=15, pad=12, fontweight='bold')

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    # Add dummy handle for 20um open marker explanation
    from matplotlib.lines import Line2D
    dummy_open = Line2D([0], [0], marker='o', color='w', markeredgecolor='black', markeredgewidth=2.0,
                        markersize=9, label=r'Open: $20\,\mu\mathrm{m}$ Cargo ($R_c > \xi$)')
    handles.append(dummy_open)
    labels.append(r'Open: $20\,\mu\mathrm{m}$ Cargo ($R_c > \xi$)')

    ax.legend(handles=handles, labels=labels, loc='upper right', frameon=True,
              framealpha=0.95, facecolor='white', edgecolor='#CCCCCC', fontsize=10.0)

    ax.grid(True, which='both', linestyle='--', alpha=0.4)
    plt.tight_layout()

    save_fig(fig, "contrast_fig9_velocity_decomposition_identity")
    plt.close(fig)

    # =========================================================================
    # Figure 9b: 2-Panel Comprehensive Edition (Linear vs Log-Log & Method Comparison)
    # =========================================================================
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 5.8))

    # Panel 1: Linear scale (Identical to main Fig 9)
    ax1.plot(y_fine, phi_fine, color=c_A, linestyle='--', linewidth=1.8, alpha=0.8,
             label=r'Theory $\Phi(R_c/\xi)$', zorder=2)
    ax1.errorbar(
        rc_std, df_sorted.loc[std_mask, 'm2_A'],
        yerr=[df_sorted.loc[std_mask, 'm2_A'] - df_sorted.loc[std_mask, 'm2_A_ci95'].apply(lambda x: x[0]),
              df_sorted.loc[std_mask, 'm2_A_ci95'].apply(lambda x: x[1]) - df_sorted.loc[std_mask, 'm2_A']],
        fmt='s-', color=c_A, ecolor=c_A, elinewidth=1.6, capsize=4, capthick=1.2,
        markersize=8, linewidth=2.0, label=r'$A = \langle|\langle\mathbf{u}\rangle_{R_c}|\rangle / \langle|\mathbf{u}|\rangle_{R_c}$',
        zorder=4
    )
    ax1.errorbar(
        rc_sp, df_sorted.loc[sp_mask, 'm2_A'],
        yerr=[df_sorted.loc[sp_mask, 'm2_A'] - df_sorted.loc[sp_mask, 'm2_A_ci95'].apply(lambda x: x[0]),
              df_sorted.loc[sp_mask, 'm2_A_ci95'].apply(lambda x: x[1]) - df_sorted.loc[sp_mask, 'm2_A']],
        fmt='s', mfc='white', mec=c_A, mew=2.2, color=c_A, ecolor=c_A, elinewidth=1.6, capsize=4, capthick=1.2,
        markersize=9, zorder=5
    )

    ax1.errorbar(
        rc_std, df_sorted.loc[std_mask, 'm2_eta'],
        yerr=[df_sorted.loc[std_mask, 'm2_eta'] - df_sorted.loc[std_mask, 'm2_eta_ci95'].apply(lambda x: x[0]),
              df_sorted.loc[std_mask, 'm2_eta_ci95'].apply(lambda x: x[1]) - df_sorted.loc[std_mask, 'm2_eta']],
        fmt='o-', color=c_eta, ecolor=c_eta, elinewidth=1.6, capsize=4, capthick=1.2,
        markersize=8, linewidth=2.0, label=r'$\eta = \langle v_\parallel\rangle / \langle|\langle\mathbf{u}\rangle_{R_c}|\rangle$ (Efficiency)',
        zorder=6
    )
    ax1.errorbar(
        rc_sp, df_sorted.loc[sp_mask, 'm2_eta'],
        yerr=[df_sorted.loc[sp_mask, 'm2_eta'] - df_sorted.loc[sp_mask, 'm2_eta_ci95'].apply(lambda x: x[0]),
              df_sorted.loc[sp_mask, 'm2_eta_ci95'].apply(lambda x: x[1]) - df_sorted.loc[sp_mask, 'm2_eta']],
        fmt='o', mfc='white', mec=c_eta, mew=2.2, color=c_eta, ecolor=c_eta, elinewidth=1.6, capsize=4, capthick=1.2,
        markersize=9, zorder=7
    )

    ax1.errorbar(
        rc_std, df_sorted.loc[std_mask, 'm2_tot'],
        yerr=[df_sorted.loc[std_mask, 'm2_tot'] - df_sorted.loc[std_mask, 'm2_tot_ci95'].apply(lambda x: x[0]),
              df_sorted.loc[std_mask, 'm2_tot_ci95'].apply(lambda x: x[1]) - df_sorted.loc[std_mask, 'm2_tot']],
        fmt='d--', color=c_tot, ecolor=c_tot, elinewidth=1.4, capsize=3.5, capthick=1.0,
        markersize=7, linewidth=1.6, label=r'$\eta \times A$ (Total Transport Ratio)',
        alpha=0.85, zorder=3
    )
    ax1.errorbar(
        rc_sp, df_sorted.loc[sp_mask, 'm2_tot'],
        yerr=[df_sorted.loc[sp_mask, 'm2_tot'] - df_sorted.loc[sp_mask, 'm2_tot_ci95'].apply(lambda x: x[0]),
              df_sorted.loc[sp_mask, 'm2_tot_ci95'].apply(lambda x: x[1]) - df_sorted.loc[sp_mask, 'm2_tot']],
        fmt='d', mfc='white', mec=c_tot, mew=2.2, color=c_tot, ecolor=c_tot, elinewidth=1.4, capsize=3.5, capthick=1.0,
        markersize=8, zorder=4
    )

    ax1.axhline(1.0, color='gray', linestyle=':', linewidth=1.2, alpha=0.7)
    ax1.axhline(0.0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    ax1.set_xscale('log')
    ax1.set_xlim(0.04, 4.0)
    ax1.set_ylim(-0.10, 1.45)
    ax1.set_xlabel(r'Normalized Cargo Size $R_c / \xi$', fontsize=13)
    ax1.set_ylabel('Velocity Ratio [-]', fontsize=13)
    ax1.set_title('(a) Exact Identity Decomposition (Linear Scale)', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', frameon=True, framealpha=0.9, fontsize=9.5)
    ax1.grid(True, which='both', linestyle='--', alpha=0.4)

    # Panel 2: Log-Log Scale (Highlighting Power-Law Decay of eta vs A)
    # Only positive values plotted on log-log
    pos_mask = df_sorted['m2_eta'] > 0
    pos_std = std_mask & pos_mask
    pos_sp  = sp_mask & pos_mask

    ax2.errorbar(
        df_sorted.loc[pos_std, 'rc_over_xi'], df_sorted.loc[pos_std, 'm2_A'],
        yerr=[df_sorted.loc[pos_std, 'm2_A'] - df_sorted.loc[pos_std, 'm2_A_ci95'].apply(lambda x: x[0]),
              df_sorted.loc[pos_std, 'm2_A_ci95'].apply(lambda x: x[1]) - df_sorted.loc[pos_std, 'm2_A']],
        fmt='s-', color=c_A, ecolor=c_A, elinewidth=1.6, capsize=4, capthick=1.2,
        markersize=8, linewidth=2.0, label=r'Averaging Factor $A$', zorder=4
    )
    if pos_sp.any():
        ax2.errorbar(
            df_sorted.loc[pos_sp, 'rc_over_xi'], df_sorted.loc[pos_sp, 'm2_A'],
            yerr=[df_sorted.loc[pos_sp, 'm2_A'] - df_sorted.loc[pos_sp, 'm2_A_ci95'].apply(lambda x: x[0]),
                  df_sorted.loc[pos_sp, 'm2_A_ci95'].apply(lambda x: x[1]) - df_sorted.loc[pos_sp, 'm2_A']],
            fmt='s', mfc='white', mec=c_A, mew=2.2, color=c_A, ecolor=c_A, elinewidth=1.6, capsize=4, capthick=1.2,
            markersize=9, zorder=5
        )

    ax2.errorbar(
        df_sorted.loc[pos_std, 'rc_over_xi'], df_sorted.loc[pos_std, 'm2_eta'],
        yerr=[df_sorted.loc[pos_std, 'm2_eta'] - df_sorted.loc[pos_std, 'm2_eta_ci95'].apply(lambda x: x[0]),
              df_sorted.loc[pos_std, 'm2_eta_ci95'].apply(lambda x: x[1]) - df_sorted.loc[pos_std, 'm2_eta']],
        fmt='o-', color=c_eta, ecolor=c_eta, elinewidth=1.6, capsize=4, capthick=1.2,
        markersize=8, linewidth=2.0, label=r'Efficiency $\eta$', zorder=6
    )
    if pos_sp.any():
        ax2.errorbar(
            df_sorted.loc[pos_sp, 'rc_over_xi'], df_sorted.loc[pos_sp, 'm2_eta'],
            yerr=[df_sorted.loc[pos_sp, 'm2_eta'] - df_sorted.loc[pos_sp, 'm2_eta_ci95'].apply(lambda x: x[0]),
                  df_sorted.loc[pos_sp, 'm2_eta_ci95'].apply(lambda x: x[1]) - df_sorted.loc[pos_sp, 'm2_eta']],
            fmt='o', mfc='white', mec=c_eta, mew=2.2, color=c_eta, ecolor=c_eta, elinewidth=1.6, capsize=4, capthick=1.2,
            markersize=9, zorder=7
        )

    ax2.errorbar(
        df_sorted.loc[pos_std, 'rc_over_xi'], df_sorted.loc[pos_std, 'm2_tot'],
        yerr=[df_sorted.loc[pos_std, 'm2_tot'] - df_sorted.loc[pos_std, 'm2_tot_ci95'].apply(lambda x: x[0]),
              df_sorted.loc[pos_std, 'm2_tot_ci95'].apply(lambda x: x[1]) - df_sorted.loc[pos_std, 'm2_tot']],
        fmt='d--', color=c_tot, ecolor=c_tot, elinewidth=1.4, capsize=3.5, capthick=1.0,
        markersize=7, linewidth=1.6, label=r'Total Ratio $\eta \times A$', alpha=0.85, zorder=3
    )
    if pos_sp.any():
        ax2.errorbar(
            df_sorted.loc[pos_sp, 'rc_over_xi'], df_sorted.loc[pos_sp, 'm2_tot'],
            yerr=[df_sorted.loc[pos_sp, 'm2_tot'] - df_sorted.loc[pos_sp, 'm2_tot_ci95'].apply(lambda x: x[0]),
                  df_sorted.loc[pos_sp, 'm2_tot_ci95'].apply(lambda x: x[1]) - df_sorted.loc[pos_sp, 'm2_tot']],
            fmt='d', mfc='white', mec=c_tot, mew=2.2, color=c_tot, ecolor=c_tot, elinewidth=1.4, capsize=3.5, capthick=1.0,
            markersize=8, zorder=4
        )

    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlim(0.04, 4.0)
    ax2.set_ylim(0.015, 1.8)
    ax2.set_xlabel(r'Normalized Cargo Size $R_c / \xi$', fontsize=13)
    ax2.set_ylabel('Velocity Ratio [-] (Log Scale)', fontsize=13)
    ax2.set_title('(b) Scaling Comparison (Log-Log Scale)', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower left', frameon=True, framealpha=0.9, fontsize=9.5)
    ax2.grid(True, which='both', linestyle='--', alpha=0.4)

    plt.tight_layout()
    save_fig(fig2, "contrast_fig9b_velocity_decomposition_2panels")
    plt.close(fig2)


COMPARE_BEADS_DEFAULT = ('beads06um', 'beads20um')


def _particle_boot_ci(p_means: pd.DataFrame, col: str, rng: np.random.Generator):
    vals = p_means[col].values
    n = len(vals)
    boots = vals[rng.integers(0, n, size=(N_BOOT, n))].mean(axis=1)
    return float(vals.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def plot_size_comparison(df_merged: pd.DataFrame, bead_names=COMPARE_BEADS_DEFAULT):
    """2 サイズ (既定: beads06um vs beads20um) の速度量を比較する 3 パネル図.

    (a) 粒子平均速度 <v_par>, <|<u_Rc>|>, <|u|> の棒グラフ (粒子ブートストラップ 95% CI)
    (b) 瞬時 v_parallel の分布 (密度ヒストグラム)
    (c) 粒子平均 v_parallel vs |<u_Rc>| 散布図 (y=x 参照線)
    """
    setup_matplotlib()
    info = {b['name']: b for b in vcd.BEADS_INFO}
    rng = np.random.default_rng(SEED)
    qty = [('v_parallel', r'$\langle v_\parallel\rangle$'),
           ('u_fp', r'$\langle|\langle\mathbf{u}\rangle_{R_c}|\rangle$'),
           ('u_px', r'$\langle|\mathbf{u}|\rangle$')]

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8))
    width = 0.8 / len(bead_names)
    rows = []
    all_vpar = np.concatenate([df_merged.loc[df_merged['bead_name'] == b, 'v_parallel'].values
                               for b in bead_names])
    lo, hi = np.percentile(all_vpar, [0.5, 99.5])
    bins = np.linspace(lo, hi, 60)
    ax_max = 0.0

    for k, bname in enumerate(bead_names):
        sub = df_merged[df_merged['bead_name'] == bname]
        if sub.empty:
            print(f"[warn] no data for {bname}")
            continue
        b = info[bname]
        p_means = sub.groupby('unique_pid')[['v_parallel', 'u_fp', 'u_px']].mean()
        label = f"{b['label']} (N={len(p_means)} particles)"

        # (a) bar chart of means
        stats = [_particle_boot_ci(p_means, c, rng) for c, _ in qty]
        x = np.arange(len(qty)) + (k - (len(bead_names) - 1) / 2) * width
        m = np.array([s[0] for s in stats])
        err = np.array([[s[0] - s[1] for s in stats], [s[2] - s[0] for s in stats]])
        axes[0].bar(x, m, width=width, color=b['color'], yerr=err, capsize=4,
                    edgecolor='black', linewidth=0.8, label=label)
        rows.append({'bead_name': bname, 'n_particles': len(p_means), 'n_frames': len(sub),
                     **{f'{c}_mean': s[0] for (c, _), s in zip(qty, stats)},
                     **{f'{c}_ci95': (s[1], s[2]) for (c, _), s in zip(qty, stats)},
                     'eta': stats[0][0] / stats[1][0], 'A': stats[1][0] / stats[2][0]})

        # (b) distribution of instantaneous v_parallel
        axes[1].hist(sub['v_parallel'], bins=bins, density=True, histtype='stepfilled',
                     alpha=0.35, color=b['color'])
        axes[1].hist(sub['v_parallel'], bins=bins, density=True, histtype='step',
                     linewidth=1.8, color=b['color'], label=b['label'])
        axes[1].axvline(sub['v_parallel'].mean(), color=b['color'], linestyle='--', linewidth=1.5)

        # (c) particle-level scatter
        axes[2].scatter(p_means['u_fp'], p_means['v_parallel'], s=28, alpha=0.75,
                        color=b['color'], marker=b['marker'], edgecolor='black', linewidth=0.4,
                        label=b['label'])
        ax_max = max(ax_max, p_means['u_fp'].max(), p_means['v_parallel'].max())

    axes[0].set_xticks(np.arange(len(qty)))
    axes[0].set_xticklabels([q[1] for q in qty])
    axes[0].set_ylabel(r'Speed [$\mu$m/s]')
    axes[0].set_title('(a) Mean speeds (particle bootstrap 95% CI)')
    axes[0].legend(fontsize=9)

    axes[1].axvline(0, color='gray', linewidth=0.8)
    axes[1].set_xlabel(r'$v_\parallel$ [$\mu$m/s] (instantaneous)')
    axes[1].set_ylabel('Probability density')
    axes[1].set_title(r'(b) Distribution of $v_\parallel$ (dashed: mean)')
    axes[1].legend(fontsize=9)

    lim = ax_max * 1.05 if ax_max > 0 else 1.0
    axes[2].plot([0, lim], [0, lim], color='gray', linestyle=':', label=r'$y=x$ ($\eta=1$)')
    axes[2].axhline(0, color='gray', linewidth=0.8)
    axes[2].set_xlim(0, lim)
    axes[2].set_xlabel(r'$\langle|\langle\mathbf{u}\rangle_{R_c}|\rangle_t$ per particle [$\mu$m/s]')
    axes[2].set_ylabel(r'$\langle v_\parallel\rangle_t$ per particle [$\mu$m/s]')
    axes[2].set_title('(c) Particle-level cargo vs footprint flow')
    axes[2].legend(fontsize=9)

    for ax in axes:
        ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    tag = '_vs_'.join(bead_names)
    save_fig(fig, f"contrast_fig9c_speed_comparison_{tag}")
    plt.close(fig)

    df_cmp = pd.DataFrame(rows)
    for d in OUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        df_cmp.to_csv(d / f"speed_comparison_{tag}.csv", index=False)
    print(f"\n=== Speed comparison: {' vs '.join(bead_names)} [um/s] ===")
    print(df_cmp[['bead_name', 'n_particles', 'n_frames', 'v_parallel_mean', 'u_fp_mean',
                  'u_px_mean', 'eta', 'A']].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return df_cmp


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--compare', nargs='+', default=list(COMPARE_BEADS_DEFAULT),
                    help='比較するビーズ名 (既定: beads06um beads20um)')
    ap.add_argument('--compare-only', action='store_true',
                    help='サイズ比較図のみ出力 (全サイズの分解統計を省略)')
    args = ap.parse_args()

    print("Loading data...")
    df_merged = load_and_preprocess_data()

    if args.compare_only:
        plot_size_comparison(df_merged, tuple(args.compare))
        print("All tasks completed successfully!")
        return
    print(f"Loaded {len(df_merged)} valid instantaneous measurements across 22 experiments.")

    print("Computing decomposition statistics & particle bootstrap (N=2000)...")
    df_res = compute_decomposition_statistics(df_merged)

    # Save summary tables
    for d in OUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        df_res.to_csv(d / "velocity_decomposition_identity_summary.csv", index=False)
        df_res.to_pickle(d / "velocity_decomposition_identity_summary.pkl")
    (CURRENT_DIR / "scratch" / "velocity_decomposition_identity_summary.pkl").write_bytes(
        (OUT_DIRS[0] / "velocity_decomposition_identity_summary.pkl").read_bytes()
    )

    print("\n=== Velocity Decomposition Summary (Method 2: Cluster Bootstrap) ===")
    cols_show = ['diameter_um', 'rc_over_xi', 'n_particles', 'm2_A', 'm2_eta', 'm2_tot']
    print(df_res[cols_show].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nGenerating figures...")
    plot_identity_decomposition(df_res)
    plot_size_comparison(df_merged, tuple(args.compare))
    print("All tasks completed successfully!")


if __name__ == '__main__':
    main()

