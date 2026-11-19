#!/usr/bin/env python3
"""
plot_propulsion_potential.py

有効推進ポテンシャル:
  U_prop(Theta_bulk; R_c) = v_tilde(R_c) * Phi(R_c) * U_eff_tilde(Theta_bulk)
を計算・作図する。
ここで:
  - v_tilde(R_c) = <v_cargo_norm> / <v_flow_norm> (局所平均流速に対する貨物の無次元実効速度)
  - Phi(R_c): 貨物直下の局所極性秩序度
  - U_eff_tilde(Theta_bulk): 周期核モデル B (sigma_pin = sigma_MT 固定) の無次元ポテンシャル
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from verify_cos_decomposition import (
    CURRENT_DIR, ROOT_DIR, BEADS_INFO,
    setup_matplotlib, save_figure, load_active_particles, load_all_data
)
from plot_fig5_linear_stability import log_Z_periodic

OUT_DIRS = [
    CURRENT_DIR / "figure" / "drive_decomposition",
    ROOT_DIR / "figure" / "drive_decomposition",
]
for d in OUT_DIRS:
    d.mkdir(parents=True, exist_ok=True)


def main():
    setup_matplotlib()
    df_stab = pd.read_csv(CURRENT_DIR / "fig5_linear_stability_summary.csv")

    active_set = load_active_particles(ROOT_DIR)
    df_instant, _ = load_all_data(ROOT_DIR, active_set)

    # 各ビーズサイズの無次元速度 v_tilde(R_c) = <v_cargo> / <v_flow> を算出
    v_rows = []
    for b_item in BEADS_INFO:
        bname = b_item['name']
        sub = df_instant[df_instant['bead_name'] == bname]
        vc = float(sub['v_cargo_norm'].mean())
        vf = float(sub['v_flow_norm'].mean())
        v_tilde = vc / vf if vf > 0 else np.nan
        v_rows.append({'bead_name': bname, 'v_cargo_mean': vc, 'v_flow_mean': vf, 'v_tilde': v_tilde})
    df_v = pd.DataFrame(v_rows)

    df_m = df_stab.merge(df_v, on='bead_name').sort_values('rc_over_xi')
    df_m['prefactor'] = df_m['v_tilde'] * df_m['phi_mean']
    df_m['delta_U_prop'] = df_m['prefactor'] * df_m['delta_U_tot']
    df_m['lambda_prop'] = df_m['prefactor'] * df_m['lambda0_abs']

    df_m.to_csv(CURRENT_DIR / "propulsion_potential_summary.csv", index=False)
    print("=== Effective Propulsion Potential U_prop Summary ===")
    print(df_m[['bead_name', 'rc_over_xi', 'v_tilde', 'phi_mean', 'prefactor',
                'delta_U_tot', 'delta_U_prop', 'lambda0_abs', 'lambda_prop']].to_string(index=False))

    # ------------------------------------------------------------------
    # 2-Panel Figure: (a) U_prop(Theta_bulk; R_c), (b) Scaling vs R_c/xi
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 5.8))
    th_fine = np.linspace(-np.pi, np.pi, 400)

    ax1.axhline(0.0, color='#888888', linestyle='--', linewidth=1.0)
    ax1.axvline(0.0, color='#888888', linestyle=':', linewidth=1.1)

    for b_item in BEADS_INFO:
        bname = b_item['name']
        r = df_m[df_m['bead_name'] == bname].iloc[0]
        k1, upin, sig = r['kappa1'], r['upin'], r['sigma_MT']
        lz = log_Z_periodic(k1, upin, sig)
        u0 = lz - np.log(2.0 * np.pi)
        u_eff = u0 - k1 * np.cos(th_fine) - upin * np.exp((np.cos(th_fine) - 1.0) / (sig ** 2))
        u_prop = r['prefactor'] * u_eff

        ax1.plot(th_fine, u_prop, color=b_item['color'], linewidth=2.3,
                 label=rf"{b_item['label']} ($\tilde{{v}}\Phi={r['prefactor']:.2f}, \Delta\tilde{{U}}_{{\mathrm{{prop}}}}={r['delta_U_prop']:.2f}$)")

    ax1.set_xlim(-np.pi, np.pi)
    ax1.set_xticks([-np.pi, -np.pi/2, 0.0, np.pi/2, np.pi])
    ax1.set_xticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
    ax1.set_xlabel(r'Bulk Alignment Angle $\Theta_{\mathrm{bulk}}$ (rad)', fontsize=13)
    ax1.set_ylabel(r'Effective Propulsion Potential $\tilde{U}_{\mathrm{prop}}(\Theta_{\mathrm{bulk}}; R_c)$', fontsize=12.5)
    ax1.set_title(r'(a) $\tilde{U}_{\mathrm{prop}}(\Theta_{\mathrm{bulk}}; R_c) = \tilde{v}(R_c)\,\Phi(R_c)\,\tilde{U}_{\mathrm{eff}}(\Theta_{\mathrm{bulk}})$',
                  fontsize=12.5, fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='lower center', fontsize=8.3, framealpha=0.92)

    # Panel (b): Prefactor, Delta U_prop, and lambda_prop vs R_c/xi
    ax2_r = ax2.twinx()
    ax2.plot(df_m['rc_over_xi'], df_m['delta_U_prop'], color='#1f77b4', marker='o', markersize=8.0,
             linewidth=2.2, label=r'Propulsion well depth $\Delta\tilde{U}_{\mathrm{prop}}$ (left)')
    ax2.plot(df_m['rc_over_xi'], df_m['prefactor'], color='#2ca02c', marker='s', markersize=7.0,
             linestyle='--', linewidth=1.8, label=r'Prefactor $\tilde{v}(R_c)\,\Phi(R_c)$ (left)')
    ax2_r.plot(df_m['rc_over_xi'], df_m['lambda_prop'], color='#d62728', marker='^', markersize=8.0,
               linestyle='-', linewidth=2.0, label=r'Propulsion stiffness $|\tilde{\lambda}_{\mathrm{prop}}|$ (right, log)')

    ax2.set_xscale('log')
    ax2_r.set_yscale('log')
    ax2.set_xlim(0.04, 3.2)
    ax2.set_ylim(0.0, 1.8)
    ax2_r.set_ylim(0.1, 50.0)
    ax2.set_xlabel(r'Dimensionless Radius $R_c / \xi$', fontsize=13)
    ax2.set_ylabel(r'Well Depth $\Delta\tilde{U}_{\mathrm{prop}}$ & Prefactor $\tilde{v}\Phi$', fontsize=12.5, color='#1f77b4')
    ax2_r.set_ylabel(r'Propulsion Stiffness $|\tilde{\lambda}_{\mathrm{prop}}| = \tilde{v}\Phi|\tilde{\lambda}_0|$', fontsize=12.5, color='#d62728')
    ax2.set_title(r'(b) Attenuation of Propulsion Potential vs $R_c/\xi$', fontsize=12.5, fontweight='bold')
    ax2.grid(True, which='both', linestyle=':', alpha=0.6)
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2_r.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc='upper right', fontsize=8.5, framealpha=0.92)

    fig.tight_layout()
    save_figure(fig, "fig_propulsion_potential", OUT_DIRS)
    plt.close(fig)
    print("\nSaved fig_propulsion_potential.png and .svg")


if __name__ == "__main__":
    main()

