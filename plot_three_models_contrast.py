#!/usr/bin/env python3
"""
plot_three_models_contrast.py
3つの運動モードの完全比較図を作成する:
1. Real Cargo (実測貨物)
2. Lagrangian Virtual Cargo (本来の仮想貨物: 自律移流トレーサー) [真の上限]
3. Eulerian Footprint Flow (累積通過流 / 滑り量指標) [素通りフラックス]
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = Path("/mnt/NAS-Ebanaru/sasaki/MTsingleBeads")
if not ROOT_DIR.exists():
    ROOT_DIR = CURRENT_DIR

OUT_DIRS = [
    CURRENT_DIR / "figure" / "contrast_analysis",
    ROOT_DIR / "figure" / "contrast_analysis"
]

BEADS_INFO = [
    {'name': 'beads06um', 'label': r'0.63 $\mu\mathrm{m}$', 'diameter_um': 0.63, 'radius_um': 0.315,
     'color': '#882255', 'marker': '^'},
    {'name': 'beads1um',  'label': r'1.18 $\mu\mathrm{m}$', 'diameter_um': 1.18, 'radius_um': 0.590,
     'color': '#CC6677', 'marker': 'o'},
    {'name': 'beads3um',  'label': r'3.37 $\mu\mathrm{m}$', 'diameter_um': 3.37, 'radius_um': 1.685,
     'color': '#DDCC77', 'marker': 'd'},
    {'name': 'beads7um',  'label': r'7.24 $\mu\mathrm{m}$', 'diameter_um': 7.24, 'radius_um': 3.620,
     'color': '#117733', 'marker': 'h'},
    {'name': 'beads5um',  'label': r'5.00 $\mu\mathrm{m}$', 'diameter_um': 5.00, 'radius_um': 2.500,
     'color': '#999933', 'marker': 'p'},
    {'name': 'beads20um', 'label': r'20.0 $\mu\mathrm{m}$', 'diameter_um': 20.0, 'radius_um': 10.000,
     'color': '#44AA99', 'marker': 's'},
]

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
        'legend.fontsize': 10.5,
        'figure.titlesize': 16,
        'axes.linewidth': 1.2,
        'lines.linewidth': 2.2,
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

def main():
    setup_matplotlib()
    merged_imsd = pd.read_pickle("scratch/three_models_imsd_cache.pkl")
    
    # -------------------------------------------------------------
    # Figure 2 Revised: MSD 3者直接比較 (6パネル)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), sharex=True, sharey=True)
    axes = axes.flatten()

    for idx, b_item in enumerate(BEADS_INFO):
        ax = axes[idx]
        bname = b_item['name']
        sub = merged_imsd[merged_imsd['bead_name'] == bname].copy()
        
        rc_xi = sub['rc_over_xi'].mean()
        color = b_item['color']
        
        grp = sub.groupby('lag_time_s').agg({
            'msd_real': ['mean', stats.sem],
            'msd_lagrangian_virt': ['mean', stats.sem],
            'msd_eulerian_pass': ['mean', stats.sem],
        })
        
        lags = grp.index.values
        mr = grp['msd_real']['mean'].values
        sr = grp['msd_real']['sem'].values
        ml = grp['msd_lagrangian_virt']['mean'].values
        sl = grp['msd_lagrangian_virt']['sem'].values
        mp = grp['msd_eulerian_pass']['mean'].values
        sp = grp['msd_eulerian_pass']['sem'].values
        
        # 1. 累積通過流 (Eulerian Pass: 素通りフラックス / 滑り量基準)
        ax.plot(lags, mp, color='#e66101', linestyle=':', linewidth=2.2, label='Eulerian Flow (Pass-through / Slip Base)')
        ax.fill_between(lags, np.maximum(1e-4, mp - sp), mp + sp, color='#e66101', alpha=0.10)

        # 2. 本来の仮想貨物 (Lagrangian Virt: 自律移流トレーサー / 理想追従上限)
        ax.plot(lags, ml, color='#5e3c99', linestyle='--', linewidth=2.5, label='Lagrangian Virtual Cargo (Ideal Upper Bound)')
        ax.fill_between(lags, np.maximum(1e-4, ml - sl), ml + sl, color='#5e3c99', alpha=0.15)
        
        # 3. 実測貨物 (Real Cargo)
        ax.plot(lags, mr, color=color, linestyle='-', linewidth=2.5, marker=b_item['marker'],
                markersize=6, markevery=5, label=f'Real Cargo ({b_item["label"]})')
        ax.fill_between(lags, np.maximum(1e-4, mr - sr), mr + sr, color=color, alpha=0.25)
        
        # 傾き参照
        t_ref = np.array([4.0, 40.0])
        ref_scale = mr[0] / t_ref[0]
        ax.plot(t_ref, ref_scale * t_ref, color='gray', linestyle=':', linewidth=1.0)
        ax.plot(t_ref, (ref_scale / 10.0) * (t_ref**2), color='dimgray', linestyle='-.', linewidth=1.0)
        
        # インセット: 追従度 (mr / ml) と 滑り率 (1 - mr / mp)
        ax_ins = ax.inset_axes([0.52, 0.10, 0.44, 0.35])
        ratio_virt = mr / np.where(ml > 0, ml, np.nan)
        slip_pct = (1.0 - mr / np.where(mp > 0, mp, np.nan)) * 100.0
        
        ax_ins.plot(lags, ratio_virt, color='#5e3c99', linewidth=1.8, label=r'$\mathrm{Real}/\mathrm{Virt}$')
        ax_ins.plot(lags, slip_pct / 100.0, color='#e66101', linestyle=':', linewidth=1.8, label=r'Slip ratio')
        ax_ins.axhline(1.0, color='gray', linestyle='--', linewidth=0.8)
        ax_ins.set_xscale('log')
        ax_ins.set_xlim(4, 300)
        ax_ins.set_ylim(-0.1, 1.4)
        ax_ins.set_title('Ratio & Slip', fontsize=9.0, pad=2)
        ax_ins.tick_params(labelsize=8)
        ax_ins.grid(True, linestyle=':', alpha=0.5)
        if idx == 0:
            ax_ins.legend(loc='lower left', fontsize=7.5)
        
        n_p = sub['particle'].nunique()
        ax.set_title(f"{b_item['label']}  ($R_c/\\xi = {rc_xi:.2f}, N={n_p}$)", fontsize=13, fontweight='bold')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(3.5, 350)
        ax.set_ylim(0.01, 1000)
        ax.grid(True)
        
        if idx % 3 == 0:
            ax.set_ylabel(r'MSD $\langle \Delta r^2(\tau) \rangle$ [$\mu\mathrm{m}^2$]')
        if idx >= 3:
            ax.set_xlabel(r'Lag time $\tau$ [s]')
        if idx == 0:
            ax.legend(loc='upper left', fontsize=9.0, framealpha=0.92)

    fig.suptitle(r'Ensemble MSD: Real Cargo vs Lagrangian Virtual Cargo vs Eulerian Footprint Flow',
                 fontsize=16, fontweight='bold', y=0.995)
    save_fig(fig, "contrast_fig2_three_models_msd_6panels")
    plt.close(fig)

    # -------------------------------------------------------------
    # Figure 5: 滑り率 (Slip Ratio) vs Rc/xi のサマリー
    # -------------------------------------------------------------
    fig_slip, (ax_s1, ax_s2) = plt.subplots(1, 2, figsize=(15, 6))

    rc_list, slip_64_list, ratio_64_list = [], [], []
    for b_item in BEADS_INFO:
        bname = b_item['name']
        sub = merged_imsd[merged_imsd['bead_name'] == bname]
        rc_xi = sub['rc_over_xi'].mean()
        sub64 = sub[sub['lag_time_s'] == 64.0]
        if len(sub64) > 0:
            mr = sub64['msd_real'].mean()
            ml = sub64['msd_lagrangian_virt'].mean()
            mp = sub64['msd_eulerian_pass'].mean()
            slip = 1.0 - mr / mp
            rat = mr / ml
        else:
            slip, rat = np.nan, np.nan
        rc_list.append(rc_xi)
        slip_64_list.append(slip)
        ratio_64_list.append(rat)

    sort_idx = np.argsort(rc_list)
    rc_arr = np.array(rc_list)[sort_idx]
    slip_arr = np.array(slip_64_list)[sort_idx]
    rat_arr = np.array(ratio_64_list)[sort_idx]

    # Panel 1: 滑り率 (Slip Ratio = 1 - Real / Pass)
    for i in sort_idx:
        b = BEADS_INFO[i]
        ax_s1.scatter(rc_list[i], slip_64_list[i] * 100, color=b['color'], marker=b['marker'], s=120, zorder=5, label=b['label'])
    ax_s1.plot(rc_arr, slip_arr * 100, color='#e66101', linewidth=2.0, linestyle='-', zorder=4)
    ax_s1.axhline(0, color='gray', linestyle='--', linewidth=1.0)
    ax_s1.axhline(100, color='#cc2222', linestyle=':', linewidth=1.2, label='100% Complete Slip')
    ax_s1.axvline(1.0, color='gray', linestyle='--', linewidth=1.2, alpha=0.7)
    ax_s1.set_xscale('log')
    ax_s1.set_xlim(0.04, 4.0)
    ax_s1.set_ylim(-20, 110)
    ax_s1.set_xlabel(r'Dimensionless cargo size $R_c / \xi$')
    ax_s1.set_ylabel(r'Slip Ratio $(1 - \mathrm{MSD}_{\mathrm{real}}/\mathrm{MSD}_{\mathrm{pass}}) \times 100$ [%]')
    ax_s1.set_title(r'(a) Hydrodynamic Slip Ratio vs $R_c/\xi$ ($\tau = 64\ \mathrm{s}$)', fontweight='bold')
    ax_s1.legend(loc='lower right', fontsize=10)
    ax_s1.grid(True)

    # Panel 2: 追従度 (Tracking Ratio = Real / Lagrangian Virt)
    for i in sort_idx:
        b = BEADS_INFO[i]
        ax_s2.scatter(rc_list[i], ratio_64_list[i], color=b['color'], marker=b['marker'], s=120, zorder=5, label=b['label'])
    ax_s2.plot(rc_arr, rat_arr, color='#5e3c99', linewidth=2.0, linestyle='-', zorder=4)
    ax_s2.axhline(1.0, color='#2b83ba', linestyle='--', linewidth=1.5, label='Perfect Advection (1.0)')
    ax_s2.axvline(1.0, color='gray', linestyle='--', linewidth=1.2, alpha=0.7)
    ax_s2.set_xscale('log')
    ax_s2.set_xlim(0.04, 4.0)
    ax_s2.set_ylim(-0.05, 1.2)
    ax_s2.set_xlabel(r'Dimensionless cargo size $R_c / \xi$')
    ax_s2.set_ylabel(r'Tracking Ratio $\mathrm{MSD}_{\mathrm{real}} / \mathrm{MSD}_{\mathrm{virt}}$')
    ax_s2.set_title(r'(b) Lagrangian Tracking Ratio vs $R_c/\xi$ ($\tau = 64\ \mathrm{s}$)', fontweight='bold')
    ax_s2.legend(loc='lower left', fontsize=10)
    ax_s2.grid(True)

    save_fig(fig_slip, "contrast_fig5_slip_and_tracking_ratio_summary")
    plt.close(fig_slip)
    print("Done creating plots!")

if __name__ == '__main__':
    main()
