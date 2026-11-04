#!/usr/bin/env python3
"""
plot_three_models_trajectories.py
代表粒子における3つの運動モードの2D軌道比較プロット:
- Real Cargo Track (実測軌道)
- Lagrangian Virtual Cargo (本来の仮想貨物: 自律移流)
- Eulerian Footprint Flow (累積通過流 / 滑り量モデル)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = Path("/mnt/NAS-Ebanaru/sasaki/MTsingleBeads")
if not ROOT_DIR.exists():
    ROOT_DIR = CURRENT_DIR

OUT_DIRS = [
    CURRENT_DIR / "figure" / "contrast_analysis",
    ROOT_DIR / "figure" / "contrast_analysis"
]

SCALE = 0.11
DT = 4.0

def main():
    # 抽出済みの軌道キャッシュを読み込み
    df_lag = pd.read_pickle("scratch/lagrangian_virt_tracks.pkl")
    
    # 代表的な2つの条件を選択:
    # 1. beads1um (1.18 um, particle 7): 活発に流れる小粒子
    # 2. beads20um (20.0 um, particle 1 or 2): ほぼ停止した大粒子
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))
    
    # --- Panel 1: beads1um Particle 7 ---
    exp_1um = "beads1um001_crop"
    p_1um = 7
    sub_lag1 = df_lag[(df_lag['exp_dir'] == exp_1um) & (df_lag['particle'] == p_1um)].sort_values('frame')
    
    # 実測トラック
    trk_1um = pd.read_csv(ROOT_DIR / "beads1um" / "20260717" / exp_1um / "beads_tracks.csv")
    sub_real1 = trk_1um[trk_1um['particle'] == p_1um].sort_values('frame')
    
    # 累積通過流
    flw_1um = pd.read_csv(ROOT_DIR / "beads1um" / "20260717" / exp_1um / "beads_flow_interaction.csv")
    sub_flw1 = flw_1um[(flw_1um['particle'] == p_1um) & (flw_1um['radius'] == 10)].sort_values('frame')
    
    m1 = pd.merge(sub_real1[['frame', 'x', 'y']], sub_flw1[['frame', 'mean_flow_u', 'mean_flow_v']], on='frame').sort_values('frame')
    x_r1 = (m1['x'].values - m1['x'].iloc[0]) * SCALE
    y_r1 = (m1['y'].values - m1['y'].iloc[0]) * SCALE
    
    u_p1 = m1['mean_flow_u'].values * SCALE
    v_p1 = m1['mean_flow_v'].values * SCALE
    x_p1 = np.zeros(len(m1))
    y_p1 = np.zeros(len(m1))
    x_p1[1:] = np.cumsum(u_p1[:-1])
    y_p1[1:] = np.cumsum(v_p1[:-1])
    
    x_v1 = (sub_lag1['x_virt_px'].values - sub_lag1['x_virt_px'].iloc[0]) * SCALE
    y_v1 = (sub_lag1['y_virt_px'].values - sub_lag1['y_virt_px'].iloc[0]) * SCALE
    
    ax1.plot(x_r1, y_r1, color='#CC6677', linewidth=2.6, label='Real Cargo (Measured Track)')
    ax1.plot(x_v1, y_v1, color='#5e3c99', linestyle='--', linewidth=2.6, label='Lagrangian Virtual Cargo (Ideal Upper Bound)')
    ax1.plot(x_p1, y_p1, color='#e66101', linestyle=':', linewidth=2.2, label='Eulerian Flow (Pass-through Flux / Slip Base)')
    ax1.scatter([0], [0], color='black', s=90, marker='x', label='Initial Position $(0,0)$', zorder=6)
    
    ax1.set_xlabel(r'$x - x_0$ [$\mu\mathrm{m}$]', fontsize=13)
    ax1.set_ylabel(r'$y - y_0$ [$\mu\mathrm{m}$]', fontsize=13)
    ax1.set_title(r'(a) Small Cargo ($d=1.18\,\mu\mathrm{m}$, $R_c/\xi=0.13$): Perfect Streamline Advection', fontsize=13, fontweight='bold')
    ax1.legend(loc='best', fontsize=9.5, framealpha=0.92)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.axis('equal')
    
    # --- Panel 2: beads20um Particle ---
    exp_20um = df_lag[df_lag['bead_name'] == 'beads20um']['exp_dir'].iloc[0]
    p_20um = df_lag[df_lag['bead_name'] == 'beads20um']['particle'].iloc[0]
    sub_lag20 = df_lag[(df_lag['exp_dir'] == exp_20um) & (df_lag['particle'] == p_20um)].sort_values('frame')
    
    # 探索
    exp_dir_20 = next(d for d in ROOT_DIR.glob("beads20um/*/*") if d.name == exp_20um or d.parent.name == exp_20um)
    trk_20um = pd.read_csv(exp_dir_20 / "beads_tracks.csv")
    sub_real20 = trk_20um[trk_20um['particle'] == p_20um].sort_values('frame')
    flw_20um = pd.read_csv(exp_dir_20 / "beads_flow_interaction.csv")
    rad_20 = np.sort(flw_20um['radius'].unique())[-1]
    sub_flw20 = flw_20um[(flw_20um['particle'] == p_20um) & (flw_20um['radius'] == rad_20)].sort_values('frame')
    
    m20 = pd.merge(sub_real20[['frame', 'x', 'y']], sub_flw20[['frame', 'mean_flow_u', 'mean_flow_v']], on='frame').sort_values('frame')
    x_r20 = (m20['x'].values - m20['x'].iloc[0]) * SCALE
    y_r20 = (m20['y'].values - m20['y'].iloc[0]) * SCALE
    
    u_p20 = m20['mean_flow_u'].values * SCALE
    v_p20 = m20['mean_flow_v'].values * SCALE
    x_p20 = np.zeros(len(m20))
    y_p20 = np.zeros(len(m20))
    x_p20[1:] = np.cumsum(u_p20[:-1])
    y_p20[1:] = np.cumsum(v_p20[:-1])
    
    x_v20 = (sub_lag20['x_virt_px'].values - sub_lag20['x_virt_px'].iloc[0]) * SCALE
    y_v20 = (sub_lag20['y_virt_px'].values - sub_lag20['y_virt_px'].iloc[0]) * SCALE
    
    ax2.plot(x_r20, y_r20, color='#44AA99', linewidth=2.8, label='Real Cargo (Measured Track / Stagnant)')
    ax2.plot(x_v20, y_v20, color='#5e3c99', linestyle='--', linewidth=2.6, label='Lagrangian Virtual Cargo (Ideal Upper Bound)')
    ax2.plot(x_p20, y_p20, color='#e66101', linestyle=':', linewidth=2.2, label='Eulerian Flow (Pass-through Flux / 99% Slip)')
    ax2.scatter([0], [0], color='black', s=90, marker='x', label='Initial Position $(0,0)$', zorder=6)
    
    ax2.set_xlabel(r'$x - x_0$ [$\mu\mathrm{m}$]', fontsize=13)
    ax2.set_ylabel(r'$y - y_0$ [$\mu\mathrm{m}$]', fontsize=13)
    ax2.set_title(r'(b) Giant Cargo ($d=20.0\,\mu\mathrm{m}$, $R_c/\xi=2.56$): Complete Slip & Stagnation', fontsize=13, fontweight='bold')
    ax2.legend(loc='best', fontsize=9.5, framealpha=0.92)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.axis('equal')
    
    plt.tight_layout()
    for out_dir in OUT_DIRS:
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / "contrast_fig6_three_models_trajectories.png", dpi=300, bbox_inches='tight')
        fig.savefig(out_dir / "contrast_fig6_three_models_trajectories.svg", format='svg', bbox_inches='tight')
    print("Saved contrast_fig6_three_models_trajectories.png and .svg")

if __name__ == '__main__':
    main()

