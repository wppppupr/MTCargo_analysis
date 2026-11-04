#!/usr/bin/env python3
"""
scratch/compare_three_models.py
3つの運動モードの直接比較:
1. Real Cargo (実測貨物)
2. Eulerian Footprint Flow (累積通過流 / 滑り量)
3. Lagrangian Virtual Cargo (本来の仮想貨物: 自律移流)
"""

import sys
import time
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import verify_cos_decomposition as vcd

ROOT_DIR = vcd.ROOT_DIR
WORKSPACE_DIR = vcd.CURRENT_DIR
SCALE = 0.11  # um/px
DT = 4.0      # s/frame

# beads1um001_crop で 3 者比較
exp_dir = ROOT_DIR / "beads1um" / "20260717" / "beads1um001_crop"
flow_path = exp_dir / "GFP_flows.h5"
tracks_path = exp_dir / "beads_tracks.csv"
interaction_path = exp_dir / "beads_flow_interaction.csv"

df_tracks = pd.read_csv(tracks_path)
df_flow = pd.read_csv(interaction_path)
active_set = vcd.load_active_particles(WORKSPACE_DIR)

r_um = 0.59
r_px = r_um / SCALE

# 最寄り有効半径
valid_rads = np.sort(df_flow[(df_flow['mean_flow_u'] != 0) | (df_flow['mean_flow_v'] != 0)]['radius'].unique())
best_rad = valid_rads[np.argmin(np.abs(valid_rads - r_px))]
sub_flow = df_flow[df_flow['radius'] == best_rad].copy()

# 代表粒子として Particle 5 (中程度) と Particle 7 (高速) を選択
test_pids = [5, 7]

results = {}

for pid in test_pids:
    p_track = df_tracks[df_tracks['particle'] == pid].sort_values('frame')
    p_f = sub_flow[sub_flow['particle'] == pid].sort_values('frame')
    merged = pd.merge(p_track[['frame', 'x', 'y']], 
                      p_f[['frame', 'mean_flow_u', 'mean_flow_v']], 
                      on='frame', how='inner').sort_values('frame')
    
    # 連続フレーム区間
    merged['fdiff'] = merged['frame'].diff()
    max_seg_id = (merged['fdiff'] != 1).cumsum().value_counts().idxmax()
    seg = merged[(merged['fdiff'] != 1).cumsum() == max_seg_id].copy()
    
    # 1. Real Cargo
    x_real = seg['x'].values * SCALE
    y_real = seg['y'].values * SCALE
    x_real -= x_real[0]
    y_real -= y_real[0]
    
    # 2. Eulerian Footprint Flow (累積通過流 / 滑り量)
    u_pass = seg['mean_flow_u'].values * SCALE
    v_pass = seg['mean_flow_v'].values * SCALE
    x_pass = np.zeros(len(seg))
    y_pass = np.zeros(len(seg))
    x_pass[1:] = np.cumsum(u_pass[:-1])
    y_pass[1:] = np.cumsum(v_pass[:-1])
    
    # 3. Lagrangian Virtual Cargo (本来の仮想貨物: 自律移流)
    # HDF5 から積分
    start_frame = int(seg['frame'].iloc[0])
    end_frame = int(seg['frame'].iloc[-1])
    start_x_px = float(seg['x'].iloc[0])
    start_y_px = float(seg['y'].iloc[0])
    
    x_virt_px = [start_x_px]
    y_virt_px = [start_y_px]
    cur_x, cur_y = start_x_px, start_y_px
    
    rad_ceil = int(np.ceil(r_px))
    y_g, x_g = np.ogrid[-rad_ceil:rad_ceil+1, -rad_ceil:rad_ceil+1]
    disk_mask = (x_g**2 + y_g**2) <= r_px**2
    
    with h5py.File(flow_path, 'r') as f:
        flows_ds = f['flows']
        H, W = flows_ds.shape[2], flows_ds.shape[3]
        for frm in range(start_frame, end_frame):
            flow_frame = flows_ds[frm]
            fu = flow_frame[0].astype(np.float32)
            fv = flow_frame[1].astype(np.float32)
            
            x_int = int(round(cur_x))
            y_int = int(round(cur_y))
            y_min = y_int - rad_ceil
            y_max = y_int + rad_ceil + 1
            x_min = x_int - rad_ceil
            x_max = x_int + rad_ceil + 1
            
            if y_min < 0 or y_max > H or x_min < 0 or x_max > W:
                u_m, v_m = 0.0, 0.0
            else:
                u_m = float(np.mean(fu[y_min:y_max, x_min:x_max][disk_mask]))
                v_m = float(np.mean(fv[y_min:y_max, x_min:x_max][disk_mask]))
                
            cur_x += u_m
            cur_y += v_m
            x_virt_px.append(cur_x)
            y_virt_px.append(cur_y)
            
    x_virt = (np.array(x_virt_px) - start_x_px) * SCALE
    y_virt = (np.array(y_virt_px) - start_y_px) * SCALE
    
    results[pid] = {
        'real': (x_real, y_real),
        'pass': (x_pass, y_pass),
        'virt': (x_virt, y_virt),
        'time': np.arange(len(seg)) * DT,
    }

print("Calculated 3 models for test particles successfully!")

# プロット作成
fig, axes = plt.subplots(2, 2, figsize=(14, 11))

for idx, pid in enumerate(test_pids):
    res = results[pid]
    xr, yr = res['real']
    xp, yp = res['pass']
    xv, yv = res['virt']
    t_arr = res['time']
    
    # 2D 軌跡
    ax_traj = axes[idx, 0]
    ax_traj.plot(xr, yr, color='#CC6677', linewidth=2.5, label='Real Cargo (実測貨物)')
    ax_traj.plot(xp, yp, color='#DDCC77', linestyle=':', linewidth=2.5, label='Eulerian Flow (累積通過流 / 滑り量)')
    ax_traj.plot(xv, yv, color='#332288', linestyle='--', linewidth=2.5, label='Lagrangian Virtual (本来の仮想貨物)')
    ax_traj.scatter([0], [0], color='black', s=80, marker='x', label='Start (0,0)', zorder=6)
    ax_traj.set_xlabel(r'$x - x_0$ [$\mu\mathrm{m}$]')
    ax_traj.set_ylabel(r'$y - y_0$ [$\mu\mathrm{m}$]')
    ax_traj.set_title(f'Trajectory Comparison (Particle {pid}, $d=1.18\,\mu\mathrm{{m}}$)', fontweight='bold')
    ax_traj.legend(loc='best', fontsize=10)
    ax_traj.grid(True)
    ax_traj.axis('equal')
    
    # MSD 計算 & プロット
    ax_msd = axes[idx, 1]
    def calc_msd(x, y):
        n = len(x)
        msds = []
        taus = np.arange(1, min(n, 50))
        for tau in taus:
            dx = x[tau:] - x[:-tau]
            dy = y[tau:] - y[:-tau]
            msds.append(np.mean(dx**2 + dy**2))
        return taus * DT, np.array(msds)
        
    l_r, m_r = calc_msd(xr, yr)
    l_p, m_p = calc_msd(xp, yp)
    l_v, m_v = calc_msd(xv, yv)
    
    ax_msd.plot(l_r, m_r, color='#CC6677', linewidth=2.5, label='Real Cargo')
    ax_msd.plot(l_p, m_p, color='#DDCC77', linestyle=':', linewidth=2.5, label='Eulerian Flow (累積通過流)')
    ax_msd.plot(l_v, m_v, color='#332288', linestyle='--', linewidth=2.5, label='Lagrangian Virtual (本来の仮想貨物)')
    
    ax_msd.set_xscale('log')
    ax_msd.set_yscale('log')
    ax_msd.set_xlabel(r'Lag time $\tau$ [s]')
    ax_msd.set_ylabel(r'MSD $\langle \Delta r^2 \rangle$ [$\mu\mathrm{m}^2$]')
    ax_msd.set_title(f'Individual MSD Comparison (Particle {pid})', fontweight='bold')
    ax_msd.legend(loc='upper left', fontsize=10)
    ax_msd.grid(True)

plt.tight_layout()
fig_p = Path("figure/contrast_analysis/three_models_comparison_trajectory_msd.png")
fig_p.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(fig_p, dpi=300, bbox_inches='tight')
print(f"Saved figure to {fig_p}")

