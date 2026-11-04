#!/usr/bin/env python3
"""
scratch/test_advection.py
真の仮想貨物 (Lagrangian Advection Tracer) のテスト
実測粒子の初期位置から、各フレーム自身の位置における半径 Rc の足印平均流速で移流させる。
"""

import sys
import time
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import h5py
import verify_cos_decomposition as vcd

ROOT_DIR = vcd.ROOT_DIR
WORKSPACE_DIR = vcd.CURRENT_DIR
SCALE = 0.11
DT = 4.0

exp_dir = ROOT_DIR / "beads1um" / "20260717" / "beads1um001_crop"
flow_path = exp_dir / "GFP_flows.h5"
tracks_path = exp_dir / "beads_tracks.csv"

df_tracks = pd.read_csv(tracks_path)
active_set = vcd.load_active_particles(WORKSPACE_DIR)

r_um = 0.59  # 1.18 um bead
r_px = r_um / SCALE

print(f"Testing on {exp_dir.name}...")
print(f"Cargo radius: {r_um} um ({r_px:.2f} px)")

pids = [p for p in df_tracks['particle'].unique() if (exp_dir.name, p) in active_set]
print(f"Active particles: {pids}")

particle_meta = {}
for pid in pids:
    sub = df_tracks[df_tracks['particle'] == pid].sort_values('frame')
    particle_meta[pid] = {
        'start_frame': int(sub['frame'].min()),
        'end_frame': int(sub['frame'].max()),
        'start_x': float(sub['x'].iloc[0]),
        'start_y': float(sub['y'].iloc[0]),
        'real_sub': sub,
    }

t0 = time.time()
with h5py.File(flow_path, 'r') as f:
    flows_ds = f['flows']
    n_frames, _, H, W = flows_ds.shape
    print(f"Flow dataset shape: {flows_ds.shape}, H={H}, W={W}, n_frames={n_frames}")
    
    virt_tracks = {pid: [{'frame': particle_meta[pid]['start_frame'],
                          'x': particle_meta[pid]['start_x'],
                          'y': particle_meta[pid]['start_y'],
                          'u': 0.0, 'v': 0.0}] for pid in pids}
    
    cur_pos = {pid: [particle_meta[pid]['start_x'], particle_meta[pid]['start_y']] for pid in pids}
    
    rad_ceil = int(np.ceil(r_px))
    y_g, x_g = np.ogrid[-rad_ceil:rad_ceil+1, -rad_ceil:rad_ceil+1]
    disk_mask = (x_g**2 + y_g**2) <= r_px**2
    mask_h, mask_w = disk_mask.shape
    
    max_frame = max(m['end_frame'] for m in particle_meta.values())
    min_frame = min(m['start_frame'] for m in particle_meta.values())
    
    print(f"Simulating frames {min_frame} to {max_frame}...")
    
    for frm in range(min_frame, min(max_frame, n_frames)):
        flow_frame = flows_ds[frm] # float16
        flow_u = flow_frame[0].astype(np.float32)
        flow_v = flow_frame[1].astype(np.float32)
        
        for pid in pids:
            meta = particle_meta[pid]
            if frm < meta['start_frame'] or frm >= meta['end_frame']:
                continue
            
            cx, cy = cur_pos[pid]
            x_int = int(round(cx))
            y_int = int(round(cy))
            
            y_min = y_int - rad_ceil
            y_max = y_int + rad_ceil + 1
            x_min = x_int - rad_ceil
            x_max = x_int + rad_ceil + 1
            
            if y_min < 0 or y_max > H or x_min < 0 or x_max > W:
                y_idx, x_idx = np.ogrid[max(0, y_min):min(H, y_max), max(0, x_min):min(W, x_max)]
                sub_mask = ((x_idx - cx)**2 + (y_idx - cy)**2) <= r_px**2
                if np.any(sub_mask):
                    u_mean = float(np.mean(flow_u[y_idx, x_idx][sub_mask]))
                    v_mean = float(np.mean(flow_v[y_idx, x_idx][sub_mask]))
                else:
                    u_mean, v_mean = 0.0, 0.0
            else:
                patch_u = flow_u[y_min:y_max, x_min:x_max]
                patch_v = flow_v[y_min:y_max, x_min:x_max]
                u_mean = float(np.mean(patch_u[disk_mask]))
                v_mean = float(np.mean(patch_v[disk_mask]))
                
            new_x = cx + u_mean
            new_y = cy + v_mean
            cur_pos[pid] = [new_x, new_y]
            
            virt_tracks[pid].append({
                'frame': frm + 1,
                'x': new_x,
                'y': new_y,
                'u': u_mean,
                'v': v_mean,
            })

t1 = time.time()
print(f"Simulation completed in {t1 - t0:.2f} seconds!")

for pid in pids:
    df_v = pd.DataFrame(virt_tracks[pid])
    df_r = particle_meta[pid]['real_sub']
    
    disp_r = np.hypot(df_r['x'].iloc[-1] - df_r['x'].iloc[0], df_r['y'].iloc[-1] - df_r['y'].iloc[0]) * SCALE
    disp_v = np.hypot(df_v['x'].iloc[-1] - df_v['x'].iloc[0], df_v['y'].iloc[-1] - df_v['y'].iloc[0]) * SCALE
    
    print(f"Particle {pid}: Frames={len(df_r)} | Real Net Disp={disp_r:.2f} um | Virt Net Disp={disp_v:.2f} um")
