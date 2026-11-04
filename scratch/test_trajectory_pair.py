#!/usr/bin/env python3
"""
scratch/test_trajectory_pair.py
実測軌道と仮想軌道（足印流速積分）のペア構築とMSD比較テスト
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd
import verify_cos_decomposition as vcd

ROOT_DIR = vcd.ROOT_DIR
SCALE = 0.11 # um/px
DT = 4.0 # s

active_set = vcd.load_active_particles(vcd.CURRENT_DIR)
print(f"Active set size: {len(active_set)}")

# beads1um の 1 つの実験でテスト
exp_dirs = vcd.find_experiment_dirs(ROOT_DIR, 'beads1um')
exp_dir = exp_dirs[0]
print(f"Testing experiment: {exp_dir.name}")

df_flow = pd.read_csv(exp_dir / "beads_flow_interaction.csv")
df_tracks = pd.read_csv(exp_dir / "beads_tracks.csv")

# 1um の半径 (0.59 um -> ~5.36 px)
target_rad = 0.59 / SCALE
valid_rads = np.sort(df_flow[(df_flow['mean_flow_u'] != 0) | (df_flow['mean_flow_v'] != 0)]['radius'].unique())
best_rad = valid_rads[np.argmin(np.abs(valid_rads - target_rad))]
print(f"Target rad: {target_rad:.2f} px, best rad: {best_rad} px")

sub_flow = df_flow[df_flow['radius'] == best_rad].copy()

# 1つの粒子を選択
pids = [p for p in sub_flow['particle'].unique() if (exp_dir.name, p) in active_set]
print(f"Active particles in this exp: {pids}")

if pids:
    pid = pids[0]
    p_flow = sub_flow[sub_flow['particle'] == pid].sort_values('frame')
    p_track = df_tracks[df_tracks['particle'] == pid].sort_values('frame')
    
    # マージ
    merged = pd.merge(p_track, p_flow[['frame', 'mean_flow_u', 'mean_flow_v', 'cos_sim']], on='frame', how='inner')
    merged = merged.sort_values('frame')
    print(f"Merged points: {len(merged)}")
    
    # 連続フレームの抽出
    # frame が連続している区間
    merged['frame_diff'] = merged['frame'].diff()
    print("Frame diffs value counts:")
    print(merged['frame_diff'].value_counts())
