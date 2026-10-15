#!/usr/bin/env python3
"""
calc_missing_interactions.py

2026年10月8日のトラック再生成で追加された粒子について、
beads_flow_interaction_gaussian.csv に未反映だったフロー相互作用データを追加計算し、追記するスクリプト。
"""

import shutil
import time
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from libs.bead_flow_interaction_gaussian import calculate_flow_bead_dot_product_numba

BASE_NAS = Path('/mnt/NAS-Ebanaru/sasaki/MTsingleBeads')

TARGETS = [
    BASE_NAS / 'beads5um' / '20260715' / 'beads5um',
    BASE_NAS / 'beads5um' / '20260715' / 'beads5um001',
    BASE_NAS / 'beads5um' / '20260715' / 'beads5um002',
    BASE_NAS / 'beads5um' / '20260715' / 'beads5um003',
    BASE_NAS / 'beads7um' / '20260608' / 'beads7um',
    BASE_NAS / 'beads7um' / '20260608' / 'beads7um003',
]

def process_exp_dir(exp_dir: Path):
    tracks_p = exp_dir / 'beads_tracks.csv'
    flow_csv_p = exp_dir / 'beads_flow_interaction_gaussian.csv'
    h5_p = exp_dir / 'GFP_flows.h5'

    if not tracks_p.exists() or not flow_csv_p.exists() or not h5_p.exists():
        print(f"Skipping {exp_dir}: Missing required files.")
        return

    df_t = pd.read_csv(tracks_p)
    df_f = pd.read_csv(flow_csv_p)

    particles_t = set(df_t['particle'].unique())
    particles_f = set(df_f['particle'].unique())
    missing_particles = sorted(particles_t - particles_f)

    if not missing_particles:
        print(f"[{exp_dir.name}] All {len(particles_t)} particles already calculated. OK!")
        return

    sigmas = sorted(df_f['sigma'].unique())
    print(f"\n=======================================================")
    print(f"[{exp_dir.name}] Found {len(missing_particles)} missing particles: {missing_particles}")
    print(f"Target sigmas ({len(sigmas)}): {sigmas}")
    print(f"=======================================================")

    # バックアップ作成
    bak_p = exp_dir / 'beads_flow_interaction_gaussian.csv.bak'
    if not bak_p.exists():
        print(f"Creating backup: {bak_p}")
        shutil.copy2(flow_csv_p, bak_p)

    # missing 粒子のみ抽出
    sub_t = df_t[df_t['particle'].isin(missing_particles)].copy()
    grouped = sub_t.groupby('frame')
    frames = sorted(grouped.groups.keys())

    # JIT ウォームアップ
    dummy_arr = np.array([10.0], dtype=np.float32)
    dummy_flow = np.zeros((10, 10), dtype=np.float32)
    calculate_flow_bead_dot_product_numba(dummy_arr, dummy_arr, dummy_arr, dummy_arr, dummy_flow, dummy_flow, 5.0, 0.0)

    t0 = time.time()
    records_to_append = []

    with h5py.File(str(h5_p), 'r') as f:
        ds_key = list(f.keys())[0]
        flow_ds = f[ds_key]
        num_frames = flow_ds.shape[0]

        active_frames = [t for t in frames if t < num_frames and (t + 1) in grouped.groups]
        print(f"Frames to compute: {len(active_frames)} frames")

        for t in tqdm(active_frames, desc=f"Processing {exp_dir.name}"):
            st0 = sub_t.loc[grouped.groups[t]]
            st1 = sub_t.loc[grouped.groups[t + 1]]
            merged = pd.merge(st0, st1, on='particle', suffixes=('_0', '_t'))
            if merged.empty:
                continue

            bx = merged['x_0'].to_numpy(dtype=np.float32)
            by = merged['y_0'].to_numpy(dtype=np.float32)
            b_dx = (merged['x_t'] - merged['x_0']).to_numpy(dtype=np.float32)
            b_dy = (merged['y_t'] - merged['y_0']).to_numpy(dtype=np.float32)
            p_ids = merged['particle'].to_numpy()
            n_part = len(p_ids)

            # 1フレーム分の flow データを読み出し
            flow_frame = flow_ds[t] # shape: (2, H, W)
            m_x = np.ascontiguousarray(flow_frame[0], dtype=np.float32)
            m_y = np.ascontiguousarray(flow_frame[1], dtype=np.float32)

            for rad in sigmas:
                dot, cos_sim, mu, mv, cos_std = calculate_flow_bead_dot_product_numba(
                    bx, by, b_dx, b_dy, m_x, m_y, float(rad), 0.0
                )
                for i in range(n_part):
                    records_to_append.append({
                        'frame': t,
                        'particle': p_ids[i],
                        'sigma': rad,
                        'dot_product': dot[i],
                        'cos_sim': cos_sim[i],
                        'mean_flow_u': mu[i],
                        'mean_flow_v': mv[i],
                        'cos_std': cos_std[i]
                    })

    if records_to_append:
        df_new = pd.DataFrame(records_to_append)
        cols = ['frame', 'particle', 'sigma', 'dot_product', 'cos_sim', 'mean_flow_u', 'mean_flow_v', 'cos_std']
        df_new.to_csv(flow_csv_p, mode='a', header=False, index=False, columns=cols)
        print(f"Successfully appended {len(df_new)} rows for {len(missing_particles)} particles to {flow_csv_p.name} in {time.time() - t0:.1f} s")
    else:
        print(f"No records generated for {exp_dir.name}")

def main():
    print("Starting calculation of missing interactions...")
    for target in TARGETS:
        process_exp_dir(target)
    print("\nAll missing calculations completed successfully!")

if __name__ == '__main__':
    main()
