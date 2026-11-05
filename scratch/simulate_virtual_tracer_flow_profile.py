#!/usr/bin/env python3
"""
scratch/simulate_virtual_tracer_flow_profile.py

Lagrange 仮想トレーサーによる動径流速プロファイル検証スクリプト:
「受動粒子が流速 |u| の低い領域に集まるか」を検証する。

1. 点トレーサーを使う: 貨物のサイズを持たせず、初期フレームで貨物のない領域に一様に散布。
2. 測定は実際の貨物と揃える: 各サイズについて半径 R_c の仮想マスクを付け、まったく同じ計算で |u|(r/ξ)/u_∞ を求める。
3. 時間を揃える: まいてから、実験で貨物を追跡した時間経過後の分布（サンプリングフレーム）を使う。
"""

import sys
import time
import pickle
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from scipy.ndimage import map_coordinates
from multiprocessing import Pool, cpu_count

# パラメータ設定
SCALE_UM_PER_PX = 0.11
DT_SEC = 4.0
V_UNIT = SCALE_UM_PER_PX / DT_SEC # um/s

MAX_R_EDGE_UM = 35.0
BIN_WIDTH_UM = 0.50
MAX_SAMPLES = 25 # 実測と同じフレーム数
N_TRACERS = 2000 # 仮想点トレーサー数
N_PROFILE_SUBSET = 200 # 各サンプルフレームでプロファイル計測に使うトレーサー数（統計十分・高速化）

# ビーズ情報
BEADS_INFO = {
    "beads06um": {"diameter_um": 0.63, "radius_um": 0.315, "rc_px": 3.0},
    "beads1um":  {"diameter_um": 1.18, "radius_um": 0.590, "rc_px": 6.0},
    "beads3um":  {"diameter_um": 3.37, "radius_um": 1.685, "rc_px": 15.0},
    "beads5um":  {"diameter_um": 5.00, "radius_um": 2.500, "rc_px": 25.0},
    "beads7um":  {"diameter_um": 7.24, "radius_um": 3.620, "rc_px": 35.0},
    "beads20um": {"diameter_um": 20.0, "radius_um": 10.00, "rc_px": 95.0},
}

TARGET_DIRS = [
    "/mnt/NAS-Ebanaru/sasaki/MTsingleBeads/beads06um/20260716/beads06um",
    "/mnt/NAS-Ebanaru/sasaki/MTsingleBeads/beads06um/20260716/beads06um001",
    "/mnt/NAS-Ebanaru/sasaki/MTsingleBeads/beads06um/20260716/beads06um002",
    "/mnt/NAS-Ebanaru/sasaki/MTsingleBeads/beads06um/20260716/beads06um003",
    "/mnt/NAS-Ebanaru/sasaki/MTsingleBeads/beads1um/20260717/beads1um001_crop",
    "/mnt/NAS-Ebanaru/sasaki/MTsingleBeads/beads1um/20260717/beads1um002",
    "/mnt/NAS-Ebanaru/sasaki/MTsingleBeads/beads1um/20260717/beads1um003",
    "/mnt/NAS-Ebanaru/sasaki/MTsingleBeads/beads1um/20260717/beads1um_crop",
    "/mnt/NAS-Ebanaru/sasaki/MTsingleBeads/beads3um/20260708/beads3um",
    "/mnt/NAS-Ebanaru/sasaki/MTsingleBeads/beads3um/20260708/beads3um001",
    "/mnt/NAS-Ebanaru/sasaki/MTsingleBeads/beads3um/20260708/beads3um002",
    "/mnt/NAS-Ebanaru/sasaki/MTsingleBeads/beads3um/20260708/beads3um003",
    "/mnt/NAS-Ebanaru/sasaki/MTsingleBeads/beads5um/20260715/beads5um",
    "/mnt/NAS-Ebanaru/sasaki/MTsingleBeads/beads5um/20260715/beads5um001",
    "/mnt/NAS-Ebanaru/sasaki/MTsingleBeads/beads5um/20260715/beads5um002",
    "/mnt/NAS-Ebanaru/sasaki/MTsingleBeads/beads5um/20260715/beads5um003",
    "/mnt/NAS-Ebanaru/sasaki/MTsingleBeads/beads7um/20260608/beads7um",
    "/mnt/NAS-Ebanaru/sasaki/MTsingleBeads/beads7um/20260608/beads7um002",
    "/mnt/NAS-Ebanaru/sasaki/MTsingleBeads/beads7um/20260608/beads7um003",
    "/mnt/NAS-Ebanaru/sasaki/MTsingleBeads/beads20um/20260723/beads20um001",
    "/mnt/NAS-Ebanaru/sasaki/MTsingleBeads/beads20um/20260723/beads20um002",
    "/mnt/NAS-Ebanaru/sasaki/MTsingleBeads/beads20um/20260723/beads20um_crop"
]

def get_bead_info_from_dir(exp_dir_str):
    for b_key, info in BEADS_INFO.items():
        if f"/{b_key}/" in exp_dir_str:
            return b_key, info
    raise ValueError(f"Unknown bead type for {exp_dir_str}")

def process_single_experiment(exp_dir_str):
    t0 = time.time()
    exp_dir = Path(exp_dir_str)
    b_name, b_info = get_bead_info_from_dir(exp_dir_str)
    h5_path = exp_dir / "GFP_flows.h5"
    tracks_path = exp_dir / "beads_tracks.csv"
    
    if not h5_path.exists() or not tracks_path.exists():
        print(f"Skipping {exp_dir_str}: files not found")
        return None
        
    df_tracks = pd.read_csv(tracks_path)
    Rc_px = b_info['rc_px']
    max_r_edge_px = MAX_R_EDGE_UM / SCALE_UM_PER_PX
    bin_width_px = BIN_WIDTH_UM / SCALE_UM_PER_PX
    bins_px = np.arange(0, max_r_edge_px + bin_width_px, bin_width_px)
    n_bins = len(bins_px) - 1
    
    bin_sums = np.zeros(n_bins, dtype=np.float64)
    bin_counts = np.zeros(n_bins, dtype=np.int64)
    
    with h5py.File(str(h5_path), 'r') as hf:
        dataset_key = list(hf.keys())[0]
        flows = hf[dataset_key]
        num_frames = flows.shape[0]
        H, W = flows.shape[2], flows.shape[3]
        
        # 実測と同じサンプリングフレームを取得
        available_frames = [f for f in sorted(df_tracks['frame'].unique()) if f < num_frames]
        if not available_frames:
            return None
        n_samples = min(len(available_frames), MAX_SAMPLES)
        sample_indices = np.linspace(0, len(available_frames) - 1, n_samples, dtype=int)
        sampled_frames = set(available_frames[i] for i in sample_indices)
        max_frame_needed = max(sampled_frames)
        
        # 点トレーサーの初期化 (t=0)
        # 境界マージンを確保
        margin = max_r_edge_px + Rc_px + 20.0
        np.random.seed(42 + hash(exp_dir_str) % 10000)
        
        # 貨物の初期位置
        init_tracks = df_tracks[df_tracks['frame'] == available_frames[0]]
        cargo_pts = init_tracks[['x', 'y']].values if len(init_tracks) > 0 else np.empty((0, 2))
        
        # 初期均一散布（貨物の初期位置近傍は除外）
        tracers_x = []
        tracers_y = []
        while len(tracers_x) < N_TRACERS:
            cand_x = np.random.uniform(margin, W - margin, N_TRACERS * 2)
            cand_y = np.random.uniform(margin, H - margin, N_TRACERS * 2)
            for cx, cy in zip(cand_x, cand_y):
                if len(cargo_pts) > 0:
                    dists = np.hypot(cargo_pts[:, 0] - cx, cargo_pts[:, 1] - cy)
                    if np.min(dists) < Rc_px * 2.0:
                        continue
                tracers_x.append(cx)
                tracers_y.append(cy)
                if len(tracers_x) == N_TRACERS:
                    break
                    
        tr_x = np.array(tracers_x, dtype=np.float64)
        tr_y = np.array(tracers_y, dtype=np.float64)
        
        # フレームごとに時間発展 (Advection)
        for fr in range(max_frame_needed + 1):
            flow_fr = flows[fr]
            u = flow_fr[0].astype(np.float32)
            v = flow_fr[1].astype(np.float32)
            
            # もし現在のフレームがサンプリングフレームなら、動径プロファイルを計測
            if fr in sampled_frames:
                speed = np.hypot(u, v)
                
                # 有効領域（境界から十分離れている）のトレーサーを抽出
                valid_tr = (tr_x >= margin) & (tr_x < W - margin) & (tr_y >= margin) & (tr_y < H - margin)
                idx_valid = np.where(valid_tr)[0]
                
                if len(idx_valid) > N_PROFILE_SUBSET:
                    idx_subset = np.random.choice(idx_valid, N_PROFILE_SUBSET, replace=False)
                else:
                    idx_subset = idx_valid
                    
                for tid in idx_subset:
                    bx, by = tr_x[tid], tr_y[tid]
                    max_dist = Rc_px + max_r_edge_px
                    
                    x_min = max(0, int(np.floor(bx - max_dist)))
                    x_max = min(W, int(np.ceil(bx + max_dist)) + 1)
                    y_min = max(0, int(np.floor(by - max_dist)))
                    y_max = min(H, int(np.ceil(by + max_dist)) + 1)
                    
                    sub_spd = speed[y_min:y_max, x_min:x_max]
                    y_idx, x_idx = np.ogrid[y_min:y_max, x_min:x_max]
                    dist = np.hypot(x_idx - bx, y_idx - by)
                    
                    # 仮想マスク: 貨物の縁からの距離 r = dist - Rc_px
                    r_edge = dist - Rc_px
                    valid_pixels = (r_edge >= 0) & (r_edge < max_r_edge_px)
                    
                    if not np.any(valid_pixels):
                        continue
                        
                    b_idx = np.digitize(r_edge[valid_pixels], bins_px) - 1
                    bin_counts += np.bincount(b_idx, minlength=n_bins)
                    bin_sums += np.bincount(b_idx, weights=sub_spd[valid_pixels], minlength=n_bins)
            
            # トレーサーの移流 (Euler forward with bilinear interpolation)
            if fr < max_frame_needed:
                # 画面内のトレーサーのみ速度を補間
                in_bounds = (tr_x >= 0) & (tr_x < W - 1) & (tr_y >= 0) & (tr_y < H - 1)
                coords = np.vstack([tr_y[in_bounds], tr_x[in_bounds]])
                
                vx = map_coordinates(u, coords, order=1, mode='nearest')
                vy = map_coordinates(v, coords, order=1, mode='nearest')
                
                tr_x[in_bounds] += vx
                tr_y[in_bounds] += vy
                
    mean_speed_px = bin_sums / np.maximum(bin_counts, 1)
    mean_speed_um_s = mean_speed_px * V_UNIT
    
    bin_centers_px = 0.5 * (bins_px[:-1] + bins_px[1:])
    bin_centers_um = bin_centers_px * SCALE_UM_PER_PX
    
    far_mask = bin_centers_um >= 20.0
    if np.sum(bin_counts[far_mask]) > 100:
        u_inf = np.average(mean_speed_um_s[far_mask], weights=bin_counts[far_mask])
    else:
        u_inf = np.mean(mean_speed_um_s[int(n_bins * 0.8):])
        
    norm_speed = mean_speed_um_s / u_inf if (u_inf > 0 and np.isfinite(u_inf)) else np.zeros_like(mean_speed_um_s)
    
    elapsed = time.time() - t0
    print(f"[{b_name}] Done {Path(exp_dir_str).name} in {elapsed:.1f}s | u_inf={u_inf:.4f} um/s, edge_norm={norm_speed[0]:.4f}")
    
    return {
        'exp_dir': exp_dir_str,
        'bead_name': b_name,
        'diameter_um': b_info['diameter_um'],
        'radius_um': b_info['radius_um'],
        'r_edge_um': bin_centers_um,
        'dist_um': bin_centers_um + b_info['radius_um'],
        'speed_um_s': mean_speed_um_s,
        'u_inf': u_inf,
        'norm_speed': norm_speed,
        'bin_counts': bin_counts
    }

def main():
    print(f"Starting simulation of virtual tracers on {len(TARGET_DIRS)} experiments...")
    num_workers = min(10, cpu_count())
    print(f"Using {num_workers} parallel workers.")
    
    with Pool(processes=num_workers) as pool:
        results = pool.map(process_single_experiment, TARGET_DIRS)
        
    valid_results = [r for r in results if r is not None]
    df_results = pd.DataFrame(valid_results)
    
    out_cache = Path("scratch/virtual_tracer_flow_profile_cache.pkl")
    with open(out_cache, "wb") as f:
        pickle.dump(df_results, f)
    print(f"Successfully saved virtual tracer flow profile results to {out_cache}")

if __name__ == "__main__":
    main()
