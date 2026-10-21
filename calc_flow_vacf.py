#!/usr/bin/env python3
"""
calc_flow_vacf.py

微小管の流ればの速度の時間自己相関関数 C_v(Δt) を計算・可視化するスクリプト。

空間上の固定点 r において、時刻 t の流速ベクトルと時刻 t + Δt の流速ベクトルの内積を計算し、
全空間・全時刻で平均します:
    C_v(Δt) = < v(r, t) · v(r, t + Δt) >_{r, t} / < |v(r, t)|^2 >_{r, t}

- 入力: 各実験フォルダの GFP_flows.h5
- 高速化: 時間方向の線形FFT畳み込み (Wiener-Khinchin) と空間グリッド総和の可換性を利用
- 並列化: 各動画をマルチプロセスで並列処理
- 可視化: MSD.py と統一されたカラーパレット・マーカー・スタイルを採用
- 出力: figure/flow_vacf/ および各実験ディレクトリへの個別・アンサンブルプロットとCSV保存
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import glob
import multiprocessing
import os
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# スタイルの適用
current_dir = Path(__file__).parent.resolve()
style_path = current_dir / 'libs' / 'my_style.mplstyle'
if style_path.exists():
    plt.style.use(str(style_path))
    style_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
else:
    style_colors = ['#882255', '#CC6677', '#DDCC77', '#999933', '#117733', '#44AA99']

# データルートディレクトリの候補
POSSIBLE_ROOTS = [
    Path('/mnt/NAS-Ebanaru/sasaki/MTsingleBeads'),
    Path('/mnt/NAS-Ebanaru/Sasaki/MTsingleBeads'),
    Path('/Volumes/data-1/Sasaki/MTsingleBeads'),
    Path('/Volumes/data-1/sasaki/MTsingleBeads'),
    Path('/Volumes/data/Sasaki/MTsingleBeads'),
    Path('/Volumes/data/Sasaki/MTSingleBeads'),
]

# MSD.py と共通のビーズ条件定義
BEADS_INFO = [
    {"name": "beads06um", "diameter_um": 0.63, "marker": "^", "color": style_colors[0]},
    {"name": "beads1um",  "diameter_um": 1.18, "marker": "o", "color": style_colors[1]},
    {"name": "beads3um",  "diameter_um": 3.37, "marker": "d", "color": style_colors[2]},
    {"name": "beads5um",  "diameter_um": 5.00, "marker": "p", "color": style_colors[3]},
    {"name": "beads7um",  "diameter_um": 7.24, "marker": "h", "color": style_colors[4]},
    {"name": "beads20um", "diameter_um": 20.0, "marker": "s", "color": style_colors[5]},
]


def find_default_root() -> Path:
    for r in POSSIBLE_ROOTS:
        if r.exists():
            for b in ['beads1um', 'beads06um', 'beads3um', 'beads5um', 'beads7um', 'beads20um']:
                if (r / b).exists() and len(list((r / b).glob('*/*beads_tracks.csv'))) > 0:
                    return r
    for r in POSSIBLE_ROOTS:
        if r.exists():
            return r
    return POSSIBLE_ROOTS[0]


def save_figure_to_all(fig, basename: str, out_dirs: List[Path]):
    """Save matplotlib Figure as both .svg and .png to all valid output directories."""
    for d in out_dirs:
        try:
            d.mkdir(parents=True, exist_ok=True)
            fig.savefig(d / f"{basename}.svg", bbox_inches='tight')
            fig.savefig(d / f"{basename}.png", bbox_inches='tight')
        except Exception as e:
            print(f"Warning: Failed to save {basename} to {d}: {e}")


def save_csv_to_all(df: pd.DataFrame, filename: str, out_dirs: List[Path]):
    """Save DataFrame as CSV to all valid output directories."""
    for d in out_dirs:
        try:
            d.mkdir(parents=True, exist_ok=True)
            df.to_csv(d / filename, index=False)
        except Exception as e:
            print(f"Warning: Failed to save {filename} to {d}: {e}")


def compute_movie_flow_vacf(h5_path_str: str, stride: int = 16, max_frames: Optional[int] = None) -> Dict:
    """
    1本の動画の GFP_flows.h5 から微小管流速の時間自己相関関数 C_v(Δt) を計算する。
    """
    h5_path = Path(h5_path_str)
    exp_dir = h5_path.parent
    cache_csv = exp_dir / f"flow_vacf_stride{stride}.csv"

    # キャッシュが存在する場合は高速読み込み
    if cache_csv.exists():
        try:
            df_cached = pd.read_csv(cache_csv)
            if 'lag_frames' in df_cached.columns and 'Cv' in df_cached.columns:
                return {
                    'exp_name': exp_dir.name,
                    'h5_path': str(h5_path),
                    'exp_dir': str(exp_dir),
                    'num_frames': len(df_cached),
                    'lag_frames': df_cached['lag_frames'].values,
                    'Cv': df_cached['Cv'].values,
                    'mean_v_sq': float(df_cached['mean_v_sq'].iloc[0]) if 'mean_v_sq' in df_cached.columns else np.nan,
                    'from_cache': True,
                }
        except Exception as e:
            print(f"Warning: Failed to read cache {cache_csv}: {e}")

    t0 = time.time()
    with h5py.File(h5_path, 'r') as f:
        total_frames = f['flows'].shape[0]
        T = min(total_frames, max_frames) if max_frames else total_frames
        
    # 空間グリッド数
    H = (2160 + stride - 1) // stride
    W = (2560 + stride - 1) // stride
    
    # マルチスレッドによるフレーム並行読み出し (I/O & lzf 展開の並行化)
    arr = np.empty((T, 2, H, W), dtype=np.float32)
    n_threads = min(8, T)
    
    def thread_read_worker(frame_indices):
        with h5py.File(h5_path, 'r') as f_thread:
            d_thread = f_thread['flows']
            for i in frame_indices:
                arr[i] = d_thread[i, :, ::stride, ::stride]

    frame_chunks = np.array_split(np.arange(T), n_threads)
    with ThreadPoolExecutor(max_workers=n_threads) as t_executor:
        list(t_executor.map(thread_read_worker, frame_chunks))

    # 時間方向 FFT による高速相関計算
    # 線形相関のため長さ 2*T にゼロパディング
    V = np.fft.rfft(arr, n=2 * T, axis=0) # shape: (T + 1, 2, H, W)
    P = np.sum(np.abs(V) ** 2, axis=1)    # shape: (T + 1, H, W)
    P_sum = np.sum(P, axis=(1, 2))        # shape: (T + 1,)
    sum_dot = np.fft.irfft(P_sum, n=2 * T)[:T] # shape: (T,)

    # 正規化係数: 各ラグ Δt における時空間サンプル数 (T - Δt) * (H * W)
    sample_counts = (T - np.arange(T, dtype=np.float64)) * (H * W)
    numer = sum_dot / sample_counts

    # 分母: < |v(r, t)|^2 >_{r, t} = numer[0]
    mean_v_sq = float(numer[0])
    Cv = (numer / mean_v_sq).astype(np.float64)

    df_out = pd.DataFrame({
        'lag_frames': np.arange(T),
        'Cv': Cv,
        'mean_v_sq': mean_v_sq,
        'num_frames': T,
        'stride': stride,
    })
    try:
        df_out.to_csv(cache_csv, index=False)
    except Exception:
        pass

    elapsed = time.time() - t0
    print(f"Computed flow VACF for {exp_dir.name} (T={T}, stride={stride}) in {elapsed:.1f} s")

    return {
        'exp_name': exp_dir.name,
        'h5_path': str(h5_path),
        'exp_dir': str(exp_dir),
        'num_frames': T,
        'lag_frames': np.arange(T),
        'Cv': Cv,
        'mean_v_sq': mean_v_sq,
        'from_cache': False,
    }


def main():
    parser = argparse.ArgumentParser(description="Calculate and plot microtubule flow velocity autocorrelation function Cv(Δt).")
    parser.add_argument("--root", type=str, default=None, help="Root directory of dataset")
    parser.add_argument("--stride", type=int, default=16, help="Spatial subsampling stride (default: 16)")
    parser.add_argument("--max_lag_s", type=float, default=600.0, help="Maximum lag time in seconds for plotting (default: 600.0)")
    parser.add_argument("--frame_interval", type=float, default=4.0, help="Frame interval in seconds (default: 4.0)")
    parser.add_argument("--n_workers", type=int, default=6, help="Number of parallel worker processes")
    parser.add_argument("--max_frames", type=int, default=None, help="Maximum number of frames to load per movie (for fast check)")
    args = parser.parse_args()

    root_dir = Path(args.root) if args.root else find_default_root()
    workspace_dir = Path(__file__).parent.resolve()
    
    out_dirs = [
        workspace_dir / "figure" / "flow_vacf",
        root_dir / "figure" / "flow_vacf",
    ]
    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)

    print("==================================================================")
    print("Microtubule Flow Velocity Autocorrelation Function: Cv(Δt)")
    print(f"Data Root Directory: {root_dir}")
    print(f"Spatial Stride: {args.stride} (approx. {(2160//args.stride)*(2560//args.stride)} grid points per frame)")
    print(f"Frame Interval: {args.frame_interval} s")
    print(f"Max Lag Time: {args.max_lag_s} s")
    print(f"Output Dirs: {[str(d) for d in out_dirs]}")
    print("==================================================================")

    # 1. 各ビーズ条件の HDF5 ファイルを探索
    movie_tasks = []
    for b_idx, b_item in enumerate(BEADS_INFO):
        b_name = b_item["name"]
        b_dir = root_dir / b_name
        h5_list = sorted(list(b_dir.glob("*/*/GFP_flows.h5")))
        for h5_p in h5_list:
            movie_tasks.append((b_name, b_idx, str(h5_p)))

    print(f"Found {len(movie_tasks)} movies across {len(BEADS_INFO)} bead conditions.")

    # 2. 並列計算
    results_by_bead = {b["name"]: [] for b in BEADS_INFO}
    all_movie_results = []

    print(f"\nComputing flow VACF using {args.n_workers} processes...")
    with ProcessPoolExecutor(max_workers=args.n_workers) as executor:
        future_map = {
            executor.submit(compute_movie_flow_vacf, task[2], args.stride, args.max_frames): task
            for task in movie_tasks
        }
        for future in as_completed(future_map):
            task = future_map[future]
            b_name, b_idx, h5_p = task
            try:
                res = future.result()
                res["bead_name"] = b_name
                results_by_bead[b_name].append(res)
                all_movie_results.append(res)
            except Exception as e:
                print(f"Error computing {h5_p}: {e}")

    # 3. 条件別および全アンサンブルの平均・SEM集計
    max_lag_frames = int(np.ceil(args.max_lag_s / args.frame_interval))
    lags_common_s = np.arange(max_lag_frames + 1) * args.frame_interval

    summary_rows = []
    beads_ensemble = {}

    for b_item in BEADS_INFO:
        b_name = b_item["name"]
        d_um = b_item["diameter_um"]
        res_list = results_by_bead[b_name]
        if not res_list:
            continue

        # 各動画の Cv を共通ラグ時間に切り出し／補間
        cv_matrix = []
        for r in res_list:
            t_frames = r['lag_frames']
            cv_vals = r['Cv']
            # max_lag_frames まで
            interp_cv = np.interp(np.arange(max_lag_frames + 1), t_frames, cv_vals, right=np.nan)
            cv_matrix.append(interp_cv)
        
        cv_arr = np.array(cv_matrix) # shape: (N_movies, max_lag_frames + 1)
        n_valid = np.sum(~np.isnan(cv_arr), axis=0)
        valid_mask = n_valid > 0
        mean_cv = np.full(max_lag_frames + 1, np.nan)
        std_cv = np.full(max_lag_frames + 1, np.nan)
        sem_cv = np.full(max_lag_frames + 1, np.nan)

        if np.any(valid_mask):
            mean_cv[valid_mask] = np.nanmean(cv_arr[:, valid_mask], axis=0)
            if cv_arr.shape[0] > 1:
                std_cv[valid_mask] = np.nanstd(cv_arr[:, valid_mask], axis=0, ddof=1)
                sem_cv[valid_mask] = np.where(n_valid[valid_mask] > 1, std_cv[valid_mask] / np.sqrt(n_valid[valid_mask]), 0.0)
            else:
                std_cv[valid_mask] = 0.0
                sem_cv[valid_mask] = 0.0

        beads_ensemble[b_name] = {
            'mean': mean_cv,
            'std': std_cv,
            'sem': sem_cv,
            'n_movies': len(res_list),
            'cv_matrix': cv_arr,
        }

        for lag_idx, t_s in enumerate(lags_common_s):
            summary_rows.append({
                'bead_name': b_name,
                'diameter_um': d_um,
                'lag_time_s': t_s,
                'Cv_mean': mean_cv[lag_idx],
                'Cv_std': std_cv[lag_idx],
                'Cv_sem': sem_cv[lag_idx],
                'n_movies': int(n_valid[lag_idx]),
            })

    # 全動画（全体のバルク微小管流動）アンサンブル
    all_cv_matrix = []
    for r in all_movie_results:
        t_frames = r['lag_frames']
        cv_vals = r['Cv']
        interp_cv = np.interp(np.arange(max_lag_frames + 1), t_frames, cv_vals, right=np.nan)
        all_cv_matrix.append(interp_cv)
    all_cv_arr = np.array(all_cv_matrix)
    all_n_valid = np.sum(~np.isnan(all_cv_arr), axis=0)
    all_valid_mask = all_n_valid > 0
    all_mean_cv = np.full(max_lag_frames + 1, np.nan)
    all_std_cv = np.full(max_lag_frames + 1, np.nan)
    all_sem_cv = np.full(max_lag_frames + 1, np.nan)

    if np.any(all_valid_mask):
        all_mean_cv[all_valid_mask] = np.nanmean(all_cv_arr[:, all_valid_mask], axis=0)
        all_std_cv[all_valid_mask] = np.nanstd(all_cv_arr[:, all_valid_mask], axis=0, ddof=1)
        all_sem_cv[all_valid_mask] = np.where(all_n_valid[all_valid_mask] > 1, all_std_cv[all_valid_mask] / np.sqrt(all_n_valid[all_valid_mask]), 0.0)

    for lag_idx, t_s in enumerate(lags_common_s):
        summary_rows.append({
            'bead_name': 'all_bulk_flow',
            'diameter_um': np.nan,
            'lag_time_s': t_s,
            'Cv_mean': all_mean_cv[lag_idx],
            'Cv_std': all_std_cv[lag_idx],
            'Cv_sem': all_sem_cv[lag_idx],
            'n_movies': int(all_n_valid[lag_idx]),
        })

    df_summary = pd.DataFrame(summary_rows)
    save_csv_to_all(df_summary, "flow_vacf_summary.csv", out_dirs)

    # 4. 可視化 (MSD.py スタイルに準拠)
    print("\nPlotting flow velocity autocorrelation functions...")

    # 図1: 全体バルク微小管流動の平均自己相関関数 Cv(Δt)
    fig1, ax1 = plt.subplots(figsize=(8.5, 6.2))
    ax1.axhline(0, color='#888888', linestyle=':', linewidth=1.2)
    # 個別動画（薄くプロット）
    for i in range(len(all_movie_results)):
        ax1.plot(lags_common_s, all_cv_arr[i], color='#999999', alpha=0.25, linewidth=1.0)
    # 全体平均 ± SEM
    ax1.fill_between(lags_common_s, all_mean_cv - all_sem_cv, all_mean_cv + all_sem_cv,
                     color='#332288', alpha=0.25, edgecolor='none')
    ax1.plot(lags_common_s, all_mean_cv, color='#332288', linestyle='-', linewidth=2.5,
             label=f'All bulk MT flow ($N = {len(all_movie_results)}$ movies)')

    ax1.set_xlim(0, args.max_lag_s)
    ax1.set_ylim(-0.2, 1.05)
    ax1.set_xlabel(r"Lag time $\Delta t$ [s]")
    ax1.set_ylabel(r"Flow Velocity Autocorrelation $C_v(\Delta t)$")
    ax1.set_title("Microtubule Flow Velocity Autocorrelation Function")
    ax1.legend(loc="upper right", framealpha=0.9)
    save_figure_to_all(fig1, "flow_vacf_bulk_all", out_dirs)
    plt.close(fig1)

    # 図2: ビーズ条件ごとの微小管流速自己相関関数 (MSD.py カラー・マーカー対応)
    fig2, ax2 = plt.subplots(figsize=(8.5, 6.2))
    ax2.axhline(0, color='#888888', linestyle=':', linewidth=1.2)

    # マーカー間隔（プロットがマーカーで埋もれないよう間引き表示）
    mark_stride = max(1, len(lags_common_s) // 25)

    for b_item in BEADS_INFO:
        b_name = b_item["name"]
        d_um = b_item["diameter_um"]
        color = b_item["color"]
        marker = b_item["marker"]

        if b_name not in beads_ensemble:
            continue
        st = beads_ensemble[b_name]
        mean_cv = st['mean']
        sem_cv = st['sem']
        n_m = st['n_movies']

        # SEM 帯
        ax2.fill_between(lags_common_s, mean_cv - sem_cv, mean_cv + sem_cv,
                         color=color, alpha=0.18, edgecolor='none')
        # 実線 + マーカー
        ax2.plot(lags_common_s, mean_cv, marker=marker, markevery=mark_stride,
                 linestyle='-', linewidth=2.0, markersize=8, color=color,
                 label=f'{d_um:.2f} \u03bcm ($N = {n_m}$)')

    ax2.set_xlim(0, args.max_lag_s)
    ax2.set_ylim(-0.2, 1.05)
    ax2.set_xlabel(r"Lag time $\Delta t$ [s]")
    ax2.set_ylabel(r"Flow Velocity Autocorrelation $C_v(\Delta t)$")
    ax2.set_title("Microtubule Flow Autocorrelation by Condition")
    ax2.legend(loc="upper right", framealpha=0.9, fontsize=12)
    save_figure_to_all(fig2, "flow_vacf_by_condition", out_dirs)
    plt.close(fig2)

    # 図3: 片対数プロット (緩和時間の比較・指数減衰の検証)
    fig3, ax3 = plt.subplots(figsize=(8.5, 6.2))
    ax3.axhline(0, color='#888888', linestyle=':', linewidth=1.2)
    for b_item in BEADS_INFO:
        b_name = b_item["name"]
        d_um = b_item["diameter_um"]
        color = b_item["color"]
        marker = b_item["marker"]

        if b_name not in beads_ensemble:
            continue
        st = beads_ensemble[b_name]
        mean_cv = st['mean']
        n_m = st['n_movies']

        # 正の領域のみプロット
        pos_mask = mean_cv > 1e-4
        ax3.plot(lags_common_s[pos_mask], mean_cv[pos_mask], marker=marker, markevery=mark_stride,
                 linestyle='-', linewidth=2.0, markersize=8, color=color,
                 label=f'{d_um:.2f} \u03bcm ($N = {n_m}$)')

    ax3.set_yscale('log')
    ax3.set_xlim(0, args.max_lag_s)
    ax3.set_ylim(1e-2, 1.05)
    ax3.set_xlabel(r"Lag time $\Delta t$ [s]")
    ax3.set_ylabel(r"Flow Velocity Autocorrelation $C_v(\Delta t)$")
    ax3.set_title("Semilog: Flow Velocity Autocorrelation $C_v(\Delta t)$")
    ax3.legend(loc="upper right", framealpha=0.9, fontsize=12)
    save_figure_to_all(fig3, "flow_vacf_by_condition_semilog", out_dirs)
    plt.close(fig3)

    print("\nCalculation and plotting completed successfully!")
    print(f"Results saved to:\n  - {out_dirs[0]}\n  - {out_dirs[1]}")


if __name__ == "__main__":
    main()
