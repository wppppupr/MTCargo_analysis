#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_mt_spatial_correlation_histograms.py
=========================================
微小管（Microtubules / MT）アクティブフロー場の空間配向相関のヒストグラムを
さまざまな大きさの r (例: r = 2, 16, 32 um) で高速かつ高精度にプロット・解析するスクリプト。

主な最適化 & 特徴:
1. 24方向固定サンプリング (24 Fixed Angle Directions):
   元解析との統計的一貫性を保つため、angles = np.linspace(0, 2*pi, 24, endpoint=False)
   の各方向ごとにランダム点サンプリングを実施（巨大配列スライス・コピーを完全排除）。
2. Adaptive Sampling による最低サンプル数保証:
   境界落ちや無効点によるサンプル不足を防ぐ while ループ再試行（max_trials=5）。
3. list.extend() の排除:
   NumPy 配列の append & 最後に np.concatenate() による高速結合。
4. HDF5 読み込み最適化:
   ds[t, 0, ::stride, ::stride] で必要データのみを直接スライス読み込み。
5. ProcessPoolExecutor による実験単位の並列化:
   マルチコア CPU による並列 I/O & サンプリング処理。
6. np.random.default_rng(seed) による高速乱数 & 再現性保証。
7. ステージ別プロファイル計測 (I/O, Normalization, Sampling, Histogram, Plotting) & cProfile サポート。
"""

import argparse
import concurrent.futures
import cProfile
import io
import os
import pstats
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# NAS / 共有ボリュームでの HDF5 ファイルロックエラー防止
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# 親ディレクトリのパス設定
current_dir = Path(__file__).parent.resolve()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# スタイルの適用
style_path = current_dir / 'libs' / 'my_style.mplstyle'
if style_path.exists():
    try:
        plt.style.use(str(style_path))
        style_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    except Exception:
        style_colors = ['#882255', '#CC6677', '#DDCC77', '#999933', '#117733', '#44AA99']
else:
    style_colors = ['#882255', '#CC6677', '#DDCC77', '#999933', '#117733', '#44AA99']

BEADS_INFO = [
    {"name": "beads06um", "diameter_um": 0.63, "marker": "^", "color": style_colors[0]},
    {"name": "beads1um",  "diameter_um": 1.18, "marker": "o", "color": style_colors[1]},
    {"name": "beads3um",  "diameter_um": 3.37, "marker": "d", "color": style_colors[2]},
    {"name": "beads5um",  "diameter_um": 5.00, "marker": "p", "color": style_colors[3]},
    {"name": "beads7um",  "diameter_um": 7.24, "marker": "h", "color": style_colors[4]},
    {"name": "beads20um", "diameter_um": 20.0, "marker": "s", "color": style_colors[5]},
]

POSSIBLE_ROOTS = [
    Path('/Volumes/data/Sasaki/MTsingleBeads'),
    Path('/Volumes/data-1/Sasaki/MTsingleBeads'),
    Path('/Volumes/data/sasaki/MTsingleBeads'),
    Path('/Volumes/data-1/sasaki/MTsingleBeads'),
    Path('/mnt/NAS-Ebanaru/Sasaki/MTsingleBeads'),
]

R_COLORS = ['#d95f02', '#7570b3', '#1b9e77', '#e7298a', '#66a61e', '#e6ab02']


def find_default_root() -> Path:
    for r in POSSIBLE_ROOTS:
        if r.exists():
            for b in ['beads1um', 'beads06um', 'beads3um', 'beads5um', 'beads7um', 'beads20um']:
                if (r / b).exists() and len(list((r / b).glob('*/*GFP_flows.h5'))) > 0:
                    return r
    for r in POSSIBLE_ROOTS:
        if r.exists():
            return r
    return POSSIBLE_ROOTS[0]


def find_experiment_dirs(root_dir: Path, bead_name: str) -> List[Path]:
    base = Path(root_dir) / bead_name
    if not base.exists():
        return []
    exp_dirs = []
    for p in sorted(base.glob("*/*")):
        if p.is_dir() and (p / "GFP_flows.h5").exists():
            exp_dirs.append(p)
    if not exp_dirs:
        for p in sorted(base.glob("*")):
            if p.is_dir() and (p / "GFP_flows.h5").exists():
                exp_dirs.append(p)
    return exp_dirs


# =========================================================================
# 単一実験の並列処理ワーカー関数
# =========================================================================

def process_single_experiment_flow(
    h5_path_str: str,
    r_targets: List[float],
    scale: float = 0.11,
    stride: int = 4,
    target_pairs_per_r: int = 12000,  # 24 directions x 500 pairs
    n_directions: int = 24,
    seed: int = 42,
    max_trials: int = 5,
) -> Tuple[Dict[float, np.ndarray], Dict[str, float], Dict[str, int]]:
    """
    1つの実験 HDF5 ファイルからオプティカルフローを読み込み、
    24方向固定サンプリング & Adaptive Sampling により各 r の内積分布を計算する。

    Returns:
    --------
    corr_by_r : Dict[float, np.ndarray]
    timings : Dict[str, float] (io, norm, sampling)
    stats_info : Dict[str, int] (requested, collected)
    """
    h5_path = Path(h5_path_str)
    step_scale = scale * stride  # 0.11 * 4 = 0.44 um/pixel
    rng = np.random.default_rng(seed)

    timings = {'io': 0.0, 'norm': 0.0, 'sampling': 0.0}
    stats_info = {'requested': len(r_targets) * target_pairs_per_r, 'collected': 0}
    corr_by_r = {r: np.empty(0, dtype=np.float32) for r in r_targets}

    if not h5_path.exists():
        return corr_by_r, timings, stats_info

    # 1. HDF5 最適化読み込み (中央フレームの fx, fy のみをストライドスライス取得)
    t0 = time.perf_counter()
    try:
        with h5py.File(str(h5_path), 'r', locking=False) as f:
            dataset_key = 'flows' if 'flows' in f else list(f.keys())[0]
            flow_ds = f[dataset_key]
            T = flow_ds.shape[0]
            target_frame = int(T * 0.5)

            # ストライド付きで直接読み込み (メモリ消費を 1/16 に削減)
            fx = flow_ds[target_frame, 0, ::stride, ::stride].astype(np.float32)
            fy = flow_ds[target_frame, 1, ::stride, ::stride].astype(np.float32)
    except Exception as e:
        return corr_by_r, timings, stats_info
    timings['io'] = time.perf_counter() - t0

    # 2. ベクトル正規化 & 有効点抽出
    t0 = time.perf_counter()
    mag = np.hypot(fx, fy)
    valid = (mag > 1e-3) & np.isfinite(fx) & np.isfinite(fy)

    ux = np.zeros_like(fx)
    uy = np.zeros_like(fy)
    ux[valid] = fx[valid] / np.maximum(mag[valid], 1e-6)
    uy[valid] = fy[valid] / np.maximum(mag[valid], 1e-6)

    valid_y, valid_x = np.nonzero(valid)
    n_valid = len(valid_y)
    H, W = ux.shape
    timings['norm'] = time.perf_counter() - t0

    if n_valid == 0:
        return corr_by_r, timings, stats_info

    # 3. 24方向固定サンプリング & Adaptive Sampling
    t0 = time.perf_counter()
    angles = np.linspace(0, 2 * np.pi, n_directions, endpoint=False)
    pairs_per_dir = int(np.ceil(target_pairs_per_r / n_directions))

    for r_val in r_targets:
        r_pix = r_val / step_scale
        r_dots_list = []

        for phi in angles:
            dy = int(round(r_pix * np.sin(phi)))
            dx = int(round(r_pix * np.cos(phi)))

            collected_dir = []
            n_collected = 0
            trial = 0

            # Adaptive sampling: 目標点数に達するまで最大 max_trials 回サンプリング
            while n_collected < pairs_per_dir and trial < max_trials:
                trial += 1
                needed = pairs_per_dir - n_collected
                # 境界落ちや無効点落ちを見越して 1.3倍サンプリング
                batch_size = max(10, int(needed * 1.3))

                idx = rng.integers(0, n_valid, size=batch_size)
                y1 = valid_y[idx]
                x1 = valid_x[idx]

                y2 = y1 + dy
                x2 = x1 + dx

                in_bounds = (y2 >= 0) & (y2 < H) & (x2 >= 0) & (x2 < W)
                y2_safe = np.clip(y2, 0, H - 1)
                x2_safe = np.clip(x2, 0, W - 1)

                pair_valid = in_bounds & valid[y2_safe, x2_safe]
                valid_count = np.count_nonzero(pair_valid)

                if valid_count > 0:
                    y1_v = y1[pair_valid][:needed]
                    x1_v = x1[pair_valid][:needed]
                    y2_v = y2[pair_valid][:needed]
                    x2_v = x2[pair_valid][:needed]

                    dots = ux[y1_v, x1_v] * ux[y2_v, x2_v] + uy[y1_v, x1_v] * uy[y2_v, x2_v]
                    collected_dir.append(dots)
                    n_collected += len(dots)

            if collected_dir:
                r_dots_list.append(np.concatenate(collected_dir))

        if r_dots_list:
            combined_dots = np.concatenate(r_dots_list)
            corr_by_r[r_val] = combined_dots
            stats_info['collected'] += len(combined_dots)

    timings['sampling'] = time.perf_counter() - t0
    return corr_by_r, timings, stats_info


# =========================================================================
# 並列実行制御関数
# =========================================================================

def sample_mt_flow_parallel(
    exp_dirs: List[Path],
    r_targets: List[float],
    scale: float = 0.11,
    stride: int = 4,
    target_pairs_per_r: int = 12000,
    n_directions: int = 24,
    max_workers: Optional[int] = None,
    base_seed: int = 42,
) -> Tuple[Dict[float, np.ndarray], Dict[str, float]]:
    """
    全実験を ProcessPoolExecutor で並列処理し、結果を集約する。
    """
    if max_workers is None:
        max_workers = min(8, os.cpu_count() or 4)

    corr_by_r_chunks = {r: [] for r in r_targets}
    total_timings = {'io': 0.0, 'norm': 0.0, 'sampling': 0.0}

    tasks = []
    for exp_idx, edir in enumerate(exp_dirs):
        h5_path = edir / "GFP_flows.h5"
        if not h5_path.exists():
            continue
        tasks.append((str(h5_path), base_seed + exp_idx))

    if not tasks:
        return {r: np.empty(0, dtype=np.float32) for r in r_targets}, total_timings

    # 並列ワーカーの実行
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_exp = {
            executor.submit(
                process_single_experiment_flow,
                h5_str,
                r_targets,
                scale,
                stride,
                target_pairs_per_r,
                n_directions,
                seed,
            ): h5_str for h5_str, seed in tasks
        }

        for future in concurrent.futures.as_completed(future_to_exp):
            h5_str = future_to_exp[future]
            try:
                corr_res, t_res, s_res = future.result()
                for r_val in r_targets:
                    if len(corr_res[r_val]) > 0:
                        corr_by_r_chunks[r_val].append(corr_res[r_val])
                for k in total_timings:
                    total_timings[k] += t_res[k]
            except Exception as e:
                print(f"[WARNING] Experiment failed: {h5_str} ({e})", flush=True)

    # np.concatenate による一括統合 (list.extend を排除)
    final_corr_by_r = {}
    for r_val in r_targets:
        if corr_by_r_chunks[r_val]:
            final_corr_by_r[r_val] = np.concatenate(corr_by_r_chunks[r_val])
        else:
            final_corr_by_r[r_val] = np.empty(0, dtype=np.float32)

    return final_corr_by_r, total_timings


# =========================================================================
# 可視化関数群
# =========================================================================

def plot_focused_mt_flow_histograms(
    flow_corr_by_r: Dict[float, np.ndarray],
    r_targets: List[float],
    output_dir: Path,
) -> float:
    """
    微小管フロー場の空間配向相関ヒストグラム (cos(theta_ij) の単一パネル)。
    """
    t0 = time.perf_counter()
    fig, ax1 = plt.subplots(1, 1, figsize=(7.5, 5.5))

    # cos(Delta theta) ヒストグラム
    bins_cos = np.linspace(-1.0, 1.0, 31)
    x_theory = np.linspace(-0.99, 0.99, 300)
    p_theory_cos = 1.0 / (np.pi * np.sqrt(1.0 - x_theory**2))

    ax1.plot(x_theory, p_theory_cos, 'k--', lw=1.5, alpha=0.6, label="Isotropic / Random (Theory)")

    for r_idx, r_val in enumerate(r_targets):
        color = R_COLORS[r_idx % len(R_COLORS)]
        vals = flow_corr_by_r.get(r_val, np.array([]))
        if len(vals) < 10:
            continue

        counts, edges = np.histogram(vals, bins=bins_cos, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        mean_c = np.mean(vals)
        sem_c = stats.sem(vals)

        ax1.plot(
            centers, counts,
            drawstyle='steps-mid',
            color=color,
            lw=2.5,
            label=f"$r = {r_val:.0f}\\,\\mu\\mathrm{{m}}$ ($N={len(vals):,}$, $\\langle \\cos\\theta \\rangle = {mean_c:+.3f} \\pm {sem_c:.3f}$)",
            alpha=0.9,
        )
        ax1.fill_between(centers, counts, step='mid', color=color, alpha=0.15)

    ax1.set_xlim(-1.05, 1.05)
    ax1.set_ylim(bottom=0)
    ax1.set_xlabel(r"Microtubule Flow Direction Correlation $\cos(\theta_{ij}) = \hat{\mathbf{u}}_i \cdot \hat{\mathbf{u}}_j$", fontsize=11)
    ax1.set_ylabel("Probability Density Function (PDF)", fontsize=11)
    r_str = ", ".join([f"{r:.0f}" for r in r_targets])
    ax1.set_title(f"Microtubule Flow Spatial Orientational Correlation ($r = {r_str}\\,\\mu\\mathrm{{m}}$)", fontsize=12, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.4)
    ax1.legend(fontsize=9.5, loc='upper center', framealpha=0.9)

    plt.tight_layout()

    out_base = output_dir / "mt_flow_spatial_correlation_histograms_focused_r"
    fig.savefig(f"{out_base}.svg", dpi=300, bbox_inches='tight')
    fig.savefig(f"{out_base}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out_base}.svg / .png", flush=True)
    return time.perf_counter() - t0


def plot_mt_flow_by_condition_grid(
    flow_corr_by_bead_r: Dict[str, Dict[float, np.ndarray]],
    flow_pooled_by_r: Dict[float, np.ndarray],
    r_targets: List[float],
    output_dir: Path,
) -> float:
    """
    ビーズ条件ごとの微小管フロー空間相関ヒストグラム (8パネルグリッド)。
    """
    t0 = time.perf_counter()
    fig, axes = plt.subplots(2, 4, figsize=(18, 8.5))
    axes = axes.flatten()

    bins_cos = np.linspace(-1.0, 1.0, 25)
    x_theory = np.linspace(-0.98, 0.98, 200)
    p_theory_cos = 1.0 / (np.pi * np.sqrt(1.0 - x_theory**2))

    for idx, binfo in enumerate(BEADS_INFO):
        ax = axes[idx]
        bname = binfo['name']
        dia = binfo['diameter_um']

        ax.set_title(f"$d = {dia:.2f}\\,\\mu\\mathrm{{m}}$ Background Flow", fontsize=12, fontweight='bold')
        ax.plot(x_theory, p_theory_cos, 'k--', lw=1.2, alpha=0.5, label="Isotropic" if idx == 0 else "")

        b_data = flow_corr_by_bead_r.get(bname, {})

        for r_idx, r_val in enumerate(r_targets):
            color = R_COLORS[r_idx % len(R_COLORS)]
            vals = b_data.get(r_val, np.array([]))
            if len(vals) < 10:
                continue

            counts, edges = np.histogram(vals, bins=bins_cos, density=True)
            centers = 0.5 * (edges[:-1] + edges[1:])
            mean_c = np.mean(vals)

            ax.plot(
                centers, counts,
                drawstyle='steps-mid',
                color=color,
                lw=2.0,
                label=f"$r \\approx {r_val:.0f}\\,\\mu\\mathrm{{m}}$ ($N={len(vals):,}$, $\\langle C \\rangle={mean_c:.2f}$)",
                alpha=0.9,
            )
            ax.fill_between(centers, counts, step='mid', color=color, alpha=0.15)

        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(bottom=0)
        ax.set_xlabel(r"$\cos(\theta_{ij}) = \hat{\mathbf{u}}_i \cdot \hat{\mathbf{u}}_j$", fontsize=10)
        ax.set_ylabel("Probability Density (PDF)", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.4)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, fontsize=8, loc='upper center', framealpha=0.85)

    # 7番目: 全体プール (All Conditions Pooled)
    ax_pool = axes[6]
    ax_pool.set_title("All Background Flows Pooled", fontsize=12, fontweight='bold', color='#333333')
    ax_pool.plot(x_theory, p_theory_cos, 'k--', lw=1.2, alpha=0.5, label="Isotropic")

    for r_idx, r_val in enumerate(r_targets):
        color = R_COLORS[r_idx % len(R_COLORS)]
        vals = flow_pooled_by_r.get(r_val, np.array([]))
        if len(vals) < 10:
            continue

        counts, edges = np.histogram(vals, bins=bins_cos, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        mean_c = np.mean(vals)

        ax_pool.plot(
            centers, counts,
            drawstyle='steps-mid',
            color=color,
            lw=2.2,
            label=f"$r \\approx {r_val:.0f}\\,\\mu\\mathrm{{m}}$ ($N={len(vals):,}$, $\\langle C \\rangle={mean_c:.2f}$)",
            alpha=0.95,
        )
        ax_pool.fill_between(centers, counts, step='mid', color=color, alpha=0.15)

    ax_pool.set_xlim(-1.05, 1.05)
    ax_pool.set_ylim(bottom=0)
    ax_pool.set_xlabel(r"$\cos(\theta_{ij}) = \hat{\mathbf{u}}_i \cdot \hat{\mathbf{u}}_j$", fontsize=10)
    ax_pool.set_ylabel("Probability Density (PDF)", fontsize=10)
    ax_pool.grid(True, linestyle='--', alpha=0.4)
    ax_pool.legend(fontsize=8, loc='upper center', framealpha=0.85)

    # 8番目: サマリーパネル (平均フロー相関 <C(r)> vs r)
    ax_sum = axes[7]
    ax_sum.set_title(r"Mean Flow Correlation $\langle C(r) \rangle$ vs $r$", fontsize=12, fontweight='bold')

    for idx, binfo in enumerate(BEADS_INFO):
        bname = binfo['name']
        dia = binfo['diameter_um']
        col = binfo['color']
        m = binfo['marker']
        b_data = flow_corr_by_bead_r.get(bname, {})

        r_pts = []
        c_pts = []
        for r_val in r_targets:
            vals = b_data.get(r_val, np.array([]))
            if len(vals) > 0:
                r_pts.append(r_val)
                c_pts.append(np.mean(vals))

        if r_pts:
            ax_sum.plot(r_pts, c_pts, marker=m, color=col, lw=1.8, ms=6, label=f"{dia:.2f} $\\mu$m", alpha=0.85)

    ax_sum.axhline(0, color='gray', linestyle='--', lw=1.0, alpha=0.6)
    ax_sum.set_xlim(0, max(r_targets) + 10)
    ax_sum.set_ylim(-0.1, 1.0)
    ax_sum.set_xlabel(r"Distance $r$ [$\mu\mathrm{m}$]", fontsize=10)
    ax_sum.set_ylabel(r"Mean $\langle \cos\theta \rangle$", fontsize=10)
    ax_sum.grid(True, linestyle='--', alpha=0.4)
    ax_sum.legend(fontsize=8, loc='upper right', framealpha=0.85)

    fig.suptitle(f"Microtubule Flow Spatial Orientational Correlation Histograms across Conditions ($r = {', '.join([f'{r:.0f}' for r in r_targets])}\\,\\mu\\mathrm{{m}}$)", fontsize=15, fontweight='bold', y=0.99)
    plt.tight_layout()

    out_base = output_dir / "mt_flow_spatial_corr_grid_across_conditions"
    fig.savefig(f"{out_base}.svg", dpi=300, bbox_inches='tight')
    fig.savefig(f"{out_base}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out_base}.svg / .png", flush=True)
    return time.perf_counter() - t0


def save_mt_histogram_summary_csv(
    flow_corr_by_r: Dict[float, np.ndarray],
    r_targets: List[float],
    output_dir: Path,
) -> float:
    """
    微小管フロー空間相関ヒストグラムの頻度・統計サマリー CSV を保存。
    """
    t0 = time.perf_counter()
    bins_cos = np.linspace(-1.0, 1.0, 21)
    bin_centers = 0.5 * (bins_cos[:-1] + bins_cos[1:])

    records = []
    for r_val in r_targets:
        f_vals = flow_corr_by_r.get(r_val, np.array([]))
        if len(f_vals) > 0:
            f_pdf, _ = np.histogram(f_vals, bins=bins_cos, density=True)
            f_raw, _ = np.histogram(f_vals, bins=bins_cos, density=False)
            for b_idx in range(len(bin_centers)):
                records.append({
                    'data_type': 'mt_optical_flow',
                    'r_target_um': r_val,
                    'cos_bin_center': bin_centers[b_idx],
                    'cos_bin_min': bins_cos[b_idx],
                    'cos_bin_max': bins_cos[b_idx+1],
                    'pdf_density': f_pdf[b_idx],
                    'raw_count': f_raw[b_idx],
                    'total_pairs_in_r': len(f_vals),
                    'mean_correlation': np.mean(f_vals),
                    'std_correlation': np.std(f_vals),
                    'sem_correlation': stats.sem(f_vals) if len(f_vals) > 1 else np.nan,
                })

    df_csv = pd.DataFrame(records)
    csv_path = output_dir / "mt_spatial_correlation_histogram_summary.csv"
    df_csv.to_csv(csv_path, index=False)
    print(f"  Saved MT histogram summary CSV: {csv_path}", flush=True)
    return time.perf_counter() - t0


# =========================================================================
# パイプライン実行 & タイミング計測
# =========================================================================

def run_pipeline(args):
    root_dir = Path(args.root_dir) if args.root_dir else find_default_root()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=================================================================", flush=True)
    print(" Microtubule (MT) Flow Spatial Orientational Correlation Analysis", flush=True)
    print("=================================================================", flush=True)
    print(f"Data Root Directory: {root_dir}", flush=True)
    print(f"Output Directory:    {output_dir}", flush=True)
    print(f"Target r distances:  {args.r_list} um", flush=True)
    print(f"Parallel Workers:    {args.max_workers or min(8, os.cpu_count() or 4)}", flush=True)
    print("=================================================================\n", flush=True)

    flow_corr_by_bead_r = {}
    all_flow_pairs = {r: [] for r in args.r_list}

    timings = {
        "HDF5 read time (CPU total)": 0.0,
        "Normalization time (CPU total)": 0.0,
        "Sampling time (CPU total)": 0.0,
        "Histogram & PDF time": 0.0,
        "Figure save time": 0.0,
        "Total pipeline wall time": 0.0,
    }

    t_wall_start = time.perf_counter()

    for binfo in BEADS_INFO:
        bname = binfo['name']
        dia = binfo['diameter_um']
        print(f"--- Sampling MT flow data for {bname} (d = {dia:.2f} um) ---", flush=True)
        edirs = find_experiment_dirs(root_dir, bname)
        if not edirs:
            print(f"  [WARNING] No experiment dirs found for {bname}", flush=True)
            continue

        print(f"  Found {len(edirs)} experiment directories. Running parallel sampling...", flush=True)
        t0 = time.perf_counter()
        f_data, t_data = sample_mt_flow_parallel(
            edirs,
            args.r_list,
            max_workers=args.max_workers,
            base_seed=args.seed,
        )
        flow_corr_by_bead_r[bname] = f_data
        dt_cond = time.perf_counter() - t0

        timings["HDF5 read time (CPU total)"] += t_data["io"]
        timings["Normalization time (CPU total)"] += t_data["norm"]
        timings["Sampling time (CPU total)"] += t_data["sampling"]

        for r_val, arr in f_data.items():
            if len(arr) > 0:
                all_flow_pairs[r_val].append(arr)
                mean_c = np.mean(arr)
                print(f"    Flow r={r_val:.0f} um: N={len(arr):,}, mean_cos={mean_c:+.3f}", flush=True)
        print(f"  Completed {bname} in {dt_cond:.2f}s (Wall time)", flush=True)

    # 全条件のプール
    t_hist_0 = time.perf_counter()
    flow_pooled_by_r = {}
    for r_val in args.r_list:
        if all_flow_pairs[r_val]:
            flow_pooled_by_r[r_val] = np.concatenate(all_flow_pairs[r_val])
        else:
            flow_pooled_by_r[r_val] = np.empty(0, dtype=np.float32)
    timings["Histogram & PDF time"] += (time.perf_counter() - t_hist_0)

    print("\n--- Generating Plots ---", flush=True)
    # 1. フロー相関 2パネルフォーカスプロット (cos & angle)
    t_plot1 = plot_focused_mt_flow_histograms(flow_pooled_by_r, args.r_list, output_dir)
    timings["Figure save time"] += t_plot1

    # 2. 条件別 8パネルグリッド (フロー相関)
    t_plot2 = plot_mt_flow_by_condition_grid(flow_corr_by_bead_r, flow_pooled_by_r, args.r_list, output_dir)
    timings["Figure save time"] += t_plot2

    # 3. サマリー CSV 出力
    t_csv = save_mt_histogram_summary_csv(flow_pooled_by_r, args.r_list, output_dir)
    timings["Figure save time"] += t_csv

    timings["Total pipeline wall time"] = time.perf_counter() - t_wall_start

    print("\n=================================================================", flush=True)
    print(" Execution Time & Performance Breakdown", flush=True)
    print("=================================================================", flush=True)
    for k, v in timings.items():
        print(f"  {k:<35}: {v:8.3f} s", flush=True)
    print("=================================================================", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Plot Microtubule (MT) active flow spatial orientational correlation histograms at specified distances r."
    )
    parser.add_argument('--root_dir', type=str, default=None, help="Root directory containing beads data.")
    parser.add_argument('--output_dir', type=str, default='figure/spatial_correlation', help="Output directory for figures.")
    parser.add_argument('--r_list', type=float, nargs='+', default=[2.0, 8.0, 32.0], help="Target distances r in um.")
    parser.add_argument('--max_workers', type=int, default=None, help="Maximum worker processes for parallel I/O & sampling.")
    parser.add_argument('--seed', type=int, default=42, help="Base random seed for reproducibility.")
    parser.add_argument('--profile', action='store_true', help="Enable detailed cProfile profiling report.")
    args = parser.parse_args()

    if args.profile:
        profiler = cProfile.Profile()
        profiler.enable()
        run_pipeline(args)
        profiler.disable()

        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('tottime')
        ps.print_stats(20)
        print("\n=================================================================", flush=True)
        print(" cProfile Function-Level Profile Report (Top 20 by internal time)", flush=True)
        print("=================================================================", flush=True)
        print(s.getvalue(), flush=True)
    else:
        run_pipeline(args)


if __name__ == "__main__":
    main()
