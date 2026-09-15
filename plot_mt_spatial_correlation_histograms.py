#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_mt_spatial_correlation_histograms.py
=========================================
微小管（Microtubules / MT）アクティブフロー場の空間配向相関のヒストグラムを
さまざまな大きさの r (例: r = 2, 16, 32 um) でプロット・解析するスクリプト。

計算内容:
- 微小管フロー単位ベクトル u_flow(x) の空間相関:
    cos(theta_ij) = u_flow(x) · u_flow(x + r)  in [-1, 1]
    相対運動角度 Delta theta_ij = arccos(u_flow(x) · u_flow(x + r))  in [0, 180 deg]
"""

import argparse
import os
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
# 微小管フロー場の空間配向相関サンプリング
# =========================================================================
def sample_mt_flow_correlations(
    exp_dirs: List[Path],
    r_targets: List[float],
    scale: float = 0.11,
    stride: int = 4,
    max_pairs_per_r: int = 30000,
) -> Dict[float, np.ndarray]:
    """
    各実験の GFP_flows.h5 から代表フレームのオプティカルフロー場を取得し、
    距離 r における cos(Delta theta) の分布を計算する。
    """
    step_scale = scale * stride  # um/pixel in downsampled grid (0.11 * 4 = 0.44 um/pixel)
    corr_by_r = {r: [] for r in r_targets}

    for edir in exp_dirs:
        h5_path = edir / "GFP_flows.h5"
        if not h5_path.exists():
            continue

        try:
            with h5py.File(str(h5_path), 'r', locking=False) as f:
                dataset_key = 'flows' if 'flows' in f else list(f.keys())[0]
                flow_ds = f[dataset_key]
                T = flow_ds.shape[0]
                # 中盤の代表フレーム (T/2) を取得
                target_frame = int(T * 0.5)

                full_f = flow_ds[target_frame]  # shape (2, 2160, 2560)
                fx = full_f[0, ::stride, ::stride].astype(np.float32)
                fy = full_f[1, ::stride, ::stride].astype(np.float32)

                mag = np.hypot(fx, fy)
                valid = (mag > 1e-3) & np.isfinite(fx) & np.isfinite(fy)

                ux = np.zeros_like(fx)
                uy = np.zeros_like(fy)
                ux[valid] = fx[valid] / np.maximum(mag[valid], 1e-6)
                uy[valid] = fy[valid] / np.maximum(mag[valid], 1e-6)

                H, W = ux.shape

                for r_val in r_targets:
                    r_pix = r_val / step_scale
                    n_angles = 24
                    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)

                    for phi in angles:
                        dy = int(round(r_pix * np.sin(phi)))
                        dx = int(round(r_pix * np.cos(phi)))

                        if dy >= 0:
                            y1_s, y1_e = 0, H - dy
                            y2_s, y2_e = dy, H
                        else:
                            y1_s, y1_e = -dy, H
                            y2_s, y2_e = 0, H + dy

                        if dx >= 0:
                            x1_s, x1_e = 0, W - dx
                            x2_s, x2_e = dx, W
                        else:
                            x1_s, x1_e = -dx, W
                            x2_s, x2_e = 0, W + dx

                        v1 = valid[y1_s:y1_e, x1_s:x1_e]
                        v2 = valid[y2_s:y2_e, x2_s:x2_e]
                        pv = v1 & v2

                        if np.any(pv):
                            u1x = ux[y1_s:y1_e, x1_s:x1_e][pv]
                            u1y = uy[y1_s:y1_e, x1_s:x1_e][pv]
                            u2x = ux[y2_s:y2_e, x2_s:x2_e][pv]
                            u2y = uy[y2_s:y2_e, x2_s:x2_e][pv]

                            dots = u1x * u2x + u1y * u2y
                            if len(dots) > 500:
                                dots = np.random.choice(dots, 500, replace=False)
                            corr_by_r[r_val].extend(dots)

        except Exception as e:
            continue

    result = {}
    for r_val, arr in corr_by_r.items():
        if len(arr) > max_pairs_per_r:
            result[r_val] = np.random.choice(arr, max_pairs_per_r, replace=False)
        else:
            result[r_val] = np.array(arr)
    return result


# =========================================================================
# 可視化関数
# =========================================================================

def plot_focused_mt_flow_histograms(
    flow_corr_by_r: Dict[float, np.ndarray],
    r_targets: List[float],
    output_dir: Path,
):
    """
    微小管フロー場の空間配向相関ヒストグラム (2パネル: cos(Delta theta) & Delta theta [deg])。
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # パネル1: cos(Delta theta)
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
    ax1.set_title(r"(a) Distribution of Flow Dot Products $\cos(\theta_{ij})$", fontsize=12, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.4)
    ax1.legend(fontsize=9, loc='upper center', framealpha=0.9)

    # パネル2: 相対角度 Delta theta [deg]
    bins_ang = np.linspace(0, 180, 25)
    p_theory_ang = np.ones(50) / 180.0
    ax2.plot(np.linspace(0, 180, 50), p_theory_ang, 'k--', lw=1.5, alpha=0.6, label="Isotropic / Random (Theory)")

    for r_idx, r_val in enumerate(r_targets):
        color = R_COLORS[r_idx % len(R_COLORS)]
        vals = flow_corr_by_r.get(r_val, np.array([]))
        if len(vals) < 10:
            continue

        cos_clipped = np.clip(vals, -1.0, 1.0)
        angles_deg = np.degrees(np.arccos(cos_clipped))
        counts, edges = np.histogram(angles_deg, bins=bins_ang, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        mean_ang = np.mean(angles_deg)

        ax2.plot(
            centers, counts,
            drawstyle='steps-mid',
            color=color,
            lw=2.5,
            label=f"$r = {r_val:.0f}\\,\\mu\\mathrm{{m}}$ ($N={len(vals):,}$, $\\langle \\Delta\\theta \\rangle = {mean_ang:.1f}^\\circ$)",
            alpha=0.9,
        )
        ax2.fill_between(centers, counts, step='mid', color=color, alpha=0.15)

    ax2.set_xlim(0, 180)
    ax2.set_ylim(bottom=0)
    ax2.set_xlabel(r"Relative Flow Direction Angle $\Delta \theta_{ij}$ [deg]", fontsize=11)
    ax2.set_ylabel("Probability Density Function (PDF)", fontsize=11)
    ax2.set_title(r"(b) Distribution of Relative Angles $\Delta \theta_{ij}$", fontsize=12, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.4)
    ax2.legend(fontsize=9, loc='upper right', framealpha=0.9)

    plt.suptitle(r"Microtubule Flow Spatial Orientational Correlation Distributions ($r = 2, 16, 32\,\mu\mathrm{m}$)", fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()

    out_base = output_dir / "mt_flow_spatial_correlation_histograms_focused_r"
    fig.savefig(f"{out_base}.svg", dpi=300, bbox_inches='tight')
    fig.savefig(f"{out_base}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out_base}.svg / .png", flush=True)


def plot_mt_flow_by_condition_grid(
    flow_corr_by_bead_r: Dict[str, Dict[float, np.ndarray]],
    flow_pooled_by_r: Dict[float, np.ndarray],
    r_targets: List[float],
    output_dir: Path,
):
    """
    ビーズ条件ごとの微小管フロー空間相関ヒストグラム (8パネルグリッド)。
    """
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


def save_mt_histogram_summary_csv(
    flow_corr_by_r: Dict[float, np.ndarray],
    r_targets: List[float],
    output_dir: Path,
):
    """
    微小管フロー空間相関ヒストグラムの頻度・統計サマリー CSV を保存。
    """
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


def main():
    parser = argparse.ArgumentParser(
        description="Plot Microtubule (MT) active flow spatial orientational correlation histograms at specified distances r."
    )
    parser.add_argument('--root_dir', type=str, default=None, help="Root directory containing beads data.")
    parser.add_argument('--output_dir', type=str, default='figure/spatial_correlation', help="Output directory for figures.")
    parser.add_argument('--r_list', type=float, nargs='+', default=[2.0, 16.0, 32.0], help="Target distances r in um.")
    args = parser.parse_args()

    root_dir = Path(args.root_dir) if args.root_dir else find_default_root()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=================================================================", flush=True)
    print(" Microtubule (MT) Flow Spatial Orientational Correlation Analysis", flush=True)
    print("=================================================================", flush=True)
    print(f"Data Root Directory: {root_dir}", flush=True)
    print(f"Output Directory:    {output_dir}", flush=True)
    print(f"Target r distances:  {args.r_list} um", flush=True)
    print("=================================================================\n", flush=True)

    flow_corr_by_bead_r = {}
    all_flow_pairs = {r: [] for r in args.r_list}

    for binfo in BEADS_INFO:
        bname = binfo['name']
        dia = binfo['diameter_um']
        print(f"--- Sampling MT flow data for {bname} (d = {dia:.2f} um) ---", flush=True)
        edirs = find_experiment_dirs(root_dir, bname)
        if not edirs:
            print(f"  [WARNING] No experiment dirs found for {bname}", flush=True)
            continue

        print(f"  Found {len(edirs)} experiment directories. Sampling optical flows...", flush=True)
        f_data = sample_mt_flow_correlations(edirs, args.r_list)
        flow_corr_by_bead_r[bname] = f_data
        for r_val, arr in f_data.items():
            all_flow_pairs[r_val].extend(arr)
            mean_c = np.mean(arr) if len(arr) > 0 else np.nan
            print(f"    Flow r={r_val:.0f} um: N={len(arr):,}, mean_cos={mean_c:+.3f}", flush=True)

    flow_pooled_by_r = {r: np.array(arr) for r, arr in all_flow_pairs.items()}

    print("\n--- Generating Plots ---", flush=True)
    # 1. フロー相関 2パネルフォーカスプロット (cos & angle)
    plot_focused_mt_flow_histograms(flow_pooled_by_r, args.r_list, output_dir)

    # 2. 条件別 8パネルグリッド (フロー相関)
    plot_mt_flow_by_condition_grid(flow_corr_by_bead_r, flow_pooled_by_r, args.r_list, output_dir)

    # 3. サマリー CSV 出力
    save_mt_histogram_summary_csv(flow_pooled_by_r, args.r_list, output_dir)

    print("\n=================================================================", flush=True)
    print(" All MT flow spatial correlation histogram plots completed successfully!", flush=True)
    print("=================================================================", flush=True)


if __name__ == "__main__":
    main()
