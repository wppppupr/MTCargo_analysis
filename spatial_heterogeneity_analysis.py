"""
spatial_heterogeneity_analysis.py

微小管アクティブフロー（GFP_flows.h5）および貨物微粒子軌跡データから、
空間配向不均一性（Spatial Orientational Heterogeneity）を包括的に解析・可視化・定量化するスクリプトです。

解析項目:
1. 4点配向相関関数 G_4(r) / chi_orient(r) (局所ペア相関の空間分散)
   chi_orient(r) = < (u(x) · u(x+r))^2 >_x - C(r)^2
2. 内積確率密度関数 P(c; r) の非ガウス性パラメータ (NGP)
   alpha_{2, C}(r) = < c^4 >_r / (3 * < c^2 >_r^2) - 1   (c in [-1, 1])
3. 局所相関長 xi(x) の空間分布の直接評価 & 空間NGP
   alpha_{2, xi} = < xi^4 > / (3 * < xi^2 >^2) - 1

使用方法例:
  # 単一の実験ディレクトリを解析
  python spatial_heterogeneity_analysis.py --input_dir /path/to/experiment

  # 全ビーズ粒子径条件を一括バッチ解析 & 粒子径依存性サマリーを出力
  python spatial_heterogeneity_analysis.py --root_dir /Volumes/data-1/Sasaki/MTsingleBeads --batch
"""

import argparse
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from libs import spatial_heterogeneity as sh

# スタイル設定
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
    Path('/Volumes/data-1/Sasaki/MTsingleBeads'),
    Path('/Volumes/data-1/sasaki/MTsingleBeads'),
    Path('/Volumes/data/Sasaki/MTsingleBeads'),
    Path('/Volumes/data/sasaki/MTsingleBeads'),
    Path('/mnt/NAS-Ebanaru/Sasaki/MTsingleBeads'),
]


def parse_distances(distance_args: List[str]) -> List[int]:
    """Parse distance arguments (e.g., '2:100:2', '10 20 30')."""
    sizes = set()
    for arg_w in distance_args:
        for p in arg_w.split(','):
            p = p.strip()
            if not p:
                continue
            if ':' in p:
                parts = p.split(':')
                w_start = int(parts[0])
                w_stop = int(parts[1]) if len(parts) > 1 else w_start
                w_step = int(parts[2]) if len(parts) > 2 else 1
                sizes.update(range(w_start, w_stop + 1, w_step))
            else:
                sizes.add(int(p))
    return sorted(list(sizes))


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


def find_flow_exp_dirs(root_dir: Path, bead_name: str) -> List[Path]:
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


def analyze_single_flow_movie(
    flow_h5_path: Path,
    distances_px: List[int],
    scale: float = 0.11,
    grid_step: int = 8,
    max_frames: Optional[int] = None,
    frame_step: int = 1,
    device: Optional[str] = None,
) -> Tuple[dict, dict, dict]:
    """
    1つの微小管オプティカルフロー動画 (GFP_flows.h5) を時間平均して
    4点相関、内積NGP、局所相関長マップを算出する。
    """
    with h5py.File(str(flow_h5_path), 'r') as f:
        k = list(f.keys())[0]
        flow_data = f[k]
        shape = flow_data.shape
        num_frames = shape[0]

        if shape[1] == 2:
            channel_first = True
            rows, cols = shape[2], shape[3]
        elif shape[3] == 2:
            channel_first = False
            rows, cols = shape[1], shape[2]
        else:
            channel_first = True
            rows, cols = shape[2], shape[3]

        frames_to_process = list(range(0, num_frames, frame_step))
        if max_frames is not None:
            frames_to_process = frames_to_process[:max_frames]

        c_r_accum = []
        c2_r_accum = []
        chi_orient_accum = []
        chi_var_accum = []

        # NGP用 サンプルプール
        all_samples_by_dist = {d: [] for d in distances_px}

        # 局所相関長マップ用: フレームごとのマップを平均
        xi_maps = []
        alpha_xi_list = []

        rng = np.random.default_rng(42)

        for t in tqdm(frames_to_process, desc=f"Analyzing {flow_h5_path.parent.name}", leave=False):
            if channel_first:
                mx = flow_data[t, 0, ...].astype(np.float32)
                my = flow_data[t, 1, ...].astype(np.float32)
            else:
                mx = flow_data[t, ..., 0].astype(np.float32)
                my = flow_data[t, ..., 1].astype(np.float32)

            mag = np.hypot(mx, my)
            valid_mask = (mag > 1e-4).astype(np.float32)

            # 1. 4点配向相関
            res_4p = sh.calc_4point_correlation_fft(
                mx, my, distances_px=distances_px, valid_mask=valid_mask, device=device
            )
            c_r_accum.append(res_4p['C_r'])
            c2_r_accum.append(res_4p['C2_r'])
            chi_orient_accum.append(res_4p['chi_orient'])
            chi_var_accum.append(res_4p['chi_orient_local_var'])

            # 2. 内積サンプリング
            samples_t = sh.sample_field_dot_products(
                mx, my, distances_px=distances_px, n_samples_per_dist=10000, valid_mask=valid_mask > 0.5, rng=rng
            )
            for d in distances_px:
                if len(samples_t[d]) > 0:
                    all_samples_by_dist[d].append(samples_t[d])

            # 3. 局所相関長マップ (5フレーム毎または代表フレーム)
            if t % (frame_step * 5) == 0 or t == frames_to_process[-1]:
                res_xi = sh.calc_local_correlation_length_map(
                    mx, my, distances_px=distances_px, scale=scale, grid_step=grid_step, valid_mask=valid_mask, device=device
                )
                xi_maps.append(res_xi['xi_map_um'])
                if not np.isnan(res_xi['alpha_2_xi']):
                    alpha_xi_list.append(res_xi['alpha_2_xi'])

        # 時間平均の集計
        mean_c_r = np.nanmean(np.array(c_r_accum), axis=0)
        mean_c2_r = np.nanmean(np.array(c2_r_accum), axis=0)
        mean_chi = mean_c2_r - (mean_c_r ** 2)
        mean_chi_var = np.nanmean(np.array(chi_var_accum), axis=0)

        four_point_summary = {
            'distances_px': np.array(distances_px, dtype=np.float32),
            'C_r': mean_c_r,
            'C2_r': mean_c2_r,
            'chi_orient': mean_chi,
            'chi_orient_local_var': mean_chi_var,
        }

        # NGP集計
        merged_samples = {}
        for d in distances_px:
            if len(all_samples_by_dist[d]) > 0:
                merged_samples[d] = np.concatenate(all_samples_by_dist[d])
            else:
                merged_samples[d] = np.empty(0, dtype=np.float32)

        ngp_summary = sh.calc_dot_product_distribution_and_ngp(merged_samples, n_bins=50)

        # 相関長マップ集計
        if xi_maps:
            mean_xi_map = np.nanmean(np.array(xi_maps), axis=0)
            valid_xi = mean_xi_map[~np.isnan(mean_xi_map)]
            if len(valid_xi) >= 10:
                xi2_m = float(np.mean(valid_xi ** 2))
                xi4_m = float(np.mean(valid_xi ** 4))
                mean_alpha_xi = (xi4_m / (3.0 * (xi2_m ** 2))) - 1.0 if xi2_m > 1e-8 else np.nan
                mean_xi_val = float(np.mean(valid_xi))
                median_xi_val = float(np.median(valid_xi))
                std_xi_val = float(np.std(valid_xi))
            else:
                mean_alpha_xi = float(np.nanmean(alpha_xi_list)) if alpha_xi_list else np.nan
                mean_xi_val = np.nan
                median_xi_val = np.nan
                std_xi_val = np.nan
        else:
            mean_xi_map = np.empty((0, 0))
            valid_xi = np.empty(0)
            mean_alpha_xi = np.nan
            mean_xi_val = np.nan
            median_xi_val = np.nan
            std_xi_val = np.nan

        local_xi_summary = {
            'grid_x_um': np.arange(0, cols, grid_step) * scale,
            'grid_y_um': np.arange(0, rows, grid_step) * scale,
            'xi_map_um': mean_xi_map,
            'xi_valid_um': valid_xi,
            'alpha_2_xi': float(mean_alpha_xi),
            'xi_mean_um': float(mean_xi_val),
            'xi_median_um': float(median_xi_val),
            'xi_std_um': float(std_xi_val),
        }

        return four_point_summary, ngp_summary, local_xi_summary


def plot_cross_condition_comparison(
    summary_df: pd.DataFrame,
    cond_curves: dict,
    out_fig_path: Path,
):
    """
    全粒子径条件（beads06um 〜 beads20um）を横断した総合比較図（4パネル）を作成する。
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes = axes.flatten()

    cond_color_map = {b['name']: b['color'] for b in BEADS_INFO}
    cond_marker_map = {b['name']: b['marker'] for b in BEADS_INFO}
    cond_diam_map = {b['name']: b['diameter_um'] for b in BEADS_INFO}

    # -------------------------------------------------------------
    # Panel A: 条件別 4点配向相関 chi_orient(r) 曲線比較
    # -------------------------------------------------------------
    ax_a = axes[0]
    for b in BEADS_INFO:
        c_name = b['name']
        if c_name in cond_curves and 'chi_orient' in cond_curves[c_name]:
            r_um = cond_curves[c_name]['distances_um']
            chi = cond_curves[c_name]['chi_orient']
            ax_a.plot(r_um, chi, color=b['color'], marker=b['marker'], ms=4, lw=2, label=f"{c_name} ({b['diameter_um']}$\\mu\\mathrm{{m}}$)")

    ax_a.set_xlabel(r'Distance $r\ [\mu\mathrm{m}]$', fontsize=12)
    ax_a.set_ylabel(r'4-Point Susceptibility $\chi_{\mathrm{orient}}(r)$', fontsize=12)
    ax_a.set_title(r'(a) 4-Point Correlation $\chi_{\mathrm{orient}}(r)$ across Conditions', fontsize=13, fontweight='bold')
    ax_a.grid(True, linestyle=':', alpha=0.6)
    ax_a.legend(loc='best', fontsize=9, frameon=True)

    # -------------------------------------------------------------
    # Panel B: 条件別 内積 NGP alpha_{2, C}(r) 曲線比較
    # -------------------------------------------------------------
    ax_b = axes[1]
    for b in BEADS_INFO:
        c_name = b['name']
        if c_name in cond_curves and 'ngp' in cond_curves[c_name]:
            r_um = cond_curves[c_name]['distances_um']
            ngp = cond_curves[c_name]['ngp']
            ax_b.plot(r_um, ngp, color=b['color'], marker=b['marker'], ms=4, lw=2, label=f"{c_name} ({b['diameter_um']}$\\mu\\mathrm{{m}}$)")

    ax_b.set_xlabel(r'Distance $r\ [\mu\mathrm{m}]$', fontsize=12)
    ax_b.set_ylabel(r'Dot Product NGP $\alpha_{2, C}(r)$', fontsize=12)
    ax_b.set_title(r'(b) Dot Product NGP $\alpha_{2, C}(r)$ across Conditions', fontsize=13, fontweight='bold')
    ax_b.grid(True, linestyle=':', alpha=0.6)
    ax_b.legend(loc='best', fontsize=9, frameon=True)

    # -------------------------------------------------------------
    # Panel C: 粒子径 vs 4点相関ピーク強度 chi_max
    # -------------------------------------------------------------
    ax_c = axes[2]
    if 'diameter_um' in summary_df.columns and 'chi_max' in summary_df.columns:
        grouped = summary_df.groupby('condition')
        for b in BEADS_INFO:
            c_name = b['name']
            if c_name in grouped.groups:
                sub = grouped.get_group(c_name)
                d = b['diameter_um']
                chi_m = sub['chi_max'].mean()
                chi_err = sub['chi_max'].sem() if len(sub) > 1 else 0.0
                ax_c.errorbar([d], [chi_m], yerr=[chi_err], fmt=b['marker'], color=b['color'],
                             ms=8, capsize=4, elinewidth=1.5, label=c_name)

        ax_c.set_xscale('log')
        ax_c.set_xlabel(r'Bead Diameter $d\ [\mu\mathrm{m}]$', fontsize=12)
        ax_c.set_ylabel(r'Peak 4-Point Susceptibility $\chi_{\mathrm{max}}$', fontsize=12)
        ax_c.set_title(r'(c) Domain Contrast $\chi_{\mathrm{max}}$ vs Bead Diameter', fontsize=13, fontweight='bold')
        ax_c.grid(True, linestyle=':', alpha=0.6, which='both')

    # -------------------------------------------------------------
    # Panel D: 粒子径 vs 空間相関長 NGP alpha_{2, xi}
    # -------------------------------------------------------------
    ax_d = axes[3]
    if 'diameter_um' in summary_df.columns and 'alpha_2_xi' in summary_df.columns:
        grouped = summary_df.groupby('condition')
        for b in BEADS_INFO:
            c_name = b['name']
            if c_name in grouped.groups:
                sub = grouped.get_group(c_name)
                d = b['diameter_um']
                a_m = sub['alpha_2_xi'].mean()
                a_err = sub['alpha_2_xi'].sem() if len(sub) > 1 else 0.0
                ax_d.errorbar([d], [a_m], yerr=[a_err], fmt=b['marker'], color=b['color'],
                             ms=8, capsize=4, elinewidth=1.5, label=c_name)

        ax_d.set_xscale('log')
        ax_d.set_xlabel(r'Bead Diameter $d\ [\mu\mathrm{m}]$', fontsize=12)
        ax_d.set_ylabel(r'Spatial Correlation Length NGP $\alpha_{2, \xi}$', fontsize=12)
        ax_d.set_title(r'(d) Spatial Heterogeneity $\alpha_{2, \xi}$ vs Bead Diameter', fontsize=13, fontweight='bold')
        ax_d.grid(True, linestyle=':', alpha=0.6, which='both')

    fig.suptitle('Spatial Orientational Heterogeneity Summary across Bead Diameters', fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()

    out_fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig_path, dpi=300, bbox_inches='tight')
    print(f"[INFO] Saved condition comparison figure to {out_fig_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze spatial orientational heterogeneity (4-point correlation, dot-product NGP, local xi map).")
    parser.add_argument('--input_dir', type=str, default=None, help='Single experiment directory path containing GFP_flows.h5')
    parser.add_argument('--root_dir', type=str, default=None, help='Root dataset directory')
    parser.add_argument('--batch', action='store_true', help='Run batch analysis across all bead diameter conditions')
    parser.add_argument('--distances', nargs='+', default=['2:100:2', '110:300:10'],
                        help='Distance shells r (pixels) (default: 2:100:2 110:300:10)')
    parser.add_argument('--scale', type=float, default=0.11, help='Spatial scale (um/pixel, default: 0.11)')
    parser.add_argument('--grid_step', type=int, default=8, help='Grid step (pixels) for local xi map (default: 8)')
    parser.add_argument('--max_frames', type=int, default=100, help='Max frames per movie to analyze (default: 100)')
    parser.add_argument('--frame_step', type=int, default=2, help='Frame subsampling step (default: 2)')
    parser.add_argument('--device', type=str, default=None, help="Execution device: 'cuda', 'scipy', or None (auto)")
    parser.add_argument('--out_dir', type=str, default='figure/spatial_heterogeneity', help='Output figure/results directory')
    args = parser.parse_args()

    distances_px = parse_distances(args.distances)
    scale = args.scale
    distances_um = np.array(distances_px, dtype=np.float32) * scale
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 単一ディレクトリ解析モード
    if args.input_dir:
        in_path = Path(args.input_dir)
        flow_path = in_path / 'GFP_flows.h5'
        if not flow_path.exists():
            raise FileNotFoundError(f"GFP_flows.h5 not found in {in_path}")

        print(f"\n=======================================================")
        print(f"Analyzing Single Experiment: {in_path.name}")
        print(f"=======================================================")
        four_p, ngp_res, xi_res = analyze_single_flow_movie(
            flow_path, distances_px=distances_px, scale=scale, grid_step=args.grid_step,
            max_frames=args.max_frames, frame_step=args.frame_step, device=args.device,
        )

        fig_path = out_dir / f"{in_path.name}_heterogeneity_dashboard.png"
        sh.plot_spatial_heterogeneity_summary(
            distances_um=distances_um,
            four_point_result=four_p,
            ngp_result=ngp_res,
            local_xi_result=xi_res,
            condition_name=in_path.name,
            save_path=fig_path,
        )
        return

    # 2. バッチ解析モード
    root_dir = Path(args.root_dir) if args.root_dir else find_default_root()
    print(f"[INFO] Using root directory: {root_dir}")

    summary_records = []
    cond_curves = {}

    for b in BEADS_INFO:
        cond_name = b['name']
        exp_dirs = find_flow_exp_dirs(root_dir, cond_name)
        if not exp_dirs:
            print(f"[INFO] Skipping {cond_name} (no GFP_flows.h5 found)")
            continue

        print(f"\n=======================================================")
        print(f"Condition: {cond_name} ({b['diameter_um']} um) - Found {len(exp_dirs)} experiments")
        print(f"=======================================================")

        cond_chi_list = []
        cond_ngp_list = []
        cond_xi_mean_list = []
        cond_alpha_xi_list = []

        for edir in exp_dirs:
            flow_path = edir / 'GFP_flows.h5'
            try:
                four_p, ngp_res, xi_res = analyze_single_flow_movie(
                    flow_path, distances_px=distances_px, scale=scale, grid_step=args.grid_step,
                    max_frames=args.max_frames, frame_step=args.frame_step, device=args.device,
                )
            except Exception as e:
                print(f"[WARNING] Failed analyzing {edir.name}: {e}")
                continue

            # 個別実験の図を出力
            exp_fig_path = out_dir / cond_name / f"{edir.name}_heterogeneity.png"
            sh.plot_spatial_heterogeneity_summary(
                distances_um=distances_um,
                four_point_result=four_p,
                ngp_result=ngp_res,
                local_xi_result=xi_res,
                condition_name=f"{cond_name} - {edir.name}",
                save_path=exp_fig_path,
            )

            # サマリー指標
            chi_vals = four_p['chi_orient']
            valid_chi = chi_vals[~np.isnan(chi_vals)]
            chi_max = float(np.max(valid_chi)) if len(valid_chi) > 0 else np.nan
            chi_max_r = float(distances_um[np.argmax(chi_vals)]) if len(valid_chi) > 0 else np.nan

            ngp_vals = ngp_res['ngp']
            valid_ngp = ngp_vals[~np.isnan(ngp_vals)]
            ngp_max = float(np.max(valid_ngp)) if len(valid_ngp) > 0 else np.nan

            summary_records.append({
                'condition': cond_name,
                'diameter_um': b['diameter_um'],
                'exp_dir': edir.name,
                'chi_max': chi_max,
                'chi_max_r_um': chi_max_r,
                'ngp_max': ngp_max,
                'alpha_2_xi': xi_res['alpha_2_xi'],
                'xi_mean_um': xi_res['xi_mean_um'],
                'xi_median_um': xi_res['xi_median_um'],
                'xi_std_um': xi_res['xi_std_um'],
            })

            cond_chi_list.append(chi_vals)
            cond_ngp_list.append(ngp_vals)
            if not np.isnan(xi_res['alpha_2_xi']):
                cond_alpha_xi_list.append(xi_res['alpha_2_xi'])

        if cond_chi_list:
            cond_curves[cond_name] = {
                'distances_um': distances_um,
                'chi_orient': np.nanmean(np.array(cond_chi_list), axis=0),
                'ngp': np.nanmean(np.array(cond_ngp_list), axis=0),
            }

    if summary_records:
        df_summary = pd.DataFrame(summary_records)
        csv_path = out_dir / 'spatial_heterogeneity_summary.csv'
        df_summary.to_csv(csv_path, index=False)
        print(f"\n[INFO] Saved summary CSV to {csv_path}")

        # 条件間比較プロット
        summary_fig_path = out_dir / 'heterogeneity_beads_summary.png'
        plot_cross_condition_comparison(df_summary, cond_curves, summary_fig_path)
        print(f"[INFO] Batch analysis complete!")


if __name__ == '__main__':
    main()
