"""
angle_change_analysis.py

貨物微粒子（蛍光ビーズ）の進行方向角度変化（Turning Angle / Angle Change Δθ）の
確率密度関数 (PDF) および円統計（Circular Statistics）を一括解析・可視化するスクリプトです。

全ビーズサイズ（0.63μm, 1.18μm, 3.37μm, 5.0μm, 7.24μm, 20μm）および
複数のラグタイム Δt において、
1. 角度変化 PDF ヒストグラムの算出 & 実験間標準偏差エラーバンド描画
2. von Mises 分布（円正規分布）およびガウス分布によるフィッティング
3. 方向持続性パラメータ <cos Δθ> および集中度 κ の時間発展解析
4. 各種統計サマリー CSV の出力
を行います。
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

# 親ディレクトリのパス設定
current_dir = Path(__file__).parent.resolve()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from libs import angular_distribution as ang_dist

# スタイルの適用
style_path = current_dir / 'libs' / 'my_style.mplstyle'
if style_path.exists():
    try:
        plt.style.use(str(style_path))
        style_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    except Exception:
        style_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
else:
    style_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

# ビーズ条件設定
BEADS_INFO = [
    {"name": "beads06um", "diameter_um": 0.63, "marker": "^", "color": style_colors[0]},
    {"name": "beads1um",  "diameter_um": 1.18, "marker": "o", "color": style_colors[1]},
    {"name": "beads3um",  "diameter_um": 3.37, "marker": "d", "color": style_colors[2]},
    {"name": "beads5um",  "diameter_um": 5.00, "marker": 10,  "color": style_colors[3]},
    {"name": "beads7um",  "diameter_um": 7.24, "marker": 11,  "color": style_colors[4]},
    {"name": "beads20um", "diameter_um": 20.0, "marker": "s", "color": style_colors[5]},
]

POSSIBLE_ROOTS = [
    Path('/Volumes/data-1/Sasaki/MTsingleBeads'),
    Path('/Volumes/data-1/sasaki/MTsingleBeads'),
    Path('/Volumes/data/Sasaki/MTsingleBeads'),
    Path('/Volumes/data/sasaki/MTsingleBeads'),
    Path('/mnt/NAS-Ebanaru/Sasaki/MTsingleBeads'),
    Path('/mnt/NAS-Ebanaru/sasaki/MTsingleBeads'),
]


def find_default_root():
    for r in POSSIBLE_ROOTS:
        if r.exists():
            for b in ['beads1um', 'beads06um', 'beads3um', 'beads5um', 'beads7um', 'beads20um']:
                if (r / b).exists() and len(list((r / b).glob('*/*beads_tracks.csv'))) > 0:
                    return r
    for r in POSSIBLE_ROOTS:
        if r.exists():
            return r
    return POSSIBLE_ROOTS[0]


def find_experiment_dirs(root_dir, bead_name):
    base = Path(root_dir) / bead_name
    if not base.exists():
        return []
    exp_dirs = []
    for p in sorted(base.glob("*/*")):
        if p.is_dir() and (p / "beads_tracks.csv").exists():
            exp_dirs.append(p)
    if not exp_dirs:
        for p in sorted(base.glob("*")):
            if p.is_dir() and (p / "beads_tracks.csv").exists():
                exp_dirs.append(p)
    return exp_dirs


def safe_save_csv(df, target_path, max_retries=5):
    target_path = Path(target_path)
    for attempt in range(max_retries):
        try:
            df.to_csv(target_path, index=False)
            return
        except Exception as e:
            if attempt == max_retries - 1:
                try:
                    csv_text = df.to_csv(index=False)
                    with open(str(target_path), 'w', encoding='utf-8') as f:
                        f.write(csv_text)
                    return
                except Exception:
                    print(f"[WARNING] Could not save {target_path}: {e}", flush=True)
                    return
            import time
            time.sleep(0.5)


_TRACKS_CACHE = {}


def collect_angle_changes(exp_dirs, tau, scale=0.11, frame_interval=4.0, signed=True, unit='deg'):
    """
    複数の実験ディレクトリから指定 tau の角度変化データを集約する。
    """
    all_angles = []
    exp_angles = []

    for d in exp_dirs:
        d_str = str(d)
        if d_str in _TRACKS_CACHE:
            df_tracks = _TRACKS_CACHE[d_str]
        else:
            tracks_path = d / "beads_tracks.csv"
            try:
                df_tracks = pd.read_csv(tracks_path)
                _TRACKS_CACHE[d_str] = df_tracks
            except Exception as e:
                print(f"[WARNING] Failed to read {tracks_path}: {e}", flush=True)
                _TRACKS_CACHE[d_str] = None
                continue

        if df_tracks is None:
            continue

        try:
            angles = ang_dist.calc_angular_changes(
                df_tracks,
                tau=tau,
                scale=scale,
                frame_interval=frame_interval,
                signed=signed,
                unit=unit
            )
            if len(angles) > 0:
                all_angles.extend(angles)
                exp_angles.append(np.asarray(angles))
        except Exception as e:
            print(f"[ERROR] Error calculating angle changes for {d}: {e}", flush=True)

    return {
        "pooled": np.array(all_angles),
        "per_exp": exp_angles
    }


def plot_angle_pdf_across_beads(beads_data, tau, frame_interval, signed, unit, out_path,
                                bins=50, error_style='band', fit_model='von_mises'):
    """
    全ビーズサイズを1つの図で比較する角度変化PDFプロットを作成・保存する。
    """
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    tau_sec = tau * frame_interval
    unit_str = r'^\circ' if unit == 'deg' else r'\mathrm{rad}'
    xlabel = rf'Angular change $\Delta\theta$ [${unit_str}$]' if signed else rf'Absolute angular change $|\Delta\theta|$ [${unit_str}$]'
    ylabel = r'Probability density $P(\Delta\theta)$'
    fit_results = {}

    bin_range = (-180.0, 180.0) if (unit == 'deg' and signed) else (
        (0.0, 180.0) if (unit == 'deg' and not signed) else (
            (-np.pi, np.pi) if signed else (0.0, np.pi)
        )
    )

    for item in BEADS_INFO:
        b_name = item["name"]
        if b_name not in beads_data or len(beads_data[b_name]["per_exp"]) == 0:
            continue

        exp_list = beads_data[b_name]["per_exp"]
        centers, mean_pdf, std_pdf, _ = ang_dist.calc_ensemble_angle_pdf(
            exp_list, bins=bins, bin_range=bin_range, density=True, signed=signed, unit=unit
        )
        valid = (mean_pdf > 0) & np.isfinite(centers)

        n_exps = len(exp_list)
        label_text = f'{item["diameter_um"]:.2f} $\\mu\\mathrm{{m}}$ ($N={n_exps}$)'

        # 分布フィッティング
        fit_res = None
        if fit_model == 'von_mises' and signed:
            fit_res = ang_dist.fit_von_mises_pdf(centers[valid], mean_pdf[valid], pdf_std=std_pdf[valid], unit=unit)
            if fit_res is not None:
                fit_results[b_name] = fit_res
                label_text = f'{item["diameter_um"]:.2f} $\\mu\\mathrm{{m}}$ ($\kappa={fit_res["kappa"]:.2f}$, $R^2={fit_res["r_squared"]:.2f}$)'
        elif fit_model == 'gaussian':
            fit_res = ang_dist.fit_gaussian_pdf(centers[valid], mean_pdf[valid], pdf_std=std_pdf[valid])
            if fit_res is not None:
                fit_results[b_name] = fit_res
                label_text = f'{item["diameter_um"]:.2f} $\\mu\\mathrm{{m}}$ ($\sigma={fit_res["sigma"]:.1f}{unit_str}$)'

        # 1. 平均曲線のプロット
        ax.plot(
            centers[valid],
            mean_pdf[valid],
            marker=item["marker"],
            color=item["color"],
            label=label_text,
            markersize=5,
            alpha=0.9,
            linestyle='none' if fit_res is not None else '-'
        )

        # 2. フィッティング曲線（破線）
        if fit_res is not None:
            ax.plot(
                fit_res['fit_x'],
                fit_res['fit_y'],
                linestyle='--',
                color=item["color"],
                alpha=0.85,
                linewidth=1.5
            )

        # 3. 実験間標準偏差エラーバンド
        if error_style in ['band', 'both']:
            ax.fill_between(
                centers[valid],
                np.clip(mean_pdf[valid] - std_pdf[valid], 0, None),
                mean_pdf[valid] + std_pdf[valid],
                edgecolor=item["color"],
                facecolor=mcolors.to_rgba(item["color"], alpha=0.2),
                linewidth=0.5
            )
        if error_style in ['bar', 'both']:
            ax.errorbar(
                centers[valid],
                mean_pdf[valid],
                yerr=std_pdf[valid],
                fmt='none',
                ecolor=item["color"],
                elinewidth=1.0,
                capsize=2,
                alpha=0.6
            )

    # 等方的一様分布の基準線（完全ランダムウォーク）
    if signed:
        uniform_y = (1.0 / 360.0) if unit == 'deg' else (1.0 / (2.0 * np.pi))
        ax.axhline(uniform_y, color='black', linestyle=':', alpha=0.6, label='Isotropic uniform')

    if unit == 'deg':
        ax.set_xlim(-180, 180) if signed else ax.set_xlim(0, 180)
        if signed:
            ax.set_xticks([-180, -90, 0, 90, 180])
        else:
            ax.set_xticks([0, 45, 90, 135, 180])
    else:
        ax.set_xlim(-np.pi, np.pi) if signed else ax.set_xlim(0, np.pi)
        if signed:
            ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
            ax.set_xticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
        else:
            ax.set_xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
            ax.set_xticklabels([r'$0$', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$'])

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f'Angular Change PDF ($\Delta t = {tau_sec:.1f}\\mathrm{{s}}$, $\\tau = {tau}$ frames)')
    ax.legend(frameon=True, fontsize=8, loc='upper right')
    ax.grid(True, which="both", ls="--", alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVED] {out_path}", flush=True)
    return fit_results


def plot_angle_pdf_grid(all_tau_data, frame_interval, signed, unit, out_path,
                        plot_tau_list=None, bins=40, error_style='band', fit_model='von_mises'):
    """
    複数の代表ラグタイム tau をグリッド状に並べたサマリープロットを作成・保存する。
    """
    if plot_tau_list is None:
        plot_tau_list = list(all_tau_data.keys())

    n_tau = len(plot_tau_list)
    if n_tau == 0:
        return

    n_cols = min(3, n_tau)
    n_rows = (n_tau + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.8 * n_cols, 4.6 * n_rows), squeeze=False)
    unit_str = r'^\circ' if unit == 'deg' else r'\mathrm{rad}'
    xlabel = rf'$\Delta\theta$ [${unit_str}$]' if signed else rf'$|\Delta\theta|$ [${unit_str}$]'
    ylabel = r'$P(\Delta\theta)$'

    bin_range = (-180.0, 180.0) if (unit == 'deg' and signed) else (
        (0.0, 180.0) if (unit == 'deg' and not signed) else (
            (-np.pi, np.pi) if signed else (0.0, np.pi)
        )
    )

    for idx, tau in enumerate(plot_tau_list):
        r = idx // n_cols
        c = idx % n_cols
        ax = axes[r, c]
        tau_sec = tau * frame_interval

        if tau not in all_tau_data:
            continue
        beads_data = all_tau_data[tau]

        for item in BEADS_INFO:
            b_name = item["name"]
            if b_name not in beads_data or len(beads_data[b_name]["per_exp"]) == 0:
                continue

            exp_list = beads_data[b_name]["per_exp"]
            centers, mean_pdf, std_pdf, _ = ang_dist.calc_ensemble_angle_pdf(
                exp_list, bins=bins, bin_range=bin_range, density=True, signed=signed, unit=unit
            )
            valid = (mean_pdf > 0) & np.isfinite(centers)

            label_text = f'{item["diameter_um"]:.2f} $\\mu\\mathrm{{m}}$'
            fit_res = None
            if fit_model == 'von_mises' and signed:
                fit_res = ang_dist.fit_von_mises_pdf(centers[valid], mean_pdf[valid], pdf_std=std_pdf[valid], unit=unit)
                if fit_res is not None:
                    label_text = f'{item["diameter_um"]:.2f} $\\mu\\mathrm{{m}}$ ($\kappa={fit_res["kappa"]:.1f}$)'

            ax.plot(
                centers[valid],
                mean_pdf[valid],
                marker=item["marker"],
                color=item["color"],
                label=label_text,
                markersize=4.5,
                alpha=0.9,
                linestyle='none' if fit_res is not None else '-'
            )

            if fit_res is not None:
                ax.plot(
                    fit_res['fit_x'],
                    fit_res['fit_y'],
                    linestyle='--',
                    color=item["color"],
                    alpha=0.85,
                    linewidth=1.2
                )

            if error_style in ['band', 'both']:
                ax.fill_between(
                    centers[valid],
                    np.clip(mean_pdf[valid] - std_pdf[valid], 0, None),
                    mean_pdf[valid] + std_pdf[valid],
                    edgecolor=item["color"],
                    facecolor=mcolors.to_rgba(item["color"], alpha=0.2),
                    linewidth=0.5
                )

        if signed:
            uniform_y = (1.0 / 360.0) if unit == 'deg' else (1.0 / (2.0 * np.pi))
            ax.axhline(uniform_y, color='black', linestyle=':', alpha=0.5)

        if unit == 'deg':
            ax.set_xlim(-180, 180) if signed else ax.set_xlim(0, 180)
            if signed:
                ax.set_xticks([-180, -90, 0, 90, 180])
            else:
                ax.set_xticks([0, 90, 180])
        else:
            ax.set_xlim(-np.pi, np.pi) if signed else ax.set_xlim(0, np.pi)
            if signed:
                ax.set_xticks([-np.pi, 0, np.pi])
                ax.set_xticklabels([r'$-\pi$', r'$0$', r'$\pi$'])
            else:
                ax.set_xticks([0, np.pi/2, np.pi])
                ax.set_xticklabels([r'$0$', r'$\pi/2$', r'$\pi$'])

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f'$\Delta t = {tau_sec:.1f}\\mathrm{{s}}$ ($\Delta t = {tau}\\mathrm{{ frames}}$)')
        ax.grid(True, which="both", ls="--", alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=7, frameon=True, loc='upper right')

    for idx in range(n_tau, n_rows * n_cols):
        r = idx // n_cols
        c = idx % n_cols
        axes[r, c].axis('off')

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVED] {out_path}", flush=True)


def plot_persistence_evolution(df_stats, out_path):
    """
    方向持続性パラメータ <cos Δθ> のラグタイム Δt に対する減衰時間発展プロットを作成・保存する。
    """
    fig, ax = plt.subplots(figsize=(6.5, 5.0))

    for item in BEADS_INFO:
        b_name = item["name"]
        df_b = df_stats[df_stats['bead_name'] == b_name].sort_values('lag_time_s')
        if df_b.empty or 'mean_cos' not in df_b.columns:
            continue

        tau_s = df_b['lag_time_s'].to_numpy()
        mean_cos = df_b['mean_cos'].to_numpy()

        ax.plot(
            tau_s,
            mean_cos,
            marker=item["marker"],
            color=item["color"],
            label=f'{item["diameter_um"]:.2f} $\\mu\\mathrm{{m}}$',
            markersize=5.5,
            alpha=0.9,
            linestyle='-'
        )

    ax.set_xscale('log')
    ax.set_xlabel(r'Lag time $\Delta t$ [$\mathrm{s}$]')
    ax.set_ylabel(r'Directional persistence $\langle \cos\Delta\theta \rangle$')
    ax.axhline(0, color='gray', linestyle=':', alpha=0.6)
    ax.set_ylim(-0.2, 1.05)
    ax.legend(frameon=True, fontsize=8, loc='upper right')
    ax.grid(True, which="both", ls="--", alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVED] {out_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Cargo particle angular change PDF and circular statistics analysis.")
    parser.add_argument('--root_dir', type=str, default=None,
                        help="Root directory containing bead conditions.")
    parser.add_argument('--beads', type=str, nargs='+', default=['all'],
                        help="Beads conditions to analyze (e.g. beads06um beads1um ... or 'all').")
    DEFAULT_FINE_TAU = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 22, 25, 28, 30, 35, 40, 45, 50]
    DEFAULT_PLOT_TAU = [1, 2, 5, 10, 25]

    parser.add_argument('--tau', type=int, nargs='+', default=DEFAULT_FINE_TAU,
                        help="Lag time list in frames for time evolution.")
    parser.add_argument('--tau_seconds', type=float, nargs='+', default=None,
                        help="Lag time in seconds (overrides --tau if provided).")
    parser.add_argument('--plot_tau', type=int, nargs='+', default=DEFAULT_PLOT_TAU,
                        help="Representative lag times in frames to plot PDF histograms for.")
    parser.add_argument('--plot_tau_seconds', type=float, nargs='+', default=None,
                        help="Representative lag times in seconds to plot PDF histograms for.")
    parser.add_argument('--scale', type=float, default=0.11,
                        help="Spatial conversion scale (um/pixel, default: 0.11).")
    parser.add_argument('--frame_interval', type=float, default=4.0,
                        help="Time interval between frames in seconds (default: 4.0).")
    parser.add_argument('--signed', action='store_true', default=True,
                        help="Calculate signed angular change [-π, π] (default: True).")
    parser.add_argument('--abs', dest='signed', action='store_false',
                        help="Calculate absolute angular change |Δθ| [0, π].")
    parser.add_argument('--unit', type=str, default='deg', choices=['deg', 'rad'],
                        help="Angular unit: 'deg' (degrees) or 'rad' (radians) (default: deg).")
    parser.add_argument('--bins', type=int, default=50,
                        help="Number of bins for PDF histogram (default: 50).")
    parser.add_argument('--fit_model', type=str, default='von_mises', choices=['von_mises', 'gaussian', 'none'],
                        help="Distribution fitting model (default: von_mises).")
    parser.add_argument('--error_style', type=str, default='band', choices=['band', 'bar', 'both', 'none'],
                        help="Error representation style across experiments (default: band).")
    parser.add_argument('--out_dir', type=str, default=None,
                        help="Output directory to save plots and CSVs. Defaults to root_dir/figure/angle_change.")
    args = parser.parse_args()

    # ルートディレクトリの確定
    if args.root_dir is not None:
        root_dir = Path(args.root_dir)
    else:
        root_dir = find_default_root()

    print(f"=== Cargo Particle Angular Change PDF Analysis ===", flush=True)
    print(f"Root Directory: {root_dir}", flush=True)

    # 出力先ディレクトリ
    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
    else:
        out_dir = root_dir / 'figure' / 'angle_change'
    out_dir.mkdir(parents=True, exist_ok=True)

    # 対象ビーズ条件の抽出
    if 'all' in args.beads:
        selected_beads = [b['name'] for b in BEADS_INFO]
    else:
        selected_beads = []
        for item in args.beads:
            for b in item.split(','):
                b = b.strip()
                if b == 'all':
                    selected_beads = [info['name'] for info in BEADS_INFO]
                    break
                if any(info['name'] == b for info in BEADS_INFO):
                    selected_beads.append(b)
                else:
                    print(f"[WARNING] Unknown bead condition: {b}", flush=True)

    # ラグタイムの決定
    if args.tau_seconds is not None:
        tau_list = sorted(list(set([max(1, int(round(ts / args.frame_interval))) for ts in args.tau_seconds])))
    else:
        tau_list = sorted(list(set(args.tau)))

    if args.plot_tau_seconds is not None:
        plot_tau_list = sorted(list(set([max(1, int(round(ts / args.frame_interval))) for ts in args.plot_tau_seconds])))
    else:
        plot_tau_list = sorted(list(set(args.plot_tau)))

    for pt in plot_tau_list:
        if pt not in tau_list:
            tau_list.append(pt)
    tau_list = sorted(tau_list)

    print(f"Evaluation lag times (frames): {tau_list}", flush=True)
    print(f"Histogram plotting lag times (frames): {plot_tau_list}", flush=True)

    stats_records = []
    fitting_records = []
    plot_tau_data = {}

    for tau in tau_list:
        tau_sec = tau * args.frame_interval
        beads_data = {}

        for bead_name in selected_beads:
            exp_dirs = find_experiment_dirs(root_dir, bead_name)
            if not exp_dirs:
                continue

            angle_dict = collect_angle_changes(
                exp_dirs, tau=tau, scale=args.scale, frame_interval=args.frame_interval,
                signed=args.signed, unit=args.unit
            )
            beads_data[bead_name] = angle_dict

            pooled = angle_dict["pooled"]
            exp_list = angle_dict["per_exp"]

            if len(pooled) > 0:
                # 円統計の算出用 (rad)
                pooled_rad = np.deg2rad(pooled) if args.unit == 'deg' else pooled
                circ_stats = ang_dist.calc_circular_stats(pooled_rad)

                stats_records.append({
                    'bead_name': bead_name,
                    'tau_frame': tau,
                    'lag_time_s': tau_sec,
                    'signed': args.signed,
                    'unit': args.unit,
                    'n_experiments': len(exp_list),
                    'count': len(pooled),
                    'mean': float(np.mean(pooled)),
                    'std': float(np.std(pooled)),
                    'median': float(np.median(pooled)),
                    'mean_cos': circ_stats['mean_cos'],
                    'mean_sin': circ_stats['mean_sin'],
                    'mean_resultant_length': circ_stats['mean_resultant_length'],
                    'circular_mean_deg': float(np.rad2deg(circ_stats['circular_mean'])),
                    'circular_variance': circ_stats['circular_variance'],
                    'circular_std_deg': float(np.rad2deg(circ_stats['circular_std'])),
                    'mean_abs_angle_deg': circ_stats['mean_abs_angle_deg']
                })

        # 代表ラグタイムの PDF プロット保存 & フィッティング
        if tau in plot_tau_list:
            print(f"  Plotting PDF for tau={tau} frames ({tau_sec:.1f} s)...", flush=True)
            plot_tau_data[tau] = beads_data
            pdf_save_path = out_dir / f"angle_change_PDF_tau{tau_sec:.0f}s.svg"
            fit_res_dict = plot_angle_pdf_across_beads(
                beads_data,
                tau=tau,
                frame_interval=args.frame_interval,
                signed=args.signed,
                unit=args.unit,
                out_path=pdf_save_path,
                bins=args.bins,
                error_style=args.error_style,
                fit_model=args.fit_model
            )
            plot_angle_pdf_across_beads(
                beads_data,
                tau=tau,
                frame_interval=args.frame_interval,
                signed=args.signed,
                unit=args.unit,
                out_path=pdf_save_path.with_suffix('.png'),
                bins=args.bins,
                error_style=args.error_style,
                fit_model=args.fit_model
            )

            if fit_res_dict:
                for b_name, f_info in fit_res_dict.items():
                    fitting_records.append({
                        'bead_name': b_name,
                        'tau_frame': tau,
                        'lag_time_s': tau_sec,
                        'fit_model': args.fit_model,
                        'kappa': f_info.get('kappa', np.nan),
                        'kappa_err': f_info.get('kappa_err', np.nan),
                        'mu': f_info.get('mu', np.nan),
                        'r_squared': f_info.get('r_squared', np.nan)
                    })

    # 代表ラグタイムのグリッドプロット保存
    if plot_tau_data:
        grid_save_path = out_dir / "angle_change_PDF_grid.svg"
        plot_angle_pdf_grid(
            plot_tau_data,
            frame_interval=args.frame_interval,
            signed=args.signed,
            unit=args.unit,
            out_path=grid_save_path,
            plot_tau_list=plot_tau_list,
            bins=args.bins,
            error_style=args.error_style,
            fit_model=args.fit_model
        )
        plot_angle_pdf_grid(
            plot_tau_data,
            frame_interval=args.frame_interval,
            signed=args.signed,
            unit=args.unit,
            out_path=grid_save_path.with_suffix('.png'),
            plot_tau_list=plot_tau_list,
            bins=args.bins,
            error_style=args.error_style,
            fit_model=args.fit_model
        )

    # 統計サマリー CSV の保存
    if stats_records:
        df_stats = pd.DataFrame(stats_records)
        csv_stats_path = out_dir / "angle_change_statistics_summary.csv"
        safe_save_csv(df_stats, csv_stats_path)
        print(f"\n[SAVED] Statistics summary saved to {csv_stats_path}", flush=True)

        # 持続性減衰プロット <cos Δθ>(Δt)
        persist_path = out_dir / "angle_change_persistence_evolution.svg"
        plot_persistence_evolution(df_stats, persist_path)
        plot_persistence_evolution(df_stats, persist_path.with_suffix('.png'))

    if fitting_records:
        df_fits = pd.DataFrame(fitting_records)
        csv_fits_path = out_dir / "angle_change_fitting_summary.csv"
        safe_save_csv(df_fits, csv_fits_path)
        print(f"[SAVED] Fitting summary saved to {csv_fits_path}", flush=True)

    print(f"\n[DONE] Angular change analysis finished successfully!", flush=True)


if __name__ == '__main__':
    main()
