"""
ngp_analysis.py

貨物微粒子（蛍光ビーズ）のノンガウシアンパラメータ (Non-Gaussian Parameter: NGP / alpha_2)
を全ビーズサイズ条件（0.63μm, 1.18μm, 3.37μm, 5.00μm, 7.24μm, 20.0μm）に対して
一括解析・可視化するスクリプトです。
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import zarr

# libsディレクトリのインポートパス解決
current_dir = Path(__file__).parent.resolve()
parent_dir = current_dir.parent if current_dir.name == 'libs' else current_dir
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from libs import ngp

# スタイルの適用
style_path = parent_dir / 'libs' / 'my_style.mplstyle'
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
    {"name": "beads06um", "diameter_um": 0.63, "marker": "^", "label": "0.63 μm", "color": style_colors[0]},
    {"name": "beads1um",  "diameter_um": 1.18, "marker": "o", "label": "1.18 μm", "color": style_colors[1]},
    {"name": "beads3um",  "diameter_um": 3.37, "marker": "d", "label": "3.37 μm", "color": style_colors[2]},
    {"name": "beads5um",  "diameter_um": 5.00, "marker": 10,  "label": "5.00 μm", "color": style_colors[3]},
    {"name": "beads7um",  "diameter_um": 7.24, "marker": 11,  "label": "7.24 μm", "color": style_colors[4]},
    {"name": "beads20um", "diameter_um": 20.0, "marker": "s", "label": "20.0 μm", "color": style_colors[5]},
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


_THETA_CACHE = {}

def load_theta_array(exp_dir):
    """
    実験ディレクトリ内の MTs_im_theta.zarr から大域ネマチック平均配向角 theta(t) を取得する。
    """
    exp_dir_str = str(exp_dir)
    if exp_dir_str in _THETA_CACHE:
        return _THETA_CACHE[exp_dir_str]

    theta_path = Path(exp_dir) / "MTs_im_theta.zarr"
    if not theta_path.exists():
        _THETA_CACHE[exp_dir_str] = None
        return None
    try:
        MTs = zarr.open_array(str(theta_path), mode='r')
        theta = np.nanmean(MTs, axis=(1, 2))
        _THETA_CACHE[exp_dir_str] = theta
        return theta
    except Exception as e:
        print(f"[WARNING] Could not load theta from {theta_path}: {e}", flush=True)
        _THETA_CACHE[exp_dir_str] = None
        return None


def analyze_single_bead_condition(target_dir, max_tau=50, scale=0.11, frame_interval=4.0,
                                  component='2d', min_samples=5):
    """
    1つのビーズディレクトリ以下の全実験について NGP を算出する。
    """
    target_dir = Path(target_dir)
    exp_dirs = [Path(p) for p in glob.glob(str(target_dir / "*" / "*"))]
    exp_dirs = [d for d in exp_dirs if d.is_dir() and (d / "beads_tracks.csv").exists()]
    if not exp_dirs:
        exp_dirs = [Path(p) for p in glob.glob(str(target_dir / "*"))]
        exp_dirs = [d for d in exp_dirs if d.is_dir() and (d / "beads_tracks.csv").exists()]

    if not exp_dirs:
        return None

    per_exp_ngp = []
    pooled_disps = {tau: [] for tau in range(1, max_tau + 1)}
    total_particles = 0

    for i, exp_dir in enumerate(exp_dirs):
        tracks_path = exp_dir / "beads_tracks.csv"
        try:
            df_tracks = pd.read_csv(tracks_path)
        except Exception as e:
            print(f"    [WARNING] 読み込み失敗 ({tracks_path}): {e}")
            continue

        theta_array = None
        if component.lower() in ['parallel', 'par', 'perpendicular', 'perp']:
            theta_array = load_theta_array(exp_dir)
            if theta_array is None:
                continue

        total_particles += df_tracks['particle'].nunique()

        # 実験ごとの NGP を計算
        df_exp_ngp = ngp.calc_ngp_evolution(
            df_tracks,
            max_tau=max_tau,
            scale=scale,
            frame_interval=frame_interval,
            component=component,
            theta_array=theta_array,
            min_samples=min_samples
        )

        if not df_exp_ngp.empty:
            df_exp_ngp['exp_idx'] = i
            df_exp_ngp['exp_name'] = exp_dir.name
            per_exp_ngp.append(df_exp_ngp)

        # プール用変位データの抽出
        for tau in range(1, max_tau + 1):
            disp = ngp.calc_displacements_array(
                df_tracks,
                tau=tau,
                scale=scale,
                component=component,
                theta_array=theta_array
            )
            if len(disp) > 0:
                pooled_disps[tau].extend(disp)

    # 全粒子プールでの NGP 算出
    pooled_records = []
    for tau in range(1, max_tau + 1):
        disp_arr = np.asarray(pooled_disps[tau])
        if len(disp_arr) < min_samples:
            continue
        res = ngp.calc_ngp_from_displacements(disp_arr, component=component)
        if not np.isnan(res['ngp']):
            pooled_records.append({
                'tau': tau,
                'lag_time': tau * frame_interval,
                'ngp': res['ngp'],
                'msd': res['msd'],
                'm4': res['m4'],
                'count': res['count']
            })

    pooled_ngp_df = pd.DataFrame(pooled_records)

    # 実験ごとの平均 NGP と標準偏差 (実験間エラーバー用)
    if per_exp_ngp:
        combined_exp_ngp = pd.concat(per_exp_ngp, ignore_index=True)
        exp_mean = combined_exp_ngp.groupby(['tau', 'lag_time'])['ngp'].mean().reset_index()
        exp_std = combined_exp_ngp.groupby(['tau', 'lag_time'])['ngp'].std().fillna(0.0).reset_index()
        exp_cnt = combined_exp_ngp.groupby(['tau', 'lag_time'])['ngp'].count().reset_index()

        exp_ngp_summary = pd.DataFrame({
            'tau': exp_mean['tau'],
            'lag_time': exp_mean['lag_time'],
            'ngp_mean': exp_mean['ngp'],
            'ngp_std': exp_std['ngp'],
            'ngp_sem': exp_std['ngp'] / np.sqrt(exp_cnt['ngp']),
            'n_experiments': exp_cnt['ngp']
        })
    else:
        exp_ngp_summary = pd.DataFrame()

    return {
        'pooled_ngp': pooled_ngp_df,
        'exp_ngp_summary': exp_ngp_summary,
        'n_experiments': len(exp_dirs),
        'n_particles': total_particles
    }


def run_all_beads_ngp_analysis(root_dir, out_dir=None, component='2d', max_tau=50,
                               frame_interval=4.0, scale=0.11, min_samples=5,
                               xlim=(0, 50)):
    """
    全ビーズサイズ条件についてノンガウシアンパラメータ alpha_2(Δt) を一括計算し、
    グラフ（SVG/PNG）および CSV を出力する。
    """
    root = Path(root_dir)
    if out_dir is None:
        out = root / 'figure' / 'ngp'
    else:
        out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"ノンガウシアンパラメータ (NGP / α_2) 一括解析開始")
    print(f"ルートディレクトリ: {root}")
    print(f"出力先ディレクトリ: {out}")
    print(f"成分: {component}")
    print(f"パラメータ: max_tau={max_tau} frames, frame_interval={frame_interval}s, scale={scale} um/px")
    print(f"{'='*70}\n")

    beads_results = {}
    summary_rows = []

    for item in BEADS_INFO:
        b_name = item["name"]
        target_dir = root / b_name
        if not target_dir.exists():
            print(f"  ディレクトリが見つかりません: {target_dir}")
            continue

        print(f"  処理中: {item['label']} ({b_name}) ...", flush=True)
        res = analyze_single_bead_condition(
            target_dir,
            max_tau=max_tau,
            scale=scale,
            frame_interval=frame_interval,
            component=component,
            min_samples=min_samples
        )

        if res is None:
            print(f"    -> 有効なデータがありません: {b_name}")
            continue

        beads_results[b_name] = res
        print(f"    -> 実験数: {res['n_experiments']}, 推定粒子数: {res['n_particles']}")

        # サマリー行の作成
        pooled_df = res['pooled_ngp']
        exp_sum = res['exp_ngp_summary']

        for _, row in pooled_df.iterrows():
            tau_val = int(row['tau'])
            lag_t = row['lag_time']
            ngp_p = row['ngp']
            msd_val = row['msd']
            m4_val = row['m4']
            n_disp = int(row['count'])

            ngp_exp_mean = np.nan
            ngp_exp_std = np.nan
            ngp_exp_sem = np.nan
            if not exp_sum.empty:
                match_exp = exp_sum[exp_sum['tau'] == tau_val]
                if not match_exp.empty:
                    ngp_exp_mean = match_exp['ngp_mean'].iloc[0]
                    ngp_exp_std = match_exp['ngp_std'].iloc[0]
                    ngp_exp_sem = match_exp['ngp_sem'].iloc[0]

            summary_rows.append({
                'bead_name': b_name,
                'diameter_um': item["diameter_um"],
                'component': component,
                'tau_frame': tau_val,
                'lag_time_s': lag_t,
                'ngp_pooled': ngp_p,
                'ngp_exp_mean': ngp_exp_mean,
                'ngp_exp_std': ngp_exp_std,
                'ngp_exp_sem': ngp_exp_sem,
                'msd_um2': msd_val,
                'm4_um4': m4_val,
                'n_displacements': n_disp,
                'n_experiments': res['n_experiments']
            })

    if not beads_results:
        print("[ERROR] 解析可能なビーズデータがありませんでした。")
        return {}

    # サマリー CSV の保存
    df_summary = pd.DataFrame(summary_rows)
    csv_path = out / f"NGP_summary_{component}.csv"
    df_summary.to_csv(csv_path, index=False)
    print(f"\n[保存完了] サマリー CSV: {csv_path}")

    # ==========================================
    # プロット 1: 全ビーズ NGP(Δt) 比較 (線形 x, 線形 y)
    # ==========================================
    _plot_all_beads_ngp(beads_results, out, component, frame_interval, xlim)

    # ==========================================
    # プロット 2: 全ビーズ NGP(Δt) 比較 (両対数 log-log)
    # ==========================================
    _plot_all_beads_ngp_loglog(beads_results, out, component, frame_interval, xlim)

    print(f"\n{'='*70}")
    print(f"ノンガウシアンパラメータ解析が完了しました。出力先: {out}")
    print(f"{'='*70}\n")
    return beads_results


def _get_ylabel_and_title(component):
    comp = component.lower()
    if comp in ['2d', 'norm', 'magnitude', 'r']:
        ylabel = r'$\alpha_2(\Delta t) = \frac{1}{2} \frac{\langle |\Delta\mathbf{r}|^4 \rangle}{\langle |\Delta\mathbf{r}|^2 \rangle^2} - 1$'
        title = 'Non-Gaussian Parameter $\\alpha_2(\\Delta t)$ (2D norm)'
    elif comp == 'x':
        ylabel = r'$\alpha_2(\Delta t) = \frac{1}{3} \frac{\langle \Delta x^4 \rangle}{\langle \Delta x^2 \rangle^2} - 1$'
        title = 'Non-Gaussian Parameter $\\alpha_2(\\Delta t)$ ($x$-component)'
    elif comp == 'y':
        ylabel = r'$\alpha_2(\Delta t) = \frac{1}{3} \frac{\langle \Delta y^4 \rangle}{\langle \Delta y^2 \rangle^2} - 1$'
        title = 'Non-Gaussian Parameter $\\alpha_2(\\Delta t)$ ($y$-component)'
    elif comp in ['parallel', 'par']:
        ylabel = r'$\alpha_2(\Delta t) = \frac{1}{3} \frac{\langle \Delta r_\parallel^4 \rangle}{\langle \Delta r_\parallel^2 \rangle^2} - 1$'
        title = 'Non-Gaussian Parameter $\\alpha_2(\\Delta t)$ (Parallel to director)'
    elif comp in ['perpendicular', 'perp']:
        ylabel = r'$\alpha_2(\Delta t) = \frac{1}{3} \frac{\langle \Delta r_\perp^4 \rangle}{\langle \Delta r_\perp^2 \rangle^2} - 1$'
        title = 'Non-Gaussian Parameter $\\alpha_2(\\Delta t)$ (Perpendicular to director)'
    else:
        ylabel = r'$\alpha_2(\Delta t)$'
        title = f'Non-Gaussian Parameter $\\alpha_2(\\Delta t)$ ({component})'
    return ylabel, title


def _plot_all_beads_ngp(beads_results, out_dir, component, frame_interval, xlim):
    """
    全ビーズサイズの NGP(Δt) 比較プロットを作成・保存する（線形プロット）。
    """
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    ylabel, title = _get_ylabel_and_title(component)

    for item in BEADS_INFO:
        b_name = item["name"]
        if b_name not in beads_results:
            continue
        res = beads_results[b_name]
        exp_sum = res['exp_ngp_summary']
        pooled_df = res['pooled_ngp']

        if not exp_sum.empty and len(exp_sum) > 0:
            x_val = exp_sum['lag_time']
            y_val = exp_sum['ngp_mean']
            y_err = exp_sum['ngp_sem']
            n_exp = res['n_experiments']
        else:
            x_val = pooled_df['lag_time']
            y_val = pooled_df['ngp']
            y_err = None
            n_exp = 1

        valid = np.isfinite(x_val) & np.isfinite(y_val)
        if not np.any(valid):
            continue

        label_text = f'{item["label"]} ($N={n_exp}$)'
        ax.errorbar(
            x_val[valid],
            y_val[valid],
            yerr=y_err[valid] if y_err is not None else None,
            marker=item["marker"],
            color=item["color"],
            label=label_text,
            capsize=3,
            elinewidth=1.0,
            markersize=5.5,
            alpha=0.9
        )

    max_x = xlim[1] if xlim is not None else 50.0
    ax.hlines(0, 0, max_x, colors='#333333', linestyles='dashed', alpha=0.6, zorder=-1, label=r'Gaussian ($\alpha_2=0$)')

    ax.set_xlabel(r'Lag time $\Delta t$ [s]')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.legend(frameon=True, fontsize=8)
    ax.grid(True, which="both", ls="--", alpha=0.3)

    plt.tight_layout()
    svg_path = out_dir / f"NGP_linear_{component}.svg"
    png_path = out_dir / f"NGP_linear_{component}.png"
    fig.savefig(svg_path, bbox_inches='tight')
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  [保存完了] NGP 線形プロット: {svg_path}")


def _plot_all_beads_ngp_loglog(beads_results, out_dir, component, frame_interval, xlim):
    """
    全ビーズサイズの NGP(Δt) 比較プロットを作成・保存する（両対数 log-log プロット）。
    """
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    ylabel, title = _get_ylabel_and_title(component)

    for item in BEADS_INFO:
        b_name = item["name"]
        if b_name not in beads_results:
            continue
        res = beads_results[b_name]
        exp_sum = res['exp_ngp_summary']
        pooled_df = res['pooled_ngp']

        if not exp_sum.empty and len(exp_sum) > 0:
            x_val = exp_sum['lag_time']
            y_val = exp_sum['ngp_mean']
            y_err = exp_sum['ngp_sem']
        else:
            x_val = pooled_df['lag_time']
            y_val = pooled_df['ngp']
            y_err = None

        valid = (x_val > 0) & (y_val > 0) & np.isfinite(x_val) & np.isfinite(y_val)
        if not np.any(valid):
            continue

        label_text = f'{item["label"]}'
        ax.errorbar(
            x_val[valid],
            y_val[valid],
            yerr=y_err[valid] if y_err is not None else None,
            marker=item["marker"],
            color=item["color"],
            label=label_text,
            capsize=3,
            elinewidth=1.0,
            markersize=5.5,
            alpha=0.9
        )

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Lag time $\Delta t$ [s]')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=True, fontsize=8)
    ax.grid(True, which="both", ls="--", alpha=0.3)

    plt.tight_layout()
    svg_path = out_dir / f"NGP_loglog_{component}.svg"
    png_path = out_dir / f"NGP_loglog_{component}.png"
    fig.savefig(svg_path, bbox_inches='tight')
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  [保存完了] NGP 両対数プロット: {svg_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Cargo particle Non-Gaussian Parameter (NGP / alpha_2) batch analysis across all bead sizes."
    )
    parser.add_argument('--root_dir', type=str, default=None,
                        help="Root directory containing bead conditions (e.g. /Volumes/data/Sasaki/MTsingleBeads).")
    parser.add_argument('--out_dir', type=str, default=None,
                        help="Directory to save figures and CSVs (default: root_dir/figure/ngp).")
    parser.add_argument('--component', type=str, default='2d',
                        choices=['2d', 'x', 'y', 'parallel', 'perpendicular'],
                        help="Displacement component: '2d' (default), 'x', 'y', 'parallel', 'perpendicular'.")
    parser.add_argument('--max_tau', type=int, default=50,
                        help="Maximum lag time in frames (default: 50).")
    parser.add_argument('--frame_interval', type=float, default=4.0,
                        help="Time interval between frames in seconds (default: 4.0).")
    parser.add_argument('--scale', type=float, default=0.11,
                        help="Spatial conversion scale in um/pixel (default: 0.11).")
    parser.add_argument('--min_samples', type=int, default=5,
                        help="Minimum number of displacement samples required (default: 5).")
    parser.add_argument('--xlim_max', type=float, default=50.0,
                        help="Maximum lag time for x-axis in plots (default: 50.0 s).")

    args = parser.parse_args()

    root_dir = args.root_dir
    if root_dir is None:
        root_dir = find_default_root()
    root_dir = Path(root_dir)

    run_all_beads_ngp_analysis(
        root_dir=root_dir,
        out_dir=args.out_dir,
        component=args.component,
        max_tau=args.max_tau,
        frame_interval=args.frame_interval,
        scale=args.scale,
        min_samples=args.min_samples,
        xlim=(0, args.xlim_max) if args.xlim_max > 0 else None
    )


if __name__ == "__main__":
    main()
