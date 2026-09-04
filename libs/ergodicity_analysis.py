"""
ergodicity_analysis.py

貨物微粒子（蛍光ビーズ）のエルゴード性破壊パラメータ (Ergodicity Breaking Parameter: EB)
および時間平均二乗変位 (Time-Averaged MSD: TAMSD) を全ビーズサイズ条件
（0.63μm, 1.18μm, 3.37μm, 5.00μm, 7.24μm, 20.0μm）に対して一括解析・可視化するスクリプトです。
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

from libs import ergodicity as erg

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
                                  component='2d', min_track_length=10):
    """
    1つのビーズディレクトリ以下の全実験について TAMSD および EB パラメータを算出する。
    """
    target_dir = Path(target_dir)
    exp_dirs = [Path(p) for p in glob.glob(str(target_dir / "*" / "*"))]
    exp_dirs = [d for d in exp_dirs if d.is_dir() and (d / "beads_tracks.csv").exists()]
    if not exp_dirs:
        exp_dirs = [Path(p) for p in glob.glob(str(target_dir / "*"))]
        exp_dirs = [d for d in exp_dirs if d.is_dir() and (d / "beads_tracks.csv").exists()]

    if not exp_dirs:
        return None

    per_exp_eb = []
    per_exp_tamsd = []
    all_tamsd_list = []

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

        tamsd_df = erg.calc_all_particles_tamsd(
            df_tracks,
            max_tau=max_tau,
            scale=scale,
            frame_interval=frame_interval,
            component=component,
            min_track_length=min_track_length,
            theta_array=theta_array
        )

        if tamsd_df.empty:
            continue

        tamsd_df['exp_idx'] = i
        tamsd_df['exp_name'] = exp_dir.name
        all_tamsd_list.append(tamsd_df)

        # 実験ごとの EB を算出
        eb_df = erg.calc_eb_parameter(tamsd_df, min_particles=3)
        if not eb_df.empty:
            eb_df['exp_idx'] = i
            eb_df['exp_name'] = exp_dir.name
            per_exp_eb.append(eb_df)

    if not all_tamsd_list:
        return None

    combined_tamsd = pd.concat(all_tamsd_list, ignore_index=True)

    # 全粒子プールでの EB 算出
    pooled_eb = erg.calc_eb_parameter(combined_tamsd, min_particles=3)

    # 実験ごとの平均 EB と標準偏差 (実験間エラーバー用)
    if per_exp_eb:
        combined_exp_eb = pd.concat(per_exp_eb, ignore_index=True)
        exp_eb_mean = combined_exp_eb.groupby(['tau', 'lag_time'])['eb'].mean().reset_index()
        exp_eb_std = combined_exp_eb.groupby(['tau', 'lag_time'])['eb'].std().fillna(0.0).reset_index()
        exp_eb_count = combined_exp_eb.groupby(['tau', 'lag_time'])['eb'].count().reset_index()

        exp_eb_summary = pd.DataFrame({
            'tau': exp_eb_mean['tau'],
            'lag_time': exp_eb_mean['lag_time'],
            'eb_mean': exp_eb_mean['eb'],
            'eb_std': exp_eb_std['eb'],
            'eb_sem': exp_eb_std['eb'] / np.sqrt(exp_eb_count['eb']),
            'n_experiments': exp_eb_count['eb']
        })
    else:
        exp_eb_summary = pd.DataFrame()

    return {
        'all_tamsd': combined_tamsd,
        'pooled_eb': pooled_eb,
        'exp_eb_summary': exp_eb_summary,
        'n_experiments': len(exp_dirs),
        'n_particles': combined_tamsd['particle'].nunique()
    }


def run_all_beads_ergodicity_analysis(root_dir, out_dir=None, component='2d', max_tau=50,
                                      frame_interval=4.0, scale=0.11, min_track_length=10,
                                      xlim=(0, 50)):
    """
    全ビーズサイズ条件についてエルゴード性破壊パラメータ EB(Δt) を一括計算し、
    グラフ（SVG/PNG）および CSV を出力する。
    """
    root = Path(root_dir)
    if out_dir is None:
        out = root / 'figure' / 'ergodicity'
    else:
        out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"エルゴード性破壊パラメータ (EB) & TAMSD 一括解析開始")
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
            min_track_length=min_track_length
        )

        if res is None:
            print(f"    -> 有効なデータがありません: {b_name}")
            continue

        beads_results[b_name] = res
        print(f"    -> 粒子数: {res['n_particles']}, 実験数: {res['n_experiments']}")

        # サマリー行の作成
        pooled_eb = res['pooled_eb']
        exp_sum = res['exp_eb_summary']

        for _, row in pooled_eb.iterrows():
            tau_val = int(row['tau'])
            lag_t = row['lag_time']
            eb_p = row['eb']
            eb_p_err = row['eb_err']
            mean_tamsd = row['mean_tamsd']
            std_tamsd = row['std_tamsd']
            n_pts = int(row['n_particles'])

            # 実験平均値の取得
            eb_exp_mean = np.nan
            eb_exp_std = np.nan
            eb_exp_sem = np.nan
            if not exp_sum.empty:
                match_exp = exp_sum[exp_sum['tau'] == tau_val]
                if not match_exp.empty:
                    eb_exp_mean = match_exp['eb_mean'].iloc[0]
                    eb_exp_std = match_exp['eb_std'].iloc[0]
                    eb_exp_sem = match_exp['eb_sem'].iloc[0]

            summary_rows.append({
                'bead_name': b_name,
                'diameter_um': item["diameter_um"],
                'component': component,
                'tau_frame': tau_val,
                'lag_time_s': lag_t,
                'eb_pooled': eb_p,
                'eb_pooled_err': eb_p_err,
                'eb_exp_mean': eb_exp_mean,
                'eb_exp_std': eb_exp_std,
                'eb_exp_sem': eb_exp_sem,
                'mean_tamsd_um2': mean_tamsd,
                'std_tamsd_um2': std_tamsd,
                'n_particles': n_pts,
                'n_experiments': res['n_experiments']
            })

    if not beads_results:
        print("[ERROR] 解析可能なビーズデータがありませんでした。")
        return {}

    # サマリー CSV の保存
    df_summary = pd.DataFrame(summary_rows)
    csv_path = out / f"EB_summary_{component}.csv"
    df_summary.to_csv(csv_path, index=False)
    print(f"\n[保存完了] サマリー CSV: {csv_path}")

    # ==========================================
    # プロット 1: 全ビーズ EB(Δt) 比較 (両対数 & 片対数)
    # ==========================================
    _plot_all_beads_eb(beads_results, out, component, frame_interval, xlim)

    # ==========================================
    # プロット 2: 全ビーズ アンサンブル平均 TAMSD <δ^2(Δt)>
    # ==========================================
    _plot_all_beads_tamsd(beads_results, out, component, frame_interval, xlim)

    # ==========================================
    # プロット 3: ビーズ別 個別粒子 TAMSD 曲線群 (6パネルグリッド)
    # ==========================================
    _plot_individual_tamsd_grids(beads_results, out, component, frame_interval, xlim)

    print(f"\n{'='*70}")
    print(f"エルゴード性破壊パラメータ解析が完了しました。出力先: {out}")
    print(f"{'='*70}\n")
    return beads_results


def _plot_all_beads_eb(beads_results, out_dir, component, frame_interval, xlim):
    """
    全ビーズサイズの EB(Δt) 比較プロットを作成・保存する（両対数 & 線形/片対数）。
    """
    # 1. 両対数 (log-log) プロット
    fig, ax = plt.subplots(figsize=(7.0, 5.2))

    for item in BEADS_INFO:
        b_name = item["name"]
        if b_name not in beads_results:
            continue
        res = beads_results[b_name]
        exp_sum = res['exp_eb_summary']
        pooled_eb = res['pooled_eb']

        if not exp_sum.empty and len(exp_sum) > 0:
            x_val = exp_sum['lag_time']
            y_val = exp_sum['eb_mean']
            y_err = exp_sum['eb_sem'] if 'eb_sem' in exp_sum.columns else exp_sum['eb_std']
            n_exp = res['n_experiments']
        else:
            x_val = pooled_eb['lag_time']
            y_val = pooled_eb['eb']
            y_err = pooled_eb['eb_err']
            n_exp = 1

        valid = (x_val > 0) & (y_val > 0) & np.isfinite(x_val) & np.isfinite(y_val)
        if not np.any(valid):
            continue

        label_text = f'{item["label"]} ($N={res["n_particles"]}$)'
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
    ax.set_ylabel(r'Ergodicity Breaking Parameter $EB(\Delta t)$')
    ax.set_title(f'Ergodicity Breaking Parameter $EB(\\Delta t)$ ({component})')
    ax.legend(frameon=True, fontsize=8)
    ax.grid(True, which="both", ls="--", alpha=0.3)

    plt.tight_layout()
    svg_log = out_dir / f"EB_loglog_{component}.svg"
    png_log = out_dir / f"EB_loglog_{component}.png"
    fig.savefig(svg_log, bbox_inches='tight')
    fig.savefig(png_log, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  [保存完了] EB 両対数プロット: {svg_log}")

    # 2. 線形-片対数 (linear x, log y / linear y) プロット
    fig, ax = plt.subplots(figsize=(7.0, 5.2))

    for item in BEADS_INFO:
        b_name = item["name"]
        if b_name not in beads_results:
            continue
        res = beads_results[b_name]
        exp_sum = res['exp_eb_summary']
        pooled_eb = res['pooled_eb']

        if not exp_sum.empty and len(exp_sum) > 0:
            x_val = exp_sum['lag_time']
            y_val = exp_sum['eb_mean']
            y_err = exp_sum['eb_sem']
        else:
            x_val = pooled_eb['lag_time']
            y_val = pooled_eb['eb']
            y_err = pooled_eb['eb_err']

        valid = (x_val >= 0) & (y_val >= 0) & np.isfinite(x_val) & np.isfinite(y_val)
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

    ax.set_xlabel(r'Lag time $\Delta t$ [s]')
    ax.set_ylabel(r'$EB(\Delta t)$')
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.set_title(f'Ergodicity Breaking Parameter $EB(\\Delta t)$ ({component})')
    ax.legend(frameon=True, fontsize=8)
    ax.grid(True, which="both", ls="--", alpha=0.3)

    plt.tight_layout()
    svg_lin = out_dir / f"EB_linear_{component}.svg"
    png_lin = out_dir / f"EB_linear_{component}.png"
    fig.savefig(svg_lin, bbox_inches='tight')
    fig.savefig(png_lin, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  [保存完了] EB 線形プロット: {svg_lin}")


def _plot_all_beads_tamsd(beads_results, out_dir, component, frame_interval, xlim):
    """
    全ビーズのアンサンブル平均 TAMSD <δ^2(Δt)> をプロットする。
    """
    fig, ax = plt.subplots(figsize=(7.0, 5.2))

    for item in BEADS_INFO:
        b_name = item["name"]
        if b_name not in beads_results:
            continue
        res = beads_results[b_name]
        pooled_eb = res['pooled_eb']
        if pooled_eb.empty:
            continue

        x_val = pooled_eb['lag_time']
        y_val = pooled_eb['mean_tamsd']
        y_std = pooled_eb['std_tamsd']

        valid = (x_val > 0) & (y_val > 0) & np.isfinite(x_val) & np.isfinite(y_val)
        if not np.any(valid):
            continue

        label_text = f'{item["label"]}'
        ax.plot(
            x_val[valid],
            y_val[valid],
            marker=item["marker"],
            color=item["color"],
            label=label_text,
            markersize=5.5,
            alpha=0.9
        )
        ax.fill_between(
            x_val[valid],
            np.clip(y_val[valid] - y_std[valid], 1e-5, None),
            y_val[valid] + y_std[valid],
            facecolor=mcolors.to_rgba(item["color"], alpha=0.15),
            edgecolor=item["color"],
            linewidth=0.5
        )

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Lag time $\Delta t$ [s]')
    ax.set_ylabel(r'Ensemble-averaged TAMSD $\langle \overline{\delta^2(\Delta t)} \rangle$ [$\mu\mathrm{m}^2$]')
    ax.set_title(f'Time-Averaged MSD $\\langle \\overline{{\\delta^2(\\Delta t)}} \\rangle$ ({component})')
    ax.legend(frameon=True, fontsize=8)
    ax.grid(True, which="both", ls="--", alpha=0.3)

    plt.tight_layout()
    svg_path = out_dir / f"TAMSD_all_beads_{component}.svg"
    png_path = out_dir / f"TAMSD_all_beads_{component}.png"
    fig.savefig(svg_path, bbox_inches='tight')
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  [保存完了] 全ビーズ TAMSD プロット: {svg_path}")


def _plot_individual_tamsd_grids(beads_results, out_dir, component, frame_interval, xlim):
    """
    ビーズ条件ごとにサブプロットを作成し、個別粒子の TAMSD 曲線群と平均線を描画する（6パネルグリッド）。
    """
    n_beads = len(BEADS_INFO)
    n_cols = 3
    n_rows = (n_beads + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 4.5 * n_rows), squeeze=False)

    for idx, item in enumerate(BEADS_INFO):
        r = idx // n_cols
        c = idx % n_cols
        ax = axes[r, c]
        b_name = item["name"]

        if b_name not in beads_results:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{item["label"]}')
            continue

        res = beads_results[b_name]
        all_tamsd = res['all_tamsd']
        pooled_eb = res['pooled_eb']

        # 個別粒子のプロット (薄い線)
        for _, p_group in all_tamsd.groupby(['exp_idx', 'particle']):
            p_group = p_group.sort_values('lag_time')
            ax.plot(
                p_group['lag_time'],
                p_group['tamsd'],
                color='#666666',
                alpha=0.15,
                linewidth=0.8
            )

        # アンサンブル平均線の描画（太線）
        if not pooled_eb.empty:
            ax.plot(
                pooled_eb['lag_time'],
                pooled_eb['mean_tamsd'],
                color=item["color"],
                linewidth=2.2,
                label=r'$\langle \overline{\delta^2} \rangle$'
            )

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'Lag time $\Delta t$ [s]')
        ax.set_ylabel(r'$\overline{\delta^2(\Delta t)}$ [$\mu\mathrm{m}^2$]')
        ax.set_title(f'{item["label"]} ($N={res["n_particles"]}$)')
        ax.grid(True, which="both", ls="--", alpha=0.3)
        ax.legend(fontsize=7, frameon=True, loc='upper left')

    # 余分な軸の非表示
    for idx in range(n_beads, n_rows * n_cols):
        r = idx // n_cols
        c = idx % n_cols
        axes[r, c].axis('off')

    plt.tight_layout()
    svg_path = out_dir / f"TAMSD_individual_grids_{component}.svg"
    png_path = out_dir / f"TAMSD_individual_grids_{component}.png"
    fig.savefig(svg_path, bbox_inches='tight')
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  [保存完了] 個別 TAMSD グリッドプロット: {svg_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Cargo particle Ergodicity Breaking Parameter (EB) and TAMSD batch analysis across all bead sizes."
    )
    parser.add_argument('--root_dir', type=str, default=None,
                        help="Root directory containing bead conditions (e.g. /Volumes/data/Sasaki/MTsingleBeads).")
    parser.add_argument('--out_dir', type=str, default=None,
                        help="Directory to save figures and CSVs (default: root_dir/figure/ergodicity).")
    parser.add_argument('--component', type=str, default='2d',
                        choices=['2d', 'x', 'y', 'parallel', 'perpendicular'],
                        help="Displacement component: '2d' (default), 'x', 'y', 'parallel', 'perpendicular'.")
    parser.add_argument('--max_tau', type=int, default=50,
                        help="Maximum lag time in frames (default: 50).")
    parser.add_argument('--frame_interval', type=float, default=4.0,
                        help="Time interval between frames in seconds (default: 4.0).")
    parser.add_argument('--scale', type=float, default=0.11,
                        help="Spatial conversion scale in um/pixel (default: 0.11).")
    parser.add_argument('--min_track_length', type=int, default=10,
                        help="Minimum trajectory length in frames to include (default: 10).")
    parser.add_argument('--xlim_max', type=float, default=50.0,
                        help="Maximum lag time for x-axis in plots (default: 50.0 s).")

    args = parser.parse_args()

    root_dir = args.root_dir
    if root_dir is None:
        root_dir = find_default_root()
    root_dir = Path(root_dir)

    run_all_beads_ergodicity_analysis(
        root_dir=root_dir,
        out_dir=args.out_dir,
        component=args.component,
        max_tau=args.max_tau,
        frame_interval=args.frame_interval,
        scale=args.scale,
        min_track_length=args.min_track_length,
        xlim=(0, args.xlim_max) if args.xlim_max > 0 else None
    )


if __name__ == "__main__":
    main()
