#!/usr/bin/env python3
"""
plot_polar_order_vs_displacement.py

横軸を貨物近傍のポーラーオーダー（lag time 内の平均値）、
縦軸を貨物の変位 |Delta r| とした散布図の解析・描画スクリプト。

- 任意の lag time で解析可能（CLI 引数 --lag_times で複数指定可能）
- ポーラーオーダーは lag time 内の平均値
- マーカーの色やグラフの形状・スタイルは MSD.py に準拠
- 貨物近傍領域は貨物中心を原点とし、粒子のある領域のポーラーオーダーを抽出
- 小粒子 (0.63 um, 1.18 um) はサンプリング数を確保するため近傍領域の半径を 3.37 um として扱う
- 散布図に加え、ポーラーオーダービンごとの平均値 +- SEM を重ねて描画し傾向を明瞭化
- 全ビーズ重ね描き図、6パネル個別図、複数 lag time 比較図、サマリー CSV を自動出力
"""

import argparse
import glob
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import xarray as xr

# ワークスペースと libs のインポート
CURRENT_DIR = Path(__file__).parent.resolve()
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from libs import cal_vel as cv
from libs import displacement as dpm

# スタイルの適用
style_path = CURRENT_DIR / 'libs' / 'my_style.mplstyle'
if style_path.exists():
    try:
        plt.style.use(str(style_path))
        style_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    except Exception:
        style_colors = ['#882255', '#CC6677', '#DDCC77', '#999933', '#117733', '#44AA99']
else:
    style_colors = ['#882255', '#CC6677', '#DDCC77', '#999933', '#117733', '#44AA99']

# ビーズ基本情報 (MSD.py 準拠)
BEADS_INFO = [
    {"name": "beads06um", "diameter_um": 0.63, "radius_um": 0.315, "marker": "^", "color": style_colors[0], "label": r"$0.63\,\mu\mathrm{m}$"},
    {"name": "beads1um",  "diameter_um": 1.18, "radius_um": 0.590, "marker": "o", "color": style_colors[1], "label": r"$1.18\,\mu\mathrm{m}$"},
    {"name": "beads3um",  "diameter_um": 3.37, "radius_um": 1.685, "marker": "d", "color": style_colors[2], "label": r"$3.37\,\mu\mathrm{m}$"},
    {"name": "beads5um",  "diameter_um": 5.00, "radius_um": 2.500, "marker": "p", "color": style_colors[3], "label": r"$5.00\,\mu\mathrm{m}$"},
    {"name": "beads7um",  "diameter_um": 7.24, "radius_um": 3.620, "marker": "h", "color": style_colors[4], "label": r"$7.24\,\mu\mathrm{m}$"},
    {"name": "beads20um", "diameter_um": 20.0, "radius_um": 10.00, "marker": "s", "color": style_colors[5], "label": r"$20.0\,\mu\mathrm{m}$"},
]

POSSIBLE_ROOTS = [
    Path('/mnt/NAS-Ebanaru/sasaki/MTsingleBeads'),
    Path('/mnt/NAS-Ebanaru/Sasaki/MTsingleBeads'),
    Path('/Volumes/data-1/Sasaki/MTsingleBeads'),
    Path('/Volumes/data-1/sasaki/MTsingleBeads'),
    Path('/Volumes/data/Sasaki/MTsingleBeads'),
    Path('/Volumes/data/sasaki/MTsingleBeads'),
    Path('/Volumes/data/Sasaki/MTSingleBeads'),
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
    return CURRENT_DIR


def find_experiment_dirs(root_dir: Path, bead_name: str) -> List[Path]:
    """beads_tracks.csv と local_polar_w.zarr の両方が存在する実験ディレクトリを検索"""
    base = Path(root_dir) / bead_name
    if not base.exists():
        return []
    exp_dirs = []
    for p in sorted(base.glob("*/*")):
        if p.is_dir() and (p / "beads_tracks.csv").exists() and (p / "local_polar_w.zarr").exists():
            exp_dirs.append(p)
    return exp_dirs


def get_region_radius_um(bead_item: dict, region_mode: str = 'diameter', min_region_um: float = 3.37) -> float:
    """
    貨物近傍領域の評価半径 [um] を算出する。
    - 小粒子 (0.63, 1.18 um) はサンプリング数を確保するため min_region_um (3.37 um) を下限とする。
    - region_mode == 'diameter' (既定): R_region = max(diameter, min_region_um)
    - region_mode == 'radius': R_region = max(radius, min_region_um)
    """
    d_um = float(bead_item['diameter_um'])
    r_um = float(bead_item['radius_um'])
    if d_um in [0.63, 1.18]:
        return float(min_region_um)
    if region_mode == 'diameter':
        return float(max(d_um, min_region_um))
    else:
        return float(max(r_um, min_region_um))


def calc_particle_alpha(sub_df: pd.DataFrame, min_t: float = 4.0, max_t: float = 300.0) -> float:
    """個別粒子MSDから異常拡散指数 alpha を算出"""
    mask = (sub_df['lag time'] >= min_t) & (sub_df['lag time'] <= max_t) & (sub_df['MSD'] > 0)
    if np.sum(mask) < 3:
        mask = (sub_df['MSD'] > 0)
    if np.sum(mask) < 3:
        return np.nan
    dt = sub_df['lag time'][mask].to_numpy(dtype=float)
    msd = sub_df['MSD'][mask].to_numpy(dtype=float)
    try:
        slope, _ = np.polyfit(np.log10(dt), np.log10(msd), 1)
        return float(slope)
    except Exception:
        return np.nan


def filter_active_particles(
    exp_dirs: List[Path],
    interval: float = 4.0,
    scale: float = 0.11,
    alpha_threshold: float = 0.5
) -> Dict[str, bool]:
    """
    各動画の全粒子について iMSD を計算し、alpha > alpha_threshold の粒子を判定する。
    戻り値: (exp_name, particle_id) -> is_active (bool)
    """
    active_map: Dict[Tuple[str, int], bool] = {}
    for exp_dir in exp_dirs:
        csv_p = exp_dir / "beads_tracks.csv"
        try:
            raw_df = pd.read_csv(csv_p)
            track = cv.cal(raw_df, scale=scale, frame_interval=interval)
            imsd_df = dpm.imsd(track, scale=scale, time_scale=interval, display=False)
            imsd_df['MSD'] = pd.to_numeric(imsd_df['MSD'], errors='coerce')
            imsd_df['lag time'] = pd.to_numeric(imsd_df['lag time'], errors='coerce')
            imsd_df = imsd_df.dropna(subset=['MSD', 'lag time'])
            for pid, grp in imsd_df.groupby('particle'):
                alp = calc_particle_alpha(grp, 4.0, 300.0)
                is_active = (np.isfinite(alp) and alp > alpha_threshold) if alpha_threshold > 0 else True
                active_map[(exp_dir.name, int(pid))] = is_active
        except Exception as e:
            print(f"Warning: Failed alpha calculation for {csv_p}: {e}")
    return active_map


def extract_polar_and_displacement_for_exp(
    exp_dir: Path,
    target_radius_um: float,
    lag_time_s: float,
    frame_interval: float = 4.0,
    scale: float = 0.11,
    stride: int = 1,
    active_particles_set: Optional[set] = None,
) -> pd.DataFrame:
    """
    1つの実験ディレクトリから、指定された lag time に対する
    各粒子・各フレームペアの (polar_order_mean, displacement_um) を抽出する。
    """
    tracks_csv = exp_dir / "beads_tracks.csv"
    polar_zarr = exp_dir / "local_polar_w.zarr"

    if not tracks_csv.exists() or not polar_zarr.exists():
        return pd.DataFrame()

    df_tracks = pd.read_csv(tracks_csv)
    if 'frame' not in df_tracks.columns or 'particle' not in df_tracks.columns:
        return pd.DataFrame()

    try:
        ds_polar = xr.open_zarr(str(polar_zarr), consolidated=False)
    except Exception as e:
        print(f"Error opening zarr {polar_zarr}: {e}")
        return pd.DataFrame()

    w_dim = 'window size' if 'window size' in ds_polar.dims else 'window_size'
    windows_px = ds_polar.coords[w_dim].values
    windows_um = windows_px * scale

    # 最も近い窓サイズを選択
    target_r_px = target_radius_um / scale
    w_idx = int(np.argmin(np.abs(windows_px - target_r_px)))
    actual_w_px = float(windows_px[w_idx])
    actual_w_um = float(windows_um[w_idx])

    # 選択した窓サイズの DataArray をロード (frame x particle)
    polar_da = ds_polar['polar_order'].isel({w_dim: w_idx}).load()

    lag_frames = int(round(lag_time_s / frame_interval))
    if lag_frames < 1:
        lag_frames = 1
    actual_lag_s = lag_frames * frame_interval

    records = []
    exp_name = exp_dir.name

    for pid, grp in df_tracks.groupby('particle'):
        pid_int = int(pid)
        if active_particles_set is not None and (exp_name, pid_int) not in active_particles_set:
            continue

        grp_sorted = grp.sort_values('frame')
        frames = grp_sorted['frame'].to_numpy(dtype=int)
        xs = grp_sorted['x'].to_numpy(dtype=float)
        ys = grp_sorted['y'].to_numpy(dtype=float)
        frame_to_idx = {f: i for i, f in enumerate(frames)}

        # 粒子が polar_da に存在するか確認
        if pid_int not in polar_da.coords['particle'].values:
            continue
        particle_polar = polar_da.sel(particle=pid_int).to_pandas()

        n_frames = len(frames)
        for i in range(0, n_frames, max(1, stride)):
            f_start = frames[i]
            f_end = f_start + lag_frames

            if f_end in frame_to_idx:
                idx_end = frame_to_idx[f_end]
                dx_um = (xs[idx_end] - xs[i]) * scale
                dy_um = (ys[idx_end] - ys[i]) * scale
                dr_um = np.hypot(dx_um, dy_um)

                # 区間 [f_start, f_end] 内のポーラーオーダー平均値
                sub_polar = particle_polar.loc[f_start:f_end]
                valid_vals = sub_polar.dropna().values
                if len(valid_vals) > 0:
                    mean_p = float(np.mean(valid_vals))
                    records.append({
                        'exp_name': exp_name,
                        'particle': pid_int,
                        'frame_start': f_start,
                        'frame_end': f_end,
                        'lag_time_s': actual_lag_s,
                        'polar_order': mean_p,
                        'displacement_um': dr_um,
                        'dx_um': dx_um,
                        'dy_um': dy_um,
                        'target_region_radius_um': target_radius_um,
                        'actual_window_um': actual_w_um,
                        'actual_window_px': actual_w_px,
                    })

    return pd.DataFrame(records)


def compute_binned_stats(
    df: pd.DataFrame,
    n_bins: int = 8,
    min_samples: int = 3
) -> pd.DataFrame:
    """
    ポーラーオーダーをビン分割し、各ビンにおける変位 |Delta r| の
    平均値, SEM, 中央値, IQR (25%, 75%), サンプル数を算出する。
    """
    if df.empty or 'polar_order' not in df or 'displacement_um' not in df:
        return pd.DataFrame()

    valid_df = df.dropna(subset=['polar_order', 'displacement_um']).copy()
    valid_df = valid_df[(valid_df['polar_order'] >= 0.0) & (valid_df['polar_order'] <= 1.0)]

    if len(valid_df) < min_samples:
        return pd.DataFrame()

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    valid_df['bin_idx'] = pd.cut(valid_df['polar_order'], bins=bins, include_lowest=True, labels=False)

    rows = []
    for b_i in range(n_bins):
        bin_low = bins[b_i]
        bin_high = bins[b_i + 1]
        bin_center = (bin_low + bin_high) / 2.0
        sub = valid_df[valid_df['bin_idx'] == b_i]
        n_count = len(sub)

        if n_count >= min_samples:
            p_mean = float(sub['polar_order'].mean())
            dr_vals = sub['displacement_um'].to_numpy(dtype=float)
            dr_mean = float(np.mean(dr_vals))
            dr_std = float(np.std(dr_vals, ddof=1)) if n_count > 1 else 0.0
            dr_sem = float(dr_std / np.sqrt(n_count))
            dr_median = float(np.median(dr_vals))
            dr_q25 = float(np.percentile(dr_vals, 25))
            dr_q75 = float(np.percentile(dr_vals, 75))
        else:
            p_mean = bin_center
            dr_mean = np.nan
            dr_std = np.nan
            dr_sem = np.nan
            dr_median = np.nan
            dr_q25 = np.nan
            dr_q75 = np.nan

        rows.append({
            'bin_idx': b_i,
            'bin_low': bin_low,
            'bin_high': bin_high,
            'bin_center': bin_center,
            'polar_order_mean': p_mean,
            'displacement_mean_um': dr_mean,
            'displacement_sem_um': dr_sem,
            'displacement_std_um': dr_std,
            'displacement_median_um': dr_median,
            'displacement_q25_um': dr_q25,
            'displacement_q75_um': dr_q75,
            'n_samples': n_count,
        })

    return pd.DataFrame(rows)


def compute_binned_stats_by_displacement(
    df: pd.DataFrame,
    n_bins: int = 8,
    min_samples: int = 3
) -> pd.DataFrame:
    """
    変位 |Delta r| (エリート度) をビン分割し、各ビンにおける
    貨物真下のポーラーオーダー Phi の平均値, SEM, 中央値, IQR (25%, 75%), サンプル数を算出する。
    """
    if df.empty or 'polar_order' not in df or 'displacement_um' not in df:
        return pd.DataFrame()

    valid_df = df.dropna(subset=['polar_order', 'displacement_um']).copy()
    valid_df = valid_df[(valid_df['displacement_um'] >= 0.0) & (valid_df['polar_order'] >= 0.0) & (valid_df['polar_order'] <= 1.0)]

    if len(valid_df) < min_samples:
        return pd.DataFrame()

    dr_max = float(valid_df['displacement_um'].quantile(0.99))
    if dr_max <= 0:
        dr_max = float(valid_df['displacement_um'].max())
    if dr_max <= 0:
        return pd.DataFrame()

    bins = np.linspace(0.0, dr_max, n_bins + 1)
    valid_df['bin_idx'] = pd.cut(valid_df['displacement_um'], bins=bins, include_lowest=True, labels=False)

    rows = []
    for b_i in range(n_bins):
        bin_low = bins[b_i]
        bin_high = bins[b_i + 1]
        bin_center = (bin_low + bin_high) / 2.0
        sub = valid_df[valid_df['bin_idx'] == b_i]
        n_count = len(sub)

        if n_count >= min_samples:
            dr_mean = float(sub['displacement_um'].mean())
            p_vals = sub['polar_order'].to_numpy(dtype=float)
            p_mean = float(np.mean(p_vals))
            p_std = float(np.std(p_vals, ddof=1)) if n_count > 1 else 0.0
            p_sem = float(p_std / np.sqrt(n_count))
            p_median = float(np.median(p_vals))
            p_q25 = float(np.percentile(p_vals, 25))
            p_q75 = float(np.percentile(p_vals, 75))
        else:
            dr_mean = bin_center
            p_mean = np.nan
            p_std = np.nan
            p_sem = np.nan
            p_median = np.nan
            p_q25 = np.nan
            p_q75 = np.nan

        rows.append({
            'bin_idx': b_i,
            'bin_low_dr_um': bin_low,
            'bin_high_dr_um': bin_high,
            'bin_center_dr_um': bin_center,
            'displacement_mean_um': dr_mean,
            'polar_order_mean': p_mean,
            'polar_order_sem': p_sem,
            'polar_order_std': p_std,
            'polar_order_median': p_median,
            'polar_order_q25': p_q25,
            'polar_order_q75': p_q75,
            'n_samples': n_count,
        })

    return pd.DataFrame(rows)


def save_figure_to_all(fig: plt.Figure, basename: str, out_dirs: List[Path], dpi: int = 300):
    """Figure を指定された全ディレクトリに png と svg で保存"""
    for d in out_dirs:
        try:
            d.mkdir(parents=True, exist_ok=True)
            png_p = d / f"{basename}.png"
            svg_p = d / f"{basename}.svg"
            fig.savefig(png_p, dpi=dpi, bbox_inches='tight')
            fig.savefig(svg_p, bbox_inches='tight')
            print(f"  Saved figure: {basename}.png / .svg -> {d}")
        except Exception as e:
            print(f"  Warning: Failed to save figure {basename} to {d}: {e}")


def save_csv_to_all(df: pd.DataFrame, filename: str, out_dirs: List[Path]):
    """DataFrame を指定された全ディレクトリに CSV として保存"""
    for d in out_dirs:
        try:
            d.mkdir(parents=True, exist_ok=True)
            csv_p = d / filename
            df.to_csv(csv_p, index=False)
            print(f"  Saved CSV: {filename} -> {d}")
        except Exception as e:
            print(f"  Warning: Failed to save CSV {filename} to {d}: {e}")


def plot_all_beads_overlay(
    all_points: Dict[str, pd.DataFrame],
    all_binned: Dict[str, pd.DataFrame],
    lag_time_s: float,
    beads_info: List[dict] = BEADS_INFO,
    out_dirs: Optional[List[Path]] = None,
    alpha_points: float = 0.25,
    point_size: float = 24.0,
):
    """
    全ビーズサイズを同一図上に重ねて描画する散布図。
    - 各個別データ点の散布図（淡色プロット）
    - ビーズごとのビン平均値 +- SEM を実線＋大型マーカーで重ね描き
    """
    fig, ax = plt.subplots(figsize=(8.8, 6.6))

    for item in beads_info:
        b_name = item['name']
        d_um = item['diameter_um']
        marker = item['marker']
        color = item['color']
        lbl = item['label']

        df_pts = all_points.get(b_name, pd.DataFrame())
        df_bin = all_binned.get(b_name, pd.DataFrame())

        if df_pts.empty:
            continue

        n_pts = len(df_pts)
        n_particles = df_pts['particle'].nunique() if 'particle' in df_pts else 0

        # 1. 散布点 (個別データ点)
        ax.scatter(
            df_pts['polar_order'],
            df_pts['displacement_um'],
            color=color,
            marker=marker,
            s=point_size,
            alpha=alpha_points,
            edgecolors='none',
            zorder=2
        )

        # 2. ビニング平均曲線 +- SEM
        if not df_bin.empty:
            valid_bin = df_bin.dropna(subset=['displacement_mean_um'])
            if len(valid_bin) >= 2:
                bx = valid_bin['polar_order_mean']
                by = valid_bin['displacement_mean_um']
                by_err = valid_bin['displacement_sem_um']
                ax.errorbar(
                    bx, by, yerr=by_err,
                    color=color,
                    marker=marker,
                    markersize=9.0,
                    markeredgewidth=1.8,
                    markeredgecolor='#111111',
                    linestyle='-',
                    linewidth=2.8,
                    capsize=3.5,
                    capthick=1.5,
                    label=f"{lbl} ($N={n_pts}$)",
                    zorder=5,
                )
            else:
                # ビンが少ない場合も凡例用にプロット
                ax.plot([], [], color=color, marker=marker, markersize=9.0, markeredgewidth=1.8,
                        markeredgecolor='#111111', linestyle='-', linewidth=2.8,
                        label=f"{lbl} ($N={n_pts}$)")
        else:
            ax.plot([], [], color=color, marker=marker, markersize=9.0, markeredgewidth=1.8,
                    markeredgecolor='#111111', linestyle='-', linewidth=2.8,
                    label=f"{lbl} ($N={n_pts}$)")

    ax.set_xlabel(r"Cargo Vicinity Polar Order $\Phi$", fontsize=18)
    ax.set_ylabel(r"Cargo Displacement $|\Delta \boldsymbol{r}|$ [$\mu\mathrm{m}$]", fontsize=18)
    ax.set_title(f"Polar Order vs Displacement ($\Delta t = {lag_time_s:.0f}\\,\\mathrm{{s}}$)", fontsize=18, fontweight='bold')
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(bottom=0.0)

    # 凡例
    ax.legend(
        fontsize=11.5,
        loc='upper left',
        framealpha=0.92,
        markerscale=1.2,
        handlelength=1.5,
        borderpad=0.4,
        labelspacing=0.3
    )

    fig.tight_layout()
    if out_dirs:
        save_figure_to_all(fig, f"polar_vs_displacement_all_beads_lag{lag_time_s:.0f}s", out_dirs)
    plt.close(fig)


def plot_6panels_grid(
    all_points: Dict[str, pd.DataFrame],
    all_binned: Dict[str, pd.DataFrame],
    lag_time_s: float,
    beads_info: List[dict] = BEADS_INFO,
    out_dirs: Optional[List[Path]] = None,
    alpha_points: float = 0.35,
    point_size: float = 28.0,
):
    """
    全 6 つのビーズサイズを 2行 3列 のグリッドに並べた個別散布図パネル。
    """
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 9.5), sharex=True, sharey=False)
    axes_flat = axes.flatten()

    for idx, item in enumerate(beads_info):
        ax = axes_flat[idx]
        b_name = item['name']
        d_um = item['diameter_um']
        marker = item['marker']
        color = item['color']
        lbl = item['label']

        df_pts = all_points.get(b_name, pd.DataFrame())
        df_bin = all_binned.get(b_name, pd.DataFrame())

        if df_pts.empty:
            ax.text(0.5, 0.5, "No Data", ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title(f"{lbl}", fontsize=14, fontweight='bold')
            continue

        n_pts = len(df_pts)
        r_region = df_pts['target_region_radius_um'].iloc[0] if 'target_region_radius_um' in df_pts else d_um

        # 散布図プロット
        ax.scatter(
            df_pts['polar_order'],
            df_pts['displacement_um'],
            color=color,
            marker=marker,
            s=point_size,
            alpha=alpha_points,
            edgecolors='none',
            label="Points",
            zorder=2
        )

        # ビニング平均 +- SEM
        if not df_bin.empty:
            valid_bin = df_bin.dropna(subset=['displacement_mean_um'])
            if len(valid_bin) >= 2:
                bx = valid_bin['polar_order_mean']
                by = valid_bin['displacement_mean_um']
                by_err = valid_bin['displacement_sem_um']
                ax.errorbar(
                    bx, by, yerr=by_err,
                    color='#222222',
                    marker=marker,
                    markersize=8.0,
                    markeredgewidth=1.5,
                    markeredgecolor=color,
                    markerfacecolor=color,
                    linestyle='-',
                    linewidth=2.5,
                    capsize=3.0,
                    capthick=1.2,
                    label="Binned Mean $\\pm$ SEM",
                    zorder=5,
                )

        # 相関係数の算出 (Pearson r & Spearman rho)
        r_corr = np.corrcoef(df_pts['polar_order'], df_pts['displacement_um'])[0, 1] if len(df_pts) > 2 else np.nan
        info_str = f"$N = {n_pts}$\n$R_{{\\mathrm{{region}}}} = {r_region:.2f}\\,\\mu\\mathrm{{m}}$\n$r = {r_corr:+.2f}$"
        ax.text(
            0.05, 0.95, info_str,
            transform=ax.transAxes,
            fontsize=11.5,
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.35', facecolor='white', alpha=0.88, edgecolor='#cccccc')
        )

        ax.set_title(f"{lbl} ($2R_c = {d_um:.2f}\\,\\mu\\mathrm{{m}}$)", fontsize=15, fontweight='bold')
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(bottom=0.0)

        if idx >= 3:
            ax.set_xlabel(r"Cargo Vicinity Polar Order $\Phi$", fontsize=14)
        if idx % 3 == 0:
            ax.set_ylabel(r"Displacement $|\Delta \boldsymbol{r}|$ [$\mu\mathrm{m}$]", fontsize=14)

    fig.suptitle(f"Cargo Vicinity Polar Order vs Displacement ($\Delta t = {lag_time_s:.0f}\\,\\mathrm{{s}}$)", fontsize=18, fontweight='bold', y=0.995)
    fig.tight_layout()

    if out_dirs:
        save_figure_to_all(fig, f"polar_vs_displacement_6panels_lag{lag_time_s:.0f}s", out_dirs)
    plt.close(fig)


def plot_multilag_comparison(
    multilag_data: Dict[float, Dict[str, pd.DataFrame]],
    beads_info: List[dict] = BEADS_INFO,
    out_dirs: Optional[List[Path]] = None,
):
    """
    複数の lag time におけるビニング平均曲線 (Displacement vs Polar Order) の比較図。
    """
    lag_list = sorted(list(multilag_data.keys()))
    if len(lag_list) <= 1:
        return

    n_lags = len(lag_list)
    fig, axes = plt.subplots(1, n_lags, figsize=(5.0 * n_lags, 5.2), sharex=True, sharey=False)
    if n_lags == 1:
        axes = [axes]

    for lag_idx, lag_s in enumerate(lag_list):
        ax = axes[lag_idx]
        binned_dict = multilag_data[lag_s]

        for item in beads_info:
            b_name = item['name']
            marker = item['marker']
            color = item['color']
            lbl = item['label']

            df_bin = binned_dict.get(b_name, pd.DataFrame())
            if df_bin.empty:
                continue

            valid_bin = df_bin.dropna(subset=['displacement_mean_um'])
            if len(valid_bin) >= 2:
                bx = valid_bin['polar_order_mean']
                by = valid_bin['displacement_mean_um']
                by_err = valid_bin['displacement_sem_um']
                ax.errorbar(
                    bx, by, yerr=by_err,
                    color=color,
                    marker=marker,
                    markersize=7.0,
                    markeredgewidth=1.2,
                    linestyle='-',
                    linewidth=2.0,
                    capsize=2.5,
                    label=lbl,
                )

        ax.set_title(f"$\Delta t = {lag_s:.0f}\\,\\mathrm{{s}}$", fontsize=15, fontweight='bold')
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(bottom=0.0)
        ax.set_xlabel(r"Polar Order $\Phi$", fontsize=14)
        if lag_idx == 0:
            ax.set_ylabel(r"Mean $|\Delta \boldsymbol{r}|$ [$\mu\mathrm{m}$]", fontsize=14)
            ax.legend(fontsize=10.0, loc='upper left', framealpha=0.88)

    fig.suptitle("Displacement vs Cargo Vicinity Polar Order across Lag Times", fontsize=17, fontweight='bold', y=1.02)
    fig.tight_layout()

    if out_dirs:
        save_figure_to_all(fig, "polar_vs_displacement_multilag_comparison", out_dirs)
    plt.close(fig)


def plot_all_beads_overlay_disp_as_x(
    all_points: Dict[str, pd.DataFrame],
    all_binned_disp: Dict[str, pd.DataFrame],
    lag_time_s: float,
    beads_info: List[dict] = BEADS_INFO,
    out_dirs: Optional[List[Path]] = None,
    alpha_points: float = 0.25,
    point_size: float = 24.0,
):
    """
    横軸を変位 |Delta r| (エリート度)、縦軸を貨物真下のポーラーオーダー Phi とした全ビーズ重ね描き散布図。
    - 各個別データ点の散布図（淡色プロット）
    - ビーズごとの変位ビン平均値 +- SEM を実線＋大型マーカーで重ね描き
    """
    fig, ax = plt.subplots(figsize=(8.8, 6.6))

    for item in beads_info:
        b_name = item['name']
        d_um = item['diameter_um']
        marker = item['marker']
        color = item['color']
        lbl = item['label']

        df_pts = all_points.get(b_name, pd.DataFrame())
        df_bin = all_binned_disp.get(b_name, pd.DataFrame())

        if df_pts.empty:
            continue

        n_pts = len(df_pts)

        # 1. 散布点 (個別データ点)
        ax.scatter(
            df_pts['displacement_um'],
            df_pts['polar_order'],
            color=color,
            marker=marker,
            s=point_size,
            alpha=alpha_points,
            edgecolors='none',
            zorder=2
        )

        # 2. 変位ビン平均曲線 +- SEM
        if not df_bin.empty:
            valid_bin = df_bin.dropna(subset=['polar_order_mean'])
            if len(valid_bin) >= 2:
                bx = valid_bin['displacement_mean_um']
                by = valid_bin['polar_order_mean']
                by_err = valid_bin['polar_order_sem']
                ax.errorbar(
                    bx, by, yerr=by_err,
                    color=color,
                    marker=marker,
                    markersize=9.0,
                    markeredgewidth=1.8,
                    markeredgecolor='#111111',
                    linestyle='-',
                    linewidth=2.8,
                    capsize=3.5,
                    capthick=1.5,
                    label=f"{lbl} ($N={n_pts}$)",
                    zorder=5,
                )
            else:
                ax.plot([], [], color=color, marker=marker, markersize=9.0, markeredgewidth=1.8,
                        markeredgecolor='#111111', linestyle='-', linewidth=2.8,
                        label=f"{lbl} ($N={n_pts}$)")
        else:
            ax.plot([], [], color=color, marker=marker, markersize=9.0, markeredgewidth=1.8,
                    markeredgecolor='#111111', linestyle='-', linewidth=2.8,
                    label=f"{lbl} ($N={n_pts}$)")

    ax.set_xlabel(r"Cargo Displacement $|\Delta \boldsymbol{r}|$ [$\mu\mathrm{m}$] (Elite Degree)", fontsize=18)
    ax.set_ylabel(r"Polar Order $\Phi$ under Cargo", fontsize=18)
    ax.set_title(f"Displacement vs Polar Order ($\Delta t = {lag_time_s:.0f}\\,\\mathrm{{s}}$)", fontsize=18, fontweight='bold')
    ax.set_xlim(left=0.0)
    ax.set_ylim(0.0, 1.0)

    # 凡例
    ax.legend(
        fontsize=11.5,
        loc='upper right',
        framealpha=0.92,
        markerscale=1.2,
        handlelength=1.5,
        borderpad=0.4,
        labelspacing=0.3
    )

    fig.tight_layout()
    if out_dirs:
        save_figure_to_all(fig, f"displacement_vs_polar_all_beads_lag{lag_time_s:.0f}s", out_dirs)
    plt.close(fig)


def plot_6panels_grid_disp_as_x(
    all_points: Dict[str, pd.DataFrame],
    all_binned_disp: Dict[str, pd.DataFrame],
    lag_time_s: float,
    beads_info: List[dict] = BEADS_INFO,
    out_dirs: Optional[List[Path]] = None,
    alpha_points: float = 0.35,
    point_size: float = 28.0,
):
    """
    全 6 つのビーズサイズを 2行 3列 のグリッドに並べた個別散布図パネル。
    横軸: 変位 |Delta r| (エリート度)、縦軸: 貨物真下のポーラーオーダー Phi。
    """
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 9.5), sharex=False, sharey=True)
    axes_flat = axes.flatten()

    for idx, item in enumerate(beads_info):
        ax = axes_flat[idx]
        b_name = item['name']
        d_um = item['diameter_um']
        marker = item['marker']
        color = item['color']
        lbl = item['label']

        df_pts = all_points.get(b_name, pd.DataFrame())
        df_bin = all_binned_disp.get(b_name, pd.DataFrame())

        if df_pts.empty:
            ax.text(0.5, 0.5, "No Data", ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title(f"{lbl}", fontsize=14, fontweight='bold')
            continue

        n_pts = len(df_pts)
        r_region = df_pts['target_region_radius_um'].iloc[0] if 'target_region_radius_um' in df_pts else d_um

        # 散布図プロット
        ax.scatter(
            df_pts['displacement_um'],
            df_pts['polar_order'],
            color=color,
            marker=marker,
            s=point_size,
            alpha=alpha_points,
            edgecolors='none',
            label="Points",
            zorder=2
        )

        # 変位ビニング平均 +- SEM
        if not df_bin.empty:
            valid_bin = df_bin.dropna(subset=['polar_order_mean'])
            if len(valid_bin) >= 2:
                bx = valid_bin['displacement_mean_um']
                by = valid_bin['polar_order_mean']
                by_err = valid_bin['polar_order_sem']
                ax.errorbar(
                    bx, by, yerr=by_err,
                    color='#222222',
                    marker=marker,
                    markersize=8.0,
                    markeredgewidth=1.5,
                    markeredgecolor=color,
                    markerfacecolor=color,
                    linestyle='-',
                    linewidth=2.5,
                    capsize=3.0,
                    capthick=1.2,
                    label="Binned Mean $\\pm$ SEM",
                    zorder=5,
                )

        # 相関係数の算出 (Pearson r)
        r_corr = np.corrcoef(df_pts['displacement_um'], df_pts['polar_order'])[0, 1] if len(df_pts) > 2 else np.nan
        info_str = f"$N = {n_pts}$\n$R_{{\\mathrm{{region}}}} = {r_region:.2f}\\,\\mu\\mathrm{{m}}$\n$r = {r_corr:+.2f}$"
        ax.text(
            0.05, 0.95, info_str,
            transform=ax.transAxes,
            fontsize=11.5,
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.35', facecolor='white', alpha=0.88, edgecolor='#cccccc')
        )

        ax.set_title(f"{lbl} ($2R_c = {d_um:.2f}\\,\\mu\\mathrm{{m}}$)", fontsize=15, fontweight='bold')
        ax.set_ylim(0.0, 1.0)
        ax.set_xlim(left=0.0)

        if idx >= 3:
            ax.set_xlabel(r"Displacement $|\Delta \boldsymbol{r}|$ [$\mu\mathrm{m}$]", fontsize=14)
        if idx % 3 == 0:
            ax.set_ylabel(r"Polar Order $\Phi$ under Cargo", fontsize=14)

    fig.suptitle(f"Cargo Displacement vs Vicinity Polar Order ($\Delta t = {lag_time_s:.0f}\\,\\mathrm{{s}}$)", fontsize=18, fontweight='bold', y=0.995)
    fig.tight_layout()

    if out_dirs:
        save_figure_to_all(fig, f"displacement_vs_polar_6panels_lag{lag_time_s:.0f}s", out_dirs)
    plt.close(fig)


def plot_multilag_comparison_disp_as_x(
    multilag_data_disp: Dict[float, Dict[str, pd.DataFrame]],
    beads_info: List[dict] = BEADS_INFO,
    out_dirs: Optional[List[Path]] = None,
):
    """
    複数の lag time におけるビニング平均曲線 (Polar Order vs Displacement: 横軸 |Delta r|) の比較図。
    """
    lag_list = sorted(list(multilag_data_disp.keys()))
    if len(lag_list) <= 1:
        return

    n_lags = len(lag_list)
    fig, axes = plt.subplots(1, n_lags, figsize=(5.0 * n_lags, 5.2), sharex=False, sharey=True)
    if n_lags == 1:
        axes = [axes]

    for lag_idx, lag_s in enumerate(lag_list):
        ax = axes[lag_idx]
        binned_dict = multilag_data_disp[lag_s]

        for item in beads_info:
            b_name = item['name']
            marker = item['marker']
            color = item['color']
            lbl = item['label']

            df_bin = binned_dict.get(b_name, pd.DataFrame())
            if df_bin.empty:
                continue

            valid_bin = df_bin.dropna(subset=['polar_order_mean'])
            if len(valid_bin) >= 2:
                bx = valid_bin['displacement_mean_um']
                by = valid_bin['polar_order_mean']
                by_err = valid_bin['polar_order_sem']
                ax.errorbar(
                    bx, by, yerr=by_err,
                    color=color,
                    marker=marker,
                    markersize=7.0,
                    markeredgewidth=1.2,
                    linestyle='-',
                    linewidth=2.0,
                    capsize=2.5,
                    label=lbl,
                )

        ax.set_title(f"$\Delta t = {lag_s:.0f}\\,\\mathrm{{s}}$", fontsize=15, fontweight='bold')
        ax.set_ylim(0.0, 1.0)
        ax.set_xlim(left=0.0)
        ax.set_xlabel(r"$|\Delta \boldsymbol{r}|$ [$\mu\mathrm{m}$]", fontsize=14)
        if lag_idx == 0:
            ax.set_ylabel(r"Polar Order $\Phi$ under Cargo", fontsize=14)
            ax.legend(fontsize=10.0, loc='upper right', framealpha=0.88)

    fig.suptitle("Cargo Vicinity Polar Order vs Displacement across Lag Times", fontsize=17, fontweight='bold', y=1.02)
    fig.tight_layout()

    if out_dirs:
        save_figure_to_all(fig, "displacement_vs_polar_multilag_comparison", out_dirs)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot scatter plot of cargo displacement |Delta r| vs cargo vicinity polar order."
    )
    parser.add_argument(
        '--root_dir', type=str, default=None,
        help="Root directory containing beads data. Default: auto-detect"
    )
    parser.add_argument(
        '--lag_times', type=float, nargs='+', default=[40.0],
        help="List of lag times [s] to evaluate (e.g. --lag_times 4 12 40 100 300). Default: [40.0]"
    )
    parser.add_argument(
        '--alpha_threshold', type=float, default=0.5,
        help="MSD anomalous diffusion exponent threshold for active particles (default: 0.5, set <=0 to disable)."
    )
    parser.add_argument(
        '--stride', type=int, default=1,
        help="Frame stride for displacement sampling (default: 1)."
    )
    parser.add_argument(
        '--region_mode', type=str, choices=['diameter', 'radius'], default='diameter',
        help="Region size definition for cargo vicinity: 'diameter' (max(diameter, 3.37um)) or 'radius' (max(radius, 3.37um)). Default: 'diameter'"
    )
    parser.add_argument(
        '--min_region_um', type=float, default=3.37,
        help="Lower bound for vicinity region radius [um] (default: 3.37 um for small beads 0.63, 1.18 um)."
    )
    parser.add_argument(
        '--n_bins', type=int, default=8,
        help="Number of bins for polar order in binned statistics (default: 8)."
    )
    parser.add_argument(
        '--beads', type=str, nargs='+', default=['all'],
        help="Target beads to analyze (e.g. all, beads06um beads1um beads3um). Default: all"
    )
    args = parser.parse_args()

    # ルートディレクトリの解決
    root_dir = Path(args.root_dir) if args.root_dir else find_default_root()
    print(f"Data Root Directory: {root_dir}")

    # 出力先ディレクトリの設定
    workspace_dir = CURRENT_DIR
    out_dirs = [
        workspace_dir / "figure" / "polar_vs_displacement",
        root_dir / "figure" / "polar_vs_displacement",
    ]
    for d in out_dirs:
        try:
            d.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    # 対象ビーズの選定
    target_names = [b.lower() for b in args.beads]
    if 'all' in target_names:
        selected_beads = BEADS_INFO
    else:
        selected_beads = [b for b in BEADS_INFO if b['name'].lower() in target_names or b['diameter_um'] in args.beads]

    print(f"\nTarget beads: {[b['name'] for b in selected_beads]}")
    print(f"Lag times [s]: {args.lag_times}")
    print(f"Region mode: {args.region_mode} (min_region: {args.min_region_um} um)")
    print(f"Alpha threshold: {args.alpha_threshold}")

    # 実験ディレクトリの収集
    exp_dirs_dict = {}
    for item in selected_beads:
        b_name = item['name']
        edirs = find_experiment_dirs(root_dir, b_name)
        exp_dirs_dict[b_name] = edirs
        print(f"  {b_name}: found {len(edirs)} experiment directories with local_polar_w.zarr")

    # アクティブ粒子のフィルタリング (alpha > threshold)
    active_particles_dict = {}
    if args.alpha_threshold > 0:
        print("\n--- Filtering active particles (alpha > threshold) ---")
        for item in selected_beads:
            b_name = item['name']
            edirs = exp_dirs_dict[b_name]
            act_map = filter_active_particles(edirs, interval=4.0, scale=0.11, alpha_threshold=args.alpha_threshold)
            active_particles_dict[b_name] = {k for k, v in act_map.items() if v}
            n_tot = len(act_map)
            n_act = len(active_particles_dict[b_name])
            print(f"  {b_name}: {n_act}/{n_tot} active particles (alpha > {args.alpha_threshold})")
    else:
        print("\n--- Particle filtering disabled (using all tracked particles) ---")
        for item in selected_beads:
            active_particles_dict[item['name']] = None

    # 各 lag time ごとのデータ抽出とプロット
    multilag_binned_data = {}
    multilag_binned_data_disp = {}

    for lag_s in args.lag_times:
        print(f"\n=======================================================")
        print(f"Processing lag time: {lag_s:.1f} s")
        print(f"=======================================================")

        all_points_dict = {}
        all_binned_dict = {}
        all_binned_disp_dict = {}
        summary_rows = []
        summary_disp_rows = []
        all_pts_rows = []

        for item in selected_beads:
            b_name = item['name']
            d_um = item['diameter_um']
            edirs = exp_dirs_dict[b_name]
            r_target = get_region_radius_um(item, region_mode=args.region_mode, min_region_um=args.min_region_um)

            dfs = []
            for edir in edirs:
                df_sub = extract_polar_and_displacement_for_exp(
                    exp_dir=edir,
                    target_radius_um=r_target,
                    lag_time_s=lag_s,
                    frame_interval=4.0,
                    scale=0.11,
                    stride=args.stride,
                    active_particles_set=active_particles_dict[b_name],
                )
                if not df_sub.empty:
                    df_sub['bead_name'] = b_name
                    df_sub['diameter_um'] = d_um
                    dfs.append(df_sub)

            if dfs:
                df_pts = pd.concat(dfs, ignore_index=True)
                all_points_dict[b_name] = df_pts
                all_pts_rows.append(df_pts)

                # 1. ビニング統計量: 横軸 Phi, 縦軸 |Delta r|
                df_bin = compute_binned_stats(df_pts, n_bins=args.n_bins, min_samples=3)
                all_binned_dict[b_name] = df_bin

                if not df_bin.empty:
                    df_bin_copy = df_bin.copy()
                    df_bin_copy['bead_name'] = b_name
                    df_bin_copy['diameter_um'] = d_um
                    df_bin_copy['lag_time_s'] = lag_s
                    summary_rows.append(df_bin_copy)

                # 2. ビニング統計量: 横軸 |Delta r| (エリート度), 縦軸 Phi
                df_bin_disp = compute_binned_stats_by_displacement(df_pts, n_bins=args.n_bins, min_samples=3)
                all_binned_disp_dict[b_name] = df_bin_disp

                if not df_bin_disp.empty:
                    df_bin_disp_copy = df_bin_disp.copy()
                    df_bin_disp_copy['bead_name'] = b_name
                    df_bin_disp_copy['diameter_um'] = d_um
                    df_bin_disp_copy['lag_time_s'] = lag_s
                    summary_disp_rows.append(df_bin_disp_copy)

                n_pts = len(df_pts)
                print(f"  {b_name} ({d_um} um): extracted {n_pts} points (region radius = {r_target:.2f} um)")
            else:
                all_points_dict[b_name] = pd.DataFrame()
                all_binned_dict[b_name] = pd.DataFrame()
                all_binned_disp_dict[b_name] = pd.DataFrame()
                print(f"  {b_name} ({d_um} um): no valid points extracted")

        multilag_binned_data[lag_s] = all_binned_dict
        multilag_binned_data_disp[lag_s] = all_binned_disp_dict

        # CSV 出力
        if summary_rows:
            df_summary = pd.concat(summary_rows, ignore_index=True)
            save_csv_to_all(df_summary, f"polar_vs_displacement_binned_summary_lag{lag_s:.0f}s.csv", out_dirs)

        if summary_disp_rows:
            df_summary_disp = pd.concat(summary_disp_rows, ignore_index=True)
            save_csv_to_all(df_summary_disp, f"displacement_vs_polar_binned_summary_lag{lag_s:.0f}s.csv", out_dirs)

        if all_pts_rows:
            df_all_pts = pd.concat(all_pts_rows, ignore_index=True)
            save_csv_to_all(df_all_pts, f"polar_vs_displacement_raw_points_lag{lag_s:.0f}s.csv", out_dirs)

        # プロット生成
        print(f"\n--- Generating plots for lag time {lag_s:.0f} s ---")
        
        # A. 横軸: ポーラーオーダー Phi, 縦軸: 変位 |Delta r|
        plot_all_beads_overlay(
            all_points=all_points_dict,
            all_binned=all_binned_dict,
            lag_time_s=lag_s,
            beads_info=selected_beads,
            out_dirs=out_dirs,
        )

        plot_6panels_grid(
            all_points=all_points_dict,
            all_binned=all_binned_dict,
            lag_time_s=lag_s,
            beads_info=selected_beads,
            out_dirs=out_dirs,
        )

        # B. 横軸: 変位 |Delta r| (エリート度), 縦軸: 貨物真下のポーラーオーダー Phi
        plot_all_beads_overlay_disp_as_x(
            all_points=all_points_dict,
            all_binned_disp=all_binned_disp_dict,
            lag_time_s=lag_s,
            beads_info=selected_beads,
            out_dirs=out_dirs,
        )

        plot_6panels_grid_disp_as_x(
            all_points=all_points_dict,
            all_binned_disp=all_binned_disp_dict,
            lag_time_s=lag_s,
            beads_info=selected_beads,
            out_dirs=out_dirs,
        )

    # 複数 lag time の比較図
    if len(args.lag_times) > 1:
        print("\n--- Generating multi-lag comparison plots ---")
        # A. 横軸 Phi, 縦軸 |Delta r|
        plot_multilag_comparison(
            multilag_data=multilag_binned_data,
            beads_info=selected_beads,
            out_dirs=out_dirs,
        )
        # B. 横軸 |Delta r|, 縦軸 Phi
        plot_multilag_comparison_disp_as_x(
            multilag_data_disp=multilag_binned_data_disp,
            beads_info=selected_beads,
            out_dirs=out_dirs,
        )

    print("\n[Done] All analyses and plots completed successfully.")


if __name__ == '__main__':
    main()
