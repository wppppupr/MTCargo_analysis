#!/usr/bin/env python3
"""
plot_cos_vs_radius.py (also plot_chi_vs_radius.py)

貨物粒子の局所微小管重心流速への方向追従度（コサイン類似度 cos(theta)）の定量化・描画スクリプト。

【理論的背景】
粒子直下の窓領域 A_c における微小管の重心流速ベクトル（面積平均ベクトル）:
    v_bar_MT(R) = (1 / A_c) * \int_{A_c} v_MT(r) d^2r
このベクトルの大きさは局所平均スピード x ポーラーオーダーそのものに一致する:
    |v_bar_MT(R)| = <|v_MT|> * Phi(R)

微小管重心流速に対する貨物速度 v_cargo の方向一致度（Directional Alignment / Cosine Similarity）:
    cos \theta \equiv \frac{v_cargo \cdot v_bar_MT(R)}{|v_cargo| |v_bar_MT(R)|}

本スクリプトは、cos(theta) をメイン指標として:
1. 全ビーズ重ね描き図 (散布図: 対数軸 & 線形軸)
2. 各ビーズ径ごとの 6 パネル個別図 (6-panels grid)
3. 各ビーズ条件単独の個別図 (Individual single-panel plots)
4. 窓サイズ依存性 cos(theta)(R) vs R の 6 パネル図
5. バイオリン＋ボックスプロットによる分布図
を生成する。
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
                if (r / b).exists() and len(list((r / b).glob('*/*/*beads_tracks.csv'))) > 0:
                    return r
                if (r / b).exists() and len(list((r / b).glob('*/*beads_tracks.csv'))) > 0:
                    return r
    for r in POSSIBLE_ROOTS:
        if r.exists():
            return r
    return CURRENT_DIR


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
) -> Dict[Tuple[str, int], bool]:
    """
    各動画の全粒子について iMSD を計算し、alpha > alpha_threshold の粒子を判定する。
    戻り値: (exp_name, particle_id) -> is_active (bool)
    """
    active_map: Dict[Tuple[str, int], bool] = {}
    if alpha_threshold <= 0:
        return active_map

    for exp_dir in exp_dirs:
        csv_p = exp_dir / "beads_tracks.csv"
        if not csv_p.exists():
            continue
        try:
            raw_df = pd.read_csv(csv_p)
            if 'frame' not in raw_df.columns or 'particle' not in raw_df.columns:
                continue
            track = cv.cal(raw_df, scale=scale, frame_interval=interval)
            imsd_df = dpm.imsd(track, scale=scale, time_scale=interval, display=False)
            imsd_df['MSD'] = pd.to_numeric(imsd_df['MSD'], errors='coerce')
            imsd_df['lag time'] = pd.to_numeric(imsd_df['lag time'], errors='coerce')
            imsd_df = imsd_df.dropna(subset=['MSD', 'lag time'])
            for pid, grp in imsd_df.groupby('particle'):
                alp = calc_particle_alpha(grp, 4.0, 300.0)
                is_active = bool(np.isfinite(alp) and alp > alpha_threshold)
                active_map[(exp_dir.name, int(pid))] = is_active
        except Exception as e:
            print(f"Warning: Failed alpha calculation for {csv_p}: {e}")
    return active_map


def find_experiment_dirs(root_dir: Path, bead_name: str) -> List[Path]:
    """beads_tracks.csv と beads_flow_interaction_gaussian.csv が存在する実験ディレクトリを検索"""
    base = Path(root_dir) / bead_name
    if not base.exists():
        return []
    exp_dirs = []
    for p in sorted(base.glob("*/*")):
        if p.is_dir() and (p / "beads_flow_interaction_gaussian.csv").exists():
            exp_dirs.append(p)
    if not exp_dirs:
        for p in sorted(base.glob("*")):
            if p.is_dir() and (p / "beads_flow_interaction_gaussian.csv").exists():
                exp_dirs.append(p)
    return exp_dirs


def get_target_region_um(
    bead_item: dict,
    region_mode: str = 'radius',
    fixed_um: float = 3.37,
    min_region_um: float = 0.0
) -> float:
    """評価領域の目標半径 [um] を返す"""
    if region_mode == 'radius':
        r = float(bead_item['radius_um'])
    elif region_mode == 'diameter':
        r = float(bead_item['diameter_um'])
    elif region_mode == 'fixed':
        return float(fixed_um)
    else:
        r = float(bead_item['radius_um'])
    
    if min_region_um > 0:
        return float(max(r, min_region_um))
    return r


def load_and_compute_data(
    exp_dirs: List[Path],
    bead_item: dict,
    region_mode: str = 'radius',
    fixed_um: float = 3.37,
    min_region_um: float = 0.0,
    scale: float = 0.11,
    active_particles_map: Optional[Dict[Tuple[str, int], bool]] = None,
    alpha_threshold: float = 0.5,
    min_points_per_particle: int = 5,
    v_flow_min_sq: float = 1e-4,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    指定ビーズ条件の全実験ディレクトリから、cos(theta) および chi を計算・集計する。
    """
    target_um = get_target_region_um(bead_item, region_mode, fixed_um, min_region_um)
    target_px = target_um / scale

    instant_records = []
    sigma_records = []

    for exp_dir in exp_dirs:
        exp_name = exp_dir.name
        csv_p = exp_dir / "beads_flow_interaction_gaussian.csv"
        tracks_p = exp_dir / "beads_tracks.csv"

        if not csv_p.exists():
            continue

        try:
            df_flow = pd.read_csv(csv_p)
        except Exception as e:
            print(f"Error reading {csv_p}: {e}")
            continue

        sigmas = np.sort(df_flow['sigma'].unique())
        if len(sigmas) == 0:
            continue

        # 最も近い sigma を選択
        best_sigma_idx = int(np.argmin(np.abs(sigmas - target_px)))
        best_sigma = float(sigmas[best_sigma_idx])
        actual_r_um = best_sigma * scale

        # 貨物自身の速度ノルムを補完計算するために tracks_csv を参照
        cargo_speed_map = {}
        if tracks_p.exists():
            try:
                tdf = pd.read_csv(tracks_p)
                if 'frame' in tdf.columns and 'particle' in tdf.columns:
                    for pid, grp in tdf.groupby('particle'):
                        grp_s = grp.sort_values('frame')
                        fs = grp_s['frame'].values
                        xs = grp_s['x'].values
                        ys = grp_s['y'].values
                        for i in range(len(fs) - 1):
                            if fs[i+1] == fs[i] + 1:
                                dx = xs[i+1] - xs[i]
                                dy = ys[i+1] - ys[i]
                                spd = np.hypot(dx, dy)
                                cargo_speed_map[(int(pid), int(fs[i]))] = spd
            except Exception:
                pass

        # 1. ターゲット sigma のデータ抽出
        sub_best = df_flow[df_flow['sigma'] == best_sigma].copy()

        for _, row in sub_best.iterrows():
            pid = int(row['particle'])
            frm = int(row['frame'])

            # アクティブ粒子フィルタリング
            if alpha_threshold > 0 and active_particles_map is not None:
                if not active_particles_map.get((exp_name, pid), False):
                    continue

            u_mt = float(row.get('mean_flow_u', 0.0))
            v_mt = float(row.get('mean_flow_v', 0.0))
            v_flow_sq = u_mt**2 + v_mt**2
            v_flow_norm = np.sqrt(v_flow_sq)

            dot_prod = float(row.get('dot_product', np.nan))
            cos_val = float(row.get('cos_sim', np.nan))

            # chi = (v_cargo . v_bar_MT) / |v_bar_MT|^2
            if v_flow_sq >= v_flow_min_sq and np.isfinite(dot_prod):
                chi_val = dot_prod / v_flow_sq
            else:
                chi_val = np.nan

            # v_cargo_norm
            v_cargo_norm = cargo_speed_map.get((pid, frm), np.nan)
            # コサインの補正 (未保存・異常値なら内積とノルムから計算)
            if not np.isfinite(cos_val) or cos_val == 0.0:
                if np.isfinite(v_cargo_norm) and v_cargo_norm > 0 and v_flow_norm > 0:
                    cos_val = np.clip(dot_prod / (v_cargo_norm * v_flow_norm), -1.0, 1.0)

            instant_records.append({
                'bead_name': bead_item['name'],
                'diameter_um': bead_item['diameter_um'],
                'radius_um': bead_item['radius_um'],
                'exp_dir': exp_name,
                'particle': pid,
                'frame': frm,
                'sigma_px': best_sigma,
                'target_region_um': target_um,
                'actual_window_um': actual_r_um,
                'mean_flow_u': u_mt,
                'mean_flow_v': v_mt,
                'v_flow_norm': v_flow_norm,
                'v_cargo_norm': v_cargo_norm,
                'dot_product': dot_prod,
                'chi': chi_val,
                'cos_sim': cos_val,
            })

        # 2. 全 sigma の窓サイズ依存性データ (粒子平均)
        for sig in sigmas:
            sub_s = df_flow[df_flow['sigma'] == sig]
            for pid, grp in sub_s.groupby('particle'):
                pid_int = int(pid)
                if alpha_threshold > 0 and active_particles_map is not None:
                    if not active_particles_map.get((exp_name, pid_int), False):
                        continue
                u_arr = grp['mean_flow_u'].to_numpy(dtype=float)
                v_arr = grp['mean_flow_v'].to_numpy(dtype=float)
                v_sq_arr = u_arr**2 + v_arr**2
                dot_arr = grp['dot_product'].to_numpy(dtype=float)
                valid_mask = (v_sq_arr >= v_flow_min_sq) & np.isfinite(dot_arr)
                if np.sum(valid_mask) >= min_points_per_particle:
                    chi_arr = dot_arr[valid_mask] / v_sq_arr[valid_mask]
                    cos_arr = grp['cos_sim'].to_numpy(dtype=float)[valid_mask]
                    sigma_records.append({
                        'bead_name': bead_item['name'],
                        'diameter_um': bead_item['diameter_um'],
                        'radius_um': bead_item['radius_um'],
                        'exp_dir': exp_name,
                        'particle': pid_int,
                        'sigma_px': float(sig),
                        'window_radius_um': float(sig * scale),
                        'chi_mean': float(np.mean(chi_arr)),
                        'chi_median': float(np.median(chi_arr)),
                        'cos_mean': float(np.mean(cos_arr)),
                        'cos_median': float(np.median(cos_arr)),
                        'n_points': int(np.sum(valid_mask))
                    })

    df_instant = pd.DataFrame(instant_records)
    df_all_sigmas = pd.DataFrame(sigma_records)

    # 粒子単位の時間平均・統計量の集約
    particle_records = []
    if not df_instant.empty:
        for (exp_n, pid), grp in df_instant.groupby(['exp_dir', 'particle']):
            valid_cos = grp['cos_sim'].dropna().values
            valid_chi = grp['chi'].dropna().values
            if len(valid_cos) < min_points_per_particle:
                continue

            particle_records.append({
                'bead_name': bead_item['name'],
                'diameter_um': bead_item['diameter_um'],
                'radius_um': bead_item['radius_um'],
                'exp_dir': exp_n,
                'particle': pid,
                'n_frames': len(valid_cos),
                'target_region_um': target_um,
                'actual_window_um': float(grp['actual_window_um'].iloc[0]),
                'cos_mean': float(np.mean(valid_cos)),
                'cos_std': float(np.std(valid_cos, ddof=1)) if len(valid_cos) > 1 else 0.0,
                'cos_sem': float(np.std(valid_cos, ddof=1) / np.sqrt(len(valid_cos))) if len(valid_cos) > 1 else 0.0,
                'cos_median': float(np.median(valid_cos)),
                'cos_q25': float(np.percentile(valid_cos, 25)),
                'cos_q75': float(np.percentile(valid_cos, 75)),
                'chi_mean': float(np.mean(valid_chi)) if len(valid_chi) > 0 else np.nan,
                'chi_std': float(np.std(valid_chi, ddof=1)) if len(valid_chi) > 1 else 0.0,
                'chi_sem': float(np.std(valid_chi, ddof=1) / np.sqrt(len(valid_chi))) if len(valid_chi) > 1 else 0.0,
                'chi_median': float(np.median(valid_chi)) if len(valid_chi) > 0 else np.nan,
                'v_flow_mean': float(grp['v_flow_norm'].mean()),
            })

    df_particles = pd.DataFrame(particle_records)
    return df_instant, df_particles, df_all_sigmas


def save_figure_to_all(fig, basename: str, out_dirs: List[Path]):
    """Save matplotlib Figure as both .svg and .png to all valid output directories."""
    for d in out_dirs:
        try:
            d.mkdir(parents=True, exist_ok=True)
            fig.savefig(d / f"{basename}.svg", bbox_inches='tight')
            fig.savefig(d / f"{basename}.png", bbox_inches='tight', dpi=300)
            print(f"Saved figure: {d / f'{basename}.png'}")
        except Exception as e:
            print(f"Warning: Failed to save figure {basename} to {d}: {e}")


def save_csv_to_all(df: pd.DataFrame, filename: str, out_dirs: List[Path]):
    """Save DataFrame as CSV to all valid output directories."""
    for d in out_dirs:
        try:
            d.mkdir(parents=True, exist_ok=True)
            df.to_csv(d / filename, index=False)
            print(f"Saved CSV: {d / filename}")
        except Exception as e:
            print(f"Warning: Failed to save CSV {filename} to {d}: {e}")


def plot_cos_vs_radius_main(
    df_all_particles: pd.DataFrame,
    df_condition_summary: pd.DataFrame,
    out_dirs: List[Path],
    xscale: str = 'log',
    region_mode: str = 'radius'
):
    """
    メイン散布図: 方向追従コサイン cos(theta) vs 貨物半径 R_c
    - 個別粒子の散布 (半透明 + ジッター)
    - アンサンブル平均 Mean +- SEM (大マーカー + 実線)
    - アンサンブル中央値 Median +- IQR (破線)
    - 基準線: 完全一致 cos = 1.0, 直交・無相関 cos = 0.0
    """
    fig, ax = plt.subplots(figsize=(8.5, 6.2))

    # 背景参照ライン
    ax.axhline(0.0, color='#666666', linestyle=':', linewidth=1.5, zorder=1, label=r'Orthogonal / No alignment ($\cos \theta = 0$)')
    ax.axhline(1.0, color='#222222', linestyle='--', linewidth=1.8, zorder=1, label=r'Perfect alignment ($\cos \theta = 1$)')

    cond_radii = []
    cond_cos_means = []
    cond_cos_sems = []
    cond_cos_medians = []
    cond_cos_q25s = []
    cond_cos_q75s = []

    np.random.seed(42)

    for b_item in BEADS_INFO:
        bname = b_item['name']
        r_c = b_item['radius_um']
        marker = b_item['marker']
        color = b_item['color']
        lbl = b_item['label']

        sub_p = df_all_particles[df_all_particles['bead_name'] == bname]
        if sub_p.empty:
            continue

        cos_vals = sub_p['cos_mean'].dropna().values
        n_p = len(cos_vals)

        # 個別粒子散布 (ジッター付き半透明マーカー)
        jitter_factor = 1.0 + np.random.uniform(-0.06, 0.06, size=n_p)
        jittered_r = r_c * jitter_factor

        ax.scatter(
            jittered_r, cos_vals,
            color=color, marker=marker, s=55, alpha=0.38,
            edgecolors='none', zorder=2
        )

        mean_v = float(np.mean(cos_vals))
        sem_v = float(np.std(cos_vals, ddof=1) / np.sqrt(n_p)) if n_p > 1 else 0.0
        med_v = float(np.median(cos_vals))
        q25_v = float(np.percentile(cos_vals, 25))
        q75_v = float(np.percentile(cos_vals, 75))

        cond_radii.append(r_c)
        cond_cos_means.append(mean_v)
        cond_cos_sems.append(sem_v)
        cond_cos_medians.append(med_v)
        cond_cos_q25s.append(q25_v)
        cond_cos_q75s.append(q75_v)

        # 代表値のエラーバー付きマーカー (アンサンブル平均)
        ax.errorbar(
            r_c, mean_v, yerr=sem_v,
            fmt=marker, color=color, ecolor=color, elinewidth=2.2, capsize=5.0, capthick=1.5,
            markersize=11.0, markeredgewidth=1.8, markeredgecolor='#333333',
            label=f'{lbl} ($N={n_p}$)', zorder=5
        )

    # 代表値同士を繋ぐ実線
    if len(cond_radii) > 0:
        sort_idx = np.argsort(cond_radii)
        r_sorted = np.array(cond_radii)[sort_idx]
        mean_sorted = np.array(cond_cos_means)[sort_idx]
        med_sorted = np.array(cond_cos_medians)[sort_idx]

        ax.plot(r_sorted, mean_sorted, color='#333333', linestyle='-', linewidth=2.0, alpha=0.8, zorder=4)
        ax.plot(r_sorted, med_sorted, color='#888888', linestyle='-.', linewidth=1.5, alpha=0.85, zorder=3)

    ax.set_xscale(xscale)
    ax.set_xlabel(r'Cargo radius $R_c$ [$\mu\mathrm{m}$]')
    ax.set_ylabel(r'Directional alignment $\cos \theta$' + '\n' + r'$[(\mathbf{v}_{\mathrm{cargo}} \cdot \bar{\mathbf{v}}_{\mathrm{MT}}) \,/\, (|\mathbf{v}_{\mathrm{cargo}}| |\bar{\mathbf{v}}_{\mathrm{MT}}|)]$', fontsize=18)
    ax.set_title(r'Cargo Directional Coupling to Local MT Flow $\cos \theta(R_c)$' + f'\n(Window: {region_mode}, Ensemble Mean $\pm$ SEM)', fontsize=18)

    if xscale == 'log':
        ax.set_xlim(0.2, 16.0)
    else:
        ax.set_xlim(-0.5, 11.0)
    
    ax.set_ylim(-0.35, 1.05)

    ax.legend(
        loc='upper right', fontsize=10.0, framealpha=0.92,
        markerscale=0.85, handlelength=1.4, borderpad=0.35, labelspacing=0.28
    )

    suffix = f"_{xscale}" if xscale != 'log' else ""
    save_figure_to_all(fig, f"cos_vs_radius_scatter{suffix}", out_dirs)
    plt.close(fig)


def load_experiment_correlation_lengths(
    root_dir: Path,
    workspace_dir: Path
) -> Dict[str, float]:
    """
    bg_angular_correlation_length_per_experiment.csv から
    各実験ごとの空間配向相関長 xi [um] を取得する。
    """
    possible_paths = [
        workspace_dir / "figure" / "bg_angular_correlation" / "bg_angular_correlation_length_per_experiment.csv",
        root_dir / "figure" / "bg_angular_correlation" / "bg_angular_correlation_length_per_experiment.csv",
        Path("/mnt/NAS-Ebanaru/Sasaki/MTsingleBeads/figure/bg_angular_correlation/bg_angular_correlation_length_per_experiment.csv"),
        Path("/mnt/NAS-Ebanaru/sasaki/MTsingleBeads/figure/bg_angular_correlation/bg_angular_correlation_length_per_experiment.csv"),
    ]
    for p in possible_paths:
        if p.exists():
            try:
                df = pd.read_csv(p)
                if 'exp_dir' in df.columns and 'xi_um' in df.columns:
                    exp_xi_dict = {}
                    for _, row in df.iterrows():
                        k = str(row['exp_dir']).strip()
                        v = float(row['xi_um'])
                        if pd.notna(v) and v > 0:
                            exp_xi_dict[k] = v
                    print(f"Loaded {len(exp_xi_dict)} correlation lengths xi from {p}")
                    return exp_xi_dict
            except Exception as e:
                print(f"Warning: Failed to read {p}: {e}")
    print("[WARNING] Could not find bg_angular_correlation_length_per_experiment.csv")
    return {}


def plot_cos_vs_scaled_radius_main(
    df_all_particles: pd.DataFrame,
    df_condition_summary: pd.DataFrame,
    out_dirs: List[Path],
    xscale: str = 'log',
    region_mode: str = 'radius'
):
    """
    メイン散布図: 方向追従コサイン cos(theta) vs スケール半径 x = R_c / xi
    - 横軸: x = R_c / xi（各実験の空間配向相関長 xi [um] で無次元化）
    - 個別粒子の散布 (半透明マーカー、各実験の xi を反映)
    - アンサンブル代表点 Mean +- SEM (横軸・縦軸両方のエラーバー付き大マーカー)
    - アンサンブル中央値 Median
    - 基準線: 完全一致 cos = 1.0, 直交・無相関 cos = 0.0
    """
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    # 背景参照ライン
    ax.axhline(0.0, color='#666666', linestyle=':', linewidth=1.5, zorder=1, label=r'Orthogonal / No alignment ($\cos \theta = 0$)')
    ax.axhline(1.0, color='#222222', linestyle='--', linewidth=1.8, zorder=1, label=r'Perfect alignment ($\cos \theta = 1$)')

    cond_x_means = []
    cond_cos_means = []
    cond_cos_sems = []
    cond_x_medians = []
    cond_cos_medians = []

    np.random.seed(42)
    for b_item in BEADS_INFO:
        bname = b_item['name']
        marker = b_item['marker']
        color = b_item['color']
        lbl = b_item['label']

        sub_p = df_all_particles[df_all_particles['bead_name'] == bname]
        if sub_p.empty or 'rc_over_xi' not in sub_p.columns:
            continue

        valid_mask = sub_p['cos_mean'].notna() & sub_p['rc_over_xi'].notna()
        sub_valid = sub_p[valid_mask]
        if sub_valid.empty:
            continue

        cos_vals = sub_valid['cos_mean'].values
        x_vals = sub_valid['rc_over_xi'].values
        n_p = len(cos_vals)

        # 個別粒子散布 (各実験の xi に基づく x = R_c / xi)
        # 同一実験内の粒子が重ならないよう微小ジッター (±1.5%) を付与
        jitter_factor = 1.0 + np.random.uniform(-0.015, 0.015, size=n_p)
        jittered_x = x_vals * jitter_factor

        ax.scatter(
            jittered_x, cos_vals,
            color=color, marker=marker, s=55, alpha=0.38,
            edgecolors='none', zorder=2
        )

        mean_cos = float(np.mean(cos_vals))
        sem_cos = float(np.std(cos_vals, ddof=1) / np.sqrt(n_p)) if n_p > 1 else 0.0
        med_cos = float(np.median(cos_vals))

        mean_x = float(np.mean(x_vals))
        sem_x = float(np.std(x_vals, ddof=1) / np.sqrt(n_p)) if n_p > 1 else 0.0
        med_x = float(np.median(x_vals))

        cond_x_means.append(mean_x)
        cond_cos_means.append(mean_cos)
        cond_cos_sems.append(sem_cos)
        cond_x_medians.append(med_x)
        cond_cos_medians.append(med_cos)

        # 代表値のエラーバー付きマーカー (アンサンブル平均: 縦横両方の SEM)
        ax.errorbar(
            mean_x, mean_cos,
            xerr=sem_x, yerr=sem_cos,
            fmt=marker, color=color, ecolor=color, elinewidth=2.2, capsize=5.0, capthick=1.5,
            markersize=11.0, markeredgewidth=1.8, markeredgecolor='#333333',
            label=f'{lbl} ($N={n_p}$)', zorder=5
        )

    # 代表値同士を繋ぐ実線 (平均) と破線 (中央値)
    if len(cond_x_means) > 0:
        sort_idx = np.argsort(cond_x_means)
        x_sorted = np.array(cond_x_means)[sort_idx]
        mean_sorted = np.array(cond_cos_means)[sort_idx]

        sort_med_idx = np.argsort(cond_x_medians)
        x_med_sorted = np.array(cond_x_medians)[sort_med_idx]
        med_sorted = np.array(cond_cos_medians)[sort_med_idx]

        ax.plot(x_sorted, mean_sorted, color='#333333', linestyle='-', linewidth=2.0, alpha=0.8, zorder=4)
        ax.plot(x_med_sorted, med_sorted, color='#888888', linestyle='-.', linewidth=1.5, alpha=0.85, zorder=3)

    ax.set_xscale(xscale)
    ax.set_xlabel(r'Scaled cargo radius $R_c / \xi$ [-]')
    ax.set_ylabel(r'Directional alignment $\cos \theta$' + '\n' + r'$[(\mathbf{v}_{\mathrm{cargo}} \cdot \bar{\mathbf{v}}_{\mathrm{MT}}) \,/\, (|\mathbf{v}_{\mathrm{cargo}}| |\bar{\mathbf{v}}_{\mathrm{MT}}|)]$', fontsize=18)
    ax.set_title(r'Cargo Directional Coupling vs Scaled Radius $R_c / \xi$' + f'\n(Window: {region_mode}, Ensemble Mean $\pm$ SEM)', fontsize=18)

    if xscale == 'log':
        ax.set_xlim(0.03, 3.5)
    else:
        ax.set_xlim(-0.05, 2.2)

    ax.set_ylim(-0.35, 1.05)

    ax.legend(
        loc='upper right', fontsize=10.0, framealpha=0.92,
        markerscale=0.85, handlelength=1.4, borderpad=0.35, labelspacing=0.28
    )

    suffix = f"_{xscale}" if xscale != 'log' else ""
    save_figure_to_all(fig, f"cos_vs_scaled_radius_scatter{suffix}", out_dirs)
    plt.close(fig)


def plot_cos_6panels_grid(
    df_all_particles: pd.DataFrame,
    df_instant: pd.DataFrame,
    out_dirs: List[Path],
    region_mode: str = 'radius'
):
    """
    全 6 つのビーズサイズを 2行 3列 のグリッドに並べた個別パネル図。
    各パネルでそのビーズ径における cos(theta) の粒子別散布および代表値を表示。
    """
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 9.5), sharex=False, sharey=True)
    axes_flat = axes.flatten()

    np.random.seed(42)

    for idx, b_item in enumerate(BEADS_INFO):
        ax = axes_flat[idx]
        bname = b_item['name']
        d_um = b_item['diameter_um']
        r_c = b_item['radius_um']
        marker = b_item['marker']
        color = b_item['color']
        lbl = b_item['label']

        sub_p = df_all_particles[df_all_particles['bead_name'] == bname].copy()
        if sub_p.empty:
            ax.text(0.5, 0.5, "No Data", ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title(f"{lbl}", fontsize=14, fontweight='bold')
            continue

        n_p = len(sub_p)
        cos_vals = sub_p['cos_mean'].dropna().values

        # 参照線
        ax.axhline(0.0, color='#888888', linestyle=':', linewidth=1.2, zorder=1)
        ax.axhline(1.0, color='#333333', linestyle='--', linewidth=1.4, zorder=1)

        # 粒子ごとの値を並べる (横軸: 粒子インデックス)
        x_indices = np.arange(1, n_p + 1)
        ax.scatter(
            x_indices, cos_vals,
            color=color, marker=marker, s=65, alpha=0.65,
            edgecolors='#333333', linewidths=0.8, zorder=3,
            label=f'Particles ($N={n_p}$)'
        )

        # 粒子のエラーバー (時間標準誤差 SEM)
        if 'cos_sem' in sub_p.columns:
            sems = sub_p['cos_sem'].values
            ax.errorbar(
                x_indices, cos_vals, yerr=sems,
                fmt='none', ecolor=color, elinewidth=1.2, capsize=3.0, alpha=0.5, zorder=2
            )

        # アンサンブル平均値とエラー帯
        mean_v = float(np.mean(cos_vals))
        sem_v = float(np.std(cos_vals, ddof=1) / np.sqrt(n_p)) if n_p > 1 else 0.0
        med_v = float(np.median(cos_vals))
        q25_v = float(np.percentile(cos_vals, 25))
        q75_v = float(np.percentile(cos_vals, 75))

        ax.axhline(mean_v, color=color, linestyle='-', linewidth=2.0, zorder=4,
                   label=f'Mean: {mean_v:.2f} $\pm$ {sem_v:.2f}')
        ax.axhspan(mean_v - sem_v, mean_v + sem_v, color=color, alpha=0.15, zorder=2)
        ax.axhline(med_v, color='#333333', linestyle='-.', linewidth=1.5, zorder=4,
                   label=f'Median: {med_v:.2f}')

        ax.set_title(f"{lbl}  ($R_c = {r_c:.3f}\,\mu\mathrm{{m}}$)", fontsize=14, fontweight='bold')
        ax.set_xlabel("Particle Index", fontsize=13)
        if idx % 3 == 0:
            ax.set_ylabel(r"$\cos \theta$", fontsize=15)

        ax.set_ylim(-0.35, 1.05)
        ax.set_xlim(0, max(n_p + 1, 5))
        ax.legend(loc='lower left', fontsize=9.5, framealpha=0.9, borderpad=0.25)

    fig.suptitle(r'Directional Coupling $\cos \theta$ for Each Cargo Size Condition', fontsize=17.0, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure_to_all(fig, "cos_distribution_6panels", out_dirs)
    plt.close(fig)


def plot_cos_individual_panels(
    df_all_particles: pd.DataFrame,
    out_dirs: List[Path],
    region_mode: str = 'radius'
):
    """
    各ビーズ条件単独の個別図（それぞれのパネルのみの図）を個別に保存。
    """
    # individual サブディレクトリの準備
    indiv_dirs = [d / "individual_panels" for d in out_dirs]
    for d in indiv_dirs:
        d.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)

    for b_item in BEADS_INFO:
        bname = b_item['name']
        d_um = b_item['diameter_um']
        r_c = b_item['radius_um']
        marker = b_item['marker']
        color = b_item['color']
        lbl = b_item['label']

        sub_p = df_all_particles[df_all_particles['bead_name'] == bname].copy()
        if sub_p.empty:
            continue

        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        n_p = len(sub_p)
        cos_vals = sub_p['cos_mean'].dropna().values

        ax.axhline(0.0, color='#888888', linestyle=':', linewidth=1.4, zorder=1, label=r'No alignment ($\cos\theta=0$)')
        ax.axhline(1.0, color='#333333', linestyle='--', linewidth=1.6, zorder=1, label=r'Perfect alignment ($\cos\theta=1$)')

        x_indices = np.arange(1, n_p + 1)
        ax.scatter(
            x_indices, cos_vals,
            color=color, marker=marker, s=80, alpha=0.75,
            edgecolors='#333333', linewidths=1.0, zorder=3,
            label=f'Active Particles ($N={n_p}$)'
        )

        if 'cos_sem' in sub_p.columns:
            sems = sub_p['cos_sem'].values
            ax.errorbar(
                x_indices, cos_vals, yerr=sems,
                fmt='none', ecolor=color, elinewidth=1.4, capsize=3.5, alpha=0.6, zorder=2
            )

        mean_v = float(np.mean(cos_vals))
        sem_v = float(np.std(cos_vals, ddof=1) / np.sqrt(n_p)) if n_p > 1 else 0.0
        med_v = float(np.median(cos_vals))

        ax.axhline(mean_v, color=color, linestyle='-', linewidth=2.4, zorder=4,
                   label=f'Ensemble Mean: {mean_v:.3f} $\pm$ {sem_v:.3f}')
        ax.axhspan(mean_v - sem_v, mean_v + sem_v, color=color, alpha=0.18, zorder=2)
        ax.axhline(med_v, color='#333333', linestyle='-.', linewidth=1.8, zorder=4,
                   label=f'Median: {med_v:.3f}')

        ax.set_title(f"Cargo Alignment: {lbl} ($R_c = {r_c:.3f}\,\mu\mathrm{{m}}$)\n" +
                     r"$\cos \theta \equiv (\mathbf{v}_{\mathrm{cargo}} \cdot \bar{\mathbf{v}}_{\mathrm{MT}}) \,/\, (|\mathbf{v}_{\mathrm{cargo}}| |\bar{\mathbf{v}}_{\mathrm{MT}}|)$",
                     fontsize=15.0)
        ax.set_xlabel("Particle Index", fontsize=14)
        ax.set_ylabel(r"Directional alignment $\cos \theta$", fontsize=15)
        ax.set_ylim(-0.35, 1.05)
        ax.set_xlim(0, max(n_p + 1, 5))
        ax.legend(loc='lower left', fontsize=10.5, framealpha=0.92)

        save_figure_to_all(fig, f"cos_single_panel_{bname}", indiv_dirs)
        save_figure_to_all(fig, f"cos_single_panel_{bname}", out_dirs)
        plt.close(fig)


def plot_cos_vs_window_6panels(
    df_all_sigmas: pd.DataFrame,
    out_dirs: List[Path],
    exp_xi_dict: Optional[Dict[str, float]] = None
):
    """
    窓サイズ依存性 cos(theta)(R) vs R の 6 パネル個別図。
    各ビーズ径ごとに、窓を広げたときのコサイン減衰カーブを個別パネルで表示。
    各実験の微小管空間配向相関長 xi の縦線および貨物半径 R_c の縦線を表示。
    """
    if df_all_sigmas.empty:
        return

    fig, axes = plt.subplots(2, 3, figsize=(15.0, 9.5), sharex=True, sharey=True)
    axes_flat = axes.flatten()

    for idx, b_item in enumerate(BEADS_INFO):
        ax = axes_flat[idx]
        bname = b_item['name']
        r_c = b_item['radius_um']
        marker = b_item['marker']
        color = b_item['color']
        lbl = b_item['label']

        sub = df_all_sigmas[df_all_sigmas['bead_name'] == bname]
        if sub.empty:
            ax.text(0.5, 0.5, "No Data", ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title(f"{lbl}", fontsize=14, fontweight='bold')
            continue

        sub = sub.copy()
        sub['unique_pid'] = sub['exp_dir'].astype(str) + '_' + sub['particle'].astype(str)
        n_p_total = sub['unique_pid'].nunique()
        n_e_total = sub['exp_dir'].nunique()

        agg = sub.groupby('window_radius_um').agg(
            cos_mean=('cos_mean', 'mean'),
            cos_sem=('cos_mean', lambda x: np.std(x, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0),
            n_parts=('unique_pid', 'nunique'),
            n_exps=('exp_dir', 'nunique')
        ).reset_index()

        agg = agg.sort_values('window_radius_um')
        r_w = agg['window_radius_um'].values
        c_m = agg['cos_mean'].values
        c_s = agg['cos_sem'].values

        ax.axhline(0.0, color='#888888', linestyle=':', linewidth=1.2, zorder=1)
        ax.axhline(1.0, color='#333333', linestyle='--', linewidth=1.4, zorder=1)

        # 貨物半径 R_c の位置に縦点線を引く
        ax.axvline(r_c, color='#cc2222', linestyle=':', linewidth=1.5, zorder=2, label=f'Cargo $R_c={r_c:.2f}\,\mu\mathrm{{m}}$')

        # 実験ごとの空間配向相関長 xi の縦線を引く
        if exp_xi_dict:
            exp_names = sub['exp_dir'].unique()
            xis = [exp_xi_dict[e] for e in exp_names if e in exp_xi_dict and not np.isnan(exp_xi_dict[e])]
            if xis:
                xi_m = float(np.mean(xis))
                xi_sem = float(np.std(xis, ddof=1) / np.sqrt(len(xis))) if len(xis) > 1 else 0.0

                # 個別実験の xi を薄い点線で表示
                for xi_val in xis:
                    ax.axvline(xi_val, color='#2b5c8f', linestyle=':', linewidth=1.2, alpha=0.4, zorder=2)

                # 平均 xi の縦破線と SEM バンド
                if xi_sem > 0:
                    ax.axvspan(xi_m - xi_sem, xi_m + xi_sem, color='#2b5c8f', alpha=0.12, zorder=1)
                    xi_lbl = rf'Corr. length $\xi={xi_m:.2f}\pm{xi_sem:.2f}\,\mu\mathrm{{m}}$'
                else:
                    xi_lbl = rf'Corr. length $\xi={xi_m:.2f}\,\mu\mathrm{{m}}$'
                ax.axvline(xi_m, color='#2b5c8f', linestyle='--', linewidth=1.8, zorder=2, label=xi_lbl)

        ax.plot(r_w, c_m, marker=marker, markersize=6.5, color=color, linewidth=2.2, label=f'{lbl} ($N={n_p_total}$, {n_e_total} movies)', zorder=4)
        ax.fill_between(r_w, c_m - c_s, c_m + c_s, color=color, alpha=0.2, zorder=3)

        ax.set_xscale('log')
        ax.set_xlim(0.4, 120.0)
        ax.set_ylim(-0.25, 0.95)
        ax.set_title(f"{lbl}  ($R_c = {r_c:.3f}\,\mu\mathrm{{m}}$)", fontsize=14, fontweight='bold')
        ax.set_xlabel(r"Window Radius $R$ [$\mu\mathrm{m}$]", fontsize=13)
        if idx % 3 == 0:
            ax.set_ylabel(r"Alignment $\cos \theta(R)$", fontsize=15)
        ax.legend(loc='upper right', fontsize=9.0, framealpha=0.9)

    fig.suptitle(r'Window Radius $R$ Dependence of Directional Alignment $\cos \theta(R)$', fontsize=17.0, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure_to_all(fig, "cos_vs_window_6panels", out_dirs)
    plt.close(fig)


def plot_cos_distribution_violin(
    df_all_particles: pd.DataFrame,
    out_dirs: List[Path]
):
    """バイオリン＋ボックスプロットで各ビーズ径の cos(theta) 分布形状を可視化"""
    fig, ax = plt.subplots(figsize=(9.2, 6.0))

    data_to_plot = []
    labels = []
    colors = []
    positions = []

    for idx, b_item in enumerate(BEADS_INFO):
        bname = b_item['name']
        sub_p = df_all_particles[df_all_particles['bead_name'] == bname]
        vals = sub_p['cos_mean'].dropna().values
        if len(vals) > 0:
            data_to_plot.append(vals)
            labels.append(b_item['label'])
            colors.append(b_item['color'])
            positions.append(idx + 1)

    if not data_to_plot:
        plt.close(fig)
        return

    parts = ax.violinplot(data_to_plot, positions=positions, showmeans=False, showmedians=False, showextrema=False, widths=0.7)
    for pc, col in zip(parts['bodies'], colors):
        pc.set_facecolor(col)
        pc.set_edgecolor('#333333')
        pc.set_alpha(0.35)

    bp = ax.boxplot(
        data_to_plot, positions=positions, widths=0.25,
        patch_artist=True, showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "#333333", "markersize": 6},
        medianprops={"color": "#333333", "linewidth": 2.0},
        boxprops={"linewidth": 1.4},
        whiskerprops={"linewidth": 1.4},
        capprops={"linewidth": 1.4},
        flierprops={"marker": "x", "markersize": 4, "alpha": 0.5}
    )
    for patch, col in zip(bp['boxes'], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.65)

    np.random.seed(42)
    for pos, vals, col in zip(positions, data_to_plot, colors):
        jit = np.random.normal(pos, 0.04, size=len(vals))
        ax.scatter(jit, vals, color=col, s=35, alpha=0.6, edgecolors='none', zorder=4)

    ax.axhline(0.0, color='#666666', linestyle=':', linewidth=1.4)
    ax.axhline(1.0, color='#222222', linestyle='--', linewidth=1.6)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_xlabel(r'Cargo Diameter ($2 R_c$)')
    ax.set_ylabel(r'Directional alignment $\cos \theta(R_c)$')
    ax.set_title(r'Distribution of Directional Alignment $\cos \theta(R_c)$ to Local MT Flow')
    ax.set_ylim(-0.35, 1.05)

    save_figure_to_all(fig, "cos_distribution_violin", out_dirs)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description='Quantify and plot cargo directional coupling cos(theta) vs cargo radius R_c.'
    )
    parser.add_argument('--root_dir', type=str, default=None,
                        help='Data root directory containing beads* folders (auto-detected if None)')
    parser.add_argument('--region_mode', type=str, default='radius', choices=['radius', 'diameter', 'fixed'],
                        help='Window evaluation mode: radius (R = R_c), diameter (R = 2*R_c), fixed (R = fixed_window_um)')
    parser.add_argument('--fixed_window_um', type=float, default=3.37,
                        help='Fixed window radius [um] if region_mode == fixed')
    parser.add_argument('--min_region_um', type=float, default=0.0,
                        help='Minimum window radius [um] floor (default: 0.0)')
    parser.add_argument('--alpha_threshold', type=float, default=0.5,
                        help='Threshold for active particles (alpha > threshold). Set <= 0 to disable.')
    parser.add_argument('--min_points_per_particle', type=int, default=5,
                        help='Minimum frames tracked per particle to be included in particle summary.')
    parser.add_argument('--scale', type=float, default=0.11,
                        help='Spatial scale [um/pixel] (default: 0.11)')
    parser.add_argument('--interval', type=float, default=4.0,
                        help='Frame interval [s] (default: 4.0)')
    parser.add_argument('--save_dir', type=str, default='figure/chi_vs_radius',
                        help='Output figure directory name under workspace/root')
    args = parser.parse_args()

    root_dir = Path(args.root_dir) if args.root_dir else find_default_root()
    print(f"Using root directory: {root_dir}")

    # 保存先ディレクトリの決定 (ワークスペース内 + root_dir 内)
    out_dirs = [CURRENT_DIR / args.save_dir]
    if root_dir != CURRENT_DIR and (root_dir / "figure").exists():
        out_dirs.append(root_dir / args.save_dir)

    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)
    print(f"Output directories: {[str(d) for d in out_dirs]}")

    # 1. アクティブ粒子判定 (alpha > threshold)
    active_particles_map = None
    if args.alpha_threshold > 0:
        print(f"\nFiltering active particles (iMSD alpha > {args.alpha_threshold})...")
        all_exp_dirs = []
        for b_item in BEADS_INFO:
            all_exp_dirs.extend(find_experiment_dirs(root_dir, b_item['name']))
        active_particles_map = filter_active_particles(
            all_exp_dirs, interval=args.interval, scale=args.scale, alpha_threshold=args.alpha_threshold
        )
        print(f"Identified {sum(active_particles_map.values())} active particles out of {len(active_particles_map)} total particles.")

    # 2. 全ビーズ条件のデータロード・集計
    all_instant_list = []
    all_particle_list = []
    all_sigma_list = []

    print(f"\nExtracting alignment cos(theta) (mode: {args.region_mode})...")
    for b_item in BEADS_INFO:
        bname = b_item['name']
        exp_dirs = find_experiment_dirs(root_dir, bname)
        if not exp_dirs:
            print(f"Warning: No valid experiment dirs found for {bname}")
            continue

        print(f"Processing {bname} ({len(exp_dirs)} experiments)...")
        df_inst, df_part, df_sig = load_and_compute_data(
            exp_dirs=exp_dirs,
            bead_item=b_item,
            region_mode=args.region_mode,
            fixed_um=args.fixed_window_um,
            min_region_um=args.min_region_um,
            scale=args.scale,
            active_particles_map=active_particles_map,
            alpha_threshold=args.alpha_threshold,
            min_points_per_particle=args.min_points_per_particle
        )

        if not df_inst.empty:
            all_instant_list.append(df_inst)
        if not df_part.empty:
            all_particle_list.append(df_part)
            print(f"  -> Extracted {len(df_part)} particles ({len(df_inst)} instant points)")
        if not df_sig.empty:
            all_sigma_list.append(df_sig)

    if not all_particle_list:
        print("Error: No particle data could be processed. Exiting.")
        sys.exit(1)

    df_all_particles = pd.concat(all_particle_list, ignore_index=True)
    df_instant = pd.concat(all_instant_list, ignore_index=True) if all_instant_list else pd.DataFrame()
    df_all_sigmas = pd.concat(all_sigma_list, ignore_index=True) if all_sigma_list else pd.DataFrame()

    # 3. 実験ごとの空間配向相関長 xi [um] の紐付け
    exp_xi_dict = load_experiment_correlation_lengths(root_dir, CURRENT_DIR)
    if exp_xi_dict:
        df_all_particles['xi_um'] = df_all_particles['exp_dir'].astype(str).str.strip().map(exp_xi_dict)
        df_all_particles['rc_over_xi'] = df_all_particles['radius_um'] / df_all_particles['xi_um']
    else:
        df_all_particles['xi_um'] = np.nan
        df_all_particles['rc_over_xi'] = np.nan

    # 4. 条件レベルのサマリー統計量作成
    condition_summary_rows = []
    for b_item in BEADS_INFO:
        bname = b_item['name']
        sub = df_all_particles[df_all_particles['bead_name'] == bname]
        if sub.empty:
            continue

        cos_arr = sub['cos_mean'].dropna().values
        chi_arr = sub['chi_mean'].dropna().values
        n_p = len(cos_arr)

        xi_arr = sub['xi_um'].dropna().values if 'xi_um' in sub.columns else np.array([])
        rc_xi_arr = sub['rc_over_xi'].dropna().values if 'rc_over_xi' in sub.columns else np.array([])

        condition_summary_rows.append({
            'bead_name': bname,
            'diameter_um': b_item['diameter_um'],
            'radius_um': b_item['radius_um'],
            'region_mode': args.region_mode,
            'target_region_um': float(sub['target_region_um'].iloc[0]),
            'actual_window_um': float(sub['actual_window_um'].iloc[0]),
            'n_particles': n_p,
            # xi & rc_over_xi 統計量
            'xi_um_mean': float(np.mean(xi_arr)) if len(xi_arr) > 0 else np.nan,
            'xi_um_sem': float(np.std(xi_arr, ddof=1) / np.sqrt(len(xi_arr))) if len(xi_arr) > 1 else 0.0,
            'rc_over_xi_mean': float(np.mean(rc_xi_arr)) if len(rc_xi_arr) > 0 else np.nan,
            'rc_over_xi_sem': float(np.std(rc_xi_arr, ddof=1) / np.sqrt(len(rc_xi_arr))) if len(rc_xi_arr) > 1 else 0.0,
            'rc_over_xi_median': float(np.median(rc_xi_arr)) if len(rc_xi_arr) > 0 else np.nan,
            # cos 統計量
            'cos_mean': float(np.mean(cos_arr)),
            'cos_std': float(np.std(cos_arr, ddof=1)) if n_p > 1 else 0.0,
            'cos_sem': float(np.std(cos_arr, ddof=1) / np.sqrt(n_p)) if n_p > 1 else 0.0,
            'cos_median': float(np.median(cos_arr)),
            'cos_q25': float(np.percentile(cos_arr, 25)),
            'cos_q75': float(np.percentile(cos_arr, 75)),
            # chi 統計量
            'chi_mean': float(np.mean(chi_arr)) if len(chi_arr) > 0 else np.nan,
            'chi_sem': float(np.std(chi_arr, ddof=1) / np.sqrt(len(chi_arr))) if len(chi_arr) > 1 else 0.0,
            'chi_median': float(np.median(chi_arr)) if len(chi_arr) > 0 else np.nan,
        })

    df_condition_summary = pd.DataFrame(condition_summary_rows)

    # 5. CSV 保存
    save_csv_to_all(df_all_particles, "cos_vs_radius_particle_summary.csv", out_dirs)
    save_csv_to_all(df_condition_summary, "cos_vs_radius_condition_summary.csv", out_dirs)

    # 6. プロット描画
    print("\nGenerating figures focusing on cos(theta)...")
    # (1) メイン散布図: cos vs R_c (対数軸)
    plot_cos_vs_radius_main(df_all_particles, df_condition_summary, out_dirs, xscale='log', region_mode=args.region_mode)
    # (2) メイン散布図: cos vs R_c (線形軸)
    plot_cos_vs_radius_main(df_all_particles, df_condition_summary, out_dirs, xscale='linear', region_mode=args.region_mode)
    # (3) 無次元化メイン散布図: cos vs R_c / xi (対数軸)
    plot_cos_vs_scaled_radius_main(df_all_particles, df_condition_summary, out_dirs, xscale='log', region_mode=args.region_mode)
    # (4) 無次元化メイン散布図: cos vs R_c / xi (線形軸)
    plot_cos_vs_scaled_radius_main(df_all_particles, df_condition_summary, out_dirs, xscale='linear', region_mode=args.region_mode)
    # (5) 各ビーズ条件 6 パネル個別図
    plot_cos_6panels_grid(df_all_particles, df_instant, out_dirs, region_mode=args.region_mode)
    # (6) 各ビーズ条件ごとの個別単体パネル図 (1パネルずつ独立)
    plot_cos_individual_panels(df_all_particles, out_dirs, region_mode=args.region_mode)
    # (7) 窓サイズ依存性 6 パネル図
    plot_cos_vs_window_6panels(df_all_sigmas, out_dirs, exp_xi_dict=exp_xi_dict)
    # (8) バイオリン分布図
    plot_cos_distribution_violin(df_all_particles, out_dirs)

    print("\n=== Cosine Alignment Summary Table ===")
    cols_disp = ['bead_name', 'radius_um', 'xi_um_mean', 'rc_over_xi_mean', 'n_particles', 'cos_mean', 'cos_sem', 'cos_median']
    print(df_condition_summary[cols_disp].to_string(index=False))
    print("\nAll cos-focused figures generated successfully!")


if __name__ == '__main__':
    main()
