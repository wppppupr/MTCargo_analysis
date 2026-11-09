#!/usr/bin/env python3
"""
MSD.py

カーゴ微粒子の平均二乗変位 (Mean Squared Displacement: MSD) 解析スクリプト。
- 個別粒子 MSD (iMSD) の算出とプール (\alpha > 0.5 フィルタリング)
- アンサンブル平均 MSD および四分位範囲 (IQR) の算出・描画
- MSD(300 s) と MSDの 300s~1000s での傾き alpha の 2軸プロット
"""

import os
import sys
import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
from scipy.optimize import curve_fit

# libsモジュールの読み込み
from libs import fit_model as fm
from libs import displacement as dpm
from libs import cal_vel as cv
from libs.experiment_cutoff import get_cutoff_frame, DEFAULT_EXPERIMENT_CUTOFFS, load_strict_active_particles

# スタイルの適用
style_path = Path(__file__).parent / 'libs' / 'my_style.mplstyle'
if style_path.exists():
    plt.style.use(str(style_path))
    style_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
else:
    style_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

# データルートディレクトリの候補
POSSIBLE_ROOTS = [
    Path('/Volumes/data-1/Sasaki/MTsingleBeads'),
    Path('/Volumes/data-1/sasaki/MTsingleBeads'),
    Path('/Volumes/data/Sasaki/MTsingleBeads'),
    Path('/mnt/NAS-Ebanaru/sasaki/MTsingleBeads'),
    Path('/mnt/NAS-Ebanaru/Sasaki/MTsingleBeads'),
    Path('/Volumes/data/Sasaki/MTSingleBeads'),
]

BEADS_INFO = [
    {"name": "beads06um", "diameter_um": 0.63, "marker": "^"},
    {"name": "beads1um",  "diameter_um": 1.18, "marker": "o"},
    {"name": "beads3um",  "diameter_um": 3.37, "marker": "d"},
    {"name": "beads5um",  "diameter_um": 5.00, "marker": "p"},
    {"name": "beads7um",  "diameter_um": 7.24, "marker": "h"},
    {"name": "beads20um", "diameter_um": 20.0, "marker": "s"},
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


mypass = find_default_root()


def calc_particle_alpha(sub_df: pd.DataFrame, min_t: float = 4.0, max_t: float = 300.0) -> float:
    """
    個別粒子MSD (iMSD) に対して対数空間でべき乗則 MSD ~ t^alpha をフィッティングし、
    異常拡散指数 alpha を算出する。
    """
    mask = (sub_df['lag time'] >= min_t) & (sub_df['lag time'] <= max_t) & (sub_df['MSD'] > 0)
    if np.sum(mask) < 3:
        mask = (sub_df['MSD'] > 0)
    if np.sum(mask) < 3:
        return np.nan
    dt = sub_df['lag time'][mask].to_numpy(dtype=float)
    msd = sub_df['MSD'][mask].to_numpy(dtype=float)
    slope, _ = np.polyfit(np.log10(dt), np.log10(msd), 1)
    return float(slope)


def concat_pooled_particles_MSD(folder, interval_list, scale = 0.11, alpha_threshold = 0.5, apply_cutoff = True, strict_particles = None):
    """
    全動画から追跡された全粒子の個別MSD (iMSD) を1つのプールに統合する。
    strict_particles (set of (exp_name, particle_id)) が指定された場合、
    全解析共通の厳格な活性貨物粒子 (N=70) のみを採用する。
    """
    all_imsds = []
    file_list = sorted(glob.glob(str(Path(folder) / "*" / "*" / "beads_tracks.csv")))
    for vid_idx, file_path in enumerate(file_list):
        exp_dir_name = Path(file_path).parent.name
        interval = interval_list[vid_idx] if vid_idx < len(interval_list) else interval_list[-1]
        raw_df = pd.read_csv(file_path)
        if apply_cutoff:
            cut = get_cutoff_frame(exp_dir_name)
            if cut is not None and 'frame' in raw_df.columns:
                raw_df = raw_df[raw_df['frame'] <= cut].copy()
                if len(raw_df) == 0:
                    continue

        if strict_particles is not None and 'particle' in raw_df.columns:
            valid_pids = [pid for pid in raw_df['particle'].unique() if (exp_dir_name, int(pid)) in strict_particles]
            raw_df = raw_df[raw_df['particle'].isin(valid_pids)].copy()
            if len(raw_df) == 0:
                continue

        track = cv.cal(raw_df, scale=scale, frame_interval=interval)
        imsd_df = dpm.imsd(track, scale=scale, time_scale=interval, display=False)
        imsd_df['video_idx'] = vid_idx
        imsd_df['exp_name'] = exp_dir_name
        imsd_df['unique_particle_id'] = f"{exp_dir_name}_" + imsd_df['particle'].astype(str)
        all_imsds.append(imsd_df)
        
    if len(all_imsds) == 0:
        empty_df = pd.DataFrame(columns=['unique_particle_id', 'video_idx', 'exp_name', 'particle', 'lag time', 'MSD', 'alpha'])
        return empty_df, empty_df, 0, 0
        
    df_pool = pd.concat(all_imsds, ignore_index=True)
    df_pool['MSD'] = pd.to_numeric(df_pool['MSD'], errors='coerce')
    df_pool['lag time'] = pd.to_numeric(df_pool['lag time'], errors='coerce')
    df_pool = df_pool.dropna(subset=['MSD', 'lag time'])
    
    # 全個別粒子ごとの alpha を算出
    p_alphas = {}
    for pid, grp in df_pool.groupby('unique_particle_id'):
        p_alphas[pid] = calc_particle_alpha(grp, 4.0, 300.0)
        
    df_pool['alpha'] = df_pool['unique_particle_id'].map(p_alphas)
    
    n_total = df_pool['unique_particle_id'].nunique()
    if strict_particles is not None:
        df_filtered = df_pool.copy()
    else:
        df_filtered = df_pool[df_pool['alpha'] > alpha_threshold].copy()
    n_filtered = df_filtered['unique_particle_id'].nunique()
    
    return df_filtered, df_pool, n_total, n_filtered


def calc_bootstrap_ci(
    data: np.ndarray,
    n_bootstraps: int = 1000,
    ci: float = 0.95,
    random_state: int = 42
) -> Tuple[float, float, float]:
    """
    ブートストラップ法 (復元抽出) により標本平均の標準誤差 (Bootstrap SEM) および信頼区間 (CI) を算出する。

    Parameters
    ----------
    data : array-like
        観測データ (1次元配列)
    n_bootstraps : int
        ブートストラップ反復回数 (デフォルト: 1000)
    ci : float
        信頼係数 (デフォルト: 0.95 -> 95%信頼区間)
    random_state : int
        乱数シード

    Returns
    -------
    boot_sem : float
        ブートストラップ標準誤差 (標本平均の標準偏差)
    ci_low : float
        信頼区間下限 (例: 95% CI なら 2.5%タイル)
    ci_high : float
        信頼区間上限 (例: 95% CI なら 97.5%タイル)
    """
    arr = np.asarray(data, dtype=float)
    arr = arr[~np.isnan(arr) & np.isfinite(arr)]
    n = len(arr)
    if n == 0:
        return np.nan, np.nan, np.nan
    if n == 1:
        return 0.0, float(arr[0]), float(arr[0])

    rng = np.random.default_rng(random_state)
    alpha = (1.0 - ci) / 2.0

    # 高速ベクトル化サンプリング
    boot_idx = rng.integers(0, n, size=(n_bootstraps, n))
    boot_means = np.mean(arr[boot_idx], axis=1)

    boot_sem = float(np.std(boot_means, ddof=1))
    ci_low = float(np.percentile(boot_means, 100.0 * alpha))
    ci_high = float(np.percentile(boot_means, 100.0 * (1.0 - alpha)))
    return boot_sem, ci_low, ci_high


def calc_pooled_MSD_stats(df_pool: pd.DataFrame, min_particles: int = 1,
                          n_bootstraps: int = 1000, ci: float = 0.95) -> pd.DataFrame:
    """
    プールされた全粒子MSDから、lag timeごとのアンサンブル平均値 (mean)、
    ブートストラップ法による標準誤差 (boot_sem) および信頼区間 (boot_ci_low, boot_ci_high)、
    四分位範囲 IQR (25%, 75%タイル)、10%, 90%タイル、標準偏差、粒子数を算出する。
    """
    if df_pool.empty:
        return pd.DataFrame(columns=[
            'mean', 'boot_sem', 'boot_ci_low', 'boot_ci_high',
            'q25', 'q75', 'q10', 'q90', 'median', 'std', 'count'
        ])
    grouped = df_pool.groupby('lag time')['MSD']
    stats_df = pd.DataFrame({
        'mean': grouped.mean(),
        'q25': grouped.quantile(0.25),
        'q75': grouped.quantile(0.75),
        'q10': grouped.quantile(0.10),
        'q90': grouped.quantile(0.90),
        'median': grouped.median(),
        'std': grouped.std(),
        'count': grouped.count()
    })

    # 各 lag time についてブートストラップ信頼区間と標準誤差を算出
    boot_sems = []
    boot_ci_lows = []
    boot_ci_highs = []
    for lag_val, grp_vals in grouped:
        arr_vals = grp_vals.to_numpy(dtype=float)
        b_sem, c_low, c_high = calc_bootstrap_ci(arr_vals, n_bootstraps=n_bootstraps, ci=ci)
        boot_sems.append(b_sem)
        boot_ci_lows.append(c_low)
        boot_ci_highs.append(c_high)

    stats_df['boot_sem'] = boot_sems
    stats_df['boot_ci_low'] = boot_ci_lows
    stats_df['boot_ci_high'] = boot_ci_highs

    if min_particles > 1:
        stats_df = stats_df[stats_df['count'] >= min_particles]
    return stats_df


def extract_msd_at_lag(df_pool: pd.DataFrame, target_t: float = 300.0, tol: float = 12.0,
                       n_bootstraps: int = 1000, ci: float = 0.95) -> dict:
    """
    個別粒子データから target_t 近傍の MSD 値を集計し、ブートストラップ誤差を含む統計量を返す。
    """
    records = []
    for pid, grp in df_pool.groupby('unique_particle_id'):
        sub = grp[(grp['lag time'] >= target_t - tol) & (grp['lag time'] <= target_t + tol) & (grp['MSD'] > 0)]
        if not sub.empty:
            best_row = sub.iloc[(sub['lag time'] - target_t).abs().argsort()[:1]]
            records.append(float(best_row['MSD'].values[0]))
    if len(records) > 0:
        arr = np.array(records, dtype=float)
        mean_val = float(np.mean(arr))
        std_val = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        sem_val = float(std_val / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
        boot_sem, boot_ci_low, boot_ci_high = calc_bootstrap_ci(arr, n_bootstraps=n_bootstraps, ci=ci)
        return {
            'mean': mean_val,
            'std': std_val,
            'sem': sem_val,
            'boot_sem': boot_sem,
            'boot_ci_low': boot_ci_low,
            'boot_ci_high': boot_ci_high,
            'median': float(np.median(arr)),
            'q25': float(np.percentile(arr, 25)),
            'q75': float(np.percentile(arr, 75)),
            'n': len(arr)
        }
    return {
        'mean': np.nan, 'std': np.nan, 'sem': np.nan,
        'boot_sem': np.nan, 'boot_ci_low': np.nan, 'boot_ci_high': np.nan,
        'median': np.nan, 'q25': np.nan, 'q75': np.nan, 'n': 0
    }


def calc_msd_slope_range(
    df_pool: pd.DataFrame,
    min_t: float = 300.0,
    max_t: float = 1000.0,
    min_total_track_length: float = 1000.0,
    min_points: int = 3,
    n_bootstraps: int = 1000,
    ci: float = 0.95
) -> dict:
    """
    長時間 (max lag time >= min_total_track_length, 既定: 1000s) 追跡できた個別粒子それぞれの
    MSD に対して、指定ラグ時間範囲 (既定: 300s~1000s) でのべき乗則傾き alpha を算出する。
    """
    part_slopes = []
    part_ids = []
    
    for pid, grp in df_pool.groupby('unique_particle_id'):
        max_lag = grp['lag time'].max()
        # 長時間 (1000s以上) 追跡できた粒子のみに絞る
        if max_lag < min_total_track_length:
            continue
            
        mask = (grp['lag time'] >= min_t) & (grp['lag time'] <= max_t) & (grp['MSD'] > 0)
        if np.sum(mask) >= min_points:
            dt = grp['lag time'][mask].to_numpy(dtype=float)
            msd = grp['MSD'][mask].to_numpy(dtype=float)
            slope, _ = np.polyfit(np.log10(dt), np.log10(msd), 1)
            part_slopes.append(slope)
            part_ids.append(pid)
            
    if len(part_slopes) > 0:
        arr = np.array(part_slopes, dtype=float)
        mean_val = float(np.mean(arr))
        std_val = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        sem_val = float(std_val / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
        boot_sem, boot_ci_low, boot_ci_high = calc_bootstrap_ci(arr, n_bootstraps=n_bootstraps, ci=ci)
        return {
            'mean': mean_val,
            'std': std_val,
            'sem': sem_val,
            'boot_sem': boot_sem,
            'boot_ci_low': boot_ci_low,
            'boot_ci_high': boot_ci_high,
            'median': float(np.median(arr)),
            'q25': float(np.percentile(arr, 25)),
            'q75': float(np.percentile(arr, 75)),
            'n': len(arr),
            'particle_ids': part_ids
        }
    return {
        'mean': np.nan, 'std': np.nan, 'sem': np.nan, 
        'boot_sem': np.nan, 'boot_ci_low': np.nan, 'boot_ci_high': np.nan,
        'median': np.nan, 'q25': np.nan, 'q75': np.nan, 
        'n': 0, 'particle_ids': []
    }


def save_figure_to_all(fig, basename: str, out_dirs: List[Path]):
    """Save matplotlib Figure as both .svg and .png to all valid output directories."""
    for d in out_dirs:
        try:
            if d.exists() or d.parent.exists():
                d.mkdir(parents=True, exist_ok=True)
                fig.savefig(d / f"{basename}.svg", bbox_inches='tight')
                fig.savefig(d / f"{basename}.png", bbox_inches='tight')
        except Exception as e:
            print(f"Warning: Failed to save {basename} to {d}: {e}")


def save_csv_to_all(df: pd.DataFrame, filename: str, out_dirs: List[Path]):
    """Save DataFrame as CSV to all valid output directories."""
    for d in out_dirs:
        try:
            if d.exists() or d.parent.exists():
                d.mkdir(parents=True, exist_ok=True)
                df.to_csv(d / filename, index=False)
        except Exception as e:
            print(f"Warning: Failed to save {filename} to {d}: {e}")


def clean_redundant_files(out_dirs: List[Path]):
    redundant_patterns = [
        "dimensionless_MSD.*",
        "dimensionless_alpha.*",
        "dimensionless_local_alpha.*",
        "local_alpha.*",
        "alpha.*",
        "MSD_RTP_fit*.*",
        "rtp_effective_diffusion.*",
        "relaxation_times_vs_diameter.*",
        "tau_eff_vs_diameter.*",
        "rtp_2state_msd_fit_summary.csv",
    ]
    for d in out_dirs:
        if not d.exists():
            continue
        for pattern in redundant_patterns:
            for p in d.glob(pattern):
                try:
                    p.unlink()
                except Exception:
                    pass


def main():
    root_dir = find_default_root()
    workspace_dir = Path(__file__).parent.resolve()
    
    out_dirs = [
        workspace_dir / "figure" / "msd",
        root_dir / "figure" / "msd",
    ]
    for d in out_dirs:
        try:
            if d.exists() or d.parent.exists():
                d.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    print(f"Data Root Directory: {root_dir}")
    print("Saving outputs to:")
    for d in out_dirs:
        if d.parent.exists():
            print(f"  {d}")

    clean_redundant_files(out_dirs)

    print("\n--- 1. Loading & Filtering MSD datasets (Strict Active Particles N=70) ---")
    strict_set = load_strict_active_particles()
    alpha_thresh = 0.5
    pool_06um, raw_06um, n_tot_06um, n_filt_06um = concat_pooled_particles_MSD(root_dir / "beads06um", [4, 4, 4, 4], alpha_threshold=alpha_thresh, strict_particles=strict_set)
    pool_1um,  raw_1um,  n_tot_1um,  n_filt_1um  = concat_pooled_particles_MSD(root_dir / "beads1um",  [4, 4, 4, 4], alpha_threshold=alpha_thresh, strict_particles=strict_set)
    pool_3um,  raw_3um,  n_tot_3um,  n_filt_3um  = concat_pooled_particles_MSD(root_dir / "beads3um",  [4, 4, 4, 4], alpha_threshold=alpha_thresh, strict_particles=strict_set)
    pool_5um,  raw_5um,  n_tot_5um,  n_filt_5um  = concat_pooled_particles_MSD(root_dir / "beads5um",  [4, 4, 4, 4], alpha_threshold=alpha_thresh, strict_particles=strict_set)
    pool_7um,  raw_7um,  n_tot_7um,  n_filt_7um  = concat_pooled_particles_MSD(root_dir / "beads7um",  [4, 4, 4],    alpha_threshold=alpha_thresh, strict_particles=strict_set)
    pool_20um, raw_20um, n_tot_20um, n_filt_20um = concat_pooled_particles_MSD(root_dir / "beads20um", [4, 4, 4],    alpha_threshold=alpha_thresh, strict_particles=strict_set)

    stats_06um = calc_pooled_MSD_stats(pool_06um)
    stats_1um = calc_pooled_MSD_stats(pool_1um)
    stats_3um = calc_pooled_MSD_stats(pool_3um)
    stats_5um = calc_pooled_MSD_stats(pool_5um)
    stats_7um = calc_pooled_MSD_stats(pool_7um)
    stats_20um = calc_pooled_MSD_stats(pool_20um)
    
    print(f"  beads06um: {n_tot_06um} -> {n_filt_06um} particles (removed {n_tot_06um - n_filt_06um})")
    print(f"  beads1um:  {n_tot_1um} -> {n_filt_1um} particles (removed {n_tot_1um - n_filt_1um})")
    print(f"  beads3um:  {n_tot_3um} -> {n_filt_3um} particles (removed {n_tot_3um - n_filt_3um})")
    print(f"  beads5um:  {n_tot_5um} -> {n_filt_5um} particles (removed {n_tot_5um - n_filt_5um})")
    print(f"  beads7um:  {n_tot_7um} -> {n_filt_7um} particles (removed {n_tot_7um - n_filt_7um})")
    print(f"  beads20um: {n_tot_20um} -> {n_filt_20um} particles (removed {n_tot_20um - n_filt_20um})")

    beads_data = [
        {"name": "beads06um", "d_um": 0.63, "pool_df": pool_06um, "raw_df": raw_06um, "stats": stats_06um, "n_tot": n_tot_06um, "n_filt": n_filt_06um, "marker": "^", "color": style_colors[0]},
        {"name": "beads1um",  "d_um": 1.18, "pool_df": pool_1um,  "raw_df": raw_1um,  "stats": stats_1um,  "n_tot": n_tot_1um,  "n_filt": n_filt_1um,  "marker": "o", "color": style_colors[1]},
        {"name": "beads3um",  "d_um": 3.37, "pool_df": pool_3um,  "raw_df": raw_3um,  "stats": stats_3um,  "n_tot": n_tot_3um,  "n_filt": n_filt_3um,  "marker": "d", "color": style_colors[2]},
        {"name": "beads5um",  "d_um": 5.00, "pool_df": pool_5um,  "raw_df": raw_5um,  "stats": stats_5um,  "n_tot": n_tot_5um,  "n_filt": n_filt_5um,  "marker": "p", "color": style_colors[3]},
        {"name": "beads7um",  "d_um": 7.24, "pool_df": pool_7um,  "raw_df": raw_7um,  "stats": stats_7um,  "n_tot": n_tot_7um,  "n_filt": n_filt_7um,  "marker": "h", "color": style_colors[4]},
        {"name": "beads20um", "d_um": 20.0, "pool_df": pool_20um, "raw_df": raw_20um, "stats": stats_20um, "n_tot": n_tot_20um, "n_filt": n_filt_20um, "marker": "s", "color": style_colors[5]},
    ]

    # 統計サマリーCSVの出力
    pooled_stats_rows = []
    for item in beads_data:
        st = item["stats"]
        b_name = item["name"]
        d_um = item["d_um"]
        n_filt = item["n_filt"]
        n_tot = item["n_tot"]
        for lag_val, row in st.iterrows():
            pooled_stats_rows.append({
                "bead_name": b_name,
                "diameter_um": d_um,
                "lag_time_s": lag_val,
                "mean_msd_um2": row['mean'],
                "boot_sem_msd_um2": row['boot_sem'],
                "boot_ci_low_msd_um2": row['boot_ci_low'],
                "boot_ci_high_msd_um2": row['boot_ci_high'],
                "q25_msd_um2": row['q25'],
                "q75_msd_um2": row['q75'],
                "q10_msd_um2": row['q10'],
                "q90_msd_um2": row['q90'],
                "median_msd_um2": row['median'],
                "std_msd_um2": row['std'],
                "n_active_particles": int(row['count']),
                "n_filtered_particles": n_filt,
                "n_total_particles": n_tot
            })
    df_pooled_summary = pd.DataFrame(pooled_stats_rows)
    save_csv_to_all(df_pooled_summary, "msd_pooled_mean_iqr_summary.csv", out_dirs)

    # ---------------------------------------------------------
    # 図1: MSD プロット (代表値: Ensemble Mean, エラーバー & エラー帯: ブートストラップ法)
    # ---------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(8.5, 6.2))
    for item in beads_data:
        st = item["stats"]
        color = item["color"]
        marker = item["marker"]
        d_um = item["d_um"]
        n_filt = item["n_filt"]
        
        valid = st['count'] >= 2
        lags = st.index[valid].to_numpy(dtype=float)
        mean_v = st['mean'][valid].to_numpy(dtype=float)
        boot_sem = st['boot_sem'][valid].to_numpy(dtype=float)
        boot_ci_low = st['boot_ci_low'][valid].to_numpy(dtype=float)
        boot_ci_high = st['boot_ci_high'][valid].to_numpy(dtype=float)
        
        # 1. ブートストラップ 95% 信頼区間帯 (Bootstrap 95% CI band)
        ax1.fill_between(
            lags,
            np.clip(boot_ci_low, 1e-4, None),
            boot_ci_high,
            color=color,
            alpha=0.18,
            edgecolor='none'
        )

        # 2. ブートストラップ 95% 信頼区間によるエラーバー (Bootstrap 95% CI Error Bars)
        # エラーバーの長さを淡色シェード帯と同じ 95% 信頼区間に統一 ([mean - ci_low, ci_high - mean])
        # 両対数プロットで点同士が密集して見づらくなるのを防ぐため、対数等間隔に配置される代表点にクッキリとエラーバーを描画
        n_pts = len(lags)
        if n_pts > 20:
            target_lags = np.geomspace(lags[0], lags[-1], num=18)
            sub_idx = np.unique([np.argmin(np.abs(lags - tg)) for tg in target_lags])
        else:
            sub_idx = np.arange(n_pts)

        yerr_ci_rep = [
            np.clip(mean_v[sub_idx] - boot_ci_low[sub_idx], 0, None),
            np.clip(boot_ci_high[sub_idx] - mean_v[sub_idx], 0, None)
        ]

        ax1.errorbar(
            lags[sub_idx],
            mean_v[sub_idx],
            yerr=yerr_ci_rep,
            fmt='none',
            ecolor=color,
            elinewidth=1.4,
            capsize=3.5,
            capthick=1.0,
            alpha=0.85,
            zorder=3
        )

        # 全点にも細いエラーバーを薄く付与
        yerr_ci_all = [
            np.clip(mean_v - boot_ci_low, 0, None),
            np.clip(boot_ci_high - mean_v, 0, None)
        ]
        ax1.errorbar(
            lags,
            mean_v,
            yerr=yerr_ci_all,
            fmt='none',
            ecolor=color,
            elinewidth=0.7,
            capsize=0,
            alpha=0.35,
            zorder=2
        )

        # 3. 主線: アンサンブル平均 Mean (実線 + マーカー)
        ax1.plot(lags, mean_v, marker=marker, linestyle='-', linewidth=2.0, markersize=7.5,
                 label=f'{d_um:.2f} \u03bcm ($N={n_filt}$)', color=color, zorder=4)

    # ガイド線
    t_g1, t_g2 = 80.0, 200.0
    ax1.plot([t_g1, t_g2], [3.0 * (t_g1/100.0)**1.0, 3.0 * (t_g2/100.0)**1.0], color='#333333', linestyle='--')
    ax1.text(130, 2.3, r'$\propto \Delta t^{1.0}$', fontsize=12)
    ax1.plot([t_g1, t_g2], [30.0 * (t_g1/100.0)**2.0, 30.0 * (t_g2/100.0)**2.0], color='#333333', linestyle='--')
    ax1.text(70, 80, r'$\propto \Delta t^{2.0}$', fontsize=12)

    ax1.legend(fontsize=10.0, loc='upper left', framealpha=0.88, markerscale=0.8, handlelength=1.4, borderpad=0.35, labelspacing=0.28)
    ax1.set(
        xlim=(4e-0, 1000),
        ylim=(1e-2, 1e4),
        xscale='log',
        yscale='log',
        xlabel='Lag time $\\Delta t$ [s]',
        ylabel='MSD $\\langle \\Delta \\boldsymbol{r}^2 \\rangle$ [$\\mu\\mathrm{m}^2$]',
        title='Ensemble Mean MSD with Bootstrap Error Bars (Filtered: $\\alpha > 0.5$)'
    )
    save_figure_to_all(fig1, "MSD", out_dirs)

    # ---------------------------------------------------------
    # 2. MSD(300 s) および MSD(300s~1000s の傾き alpha) の算出
    # ---------------------------------------------------------
    print("\n--- 2. Calculating MSD(300s) and Slope alpha (300s-1000s) vs Cargo Diameter ---")
    summary_rows = []
    
    for item in beads_data:
        d_um = item["d_um"]
        b_name = item["name"]
        pool_df = item["pool_df"]
        n_filt = item["n_filt"]
        n_tot = item["n_tot"]
        
        # MSD(300s)
        msd_300_res = extract_msd_at_lag(pool_df, target_t=300.0, tol=12.0)
        
        # 傾き alpha (300s~1000s, 長時間粒子 >= 1000s のみ)
        slope_res = calc_msd_slope_range(pool_df, min_t=300.0, max_t=1000.0, min_total_track_length=0.0, min_points=3)
        
        summary_rows.append({
            "bead_name": b_name,
            "diameter_um": d_um,
            "n_filtered_particles": n_filt,
            "n_total_particles": n_tot,
            # MSD(300s)
            "msd_300s_mean_um2": msd_300_res["mean"],
            "msd_300s_sem_um2": msd_300_res["sem"],
            "msd_300s_boot_sem_um2": msd_300_res["boot_sem"],
            "msd_300s_boot_ci_low_um2": msd_300_res["boot_ci_low"],
            "msd_300s_boot_ci_high_um2": msd_300_res["boot_ci_high"],
            "msd_300s_std_um2": msd_300_res["std"],
            "msd_300s_median_um2": msd_300_res["median"],
            "msd_300s_q25_um2": msd_300_res["q25"],
            "msd_300s_q75_um2": msd_300_res["q75"],
            "msd_300s_n_particles": msd_300_res["n"],
            # Slope alpha (300s~1000s, tracks >= 1000s)
            "alpha_300_1000s_mean": slope_res["mean"],
            "alpha_300_1000s_sem": slope_res["sem"],
            "alpha_300_1000s_boot_sem": slope_res["boot_sem"],
            "alpha_300_1000s_boot_ci_low": slope_res["boot_ci_low"],
            "alpha_300_1000s_boot_ci_high": slope_res["boot_ci_high"],
            "alpha_300_1000s_std": slope_res["std"],
            "alpha_300_1000s_median": slope_res["median"],
            "alpha_300_1000s_q25": slope_res["q25"],
            "alpha_300_1000s_q75": slope_res["q75"],
            "alpha_300_1000s_n_long_particles": slope_res["n"]
        })
        
        n_long = slope_res["n"]
        print(f"  {b_name} ({d_um} um): MSD(300s) = {msd_300_res['mean']:.2f} \u00b1 {msd_300_res['boot_sem']:.2f} um^2 (Bootstrap SEM, N={msd_300_res['n']}) | alpha(300-1000s) = {slope_res['mean']:.2f} \u00b1 {slope_res['boot_sem']:.2f} (N_long={n_long})")

    df_msd300_slope = pd.DataFrame(summary_rows)
    save_csv_to_all(df_msd300_slope, "msd300_and_slope_summary.csv", out_dirs)

    # ---------------------------------------------------------
    # 3. 2軸プロット: 第1軸 MSD(300s), 第2軸 MSDの傾き (300s-1000s, tracks >= 1000s)
    # ---------------------------------------------------------
    fig2, ax_msd = plt.subplots(figsize=(8.2, 5.8))
    ax_slope = ax_msd.twinx()

    d_vals = df_msd300_slope["diameter_um"].to_numpy(dtype=float)
    msd300_mean = df_msd300_slope["msd_300s_mean_um2"].to_numpy(dtype=float)
    msd300_ci_low = df_msd300_slope["msd_300s_boot_ci_low_um2"].to_numpy(dtype=float)
    msd300_ci_high = df_msd300_slope["msd_300s_boot_ci_high_um2"].to_numpy(dtype=float)
    msd300_yerr = [
        np.clip(msd300_mean - msd300_ci_low, 0, None),
        np.clip(msd300_ci_high - msd300_mean, 0, None)
    ]
    
    alpha_mean = df_msd300_slope["alpha_300_1000s_mean"].to_numpy(dtype=float)
    alpha_ci_low = df_msd300_slope["alpha_300_1000s_boot_ci_low"].to_numpy(dtype=float)
    alpha_ci_high = df_msd300_slope["alpha_300_1000s_boot_ci_high"].to_numpy(dtype=float)
    alpha_yerr = [
        np.clip(alpha_mean - alpha_ci_low, 0, None),
        np.clip(alpha_ci_high - alpha_mean, 0, None)
    ]
    n_long_vals = df_msd300_slope["alpha_300_1000s_n_long_particles"].to_numpy(dtype=int)

    # 第1軸 (左軸): MSD(300 s) [um^2] (log scale, 青系)
    color_msd = '#1f77b4'
    line1 = ax_msd.errorbar(
        d_vals, msd300_mean, yerr=msd300_yerr,
        fmt='o-', color=color_msd, ecolor=color_msd, elinewidth=1.6,
        capsize=4.5, capthick=1.2, markersize=8.5, linewidth=2.0,
        label=r'$\mathrm{MSD}(\Delta t = 300\,\mathrm{s})$ (Bootstrap 95% CI)',
        zorder=4
    )

    # 第2軸 (右軸): MSD 傾き alpha (300s-1000s, tracks >= 1000s) (linear scale, 赤系)
    color_slope = '#d62728'
    line2 = ax_slope.errorbar(
        d_vals, alpha_mean, yerr=alpha_yerr,
        fmt='s--', color=color_slope, ecolor=color_slope, elinewidth=1.6,
        capsize=4.5, capthick=1.2, markersize=8.0, linewidth=2.0,
        label=r'Slope $\alpha$ ($\Delta t \in [300, 1000]\,\mathrm{s}$, Bootstrap 95% CI)',
        zorder=5
    )

    # 各データ点に長時間追跡粒子数 (N_long) の注釈を表示
    for x_d, y_a, n_l in zip(d_vals, alpha_mean, n_long_vals):
        if np.isfinite(y_a):
            ax_slope.annotate(
                f"$N={n_l}$",
                (x_d, y_a),
                textcoords="offset points",
                xytext=(0, 10),
                ha='center',
                fontsize=9.0,
                fontweight='bold',
                color=color_slope,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=color_slope, alpha=0.85, linewidth=0.8),
                zorder=6
            )

    # 傾きの基準線 (alpha = 1.0, alpha = 2.0)
    ax_slope.axhline(1.0, color='#999999', linestyle=':', linewidth=1.2, zorder=2)
    ax_slope.axhline(2.0, color='#999999', linestyle=':', linewidth=1.2, zorder=2)

    # 軸設定 (横軸)
    ax_msd.set_xscale('log')
    ax_msd.set_xlim(0.4, 28.0)
    ax_msd.xaxis.set_major_locator(ticker.FixedLocator([0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0]))
    ax_msd.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
    ax_msd.set_xlabel(r'Cargo Diameter $2R_c$ [$\mu\mathrm{m}$]', fontsize=12, fontweight='bold')

    # 第1軸設定 (左軸)
    ax_msd.set_yscale('log')
    ax_msd.set_ylim(0.05, 5000.0)
    ax_msd.set_ylabel(r'MSD $\langle \Delta r^2(\Delta t = 300\,\mathrm{s}) \rangle$ [$\mu\mathrm{m}^2$]', 
                      fontsize=12, fontweight='bold', color=color_msd)
    ax_msd.tick_params(axis='y', labelcolor=color_msd)

    # 第2軸設定 (右軸)
    ax_slope.set_ylim(0.0, 2.3)
    ax_slope.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax_slope.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax_slope.set_ylabel(r'MSD Slope $\alpha$ ($\Delta t \in [300, 1000]\,\mathrm{s}$, tracks $\geq 1000\,\mathrm{s}$)', 
                        fontsize=12, fontweight='bold', color=color_slope)
    ax_slope.tick_params(axis='y', labelcolor=color_slope)

    ax_msd.grid(True, which='both', linestyle='--', alpha=0.35)

    # 凡例を1つに統合
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax_msd.legend(lines, labels, loc='upper right', framealpha=0.92, fontsize=9.5)

    ax_msd.set_title(r'MSD($\Delta t=300\,\mathrm{s}$) and Late-Stage Slope $\alpha$ vs Cargo Diameter', 
                     fontsize=12, fontweight='bold', pad=10)

    save_figure_to_all(fig2, "msd300_and_slope_vs_diameter", out_dirs)

    print("\n--- Completed all fits and figure outputs successfully! ---")


if __name__ == "__main__":
    main()