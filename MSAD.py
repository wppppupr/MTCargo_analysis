#!/usr/bin/env python3
"""
MSAD.py

カーゴ微粒子の平均二乗角度変位 (Mean Squared Angular Displacement: MSAD) 解析スクリプト。
MSD.py の設計思想・可視化スタイル（マーカー・カラーパレット・エラー帯・軸フォーマット）に準拠。

- 各軌跡の瞬時進行方向角 theta = arctan2(dy, dx) を連続セグメント上でアンラップ (unwrap)
- 個別粒子 MSAD (iMSAD) の算出とプール (MSD の alpha > 0.5 によるアクティブ粒子フィルタリング)
- アンサンブル平均 MSAD および四分位範囲 (IQR: 25-75% & 10-90% タイル帯) の算出・描画
- MSAD(300 s) および回転拡散パラメータの集計 CSV 出力
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

# スタイルの適用 (MSD.py と完全に統一)
style_path = Path(__file__).parent / 'libs' / 'my_style.mplstyle'
if style_path.exists():
    plt.style.use(str(style_path))
    style_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
else:
    style_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

# データルートディレクトリの候補 (MSD.py と統一)
POSSIBLE_ROOTS = [
    Path('/Volumes/data-1/Sasaki/MTsingleBeads'),
    Path('/Volumes/data-1/sasaki/MTsingleBeads'),
    Path('/Volumes/data/Sasaki/MTsingleBeads'),
    Path('/mnt/NAS-Ebanaru/sasaki/MTsingleBeads'),
    Path('/mnt/NAS-Ebanaru/Sasaki/MTsingleBeads'),
    Path('/Volumes/data/Sasaki/MTSingleBeads'),
]

BEADS_INFO = [
    {"name": "beads06um", "diameter_um": 0.63, "marker": "^", "intervals": [4, 4, 4, 4]},
    {"name": "beads1um",  "diameter_um": 1.18, "marker": "o", "intervals": [4, 4, 4, 4]},
    {"name": "beads3um",  "diameter_um": 3.37, "marker": "d", "intervals": [4, 4, 4, 4]},
    {"name": "beads5um",  "diameter_um": 5.00, "marker": "p", "intervals": [4, 4, 4, 4]},
    {"name": "beads7um",  "diameter_um": 7.24, "marker": "h", "intervals": [4, 4, 4]},
    {"name": "beads20um", "diameter_um": 20.0, "marker": "s", "intervals": [4, 4, 4]},
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


def calc_particle_alpha(sub_df: pd.DataFrame, min_t: float = 4.0, max_t: float = 300.0) -> float:
    """
    個別粒子MSD (iMSD) に対して対数空間でべき乗則 MSD ~ t^alpha をフィッティングし、
    異常拡散指数 alpha を算出する (MSD.py と完全に同一)。
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


def concat_pooled_particles_MSAD(
    folder: Path,
    interval_list: List[float],
    scale: float = 0.11,
    alpha_threshold: float = 0.5,
    max_tau_steps: int = 250
) -> Tuple[pd.DataFrame, pd.DataFrame, int, int]:
    """
    指定ビーズフォルダ内の全動画から個別粒子の MSAD (iMSAD) を計算し、1つのプールに統合する。
    各粒子について iMSD による異常拡散指数 alpha を算出し、
    MSD.py と同様に alpha > alpha_threshold (デフォルト: 0.5) の粒子のみを抽出する。

    Returns:
    --------
    df_filtered : pd.DataFrame
        alpha > alpha_threshold でフィルタリングされた全粒子の iMSAD
    df_pool : pd.DataFrame
        フィルタリング前の全粒子 iMSAD
    n_total : int
        総粒子数
    n_filtered : int
        フィルタ通過粒子数
    """
    file_list = sorted(glob.glob(str(Path(folder) / "*" / "*" / "beads_tracks.csv")))
    all_imsads = []

    for vid_idx, file_path in enumerate(file_list):
        interval = interval_list[vid_idx] if vid_idx < len(interval_list) else interval_list[-1]
        df_raw = pd.read_csv(file_path)

        for pid, grp in df_raw.groupby('particle'):
            grp = grp.sort_values(by='frame')
            frames = grp['frame'].to_numpy()
            x = grp['x'].to_numpy() * scale
            y = grp['y'].to_numpy() * scale
            n_pts = len(frames)
            if n_pts < 3:
                continue

            unique_pid = f"vid{vid_idx}_{pid}"

            # 1. iMSD を計算して粒子 alpha を推定
            pos = np.column_stack([x, y])
            msd_vals = []
            lag_times = []
            for tau in range(1, n_pts):
                disp = pos[tau:] - pos[:-tau]
                msd_vals.append(np.mean(np.sum(disp**2, axis=1)))
                lag_times.append(tau * interval)
            df_imsd = pd.DataFrame({'lag time': lag_times, 'MSD': msd_vals})
            alpha_val = calc_particle_alpha(df_imsd, 4.0, 300.0)

            # 2. 連続フレームセグメントごとの角度アンラップ MSAD 計算
            diff_f = np.diff(frames)
            split_idx = np.where(diff_f != 1)[0] + 1
            starts = np.concatenate([[0], split_idx])
            ends = np.concatenate([split_idx, [n_pts]])

            dtheta2_by_tau = {}
            for s, e in zip(starts, ends):
                seg_len = e - s
                if seg_len < 3:
                    continue
                seg_x = x[s:e]
                seg_y = y[s:e]
                # 瞬時進行方向角 theta = arctan2(dy, dx)
                dx_inst = np.diff(seg_x)
                dy_inst = np.diff(seg_y)
                th = np.arctan2(dy_inst, dx_inst)
                unwrapped_th = np.unwrap(th)
                n_angles = len(unwrapped_th)

                for tau in range(1, min(max_tau_steps + 1, n_angles)):
                    dth = unwrapped_th[tau:] - unwrapped_th[:-tau]
                    if tau not in dtheta2_by_tau:
                        dtheta2_by_tau[tau] = []
                    dtheta2_by_tau[tau].extend((dth**2).tolist())

            if not dtheta2_by_tau:
                continue

            # 粒子内時間平均 MSAD の作成
            for tau, vals in dtheta2_by_tau.items():
                mean_msad_rad2 = float(np.mean(vals))
                all_imsads.append({
                    'unique_particle_id': unique_pid,
                    'video_idx': vid_idx,
                    'particle': pid,
                    'tau_step': tau,
                    'lag time': tau * interval,
                    'MSAD': mean_msad_rad2,
                    'MSAD_deg2': mean_msad_rad2 * ((180.0 / np.pi)**2),
                    'alpha': alpha_val
                })

    if len(all_imsads) == 0:
        empty_df = pd.DataFrame(columns=['unique_particle_id', 'video_idx', 'particle', 'lag time', 'MSAD', 'alpha'])
        return empty_df, empty_df, 0, 0

    df_pool = pd.DataFrame(all_imsads)
    df_pool['MSAD'] = pd.to_numeric(df_pool['MSAD'], errors='coerce')
    df_pool['lag time'] = pd.to_numeric(df_pool['lag time'], errors='coerce')
    df_pool = df_pool.dropna(subset=['MSAD', 'lag time'])

    n_total = df_pool['unique_particle_id'].nunique()
    df_filtered = df_pool[df_pool['alpha'] > alpha_threshold].copy()
    n_filtered = df_filtered['unique_particle_id'].nunique()

    return df_filtered, df_pool, n_total, n_filtered


def calc_pooled_MSAD_stats(df_pool: pd.DataFrame, min_particles: int = 1) -> pd.DataFrame:
    """
    プールされた全粒子MSADから、lag timeごとのアンサンブル平均値 (mean)、
    四分位範囲 IQR (25%, 75%タイル)、10%, 90%タイル、標準偏差、粒子数を算出する (MSD.py準拠)。
    """
    if df_pool.empty:
        return pd.DataFrame(columns=['mean', 'q25', 'q75', 'q10', 'q90', 'median', 'std', 'sem', 'count'])
    grouped = df_pool.groupby('lag time')['MSAD']
    mean_series = grouped.mean()
    std_series = grouped.std()
    count_series = grouped.count()
    sem_series = std_series / np.sqrt(count_series)

    stats_df = pd.DataFrame({
        'mean': mean_series,
        'q25': grouped.quantile(0.25),
        'q75': grouped.quantile(0.75),
        'q10': grouped.quantile(0.10),
        'q90': grouped.quantile(0.90),
        'median': grouped.median(),
        'std': std_series,
        'sem': sem_series,
        'count': count_series
    })
    if min_particles > 1:
        stats_df = stats_df[stats_df['count'] >= min_particles]
    return stats_df


def extract_msad_at_lag(df_pool: pd.DataFrame, target_t: float = 300.0, tol: float = 12.0) -> dict:
    """
    個別粒子データから target_t 近傍の MSAD 値を集計し、統計量を返す。
    """
    records = []
    for pid, grp in df_pool.groupby('unique_particle_id'):
        sub = grp[(grp['lag time'] >= target_t - tol) & (grp['lag time'] <= target_t + tol) & (grp['MSAD'] > 0)]
        if not sub.empty:
            best_row = sub.iloc[(sub['lag time'] - target_t).abs().argsort()[:1]]
            records.append(float(best_row['MSAD'].values[0]))
    if len(records) > 0:
        arr = np.array(records, dtype=float)
        mean_val = float(np.mean(arr))
        std_val = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        sem_val = float(std_val / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
        return {
            'mean': mean_val,
            'std': std_val,
            'sem': sem_val,
            'median': float(np.median(arr)),
            'q25': float(np.percentile(arr, 25)),
            'q75': float(np.percentile(arr, 75)),
            'n': len(arr)
        }
    return {'mean': np.nan, 'std': np.nan, 'sem': np.nan, 'median': np.nan, 'q25': np.nan, 'q75': np.nan, 'n': 0}


def calc_msad_slope_range(
    df_pool: pd.DataFrame,
    min_t: float = 4.0,
    max_t: float = 300.0,
    min_points: int = 3
) -> dict:
    """
    個別粒子それぞれの MSAD に対して指定ラグ時間範囲でのべき乗則傾き alpha_theta (MSAD ~ t^alpha_theta) を算出する。
    """
    part_slopes = []
    part_ids = []

    for pid, grp in df_pool.groupby('unique_particle_id'):
        mask = (grp['lag time'] >= min_t) & (grp['lag time'] <= max_t) & (grp['MSAD'] > 0)
        if np.sum(mask) >= min_points:
            dt = grp['lag time'][mask].to_numpy(dtype=float)
            msad = grp['MSAD'][mask].to_numpy(dtype=float)
            slope, _ = np.polyfit(np.log10(dt), np.log10(msad), 1)
            part_slopes.append(slope)
            part_ids.append(pid)

    if len(part_slopes) > 0:
        arr = np.array(part_slopes, dtype=float)
        mean_val = float(np.mean(arr))
        std_val = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        sem_val = float(std_val / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
        return {
            'mean': mean_val,
            'std': std_val,
            'sem': sem_val,
            'median': float(np.median(arr)),
            'q25': float(np.percentile(arr, 25)),
            'q75': float(np.percentile(arr, 75)),
            'n': len(arr),
            'particle_ids': part_ids
        }
    return {
        'mean': np.nan, 'std': np.nan, 'sem': np.nan,
        'median': np.nan, 'q25': np.nan, 'q75': np.nan,
        'n': 0, 'particle_ids': []
    }


def fit_msad_diffusion(
    t_data: np.ndarray,
    y_data: np.ndarray,
    min_t: float = 4.0,
    max_t: float = 40.0,
    min_points: int = 3
) -> Tuple[float, float, float, float, float, int]:
    """
    MSAD データに対してオフセット付き理論式 MSAD = 2 * D_r * \Delta t + C を線形フィッティングし、
    回転拡散係数 D_r [rad^2/s], D_r の標準誤差, 切片 C [rad^2], C の標準誤差, 決定係数 R^2, 点数を返す。
    y = m * t + C (m = 2 * D_r  =>  D_r = m / 2)
    """
    mask = (t_data >= min_t - 1e-6) & (t_data <= max_t + 1e-6) & (y_data > 0) & np.isfinite(y_data)
    if np.sum(mask) < min_points:
        return np.nan, np.nan, np.nan, np.nan, np.nan, int(np.sum(mask))

    t = t_data[mask]
    y = y_data[mask]

    p, cov = np.polyfit(t, y, 1, cov=True)
    m, c = float(p[0]), float(p[1])
    dr = m / 2.0
    dr_err = float(np.sqrt(cov[0, 0]) / 2.0) if np.isfinite(cov[0, 0]) else 0.0
    c_err = float(np.sqrt(cov[1, 1])) if np.isfinite(cov[1, 1]) else 0.0

    residuals = y - (m * t + c)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_sq = float(1.0 - np.sum(residuals**2) / ss_tot) if ss_tot > 0 else np.nan

    return dr, dr_err, c, c_err, r_sq, len(t)


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


def main():
    root_dir = find_default_root()
    workspace_dir = Path(__file__).parent.resolve()

    out_dirs = [
        workspace_dir / "figure" / "msad",
        workspace_dir / "figure" / "msd",
        root_dir / "figure" / "msad",
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

    print("\n--- 1. Loading & Computing MSAD datasets (alpha > 0.5 filtered) ---")
    alpha_thresh = 0.5

    beads_loaded = []
    for idx, b_conf in enumerate(BEADS_INFO):
        b_name = b_conf["name"]
        d_um = b_conf["diameter_um"]
        marker = b_conf["marker"]
        color = style_colors[idx % len(style_colors)]
        intervals = b_conf["intervals"]

        pool_df, raw_df, n_tot, n_filt = concat_pooled_particles_MSAD(
            root_dir / b_name,
            intervals,
            scale=0.11,
            alpha_threshold=alpha_thresh,
            max_tau_steps=250
        )
        stats = calc_pooled_MSAD_stats(pool_df)

        print(f"  {b_name} ({d_um:.2f} \u03bcm): {n_tot} -> {n_filt} active particles (MSAD curve points: {len(stats)})")

        beads_loaded.append({
            "name": b_name,
            "d_um": d_um,
            "pool_df": pool_df,
            "raw_df": raw_df,
            "stats": stats,
            "n_tot": n_tot,
            "n_filt": n_filt,
            "marker": marker,
            "color": color
        })

    # ---------------------------------------------------------
    # 統計サマリーCSVの出力 (MSD.py 準拠)
    # ---------------------------------------------------------
    pooled_stats_rows = []
    for item in beads_loaded:
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
                "mean_msad_rad2": row['mean'],
                "mean_msad_deg2": row['mean'] * ((180.0 / np.pi)**2),
                "q25_msad_rad2": row['q25'],
                "q75_msad_rad2": row['q75'],
                "q10_msad_rad2": row['q10'],
                "q90_msad_rad2": row['q90'],
                "median_msad_rad2": row['median'],
                "std_msad_rad2": row['std'],
                "sem_msad_rad2": row['sem'],
                "n_active_particles": int(row['count']),
                "n_filtered_particles": n_filt,
                "n_total_particles": n_tot
            })
    df_pooled_summary = pd.DataFrame(pooled_stats_rows)
    save_csv_to_all(df_pooled_summary, "msad_pooled_mean_iqr_summary.csv", out_dirs)

    # ---------------------------------------------------------
    # 図1: MSAD プロット (代表値: Ensemble Mean, エラー帯: 25-75% IQR & 10-90%)
    # MSD.py と完全に同一の美しい見た目 (カラー・マーカー・フォント・帯透明度)
    # ---------------------------------------------------------
    print("\n--- 2. Generating MSAD Ensemble Plot (rad^2) ---")
    fig1, ax1 = plt.subplots(figsize=(8.5, 6.2))
    for item in beads_loaded:
        st = item["stats"]
        color = item["color"]
        marker = item["marker"]
        d_um = item["d_um"]
        n_filt = item["n_filt"]

        valid = st['count'] >= 2
        lags = st.index[valid]
        mean_v = st['mean'][valid]
        q25 = st['q25'][valid]
        q75 = st['q75'][valid]
        q10 = st['q10'][valid]
        q90 = st['q90'][valid]

        # 10% - 90% タイル帯 (淡色)
        ax1.fill_between(lags, q10, q90, color=color, alpha=0.08, edgecolor='none')
        # 25% - 75% 四分位範囲 (IQR) 帯 (中間色)
        ax1.fill_between(lags, q25, q75, color=color, alpha=0.22, edgecolor='none')
        # 主線: アンサンブル平均 Mean (実線 + マーカー)
        ax1.plot(lags, mean_v, marker=marker, linestyle='-', linewidth=2.0, markersize=7.5,
                 label=f'{d_um:.2f} \u03bcm ($N={n_filt}$)', color=color)

    # ガイド線: 拡散的傾き \propto \Delta t^{1.0}
    t_g1, t_g2 = 50.0, 200.0
    ax1.plot([t_g1, t_g2], [18.0 * (t_g1 / 50.0)**1.0, 18.0 * (t_g2 / 50.0)**1.0], color='#333333', linestyle='--', linewidth=1.6)
    ax1.text(140, 30, r'$\propto \Delta t^{1.0}$', fontsize=12, color='#333333')

    ax1.legend(fontsize=10.0, loc='upper left', framealpha=0.88, markerscale=0.8, handlelength=1.4, borderpad=0.35, labelspacing=0.28)
    ax1.set(
        xlim=(4e-0, 1000),
        ylim=(1.0, 2e3),
        xscale='log',
        yscale='log',
        xlabel=r'Lag time $\Delta t$ [s]',
        ylabel=r'MSAD $\langle (\Delta \theta)^2 \rangle$ [$\mathrm{rad}^2$]',
        title=r'Ensemble Mean MSAD $\pm$ IQR (Filtered: $\alpha > 0.5$)'
    )
    save_figure_to_all(fig1, "MSAD", out_dirs)

    # 度 (deg^2) 単位版の図も保存
    fig1_deg, ax1_deg = plt.subplots(figsize=(8.5, 6.2))
    rad2deg_sq = (180.0 / np.pi)**2
    for item in beads_loaded:
        st = item["stats"]
        color = item["color"]
        marker = item["marker"]
        d_um = item["d_um"]
        n_filt = item["n_filt"]

        valid = st['count'] >= 2
        lags = st.index[valid]
        mean_v = st['mean'][valid] * rad2deg_sq
        q25 = st['q25'][valid] * rad2deg_sq
        q75 = st['q75'][valid] * rad2deg_sq
        q10 = st['q10'][valid] * rad2deg_sq
        q90 = st['q90'][valid] * rad2deg_sq

        ax1_deg.fill_between(lags, q10, q90, color=color, alpha=0.08, edgecolor='none')
        ax1_deg.fill_between(lags, q25, q75, color=color, alpha=0.22, edgecolor='none')
        ax1_deg.plot(lags, mean_v, marker=marker, linestyle='-', linewidth=2.0, markersize=7.5,
                     label=f'{d_um:.2f} \u03bcm ($N={n_filt}$)', color=color)

    ax1_deg.plot([t_g1, t_g2], [30.0 * rad2deg_sq * (t_g1 / 60.0)**1.0, 30.0 * rad2deg_sq * (t_g2 / 60.0)**1.0],
                 color='#333333', linestyle='--', linewidth=1.5)
    ax1_deg.text(120, 42 * rad2deg_sq, r'$\propto \Delta t^{1.0}$', fontsize=13, color='#333333')

    ax1_deg.legend(fontsize=10.0, loc='upper left', framealpha=0.88, markerscale=0.8, handlelength=1.4, borderpad=0.35, labelspacing=0.28)
    ax1_deg.set(
        xlim=(4e-0, 1000),
        ylim=(3e3, 6e6),
        xscale='log',
        yscale='log',
        xlabel=r'Lag time $\Delta t$ [s]',
        ylabel=r'MSAD $\langle (\Delta \theta)^2 \rangle$ [$\mathrm{deg}^2$]',
        title=r'Ensemble Mean MSAD $\pm$ IQR (Filtered: $\alpha > 0.5$)'
    )
    save_figure_to_all(fig1_deg, "MSAD_deg2", out_dirs)

    # ---------------------------------------------------------
    # 3. MSAD(300 s) および べき乗則傾き alpha_theta (4s~300s) の集計
    # ---------------------------------------------------------
    print("\n--- 3. Calculating MSAD(300s) and Slope alpha_theta vs Cargo Diameter ---")
    summary_rows = []

    for item in beads_loaded:
        d_um = item["d_um"]
        b_name = item["name"]
        pool_df = item["pool_df"]
        n_filt = item["n_filt"]
        n_tot = item["n_tot"]

        msad_300_res = extract_msad_at_lag(pool_df, target_t=300.0, tol=12.0)
        slope_res = calc_msad_slope_range(pool_df, min_t=4.0, max_t=300.0, min_points=3)

        summary_rows.append({
            "bead_name": b_name,
            "diameter_um": d_um,
            "radius_um": d_um / 2.0,
            "n_filtered_particles": n_filt,
            "n_total_particles": n_tot,
            # MSAD(300s) rad^2
            "msad_300s_mean_rad2": msad_300_res["mean"],
            "msad_300s_sem_rad2": msad_300_res["sem"],
            "msad_300s_std_rad2": msad_300_res["std"],
            "msad_300s_median_rad2": msad_300_res["median"],
            "msad_300s_q25_rad2": msad_300_res["q25"],
            "msad_300s_q75_rad2": msad_300_res["q75"],
            "msad_300s_n_particles": msad_300_res["n"],
            # MSAD(300s) deg^2
            "msad_300s_mean_deg2": msad_300_res["mean"] * rad2deg_sq if np.isfinite(msad_300_res["mean"]) else np.nan,
            "msad_300s_sem_deg2": msad_300_res["sem"] * rad2deg_sq if np.isfinite(msad_300_res["sem"]) else np.nan,
            # Slope alpha_theta (4s~300s)
            "alpha_theta_4_300s_mean": slope_res["mean"],
            "alpha_theta_4_300s_sem": slope_res["sem"],
            "alpha_theta_4_300s_std": slope_res["std"],
            "alpha_theta_4_300s_median": slope_res["median"],
            "alpha_theta_4_300s_q25": slope_res["q25"],
            "alpha_theta_4_300s_q75": slope_res["q75"],
            "alpha_theta_4_300s_n_particles": slope_res["n"]
        })

        print(f"  {b_name} ({d_um} um): MSAD(300s) = {msad_300_res['mean']:.2f} \u00b1 {msad_300_res['sem']:.2f} rad^2 (N={msad_300_res['n']}) | alpha_theta = {slope_res['mean']:.2f} \u00b1 {slope_res['sem']:.2f} (N={slope_res['n']})")

    df_msad300_slope = pd.DataFrame(summary_rows)
    save_csv_to_all(df_msad300_slope, "msad300_and_slope_summary.csv", out_dirs)

    # ---------------------------------------------------------
    # 4. MSAD を 2 * D_r * \Delta t + C で線形フィッティング & D_r vs 貨物半径 R_c
    # ---------------------------------------------------------
    print("\n--- 4. Fitting MSAD = 2 * Dr * Delta_t + C & Plotting Dr vs Cargo Radius Rc ---")
    dr_fit_rows = []

    # フィッティング範囲: 短時間拡散領域 (4s~40s, 4s~100s, 4s~300s)
    fit_ranges = [(4.0, 40.0), (4.0, 100.0), (4.0, 300.0)]

    for item in beads_loaded:
        b_name = item["name"]
        d_um = item["d_um"]
        rc_um = d_um / 2.0
        marker = item["marker"]
        color = item["color"]
        st = item["stats"]
        pool_df = item["pool_df"]
        n_filt = item["n_filt"]

        t_all = st.index.to_numpy(dtype=float)
        y_all = st['mean'].to_numpy(dtype=float)

        # 各粒子ごとの Dr, C も集計して粒子間バラツキ (SEM) を算出
        part_drs_dict = {f"{int(mn)}_{int(mx)}s": [] for mn, mx in fit_ranges}
        part_c_dict = {f"{int(mn)}_{int(mx)}s": [] for mn, mx in fit_ranges}
        for pid, grp in pool_df.groupby('unique_particle_id'):
            t_p = grp['lag time'].to_numpy(dtype=float)
            y_p = grp['MSAD'].to_numpy(dtype=float)
            for mn, mx in fit_ranges:
                p_dr, _, p_c, _, _, n_pts = fit_msad_diffusion(t_p, y_p, min_t=mn, max_t=mx, min_points=3)
                if np.isfinite(p_dr):
                    part_drs_dict[f"{int(mn)}_{int(mx)}s"].append(p_dr)
                    part_c_dict[f"{int(mn)}_{int(mx)}s"].append(p_c)

        row_dict = {
            "bead_name": b_name,
            "diameter_um": d_um,
            "radius_um": rc_um,
            "marker": marker,
            "color": color,
            "n_filtered_particles": n_filt
        }

        for mn, mx in fit_ranges:
            rng_key = f"{int(mn)}_{int(mx)}s"
            dr_ens, dr_err_fit, c_ens, c_err_fit, r_sq, n_fit = fit_msad_diffusion(t_all, y_all, min_t=mn, max_t=mx, min_points=3)
            p_arr = np.array(part_drs_dict[rng_key]) if len(part_drs_dict[rng_key]) > 0 else np.array([])
            p_mean = float(np.mean(p_arr)) if len(p_arr) > 0 else np.nan
            p_sem = float(np.std(p_arr, ddof=1) / np.sqrt(len(p_arr))) if len(p_arr) > 1 else 0.0

            p_c_arr = np.array(part_c_dict[rng_key]) if len(part_c_dict[rng_key]) > 0 else np.array([])
            p_c_mean = float(np.mean(p_c_arr)) if len(p_c_arr) > 0 else np.nan
            p_c_sem = float(np.std(p_c_arr, ddof=1) / np.sqrt(len(p_c_arr))) if len(p_c_arr) > 1 else 0.0

            row_dict[f"Dr_{rng_key}_ens_rad2_s"] = dr_ens
            row_dict[f"Dr_{rng_key}_ens_fit_err_rad2_s"] = dr_err_fit
            row_dict[f"C_{rng_key}_ens_rad2"] = c_ens
            row_dict[f"C_{rng_key}_ens_fit_err_rad2"] = c_err_fit
            row_dict[f"Dr_{rng_key}_ens_r2"] = r_sq
            row_dict[f"Dr_{rng_key}_part_mean_rad2_s"] = p_mean
            row_dict[f"Dr_{rng_key}_part_sem_rad2_s"] = p_sem
            row_dict[f"C_{rng_key}_part_mean_rad2"] = p_c_mean
            row_dict[f"C_{rng_key}_part_sem_rad2"] = p_c_sem
            row_dict[f"Dr_{rng_key}_part_n"] = len(p_arr)

        dr_40 = row_dict["Dr_4_40s_ens_rad2_s"]
        c_40 = row_dict["C_4_40s_ens_rad2"]
        sem_40 = row_dict["Dr_4_40s_part_sem_rad2_s"]
        r2_40 = row_dict["Dr_4_40s_ens_r2"]
        print(f"  {b_name} (Rc={rc_um:.2f} \u03bcm): Dr(4-40s) = {dr_40:.4f} \u00b1 {sem_40:.4f} rad^2/s, C = {c_40:.4f} (R^2 = {r2_40:.4f})")
        dr_fit_rows.append(row_dict)

    df_dr_summary = pd.DataFrame(dr_fit_rows)
    save_csv_to_all(df_dr_summary, "dr_vs_cargo_radius_summary.csv", out_dirs)

    # ---------------------------------------------------------
    # 図3: D_r vs 貨物半径 R_c の単独プロット (MSD.py スタイル準拠)
    # ---------------------------------------------------------
    fig3, ax3 = plt.subplots(figsize=(8.5, 6.0))
    rc_arr = df_dr_summary['radius_um'].to_numpy()
    dr_arr = df_dr_summary['Dr_4_40s_ens_rad2_s'].to_numpy()

    # 個別ビーズごとのシンボル（MSD.py と完全に同一のマーカー・色）でプロット
    for idx, row in df_dr_summary.iterrows():
        ax3.errorbar(
            row['radius_um'], row['Dr_4_40s_ens_rad2_s'],
            yerr=row['Dr_4_40s_part_sem_rad2_s'],
            fmt=row['marker'], color=row['color'], ecolor=row['color'],
            markersize=9, capsize=4.5, elinewidth=1.6, markeredgewidth=1.2,
            label=f"{row['diameter_um']:.2f} \u03bcm ($R_c={row['radius_um']:.2f}\,\mu\mathrm{{m}}$)",
            zorder=4
        )

    # 全体をつなぐガイドライン（薄いグレー破線）
    ax3.plot(rc_arr, dr_arr, color='#555555', linestyle='--', linewidth=1.8, alpha=0.7, zorder=2)

    ax3.set_xscale('log')
    ax3.set_xlabel(r'Cargo Radius $R_c$ [$\mu\mathrm{m}$]')
    ax3.set_ylabel(r'Rotational Diffusion $D_r$ [$\mathrm{rad}^2/\mathrm{s}$]')
    ax3.set_title(r'Rotational Diffusion $D_r$ vs Cargo Radius $R_c$ ($\langle (\Delta \theta)^2 \rangle = 2 D_r \Delta t + C$)', pad=12)
    ax3.set_xlim(0.25, 15.0)
    ax3.set_ylim(0.0, 0.65)
    ax3.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    ax3.legend(fontsize=9.0, loc='upper right', framealpha=0.9, handlelength=1.4, borderpad=0.35, labelspacing=0.28)
    fig3.tight_layout()

    save_figure_to_all(fig3, "Dr_vs_cargo_radius", out_dirs)

    # ---------------------------------------------------------
    # 図4: 2パネル構成 (左: MSAD と 2*Dr*Delta_t + C フィッティング直線, 右: Dr vs Rc)
    # ---------------------------------------------------------
    fig4, (ax4_left, ax4_right) = plt.subplots(1, 2, figsize=(14.0, 5.8))

    # 左パネル: MSAD 曲線 (4s-40s) とフィッティング直線 (実線: データ, 破線: 2*Dr*t + C)
    t_fit_line = np.linspace(0.0, 42.0, 100)
    for item in beads_loaded:
        st = item["stats"]
        color = item["color"]
        marker = item["marker"]
        d_um = item["d_um"]
        rc_um = d_um / 2.0
        n_filt = item["n_filt"]

        valid_40 = (st.index >= 4.0) & (st.index <= 40.0)
        lags = st.index[valid_40]
        means = st['mean'][valid_40]
        sems = st['sem'][valid_40]

        # データ点
        ax4_left.errorbar(
            lags, means, yerr=sems,
            fmt=marker, color=color, ecolor=color, markersize=7.0, capsize=3,
            label=f"$R_c={rc_um:.2f}\,\mu\mathrm{{m}}$ ($d={d_um:.2f}\,\mu\mathrm{{m}}$)",
            zorder=3
        )

        # フィッティング直線 y = 2 * Dr * t + C
        r_info = df_dr_summary[df_dr_summary['bead_name'] == item["name"]].iloc[0]
        dr_val = r_info['Dr_4_40s_ens_rad2_s']
        c_val = r_info['C_4_40s_ens_rad2']
        ax4_left.plot(t_fit_line, 2.0 * dr_val * t_fit_line + c_val, color=color, linestyle='--', linewidth=1.8, alpha=0.85, zorder=2)

    ax4_left.set_xlim(0.0, 42.0)
    ax4_left.set_ylim(0.0, 50.0)
    ax4_left.set_xlabel(r'Lag time $\Delta t$ [s]')
    ax4_left.set_ylabel(r'MSAD $\langle (\Delta \theta)^2 \rangle$ [$\mathrm{rad}^2$]')
    ax4_left.set_title(r'(a) Linear Fit: $\langle (\Delta \theta)^2 \rangle = 2 D_r \Delta t + C$ ($4 - 40\,\mathrm{s}$)')
    ax4_left.legend(fontsize=9.0, loc='upper left', framealpha=0.9, handlelength=1.4)

    # 右パネル: Dr vs Rc
    for idx, row in df_dr_summary.iterrows():
        ax4_right.errorbar(
            row['radius_um'], row['Dr_4_40s_ens_rad2_s'],
            yerr=row['Dr_4_40s_part_sem_rad2_s'],
            fmt=row['marker'], color=row['color'], ecolor=row['color'],
            markersize=9, capsize=4.5, elinewidth=1.6, markeredgewidth=1.2,
            label=f"$R_c={row['radius_um']:.2f}\,\mu\mathrm{{m}}$",
            zorder=4
        )
    ax4_right.plot(rc_arr, dr_arr, color='#555555', linestyle='--', linewidth=1.8, alpha=0.7, zorder=2)
    ax4_right.set_xscale('log')
    ax4_right.set_xlabel(r'Cargo Radius $R_c$ [$\mu\mathrm{m}$]')
    ax4_right.set_ylabel(r'Rotational Diffusion Coefficient $D_r$ [$\mathrm{rad}^2/\mathrm{s}$]')
    ax4_right.set_title(r'(b) Rotational Diffusion $D_r$ vs Cargo Radius $R_c$')
    ax4_right.set_xlim(0.2, 15.0)
    ax4_right.set_ylim(0.0, 0.65)
    ax4_right.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    ax4_right.legend(fontsize=9.0, loc='upper right', framealpha=0.9, handlelength=1.4)

    fig4.tight_layout()
    save_figure_to_all(fig4, "msad_linear_fit_and_Dr_vs_radius_2panel", out_dirs)

    # ---------------------------------------------------------
    # 図5: MSAD(300 s) & alpha_theta vs 粒子径 2軸プロット
    # MSD.py の msd300_and_slope_vs_diameter と完全対応
    # ---------------------------------------------------------
    fig2, ax2_left = plt.subplots(figsize=(7.5, 5.5))
    ax2_right = ax2_left.twinx()

    d_arr = df_msad300_slope['diameter_um'].to_numpy()
    msad_m = df_msad300_slope['msad_300s_mean_rad2'].to_numpy()
    msad_sem = df_msad300_slope['msad_300s_sem_rad2'].to_numpy()
    alp_m = df_msad300_slope['alpha_theta_4_300s_mean'].to_numpy()
    alp_sem = df_msad300_slope['alpha_theta_4_300s_sem'].to_numpy()

    # 左軸: MSAD(300s)
    c_left = style_colors[0]
    ax2_left.errorbar(
        d_arr, msad_m, yerr=msad_sem,
        fmt='o-', color=c_left, ecolor=c_left, capsize=4, markersize=8, linewidth=2.0,
        label=r'MSAD($\Delta t = 300\,\mathrm{s}$)'
    )
    ax2_left.set_xlabel(r'Cargo Diameter $d$ [$\mu\mathrm{m}$]')
    ax2_left.set_ylabel(r'MSAD($300\,\mathrm{s}$) [$\mathrm{rad}^2$]', color=c_left)
    ax2_left.tick_params(axis='y', labelcolor=c_left)
    ax2_left.set_xscale('log')
    ax2_left.set_xlim(0.4, 30.0)

    # 右軸: alpha_theta (傾き)
    c_right = style_colors[1]
    ax2_right.errorbar(
        d_arr, alp_m, yerr=alp_sem,
        fmt='s--', color=c_right, ecolor=c_right, capsize=4, markersize=8, linewidth=2.0,
        label=r'Slope $\alpha_\theta$ ($4 - 300\,\mathrm{s}$)'
    )
    ax2_right.axhline(1.0, color='gray', linestyle=':', linewidth=1.2, label=r'Diffusive ($\alpha_\theta = 1$)')
    ax2_right.set_ylabel(r'Slope $\alpha_\theta$ (MSAD $\propto \Delta t^{\alpha_\theta}$)', color=c_right)
    ax2_right.tick_params(axis='y', labelcolor=c_right)
    ax2_right.set_ylim(0.0, 2.0)

    # 軸目盛り
    ax2_left.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

    fig2.suptitle(r'MSAD($300\,\mathrm{s}$) & Scaling Exponent $\alpha_\theta$ vs Cargo Diameter', fontsize=13)
    fig2.tight_layout()
    save_figure_to_all(fig2, "msad300_and_slope_vs_diameter", out_dirs)

    print("\n[DONE] All MSAD figures and summaries generated successfully!")


if __name__ == '__main__':
    main()
