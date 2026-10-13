#!/usr/bin/env python3
"""
plot_transport_efficiency.py

粒子追跡区間に同期した微小管平均速度 v_MT を採用した輸送効率 eta_transport(tau) および MSD 解析スクリプト。
- 各粒子の追跡フレーム区間 [f_min, f_max] に同期した微小管平均速度 v_MT を velocities_mean.csv から算出
- 輸送効率: eta_transport(tau) = v_eff(tau) / v_MT = sqrt(<dr^2(tau)>) / (v_MT * tau)
- MSD: 次元量 MSD(tau) および 無次元化 MSD*(tau*) = <dr^2(tau)> / xi^2 vs tau* = tau / (xi / v_MT)
- 全ビーズ径 (0.63, 1.18, 3.37, 5.00, 7.24, 20.00 um) の比較描画
- 四分位範囲 (25%-75% IQR) および 10%-90% 分布帯、アンサンブル代表値の描画
- サマリー CSV の出力
"""

import os
import sys
import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# libsモジュール
CURRENT_DIR = Path(__file__).parent.resolve()
sys.path.append(str(CURRENT_DIR))
from libs import fit_model as fm
from libs import displacement as dpm
from libs import cal_vel as cv

# スタイル設定
style_path = CURRENT_DIR / 'libs' / 'my_style.mplstyle'
if style_path.exists():
    plt.style.use(str(style_path))
    style_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
else:
    style_colors = ['#882255', '#CC6677', '#DDCC77', '#117733', '#44AA99', '#332288']

POSSIBLE_ROOTS = [
    Path('/mnt/NAS-Ebanaru/sasaki/MTsingleBeads'),
    Path('/mnt/NAS-Ebanaru/Sasaki/MTsingleBeads'),
    Path('/Volumes/data/sasaki/MTsingleBeads'),
    Path('/Volumes/data/Sasaki/MTsingleBeads'),
]

# 各ビーズのバックグラウンド微小管配向相関長 xi [um]
# (angular_correlation_bg_vp.zarr / bg_angular_correlation_length_summary.csv より)
DEFAULT_XI_MEDIAN_MAP = {
    'beads06um': 4.80,
    'beads1um': 4.43,
    'beads3um': 5.73,
    'beads5um': 3.34,
    'beads7um': 8.96,
    'beads20um': 7.00,
}


def load_bg_correlation_lengths(
    root_dir: Path,
    workspace_dir: Path
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    angular_correlation_bg_vp.zarr 由来の相関長データ
    (bg_angular_correlation_length_summary.csv および bg_angular_correlation_length_per_experiment.csv)
    をロードし、(ビーズ条件別の中央値マップ, 実験ごとの相関長マップ) を返す。
    """
    cond_map = dict(DEFAULT_XI_MEDIAN_MAP)
    exp_map: Dict[str, float] = {}

    # 条件別サマリーの読み込み
    for p in [
        workspace_dir / "figure" / "bg_angular_correlation" / "bg_angular_correlation_length_summary.csv",
        root_dir / "figure" / "bg_angular_correlation" / "bg_angular_correlation_length_summary.csv"
    ]:
        if p.exists():
            try:
                df_s = pd.read_csv(p)
                for _, r in df_s.iterrows():
                    bname = str(r.get('bead_name', ''))
                    val = r.get('xi_bg_median_um', r.get('xi_bg_mean_um', np.nan))
                    if bname in cond_map and np.isfinite(val) and float(val) > 0:
                        cond_map[bname] = float(val)
                print(f"Loaded condition-level xi from: {p}")
                break
            except Exception as e:
                print(f"Warning loading {p}: {e}")

    # 実験別サマリーの読み込み
    for p in [
        workspace_dir / "figure" / "bg_angular_correlation" / "bg_angular_correlation_length_per_experiment.csv",
        root_dir / "figure" / "bg_angular_correlation" / "bg_angular_correlation_length_per_experiment.csv"
    ]:
        if p.exists():
            try:
                df_e = pd.read_csv(p)
                for _, r in df_e.iterrows():
                    edir = str(r.get('exp_dir', ''))
                    val = r.get('xi_um', np.nan)
                    if edir and np.isfinite(val) and float(val) > 0:
                        exp_map[edir] = float(val)
                print(f"Loaded {len(exp_map)} experiment-level xi values from: {p}")
                break
            except Exception as e:
                print(f"Warning loading {p}: {e}")

    return cond_map, exp_map


def find_default_root() -> Path:
    for r in POSSIBLE_ROOTS:
        if r.exists():
            return r
    return CURRENT_DIR


def save_figure_to_all(fig: plt.Figure, base_name: str, out_dirs: List[Path], dpi: int = 300):
    for d in out_dirs:
        try:
            d.mkdir(parents=True, exist_ok=True)
            png_p = d / f"{base_name}.png"
            svg_p = d / f"{base_name}.svg"
            fig.savefig(png_p, dpi=dpi, bbox_inches='tight')
            fig.savefig(svg_p, bbox_inches='tight')
            print(f"  Saved figure: {png_p.name} -> {d}")
        except Exception as e:
            print(f"  Warning: Failed saving {base_name} to {d}: {e}")


def save_csv_to_all(df: pd.DataFrame, filename: str, out_dirs: List[Path]):
    for d in out_dirs:
        try:
            d.mkdir(parents=True, exist_ok=True)
            csv_p = d / filename
            df.to_csv(csv_p, index=False)
            print(f"  Saved CSV: {filename} -> {d}")
        except Exception as e:
            print(f"  Warning: Failed saving {filename} to {d}: {e}")


def calc_particle_alpha(p_df: pd.DataFrame, t_min: float = 4.0, t_max: float = 300.0) -> float:
    """4s ~ 300s の範囲で 個別粒子 MSD のべき指数 alpha を算出"""
    sub = p_df[(p_df['lag time'] >= t_min) & (p_df['lag time'] <= t_max) & (p_df['MSD'] > 0)].copy()
    if len(sub) < 3:
        return np.nan
    lx = np.log(sub['lag time'].values.astype(float))
    ly = np.log(sub['MSD'].values.astype(float))
    try:
        slope, _ = np.polyfit(lx, ly, 1)
        return float(slope)
    except Exception:
        return np.nan


def load_condition_tracks_and_compute_eta(
    folder: Path,
    interval_list: List[int],
    scale: float = 0.11,
    alpha_threshold: float = 0.5,
    exp_xi_map: Optional[Dict[str, float]] = None,
    cond_xi_default: float = 5.0,
) -> Tuple[pd.DataFrame, int, int]:
    """
    指定条件フォルダから全粒子の iMSD を計算し、
    各粒子の追跡フレーム区間 [f_min, f_max] に同期した微小管平均速度 v_MT を取得して
    v_eff(tau) = sqrt(MSD) / tau, eta_transport(tau) = v_eff(tau) / v_MT を算出。
    また、各実験の angular_correlation_bg_vp.zarr 由来の相関長 xi [um] を付与する。
    alpha > alpha_threshold の粒子のみをフィルタリングして返す。
    """
    all_imsds = []
    file_list = sorted(glob.glob(str(folder / "*" / "*" / "beads_tracks.csv")))
    for vid_idx, file_path in enumerate(file_list):
        interval = interval_list[vid_idx] if vid_idx < len(interval_list) else interval_list[-1]
        raw_df = pd.read_csv(file_path)
        track = cv.cal(raw_df, scale=scale, frame_interval=interval)
        imsd_df = dpm.imsd(track, scale=scale, time_scale=interval, display=False)
        imsd_df['video_idx'] = vid_idx
        exp_name = Path(file_path).parent.name
        imsd_df['exp_name'] = exp_name
        imsd_df['unique_particle_id'] = f"vid{vid_idx}_" + imsd_df['particle'].astype(str)

        # 実験ごとの相関長 xi [um] (angular_correlation_bg_vp.zarr 由来)
        exp_xi = cond_xi_default
        if exp_xi_map and exp_name in exp_xi_map:
            exp_xi = exp_xi_map[exp_name]
        imsd_df['xi_um'] = exp_xi

        # 微小管速度時系列 v_MT(t) の取得 (velocities_mean.csv)
        vel_path = Path(file_path).parent / "velocities_mean.csv"
        df_vel = pd.DataFrame()
        v_video_mean = 0.2
        if vel_path.exists():
            try:
                df_vel = pd.read_csv(vel_path)
                if 'mean_velocity' in df_vel.columns and len(df_vel) > 0:
                    v_val = pd.to_numeric(df_vel['mean_velocity'], errors='coerce').dropna()
                    if len(v_val) > 0:
                        v_video_mean = float(v_val.mean())
                else:
                    num_cols = df_vel.select_dtypes(include=[np.number]).columns
                    if len(num_cols) > 0:
                        v_video_mean = float(df_vel[num_cols[0]].dropna().iloc[0])
            except Exception:
                v_video_mean = 0.2
        if not np.isfinite(v_video_mean) or v_video_mean <= 0:
            v_video_mean = 0.2

        # 各粒子の追跡フレーム区間に同期した微小管速度の計算
        particle_vmt_map = {}
        for p, grp in raw_df.groupby('particle'):
            f_min, f_max = grp['frame'].min(), grp['frame'].max()
            if not df_vel.empty and 'mean_velocity' in df_vel.columns:
                sub_vel = df_vel[(df_vel['frame'] >= f_min) & (df_vel['frame'] <= f_max)]['mean_velocity']
                sub_val = pd.to_numeric(sub_vel, errors='coerce').dropna()
                particle_vmt_map[p] = float(sub_val.mean()) if len(sub_val) > 0 else v_video_mean
            else:
                particle_vmt_map[p] = v_video_mean

        imsd_df['v_MT'] = imsd_df['particle'].map(particle_vmt_map).fillna(v_video_mean)
        all_imsds.append(imsd_df)

    if len(all_imsds) == 0:
        empty_df = pd.DataFrame(columns=[
            'unique_particle_id', 'video_idx', 'exp_name', 'particle', 'lag time', 'MSD', 'v_MT', 'alpha',
            'v_eff', 'eta_transport', 'xi_um'
        ])
        return empty_df, 0, 0

    df_pool = pd.concat(all_imsds, ignore_index=True)
    df_pool['MSD'] = pd.to_numeric(df_pool['MSD'], errors='coerce')
    df_pool['lag time'] = pd.to_numeric(df_pool['lag time'], errors='coerce')
    df_pool['v_MT'] = pd.to_numeric(df_pool['v_MT'], errors='coerce')
    df_pool['xi_um'] = pd.to_numeric(df_pool['xi_um'], errors='coerce').fillna(cond_xi_default)
    df_pool = df_pool.dropna(subset=['MSD', 'lag time', 'v_MT'])
    df_pool = df_pool[(df_pool['MSD'] > 0) & (df_pool['lag time'] > 0) & (df_pool['v_MT'] > 0)]

    # 各粒子ごとの実効前進速度 v_eff(tau) および 輸送効率 eta_transport(tau)
    df_pool['v_eff'] = np.sqrt(df_pool['MSD'].values) / df_pool['lag time'].values
    df_pool['eta_transport'] = df_pool['v_eff'] / df_pool['v_MT']

    # 各粒子の alpha 算出
    p_alphas = {}
    for pid, grp in df_pool.groupby('unique_particle_id'):
        p_alphas[pid] = calc_particle_alpha(grp, 4.0, 300.0)
    df_pool['alpha'] = df_pool['unique_particle_id'].map(p_alphas)

    n_total = df_pool['unique_particle_id'].nunique()
    df_filtered = df_pool[df_pool['alpha'] > alpha_threshold].copy()
    n_filtered = df_filtered['unique_particle_id'].nunique()

    return df_filtered, n_total, n_filtered


def calc_condition_eta_stats(df_pool: pd.DataFrame, xi_um: float = 5.0, min_particles: int = 1) -> pd.DataFrame:
    """
    プールされた全粒子データから、ラグ時間 tau (= lag time) ごとの
    アンサンブル代表値 eta_ens = sqrt(<MSD>) / (mean_v_MT * tau)、
    および個別粒子 eta, MSD の統計量 (Mean, SEM, STD, Median, IQR, 10%-90%タイル) を算出する。
    相関長 xi_um は angular_correlation_bg_vp.zarr 由来の値を使用する。
    """
    if df_pool.empty:
        return pd.DataFrame()

    records = []
    for lag_val, grp in df_pool.groupby('lag time'):
        n_p = grp['unique_particle_id'].nunique()
        if n_p < min_particles:
            continue

        tau = float(lag_val)
        mean_msd = float(grp['MSD'].mean())
        mean_vmt = float(grp['v_MT'].mean())
        xi_val = float(grp['xi_um'].median()) if ('xi_um' in grp and grp['xi_um'].notna().any()) else xi_um

        # アンサンブル平均 MSD ベースの輸送効率
        eta_ens = float(np.sqrt(mean_msd) / (mean_vmt * tau))

        # 個別粒子 eta の分布統計量
        etas = grp['eta_transport'].to_numpy(dtype=float)
        eta_mean = float(np.mean(etas))
        eta_std = float(np.std(etas, ddof=1)) if len(etas) > 1 else 0.0
        eta_sem = float(eta_std / np.sqrt(len(etas))) if len(etas) > 1 else 0.0
        eta_med = float(np.median(etas))
        q25 = float(np.percentile(etas, 25))
        q75 = float(np.percentile(etas, 75))
        q10 = float(np.percentile(etas, 10))
        q90 = float(np.percentile(etas, 90))

        # 個別粒子 MSD の分布統計量
        msds = grp['MSD'].to_numpy(dtype=float)
        msd_med = float(np.median(msds))
        msd_std = float(np.std(msds, ddof=1)) if len(msds) > 1 else 0.0
        msd_sem = float(msd_std / np.sqrt(len(msds))) if len(msds) > 1 else 0.0
        msd_q25 = float(np.percentile(msds, 25))
        msd_q75 = float(np.percentile(msds, 75))
        msd_q10 = float(np.percentile(msds, 10))
        msd_q90 = float(np.percentile(msds, 90))

        # 無次元化値: tau* = tau / (xi / mean_vmt), MSD* = MSD / xi^2
        tau_char = xi_val / mean_vmt
        tau_star = tau / tau_char
        msd_star_mean = mean_msd / (xi_val ** 2)
        msd_star_median = msd_med / (xi_val ** 2)
        msd_star_q25 = msd_q25 / (xi_val ** 2)
        msd_star_q75 = msd_q75 / (xi_val ** 2)

        records.append({
            'tau_s': tau,
            'eta_ens': eta_ens,
            'eta_mean': eta_mean,
            'eta_sem': eta_sem,
            'eta_std': eta_std,
            'eta_median': eta_med,
            'eta_q25': q25,
            'eta_q75': q75,
            'eta_q10': q10,
            'eta_q90': q90,
            'mean_msd_um2': mean_msd,
            'msd_median_um2': msd_med,
            'msd_std_um2': msd_std,
            'msd_sem_um2': msd_sem,
            'msd_q25_um2': msd_q25,
            'msd_q75_um2': msd_q75,
            'msd_q10_um2': msd_q10,
            'msd_q90_um2': msd_q90,
            'tau_star': tau_star,
            'msd_star_mean': msd_star_mean,
            'msd_star_median': msd_star_median,
            'msd_star_q25': msd_star_q25,
            'msd_star_q75': msd_star_q75,
            'xi_um': xi_val,
            'rms_disp_um': float(np.sqrt(mean_msd)),
            'v_eff_mean_um_s': float(np.sqrt(mean_msd) / tau),
            'mean_v_MT_syn_um_s': mean_vmt,
            'n_active_particles': n_p
        })

    return pd.DataFrame(records)


def log_bin_dimensionless_msd(
    pool_df: pd.DataFrame,
    method: str = 'fd',
    n_bins_per_decade: int = 10,
    min_particles: int = 2,
    min_points: int = 3
) -> pd.DataFrame:
    """
    プールされた個別粒子データに対し、無次元時間 tau* の対数空間で最適なビンを作成して
    MSD* の中央値、IQR (q25, q75)、平均値、SEM を集計する。
    method='fd': Freedman-Diaconis 則 (Delta u = 2 * IQR(u) / N^(1/3))
    """
    if pool_df.empty:
        return pd.DataFrame()

    tau_star = (pool_df['lag time'] * pool_df['v_MT'] / pool_df['xi_um']).to_numpy(dtype=float)
    msd_star = (pool_df['MSD'] / (pool_df['xi_um'] ** 2)).to_numpy(dtype=float)
    eta_arr = pool_df['eta_transport'].to_numpy(dtype=float) if 'eta_transport' in pool_df.columns else (np.sqrt(pool_df['MSD']) / (pool_df['v_MT'] * pool_df['lag time'])).to_numpy(dtype=float)
    p_ids = pool_df['unique_particle_id'].to_numpy()

    valid = np.isfinite(tau_star) & np.isfinite(msd_star) & np.isfinite(eta_arr) & (tau_star > 0) & (msd_star > 0) & (eta_arr > 0)
    tau_star = tau_star[valid]
    msd_star = msd_star[valid]
    eta_arr = eta_arr[valid]
    p_ids = p_ids[valid]

    if len(tau_star) == 0:
        return pd.DataFrame()

    u = np.log10(tau_star)
    u_min, u_max = float(np.min(u)), float(np.max(u))
    N = len(u)

    if method.lower() == 'fd':
        iqr = float(np.percentile(u, 75) - np.percentile(u, 25))
        if iqr > 0 and N >= 8:
            fd_bin_w = 2.0 * iqr / (N ** (1.0 / 3.0))
            n_bins = max(6, int(np.ceil((u_max - u_min) / fd_bin_w)))
        else:
            n_bins = max(6, int(np.round((u_max - u_min) * n_bins_per_decade)))
    else:
        decades = u_max - u_min
        n_bins = max(6, int(np.round(decades * n_bins_per_decade)))

    edges = np.geomspace(10 ** (u_min - 1e-4), 10 ** (u_max + 1e-4), n_bins + 1)
    records = []

    for k in range(n_bins):
        lo, hi = edges[k], edges[k + 1]
        mask = (tau_star >= lo) & (tau_star <= hi) if k == n_bins - 1 else (tau_star >= lo) & (tau_star < hi)
        n_pts = int(np.count_nonzero(mask))
        if n_pts < min_points:
            continue

        pts_p = len(np.unique(p_ids[mask]))
        if pts_p < min_particles:
            continue

        sub_t = tau_star[mask]
        sub_y = msd_star[mask]
        sub_eta = eta_arr[mask]

        t_center = float(np.sqrt(lo * hi))
        t_geom = float(np.exp(np.mean(np.log(sub_t))))
        t_med = float(np.median(sub_t))

        y_med = float(np.median(sub_y))
        y_q25 = float(np.percentile(sub_y, 25))
        y_q75 = float(np.percentile(sub_y, 75))
        y_mean = float(np.mean(sub_y))
        y_std = float(np.std(sub_y, ddof=1)) if n_pts > 1 else 0.0
        y_sem = float(y_std / np.sqrt(n_pts)) if n_pts > 1 else 0.0

        eta_med = float(np.median(sub_eta))
        eta_q25 = float(np.percentile(sub_eta, 25))
        eta_q75 = float(np.percentile(sub_eta, 75))
        eta_q10 = float(np.percentile(sub_eta, 10))
        eta_q90 = float(np.percentile(sub_eta, 90))
        eta_mean = float(np.mean(sub_eta))
        eta_std = float(np.std(sub_eta, ddof=1)) if n_pts > 1 else 0.0
        eta_sem = float(eta_std / np.sqrt(n_pts)) if n_pts > 1 else 0.0
        eta_ens = float(np.sqrt(y_mean) / t_geom)

        records.append({
            'tau_star': t_geom,
            'tau_star_center': t_center,
            'tau_star_med': t_med,
            'msd_star_median': y_med,
            'msd_star_q25': y_q25,
            'msd_star_q75': y_q75,
            'msd_star_mean': y_mean,
            'msd_star_sem': y_sem,
            'eta_median': eta_med,
            'eta_q25': eta_q25,
            'eta_q75': eta_q75,
            'eta_q10': eta_q10,
            'eta_q90': eta_q90,
            'eta_mean': eta_mean,
            'eta_sem': eta_sem,
            'eta_ens': eta_ens,
            'n_points': n_pts,
            'n_particles': pts_p,
            'bin_lo': lo,
            'bin_hi': hi,
            'bin_method': 'Freedman-Diaconis' if method.lower() == 'fd' else 'Fixed-Decade',
        })

    return pd.DataFrame(records)


def main():
    root_dir = find_default_root()
    workspace_dir = CURRENT_DIR

    out_dirs = [
        workspace_dir / "figure" / "msd",
        root_dir / "figure" / "msd",
    ]
    for d in out_dirs:
        try:
            d.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    print(f"Data Root Directory: {root_dir}")
    print("Saving outputs to:")
    for d in out_dirs:
        print(f"  {d}")

    alpha_thresh = 0.5
    cond_xi_map, exp_xi_map = load_bg_correlation_lengths(root_dir, workspace_dir)
    print("\nBackground Microtubule Orientation Correlation Lengths (from angular_correlation_bg_vp.zarr):")
    for bname, xi_v in cond_xi_map.items():
        print(f"  {bname:10s}: xi = {xi_v:.2f} um")

    beads_info = [
        {"name": "beads06um", "d_um": 0.63, "xi_um": cond_xi_map['beads06um'], "iv": [4, 4, 4, 4], "marker": "^", "color": style_colors[0]},
        {"name": "beads1um",  "d_um": 1.18, "xi_um": cond_xi_map['beads1um'],  "iv": [4, 4, 4, 4], "marker": "o", "color": style_colors[1]},
        {"name": "beads3um",  "d_um": 3.37, "xi_um": cond_xi_map['beads3um'],  "iv": [4, 4, 4, 4], "marker": "d", "color": style_colors[2]},
        {"name": "beads5um",  "d_um": 5.00, "xi_um": cond_xi_map['beads5um'],  "iv": [4, 4, 4, 4], "marker": "p", "color": style_colors[3]},
        {"name": "beads7um",  "d_um": 7.24, "xi_um": cond_xi_map['beads7um'],  "iv": [4, 4, 4],    "marker": "h", "color": style_colors[4]},
        {"name": "beads20um", "d_um": 20.0, "xi_um": cond_xi_map['beads20um'], "iv": [4, 4, 4],    "marker": "s", "color": style_colors[5]},
    ]

    print("\n--- 1. Loading tracks and computing particle-synchronized eta(tau) & MSD ---")
    beads_data = []
    summary_rows = []
    binned_summary_rows = []

    for b in beads_info:
        b_folder = root_dir / b["name"]
        pool_df, n_tot, n_filt = load_condition_tracks_and_compute_eta(
            b_folder, b["iv"], scale=0.11, alpha_threshold=alpha_thresh,
            exp_xi_map=exp_xi_map, cond_xi_default=b["xi_um"]
        )
        stats_df = calc_condition_eta_stats(pool_df, xi_um=b["xi_um"], min_particles=2)
        binned_df = log_bin_dimensionless_msd(pool_df, method="fd", min_particles=2, min_points=3)
        print(f"  {b['name']} ({b['d_um']:.2f} \u03bcm, \u03be={b['xi_um']:.2f} \u03bcm): {n_tot} -> {n_filt} active particles (stats: {len(stats_df)}, FD log-bins: {len(binned_df)})")

        beads_data.append({
            "name": b["name"],
            "d_um": b["d_um"],
            "xi_um": b["xi_um"],
            "pool_df": pool_df,
            "stats_df": stats_df,
            "binned_df": binned_df,
            "n_tot": n_tot,
            "n_filt": n_filt,
            "marker": b["marker"],
            "color": b["color"]
        })

        for _, row in stats_df.iterrows():
            summary_rows.append({
                "bead_name": b["name"],
                "diameter_um": b["d_um"],
                "xi_um": b["xi_um"],
                "tau_s": row['tau_s'],
                "eta_ens": row['eta_ens'],
                "eta_mean": row['eta_mean'],
                "eta_sem": row['eta_sem'],
                "eta_std": row['eta_std'],
                "eta_median": row['eta_median'],
                "eta_q25": row['eta_q25'],
                "eta_q75": row['eta_q75'],
                "eta_q10": row['eta_q10'],
                "eta_q90": row['eta_q90'],
                "mean_msd_um2": row['mean_msd_um2'],
                "msd_median_um2": row['msd_median_um2'],
                "msd_sem_um2": row['msd_sem_um2'],
                "msd_q25_um2": row['msd_q25_um2'],
                "msd_q75_um2": row['msd_q75_um2'],
                "tau_star": row['tau_star'],
                "msd_star_mean": row['msd_star_mean'],
                "msd_star_median": row['msd_star_median'],
                "msd_star_q25": row['msd_star_q25'],
                "msd_star_q75": row['msd_star_q75'],
                "mean_v_MT_syn_um_s": row['mean_v_MT_syn_um_s'],
                "n_active_particles": row['n_active_particles']
            })

        for _, row in binned_df.iterrows():
            binned_summary_rows.append({
                "bead_name": b["name"],
                "diameter_um": b["d_um"],
                "xi_um": b["xi_um"],
                "tau_star": row['tau_star'],
                "tau_star_center": row['tau_star_center'],
                "msd_star_median": row['msd_star_median'],
                "msd_star_q25": row['msd_star_q25'],
                "msd_star_q75": row['msd_star_q75'],
                "msd_star_mean": row['msd_star_mean'],
                "msd_star_sem": row['msd_star_sem'],
                "eta_median": row['eta_median'],
                "eta_q25": row['eta_q25'],
                "eta_q75": row['eta_q75'],
                "eta_q10": row['eta_q10'],
                "eta_q90": row['eta_q90'],
                "eta_mean": row['eta_mean'],
                "eta_sem": row['eta_sem'],
                "eta_ens": row['eta_ens'],
                "n_points": row['n_points'],
                "n_particles": row['n_particles'],
                "bin_lo": row['bin_lo'],
                "bin_hi": row['bin_hi'],
            })

    df_summary = pd.DataFrame(summary_rows)
    save_csv_to_all(df_summary, "transport_efficiency_vs_tau_summary.csv", out_dirs)
    save_csv_to_all(df_summary, "msd_particle_synchronized_summary.csv", out_dirs)

    df_binned_summary = pd.DataFrame(binned_summary_rows)
    save_csv_to_all(df_binned_summary, "dimensionless_msd_log_binned_summary.csv", out_dirs)
    save_csv_to_all(df_binned_summary, "dimensionless_transport_efficiency_log_binned_summary.csv", out_dirs)

    # ---------------------------------------------------------
    # 図1: Log-Log プロット (eta_transport(tau) vs tau)
    # ---------------------------------------------------------
    fig_log, ax_log = plt.subplots(figsize=(8.5, 6.2))
    for item in beads_data:
        st = item["stats_df"]
        color = item["color"]
        marker = item["marker"]
        d_um = item["d_um"]
        n_filt = item["n_filt"]

        if st.empty:
            continue

        tau_vals = st['tau_s'].values
        eta_ens = st['eta_ens'].values
        q25 = st['eta_q25'].values
        q75 = st['eta_q75'].values
        q10 = st['eta_q10'].values
        q90 = st['eta_q90'].values

        # 10% - 90% タイル帯 (淡色)
        ax_log.fill_between(tau_vals, q10, q90, color=color, alpha=0.08, edgecolor='none')
        # 25% - 75% 四分位範囲 (IQR) 帯 (中間色)
        ax_log.fill_between(tau_vals, q25, q75, color=color, alpha=0.22, edgecolor='none')
        # 主線: アンサンブル代表値 eta_ens = sqrt(<MSD>) / (v_MT_syn * tau)
        ax_log.plot(
            tau_vals, eta_ens, marker=marker, linestyle='-', linewidth=2.0, markersize=7.5,
            label=f'{d_um:.2f} \u03bcm ($N={n_filt}$)', color=color, zorder=4
        )

    # 基準線: 同期輸送限界 eta = 1 (v_eff = v_MT)
    ax_log.axhline(1.0, color='#999999', linestyle='--', linewidth=1.2, zorder=2)
    ax_log.text(5.0, 1.08, r'$\eta = 1$ (Synchronous transport: $v_{\mathrm{eff}} = v_{\mathrm{MT}}$)',
                fontsize=10.0, color='#666666', fontweight='bold')

    # ガイド線 (対数空間での漸近スケーリング)
    # 1. 弾道輸送 (alpha=2 -> slope=0: eta \propto \tau^0)
    t_g1, t_g2 = 6.0, 25.0
    ax_log.plot([t_g1, t_g2], [0.85, 0.85], color='#333333', linestyle=':', linewidth=1.4, zorder=3)
    ax_log.text(t_g1 * 1.15, 0.90, r'$\propto \tau^{0}$ ($\alpha=2$)', fontsize=11, color='#333333')

    # 2. 標準拡散 (alpha=1 -> slope=-0.5: eta \propto \tau^-0.5)
    t_d1, t_d2 = 40.0, 220.0
    A_diff = 1.3
    ax_log.plot([t_d1, t_d2], [A_diff * (t_d1**-0.5), A_diff * (t_d2**-0.5)], color='#222222', linestyle='--', linewidth=1.5, zorder=5)
    ax_log.text(45, A_diff * (45**-0.5) * 0.65, r'$\propto \tau^{-0.5}$ ($\alpha=1$)', fontsize=11.0, color='#222222')

    # 3. ケージング・停止 (alpha=0 -> slope=-1.0: eta \propto \tau^-1.0)
    t_c1, t_c2 = 120.0, 450.0
    A_cage = 8.5
    ax_log.plot([t_c1, t_c2], [A_cage * (t_c1**-1.0), A_cage * (t_c2**-1.0)], color='#666666', linestyle='-.', linewidth=1.3, zorder=3)
    ax_log.text(150, A_cage * (150**-1.0) * 0.55, r'$\propto \tau^{-1.0}$ ($\alpha=0$)', fontsize=11, color='#555555')

    ax_log.legend(fontsize=10.0, loc='lower left', framealpha=0.90, markerscale=0.82, handlelength=1.4, borderpad=0.35, labelspacing=0.28)
    ax_log.set(
        xlim=(4e0, 1000),
        ylim=(0.02, 1.5),
        xscale='log',
        yscale='log',
        xlabel=r'Time interval $\tau$ [s]',
        ylabel=r'Transport Efficiency $\eta_{\mathrm{transport}}(\tau)$',
        title=r'Cargo Transport Efficiency $\eta_{\mathrm{transport}}(\tau) \equiv \frac{\sqrt{\langle \Delta r^2(\tau) \rangle}}{v_{\mathrm{MT, syn}} \, \tau}$ vs $\tau$ (Log-Log)'
    )
    save_figure_to_all(fig_log, "transport_efficiency_vs_tau", out_dirs)

    # ---------------------------------------------------------
    # 図2: リニア縦軸 プロット (Linear y)
    # ---------------------------------------------------------
    fig_lin, ax_lin = plt.subplots(figsize=(8.5, 6.2))
    for item in beads_data:
        st = item["stats_df"]
        color = item["color"]
        marker = item["marker"]
        d_um = item["d_um"]
        n_filt = item["n_filt"]

        if st.empty:
            continue

        tau_vals = st['tau_s'].values
        eta_ens = st['eta_ens'].values
        q25 = st['eta_q25'].values
        q75 = st['eta_q75'].values
        q10 = st['eta_q10'].values
        q90 = st['eta_q90'].values

        ax_lin.fill_between(tau_vals, q10, q90, color=color, alpha=0.08, edgecolor='none')
        ax_lin.fill_between(tau_vals, q25, q75, color=color, alpha=0.22, edgecolor='none')
        ax_lin.plot(
            tau_vals, eta_ens, marker=marker, linestyle='-', linewidth=2.0, markersize=7.5,
            label=f'{d_um:.2f} \u03bcm ($N={n_filt}$)', color=color, zorder=4
        )

    ax_lin.axhline(1.0, color='#999999', linestyle='--', linewidth=1.2, zorder=2)
    ax_lin.text(5.0, 1.03, r'$\eta = 1$ ($v_{\mathrm{eff}} = v_{\mathrm{MT}}$)', fontsize=10.5, color='#666666')

    ax_lin.legend(fontsize=10.0, loc='upper right', framealpha=0.90, markerscale=0.82, handlelength=1.4)
    ax_lin.set(
        xlim=(4e0, 1000),
        ylim=(0.0, 1.15),
        xscale='log',
        xlabel=r'Time interval $\tau$ [s]',
        ylabel=r'Transport Efficiency $\eta_{\mathrm{transport}}(\tau)$',
        title=r'Cargo Transport Efficiency $\eta_{\mathrm{transport}}(\tau) \equiv \frac{\sqrt{\langle \Delta r^2(\tau) \rangle}}{v_{\mathrm{MT, syn}} \, \tau}$ vs $\tau$ (Linear Scale)'
    )
    save_figure_to_all(fig_lin, "transport_efficiency_vs_tau_linear", out_dirs)

    # ---------------------------------------------------------
    # 図3: 2パネル図 (Panel A: Linear, Panel B: Log-Log)
    # ---------------------------------------------------------
    fig_2p, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(15.6, 5.8))
    # Panel A (Linear y)
    for item in beads_data:
        st = item["stats_df"]
        if st.empty:
            continue
        ax_a.fill_between(st['tau_s'], st['eta_q25'], st['eta_q75'], color=item['color'], alpha=0.22, edgecolor='none')
        ax_a.plot(st['tau_s'], st['eta_ens'], marker=item['marker'], linestyle='-', linewidth=2.0, markersize=7.0,
                  label=f"{item['d_um']:.2f} \u03bcm ($N={item['n_filt']}$)", color=item['color'])
    ax_a.axhline(1.0, color='#999999', linestyle='--', linewidth=1.2)
    ax_a.text(5.0, 1.03, r'$\eta = 1$ ($v_{\mathrm{eff}} = v_{\mathrm{MT}}$)', fontsize=10.0, color='#666666')
    ax_a.set(
        xlim=(4e0, 1000),
        ylim=(0.0, 1.15),
        xscale='log',
        xlabel=r'Time interval $\tau$ [s]',
        ylabel=r'Transport Efficiency $\eta_{\mathrm{transport}}(\tau)$',
        title=r'(a) Linear Scale'
    )
    ax_a.legend(fontsize=9.5, loc='upper right', framealpha=0.90, markerscale=0.8)

    # Panel B (Log-Log)
    for item in beads_data:
        st = item["stats_df"]
        if st.empty:
            continue
        ax_b.fill_between(st['tau_s'], st['eta_q25'], st['eta_q75'], color=item['color'], alpha=0.22, edgecolor='none')
        ax_b.plot(st['tau_s'], st['eta_ens'], marker=item['marker'], linestyle='-', linewidth=2.0, markersize=7.0,
                  label=f"{item['d_um']:.2f} \u03bcm ($N={item['n_filt']}$)", color=item['color'])
    ax_b.axhline(1.0, color='#999999', linestyle='--', linewidth=1.2)
    ax_b.plot([t_g1, t_g2], [0.85, 0.85], color='#333333', linestyle=':', linewidth=1.4)
    ax_b.text(t_g1 * 1.15, 0.90, r'$\propto \tau^{0}$ ($\alpha=2$)', fontsize=10.5, color='#333333')
    ax_b.plot([t_d1, t_d2], [A_diff * (t_d1**-0.5), A_diff * (t_d2**-0.5)], color='#222222', linestyle='--', linewidth=1.5)
    ax_b.text(45, A_diff * (45**-0.5) * 0.65, r'$\propto \tau^{-0.5}$ ($\alpha=1$)', fontsize=10.5, color='#222222')
    ax_b.plot([t_c1, t_c2], [A_cage * (t_c1**-1.0), A_cage * (t_c2**-1.0)], color='#666666', linestyle='-.', linewidth=1.3)
    ax_b.text(150, A_cage * (150**-1.0) * 0.55, r'$\propto \tau^{-1.0}$ ($\alpha=0$)', fontsize=10.5, color='#555555')
    ax_b.set(
        xlim=(4e0, 1000),
        ylim=(0.02, 1.5),
        xscale='log',
        yscale='log',
        xlabel=r'Time interval $\tau$ [s]',
        ylabel=r'Transport Efficiency $\eta_{\mathrm{transport}}(\tau)$',
        title=r'(b) Log-Log Scale'
    )
    ax_b.legend(fontsize=9.5, loc='lower left', framealpha=0.90, markerscale=0.8)

    fig_2p.tight_layout(rect=[0, 0, 1, 0.94])
    fig_2p.suptitle(r'Cargo Transport Efficiency $\eta_{\mathrm{transport}}(\tau) \equiv \frac{\sqrt{\langle \Delta r^2(\tau) \rangle}}{v_{\mathrm{MT, syn}} \, \tau} = \frac{v_{\mathrm{eff}}(\tau)}{v_{\mathrm{MT, syn}}}$ vs $\tau$',
                    fontsize=13.5, fontweight='bold', y=0.98)
    save_figure_to_all(fig_2p, "transport_efficiency_vs_tau_2panel", out_dirs)

    # ---------------------------------------------------------
    # 図4: 次元量 MSD プロット (MSD(tau) vs tau)
    # ---------------------------------------------------------
    fig_msd, ax_msd = plt.subplots(figsize=(8.5, 6.2))
    for item in beads_data:
        st = item["stats_df"]
        if st.empty:
            continue
        tau_vals = st['tau_s'].values
        mean_msd = st['mean_msd_um2'].values
        q25_msd = st['msd_q25_um2'].values
        q75_msd = st['msd_q75_um2'].values
        q10_msd = st['msd_q10_um2'].values
        q90_msd = st['msd_q90_um2'].values

        ax_msd.fill_between(tau_vals, q10_msd, q90_msd, color=item['color'], alpha=0.08, edgecolor='none')
        ax_msd.fill_between(tau_vals, q25_msd, q75_msd, color=item['color'], alpha=0.22, edgecolor='none')
        ax_msd.plot(
            tau_vals, mean_msd, marker=item['marker'], linestyle='-', linewidth=2.0, markersize=7.5,
            label=f"{item['d_um']:.2f} \u03bcm ($N={item['n_filt']}$)", color=item['color'], zorder=4
        )

    # MSD のガイド線: 弾道輸送 \propto \tau^2, 拡散輸送 \propto \tau^1
    t_b1, t_b2 = 5.0, 30.0
    A_ball = 0.020
    ax_msd.plot([t_b1, t_b2], [A_ball * (t_b1**2), A_ball * (t_b2**2)], color='#222222', linestyle=':', linewidth=1.5, zorder=5)
    ax_msd.text(5.2, A_ball * (5.2**2) * 2.8, r'$\propto \tau^{2.0}$ (Ballistic)', fontsize=11, color='#222222')

    t_df1, t_df2 = 100.0, 600.0
    A_df = 0.35
    ax_msd.plot([t_df1, t_df2], [A_df * (t_df1**1), A_df * (t_df2**1)], color='#222222', linestyle='--', linewidth=1.5, zorder=5)
    ax_msd.text(140.0, A_df * (140.0**1) * 0.30, r'$\propto \tau^{1.0}$ (Diffusive)', fontsize=11, color='#222222')

    ax_msd.legend(fontsize=10.0, loc='upper left', framealpha=0.90, markerscale=0.82)
    ax_msd.set(
        xlim=(4e0, 1000),
        ylim=(0.02, 1200),
        xscale='log',
        yscale='log',
        xlabel=r'Time interval $\tau$ [s]',
        ylabel=r'Mean Squared Displacement $\langle \Delta r^2(\tau) \rangle$ [$\mu\mathrm{m}^2$]',
        title=r'Cargo Mean Squared Displacement $\langle \Delta r^2(\tau) \rangle$ vs $\tau$'
    )
    save_figure_to_all(fig_msd, "msd_vs_tau", out_dirs)

    # ---------------------------------------------------------
    # 図5: 対数ビニング無次元化 MSD プロット (MSD/xi^2 vs tau / (xi / v_MT_syn))
    # ---------------------------------------------------------
    fig_dim, ax_dim = plt.subplots(figsize=(8.5, 6.2))
    for item in beads_data:
        st = item["binned_df"]
        if st.empty:
            continue
        tau_s = st['tau_star'].values
        msd_s = st['msd_star_mean'].values
        q25_s = st['msd_star_q25'].values
        q75_s = st['msd_star_q75'].values

        ax_dim.fill_between(tau_s, q25_s, q75_s, color=item['color'], alpha=0.20, edgecolor='none')
        ax_dim.plot(
            tau_s, msd_s, marker=item['marker'], linestyle='-', linewidth=2.0, markersize=7.0,
            label=f"{item['d_um']:.2f} \u03bcm ($\u03be = {item['xi_um']:.1f}\\,\\mu\\mathrm{{m}}$)", color=item['color'], zorder=4
        )

    # 無次元スケーリングガイド線
    t_s1, t_s2 = 0.15, 1.5
    ax_dim.plot([t_s1, t_s2], [0.25 * (t_s1**2), 0.25 * (t_s2**2)], color='#222222', linestyle=':', linewidth=1.5, zorder=5)
    ax_dim.text(0.18, 0.25 * (0.18**2) * 3.5, r'$\propto (\tau^*)^2$', fontsize=11, color='#222222')

    t_s3, t_s4 = 2.0, 25.0
    ax_dim.plot([t_s3, t_s4], [0.15 * (t_s3**1), 0.15 * (t_s4**1)], color='#222222', linestyle='--', linewidth=1.5, zorder=5)
    ax_dim.text(3.5, 0.15 * (3.5**1) * 0.40, r'$\propto \tau^*$', fontsize=11, color='#222222')

    ax_dim.legend(fontsize=9.5, loc='upper left', framealpha=0.90, markerscale=0.82)
    ax_dim.set(
        xscale='log',
        yscale='log',
        xlabel=r'Dimensionless Time $\tau^* \equiv \tau \, v_{\mathrm{MT, syn}} / \xi_{\mathrm{bg}}$',
        ylabel=r'Dimensionless MSD $\langle \Delta r^2(\tau) \rangle / \xi_{\mathrm{bg}}^2$',
        title=r'Freedman–Diaconis Binned Dimensionless Cargo MSD: $\langle \Delta r^2 \rangle / \xi_{\mathrm{bg}}^2$ vs $\tau^*$'
    )
    save_figure_to_all(fig_dim, "dimensionless_msd_vs_tau", out_dirs)

    # ---------------------------------------------------------
    # 図6: 対数ビニング無次元化 輸送効率 プロット (Log-Log: eta_transport(tau*) vs tau*)
    # ---------------------------------------------------------
    fig_dim_eta, ax_dim_eta = plt.subplots(figsize=(8.5, 6.2))
    for item in beads_data:
        st = item["binned_df"]
        if st.empty:
            continue
        tau_s = st['tau_star'].values
        eta_s = st['eta_median'].values
        q25_s = st['eta_q25'].values
        q75_s = st['eta_q75'].values
        q10_s = st['eta_q10'].values
        q90_s = st['eta_q90'].values

        ax_dim_eta.fill_between(tau_s, q10_s, q90_s, color=item['color'], alpha=0.08, edgecolor='none')
        ax_dim_eta.fill_between(tau_s, q25_s, q75_s, color=item['color'], alpha=0.22, edgecolor='none')
        ax_dim_eta.plot(
            tau_s, eta_s, marker=item['marker'], linestyle='-', linewidth=2.0, markersize=7.0,
            label=f"{item['d_um']:.2f} \u03bcm ($\u03be = {item['xi_um']:.1f}\\,\\mu\\mathrm{{m}}$)",
            color=item['color'], zorder=4
        )

    # 基準線: 同期輸送限界 eta = 1
    ax_dim_eta.axhline(1.0, color='#999999', linestyle='--', linewidth=1.2, zorder=2)
    ax_dim_eta.text(0.09, 1.05, r'$\eta = 1$ (Synchronous limit: $v_{\mathrm{eff}} = v_{\mathrm{MT}}$)',
                    fontsize=10.0, color='#666666', fontweight='bold')

    # ガイド線 (無次元空間での漸近スケーリング)
    # 1. 弾道輸送 (alpha=2 -> slope=0: eta \propto (tau*)^0)
    t_g1, t_g2 = 0.12, 1.2
    ax_dim_eta.plot([t_g1, t_g2], [0.82, 0.82], color='#333333', linestyle=':', linewidth=1.4, zorder=3)
    ax_dim_eta.text(0.18, 0.70, r'$\propto (\tau^*)^{0}$ ($\alpha=2$)', fontsize=10.5, color='#333333')

    # 2. 標準拡散 (alpha=1 -> slope=-0.5: eta \propto (tau*)^-0.5)
    t_d1, t_d2 = 2.0, 25.0
    A_diff_star = 0.55
    ax_dim_eta.plot([t_d1, t_d2], [A_diff_star * (t_d1**-0.5), A_diff_star * (t_d2**-0.5)], color='#222222', linestyle='--', linewidth=1.5, zorder=5)
    ax_dim_eta.text(4.0, 0.35, r'$\propto (\tau^*)^{-0.5}$ ($\alpha=1$)', fontsize=10.5, color='#222222')

    # 3. ケージング・停止 (alpha=0 -> slope=-1.0: eta \propto (tau*)^-1.0)
    t_c1, t_c2 = 10.0, 45.0
    A_cage_star = 1.8
    ax_dim_eta.plot([t_c1, t_c2], [A_cage_star * (t_c1**-1.0), A_cage_star * (t_c2**-1.0)], color='#666666', linestyle='-.', linewidth=1.3, zorder=3)
    ax_dim_eta.text(12.0, 0.08, r'$\propto (\tau^*)^{-1.0}$ ($\alpha=0$)', fontsize=10.5, color='#555555')

    ax_dim_eta.legend(fontsize=9.5, loc='lower left', framealpha=0.90, markerscale=0.82)
    ax_dim_eta.set(
        xlim=(0.07, 250),
        ylim=(0.02, 1.5),
        xscale='log',
        yscale='log',
        xlabel=r'Dimensionless Time $\tau^* \equiv \tau \, v_{\mathrm{MT, syn}} / \xi_{\mathrm{bg}}$',
        ylabel=r'Transport Efficiency $\eta_{\mathrm{transport}}(\tau^*)$',
        title=r'Freedman–Diaconis Binned Dimensionless Cargo Transport Efficiency $\eta_{\mathrm{transport}}(\tau^*) \equiv \frac{\sqrt{\langle \Delta r^2 \rangle}}{\xi_{\mathrm{bg}} \, \tau^*}$ vs $\tau^*$ (Log-Log)'
    )
    save_figure_to_all(fig_dim_eta, "dimensionless_transport_efficiency_vs_tau", out_dirs)

    # ---------------------------------------------------------
    # 図7: 対数ビニング無次元化 輸送効率 プロット (Linear y: eta_transport(tau*) vs tau*)
    # ---------------------------------------------------------
    fig_dim_lin, ax_dim_lin = plt.subplots(figsize=(8.5, 6.2))
    for item in beads_data:
        st = item["binned_df"]
        if st.empty:
            continue
        tau_s = st['tau_star'].values
        eta_s = st['eta_median'].values
        q25_s = st['eta_q25'].values
        q75_s = st['eta_q75'].values
        q10_s = st['eta_q10'].values
        q90_s = st['eta_q90'].values

        ax_dim_lin.fill_between(tau_s, q10_s, q90_s, color=item['color'], alpha=0.08, edgecolor='none')
        ax_dim_lin.fill_between(tau_s, q25_s, q75_s, color=item['color'], alpha=0.22, edgecolor='none')
        ax_dim_lin.plot(
            tau_s, eta_s, marker=item['marker'], linestyle='-', linewidth=2.0, markersize=7.0,
            label=f"{item['d_um']:.2f} \u03bcm ($\u03be = {item['xi_um']:.1f}\\,\\mu\\mathrm{{m}}$)",
            color=item['color'], zorder=4
        )

    ax_dim_lin.axhline(1.0, color='#999999', linestyle='--', linewidth=1.2, zorder=2)
    ax_dim_lin.text(0.09, 1.03, r'$\eta = 1$ ($v_{\mathrm{eff}} = v_{\mathrm{MT}}$)', fontsize=10.5, color='#666666')

    ax_dim_lin.legend(fontsize=9.5, loc='upper right', framealpha=0.90, markerscale=0.82)
    ax_dim_lin.set(
        xlim=(0.07, 250),
        ylim=(0.0, 1.15),
        xscale='log',
        xlabel=r'Dimensionless Time $\tau^* \equiv \tau \, v_{\mathrm{MT, syn}} / \xi_{\mathrm{bg}}$',
        ylabel=r'Transport Efficiency $\eta_{\mathrm{transport}}(\tau^*)$',
        title=r'Freedman–Diaconis Binned Dimensionless Cargo Transport Efficiency vs $\tau^*$ (Linear Scale)'
    )
    save_figure_to_all(fig_dim_lin, "dimensionless_transport_efficiency_linear", out_dirs)

    # ---------------------------------------------------------
    # 図8: 2パネル無次元化 輸送効率 プロット (Panel A: Linear, Panel B: Log-Log)
    # ---------------------------------------------------------
    fig_dim_2p, (ax_d_a, ax_d_b) = plt.subplots(1, 2, figsize=(15.6, 5.8))
    # Panel A (Linear y)
    for item in beads_data:
        st = item["binned_df"]
        if st.empty:
            continue
        ax_d_a.fill_between(st['tau_star'], st['eta_q25'], st['eta_q75'], color=item['color'], alpha=0.22, edgecolor='none')
        ax_d_a.plot(st['tau_star'], st['eta_median'], marker=item['marker'], linestyle='-', linewidth=2.0, markersize=6.8,
                    label=f"{item['d_um']:.2f} \u03bcm ($\u03be = {item['xi_um']:.1f}\\,\\mu\\mathrm{{m}}$)", color=item['color'])
    ax_d_a.axhline(1.0, color='#999999', linestyle='--', linewidth=1.2)
    ax_d_a.text(0.09, 1.03, r'$\eta = 1$ ($v_{\mathrm{eff}} = v_{\mathrm{MT}}$)', fontsize=10.0, color='#666666')
    ax_d_a.set(
        xlim=(0.07, 250),
        ylim=(0.0, 1.15),
        xscale='log',
        xlabel=r'Dimensionless Time $\tau^* \equiv \tau \, v_{\mathrm{MT, syn}} / \xi_{\mathrm{bg}}$',
        ylabel=r'Transport Efficiency $\eta_{\mathrm{transport}}(\tau^*)$',
        title=r'(a) Linear Scale'
    )
    ax_d_a.legend(fontsize=9.2, loc='upper right', framealpha=0.90, markerscale=0.78)

    # Panel B (Log-Log)
    for item in beads_data:
        st = item["binned_df"]
        if st.empty:
            continue
        ax_d_b.fill_between(st['tau_star'], st['eta_q25'], st['eta_q75'], color=item['color'], alpha=0.22, edgecolor='none')
        ax_d_b.plot(st['tau_star'], st['eta_median'], marker=item['marker'], linestyle='-', linewidth=2.0, markersize=6.8,
                    label=f"{item['d_um']:.2f} \u03bcm ($\u03be = {item['xi_um']:.1f}\\,\\mu\\mathrm{{m}}$)", color=item['color'])
    ax_d_b.axhline(1.0, color='#999999', linestyle='--', linewidth=1.2)
    ax_d_b.plot([t_g1, t_g2], [0.82, 0.82], color='#333333', linestyle=':', linewidth=1.4)
    ax_d_b.text(0.18, 0.70, r'$\propto (\tau^*)^{0}$ ($\alpha=2$)', fontsize=10.5, color='#333333')
    ax_d_b.plot([t_d1, t_d2], [A_diff_star * (t_d1**-0.5), A_diff_star * (t_d2**-0.5)], color='#222222', linestyle='--', linewidth=1.5)
    ax_d_b.text(4.0, 0.35, r'$\propto (\tau^*)^{-0.5}$ ($\alpha=1$)', fontsize=10.5, color='#222222')
    ax_d_b.plot([t_c1, t_c2], [A_cage_star * (t_c1**-1.0), A_cage_star * (t_c2**-1.0)], color='#666666', linestyle='-.', linewidth=1.3)
    ax_d_b.text(12.0, 0.08, r'$\propto (\tau^*)^{-1.0}$ ($\alpha=0$)', fontsize=10.5, color='#555555')
    ax_d_b.set(
        xlim=(0.07, 250),
        ylim=(0.02, 1.5),
        xscale='log',
        yscale='log',
        xlabel=r'Dimensionless Time $\tau^* \equiv \tau \, v_{\mathrm{MT, syn}} / \xi_{\mathrm{bg}}$',
        ylabel=r'Transport Efficiency $\eta_{\mathrm{transport}}(\tau^*)$',
        title=r'(b) Log-Log Scale'
    )
    ax_d_b.legend(fontsize=9.2, loc='lower left', framealpha=0.90, markerscale=0.78)

    fig_dim_2p.tight_layout(rect=[0, 0, 1, 0.94])
    fig_dim_2p.suptitle(r'Freedman–Diaconis Binned Dimensionless Cargo Transport Efficiency $\eta_{\mathrm{transport}}(\tau^*) \equiv \frac{\sqrt{\langle \Delta r^2 \rangle}}{\xi_{\mathrm{bg}} \, \tau^*}$ vs $\tau^*$',
                        fontsize=13.0, fontweight='bold', y=0.98)
    save_figure_to_all(fig_dim_2p, "dimensionless_transport_efficiency_2panel", out_dirs)

    # ---------------------------------------------------------
    # 図9: 輸送効率 vs スケール半径 x = R_c / xi
    # ---------------------------------------------------------
    fig_x, ax_x = plt.subplots(figsize=(8.5, 6.2))
    target_taus = [40.0, 100.0, 300.0]
    tau_colors = ['#88CCEE', '#44AA99', '#117733']
    tau_markers = ['o', 's', '^']

    for tau_target, t_col, t_mk in zip(target_taus, tau_colors, tau_markers):
        x_vals = []
        eta_med_vals = []
        eta_q25_vals = []
        eta_q75_vals = []
        for item in beads_data:
            st = item["stats_df"]
            if st.empty:
                continue
            row = st[st['tau_s'] == tau_target]
            if not row.empty:
                r = row.iloc[0]
                rc = item['d_um'] / 2.0
                xi = item['xi_um']
                x_scale = rc / xi
                x_vals.append(x_scale)
                eta_med_vals.append(r['eta_median'])
                eta_q25_vals.append(r['eta_q25'])
                eta_q75_vals.append(r['eta_q75'])

        if x_vals:
            order = np.argsort(x_vals)
            x_arr = np.array(x_vals)[order]
            med_arr = np.array(eta_med_vals)[order]
            q25_arr = np.array(eta_q25_vals)[order]
            q75_arr = np.array(eta_q75_vals)[order]

            yerr_low = med_arr - q25_arr
            yerr_high = q75_arr - med_arr
            ax_x.errorbar(
                x_arr, med_arr, yerr=[yerr_low, yerr_high],
                fmt=f'-{t_mk}', color=t_col, linewidth=2.0, markersize=8, capsize=4,
                label=rf'$\tau = {int(tau_target)}\,\mathrm{{s}}$'
            )

    # 各ビーズのラベルを注記 (tau = 300 s の点近傍)
    for item in beads_data:
        rc = item['d_um'] / 2.0
        xi = item['xi_um']
        x_scale = rc / xi
        st = item["stats_df"]
        row300 = st[st['tau_s'] == 300.0]
        if not row300.empty:
            med_val = row300.iloc[0]['eta_median']
            ax_x.annotate(
                f"{item['d_um']:.2f} μm",
                (x_scale, med_val),
                textcoords="offset points",
                xytext=(0, -18),
                ha='center',
                fontsize=8.5,
                color=item['color'],
                fontweight='bold'
            )

    ax_x.legend(fontsize=10, loc='upper right', framealpha=0.90)
    ax_x.set(
        xscale='log',
        xlabel=r'Scaled Radius $x \equiv R_c / \xi_{\mathrm{bg}}$',
        ylabel=r'Transport Efficiency $\eta_{\mathrm{transport}}(\tau) = v_{\mathrm{eff}} / v_{\mathrm{MT, syn}}$',
        title=r'Cargo Transport Efficiency vs Scaled Radius $x = R_c / \xi_{\mathrm{bg}}$'
    )
    save_figure_to_all(fig_x, "transport_efficiency_vs_scaled_radius", out_dirs)

    print("\n--- Completed particle-synchronized transport efficiency and MSD analysis successfully! ---")


if __name__ == "__main__":
    main()
