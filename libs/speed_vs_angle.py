"""
libs/speed_vs_angle.py

貨物微粒子（蛍光ビーズ）の速さ v と進行方向角度変化 Δθ の相関解析を行うモジュールです。
- 各トラッキングステップにおける方向転換角 Δθ（符号付き / 絶対値）と移動速さ v のペア算出
- 角度ビンごとの平均速さプロファイル <v>(Δθ) および速さ比 R(Δθ) = <v>(Δθ)/<v> の集計
- 速さと角度変化の各種相関係数 (Pearson r, Spearman rho, 配向コサイン相関, 回帰傾き) の算出
- 2D結合確率密度 P(Δθ, v) の集計
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats


def wrap_angle_pi(angle_rad):
    """角度（ラジアン）を [-π, π] の範囲にラップする。"""
    return np.arctan2(np.sin(angle_rad), np.cos(angle_rad))


def calc_speed_and_turning_angle(df_tracks, tau=1, scale=0.11, frame_interval=4.0, unit='deg', speed_mode='mean', signed=False):
    """
    粒子軌跡データから、指定ラグタイム tau における角度変化 Δθ と速さ v (um/s) をペアで算出する。

    Parameters
    ----------
    df_tracks : pd.DataFrame
        'particle', 'frame', 'x', 'y' を含むトラッキングデータ
    tau : int, default 1
        ラグタイム（フレーム数）
    scale : float, default 0.11
        空間スケール (um/pixel)
    frame_interval : float, default 4.0
        フレーム時間間隔 (s)
    unit : str, default 'deg'
        角度の単位: 'deg' (-180 ~ 180度または 0 ~ 180度) または 'rad'
    speed_mode : str, default 'mean'
        速さの定義:
        'mean': 転換前後ステップの平均速さ (v1 + v2) / 2
        'incoming': 転換前ステップの速さ v1
        'outgoing': 転換後ステップの速さ v2
    signed : bool, default False
        True の場合は符号付き角度変化 Δθ ∈ [-180°, 180°]、False の場合は絶対角度変化 |Δθ| ∈ [0°, 180°]

    Returns
    -------
    angles : np.ndarray
        角度変化配列
    speeds : np.ndarray
        対応する速さ v [um/s] 配列
    """
    if df_tracks is None or df_tracks.empty:
        return np.array([]), np.array([])

    cols = ['particle', 'frame', 'x', 'y']
    for c in cols:
        if c not in df_tracks.columns:
            return np.array([]), np.array([])

    df = df_tracks[cols].sort_values(by=['particle', 'frame'])
    p = df['particle'].to_numpy()
    f = df['frame'].to_numpy()
    x = df['x'].to_numpy() * scale
    y = df['y'].to_numpy() * scale

    N = len(p)
    if N <= 2 * tau:
        return np.array([]), np.array([])

    # 3点 (t, t+tau, t+2tau)
    p0, p1, p2 = p[:-2 * tau], p[tau:-tau], p[2 * tau:]
    f0, f1, f2 = f[:-2 * tau], f[tau:-tau], f[2 * tau:]
    x0, x1, x2 = x[:-2 * tau], x[tau:-tau], x[2 * tau:]
    y0, y1, y2 = y[:-2 * tau], y[tau:-tau], y[2 * tau:]

    valid = (p0 == p1) & (p1 == p2) & (f1 == f0 + tau) & (f2 == f1 + tau)
    if not np.any(valid):
        return np.array([]), np.array([])

    dx1 = x1[valid] - x0[valid]
    dy1 = y1[valid] - y0[valid]
    dx2 = x2[valid] - x1[valid]
    dy2 = y2[valid] - y1[valid]

    dr1_sq = dx1**2 + dy1**2
    dr2_sq = dx2**2 + dy2**2
    non_zero = (dr1_sq > 1e-10) & (dr2_sq > 1e-10)
    if not np.any(non_zero):
        return np.array([]), np.array([])

    th1 = np.arctan2(dy1[non_zero], dx1[non_zero])
    th2 = np.arctan2(dy2[non_zero], dx2[non_zero])
    d_th = wrap_angle_pi(th2 - th1)
    if not signed:
        d_th = np.abs(d_th)

    dt_sec = tau * frame_interval
    v1 = np.sqrt(dr1_sq[non_zero]) / dt_sec
    v2 = np.sqrt(dr2_sq[non_zero]) / dt_sec

    if speed_mode == 'incoming':
        v = v1
    elif speed_mode == 'outgoing':
        v = v2
    else:
        v = (v1 + v2) / 2.0

    if unit == 'deg':
        d_th = np.rad2deg(d_th)

    return d_th, v


def calc_ensemble_speed_vs_angle(exp_pairs_list, bins=20, bin_range=None, signed=False, unit='deg'):
    """
    実験ごとの (角度変化, 速さ) ペアのリストから、角度ビンごとの平均速さプロファイル <v>(Δθ)
    および実験間標準偏差 (std) を算出する。
    """
    if not exp_pairs_list:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    if bin_range is None:
        if unit == 'deg':
            bin_range = (-180.0, 180.0) if signed else (0.0, 180.0)
        else:
            bin_range = (-np.pi, np.pi) if signed else (0.0, np.pi)

    bin_edges = np.linspace(bin_range[0], bin_range[1], bins + 1)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    exp_means = []
    total_counts = np.zeros(bins, dtype=int)

    for d_th, v in exp_pairs_list:
        if len(d_th) == 0 or len(v) == 0:
            continue
        valid = np.isfinite(d_th) & np.isfinite(v) & (d_th >= bin_range[0]) & (d_th <= bin_range[1])
        d_th_val = d_th[valid]
        v_val = v[valid]

        if len(d_th_val) < 5:
            continue

        bin_indices = np.digitize(d_th_val, bin_edges) - 1
        means_per_bin = np.full(bins, np.nan)
        for b in range(bins):
            in_b = (bin_indices == b)
            n_in_b = np.sum(in_b)
            total_counts[b] += n_in_b
            if n_in_b >= 2:
                means_per_bin[b] = np.mean(v_val[in_b])
        exp_means.append(means_per_bin)

    if not exp_means:
        return centers, np.full(bins, np.nan), np.full(bins, np.nan), total_counts, bin_edges

    exp_means = np.array(exp_means)
    with np.errstate(all='ignore'):
        mean_v = np.nanmean(exp_means, axis=0)
        std_v = np.nanstd(exp_means, axis=0, ddof=1) if len(exp_means) > 1 else np.zeros_like(mean_v)

    return centers, mean_v, std_v, total_counts, bin_edges


def calc_ensemble_speed_ratio_R(exp_pairs_list, bins=20, bin_range=None, signed=False, unit='deg'):
    """
    R(Δθ) = <v>(Δθ) / <v> のアンサンブル平均プロファイルおよび標準偏差を算出する。
    各実験内でまず R_exp(Δθ) = <v>_exp(Δθ) / <v>_exp を求め、実験間平均と標準偏差を集計。
    """
    if not exp_pairs_list:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    if bin_range is None:
        if unit == 'deg':
            bin_range = (-180.0, 180.0) if signed else (0.0, 180.0)
        else:
            bin_range = (-np.pi, np.pi) if signed else (0.0, np.pi)

    bin_edges = np.linspace(bin_range[0], bin_range[1], bins + 1)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    exp_R_list = []
    total_counts = np.zeros(bins, dtype=int)

    for d_th, v in exp_pairs_list:
        if len(d_th) == 0 or len(v) == 0:
            continue
        valid = np.isfinite(d_th) & np.isfinite(v) & (d_th >= bin_range[0]) & (d_th <= bin_range[1])
        d_th_val = d_th[valid]
        v_val = v[valid]

        if len(d_th_val) < 5:
            continue

        exp_mean_v = float(np.mean(v_val))
        if exp_mean_v <= 0:
            continue

        bin_indices = np.digitize(d_th_val, bin_edges) - 1
        R_per_bin = np.full(bins, np.nan)
        for b in range(bins):
            in_b = (bin_indices == b)
            n_in_b = np.sum(in_b)
            total_counts[b] += n_in_b
            if n_in_b >= 2:
                R_per_bin[b] = np.mean(v_val[in_b]) / exp_mean_v
        exp_R_list.append(R_per_bin)

    if not exp_R_list:
        return centers, np.full(bins, np.nan), np.full(bins, np.nan), total_counts, bin_edges

    exp_R_arr = np.array(exp_R_list)
    with np.errstate(all='ignore'):
        mean_R = np.nanmean(exp_R_arr, axis=0)
        std_R = np.nanstd(exp_R_arr, axis=0, ddof=1) if len(exp_R_list) > 1 else np.zeros_like(mean_R)

    return centers, mean_R, std_R, total_counts, bin_edges


def calc_speed_angle_correlation(angles, speeds, unit='deg'):
    """
    速さ v と方向転換角 |Δθ| の各種相関係数を算出する。
    """
    valid = np.isfinite(angles) & np.isfinite(speeds)
    a = np.asarray(angles)[valid]
    v = np.asarray(speeds)[valid]

    # 相関には絶対角度変化 |Δθ| を使用
    a_abs = np.abs(a)

    if len(a) < 10:
        return {
            'pearson_r': np.nan, 'pearson_pvalue': np.nan,
            'spearman_rho': np.nan, 'spearman_pvalue': np.nan,
            'cos_r': np.nan, 'linear_slope': np.nan, 'normalized_slope': np.nan,
            'n_points': len(a),
            'mean_speed': np.nan, 'std_speed': np.nan,
            'mean_angle': np.nan, 'std_angle': np.nan
        }

    pr_res = stats.pearsonr(a_abs, v)
    sr_res = stats.spearmanr(a_abs, v)

    a_rad = np.deg2rad(a_abs) if unit == 'deg' else a_abs
    cos_th = np.cos(a_rad)
    cos_res = stats.pearsonr(cos_th, v)

    slope, intercept, r_val, p_val, std_err = stats.linregress(a_abs, v)
    mean_v = float(np.mean(v))
    norm_slope = float(slope / mean_v) if mean_v > 0 else np.nan

    return {
        'pearson_r': float(pr_res.statistic),
        'pearson_pvalue': float(pr_res.pvalue),
        'spearman_rho': float(sr_res.statistic),
        'spearman_pvalue': float(sr_res.pvalue),
        'cos_r': float(cos_res.statistic),
        'linear_slope': float(slope),
        'normalized_slope': norm_slope,
        'n_points': len(a),
        'mean_speed': mean_v,
        'std_speed': float(np.std(v)),
        'mean_angle': float(np.mean(a_abs)),
        'std_angle': float(np.std(a_abs))
    }


def calc_speed_contrast_delta_v(exp_pairs_list, angle_threshold_deg=20.0, unit='deg'):
    """
    直進時（0°付近）と反転時（180°付近）の速さの差および規格化速さコントラスト:
    Δv = (<v>(0°) - <v>(180°)) / <v> = R(0°) - R(180°)
    を実験間アンサンブル平均および標準誤差で算出する。

    Parameters
    ----------
    exp_pairs_list : list of (angles, speeds) tuples
    angle_threshold_deg : float, default 20.0
        直進・反転とみなす角度の許容範囲 [deg]
        直進: |Δθ| <= angle_threshold_deg
        反転: |Δθ| >= 180.0 - angle_threshold_deg
    unit : str, default 'deg'

    Returns
    -------
    dict
        'delta_v_norm_mean': 実験間平均 ( <v>(0°) - <v>(180°) ) / <v>
        'delta_v_norm_std': 実験間標準偏差
        'delta_v_norm_sem': 実験間標準誤差
        'delta_v_abs_mean': 実験間平均 <v>(0°) - <v>(180°) [um/s]
        'delta_v_abs_std': 実験間標準偏差
        'v_0_mean': <v>(0°) 平均
        'v_180_mean': <v>(180°) 平均
        'v_overall_mean': <v> 平均
        'R_0_mean': R(0°) 平均
        'R_180_mean': R(180°) 平均
        'n_experiments': 実験数
    """
    if not exp_pairs_list:
        return {
            'delta_v_norm_mean': np.nan, 'delta_v_norm_std': np.nan, 'delta_v_norm_sem': np.nan,
            'delta_v_abs_mean': np.nan, 'delta_v_abs_std': np.nan,
            'v_0_mean': np.nan, 'v_180_mean': np.nan, 'v_overall_mean': np.nan,
            'R_0_mean': np.nan, 'R_180_mean': np.nan, 'n_experiments': 0
        }

    th_fwd = np.deg2rad(angle_threshold_deg) if unit == 'rad' else angle_threshold_deg
    th_rev = (np.pi - np.deg2rad(angle_threshold_deg)) if unit == 'rad' else (180.0 - angle_threshold_deg)

    exp_delta_norm = []
    exp_delta_abs = []
    exp_v0 = []
    exp_v180 = []
    exp_v_all = []
    exp_R0 = []
    exp_R180 = []

    for d_th, v in exp_pairs_list:
        if len(d_th) == 0 or len(v) == 0:
            continue
        valid = np.isfinite(d_th) & np.isfinite(v)
        d_th_val = np.abs(d_th[valid])
        v_val = v[valid]

        if len(d_th_val) < 10:
            continue

        mean_v_exp = float(np.mean(v_val))
        if mean_v_exp <= 0:
            continue

        fwd_mask = (d_th_val <= th_fwd)
        rev_mask = (d_th_val >= th_rev)

        if np.sum(fwd_mask) >= 3 and np.sum(rev_mask) >= 3:
            v_0 = float(np.mean(v_val[fwd_mask]))
            v_180 = float(np.mean(v_val[rev_mask]))
            d_v_abs = v_0 - v_180
            d_v_norm = (v_0 - v_180) / mean_v_exp
            R_0 = v_0 / mean_v_exp
            R_180 = v_180 / mean_v_exp

            exp_delta_norm.append(d_v_norm)
            exp_delta_abs.append(d_v_abs)
            exp_v0.append(v_0)
            exp_v180.append(v_180)
            exp_v_all.append(mean_v_exp)
            exp_R0.append(R_0)
            exp_R180.append(R_180)

    n_exp = len(exp_delta_norm)
    if n_exp == 0:
        return {
            'delta_v_norm_mean': np.nan, 'delta_v_norm_std': np.nan, 'delta_v_norm_sem': np.nan,
            'delta_v_abs_mean': np.nan, 'delta_v_abs_std': np.nan,
            'v_0_mean': np.nan, 'v_180_mean': np.nan, 'v_overall_mean': np.nan,
            'R_0_mean': np.nan, 'R_180_mean': np.nan, 'n_experiments': 0
        }

    d_norm_arr = np.array(exp_delta_norm)
    d_abs_arr = np.array(exp_delta_abs)

    return {
        'delta_v_norm_mean': float(np.mean(d_norm_arr)),
        'delta_v_norm_std': float(np.std(d_norm_arr, ddof=1)) if n_exp > 1 else 0.0,
        'delta_v_norm_sem': float(np.std(d_norm_arr, ddof=1) / np.sqrt(n_exp)) if n_exp > 1 else 0.0,
        'delta_v_abs_mean': float(np.mean(d_abs_arr)),
        'delta_v_abs_std': float(np.std(d_abs_arr, ddof=1)) if n_exp > 1 else 0.0,
        'v_0_mean': float(np.mean(exp_v0)),
        'v_180_mean': float(np.mean(exp_v180)),
        'v_overall_mean': float(np.mean(exp_v_all)),
        'R_0_mean': float(np.mean(exp_R0)),
        'R_180_mean': float(np.mean(exp_R180)),
        'n_experiments': n_exp
    }
