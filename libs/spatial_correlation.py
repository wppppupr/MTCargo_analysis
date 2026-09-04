"""
libs/spatial_correlation.py

貨物微粒子の運動軌跡および推定された運動モード（HMM 状態: Run / Tumble）に基づき、
モード別の空間配向相関（Spatial Orientational / Velocity Correlation）
C(r) = < v_i · v_j >_{r_{ij} ≈ r}
を計算・集計・フィッティング・可視化するためのモジュールです。
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# モードペアの定義とラベル・色設定
MODE_PAIR_NAMES = {
    'run_run': 'Run - Run',
    'tumble_tumble': 'Tumble - Tumble',
    'run_tumble': 'Run - Tumble (Cross)',
    'all': 'All Pairs',
}

MODE_PAIR_COLORS = {
    'run_run': '#1b9e77',         # 青緑系 (Run)
    'tumble_tumble': '#d95f02',   # オレンジ系 (Tumble)
    'run_tumble': '#7570b3',      # 紫系 (Cross)
    'all': '#333333',             # 黒/濃灰系 (全体)
}

MODE_PAIR_STYLES = {
    'run_run': '-',
    'tumble_tumble': '-',
    'run_tumble': '--',
    'all': ':',
}


def exp_decay_model(r: np.ndarray, xi: float, a: float = 1.0, c0: float = 0.0) -> np.ndarray:
    """指数減衰モデル: C(r) = a * exp(-r / xi) + c0"""
    return a * np.exp(-r / np.maximum(xi, 1e-6)) + c0


def gaussian_decay_model(r: np.ndarray, xi: float, a: float = 1.0, c0: float = 0.0) -> np.ndarray:
    """ガウス減衰モデル: C(r) = a * exp(-(r / xi)^2) + c0"""
    return a * np.exp(-((r / np.maximum(xi, 1e-6)) ** 2)) + c0


def compute_frame_pairs_correlation(
    df_frame: pd.DataFrame,
    normalize: bool = True,
    min_speed: float = 1e-6,
) -> List[dict]:
    """
    1つのフレーム内に存在する全粒子ペアについて、距離と運動方向の内積（cos類似度）を計算する。

    Parameters
    ----------
    df_frame : pd.DataFrame
        'x_um', 'y_um', 'dx_um', 'dy_um', 'pred_state', 'particle' を含むDataFrame
    normalize : bool, default True
        Trueの場合、変位ベクトルを正規化して cos(theta_i - theta_j) を計算
    min_speed : float, default 1e-6
        変位ノルムがこの値以下の場合は方向不定としてスキップ

    Returns
    -------
    pair_records : List[dict]
        各ペアの距離、相関値、モード種別を含むレコードのリスト
    """
    n_particles = len(df_frame)
    if n_particles < 2:
        return []

    x = df_frame['x_um'].to_numpy()
    y = df_frame['y_um'].to_numpy()
    dx = df_frame['dx_um'].to_numpy()
    dy = df_frame['dy_um'].to_numpy()
    states = df_frame['pred_state'].to_numpy() if 'pred_state' in df_frame.columns else np.zeros(n_particles, dtype=int)
    particles = df_frame['particle'].to_numpy()

    # 変位ベクトルのノルム
    v_norm = np.hypot(dx, dy)
    valid_motion = v_norm > min_speed

    if np.count_nonzero(valid_motion) < 2:
        return []

    # 単位方向ベクトル
    if normalize:
        with np.errstate(divide='ignore', invalid='ignore'):
            ux = np.where(valid_motion, dx / v_norm, 0.0)
            uy = np.where(valid_motion, dy / v_norm, 0.0)
    else:
        ux = dx
        uy = dy

    # 座標差行列と距離行列の計算 (N, N)
    dx_mat = x[:, np.newaxis] - x[np.newaxis, :]
    dy_mat = y[:, np.newaxis] - y[np.newaxis, :]
    dist_mat = np.hypot(dx_mat, dy_mat)

    # 内積行列 (N, N)
    corr_mat = ux[:, np.newaxis] * ux[np.newaxis, :] + uy[:, np.newaxis] * uy[np.newaxis, :]

    # 上三角成分 (i < j) を抽出
    i_idx, j_idx = np.triu_indices(n_particles, k=1)

    # 両方の粒子が有効な運動ベクトルを持つペアのみ抽出
    valid_pairs = valid_motion[i_idx] & valid_motion[j_idx]
    if not np.any(valid_pairs):
        return []

    i_valid = i_idx[valid_pairs]
    j_valid = j_idx[valid_pairs]

    dists = dist_mat[i_valid, j_valid]
    corrs = corr_mat[i_valid, j_valid]
    s_i = states[i_valid]
    s_j = states[j_valid]
    p_i = particles[i_valid]
    p_j = particles[j_valid]

    records = []
    for k in range(len(dists)):
        si = s_i[k]
        sj = s_j[k]

        if si == 1 and sj == 1:
            mode_cat = 'run_run'
        elif si == 0 and sj == 0:
            mode_cat = 'tumble_tumble'
        else:
            mode_cat = 'run_tumble'

        records.append({
            'distance_um': dists[k],
            'correlation': corrs[k],
            'state_i': si,
            'state_j': sj,
            'mode_category': mode_cat,
            'particle_i': p_i[k],
            'particle_j': p_j[k],
        })

    return records


def compute_dataset_spatial_correlation(
    df_obs: pd.DataFrame,
    normalize: bool = True,
    min_speed: float = 1e-6,
) -> pd.DataFrame:
    """
    観測データセット全体（全実験・全フレーム）に対してモード別空間配向相関ペアデータを計算する。

    Parameters
    ----------
    df_obs : pd.DataFrame
        'frame', 'x_um', 'y_um', 'dx_um', 'dy_um', 'pred_state', 'particle', 'exp_dir' を含むDataFrame

    Returns
    -------
    df_pairs : pd.DataFrame
        全ペアの距離、相関値、モードカテゴリ、フレーム、実験名を含むDataFrame
    """
    if df_obs.empty:
        return pd.DataFrame()

    all_records = []
    group_cols = ['exp_dir', 'frame'] if 'exp_dir' in df_obs.columns else ['frame']

    for gkeys, sub_group in df_obs.groupby(group_cols):
        records = compute_frame_pairs_correlation(
            sub_group,
            normalize=normalize,
            min_speed=min_speed,
        )
        if not records:
            continue

        if isinstance(gkeys, tuple):
            if len(gkeys) >= 2:
                exp_dir_val, frame_val = gkeys[0], gkeys[1]
            else:
                exp_dir_val, frame_val = 'default', gkeys[0]
        else:
            exp_dir_val, frame_val = 'default', gkeys

        for r in records:
            r['exp_dir'] = exp_dir_val
            r['frame'] = frame_val
            all_records.append(r)

    if not all_records:
        return pd.DataFrame()

    return pd.DataFrame(all_records)


def bin_spatial_correlation(
    df_pairs: pd.DataFrame,
    bin_width: float = 2.0,
    max_dist: float = 60.0,
    min_pairs_per_bin: int = 5,
) -> pd.DataFrame:
    """
    ペア距離ごとに離散化し、各モード（Run-Run, Tumble-Tumble, Run-Tumble, All）の
    平均相関 C(r)、標準偏差、SEM、サンプル数を集計する。

    Parameters
    ----------
    df_pairs : pd.DataFrame
        'distance_um', 'correlation', 'mode_category' を含むDataFrame
    bin_width : float, default 2.0
        距離ビン幅 (um)
    max_dist : float, default 60.0
        最大集計距離 (um)
    min_pairs_per_bin : int, default 5
        有効とする最小ペア数

    Returns
    -------
    df_binned : pd.DataFrame
        'mode_category', 'distance_bin_center', 'mean_corr', 'std_corr', 'sem_corr', 'n_pairs' を含むDataFrame
    """
    if df_pairs.empty:
        return pd.DataFrame()

    bin_edges = np.arange(0, max_dist + bin_width, bin_width)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    categories = ['run_run', 'tumble_tumble', 'run_tumble', 'all']
    results = []

    for cat in categories:
        if cat == 'all':
            sub = df_pairs[(df_pairs['distance_um'] >= 0) & (df_pairs['distance_um'] <= max_dist)]
        else:
            sub = df_pairs[
                (df_pairs['mode_category'] == cat) &
                (df_pairs['distance_um'] >= 0) &
                (df_pairs['distance_um'] <= max_dist)
            ]

        if sub.empty:
            continue

        dists = sub['distance_um'].to_numpy()
        corrs = sub['correlation'].to_numpy()

        bin_indices = np.digitize(dists, bin_edges) - 1

        for b_idx, b_center in enumerate(bin_centers):
            mask = (bin_indices == b_idx)
            n_pts = np.count_nonzero(mask)
            if n_pts < min_pairs_per_bin:
                continue

            vals = corrs[mask]
            mean_v = np.mean(vals)
            std_v = np.std(vals, ddof=1) if n_pts > 1 else 0.0
            sem_v = std_v / np.sqrt(n_pts) if n_pts > 0 else 0.0

            results.append({
                'mode_category': cat,
                'mode_label': MODE_PAIR_NAMES[cat],
                'distance_um': b_center,
                'mean_correlation': mean_v,
                'std_correlation': std_v,
                'sem_correlation': sem_v,
                'n_pairs': n_pts,
            })

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results)


def fit_correlation_length(
    df_mode_curve: pd.DataFrame,
    model_type: str = 'exponential',
    max_fit_dist: Optional[float] = None,
    allow_offset: bool = False,
) -> Dict[str, Union[float, np.ndarray]]:
    """
    C(r) 曲線に対して指数減衰モデルをフィッティングし、配向相関長 xi を算出する。

    Parameters
    ----------
    df_mode_curve : pd.DataFrame
        'distance_um', 'mean_correlation', 'sem_correlation' を含むDataFrame
    model_type : str, default 'exponential'
        'exponential' または 'gaussian'
    max_fit_dist : float, optional
        フィッティングに使用する最大距離 (um)
    allow_offset : bool, default False
        Trueの場合、一定オフセット c0 を含めてフィッティング

    Returns
    -------
    fit_res : dict
        'xi_um', 'xi_err_um', 'amplitude', 'c0', 'r2', 'fit_r', 'fit_c'
    """
    if df_mode_curve.empty:
        return {'xi_um': np.nan, 'xi_err_um': np.nan, 'amplitude': np.nan, 'c0': np.nan, 'r2': np.nan}

    df_fit = df_mode_curve.copy()
    if max_fit_dist is not None:
        df_fit = df_fit[df_fit['distance_um'] <= max_fit_dist]

    df_fit = df_fit.dropna(subset=['distance_um', 'mean_correlation'])
    if len(df_fit) < 3:
        return {'xi_um': np.nan, 'xi_err_um': np.nan, 'amplitude': np.nan, 'c0': np.nan, 'r2': np.nan}

    r_vals = df_fit['distance_um'].to_numpy()
    c_vals = df_fit['mean_correlation'].to_numpy()
    weights = df_fit['sem_correlation'].to_numpy()
    sigma = np.where(weights > 1e-6, weights, 1e-3) if len(weights) > 0 else None

    # 初期値推定
    init_a = float(np.clip(c_vals[0] if len(c_vals) > 0 else 1.0, 0.05, 1.0))
    init_xi = 15.0

    if allow_offset:
        fit_func = exp_decay_model if model_type == 'exponential' else gaussian_decay_model
        p0 = [init_xi, init_a, 0.0]
        bounds = ([0.1, 0.0, -1.0], [500.0, 2.0, 1.0])
    else:
        if model_type == 'exponential':
            def fit_func(r, xi, a):
                return exp_decay_model(r, xi, a, 0.0)
        else:
            def fit_func(r, xi, a):
                return gaussian_decay_model(r, xi, a, 0.0)
        p0 = [init_xi, init_a]
        bounds = ([0.1, 0.0], [500.0, 2.0])

    try:
        popt, pcov = curve_fit(
            fit_func,
            r_vals,
            c_vals,
            p0=p0,
            bounds=bounds,
            sigma=sigma,
            maxfev=5000,
        )
        perr = np.sqrt(np.diag(pcov)) if pcov is not None else [0.0] * len(popt)
        xi_val = float(popt[0])
        xi_err = float(perr[0])
        a_val = float(popt[1])
        c0_val = float(popt[2]) if allow_offset else 0.0

        # R^2 算出
        c_pred = fit_func(r_vals, *popt)
        ss_res = np.sum((c_vals - c_pred) ** 2)
        ss_tot = np.sum((c_vals - np.mean(c_vals)) ** 2)
        r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

        fit_r = np.linspace(np.min(r_vals), np.max(r_vals), 100)
        fit_c = fit_func(fit_r, *popt)

        return {
            'xi_um': xi_val,
            'xi_err_um': xi_err,
            'amplitude': a_val,
            'c0': c0_val,
            'r2': r2,
            'fit_r': fit_r,
            'fit_c': fit_c,
        }
    except Exception:
        return {'xi_um': np.nan, 'xi_err_um': np.nan, 'amplitude': np.nan, 'c0': np.nan, 'r2': np.nan}


def compute_short_range_order(
    df_pairs: pd.DataFrame,
    r_cutoff: float = 10.0,
) -> Dict[str, float]:
    """
    近接粒子間（r <= r_cutoff）における各モードの平均配向秩序度（cosθの平均値）を計算する。
    """
    if df_pairs.empty:
        return {cat: np.nan for cat in ['run_run', 'tumble_tumble', 'run_tumble', 'all']}

    sub = df_pairs[df_pairs['distance_um'] <= r_cutoff]
    results = {}
    for cat in ['run_run', 'tumble_tumble', 'run_tumble', 'all']:
        if cat == 'all':
            cat_sub = sub
        else:
            cat_sub = sub[sub['mode_category'] == cat]

        if len(cat_sub) >= 3:
            results[cat] = float(cat_sub['correlation'].mean())
        else:
            results[cat] = np.nan

    return results


def plot_mode_correlations_single_axis(
    df_binned: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
    title: str = "",
    fit_curves: bool = True,
    show_legend: bool = True,
) -> plt.Axes:
    """
    単一軸に 4つのモード（Run-Run, Tumble-Tumble, Run-Tumble, All）の C(r) 曲線を描画する。
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4.5))

    if df_binned.empty:
        ax.set_title(title)
        return ax

    for cat in ['run_run', 'tumble_tumble', 'run_tumble', 'all']:
        sub = df_binned[df_binned['mode_category'] == cat]
        if sub.empty:
            continue

        r = sub['distance_um'].to_numpy()
        c = sub['mean_correlation'].to_numpy()
        sem = sub['sem_correlation'].to_numpy()

        color = MODE_PAIR_COLORS.get(cat, 'black')
        label = MODE_PAIR_NAMES.get(cat, cat)
        ls = MODE_PAIR_STYLES.get(cat, '-')

        ax.errorbar(
            r, c, yerr=sem,
            label=label,
            color=color,
            fmt='o' if cat != 'all' else 's',
            markersize=4,
            linestyle=ls,
            linewidth=1.5,
            capsize=2,
            alpha=0.9,
        )

        if fit_curves and len(r) >= 3 and cat in ['run_run', 'all']:
            fit_res = fit_correlation_length(sub, model_type='exponential')
            if not np.isnan(fit_res.get('xi_um', np.nan)) and 'fit_r' in fit_res:
                ax.plot(
                    fit_res['fit_r'],
                    fit_res['fit_c'],
                    color=color,
                    linestyle='-',
                    linewidth=2.0,
                    alpha=0.6,
                    label=f"{label} Fit ($\\xi={fit_res['xi_um']:.1f}\\,\\mu\\mathrm{{m}}$)",
                )

    ax.axhline(0.0, color='gray', linestyle='--', linewidth=1.0, alpha=0.6)
    ax.set_xlabel(r"Interparticle Distance $r$ [$\mu\mathrm{m}$]", fontsize=11)
    ax.set_ylabel(r"Spatial Correlation $C(r) = \langle \hat{\mathbf{v}}_i \cdot \hat{\mathbf{v}}_j \rangle$", fontsize=11)
    ax.set_ylim(-0.3, 1.05)
    ax.grid(True, linestyle='--', alpha=0.4)
    if title:
        ax.set_title(title, fontsize=12, fontweight='bold')
    if show_legend:
        ax.legend(fontsize=9, framealpha=0.9, loc='upper right')

    return ax
