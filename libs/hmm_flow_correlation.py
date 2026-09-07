"""
libs/hmm_flow_correlation.py

微小管のアクティブオプティカルフロー結果（angular_correlation_w.zarr / angular_correlation_bg.zarr）と
貨物微粒子の1次元対数速力 Gaussian HMM 推定状態（Run / Tumble）を結合し、
運動モード別の微小管フロー空間配向相関およびビーズ-フロー相互作用相関を集計・フィッティング・可視化するためのモジュールです。
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from libs import hmm_cargo as hc

FLOW_MODE_NAMES = {
    'run': 'Run Particle Vicinity',
    'tumble': 'Tumble Particle Vicinity',
    'all': 'All Particle Vicinity',
    'bg': 'Background Flow (Bulk)',
}

FLOW_MODE_COLORS = {
    'run': '#1b9e77',      # 青緑 (Run)
    'tumble': '#d95f02',   # オレンジ (Tumble)
    'all': '#222222',      # 黒 (All)
    'bg': '#7570b3',       # 紫 / 灰 (Background)
}

FLOW_MODE_STYLES = {
    'run': '-',
    'tumble': '-',
    'all': '-',
    'bg': '--',
}


def exp_decay_model(r: np.ndarray, xi: float, a: float = 1.0, c0: float = 0.0) -> np.ndarray:
    """指数減衰モデル: C(r) = a * exp(-r / xi) + c0"""
    return a * np.exp(-r / np.maximum(xi, 1e-6)) + c0


def extract_experiment_mode_flow_correlations(
    exp_dir: Path,
    hmm_model: hc.CargoGaussianHMM,
    scale: float = 0.11,
    tau: int = 1,
    frame_interval: float = 4.0,
    epsilon: float = 1e-3,
) -> Optional[dict]:
    """
    1つの実験ディレクトリに対して、HMM 状態と微小管フロー空間相関 Zarr データをマッチングし、
    モード別の相関プロファイルを抽出する。

    Parameters
    ----------
    exp_dir : Path
        実験ディレクトリ (beads_tracks.csv, angular_correlation_w.zarr を含む)
    hmm_model : hc.CargoGaussianHMM
        学習済みの Gaussian HMM モデル
    scale : float, default 0.11
        空間スケール (um/pixel)

    Returns
    -------
    result : dict or None
        距離座標 (um)、Run/Tumble/All/BG の各相関配列 (距離 x サンプル数)
    """
    tracks_csv = exp_dir / "beads_tracks.csv"
    p_zarr_path = exp_dir / "angular_correlation_w.zarr"
    bg_zarr_path = exp_dir / "angular_correlation_bg.zarr"

    if not tracks_csv.exists() or not p_zarr_path.exists():
        return None

    try:
        df_tracks = pd.read_csv(tracks_csv)
    except Exception:
        return None

    X, lengths, df_obs = hc.extract_hmm_features(
        df_tracks,
        tau=tau,
        scale=scale,
        frame_interval=frame_interval,
        epsilon=epsilon,
    )

    if len(X) < 10:
        return None

    # HMM 状態予測 (0: Tumble, 1: Run)
    pred_states = hmm_model.predict(X, lengths=lengths)
    df_obs['pred_state'] = pred_states

    # (frame, particle) -> state のルックアップ辞書を作成
    state_map = {}
    for _, row in df_obs.iterrows():
        f = int(row['frame'])
        p = int(row['particle'])
        st = int(row['pred_state'])
        state_map[(f, p)] = st

    # 粒子相関 Zarr のロード
    try:
        ds_p = xr.open_zarr(str(p_zarr_path), consolidated=False)
    except Exception as e:
        print(f"[WARNING] Failed to open {p_zarr_path}: {e}")
        return None

    distances_px = ds_p.coords['distance'].values
    distances_um = distances_px * scale
    num_d = len(distances_px)

    frames = ds_p.coords['frame'].values
    particles = ds_p.coords['particle'].values

    # データ変数の取得
    has_par = 'angular_correlation_parallel' in ds_p
    has_perp = 'angular_correlation_perpendicular' in ds_p
    has_bead = 'bead_correlation' in ds_p
    has_bead_par = 'bead_correlation_parallel' in ds_p
    has_bead_perp = 'bead_correlation_perpendicular' in ds_p

    arr_total = ds_p['angular_correlation'].values  # (d, frame, particle)
    arr_par = ds_p['angular_correlation_parallel'].values if has_par else None
    arr_perp = ds_p['angular_correlation_perpendicular'].values if has_perp else None
    arr_bead = ds_p['bead_correlation'].values if has_bead else None
    arr_bead_par = ds_p['bead_correlation_parallel'].values if has_bead_par else None
    arr_bead_perp = ds_p['bead_correlation_perpendicular'].values if has_bead_perp else None

    # 各 (frame, particle) の状態マスクを作成
    run_mask = np.zeros((len(frames), len(particles)), dtype=bool)
    tumble_mask = np.zeros((len(frames), len(particles)), dtype=bool)
    all_mask = np.zeros((len(frames), len(particles)), dtype=bool)

    for f_idx, f_val in enumerate(frames):
        for p_idx, p_val in enumerate(particles):
            key = (int(f_val), int(p_val))
            if key in state_map:
                st = state_map[key]
                all_mask[f_idx, p_idx] = True
                if st == 1:
                    run_mask[f_idx, p_idx] = True
                elif st == 0:
                    tumble_mask[f_idx, p_idx] = True

    # モードごとに (distance, N_samples) の形式で抽出
    def extract_samples(arr, mask_2d):
        if arr is None or not np.any(mask_2d):
            return np.empty((num_d, 0), dtype=np.float32)
        # arr shape: (D, F, P) -> arr[:, mask_2d] shape: (D, N_valid)
        samples = arr[:, mask_2d]
        return samples

    run_flow_samples = extract_samples(arr_total, run_mask)
    tumble_flow_samples = extract_samples(arr_total, tumble_mask)
    all_flow_samples = extract_samples(arr_total, all_mask)

    run_par_samples = extract_samples(arr_par, run_mask)
    tumble_par_samples = extract_samples(arr_par, tumble_mask)
    run_perp_samples = extract_samples(arr_perp, run_mask)
    tumble_perp_samples = extract_samples(arr_perp, tumble_mask)

    run_bead_samples = extract_samples(arr_bead, run_mask)
    tumble_bead_samples = extract_samples(arr_bead, tumble_mask)
    run_bead_par_samples = extract_samples(arr_bead_par, run_mask)
    run_bead_perp_samples = extract_samples(arr_bead_perp, run_mask)

    # 背景相関 Zarr のロード
    bg_flow_samples = np.empty((num_d, 0), dtype=np.float32)
    if bg_zarr_path.exists():
        try:
            ds_bg = xr.open_zarr(str(bg_zarr_path), consolidated=False)
            if 'angular_correlation' in ds_bg:
                bg_arr = ds_bg['angular_correlation'].values  # (D, frame)
                # 距離座標が一致しているか確認
                if len(bg_arr) == num_d:
                    bg_flow_samples = bg_arr
        except Exception as e:
            print(f"[WARNING] Failed to load {bg_zarr_path}: {e}")

    return {
        'exp_dir': exp_dir.name,
        'distances_um': distances_um,
        'distances_px': distances_px,
        'run_flow': run_flow_samples,
        'tumble_flow': tumble_flow_samples,
        'all_flow': all_flow_samples,
        'bg_flow': bg_flow_samples,
        'run_par': run_par_samples,
        'tumble_par': tumble_par_samples,
        'run_perp': run_perp_samples,
        'tumble_perp': tumble_perp_samples,
        'run_bead': run_bead_samples,
        'tumble_bead': tumble_bead_samples,
        'run_bead_par': run_bead_par_samples,
        'run_bead_perp': run_bead_perp_samples,
    }


def aggregate_flow_correlation_dataset(
    exp_results: List[dict],
) -> Dict[str, pd.DataFrame]:
    """
    複数実験にわたるモード別微小管フロー相関データを集計し、
    各モードの距離依存性 DataFrame を生成する。

    Returns
    -------
    df_curves : pd.DataFrame
        'mode', 'distance_um', 'mean_correlation', 'sem_correlation', 'std_correlation', 'n_samples'
    """
    if not exp_results:
        return pd.DataFrame()

    distances_um = exp_results[0]['distances_um']
    num_d = len(distances_um)

    target_vars = [
        ('run', 'run_flow'),
        ('tumble', 'tumble_flow'),
        ('all', 'all_flow'),
        ('bg', 'bg_flow'),
        ('run_par', 'run_par'),
        ('tumble_par', 'tumble_par'),
        ('run_perp', 'run_perp'),
        ('tumble_perp', 'tumble_perp'),
        ('run_bead', 'run_bead'),
        ('tumble_bead', 'tumble_bead'),
        ('run_bead_par', 'run_bead_par'),
        ('run_bead_perp', 'run_bead_perp'),
    ]

    records = []

    for mode_key, data_field in target_vars:
        all_samples_per_dist = [[] for _ in range(num_d)]

        for res in exp_results:
            samples = res.get(data_field)
            if samples is None or samples.shape[1] == 0:
                continue

            for d_idx in range(num_d):
                vals = samples[d_idx, :]
                valid_vals = vals[~np.isnan(vals)]
                if len(valid_vals) > 0:
                    all_samples_per_dist[d_idx].extend(valid_vals)

        for d_idx, d_um in enumerate(distances_um):
            v_list = np.array(all_samples_per_dist[d_idx])
            n_pts = len(v_list)
            if n_pts < 3:
                continue

            mean_v = float(np.mean(v_list))
            std_v = float(np.std(v_list, ddof=1)) if n_pts > 1 else 0.0
            sem_v = float(std_v / np.sqrt(n_pts)) if n_pts > 0 else 0.0

            records.append({
                'mode': mode_key,
                'mode_label': FLOW_MODE_NAMES.get(mode_key, mode_key),
                'distance_um': float(d_um),
                'mean_correlation': mean_v,
                'std_correlation': std_v,
                'sem_correlation': sem_v,
                'n_samples': n_pts,
            })

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records)


def fit_flow_correlation_length(
    df_mode_curve: pd.DataFrame,
    min_fit_dist: float = 0.0,
    max_fit_dist: float = 20.0,
    min_corr_threshold: float = 0.01,
) -> Dict[str, Union[float, np.ndarray]]:
    """
    微小管フローの C(r) 曲線に対して y 軸を対数（ln(C(r))）に変換した上で
    指定された距離範囲 [min_fit_dist, max_fit_dist] において
    指数減衰モデル ln(C(r)) = ln(a) - r / xi を重み付き線形フィッティングし、配向相関長 xi を算出する。

    Parameters
    ----------
    df_mode_curve : pd.DataFrame
        'distance_um', 'mean_correlation', 'sem_correlation' を含む DataFrame
    min_fit_dist : float, default 0.0
        フィッティングに使用する最小距離 (um)
    max_fit_dist : float, default 20.0
        フィッティングに使用する最大距離 (um)
    min_corr_threshold : float, default 0.01
        対数をとるために必要な相関の最小閾値

    Returns
    -------
    result : dict
        'xi_um', 'xi_err_um', 'amplitude', 'c0', 'r2', 'r2_log', 'r_peak_um', 'r_fit_min_um', 'r_fit_max_um', 'fit_r', 'fit_c'
    """
    if df_mode_curve.empty:
        return {'xi_um': np.nan, 'xi_err_um': np.nan, 'amplitude': np.nan, 'c0': 0.0, 'r2': np.nan}

    df_fit = df_mode_curve.dropna(subset=['distance_um', 'mean_correlation']).copy()
    if len(df_fit) < 3:
        return {'xi_um': np.nan, 'xi_err_um': np.nan, 'amplitude': np.nan, 'c0': 0.0, 'r2': np.nan}

    r_all = df_fit['distance_um'].to_numpy()
    c_all = df_fit['mean_correlation'].to_numpy()
    sem_all = df_fit['sem_correlation'].to_numpy()

    # 1. 探索範囲内でのピーク位置 r_peak を検出（粒子マスク領域のゼロ回避）
    search_idx = np.where((r_all >= min_fit_dist) & (r_all <= max_fit_dist))[0]
    if len(search_idx) == 0:
        search_idx = np.where(r_all <= max_fit_dist)[0]
    if len(search_idx) == 0:
        search_idx = np.arange(min(len(r_all), 10))
    peak_idx = search_idx[np.argmax(c_all[search_idx])]
    r_peak = float(r_all[peak_idx])

    # 2. ピーク以降かつ min_fit_dist 以降 (r >= max(min_fit_dist, r_peak)) かつ r <= max_fit_dist かつ C(r) >= min_corr_threshold の減衰領域を抽出
    eff_min_r = max(min_fit_dist, r_peak)
    decay_mask = (r_all >= eff_min_r) & (r_all <= max_fit_dist) & (c_all >= min_corr_threshold)

    r_fit = r_all[decay_mask]
    c_fit = c_all[decay_mask]
    sem_fit = sem_all[decay_mask]

    if len(r_fit) < 3:
        return {'xi_um': np.nan, 'xi_err_um': np.nan, 'amplitude': np.nan, 'c0': 0.0, 'r2': np.nan}

    log_c = np.log(c_fit)

    # 誤差伝播による重み: sigma_log = sem / C
    sigma_log = np.where(sem_fit > 1e-6, sem_fit / np.maximum(c_fit, 1e-6), 0.1)
    w = 1.0 / np.maximum(sigma_log, 1e-4)

    try:
        poly, cov = np.polyfit(r_fit, log_c, deg=1, w=w, cov=True)
        slope, intercept = poly[0], poly[1]

        if slope < 0:
            xi_val = float(-1.0 / slope)
            xi_err = float((xi_val ** 2) * np.sqrt(cov[0, 0])) if cov is not None else np.nan
            a_val = float(np.exp(intercept))
        else:
            xi_val = np.nan
            xi_err = np.nan
            a_val = float(np.exp(intercept))

        c_pred = a_val * np.exp(-r_fit / xi_val) if not np.isnan(xi_val) else np.zeros_like(r_fit)
        ss_res = np.sum((c_fit - c_pred) ** 2)
        ss_tot = np.sum((c_fit - np.mean(c_fit)) ** 2)
        r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else np.nan

        log_pred = intercept + slope * r_fit
        ss_res_log = np.sum((log_c - log_pred) ** 2)
        ss_tot_log = np.sum((log_c - np.mean(log_c)) ** 2)
        r2_log = float(1.0 - (ss_res_log / ss_tot_log)) if ss_tot_log > 0 else np.nan

        fit_r = np.linspace(eff_min_r, max_fit_dist, 150)
        fit_c = a_val * np.exp(-fit_r / xi_val) if not np.isnan(xi_val) else np.full_like(fit_r, np.nan)

        return {
            'xi_um': xi_val,
            'xi_err_um': xi_err,
            'amplitude': a_val,
            'c0': 0.0,
            'r2': r2,
            'r2_log': r2_log,
            'r_peak_um': r_peak,
            'r_fit_min_um': float(r_fit[0]),
            'r_fit_max_um': float(r_fit[-1]),
            'fit_r': fit_r,
            'fit_c': fit_c,
        }
    except Exception:
        return {'xi_um': np.nan, 'xi_err_um': np.nan, 'amplitude': np.nan, 'c0': 0.0, 'r2': np.nan}


def plot_flow_correlations_single_axis(
    df_curves: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
    title: str = "",
    fit_curves: bool = True,
    min_fit_dist: float = 0.0,
    max_fit_dist: float = 20.0,
    show_legend: bool = True,
) -> plt.Axes:
    """
    単一軸に Run, Tumble, All, Background の微小管フロー空間相関曲線を描画する。
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4.5))

    if df_curves.empty:
        ax.set_title(title)
        return ax

    target_modes = ['run', 'tumble', 'all', 'bg']

    for mode in target_modes:
        sub = df_curves[df_curves['mode'] == mode]
        if sub.empty:
            continue

        r = sub['distance_um'].to_numpy()
        c = sub['mean_correlation'].to_numpy()
        sem = sub['sem_correlation'].to_numpy()

        color = FLOW_MODE_COLORS.get(mode, 'black')
        label = FLOW_MODE_NAMES.get(mode, mode)
        ls = FLOW_MODE_STYLES.get(mode, '-')

        ax.errorbar(
            r, c, yerr=sem,
            label=label,
            color=color,
            fmt='o' if mode != 'bg' else 's',
            markersize=3.5,
            linestyle=ls,
            linewidth=1.5,
            capsize=2,
            alpha=0.85,
        )

        if fit_curves and len(r) >= 3 and mode in ['run', 'bg']:
            fit_res = fit_flow_correlation_length(sub, min_fit_dist=min_fit_dist, max_fit_dist=max_fit_dist)
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
    ax.set_xlabel(r"Distance $r$ from Particle Center [$\mu\mathrm{m}$]", fontsize=11)
    ax.set_ylabel(r"Flow Angular Correlation $C_{\mathrm{flow}}(r)$", fontsize=11)
    ax.set_ylim(-0.2, 1.05)
    ax.grid(True, linestyle='--', alpha=0.4)
    if title:
        ax.set_title(title, fontsize=12, fontweight='bold')
    if show_legend:
        ax.legend(fontsize=8.5, framealpha=0.9, loc='upper right')

    return ax


# =========================================================================
# 粒子ごと（Per-Particle）の相関長抽出・可視化関数群
# =========================================================================

def extract_per_particle_flow_correlations(
    exp_dir: Path,
    hmm_model: hc.CargoGaussianHMM,
    scale: float = 0.11,
    tau: int = 1,
    frame_interval: float = 4.0,
    epsilon: float = 1e-3,
    min_fit_dist: float = 0.0,
    max_fit_dist: float = 20.0,
    min_frames: int = 10,
) -> List[dict]:
    """
    1つの実験ディレクトリ内の全粒子について、Run / Tumble 状態別の微小管フロー空間相関と
    相関長 xi_run, xi_tumble を抽出する。
    """
    tracks_csv = exp_dir / "beads_tracks.csv"
    p_zarr_path = exp_dir / "angular_correlation_w.zarr"

    if not tracks_csv.exists() or not p_zarr_path.exists():
        return []

    try:
        df_tracks = pd.read_csv(tracks_csv)
    except Exception:
        return []

    X, lengths, df_obs = hc.extract_hmm_features(
        df_tracks,
        tau=tau,
        scale=scale,
        frame_interval=frame_interval,
        epsilon=epsilon,
    )

    if len(X) < 10:
        return []

    df_obs['pred_state'] = hmm_model.predict(X, lengths=lengths)
    state_map = {
        (int(r['frame']), int(r['particle'])): int(r['pred_state'])
        for _, r in df_obs.iterrows()
    }

    try:
        ds_p = xr.open_zarr(str(p_zarr_path), consolidated=False)
    except Exception as e:
        print(f"[WARNING] Failed to open {p_zarr_path}: {e}")
        return []

    distances_um = ds_p.coords['distance'].values * scale
    frames = ds_p.coords['frame'].values
    particles = ds_p.coords['particle'].values
    arr_total = ds_p['angular_correlation'].values  # (dist, frame, particle)

    particle_results = []

    for p_idx, p_val in enumerate(particles):
        p_int = int(p_val)
        run_f_indices = []
        tumble_f_indices = []

        for f_idx, f_val in enumerate(frames):
            key = (int(f_val), p_int)
            if key in state_map:
                st = state_map[key]
                if st == 1:
                    run_f_indices.append(f_idx)
                elif st == 0:
                    tumble_f_indices.append(f_idx)

        n_run = len(run_f_indices)
        n_tumble = len(tumble_f_indices)
        n_total = n_run + n_tumble

        xi_run, xi_run_err, a_run, r2_run = np.nan, np.nan, np.nan, np.nan
        xi_tumble, xi_tumble_err, a_tumble, r2_tumble = np.nan, np.nan, np.nan, np.nan

        if n_run >= min_frames:
            c_run_samples = arr_total[:, run_f_indices, p_idx]  # (dist, n_run)
            c_run_mean = np.nanmean(c_run_samples, axis=1)
            c_run_std = np.nanstd(c_run_samples, axis=1, ddof=1) if n_run > 1 else np.zeros_like(c_run_mean)
            c_run_sem = c_run_std / np.sqrt(n_run)

            df_run = pd.DataFrame({
                'distance_um': distances_um,
                'mean_correlation': c_run_mean,
                'sem_correlation': c_run_sem,
            })
            fit_run = fit_flow_correlation_length(df_run, min_fit_dist=min_fit_dist, max_fit_dist=max_fit_dist)
            xi_run = fit_run.get('xi_um', np.nan)
            xi_run_err = fit_run.get('xi_err_um', np.nan)
            a_run = fit_run.get('amplitude', np.nan)
            r2_run = fit_run.get('r2', np.nan)

        if n_tumble >= min_frames:
            c_tumble_samples = arr_total[:, tumble_f_indices, p_idx]
            c_tumble_mean = np.nanmean(c_tumble_samples, axis=1)
            c_tumble_std = np.nanstd(c_tumble_samples, axis=1, ddof=1) if n_tumble > 1 else np.zeros_like(c_tumble_mean)
            c_tumble_sem = c_tumble_std / np.sqrt(n_tumble)

            df_tumble = pd.DataFrame({
                'distance_um': distances_um,
                'mean_correlation': c_tumble_mean,
                'sem_correlation': c_tumble_sem,
            })
            fit_tumble = fit_flow_correlation_length(df_tumble, min_fit_dist=min_fit_dist, max_fit_dist=max_fit_dist)
            xi_tumble = fit_tumble.get('xi_um', np.nan)
            xi_tumble_err = fit_tumble.get('xi_err_um', np.nan)
            a_tumble = fit_tumble.get('amplitude', np.nan)
            r2_tumble = fit_tumble.get('r2', np.nan)

        delta_xi = xi_run - xi_tumble if (np.isfinite(xi_run) and np.isfinite(xi_tumble)) else np.nan
        ratio_xi = xi_run / xi_tumble if (np.isfinite(xi_run) and np.isfinite(xi_tumble) and xi_tumble > 0) else np.nan

        particle_results.append({
            'exp_name': exp_dir.name,
            'particle': p_int,
            'n_frames_run': n_run,
            'n_frames_tumble': n_tumble,
            'n_frames_total': n_total,
            'xi_run_um': xi_run,
            'xi_run_err_um': xi_run_err,
            'a_run': a_run,
            'r2_run': r2_run,
            'xi_tumble_um': xi_tumble,
            'xi_tumble_err_um': xi_tumble_err,
            'a_tumble': a_tumble,
            'r2_tumble': r2_tumble,
            'delta_xi_um': delta_xi,
            'ratio_xi_run_to_tumble': ratio_xi,
        })

    return particle_results


def plot_per_particle_scatter_6panel(
    df_particles: pd.DataFrame,
    beads_info: List[dict],
    output_path: Path,
    max_xi: Optional[float] = None,
):
    """
    粒子ごとの xi_Run vs xi_Tumble 散布図（動的パネルレイアウト）。
    対角線 (y=x) を基準に Run 優位 (xi_Run > xi_Tumble) かどうかを可視化する。
    """
    n_plots = len(beads_info)
    ncols = min(n_plots, 3)
    nrows = int(np.ceil(n_plots / ncols)) if ncols > 0 else 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 4.8 * nrows), squeeze=False)
    axes = axes.flatten()

    for idx, binfo in enumerate(beads_info):
        ax = axes[idx]
        bname = binfo['name']
        dia = binfo['diameter_um']

        sub = df_particles[df_particles['bead_name'] == bname].dropna(subset=['xi_run_um', 'xi_tumble_um'])

        if sub.empty:
            ax.set_title(f"$d = {dia:.2f}\\,\\mu\\mathrm{{m}}$ (No data)", fontsize=12)
            ax.axis('off')
            continue

        x_tumble = sub['xi_tumble_um'].to_numpy()
        y_run = sub['xi_run_um'].to_numpy()
        n_pts = len(x_tumble)

        # 軸上限の設定
        if max_xi is None:
            curr_max = max(np.percentile(x_tumble, 98), np.percentile(y_run, 98)) * 1.25
            axis_max = max(15.0, curr_max)
        else:
            axis_max = max_xi

        # 対角線
        diag_line = np.linspace(0, axis_max, 100)
        ax.plot(diag_line, diag_line, color='gray', linestyle='--', linewidth=1.2, alpha=0.7, label=r"$\xi_{\mathrm{Run}} = \xi_{\mathrm{Tumble}}$")

        # Run 優位 (> Tumble) と Tumble 優位 (<= Run) の色分け
        is_run_higher = y_run > x_tumble
        n_higher = np.sum(is_run_higher)
        pct_higher = (n_higher / n_pts) * 100.0 if n_pts > 0 else 0.0

        ax.scatter(
            x_tumble[is_run_higher], y_run[is_run_higher],
            color=FLOW_MODE_COLORS['run'], s=45, alpha=0.85, edgecolors='black', linewidth=0.5,
            label=f"$\\xi_{{\\mathrm{{Run}}}} > \\xi_{{\\mathrm{{Tumble}}}}$ ({n_higher}/{n_pts})",
            zorder=3,
        )
        ax.scatter(
            x_tumble[~is_run_higher], y_run[~is_run_higher],
            color=FLOW_MODE_COLORS['tumble'], s=45, alpha=0.85, edgecolors='black', linewidth=0.5,
            label=f"$\\xi_{{\\mathrm{{Run}}}} \\leq \\xi_{{\\mathrm{{Tumble}}}}$ ({n_pts - n_higher}/{n_pts})",
            zorder=3,
        )

        # 統計検定 (Paired Wilcoxon signed-rank test: Run > Tumble)
        p_val_str = ""
        if n_pts >= 5:
            try:
                res_w = stats.wilcoxon(y_run, x_tumble, alternative='greater')
                p_val_str = f"Wilcoxon $p = {res_w.pvalue:.3g}$"
            except Exception:
                pass

        # テキスト注記
        annot_text = f"$N = {n_pts}$\nRun $>$ Tumble: {pct_higher:.1f}%"
        if p_val_str:
            annot_text += f"\n{p_val_str}"

        ax.text(
            0.05, 0.92, annot_text,
            transform=ax.transAxes,
            fontsize=9.5,
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85, edgecolor='lightgray'),
        )

        ax.set_xlim(0, axis_max)
        ax.set_ylim(0, axis_max)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.set_title(f"$d = {dia:.2f}\\,\\mu\\mathrm{{m}}$", fontsize=12, fontweight='bold')
        ax.set_xlabel(r"$\xi_{\mathrm{Tumble}}$ (Paused) [$\mu\mathrm{m}$]", fontsize=11)
        ax.set_ylabel(r"$\xi_{\mathrm{Run}}$ (Active) [$\mu\mathrm{m}$]", fontsize=11)

        if idx == 0:
            ax.legend(fontsize=8.5, loc='lower right', framealpha=0.9)

    for i in range(n_plots, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved 6-panel per-particle scatter plot: {output_path}")


def plot_per_particle_box_violin_vs_diameter(
    df_particles: pd.DataFrame,
    beads_info: List[dict],
    output_path: Path,
):
    """
    全粒子径にわたる粒子ごとの xi_Run と xi_Tumble のペア比較（Box & Paired Jitter Strip Plot）。
    各粒子の Run 状態と Tumble 状態を灰色の実線で結ぶことで、個体ごとの相関長変化を明示する。
    外れ値に頑健なノンパラメトリック検定（Wilcoxon 符号付順位検定: Run > Tumble）を行い、
    有意差ブラケットと p 値・スター（***: p<0.001, **: p<0.01, *: p<0.05, n.s.）を描画する。
    """
    fig, ax = plt.subplots(figsize=(max(7.0, 2.5 * len(beads_info)), 6.2))

    dia_positions = []
    width = 0.28

    valid_particles = df_particles.dropna(subset=['xi_run_um', 'xi_tumble_um'])

    for idx, binfo in enumerate(beads_info):
        bname = binfo['name']
        dia = binfo['diameter_um']
        pos = idx

        sub = valid_particles[valid_particles['bead_name'] == bname]
        if sub.empty:
            continue

        dia_positions.append((pos, f"{dia:.2f} $\\mu$m\n($N={len(sub)}$)"))

        pos_run = pos - width / 2.0
        pos_tumble = pos + width / 2.0

        xi_r = sub['xi_run_um'].to_numpy()
        xi_t = sub['xi_tumble_um'].to_numpy()

        # 1. 各粒子の Run - Tumble ペアを結ぶ線
        np.random.seed(42)
        jitter = (np.random.rand(len(sub)) - 0.5) * 0.08
        for i in range(len(sub)):
            ax.plot(
                [pos_run + jitter[i], pos_tumble + jitter[i]],
                [xi_r[i], xi_t[i]],
                color='gray', alpha=0.35, linewidth=1.0, zorder=2
            )

        # 2. Box plots
        bp_run = ax.boxplot(
            [xi_r], positions=[pos_run], widths=width * 0.8,
            patch_artist=True, showfliers=False,
            boxprops=dict(facecolor=FLOW_MODE_COLORS['run'], alpha=0.35, edgecolor=FLOW_MODE_COLORS['run'], linewidth=1.5),
            medianprops=dict(color='black', linewidth=2.0),
            whiskerprops=dict(color=FLOW_MODE_COLORS['run'], linewidth=1.2),
            capprops=dict(color=FLOW_MODE_COLORS['run'], linewidth=1.2),
        )
        bp_tumble = ax.boxplot(
            [xi_t], positions=[pos_tumble], widths=width * 0.8,
            patch_artist=True, showfliers=False,
            boxprops=dict(facecolor=FLOW_MODE_COLORS['tumble'], alpha=0.35, edgecolor=FLOW_MODE_COLORS['tumble'], linewidth=1.5),
            medianprops=dict(color='black', linewidth=2.0),
            whiskerprops=dict(color=FLOW_MODE_COLORS['tumble'], linewidth=1.2),
            capprops=dict(color=FLOW_MODE_COLORS['tumble'], linewidth=1.2),
        )

        # 3. Scatter points
        ax.scatter(
            pos_run + jitter, xi_r,
            color=FLOW_MODE_COLORS['run'], s=35, alpha=0.85, edgecolors='black', linewidth=0.5,
            label='Run Mode' if idx == 0 else "", zorder=3,
        )
        ax.scatter(
            pos_tumble + jitter, xi_t,
            color=FLOW_MODE_COLORS['tumble'], s=35, alpha=0.85, edgecolors='black', linewidth=0.5,
            label='Tumble Mode' if idx == 0 else "", zorder=3,
        )

        # 4. ノンパラメトリック Wilcoxon 符号付順位検定 (Run > Tumble) & 有意差ブラケット
        if len(sub) >= 5:
            try:
                res_w = stats.wilcoxon(xi_r, xi_t, alternative='greater')
                p_val = float(res_w.pvalue)

                # スター表記
                if p_val < 0.001:
                    stars = "***"
                elif p_val < 0.01:
                    stars = "**"
                elif p_val < 0.05:
                    stars = "*"
                else:
                    stars = "n.s."

                # ブラケットの y 座標位置を算出
                filtered_vals = [v for v in np.concatenate([xi_r, xi_t]) if v <= 24.0]
                local_max = max(filtered_vals) if filtered_vals else 18.0
                y_bar = min(24.5, local_max + 1.8)
                bar_h = 0.5

                # ブラケット線
                ax.plot(
                    [pos_run, pos_run, pos_tumble, pos_tumble],
                    [y_bar - bar_h, y_bar, y_bar, y_bar - bar_h],
                    color='#333333', linewidth=1.2, zorder=4
                )

                # テキストラベル (p値とスター)
                p_str = f"p = {p_val:.3f}" if p_val >= 0.001 else f"p = {p_val:.2e}"
                label_text = f"{stars}\n({p_str})"
                ax.text(
                    pos, y_bar + 0.3, label_text,
                    ha='center', va='bottom', fontsize=9.0, fontweight='bold' if stars != 'n.s.' else 'normal',
                    color='#111111' if stars != 'n.s.' else '#666666',
                    zorder=4
                )
            except Exception as e:
                pass

    if dia_positions:
        ax.set_xticks([p[0] for p in dia_positions])
        ax.set_xticklabels([p[1] for p in dia_positions], fontsize=10.5)

    ax.set_ylabel(r"Flow Orientational Correlation Length $\xi^{(p)}$ [$\mu\mathrm{m}$]", fontsize=12)
    ax.set_title("Per-Particle MT Flow Correlation Length: Paired Wilcoxon Test (Run > Tumble)", fontsize=13, fontweight='bold')
    ax.set_ylim(0, 30)
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    ax.legend(fontsize=10.5, loc='upper right', framealpha=0.9)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved per-particle box/jitter plot: {output_path}")


def plot_per_particle_diff_and_ratio_vs_diameter(
    df_particles: pd.DataFrame,
    beads_info: List[dict],
    output_path: Path,
):
    """
    粒子ごとの相関長差 Delta xi = xi_Run - xi_Tumble および
    相関長比 Ratio = xi_Run / xi_Tumble vs 粒子径のプロット（2パネル）。
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    valid_particles = df_particles.dropna(subset=['delta_xi_um', 'ratio_xi_run_to_tumble'])

    dia_list = []
    mean_diffs = []
    sem_diffs = []
    mean_ratios = []
    sem_ratios = []

    for binfo in beads_info:
        bname = binfo['name']
        dia = binfo['diameter_um']
        col = binfo['color']
        m = binfo['marker']

        sub = valid_particles[valid_particles['bead_name'] == bname]
        if sub.empty:
            continue

        d_vals = sub['delta_xi_um'].to_numpy()
        r_vals = sub['ratio_xi_run_to_tumble'].to_numpy()

        dia_list.append(dia)
        mean_diffs.append(np.mean(d_vals))
        sem_diffs.append(np.std(d_vals, ddof=1) / np.sqrt(len(d_vals)) if len(d_vals) > 1 else 0.0)
        mean_ratios.append(np.mean(r_vals))
        sem_ratios.append(np.std(r_vals, ddof=1) / np.sqrt(len(r_vals)) if len(r_vals) > 1 else 0.0)

        # 個別粒子のプロット (Jitter)
        jitter_x = dia * (1.0 + (np.random.rand(len(d_vals)) - 0.5) * 0.08)
        axes[0].scatter(jitter_x, d_vals, color=col, marker=m, s=35, alpha=0.55, edgecolors='black', linewidth=0.4)
        axes[1].scatter(jitter_x, r_vals, color=col, marker=m, s=35, alpha=0.55, edgecolors='black', linewidth=0.4)

    # 集約平均・SEM エラーバー
    if dia_list:
        axes[0].errorbar(
            dia_list, mean_diffs, yerr=sem_diffs,
            color='black', fmt='-o', markersize=6, linewidth=2.0, capsize=4, label='Mean $\\pm$ SEM', zorder=4
        )
        axes[1].errorbar(
            dia_list, mean_ratios, yerr=sem_ratios,
            color='black', fmt='-o', markersize=6, linewidth=2.0, capsize=4, label='Mean $\\pm$ SEM', zorder=4
        )

    # Panel 0: Delta xi
    axes[0].axhline(0.0, color='red', linestyle='--', linewidth=1.2, alpha=0.7, label=r"No Difference ($\Delta \xi = 0$)")
    axes[0].set_xscale('log')
    axes[0].set_xlabel(r"Cargo Bead Diameter $d$ [$\mu\mathrm{m}$]", fontsize=11.5)
    axes[0].set_ylabel(r"Correlation Length Difference $\Delta \xi = \xi_{\mathrm{Run}} - \xi_{\mathrm{Tumble}}$ [$\mu\mathrm{m}$]", fontsize=11.5)
    axes[0].set_title("Per-Particle Difference in Flow Correlation Length", fontsize=12, fontweight='bold')
    axes[0].grid(True, which="both", linestyle='--', alpha=0.4)
    axes[0].legend(fontsize=9.5, loc='upper right')

    # Panel 1: Ratio
    axes[1].axhline(1.0, color='red', linestyle='--', linewidth=1.2, alpha=0.7, label=r"Equal ($\xi_{\mathrm{Run}} / \xi_{\mathrm{Tumble}} = 1$)")
    axes[1].set_xscale('log')
    axes[1].set_xlabel(r"Cargo Bead Diameter $d$ [$\mu\mathrm{m}$]", fontsize=11.5)
    axes[1].set_ylabel(r"Correlation Length Ratio $\xi_{\mathrm{Run}} / \xi_{\mathrm{Tumble}}$", fontsize=11.5)
    axes[1].set_title("Per-Particle Ratio of Flow Correlation Length", fontsize=12, fontweight='bold')
    axes[1].grid(True, which="both", linestyle='--', alpha=0.4)
    axes[1].legend(fontsize=9.5, loc='upper right')

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved per-particle diff & ratio plot: {output_path}")


def plot_per_particle_cdf_6panel(
    df_particles: pd.DataFrame,
    beads_info: List[dict],
    output_path: Path,
):
    """
    粒子ごとの xi_Run と xi_Tumble の累積確率分布 (eCDF) 比較（動的パネルレイアウト）。
    """
    n_plots = len(beads_info)
    ncols = min(n_plots, 3)
    nrows = int(np.ceil(n_plots / ncols)) if ncols > 0 else 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 4.8 * nrows), squeeze=False)
    axes = axes.flatten()

    for idx, binfo in enumerate(beads_info):
        ax = axes[idx]
        bname = binfo['name']
        dia = binfo['diameter_um']

        sub = df_particles[df_particles['bead_name'] == bname].dropna(subset=['xi_run_um', 'xi_tumble_um'])

        if sub.empty:
            ax.set_title(f"$d = {dia:.2f}\\,\\mu\\mathrm{{m}}$ (No data)", fontsize=12)
            ax.axis('off')
            continue

        x_r = np.sort(sub['xi_run_um'].to_numpy())
        x_t = np.sort(sub['xi_tumble_um'].to_numpy())
        y_r = np.arange(1, len(x_r) + 1) / len(x_r)
        y_t = np.arange(1, len(x_t) + 1) / len(x_t)

        ax.step(x_r, y_r, where='post', color=FLOW_MODE_COLORS['run'], linewidth=2.2, label=r"Run ($\xi_{\mathrm{Run}}$)")
        ax.step(x_t, y_t, where='post', color=FLOW_MODE_COLORS['tumble'], linewidth=2.2, linestyle='--', label=r"Tumble ($\xi_{\mathrm{Tumble}}$)")

        # 2-sample Kolmogorov-Smirnov test
        if len(x_r) >= 4 and len(x_t) >= 4:
            ks_res = stats.ks_2samp(x_r, x_t)
            ax.text(
                0.05, 0.85, f"KS test $p = {ks_res.pvalue:.3g}$\n$N = {len(x_r)}$",
                transform=ax.transAxes, fontsize=9.5,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85, edgecolor='lightgray'),
            )

        ax.set_xlabel(r"Flow Correlation Length $\xi$ [$\mu\mathrm{m}$]", fontsize=11)
        ax.set_ylabel("Cumulative Probability", fontsize=11)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.set_title(f"$d = {dia:.2f}\\,\\mu\\mathrm{{m}}$", fontsize=12, fontweight='bold')

        if idx == 0:
            ax.legend(fontsize=9, loc='lower right')

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved per-particle eCDF 6-panel plot: {output_path}")

