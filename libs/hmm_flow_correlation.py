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
    max_fit_dist: Optional[float] = None,
    allow_offset: bool = False,
) -> Dict[str, Union[float, np.ndarray]]:
    """
    微小管フローの C(r) 曲線に対して指数減衰モデルをフィッティングし、配向相関長 xi を算出する。
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

    init_a = float(np.clip(c_vals[0] if len(c_vals) > 0 else 1.0, 0.05, 1.0))
    init_xi = 20.0

    if allow_offset:
        p0 = [init_xi, init_a, 0.0]
        bounds = ([0.1, 0.0, -1.0], [500.0, 2.0, 1.0])
        def fit_func(r, xi, a, c0):
            return exp_decay_model(r, xi, a, c0)
    else:
        p0 = [init_xi, init_a]
        bounds = ([0.1, 0.0], [500.0, 2.0])
        def fit_func(r, xi, a):
            return exp_decay_model(r, xi, a, 0.0)

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


def plot_flow_correlations_single_axis(
    df_curves: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
    title: str = "",
    fit_curves: bool = True,
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
            fit_res = fit_flow_correlation_length(sub)
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
