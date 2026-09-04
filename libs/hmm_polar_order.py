"""
libs/hmm_polar_order.py

微小管の局所ポーラーオーダー結果（local_polar_w.zarr / local_polar_bg.zarr）と
貨物微粒子の1次元対数速力 Gaussian HMM 推定状態（Run / Tumble）を結合し、
運動モード別の局所ポーラーオーダーを集計・可視化するためのモジュールです。
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from libs import hmm_cargo as hc

POLAR_MODE_NAMES = {
    'run': 'Run Particle Vicinity',
    'tumble': 'Tumble Particle Vicinity',
    'all': 'All Particle Vicinity',
    'bg': 'Background Flow (Bulk)',
}

POLAR_MODE_COLORS = {
    'run': '#1b9e77',      # 青緑 (Run)
    'tumble': '#d95f02',   # オレンジ (Tumble)
    'all': '#222222',      # 黒 (All)
    'bg': '#7570b3',       # 紫 / 灰 (Background)
}

POLAR_MODE_STYLES = {
    'run': '-',
    'tumble': '-',
    'all': '-',
    'bg': '--',
}


def extract_experiment_mode_polar_orders(
    exp_dir: Path,
    hmm_model: hc.CargoGaussianHMM,
    scale: float = 0.11,
    tau: int = 1,
    frame_interval: float = 4.0,
    epsilon: float = 1e-3,
) -> Optional[dict]:
    """
    1つの実験ディレクトリに対して、HMM 状態と局所ポーラーオーダー Zarr データをマッチングし、
    モード別のポーラーオーダープロファイルを抽出する。

    Parameters
    ----------
    exp_dir : Path
        実験ディレクトリ (beads_tracks.csv, local_polar_w.zarr を含む)
    hmm_model : hc.CargoGaussianHMM
        学習済みの Gaussian HMM モデル
    scale : float, default 0.11
        空間スケール (um/pixel)

    Returns
    -------
    result : dict or None
        窓サイズ座標 (um)、Run/Tumble/All/BG の各ポーラーオーダー配列 (窓サイズ x サンプル数)
    """
    tracks_csv = exp_dir / "beads_tracks.csv"
    p_zarr_path = exp_dir / "local_polar_w.zarr"
    bg_zarr_path = exp_dir / "local_polar_bg.zarr"

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

    # 粒子局所ポーラーオーダー Zarr のロード
    try:
        ds_p = xr.open_zarr(str(p_zarr_path), consolidated=False)
    except Exception as e:
        print(f"[WARNING] Failed to open {p_zarr_path}: {e}")
        return None

    # 窓サイズ座標名の取得 ('window size' or 'window_size')
    w_dim = 'window size' if 'window size' in ds_p.dims else 'window_size'
    windows_px = ds_p.coords[w_dim].values
    windows_um = windows_px * scale
    num_w = len(windows_px)

    frames = ds_p.coords['frame'].values
    particles = ds_p.coords['particle'].values

    arr_total = ds_p['polar_order'].values  # (W, frame, particle)

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

    def extract_samples(arr, mask_2d):
        if arr is None or not np.any(mask_2d):
            return np.empty((num_w, 0), dtype=np.float32)
        samples = arr[:, mask_2d]
        return samples

    run_polar_samples = extract_samples(arr_total, run_mask)
    tumble_polar_samples = extract_samples(arr_total, tumble_mask)
    all_polar_samples = extract_samples(arr_total, all_mask)

    # 背景局所ポーラーオーダー Zarr のロード
    bg_polar_samples = np.empty((num_w, 0), dtype=np.float32)
    if bg_zarr_path.exists():
        try:
            ds_bg = xr.open_zarr(str(bg_zarr_path), consolidated=False)
            if 'polar_order' in ds_bg:
                bg_w_dim = 'window size' if 'window size' in ds_bg.dims else 'window_size'
                bg_arr = ds_bg['polar_order'].values  # (W, frame)
                # 窓サイズ座標が一致しているか確認 (または共通部分をスライス)
                bg_w_px = ds_bg.coords[bg_w_dim].values
                if np.array_equal(bg_w_px, windows_px):
                    bg_polar_samples = bg_arr
                else:
                    # 共通の窓サイズのみ抽出
                    common_w, p_idx_common, bg_idx_common = np.intersect1d(windows_px, bg_w_px, return_indices=True)
                    if len(common_w) > 0:
                        bg_polar_samples = np.full((num_w, bg_arr.shape[1]), np.nan, dtype=np.float32)
                        bg_polar_samples[p_idx_common, :] = bg_arr[bg_idx_common, :]
        except Exception as e:
            print(f"[WARNING] Failed to load {bg_zarr_path}: {e}")

    return {
        'exp_dir': exp_dir.name,
        'windows_um': windows_um,
        'windows_px': windows_px,
        'run_polar': run_polar_samples,
        'tumble_polar': tumble_polar_samples,
        'all_polar': all_polar_samples,
        'bg_polar': bg_polar_samples,
    }


def aggregate_polar_order_dataset(
    exp_results: List[dict],
) -> pd.DataFrame:
    """
    複数実験にわたるモード別局所ポーラーオーダーデータを集計し、
    各モードの窓サイズ依存性 DataFrame を生成する。

    Returns
    -------
    df_curves : pd.DataFrame
        'mode', 'window_size_um', 'mean_polar_order', 'sem_polar_order', 'std_polar_order', 'n_samples'
    """
    if not exp_results:
        return pd.DataFrame()

    windows_um = exp_results[0]['windows_um']
    num_w = len(windows_um)

    target_vars = [
        ('run', 'run_polar'),
        ('tumble', 'tumble_polar'),
        ('all', 'all_polar'),
        ('bg', 'bg_polar'),
    ]

    records = []

    for mode_key, data_field in target_vars:
        all_samples_per_window = [[] for _ in range(num_w)]

        for res in exp_results:
            samples = res.get(data_field)
            if samples is None or samples.shape[1] == 0:
                continue

            for w_idx in range(num_w):
                vals = samples[w_idx, :]
                valid_vals = vals[np.isfinite(vals) & (vals <= 1.05) & (vals >= 0.0)]
                if len(valid_vals) > 0:
                    all_samples_per_window[w_idx].extend(valid_vals)

        for w_idx, w_um in enumerate(windows_um):
            v_list = np.array(all_samples_per_window[w_idx])
            n_pts = len(v_list)
            if n_pts < 3:
                continue

            mean_v = float(np.mean(v_list))
            std_v = float(np.std(v_list, ddof=1)) if n_pts > 1 else 0.0
            sem_v = float(std_v / np.sqrt(n_pts)) if n_pts > 0 else 0.0

            records.append({
                'mode': mode_key,
                'mode_label': POLAR_MODE_NAMES.get(mode_key, mode_key),
                'window_size_um': float(w_um),
                'mean_polar_order': mean_v,
                'std_polar_order': std_v,
                'sem_polar_order': sem_v,
                'n_samples': n_pts,
            })

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records)


def plot_polar_orders_single_axis(
    df_curves: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
    title: str = "",
    show_legend: bool = True,
) -> plt.Axes:
    """
    単一軸に Run, Tumble, All, Background の局所ポーラーオーダー曲線を描画する。
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

        w = sub['window_size_um'].to_numpy()
        p = sub['mean_polar_order'].to_numpy()
        sem = sub['sem_polar_order'].to_numpy()

        color = POLAR_MODE_COLORS.get(mode, 'black')
        label = POLAR_MODE_NAMES.get(mode, mode)
        ls = POLAR_MODE_STYLES.get(mode, '-')

        ax.errorbar(
            w, p, yerr=sem,
            label=label,
            color=color,
            fmt='o' if mode != 'bg' else 's',
            markersize=3.5,
            linestyle=ls,
            linewidth=1.5,
            capsize=2,
            alpha=0.85,
        )

    ax.set_xlabel(r"Window Size $R$ [$\mu\mathrm{m}$]", fontsize=11)
    ax.set_ylabel(r"Local Polar Order $\Phi(R)$", fontsize=11)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, linestyle='--', alpha=0.4)
    if title:
        ax.set_title(title, fontsize=12, fontweight='bold')
    if show_legend:
        ax.legend(fontsize=8.5, framealpha=0.9, loc='upper right')

    return ax
