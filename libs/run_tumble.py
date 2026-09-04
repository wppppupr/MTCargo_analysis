"""
libs/run_tumble.py

貨物微粒子のRun (能動輸送/走行) と Tumble (停滞/方向転換) のセグメンテーション、
持続時間（Duration）の抽出、確率密度関数 (PDF)・累積補確率 (CCDF) の算出、
および指数分布フィッティングを行うモジュールです。
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def calc_instantaneous_speeds(df_tracks, scale=0.11, frame_interval=4.0):
    """
    トラッキング DataFrame から各粒子の瞬時速度 [μm/s] を計算する。

    Parameters:
    -----------
    df_tracks : pd.DataFrame
        'particle', 'frame', 'x', 'y' カラムを含む DataFrame
    scale : float
        ピクセル -> μm の変換スケール (default: 0.11 μm/pixel)
    frame_interval : float
        フレーム間隔 (default: 4.0 s)

    Returns:
    --------
    df_result : pd.DataFrame
        各ステップごとの瞬時速度 'v' [μm/s] および粒子情報が付与された DataFrame
    """
    required_cols = {'particle', 'frame', 'x', 'y'}
    if not required_cols.issubset(df_tracks.columns):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")

    # ソート
    df_sorted = df_tracks.sort_values(by=['particle', 'frame']).copy()

    # 各粒子ごとの差分
    grouped = df_sorted.groupby('particle', group_keys=False)

    df_sorted['frame_diff'] = grouped['frame'].diff()
    df_sorted['dx_px'] = grouped['x'].diff()
    df_sorted['dy_px'] = grouped['y'].diff()

    # 連続フレーム（frame_diff == 1）のみ有効とする
    valid_step = df_sorted['frame_diff'] == 1
    df_sorted['dr_um'] = np.where(
        valid_step,
        np.sqrt(df_sorted['dx_px']**2 + df_sorted['dy_px']**2) * scale,
        np.nan
    )
    df_sorted['dt_sec'] = np.where(valid_step, frame_interval, np.nan)
    df_sorted['v'] = df_sorted['dr_um'] / df_sorted['dt_sec']

    return df_sorted


def segment_run_tumble_from_speeds(speed_series, threshold, frame_interval=4.0, drop_edges=False, min_consecutive=1):
    """
    瞬時速度の時系列から Run (>= threshold) と Tumble (< threshold) の
    セグメンテーションを行い、それぞれの持続時間 [s] のリストを抽出する。

    Parameters:
    -----------
    speed_series : np.ndarray or pd.Series
        1つの粒子の瞬時速度配列 (NaNはトラックの切れ目を表す)
    threshold : float
        Run/Tumble 判定の速度閾値 [μm/s]
    frame_interval : float
        フレーム間隔 [s]
    drop_edges : bool
        開始時・終了時の不完全なセグメント（右側/左側打ち切りデータ）を除外するかどうか
    min_consecutive : int
        セグメントとみなす最小連続フレーム数 (default: 1)

    Returns:
    --------
    run_durations : list of float
        Run 持続時間 [s] のリスト
    tumble_durations : list of float
        Tumble 持続時間 [s] のリスト
    states : np.ndarray
        各ステップの状態配列 (1: Run, 0: Tumble, -1: Undefined/NaN)
    """
    speeds = np.asarray(speed_series)
    n = len(speeds)
    states = np.full(n, -1, dtype=int)

    valid_mask = np.isfinite(speeds)
    states[valid_mask] = (speeds[valid_mask] >= threshold).astype(int)

    run_durations = []
    tumble_durations = []

    # 連続した同一状態区間 (Run-length encoding) を検出
    i = 0
    while i < n:
        if states[i] == -1:
            i += 1
            continue

        curr_state = states[i]
        start_idx = i
        while i < n and states[i] == curr_state:
            i += 1
        end_idx = i
        count = end_idx - start_idx

        # 端点判定（トラックの先頭または末尾、あるいはNaN境界）
        is_edge = False
        if start_idx == 0 or (start_idx > 0 and states[start_idx - 1] == -1):
            is_edge = True
        if end_idx == n or (end_idx < n and states[end_idx] == -1):
            is_edge = True

        if drop_edges and is_edge:
            continue

        if count >= min_consecutive:
            duration = count * frame_interval
            if curr_state == 1:
                run_durations.append(duration)
            else:
                tumble_durations.append(duration)

    return run_durations, tumble_durations, states


def extract_durations_from_df(df_with_v, threshold, frame_interval=4.0, drop_edges=False, min_track_len=5):
    """
    全粒子のトラッキングデータから Run/Tumble 持続時間を集約する。

    Parameters:
    -----------
    df_with_v : pd.DataFrame
        'particle', 'v', 'frame' 等を含む DataFrame
    threshold : float
        速度閾値 [μm/s]
    frame_interval : float
        フレーム間隔 [s]
    drop_edges : bool
        端点セグメントを除外するかどうか
    min_track_len : int
        解析対象とする粒子の最小有効データ点数

    Returns:
    --------
    all_run_durations : np.ndarray
        全粒子の Run 持続時間 [s]
    all_tumble_durations : np.ndarray
        全粒子の Tumble 持続時間 [s]
    states_dict : dict
        particle_id -> (frames, speeds, states)
    """
    all_run_durations = []
    all_tumble_durations = []
    states_dict = {}

    for particle_id, group in df_with_v.groupby('particle'):
        valid_v = group['v'].dropna()
        if len(valid_v) < min_track_len:
            continue

        speeds = group['v'].values
        frames = group['frame'].values

        r_durs, t_durs, states = segment_run_tumble_from_speeds(
            speeds, threshold=threshold, frame_interval=frame_interval, drop_edges=drop_edges
        )

        all_run_durations.extend(r_durs)
        all_tumble_durations.extend(t_durs)
        states_dict[particle_id] = {
            'frame': frames,
            'x': group['x'].values,
            'y': group['y'].values,
            'v': speeds,
            'state': states
        }

    return np.array(all_run_durations, dtype=float), np.array(all_tumble_durations, dtype=float), states_dict


def calc_duration_pdf(durations, bins=25, bin_range=None, density=True):
    """
    持続時間データの確率密度関数 (PDF) または頻度ヒストグラムを計算する。

    Parameters:
    -----------
    durations : array-like
        持続時間 [s] の配列
    bins : int or array-like
        ビン数またはビン境界
    bin_range : tuple of (float, float), optional
        (min_val, max_val)
    density : bool
        True の場合、積分が 1 になるように正規化 (PDF)

    Returns:
    --------
    bin_centers : np.ndarray
        ビン中心値 [s]
    hist : np.ndarray
        確率密度または頻度
    bin_edges : np.ndarray
        ビンの境界
    """
    arr = np.asarray(durations)
    arr = arr[np.isfinite(arr) & (arr > 0)]
    if len(arr) == 0:
        return np.array([]), np.array([]), np.array([])

    if bin_range is None:
        bin_range = (np.min(arr), np.max(arr))

    hist, bin_edges = np.histogram(arr, bins=bins, range=bin_range, density=density)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return bin_centers, hist, bin_edges


def calc_duration_ccdf(durations):
    """
    持続時間データの相補累積分布関数 (CCDF: P(T >= t)) を計算する。

    Parameters:
    -----------
    durations : array-like
        持続時間 [s] の配列

    Returns:
    --------
    sorted_durations : np.ndarray
        ソートされた持続時間 [s]
    ccdf : np.ndarray
        P(T >= t) の値 (1.0 -> 0.0)
    """
    arr = np.asarray(durations)
    arr = arr[np.isfinite(arr) & (arr > 0)]
    if len(arr) == 0:
        return np.array([]), np.array([])

    sorted_durations = np.sort(arr)
    n = len(sorted_durations)
    # 累積確率 P(T <= t) -> 1 - (i / n)
    ccdf = 1.0 - (np.arange(n) / n)
    return sorted_durations, ccdf


def exp_decay_func(t, tau, a):
    """指数減衰関数: a * exp(-t / tau)"""
    return a * np.exp(-t / tau)


def fit_exponential(bin_centers, pdf_values, initial_tau=None):
    """
    PDF に対して指数減衰関数 P(t) = a * exp(-t / tau) をフィッティングする。

    Parameters:
    -----------
    bin_centers : array-like
        ビン中心値 [s]
    pdf_values : array-like
        PDF 値
    initial_tau : float, optional
        初期推定値

    Returns:
    --------
    fit_result : dict
        {'tau': float, 'tau_err': float, 'a': float, 'r_squared': float}
    """
    valid = np.isfinite(pdf_values) & (pdf_values > 0) & np.isfinite(bin_centers)
    x = np.asarray(bin_centers)[valid]
    y = np.asarray(pdf_values)[valid]

    if len(x) < 3:
        return {'tau': np.nan, 'tau_err': np.nan, 'a': np.nan, 'r_squared': np.nan}

    if initial_tau is None:
        initial_tau = np.mean(x)
    initial_a = y[0] if len(y) > 0 else 1.0 / initial_tau

    try:
        popt, pcov = curve_fit(
            exp_decay_func, x, y,
            p0=[initial_tau, initial_a],
            bounds=([1e-3, 0], [1e5, np.inf]),
            maxfev=5000
        )
        tau, a = popt
        tau_err = np.sqrt(pcov[0, 0]) if np.isfinite(pcov[0, 0]) else np.nan

        # R^2 の計算
        residuals = y - exp_decay_func(x, tau, a)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

        return {
            'tau': tau,
            'tau_err': tau_err,
            'a': a,
            'r_squared': r_squared
        }
    except Exception:
        return {'tau': np.nan, 'tau_err': np.nan, 'a': np.nan, 'r_squared': np.nan}
