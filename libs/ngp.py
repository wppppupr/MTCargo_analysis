"""
ngp.py

貨物微粒子（蛍光ビーズ）の軌跡データからノンガウシアンパラメータ (Non-Gaussian Parameter: NGP / alpha_2)
を計算・解析するモジュールです。

理論的背景:
  ガウス統計（標準ブラウン運動）からの変位確率分布の偏りを定量化する指標 (Rahman, 1964):

  - 2次元変位の場合 (d = 2):
      \\alpha_2(\\Delta t) = \\frac{1}{2} \\frac{\\langle |\\Delta\\mathbf{r}|^4 \\rangle}{\\langle |\\Delta\\mathbf{r}|^2 \\rangle^2} - 1

  - 1次元変位の場合 (d = 1, 例: x, y, parallel, perpendicular):
      \\alpha_2(\\Delta t) = \\frac{1}{3} \\frac{\\langle \\Delta x^4 \\rangle}{\\langle \\Delta x^2 \\rangle^2} - 1

  完全なガウス分布では \\alpha_2 = 0 となり、裾の重い非ガウス性（指数テールや不均一輸送）では \\alpha_2 > 0 となります。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def calc_displacements_array(df, tau, scale=0.11, component='2d', theta_array=None):
    """
    データフレーム内の全粒子から指定ラグタイム tau における変位配列を抽出する。

    Parameters
    ----------
    df : pd.DataFrame
        'particle', 'frame', 'x', 'y' を含むトラッキングデータ
    tau : int
        ラグタイム（フレーム数）
    scale : float
        空間スケール (μm/pixel)
    component : str
        '2d' (ノルム), 'x', 'y', 'parallel', 'perpendicular'
    theta_array : np.ndarray, optional
        大域ネマチック平均配向角 theta(t) (ラジアン)

    Returns
    -------
    np.ndarray
        変位（または2次元ノルム）の1次元配列
    """
    comp = component.lower()
    disp_list = []

    for _, group in df.groupby('particle'):
        group = group.sort_values(by='frame')
        frames = group['frame'].to_numpy()
        T = len(frames)
        if T <= tau:
            continue

        x = scale * group['x'].to_numpy()
        y = scale * group['y'].to_numpy()

        f_min = frames[0]
        f_max = frames[-1]
        n_dense = f_max - f_min + 1

        x_dense = np.full(n_dense, np.nan)
        y_dense = np.full(n_dense, np.nan)
        f_idx = frames - f_min
        x_dense[f_idx] = x
        y_dense[f_idx] = y

        dx = x_dense[tau:] - x_dense[:-tau]
        dy = y_dense[tau:] - y_dense[:-tau]

        valid = ~np.isnan(dx) & ~np.isnan(dy)
        if not np.any(valid):
            continue

        dx_v = dx[valid]
        dy_v = dy[valid]

        if comp in ['2d', 'norm', 'magnitude', 'r']:
            disp = np.sqrt(dx_v**2 + dy_v**2)
        elif comp == 'x':
            disp = dx_v
        elif comp == 'y':
            disp = dy_v
        elif comp in ['parallel', 'par']:
            if theta_array is None:
                raise ValueError("component='parallel' には theta_array が必要です。")
            start_f = f_min + np.where(valid)[0]
            valid_th = start_f < len(theta_array)
            if not np.any(valid_th):
                continue
            th = theta_array[start_f[valid_th]]
            disp = dx_v[valid_th] * np.cos(th) + dy_v[valid_th] * np.sin(th)
        elif comp in ['perpendicular', 'perp']:
            if theta_array is None:
                raise ValueError("component='perpendicular' には theta_array が必要です。")
            start_f = f_min + np.where(valid)[0]
            valid_th = start_f < len(theta_array)
            if not np.any(valid_th):
                continue
            th = theta_array[start_f[valid_th]]
            disp = -dx_v[valid_th] * np.sin(th) + dy_v[valid_th] * np.cos(th)
        else:
            disp = np.sqrt(dx_v**2 + dy_v**2)

        disp_list.extend(disp)

    return np.asarray(disp_list)


def calc_ngp_from_displacements(displacements, component='2d'):
    """
    変位配列からノンガウシアンパラメータ alpha_2 を計算する。

    Parameters
    ----------
    displacements : array-like
        変位（または2次元ノルム）配列
    component : str
        '2d', 'x', 'y', 'parallel', 'perpendicular'

    Returns
    -------
    dict
        'ngp', 'msd' (<r^2>), 'm4' (<r^4>), 'count'
    """
    arr = np.asarray(displacements)
    arr = arr[~np.isnan(arr) & np.isfinite(arr)]
    n = len(arr)
    if n < 4:
        return {'ngp': np.nan, 'msd': np.nan, 'm4': np.nan, 'count': n}

    comp = component.lower()
    dim = 2 if comp in ['2d', 'norm', 'magnitude', 'r'] else 1

    r2 = arr ** 2
    r4 = arr ** 4

    mean_r2 = np.mean(r2)
    mean_r4 = np.mean(r4)

    if mean_r2 <= 0 or np.isnan(mean_r2):
        return {'ngp': np.nan, 'msd': np.nan, 'm4': np.nan, 'count': n}

    # d / (d + 2) * <r^4> / <r^2>^2 - 1
    factor = 0.5 if dim == 2 else (1.0 / 3.0)
    ngp_val = factor * (mean_r4 / (mean_r2 ** 2)) - 1.0

    return {
        'ngp': float(ngp_val),
        'msd': float(mean_r2),
        'm4': float(mean_r4),
        'count': int(n)
    }


def calc_ngp_evolution(df, max_tau=50, scale=0.11, frame_interval=4.0, component='2d',
                       theta_array=None, min_samples=5):
    """
    データフレームから各ラグタイム tau = 1 .. max_tau に対する NGP の時間変化を算出する。

    Parameters
    ----------
    df : pd.DataFrame
        トラッキングデータ
    max_tau : int
        最大ラグタイム（フレーム数）
    scale : float
        空間スケール (μm/px)
    frame_interval : float
        時間間隔 [s]
    component : str
        '2d', 'x', 'y', 'parallel', 'perpendicular'
    theta_array : np.ndarray, optional
        大域ネマチック平均配向角 theta(t)
    min_samples : int
        計算に必要な最小サンプル数

    Returns
    -------
    pd.DataFrame
        'tau', 'lag_time', 'ngp', 'msd', 'm4', 'count'
    """
    records = []
    for tau in range(1, max_tau + 1):
        disp = calc_displacements_array(
            df,
            tau=tau,
            scale=scale,
            component=component,
            theta_array=theta_array
        )
        if len(disp) < min_samples:
            continue

        res = calc_ngp_from_displacements(disp, component=component)
        if np.isnan(res['ngp']):
            continue

        records.append({
            'tau': tau,
            'lag_time': tau * frame_interval,
            'ngp': res['ngp'],
            'msd': res['msd'],
            'm4': res['m4'],
            'count': res['count']
        })

    if not records:
        return pd.DataFrame(columns=['tau', 'lag_time', 'ngp', 'msd', 'm4', 'count'])

    return pd.DataFrame(records)


def calc_individual_particle_ngp(df, max_tau=50, scale=0.11, frame_interval=4.0, component='2d',
                                 min_track_length=15, theta_array=None):
    """
    個別粒子ごとに NGP の時間平均曲線を計算する。
    """
    records = []
    comp = component.lower()
    dim = 2 if comp in ['2d', 'norm', 'magnitude', 'r'] else 1
    factor = 0.5 if dim == 2 else (1.0 / 3.0)

    for particle_id, group in df.groupby('particle'):
        if len(group) < min_track_length:
            continue
        group = group.sort_values(by='frame')
        frames = group['frame'].to_numpy()
        x = scale * group['x'].to_numpy()
        y = scale * group['y'].to_numpy()

        f_min = frames[0]
        f_max = frames[-1]
        n_dense = f_max - f_min + 1

        x_dense = np.full(n_dense, np.nan)
        y_dense = np.full(n_dense, np.nan)
        f_idx = frames - f_min
        x_dense[f_idx] = x
        y_dense[f_idx] = y

        for tau in range(1, min(max_tau + 1, n_dense)):
            dx = x_dense[tau:] - x_dense[:-tau]
            dy = y_dense[tau:] - y_dense[:-tau]
            valid = ~np.isnan(dx) & ~np.isnan(dy)
            if np.sum(valid) < 3:
                continue

            dx_v = dx[valid]
            dy_v = dy[valid]

            if comp in ['2d', 'norm', 'magnitude', 'r']:
                r2 = dx_v**2 + dy_v**2
            elif comp == 'x':
                r2 = dx_v**2
            elif comp == 'y':
                r2 = dy_v**2
            elif comp in ['parallel', 'par']:
                if theta_array is None:
                    continue
                start_f = f_min + np.where(valid)[0]
                valid_th = start_f < len(theta_array)
                if not np.any(valid_th):
                    continue
                th = theta_array[start_f[valid_th]]
                r = dx_v[valid_th] * np.cos(th) + dy_v[valid_th] * np.sin(th)
                r2 = r**2
            elif comp in ['perpendicular', 'perp']:
                if theta_array is None:
                    continue
                start_f = f_min + np.where(valid)[0]
                valid_th = start_f < len(theta_array)
                if not np.any(valid_th):
                    continue
                th = theta_array[start_f[valid_th]]
                r = -dx_v[valid_th] * np.sin(th) + dy_v[valid_th] * np.cos(th)
                r2 = r**2
            else:
                r2 = dx_v**2 + dy_v**2

            mean_r2 = np.mean(r2)
            mean_r4 = np.mean(r2**2)
            if mean_r2 <= 0:
                continue

            ngp_val = factor * (mean_r4 / (mean_r2**2)) - 1.0
            records.append({
                'particle': particle_id,
                'tau': tau,
                'lag_time': tau * frame_interval,
                'ngp': float(ngp_val),
                'n_points': int(len(r2))
            })

    if not records:
        return pd.DataFrame(columns=['particle', 'tau', 'lag_time', 'ngp', 'n_points'])

    return pd.DataFrame(records)
