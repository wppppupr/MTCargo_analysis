"""
ergodicity.py

貨物微粒子（蛍光ビーズ）の軌跡データから時間平均二乗変位 (Time-Averaged MSD: TAMSD)
およびエルゴード性破壊パラメータ (Ergodicity Breaking Parameter: EB) を計算・解析するモジュールです。

理論的背景:
  各粒子 i の軌道 r_i(t) に対する時間平均二乗変位 (TAMSD):
    \\overline{\\delta_i^2(\\Delta t)} = \\frac{1}{T - \\Delta t} \\int_0^{T - \\Delta t} |\\mathbf{r}_i(t + \\Delta t) - \\mathbf{r}_i(t)|^2 dt

  エルゴード性破壊パラメータ EB(\\Delta t):
    EB(\\Delta t) = \\frac{\\langle (\\overline{\\delta^2(\\Delta t)})^2 \\rangle - \\langle \\overline{\\delta^2(\\Delta t)} \\rangle^2}{\\langle \\overline{\\delta^2(\\Delta t)} \\rangle^2}
                  = \\frac{\\mathrm{Var}(\\overline{\\delta^2(\\Delta t)})}{\\langle \\overline{\\delta^2(\\Delta t)} \\rangle^2}
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def calc_particle_tamsd(group, max_tau, scale=0.11, frame_interval=4.0, component='2d', theta_array=None):
    """
    1つの粒子の軌道データから各ラグタイム tau における TAMSD (時間平均二乗変位) を計算する。

    Parameters
    ----------
    group : pd.DataFrame
        1粒子のトラッキングデータ ('frame', 'x', 'y' 列を含む)
    max_tau : int
        計算する最大ラグタイム（フレーム数）
    scale : float
        ピクセルから μm への変換係数 (μm/pixel)
    frame_interval : float
        フレーム間の時間間隔 [s]
    component : str
        '2d' (2次元ノルム), 'x', 'y', 'parallel' (大域ネマチック平行), 'perpendicular' (大域ネマチック直交)
    theta_array : np.ndarray, optional
        大域ネマチック平均配向角 theta(t) (ラジアン)

    Returns
    -------
    list of dict
        各 tau に対する {'tau': tau, 'lag_time': lag_time, 'tamsd': tamsd, 'n_points': n_pts}
    """
    group = group.sort_values(by='frame')
    frames = group['frame'].to_numpy()
    T = len(frames)
    if T < 2:
        return []

    x = scale * group['x'].to_numpy()
    y = scale * group['y'].to_numpy()

    # フレームの連続性を確認（欠損フレームがある場合はフレーム番号でインデックス化）
    f_min = frames[0]
    f_max = frames[-1]
    n_total_frames = f_max - f_min + 1

    # 密な配列を構築（欠損フレームは NaN）
    x_dense = np.full(n_total_frames, np.nan)
    y_dense = np.full(n_total_frames, np.nan)
    f_indices = frames - f_min
    x_dense[f_indices] = x
    y_dense[f_indices] = y

    comp = component.lower()
    records = []

    for tau in range(1, min(max_tau + 1, n_total_frames)):
        dx = x_dense[tau:] - x_dense[:-tau]
        dy = y_dense[tau:] - y_dense[:-tau]

        valid_mask = ~np.isnan(dx) & ~np.isnan(dy)
        if not np.any(valid_mask):
            continue

        dx_v = dx[valid_mask]
        dy_v = dy[valid_mask]

        if comp in ['2d', 'norm', 'magnitude', 'r']:
            sq_disp = dx_v**2 + dy_v**2
        elif comp == 'x':
            sq_disp = dx_v**2
        elif comp == 'y':
            sq_disp = dy_v**2
        elif comp in ['parallel', 'par']:
            if theta_array is None:
                raise ValueError("component='parallel' には theta_array が必要です。")
            start_f = (frames[:-tau])[valid_mask] if len(frames) == n_total_frames else f_min + np.where(valid_mask)[0]
            valid_th = start_f < len(theta_array)
            if not np.any(valid_th):
                continue
            th = theta_array[start_f[valid_th]]
            disp_par = dx_v[valid_th] * np.cos(th) + dy_v[valid_th] * np.sin(th)
            sq_disp = disp_par**2
        elif comp in ['perpendicular', 'perp']:
            if theta_array is None:
                raise ValueError("component='perpendicular' には theta_array が必要です。")
            start_f = (frames[:-tau])[valid_mask] if len(frames) == n_total_frames else f_min + np.where(valid_mask)[0]
            valid_th = start_f < len(theta_array)
            if not np.any(valid_th):
                continue
            th = theta_array[start_f[valid_th]]
            disp_perp = -dx_v[valid_th] * np.sin(th) + dy_v[valid_th] * np.cos(th)
            sq_disp = disp_perp**2
        else:
            sq_disp = dx_v**2 + dy_v**2

        tamsd_val = np.mean(sq_disp)
        records.append({
            'tau': tau,
            'lag_time': tau * frame_interval,
            'tamsd': float(tamsd_val),
            'n_points': int(len(sq_disp))
        })

    return records


def calc_all_particles_tamsd(df, max_tau=50, scale=0.11, frame_interval=4.0, component='2d',
                             min_track_length=10, theta_array=None):
    """
    データフレーム内の全粒子について TAMSD を計算する。

    Parameters
    ----------
    df : pd.DataFrame
        'particle', 'frame', 'x', 'y' を含むトラッキングデータ
    max_tau : int
        最大ラグタイム（フレーム数）
    scale : float
        空間スケール (μm/px)
    frame_interval : float
        時間間隔 [s]
    component : str
        '2d', 'x', 'y', 'parallel', 'perpendicular'
    min_track_length : int
        解析対象とする最小トラック長（フレーム数）
    theta_array : np.ndarray, optional
        大域ネマチック平均配向角 theta(t)

    Returns
    -------
    pd.DataFrame
        'particle', 'tau', 'lag_time', 'tamsd', 'n_points' 列を持つ DataFrame
    """
    all_records = []

    for particle_id, group in df.groupby('particle'):
        if len(group) < min_track_length:
            continue
        recs = calc_particle_tamsd(
            group,
            max_tau=max_tau,
            scale=scale,
            frame_interval=frame_interval,
            component=component,
            theta_array=theta_array
        )
        for r in recs:
            r['particle'] = particle_id
            all_records.append(r)

    if not all_records:
        return pd.DataFrame(columns=['particle', 'tau', 'lag_time', 'tamsd', 'n_points'])

    return pd.DataFrame(all_records)


def calc_eb_parameter(tamsd_df, min_particles=3):
    """
    全粒子の TAMSD データフレームから各 lag time におけるエルゴード性破壊パラメータ EB(Δt) を計算する。

    EB(Δt) = Var(TAMSD) / <TAMSD>^2 = (<TAMSD^2> - <TAMSD>^2) / <TAMSD>^2

    Parameters
    ----------
    tamsd_df : pd.DataFrame
        calc_all_particles_tamsd() の出力
    min_particles : int
        EBを算出する最小粒子数

    Returns
    -------
    pd.DataFrame
        'tau', 'lag_time', 'eb', 'eb_err', 'mean_tamsd', 'std_tamsd', 'n_particles' を列に持つ DataFrame
    """
    if tamsd_df.empty:
        return pd.DataFrame()

    results = []
    for (tau, lag_t), group in tamsd_df.groupby(['tau', 'lag_time']):
        vals = group['tamsd'].dropna().to_numpy()
        n_p = len(vals)
        if n_p < min_particles:
            continue

        mean_val = np.mean(vals)
        if mean_val <= 0 or np.isnan(mean_val):
            continue

        var_val = np.var(vals, ddof=1) if n_p > 1 else 0.0
        eb_val = var_val / (mean_val ** 2)

        # EB の標準誤差（ブートストラップまたは標準正規近似誤差）
        # 相対揺らぎ xi = delta^2 / <delta^2> の分散の誤差: Var(Var) ≈ 2/(N-1) for Gaussian, bootstrap で推定
        eb_err = eb_val * np.sqrt(2.0 / (n_p - 1)) if n_p > 2 else 0.0

        results.append({
            'tau': tau,
            'lag_time': lag_t,
            'eb': float(eb_val),
            'eb_err': float(eb_err),
            'mean_tamsd': float(mean_val),
            'std_tamsd': float(np.sqrt(var_val)),
            'n_particles': int(n_p)
        })

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results).sort_values('tau').reset_index(drop=True)


def plot_eb(eb_df, ax=None, figsize=(6.5, 5.0), display=False, color='#1f77b4', marker='o',
            label=None, title='Ergodicity Breaking Parameter $EB(\\Delta t)$', xscale='log', yscale='log'):
    """
    EBパラメータ EB(Δt) をプロットする。
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    valid = eb_df['eb'] > 0 if yscale == 'log' else np.ones(len(eb_df), dtype=bool)
    df_plot = eb_df[valid]

    ax.errorbar(
        df_plot['lag_time'],
        df_plot['eb'],
        yerr=df_plot['eb_err'],
        marker=marker,
        color=color,
        label=label,
        capsize=3,
        elinewidth=1.0,
        markersize=5,
        alpha=0.9
    )

    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xlabel(r'Lag time $\Delta t$ [s]')
    ax.set_ylabel(r'$EB(\Delta t) = \mathrm{Var}(\overline{\delta^2}) / \langle \overline{\delta^2} \rangle^2$')
    ax.set_title(title)
    ax.grid(True, which="both", ls="--", alpha=0.3)

    if display:
        plt.show()

    return fig, ax
