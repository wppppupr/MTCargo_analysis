"""
libs/angular_distribution.py

貨物微粒子（蛍光ビーズ）の進行方向角度変化（Turning Angle / Angle Change Δθ）の計算、
確率密度関数 (PDF) の集計、円統計（Circular Statistics）、および各種分布モデル
（von Mises分布、ガウス分布、指数分布）によるフィッティング関数を提供します。
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.special import i0, i1


def wrap_angle_pi(angle_rad):
    """
    角度（ラジアン）を [-π, π] の範囲にラップする。
    """
    return np.arctan2(np.sin(angle_rad), np.cos(angle_rad))


def calc_angular_changes(df_tracks, tau=1, scale=0.11, frame_interval=4.0, signed=True, unit='rad', min_track_len=None):
    """
    粒子軌跡データから指定ラグタイム tau における角度変化 Δθ を NumPy 配列スライスで超高速に算出する。

    定義:
    1. 時刻 t におけるラグタイム tau の変位ベクトル:
       v(t) = (x(t+tau) - x(t), y(t+tau) - y(t))
       進行方向角: theta(t) = atan2(dy, dx)
    2. 連続するステップ間 (t -> t+tau -> t+2tau) の方向変化:
       Δθ(t, tau) = wrap(theta(t+tau) - theta(t)) in [-π, π]
    """
    if df_tracks is None or df_tracks.empty:
        return np.array([])

    cols = ['particle', 'frame', 'x', 'y']
    for c in cols:
        if c not in df_tracks.columns:
            return np.array([])

    df = df_tracks[cols].sort_values(by=['particle', 'frame'])
    p = df['particle'].to_numpy()
    f = df['frame'].to_numpy()
    x = df['x'].to_numpy() * scale
    y = df['y'].to_numpy() * scale

    N = len(p)
    if N <= 2 * tau:
        return np.array([])

    # 3点 (t, t+tau, t+2tau) のスライス
    p0, p1, p2 = p[:-2 * tau], p[tau:-tau], p[2 * tau:]
    f0, f1, f2 = f[:-2 * tau], f[tau:-tau], f[2 * tau:]
    x0, x1, x2 = x[:-2 * tau], x[tau:-tau], x[2 * tau:]
    y0, y1, y2 = y[:-2 * tau], y[tau:-tau], y[2 * tau:]

    # 同一粒子かつ tau フレーム間隔で連続している箇所を判定
    valid = (p0 == p1) & (p1 == p2) & (f1 == f0 + tau) & (f2 == f1 + tau)
    if not np.any(valid):
        return np.array([])

    dx1 = x1[valid] - x0[valid]
    dy1 = y1[valid] - y0[valid]
    dx2 = x2[valid] - x1[valid]
    dy2 = y2[valid] - y1[valid]

    # 有意な変位がある箇所
    non_zero = ((dx1**2 + dy1**2) > 1e-10) & ((dx2**2 + dy2**2) > 1e-10)
    if not np.any(non_zero):
        return np.array([])

    th1 = np.arctan2(dy1[non_zero], dx1[non_zero])
    th2 = np.arctan2(dy2[non_zero], dx2[non_zero])

    d_th = wrap_angle_pi(th2 - th1)

    if not signed:
        d_th = np.abs(d_th)

    if unit == 'deg':
        d_th = np.rad2deg(d_th)

    return d_th


def calc_ensemble_angle_pdf(exp_angles_list, bins=50, bin_range=None, density=True, signed=True, unit='rad'):
    """
    実験ごとの角度変化データリストから、実験間アンサンブル平均PDFと標準偏差 (std) を算出する。
    """
    if not exp_angles_list:
        return np.array([]), np.array([]), np.array([]), np.array([])

    if bin_range is None:
        if unit == 'deg':
            bin_range = (-180.0, 180.0) if signed else (0.0, 180.0)
        else:
            bin_range = (-np.pi, np.pi) if signed else (0.0, np.pi)

    pdf_list = []
    edges = None

    for arr in exp_angles_list:
        arr = np.asarray(arr)
        valid = np.isfinite(arr) & (arr >= bin_range[0]) & (arr <= bin_range[1])
        arr_valid = arr[valid]

        if len(arr_valid) < 5:
            continue

        counts, bin_edges = np.histogram(arr_valid, bins=bins, range=bin_range, density=density)
        pdf_list.append(counts)
        if edges is None:
            edges = bin_edges

    if not pdf_list or edges is None:
        return np.array([]), np.array([]), np.array([]), np.array([])

    pdf_matrix = np.array(pdf_list)
    mean_pdf = np.mean(pdf_matrix, axis=0)
    std_pdf = np.std(pdf_matrix, axis=0, ddof=1) if len(pdf_list) > 1 else np.zeros_like(mean_pdf)
    centers = (edges[:-1] + edges[1:]) / 2.0

    return centers, mean_pdf, std_pdf, edges


def calc_circular_stats(angles_rad):
    """
    円統計（Circular Statistics）指標を算出する。
    """
    arr = np.asarray(angles_rad)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {
            'count': 0,
            'mean_cos': np.nan,
            'mean_sin': np.nan,
            'mean_resultant_length': np.nan,
            'circular_mean': np.nan,
            'circular_variance': np.nan,
            'circular_std': np.nan,
            'mean_abs_angle_rad': np.nan,
            'mean_abs_angle_deg': np.nan
        }

    c = np.mean(np.cos(arr))
    s = np.mean(np.sin(arr))
    R = float(np.hypot(c, s))
    mu = float(np.arctan2(s, c))
    circ_var = float(1.0 - R)
    circ_std = float(np.sqrt(max(0.0, -2.0 * np.log(max(1e-12, R)))))
    mean_abs_rad = float(np.mean(np.abs(arr)))

    return {
        'count': len(arr),
        'mean_cos': float(c),
        'mean_sin': float(s),
        'mean_resultant_length': R,
        'circular_mean': mu,
        'circular_variance': circ_var,
        'circular_std': circ_std,
        'mean_abs_angle_rad': mean_abs_rad,
        'mean_abs_angle_deg': float(np.rad2deg(mean_abs_rad))
    }


def fit_von_mises_pdf(centers, pdf, pdf_std=None, unit='rad'):
    """
    角度 PDF に対して von Mises 分布（円正規分布）をフィッティングする:
    P(θ) = exp(kappa * cos(θ - mu)) / (2 * π * I0(kappa))
    """
    centers = np.asarray(centers)
    pdf = np.asarray(pdf)

    x_rad = np.deg2rad(centers) if unit == 'deg' else centers
    y = pdf * (180.0 / np.pi if unit == 'deg' else 1.0)

    mask = (y > 0) & np.isfinite(x_rad) & np.isfinite(y)
    x_data = x_rad[mask]
    y_data = y[mask]

    if len(x_data) < 4:
        return None

    def von_mises_func(x, kappa, mu):
        k = np.clip(kappa, 1e-4, 50.0)
        return np.exp(k * np.cos(x - mu)) / (2.0 * np.pi * i0(k))

    sigma = None
    if pdf_std is not None:
        std_masked = np.asarray(pdf_std)[mask]
        if np.all(std_masked > 0) and np.all(np.isfinite(std_masked)):
            sigma = std_masked * (180.0 / np.pi if unit == 'deg' else 1.0)

    try:
        popt, pcov = curve_fit(
            von_mises_func,
            x_data,
            y_data,
            p0=[1.0, 0.0],
            bounds=([0.0, -np.pi], [100.0, np.pi]),
            sigma=sigma,
            maxfev=5000
        )
        perr = np.sqrt(np.diag(pcov)) if pcov is not None else [0.0, 0.0]
    except Exception:
        return None

    kappa_fit, mu_fit = popt
    y_pred = von_mises_func(x_data, kappa_fit, mu_fit)
    ss_res = np.sum((y_data - y_pred) ** 2)
    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    fit_x_rad = np.linspace(np.min(x_rad), np.max(x_rad), 200)
    fit_y_rad = von_mises_func(fit_x_rad, kappa_fit, mu_fit)

    fit_x_out = np.rad2deg(fit_x_rad) if unit == 'deg' else fit_x_rad
    fit_y_out = fit_y_rad * (np.pi / 180.0 if unit == 'deg' else 1.0)

    return {
        'kappa': float(kappa_fit),
        'kappa_err': float(perr[0]),
        'mu': float(mu_fit),
        'mu_err': float(perr[1]),
        'r_squared': float(r2),
        'fit_x': fit_x_out,
        'fit_y': fit_y_out
    }


def fit_gaussian_pdf(centers, pdf, pdf_std=None):
    """
    角度変化 PDF に対してガウス分布をフィッティングする:
    P(θ) = A * exp(- (θ - mu)^2 / (2 * sigma^2))
    """
    centers = np.asarray(centers)
    pdf = np.asarray(pdf)
    mask = (pdf > 0) & np.isfinite(centers) & np.isfinite(pdf)
    x_data = centers[mask]
    y_data = pdf[mask]

    if len(x_data) < 4:
        return None

    def gauss(x, A, mu, sigma_val):
        return A * np.exp(-((x - mu) ** 2) / (2.0 * max(1e-6, sigma_val) ** 2))

    try:
        init_A = np.max(y_data)
        init_mu = 0.0
        init_sigma = np.std(x_data) if np.std(x_data) > 0 else 1.0
        popt, pcov = curve_fit(
            gauss,
            x_data,
            y_data,
            p0=[init_A, init_mu, init_sigma],
            bounds=([0.0, -np.inf, 1e-4], [np.inf, np.inf, np.inf]),
            maxfev=5000
        )
        perr = np.sqrt(np.diag(pcov)) if pcov is not None else [0.0, 0.0, 0.0]
    except Exception:
        return None

    A_fit, mu_fit, sigma_fit = popt
    y_pred = gauss(x_data, A_fit, mu_fit, sigma_fit)
    ss_res = np.sum((y_data - y_pred) ** 2)
    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    fit_x = np.linspace(np.min(x_data), np.max(x_data), 200)
    fit_y = gauss(fit_x, A_fit, mu_fit, sigma_fit)

    return {
        'A': float(A_fit),
        'mu': float(mu_fit),
        'sigma': float(sigma_fit),
        'sigma_err': float(perr[2]),
        'r_squared': float(r2),
        'fit_x': fit_x,
        'fit_y': fit_y
    }
