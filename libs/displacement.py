import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import chain
from scipy.optimize import curve_fit

import os
import sys

from libs import cal_vel as cv

def _calculate_displacements(pos):
    return np.vstack([pos[tau:] - pos[:-tau] for tau in range(1, len(pos))])

def PDF(df, tau, scale=1, abs_PDF = bool, xy = 'x'):
    pdf_list = []
    
    for _, group in df.groupby('particle'):
        group = group.sort_values(by='frame')  # フレーム順に並べ替え
        pos = scale * group[[xy]].to_numpy()  # xをNumPy 配列化
        displacements = pos[tau:] - pos[:-tau]
        if abs_PDF == True:
            displacements = np.abs(displacements)

        pdf_list.extend(displacements)  # 各粒子ごとにリストへ追加

    # 各粒子の結果を連結して 2 次元配列にする
    return np.array(pdf_list).flatten()


def PDF_theta(df, tau, theta_array, scale=1, abs_PDF=False, component='parallel'):
    pdf_list = []
    
    for _, group in df.groupby('particle'):
        group = group.sort_values(by='frame')
        
        frames = group['frame'].to_numpy()
        x = scale * group['x'].to_numpy()
        y = scale * group['y'].to_numpy()
        
        dx = x[tau:] - x[:-tau]
        dy = y[tau:] - y[:-tau]
        
        start_frames = frames[:-tau]
        valid_idx = start_frames < len(theta_array)
        
        dx = dx[valid_idx]
        dy = dy[valid_idx]
        valid_frames = start_frames[valid_idx]
        th = theta_array[valid_frames]
        
        if component == 'parallel':
            displacements = dx * np.cos(th) + dy * np.sin(th)
        elif component == 'perpendicular':
            displacements = -dx * np.sin(th) + dy * np.cos(th)
            
        if abs_PDF:
            displacements = np.abs(displacements)
            
        pdf_list.extend(displacements)
        
    return np.array(pdf_list).flatten()


def calc_displacement_magnitudes(df, tau, scale=0.11, component='norm', theta_array=None, signed=False):
    """
    粒子の軌跡データフレームから指定ラグタイム tau における変位（または変位の絶対値）を計算する。

    Parameters
    ----------
    df : pd.DataFrame
        'particle', 'frame', 'x', 'y' 列を持つデータフレーム。
    tau : int
        ラグタイム（フレーム数）。
    scale : float, optional
        ピクセルからμmへの変換スケール（デフォルト: 0.11 μm/px）。
    component : str, optional
        変位の成分指定:
        - 'norm', '2d', 'magnitude', 'r': 2次元変位ノルム sqrt(dx^2 + dy^2)
        - 'x': x方向変位 (|dx| または dx)
        - 'y': y方向変位 (|dy| または dy)
        - 'both_xy': x方向とy方向の変位を連結
        - 'parallel', 'par': 大域ネマチック主軸への射影成分 dx*cos(theta) + dy*sin(theta)
        - 'perpendicular', 'perp': 大域ネマチック主軸に直交する成分 -dx*sin(theta) + dy*cos(theta)
    theta_array : np.ndarray, optional
        各フレームの大域配向角 theta(t) (ラジアン)。component が 'parallel' または 'perpendicular' の場合に必須。
    signed : bool, optional
        True の場合は符号付き変位、False（デフォルト）の場合は変位の絶対値を返す。
        ※ component='norm' の場合は常に非負。

    Returns
    -------
    np.ndarray: 変位の1次元配列
    """
    comp = component.lower()
    disp_list = []

    for _, group in df.groupby('particle'):
        group = group.sort_values(by='frame')
        frames = group['frame'].to_numpy()
        if len(frames) <= tau:
            continue

        x = scale * group['x'].to_numpy()
        y = scale * group['y'].to_numpy()

        dx = x[tau:] - x[:-tau]
        dy = y[tau:] - y[:-tau]

        if comp in ['norm', '2d', 'magnitude', 'r']:
            disp = np.sqrt(dx**2 + dy**2)
        elif comp == 'x':
            disp = dx if signed else np.abs(dx)
        elif comp == 'y':
            disp = dy if signed else np.abs(dy)
        elif comp == 'both_xy':
            dx_comp = dx if signed else np.abs(dx)
            dy_comp = dy if signed else np.abs(dy)
            disp = np.concatenate([dx_comp, dy_comp])
        elif comp in ['parallel', 'par']:
            if theta_array is None:
                raise ValueError("component='parallel' requires theta_array.")
            start_frames = frames[:-tau]
            valid_idx = start_frames < len(theta_array)
            th = theta_array[start_frames[valid_idx]]
            disp = dx[valid_idx] * np.cos(th) + dy[valid_idx] * np.sin(th)
            if not signed:
                disp = np.abs(disp)
        elif comp in ['perpendicular', 'perp']:
            if theta_array is None:
                raise ValueError("component='perpendicular' requires theta_array.")
            start_frames = frames[:-tau]
            valid_idx = start_frames < len(theta_array)
            th = theta_array[start_frames[valid_idx]]
            disp = -dx[valid_idx] * np.sin(th) + dy[valid_idx] * np.cos(th)
            if not signed:
                disp = np.abs(disp)
        else:
            raise ValueError(f"Unknown component '{component}'. Choose from 'norm', 'x', 'y', 'both_xy', 'parallel', 'perpendicular'.")

        disp_list.extend(disp)

    return np.array(disp_list).flatten()


def calc_displacement_pdf(displacements, bins=50, density=True, bin_range=None):
    """
    変位配列から確率密度関数 (PDF) または度数分布を計算する。

    Parameters
    ----------
    displacements : np.ndarray or list
        変位のデータ配列。
    bins : int or array-like or str, optional
        ビンの数、境界配列、または bin アルゴリズム名（デフォルト: 50）。
    density : bool, optional
        True の場合は確率密度（全積分が1）、False の場合は度数を返す。
    bin_range : tuple of (float, float), optional
        ビンの範囲 (min, max)。None の場合はデータの最小値〜最大値。

    Returns
    -------
    bin_centers : np.ndarray
        各ビンの中心座標。
    pdf_values : np.ndarray
        確率密度（または度数）。
    bin_edges : np.ndarray
        ビンの境界配列。
    """
    displacements = np.asarray(displacements)
    displacements = displacements[~np.isnan(displacements)]
    if len(displacements) == 0:
        return np.array([]), np.array([]), np.array([])

    counts, bin_edges = np.histogram(displacements, bins=bins, range=bin_range, density=density)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return bin_centers, counts, bin_edges


def calc_displacement_evolution(df, max_tau, scale=0.11, component='norm', theta_array=None, signed=False, frame_interval=4.0):
    """
    ラグタイム tau = 1, 2, ..., max_tau に対する平均変位・標準偏差・MSD等の時間変化を算出する。

    Returns
    -------
    pd.DataFrame:
        'tau_frame', 'lag_time', 'mean_disp', 'std_disp', 'sem_disp', 'median_disp', 'msd', 'count'
    """
    records = []
    for tau in range(1, max_tau + 1):
        try:
            disp = calc_displacement_magnitudes(df, tau=tau, scale=scale, component=component, 
                                                theta_array=theta_array, signed=signed)
        except Exception:
            continue
        if len(disp) == 0:
            continue
        mean_d = float(np.mean(disp))
        std_d = float(np.std(disp))
        sem_d = float(std_d / np.sqrt(len(disp))) if len(disp) > 0 else np.nan
        med_d = float(np.median(disp))
        msd = float(np.mean(disp**2))
        records.append({
            'tau_frame': tau,
            'lag_time': tau * frame_interval,
            'mean_disp': mean_d,
            'std_disp': std_d,
            'sem_disp': sem_d,
            'median_disp': med_d,
            'msd': msd,
            'count': len(disp)
        })
    return pd.DataFrame(records)


def pdf_sim(MT_conc, S, tau, start, stop, r, abs_PDF = bool):
    dis = []
    for seed in range(start, stop):
        data =np.load(f"/Volumes/SSD-PGU3/sasaki/MTsingleBeads/analysis/ctrw/data/v2/MT{MT_conc}/S{S}/seed{seed}.npz")
        pos = data["positions"]
        displacement = r*(pos[tau:] - pos[:-tau])
        x = displacement[:,0]
        if abs_PDF == True:
            x = np.abs(x)
        dis.extend(x)

    return dis



def fit_exponential_pdf(centers, pdf, pdf_std=None, r_min=None, r_max=None, fit_mode='log'):
    """
    変位 PDF P(r) に対して指数関数 P(r) = A * exp(-r / lambda) をフィッティングする。
    デフォルトで対数空間 (ln P(r) = ln A - r / lambda) でフィッティングを行い、
    裾（テール部）まで均等な重みで指数減衰を捉える。
    
    Parameters
    ----------
    centers : np.ndarray
        ビン中心座標 r [um]
    pdf : np.ndarray
        確率密度 P(r)
    pdf_std : np.ndarray, optional
        標準偏差（重み付け用）
    r_min, r_max : float, optional
        フィッティング対象の r 範囲（テール部分のみをフィッティングしたい場合に指定）
    fit_mode : str, optional
        'log' (対数空間でフィッティング, デフォルト) または 'linear' (線形空間でフィッティング)
        
    Returns
    -------
    dict or None
        'A': 振幅 A
        'A_err': 振幅の標準誤差
        'lambda': 特性減衰長 lambda [um]
        'lambda_err': 特性減衰長の標準誤差
        'b': 減衰率 b = 1/lambda [1/um]
        'r_squared': 決定係数 R^2 (対数空間)
        'r_squared_linear': 決定係数 R^2 (線形空間)
        'popt': 最適化パラメータ [A, lambda]
        'perr': パラメータ標準誤差 [A_err, lambda_err]
        'fit_x': フィッティング評価用 x 配列
        'fit_y': フィッティング評価用 y 配列
    """
    centers = np.asarray(centers)
    pdf = np.asarray(pdf)
    
    # 有効なデータ点 (pdf > 0, centers >= 0)
    mask = (pdf > 0) & np.isfinite(pdf) & np.isfinite(centers)
    if r_min is not None:
        mask = mask & (centers >= r_min)
    if r_max is not None:
        mask = mask & (centers <= r_max)
        
    x_data = centers[mask]
    y_data = pdf[mask]
    
    if len(x_data) < 3:
        return None
        
    def exp_func(x, A, decay_len):
        return A * np.exp(-x / decay_len)
        
    def log_exp_func(x, log_A, inv_lambda):
        return log_A - inv_lambda * x

    log_y = np.log(y_data)
    
    if fit_mode == 'log':
        # 対数空間でのフィッティング: ln P(r) = ln A - (1/lambda) * r
        # 誤差伝搬: sigma_{ln y} = sigma_y / y
        sigma_log = None
        if pdf_std is not None:
            std_masked = np.asarray(pdf_std)[mask]
            valid_std = (std_masked > 0) & np.isfinite(std_masked)
            if np.all(valid_std):
                sigma_log = np.clip(std_masked / y_data, 1e-3, 10.0)

        try:
            # 初期推定値
            slope_init, intercept_init = np.polyfit(x_data, log_y, 1)
            init_inv_lambda = -slope_init if slope_init < 0 else (1.0 / np.mean(x_data) if np.mean(x_data) > 0 else 1.0)
            init_log_A = intercept_init

            popt_log, pcov_log = curve_fit(
                log_exp_func,
                x_data,
                log_y,
                p0=[init_log_A, init_inv_lambda],
                bounds=([-np.inf, 1e-6], [np.inf, np.inf]),
                sigma=sigma_log,
                maxfev=5000
            )
            log_A_fit, inv_lambda_fit = popt_log
            perr_log = np.sqrt(np.diag(pcov_log)) if pcov_log is not None else [0.0, 0.0]
            
            A_fit = np.exp(log_A_fit)
            lambda_fit = 1.0 / inv_lambda_fit if inv_lambda_fit > 0 else np.nan
            # 誤差伝搬
            A_err = A_fit * perr_log[0]
            lambda_err = (1.0 / (inv_lambda_fit ** 2)) * perr_log[1] if inv_lambda_fit > 0 else 0.0
            popt = [A_fit, lambda_fit]
            perr = [A_err, lambda_err]
        except Exception:
            slope, intercept = np.polyfit(x_data, log_y, 1)
            lambda_fit = -1.0 / slope if slope < 0 else 1.0
            A_fit = np.exp(intercept)
            popt = [A_fit, lambda_fit]
            perr = [0.0, 0.0]

    else:
        # 線形空間での直接非線形フィッティング
        slope, intercept = np.polyfit(x_data, log_y, 1)
        init_lambda = -1.0 / slope if slope < 0 else (np.mean(x_data) if np.mean(x_data) > 0 else 1.0)
        init_A = np.exp(intercept)
        try:
            sigma = None
            if pdf_std is not None:
                std_masked = np.asarray(pdf_std)[mask]
                if np.all(std_masked > 0):
                    sigma = std_masked

            popt, pcov = curve_fit(
                exp_func,
                x_data,
                y_data,
                p0=[init_A, init_lambda],
                bounds=([0.0, 1e-6], [np.inf, np.inf]),
                sigma=sigma,
                maxfev=5000
            )
            perr = np.sqrt(np.diag(pcov)) if pcov is not None else [0.0, 0.0]
        except Exception:
            popt = [init_A, init_lambda]
            perr = [0.0, 0.0]

    A_fit, lambda_fit = popt

    # 決定係数 R^2 (対数空間)
    log_y_pred = np.log(A_fit) - (x_data / lambda_fit)
    ss_res_log = np.sum((log_y - log_y_pred) ** 2)
    ss_tot_log = np.sum((log_y - np.mean(log_y)) ** 2)
    r2_log = 1.0 - (ss_res_log / ss_tot_log) if ss_tot_log > 0 else 0.0

    # 決定係数 R^2 (線形空間)
    y_pred = exp_func(x_data, A_fit, lambda_fit)
    ss_res = np.sum((y_data - y_pred) ** 2)
    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
    r2_linear = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    fit_x = np.linspace(np.min(x_data), np.max(x_data), 200)
    fit_y = exp_func(fit_x, A_fit, lambda_fit)

    return {
        'A': float(A_fit),
        'A_err': float(perr[0]),
        'lambda': float(lambda_fit),
        'lambda_err': float(perr[1]),
        'b': float(1.0 / lambda_fit) if lambda_fit > 0 else np.nan,
        'r_squared': float(r2_log),
        'r_squared_linear': float(r2_linear),
        'popt': popt,
        'perr': perr,
        'x_data': x_data,
        'y_data': y_data,
        'fit_x': fit_x,
        'fit_y': fit_y
    }


def fit_PDF(bins, counts, min_x, max_x):
    def line(x, a, b):
        return a * x + b
    popt, pcov = curve_fit(line, np.log10(bins[min_x:max_x]), np.log10(counts[min_x:max_x]))
    print(f"A {10**popt[1]}, $\\alpha$ {popt[0]}")

    return popt, pcov

def _taac(x, tau):
    As = 0
    T = len(x)
    for t in range(T-tau):
        A = np.dot(x[t+tau], x[t])
        As += A

    return As/T

def calculate_dacf(df, scale, lag_time_frames, max_timeshift_frames):
    """
    データフレームから変位自己相関関数(DACF)を計算する関数。

    Args:
        df (pd.DataFrame): x, y, frame, particle を列に持つデータフレーム。
        lag_time_frames (int): ラグタイム Δt (フレーム数)。
        max_timeshift_frames (int): 計算する最大時間シフト τ (フレーム数)。

    Returns:
        pd.DataFrame: 'time_shift_frames', 'dacf' を列に持つ結果のデータフレーム。
    """
    
    # 1. 各粒子の変位を計算
    # frameでソートしておく
    df = df.sort_values(by=['particle', 'frame']).reset_index(drop=True)

    # ラグタイムτにおける変位(dx, dy)を計算
    displacements = scale * df.groupby('particle')[['x', 'y']].diff(periods=lag_time_frames).dropna()
    displacements = df.loc[displacements.index][['particle', 'frame']].join(displacements)
    displacements.rename(columns={'x': 'dx', 'y': 'dy'}, inplace=True)
    displacements.dropna(inplace=True)

    dacf_results = []

    # 2. 時間シフトtauごとに自己相関を計算
    for tau in range(max_timeshift_frames + 1):
        # Δ=0 の場合は、変位ベクトルの二乗平均 (MSD)
        if tau == 0:
            # 内積を計算
            displacements['dot_product'] = displacements['dx']**2 + displacements['dy']**2
            # 粒子ごとに平均
            dacf_per_particle = displacements.groupby('particle')['dot_product'].mean()
            # アンサンブル平均 (全粒子で平均)
            mean_dacf = dacf_per_particle.mean()
            msd = mean_dacf # tau=0 のときはMSDに等しい
            dacf_results.append({'time_shift_frames': tau, 'dacf': mean_dacf})
            continue

        # tau>0 の場合
        # tauフレームだけシフトした変位を計算
        shifted_displacements = displacements.copy()
        shifted_displacements['frame'] -= tau
        
        # 元の変位とシフトした変位をマージ
        merged = pd.merge(
            displacements,
            shifted_displacements,
            on=['particle', 'frame'],
            suffixes=('', '_shifted')
        )
        
        if merged.empty:
            continue
            
        # 内積を計算
        merged['dot_product'] = merged['dx'] * merged['dx_shifted'] + merged['dy'] * merged['dy_shifted']
        
        # 粒子ごとに平均
        dacf_per_particle = merged.groupby('particle')['dot_product'].mean()
        
        # アンサンブル平均
        mean_dacf = dacf_per_particle.mean()
        dacf_results.append({'time_shift_frames': tau, 'dacf': mean_dacf})

    # 3. 正規化してデータフレームとして返す
    results_df = pd.DataFrame(dacf_results)
    # tau=0の値 (MSD) で正規化
    results_df['dacf'] = results_df['dacf'] / msd
    
    return results_df



def _calculate_msd(pos, time_averaged = True):
    if time_averaged == True:
        num_steps = len(pos)
        msd = []
        for tau in range(1, num_steps):
            displacements = pos[tau:] - pos[:-tau]
            squared_displacements = np.mean(displacements**2, axis=1)
            msd_tau = np.mean(squared_displacements)
            msd.append(msd_tau)
    else:
        displacements = pos[1:] - pos[0]
        msd = np.sum(displacements**2, axis = 1)
    
    return np.array(msd)

def imsd(df, scale = 1, time_scale = 1, time_averaged=True,  display=True, figsize=(10,8), title='indivisual MSD'):
    msd_df = pd.DataFrame(columns=['particle', 'lag time', 'MSD'])

    if display == True:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set(ylabel='$\\langle \\Delta r^2 \\rangle$ [\u03bcm$^2$]',
        xlabel='lag time $\Delta t [s]$')
        ax.set_title(title)

    for particle_id, group in df.groupby('particle'):
        data = pd.DataFrame()
        group = group.sort_values(by='frame')  # フレーム順に並べ替え
        x = scale * group['x'].to_numpy()
        y = scale * group['y'].to_numpy()
        pos = np.array([x, y]).T
        lag_t = time_scale * np.arange(1,pos.shape[0])
        msd=_calculate_msd(pos, time_averaged)
        data = pd.DataFrame({'particle':particle_id, 'lag time':lag_t, 'MSD':msd})

        msd_df=pd.concat([msd_df, data])


        if display == True:
            ax.plot(lag_t, msd, color='black', alpha = 0.2)

    return msd_df

def emsd(imsd_list, display=True, figsize=(10,8), title = 'Ensemble MSD'):
    emsd=imsd_list.groupby('lag time').mean()['MSD']
    N = len(imsd_list[imsd_list['lag time']==0].index)
    emsd_err = imsd_list.groupby('lag time').std()['MSD']/np.sqrt(N)
    
    if display == True:
        times = emsd.index
        fig, ax = plt.subplots(figsize=figsize)
        ax.errorbar(times, emsd, yerr= emsd_err, fmt ='o')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set(ylabel='$\\langle \\Delta r^2 \\rangle$ [\u03bcm$^2$]',
        xlabel='lag time $\Delta t [s]$')
        ax.set_title(title)
        ax.grid('True')

    return emsd, emsd_err

def _calculate_ngp(pos):
    num_steps = len(pos)
    ngps = []
    for tau in range(1, num_steps):
        displacements = pos[tau:] - pos[:-tau]
        squared_displacements =np.mean(displacements ** 2, axis=1)
        msd = np.mean(squared_displacements)
        displacements_pow4 =np.mean(displacements ** 4, axis=1)
        mdp4 = np.mean(displacements_pow4)
        ngp =  (1/2)*(mdp4 /(msd ** 2)) - 1
        ngps.append(ngp)
    
    return np.array(ngps)
    
def _calculate_related_ngp(pos):
    num_steps = len(pos)
    ngps = []
    for tau in range(1, num_steps):
        displacements = pos[tau:] - pos[:-tau]
        squared_displacements = np.sum(displacements**2, axis=1)
        msd = np.mean(squared_displacements)
        displacements_pow4 = np.sum(displacements**4, axis=1)
        mdp4 = np.mean(displacements_pow4)
        ngp = (mdp4 - 3 * msd ** 2)/msd ** 2
        ngps.append(ngp)
    
    return np.array(ngps)


def ingp(df, time_scale = 1, display=True, figsize=(10,8), title='indivisual NGP'):
    ngp_df = pd.DataFrame(columns=['particle', 'lag time', 'NGP'])

    if display == True:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set(ylabel='$\\langle \\Delta r^4 \\rangle / \\langle \\Delta r^2 \\rangle ^2$',
        xlabel='lag time $\Delta t [s]$')
        ax.set_title(title)

    for particle_id, group in df.groupby('particle'):
        data = pd.DataFrame()
        group = group.sort_values(by='frame')  # フレーム順に並べ替え
        x = group['x'].to_numpy()
        y = group['y'].to_numpy()
        pos = np.array([x, y]).T
        lag_t = time_scale * np.arange(1,pos.shape[0])
        ngp=_calculate_ngp(pos)
        data = pd.DataFrame({'particle':particle_id, 'lag time':lag_t, 'NGP':ngp})

        ngp_df=pd.concat([ngp_df, data])


        if display == True:
            ax.plot(lag_t, ngp, color='black', alpha = 0.2)

        return ngp_df
    
def engp(ingp_list, display=True, figsize=(10,8), title = 'Ensemble NGP'):
    engp=ingp_list.groupby('lag time').mean()['NGP']
    N = len(ingp_list[ingp_list['lag time']==0].index)
    engp_err = ingp_list.groupby('lag time').std()['NGP']/np.sqrt(N)
    
    if display == True:
        times = emsd.index
        fig, ax = plt.subplots(figsize=figsize)
        ax.errorbar(times, engp, yerr= engp_err, fmt ='o')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set(ylabel='$(\\langle \\Delta r^4 \\rangle - 3 \\langle \\Delta r^2 \\rangle ^2)/ \\langle \\Delta r^2 \\rangle ^2$',
        xlabel='lag time $\Delta t [s]$')
        ax.set_title(title)
        ax.grid('True')

    return engp, engp_err

def get_msd(path, scale=0.11, interval=10, threshold = 0,time_averaged = True):
    track = cv.cal(pd.read_csv(path), scale=scale ,frame_interval=interval)
    track = track[track['v'] >= threshold]
    IMSD = imsd(track, scale, interval, time_averaged, display=bool)
    EMSD, _ = emsd(IMSD, display=bool)

    return IMSD, EMSD

def get_emsd_err(IMSD):
    emsd = IMSD.groupby("lag time").mean()
    emsd_err = IMSD.groupby("lag time").sem()['MSD'].to_numpy()

    return emsd, emsd_err

def get_msd_df(path_list, scale=0.11, interval=10, threshold = 0):
    MSDs = []

    i = 0
    for path in path_list:
        imsd, _ = get_msd(path, scale, interval, threshold)
        imsd['particle'] += i*10000
        MSDs.append(imsd)
        i += 1

    MSDs = pd.concat(MSDs)

    return MSDs

def spatial_velocity_correlation(df, lag_time_frames, scale=1, normalize=True):
    """
    同じフレームにおける粒子間の運動方向の空間相関を計算し、フレームごとにアンサンブル平均をとる関数。
    
    Args:
        df (pd.DataFrame): 'frame', 'particle', 'x', 'y' を列に持つデータフレーム。
        lag_time_frames (int): 変位を計算するためのラグタイム Δt (フレーム数)。
        scale (float): 空間スケール (ピクセル -> um など)。
        normalize (bool): True なら運動方向ベクトルを正規化して cosθ を計算。
                          False なら内積をそのまま計算。
        
    Returns:
        pd.DataFrame: 各フレームにおける空間相関のアンサンブル平均を含むデータフレーム ('frame', 'correlation')。計算可能なペアが存在しない場合は空のデータフレーム。
    """
    disp_list = []
    for particle, group in df.groupby('particle'):
        group = group.sort_values(by='frame')
        frames = group['frame'].to_numpy()
        x = group['x'].to_numpy() * scale
        y = group['y'].to_numpy() * scale
        
        if len(frames) <= lag_time_frames:
            continue
            
        start_frames = frames[:-lag_time_frames]
        start_x = x[:-lag_time_frames]
        start_y = y[:-lag_time_frames]
        
        dx = x[lag_time_frames:] - x[:-lag_time_frames]
        dy = y[lag_time_frames:] - y[:-lag_time_frames]
        
        particle_df = pd.DataFrame({
            'frame': start_frames,
            'particle': particle,
            'x': start_x,
            'y': start_y,
            'dx': dx,
            'dy': dy
        })
        disp_list.append(particle_df)
        
    if not disp_list:
        return pd.DataFrame(columns=['frame', 'correlation'])
        
    disp_df = pd.concat(disp_list, ignore_index=True)
    
    correlations = []
    
    for frame, group in disp_df.groupby('frame'):
        coords = group[['x', 'y']].to_numpy()
        disp = group[['dx', 'dy']].to_numpy()
        
        if len(coords) < 2:
            continue
            
        if normalize:
            norms = np.linalg.norm(disp, axis=1, keepdims=True)
            with np.errstate(divide='ignore', invalid='ignore'):
                disp = np.where(norms > 0, disp / norms, 0)
                
        # disp の内積行列を計算
        corr_matrix = disp @ disp.T
        
        # 上三角成分（対角成分を除く）を取得
        i, j = np.triu_indices(len(coords), k=1)
        corrs = corr_matrix[i, j]
        
        if len(corrs) > 0:
            correlations.append({
                'frame': frame,
                'correlation': np.mean(corrs)
            })
            
    if not correlations:
        return pd.DataFrame(columns=['frame', 'correlation'])
        
    return pd.DataFrame(correlations)