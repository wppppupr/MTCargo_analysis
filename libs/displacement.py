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