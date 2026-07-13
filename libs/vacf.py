import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calculate_vacf(df, max_timeshift_frames, frame_interval=1, scale=1):
    """
    データフレームから速度自己相関関数(VACF)を計算する関数。
    (アンサンブル平均)
    
    Args:
        df (pd.DataFrame): 'particle', 'frame', 'x', 'y' を持つトラッキング結果のデータフレーム
        max_timeshift_frames (int): 計算する最大時間シフト τ (フレーム数)。
        frame_interval (float): フレーム間の時間間隔
        scale (float): 空間スケール (例えばピクセルからμmへの変換係数)
        
    Returns:
        pd.DataFrame: 'time_shift_frames', 'vacf' を列に持つ結果のデータフレーム。
    """
    df = df.sort_values(by=['particle', 'frame']).reset_index(drop=True)
    
    # 速度ベクトル (vx, vy) を計算
    # 前フレームとの差分を使用
    diffs = df.groupby('particle')[['x', 'y']].diff(periods=1)
    df['vx'] = (diffs['x'] * scale) / frame_interval
    df['vy'] = (diffs['y'] * scale) / frame_interval
    
    valid_df = df.dropna(subset=['vx', 'vy']).copy()
    
    vacf_results = []
    v_sq_mean = 1.0 # 正規化用
    
    for tau in range(max_timeshift_frames + 1):
        if tau == 0:
            valid_df['dot_product'] = valid_df['vx']**2 + valid_df['vy']**2
            vacf_per_particle = valid_df.groupby('particle')['dot_product'].mean()
            mean_vacf = vacf_per_particle.mean()
            v_sq_mean = mean_vacf
            vacf_results.append({'time_shift_frames': tau, 'vacf': mean_vacf})
            continue

        shifted = valid_df[['particle', 'frame', 'vx', 'vy']].copy()
        shifted['frame'] -= tau
        
        merged = pd.merge(
            valid_df[['particle', 'frame', 'vx', 'vy']],
            shifted,
            on=['particle', 'frame'],
            suffixes=('', '_shifted')
        )
        
        if merged.empty:
            continue
            
        merged['dot_product'] = merged['vx'] * merged['vx_shifted'] + merged['vy'] * merged['vy_shifted']
        vacf_per_particle = merged.groupby('particle')['dot_product'].mean()
        mean_vacf = vacf_per_particle.mean()
        
        vacf_results.append({'time_shift_frames': tau, 'vacf': mean_vacf})

    results_df = pd.DataFrame(vacf_results)
    
    # tau=0 の値で正規化
    if v_sq_mean != 0 and not pd.isna(v_sq_mean):
        results_df['vacf'] = results_df['vacf'] / v_sq_mean
        
    return results_df

def ivacf(df, max_timeshift_frames, frame_interval=1, scale=1, display=False, ax = None, figsize=(10,8), title='Individual VACF'):
    """
    個別の粒子ごとにVACFを計算する関数
    
    Args:
        df (pd.DataFrame): 'particle', 'frame', 'x', 'y' を持つトラッキング結果のデータフレーム
        max_timeshift_frames (int): 計算する最大時間シフト τ (フレーム数)。
        frame_interval (float): フレーム間の時間間隔
        scale (float): 空間スケール
    """
    df = df.sort_values(by=['particle', 'frame']).reset_index(drop=True)
    
    diffs = df.groupby('particle')[['x', 'y']].diff(periods=1)
    df['vx'] = (diffs['x'] * scale) / frame_interval
    df['vy'] = (diffs['y'] * scale) / frame_interval
    
    valid_df = df.dropna(subset=['vx', 'vy']).copy()
    
    data_list = []
    
    if display:
        if ax == None:
            fig, ax = plt.subplots(figsize=figsize)
        ax.set(ylabel=r'$\langle \vec{v}(t) \cdot \vec{v}(t+\tau) \rangle / \langle |\vec{v}|^2 \rangle$',
               xlabel=r'lag time $\Delta t$ [s]')
        ax.set_title(title)

    for particle_id, group in valid_df.groupby('particle'):
        group = group.sort_values(by='frame')
        
        vx = group['vx'].to_numpy()
        vy = group['vy'].to_numpy()
        v = np.array([vx, vy]).T
        
        T = len(v)
        vacfs = []
        lag_t = []
        
        # tau=0 の自己相関 (自己速度の二乗平均)
        v_sq = np.mean(np.sum(v**2, axis=1))
        
        for tau in range(min(max_timeshift_frames + 1, T)):
            if tau == 0:
                vacfs.append(v_sq)
                lag_t.append(0)
            else:
                dot_products = np.sum(v[tau:] * v[:-tau], axis=1)
                vacf_tau = np.mean(dot_products)
                vacfs.append(vacf_tau)
                lag_t.append(tau * frame_interval)
                
        vacfs = np.array(vacfs)
        if v_sq != 0:
            vacfs = vacfs / v_sq
            
        data = pd.DataFrame({'particle': particle_id, 'lag time': lag_t, 'VACF': vacfs})
        data_list.append(data)
        
        if display:
            ax.plot(lag_t, vacfs, color="#333333", alpha=0.2)
            
    if len(data_list) > 0:
        vacf_df = pd.concat(data_list, ignore_index=True)
    else:
        vacf_df = pd.DataFrame(columns=['particle', 'lag time', 'VACF'])
        
    return vacf_df

def evacf(ivacf_list, display=False, ax =None, figsize=(10,8), title='Ensemble VACF'):
    """
    アンサンブル平均VACFを計算する関数
    """
    evacf = ivacf_list.groupby('lag time').mean()['VACF']
    N = len(ivacf_list[ivacf_list['lag time']==0].index)
    evacf_err = ivacf_list.groupby('lag time').std()['VACF']/np.sqrt(N)
    
    if display: 
        if ax == None:
            fig, ax = plt.subplots(figsize=figsize)
        times = evacf.index
        ax.errorbar(times, evacf, yerr=evacf_err, fmt='o')
        ax.set(ylabel=r'$\langle \vec{v}(t) \cdot \vec{v}(t+\tau) \rangle / \langle |\vec{v}|^2 \rangle$',
               xlabel=r'lag time $\Delta t$ [s]')
        ax.set_title(title)
        ax.grid(True)

    return evacf, evacf_err
