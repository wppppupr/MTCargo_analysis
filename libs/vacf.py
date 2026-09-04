import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def _get_ylabel_and_title(mode, normalize, is_ensemble=False):
    prefix = "Ensemble " if is_ensemble else "Individual "
    m = mode.lower()
    if m in ['velocity', 'vec']:
        title = f"{prefix}Velocity Autocorrelation Function (VACF)"
        if normalize:
            ylabel = r'$\langle \vec{v}(t) \cdot \vec{v}(t+\tau) \rangle / \langle |\vec{v}|^2 \rangle$'
        else:
            ylabel = r'$\langle \vec{v}(t) \cdot \vec{v}(t+\tau) \rangle$'
    elif m in ['orientation', 'direction', 'unit']:
        title = f"{prefix}Orientation Autocorrelation Function (OACF)"
        ylabel = r'$\langle \hat{v}(t) \cdot \hat{v}(t+\tau) \rangle$'
    elif m in ['speed', 'scalar']:
        title = f"{prefix}Speed Autocorrelation Function (SACF)"
        if normalize:
            ylabel = r'$\langle v(t) v(t+\tau) \rangle / \langle v^2 \rangle$'
        else:
            ylabel = r'$\langle v(t) v(t+\tau) \rangle$'
    elif m in ['velocity_fluctuation', 'vel_fluc', 'v_fluc', 'velocity_fluc']:
        title = f"{prefix}Velocity Fluctuation Autocorrelation Function"
        if normalize:
            ylabel = r'$\langle \delta\vec{v}(t) \cdot \delta\vec{v}(t+\tau) \rangle / \langle |\delta\vec{v}|^2 \rangle$'
        else:
            ylabel = r'$\langle \delta\vec{v}(t) \cdot \delta\vec{v}(t+\tau) \rangle$'
    elif m in ['speed_fluctuation', 'speed_fluc', 's_fluc', 'scalar_fluc']:
        title = f"{prefix}Speed Fluctuation Autocorrelation Function"
        if normalize:
            ylabel = r'$\langle \delta v(t) \delta v(t+\tau) \rangle / \langle (\delta v)^2 \rangle$'
        else:
            ylabel = r'$\langle \delta v(t) \delta v(t+\tau) \rangle$'
    elif m in ['angle_change', 'deltatheta', 'delta_theta', 'turning_angle', 'angular_change']:
        title = f"{prefix}Angular Change Autocorrelation Function"
        if normalize:
            ylabel = r'$\langle \Delta\theta(t) \Delta\theta(t+\tau) \rangle / \langle (\Delta\theta)^2 \rangle$'
        else:
            ylabel = r'$\langle \Delta\theta(t) \Delta\theta(t+\tau) \rangle$'
    elif m in ['angle_change_fluctuation', 'deltatheta_fluc', 'delta_theta_fluc', 'angular_change_fluc']:
        title = f"{prefix}Angular Change Fluctuation Autocorrelation Function"
        if normalize:
            ylabel = r'$\langle \delta\Delta\theta(t) \delta\Delta\theta(t+\tau) \rangle / \langle (\delta\Delta\theta)^2 \rangle$'
        else:
            ylabel = r'$\langle \delta\Delta\theta(t) \delta\Delta\theta(t+\tau) \rangle$'
    else:
        title = f"{prefix}Autocorrelation Function"
        ylabel = 'Autocorrelation'
    return ylabel, title


def calculate_vacf(df, max_timeshift_frames, frame_interval=1, scale=1, mode='velocity', normalize=True):
    """
    データフレームから自己相関関数を計算する関数。(アンサンブル平均)
    
    Args:
        df (pd.DataFrame): 'particle', 'frame', 'x', 'y' を持つトラッキング結果のデータフレーム
        max_timeshift_frames (int): 計算する最大時間シフト τ (フレーム数)。
        frame_interval (float): フレーム間の時間間隔
        scale (float): 空間スケール (例えばピクセルからμmへの変換係数)
        mode (str): 計算モード:
            'velocity': 速度ベクトル
            'orientation': 配向単位ベクトル
            'speed': 速さスカラー
            'velocity_fluctuation': 速度ベクトルゆらぎ (平均速度ベクトルを引いた成分)
            'speed_fluctuation': 速さゆらぎ (平均速さを引いた成分)
            'angle_change': 進行方向の角度変化 Δθ = wrap(θ(t+1) - θ(t))
            'angle_change_fluctuation': 角度変化ゆらぎ Δθ - <Δθ>
        normalize (bool): τ=0 の値で正規化するかどうか
        
    Returns:
        pd.DataFrame: 'time_shift_frames', 'lag_time', 'vacf' を列に持つ結果のデータフレーム。
    """
    df = df.sort_values(by=['particle', 'frame']).reset_index(drop=True)
    m = mode.lower()
    
    # 速度ベクトル (vx, vy) を計算
    diffs = df.groupby('particle')[['x', 'y']].diff(periods=1)
    df['vx'] = (diffs['x'] * scale) / frame_interval
    df['vy'] = (diffs['y'] * scale) / frame_interval
    
    valid_df = df.dropna(subset=['vx', 'vy']).copy()
    
    if m in ['orientation', 'direction', 'unit']:
        speed = np.sqrt(valid_df['vx']**2 + valid_df['vy']**2)
        valid_mask = speed > 1e-12
        valid_df = valid_df[valid_mask].copy()
        speed = speed[valid_mask]
        valid_df['vx'] = valid_df['vx'] / speed
        valid_df['vy'] = valid_df['vy'] / speed
    elif m in ['speed', 'scalar']:
        valid_df['speed'] = np.sqrt(valid_df['vx']**2 + valid_df['vy']**2)
    elif m in ['velocity_fluctuation', 'vel_fluc', 'v_fluc', 'velocity_fluc']:
        # 各粒子の時間平均速度ベクトルを引く
        mean_v = valid_df.groupby('particle')[['vx', 'vy']].transform('mean')
        valid_df['vx'] = valid_df['vx'] - mean_v['vx']
        valid_df['vy'] = valid_df['vy'] - mean_v['vy']
    elif m in ['speed_fluctuation', 'speed_fluc', 's_fluc', 'scalar_fluc']:
        # 各粒子の速さの時間平均を引く
        speed = np.sqrt(valid_df['vx']**2 + valid_df['vy']**2)
        valid_df['speed'] = speed
        mean_s = valid_df.groupby('particle')['speed'].transform('mean')
        valid_df['speed'] = valid_df['speed'] - mean_s
    elif m in ['angle_change', 'deltatheta', 'delta_theta', 'turning_angle', 'angular_change',
               'angle_change_fluctuation', 'deltatheta_fluc', 'delta_theta_fluc', 'angular_change_fluc']:
        # 進行方向角 theta(t) = atan2(vy, vx)
        valid_df['theta'] = np.arctan2(valid_df['vy'], valid_df['vx'])
        # 連続フレーム間の角度変化 dtheta = wrap(theta(t) - theta(t-1))
        d_th = valid_df.groupby('particle')['theta'].diff(periods=1)
        valid_df['dtheta'] = np.arctan2(np.sin(d_th), np.cos(d_th))
        valid_df = valid_df.dropna(subset=['dtheta']).copy()
        if m in ['angle_change_fluctuation', 'deltatheta_fluc', 'delta_theta_fluc', 'angular_change_fluc']:
            mean_dth = valid_df.groupby('particle')['dtheta'].transform('mean')
            valid_df['dtheta'] = valid_df['dtheta'] - mean_dth
    
    vacf_results = []
    norm_val = 1.0 # 正規化用
    
    is_scalar_speed = m in ['speed', 'scalar', 'speed_fluctuation', 'speed_fluc', 's_fluc', 'scalar_fluc']
    is_scalar_dtheta = m in ['angle_change', 'deltatheta', 'delta_theta', 'turning_angle', 'angular_change',
                             'angle_change_fluctuation', 'deltatheta_fluc', 'delta_theta_fluc', 'angular_change_fluc']
    
    for tau in range(max_timeshift_frames + 1):
        if tau == 0:
            if is_scalar_speed:
                valid_df['dot_product'] = valid_df['speed']**2
            elif is_scalar_dtheta:
                valid_df['dot_product'] = valid_df['dtheta']**2
            else:
                valid_df['dot_product'] = valid_df['vx']**2 + valid_df['vy']**2
            vacf_per_particle = valid_df.groupby('particle')['dot_product'].mean()
            mean_vacf = vacf_per_particle.mean()
            norm_val = mean_vacf
            vacf_results.append({
                'time_shift_frames': tau,
                'lag_time': tau * frame_interval,
                'vacf': mean_vacf
            })
            continue

        if is_scalar_speed:
            cols = ['particle', 'frame', 'speed']
        elif is_scalar_dtheta:
            cols = ['particle', 'frame', 'dtheta']
        else:
            cols = ['particle', 'frame', 'vx', 'vy']

        shifted = valid_df[cols].copy()
        shifted['frame'] -= tau
        
        merged = pd.merge(
            valid_df[cols],
            shifted,
            on=['particle', 'frame'],
            suffixes=('', '_shifted')
        )
        
        if merged.empty:
            continue
            
        if is_scalar_speed:
            merged['dot_product'] = merged['speed'] * merged['speed_shifted']
        elif is_scalar_dtheta:
            merged['dot_product'] = merged['dtheta'] * merged['dtheta_shifted']
        else:
            merged['dot_product'] = merged['vx'] * merged['vx_shifted'] + merged['vy'] * merged['vy_shifted']
            
        vacf_per_particle = merged.groupby('particle')['dot_product'].mean()
        mean_vacf = vacf_per_particle.mean()
        
        vacf_results.append({
            'time_shift_frames': tau,
            'lag_time': tau * frame_interval,
            'vacf': mean_vacf
        })

    results_df = pd.DataFrame(vacf_results)
    
    # 正規化
    if normalize and m not in ['orientation', 'direction', 'unit']:
        if norm_val != 0 and not pd.isna(norm_val):
            results_df['vacf'] = results_df['vacf'] / norm_val
        
    return results_df


def ivacf(df, max_timeshift_frames, frame_interval=1, scale=1, mode='velocity', normalize=True,
          display=False, ax=None, figsize=(10, 8), title=None):
    """
    個別の粒子ごとに自己相関関数(VACF / OACF / SACF / Fluctuation / Angular Change)を計算する関数
    
    Args:
        df (pd.DataFrame): 'particle', 'frame', 'x', 'y' を持つトラッキング結果のデータフレーム
        max_timeshift_frames (int): 計算する最大時間シフト τ (フレーム数)。
        frame_interval (float): フレーム間の時間間隔 [s]
        scale (float): 空間スケール (例えばピクセルからμmへの変換係数)
        mode (str): 計算モード:
            'velocity': 速度ベクトル
            'orientation': 配向単位ベクトル
            'speed': 速さスカラー
            'velocity_fluctuation': 速度ベクトルゆらぎ
            'speed_fluctuation': 速さゆらぎ
            'angle_change': 進行方向の角度変化 Δθ
            'angle_change_fluctuation': 角度変化ゆらぎ
        normalize (bool): τ=0 の値で正規化するかどうか
        display (bool): プロットを表示するかどうか
        ax (matplotlib.axes.Axes, optional): プロット先のAxes
        figsize (tuple): 図のサイズ (axがNoneの場合に使用)
        title (str, optional): プロットのタイトル
        
    Returns:
        pd.DataFrame: 'particle', 'lag time', 'VACF' を列に持つデータフレーム
    """
    df = df.sort_values(by=['particle', 'frame']).reset_index(drop=True)
    m = mode.lower()
    
    diffs = df.groupby('particle')[['x', 'y']].diff(periods=1)
    df['vx'] = (diffs['x'] * scale) / frame_interval
    df['vy'] = (diffs['y'] * scale) / frame_interval
    
    valid_df = df.dropna(subset=['vx', 'vy']).copy()
    
    data_list = []
    
    ylabel, default_title = _get_ylabel_and_title(mode, normalize, is_ensemble=False)
    if title is None:
        title = default_title

    if display:
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        ax.set(ylabel=ylabel, xlabel=r'lag time $\Delta t$ [s]')
        ax.set_title(title)

    for particle_id, group in valid_df.groupby('particle'):
        group = group.sort_values(by='frame')
        
        vx = group['vx'].to_numpy()
        vy = group['vy'].to_numpy()
        
        if m in ['speed', 'scalar']:
            # 速さスカラー
            val = np.sqrt(vx**2 + vy**2)
            T = len(val)
            if T == 0:
                continue
            norm_val = np.mean(val**2)
            
            vacfs = []
            lag_t = []
            for tau in range(min(max_timeshift_frames + 1, T)):
                if tau == 0:
                    vacfs.append(norm_val)
                    lag_t.append(0.0)
                else:
                    prod = val[tau:] * val[:-tau]
                    vacfs.append(np.mean(prod))
                    lag_t.append(tau * frame_interval)
                    
            vacfs = np.array(vacfs)
            if normalize and norm_val != 0:
                vacfs = vacfs / norm_val

        elif m in ['speed_fluctuation', 'speed_fluc', 's_fluc', 'scalar_fluc']:
            # 速さゆらぎ: delta v(t) = v(t) - <v>_t
            val = np.sqrt(vx**2 + vy**2)
            mean_speed = np.mean(val)
            delta_val = val - mean_speed
            T = len(delta_val)
            if T == 0:
                continue
            norm_val = np.mean(delta_val**2)
            
            vacfs = []
            lag_t = []
            for tau in range(min(max_timeshift_frames + 1, T)):
                if tau == 0:
                    vacfs.append(norm_val)
                    lag_t.append(0.0)
                else:
                    prod = delta_val[tau:] * delta_val[:-tau]
                    vacfs.append(np.mean(prod))
                    lag_t.append(tau * frame_interval)
                    
            vacfs = np.array(vacfs)
            if normalize and norm_val != 0:
                vacfs = vacfs / norm_val

        elif m in ['orientation', 'direction', 'unit']:
            # 配向単位ベクトル
            speed = np.sqrt(vx**2 + vy**2)
            valid_mask = speed > 1e-12
            if not np.any(valid_mask):
                continue
            
            ux = np.zeros_like(vx)
            uy = np.zeros_like(vy)
            ux[valid_mask] = vx[valid_mask] / speed[valid_mask]
            uy[valid_mask] = vy[valid_mask] / speed[valid_mask]
            u = np.array([ux, uy]).T
            
            T = len(u)
            vacfs = []
            lag_t = []
            for tau in range(min(max_timeshift_frames + 1, T)):
                if tau == 0:
                    vacfs.append(1.0)
                    lag_t.append(0.0)
                else:
                    dot_products = np.sum(u[tau:] * u[:-tau], axis=1)
                    vacfs.append(np.mean(dot_products))
                    lag_t.append(tau * frame_interval)
            vacfs = np.array(vacfs)

        elif m in ['velocity_fluctuation', 'vel_fluc', 'v_fluc', 'velocity_fluc']:
            # 速度ベクトルゆらぎ: delta v(t) = v(t) - <v>_t
            delta_vx = vx - np.mean(vx)
            delta_vy = vy - np.mean(vy)
            v = np.array([delta_vx, delta_vy]).T
            T = len(v)
            if T == 0:
                continue
            norm_val = np.mean(np.sum(v**2, axis=1))
            
            vacfs = []
            lag_t = []
            for tau in range(min(max_timeshift_frames + 1, T)):
                if tau == 0:
                    vacfs.append(norm_val)
                    lag_t.append(0.0)
                else:
                    dot_products = np.sum(v[tau:] * v[:-tau], axis=1)
                    vacfs.append(np.mean(dot_products))
                    lag_t.append(tau * frame_interval)
                    
            vacfs = np.array(vacfs)
            if normalize and norm_val != 0:
                vacfs = vacfs / norm_val

        elif m in ['angle_change', 'deltatheta', 'delta_theta', 'turning_angle', 'angular_change',
                   'angle_change_fluctuation', 'deltatheta_fluc', 'delta_theta_fluc', 'angular_change_fluc']:
            # 角度変化 Δθ(t) = wrap(θ(t+1) - θ(t))
            theta = np.arctan2(vy, vx)
            if len(theta) < 2:
                continue
            d_th = np.arctan2(np.sin(np.diff(theta)), np.cos(np.diff(theta)))
            if m in ['angle_change_fluctuation', 'deltatheta_fluc', 'delta_theta_fluc', 'angular_change_fluc']:
                d_th = d_th - np.mean(d_th)
            
            T = len(d_th)
            if T == 0:
                continue
            norm_val = np.mean(d_th**2)
            
            vacfs = []
            lag_t = []
            for tau in range(min(max_timeshift_frames + 1, T)):
                if tau == 0:
                    vacfs.append(norm_val)
                    lag_t.append(0.0)
                else:
                    prod = d_th[tau:] * d_th[:-tau]
                    vacfs.append(np.mean(prod))
                    lag_t.append(tau * frame_interval)
                    
            vacfs = np.array(vacfs)
            if normalize and norm_val != 0:
                vacfs = vacfs / norm_val
        else:
            # 速度ベクトル (デフォルト)
            v = np.array([vx, vy]).T
            T = len(v)
            if T == 0:
                continue
            norm_val = np.mean(np.sum(v**2, axis=1))
            
            vacfs = []
            lag_t = []
            for tau in range(min(max_timeshift_frames + 1, T)):
                if tau == 0:
                    vacfs.append(norm_val)
                    lag_t.append(0.0)
                else:
                    dot_products = np.sum(v[tau:] * v[:-tau], axis=1)
                    vacfs.append(np.mean(dot_products))
                    lag_t.append(tau * frame_interval)
                    
            vacfs = np.array(vacfs)
            if normalize and norm_val != 0:
                vacfs = vacfs / norm_val
            
        data = pd.DataFrame({'particle': particle_id, 'lag time': lag_t, 'VACF': vacfs})
        data_list.append(data)
        
        if display:
            ax.plot(lag_t, vacfs, color="#333333", alpha=0.2)
            
    if len(data_list) > 0:
        vacf_df = pd.concat(data_list, ignore_index=True)
    else:
        vacf_df = pd.DataFrame(columns=['particle', 'lag time', 'VACF'])
        
    return vacf_df


def evacf(ivacf_list, display=False, ax=None, figsize=(10, 8), title=None, mode='velocity', normalize=True):
    """
    アンサンブル平均VACFを計算する関数
    """
    evacf_val = ivacf_list.groupby('lag time').mean()['VACF']
    N = len(ivacf_list[ivacf_list['lag time'] == 0].index)
    evacf_err = ivacf_list.groupby('lag time').std()['VACF'] / np.sqrt(N) if N > 0 else pd.Series(0, index=evacf_val.index)
    
    if display: 
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        times = evacf_val.index
        ylabel, default_title = _get_ylabel_and_title(mode, normalize, is_ensemble=True)
        if title is None:
            title = default_title
            
        ax.errorbar(times, evacf_val, yerr=evacf_err, fmt='o')
        ax.set(ylabel=ylabel, xlabel=r'lag time $\Delta t$ [s]')
        ax.set_title(title)
        ax.grid(True)

    return evacf_val, evacf_err

