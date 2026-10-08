import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def cal(
    tracking_df,
    scale=1,
    frame_interval=1,
    pos_window=None,
    pos_center=False,
    pos_min_periods=1,
    smooth_method=None,
    noise_std=0.05,
    k=3,
    smoothing_factor=None,
):
    """
    データフレーム内の各particleごとに速度ベクトル (vx, vy)、速さ (v)、
    角度 (theta)、角度変化 (dtheta, omega, alpha) などを計算して新しいカラムとして追加する関数。

    Parameters:
    - tracking_df: トラッキング結果のデータフレーム（'x', 'y', 'particle', 'frame'を含む）
    - scale: μm/pixel 変換スケール（デフォルト: 1）
    - frame_interval: フレーム間の時間間隔 [s]（デフォルト: 1）
    - smooth_method: None / 'none' / 'raw' (既定: 生データ差分法), 'spline' (平滑化スプライン + 解析的微分), 'moving_average' (移動平均)
    - noise_std: 観測ノイズ標準偏差 [μm] (spline の平滑化パラメータ s = N * noise_std^2 の計算用、デフォルト: 0.05)
    - k: スプラインの次数 (デフォルト: 3)
    - smoothing_factor: スプライン平滑化パラメータ s (明示指定する場合。None の場合は noise_std から自動計算)
    - pos_window: 位置移動平均ウィンドウ（int、指定時は移動平均モード）
    - pos_center: rolling の center 引数
    - pos_min_periods: rolling の min_periods

    Returns:
    - tracking_df: 速さ (v)、速度成分 (vx, vy)、角度 (theta)、平滑化座標 (x_s, y_s / x_smooth, y_smooth) 等のカラムが追加されたデータフレーム
    """
    if pos_window is not None and pos_window > 1:
        smooth_method = "moving_average"

    s_method_str = str(smooth_method).lower() if smooth_method is not None else "none"

    if s_method_str in ("spline", "smoothing_spline"):
        from libs.trajectory_velocity import SplineVelocityEstimator

        estimator = SplineVelocityEstimator(
            smoothing_factor=smoothing_factor,
            noise_std=noise_std,
            k=k,
        )

        def calculate_spline_per_particle(df):
            df = df.sort_values('frame').copy()
            N = len(df)
            t = df['frame'].values * frame_interval
            x = df['x'].values
            y = df['y'].values

            if N >= 2:
                try:
                    res = estimator.estimate(t, x, y, scale=scale)
                    df['x_s'] = res.x_smooth
                    df['y_s'] = res.y_smooth
                    df['x_smooth'] = res.x_smooth
                    df['y_smooth'] = res.y_smooth
                    df['vx'] = res.vx
                    df['vy'] = res.vy
                    df['v'] = res.v
                    df['theta'] = res.theta
                    df['distance'] = res.v * frame_interval / scale
                except Exception:
                    return _calculate_diff_per_particle(df)
            else:
                return _calculate_diff_per_particle(df)

            # 速度が微小な点の theta を NaN に
            df.loc[df['v'] <= 1e-8, 'theta'] = np.nan

            # 単位方向ベクトル dx, dy
            with np.errstate(divide='ignore', invalid='ignore'):
                dx = np.where(df['v'] > 1e-8, df['vx'] / df['v'], np.nan)
                dy = np.where(df['v'] > 1e-8, df['vy'] / df['v'], np.nan)
            df['dx'] = dx
            df['dy'] = dy

            dx_prev = df['dx'].shift(1)
            dy_prev = df['dy'].shift(1)
            cross = dx_prev * df['dy'] - dy_prev * df['dx']
            dot = dx_prev * df['dx'] + dy_prev * df['dy']

            df['dtheta'] = np.arctan2(cross, dot)
            df.loc[df['v'] <= 1e-8, 'dtheta'] = np.nan
            df['omega'] = df['dtheta'] / frame_interval

            # 進行方向変化ベクトル alpha
            df['ddx'] = df['dx'] - dx_prev
            df['ddy'] = df['dy'] - dy_prev
            df['alpha'] = np.arctan2(df['ddy'], df['ddx'])
            df.loc[df['v'] <= 1e-8, 'alpha'] = np.nan

            df['t'] = t
            return df

        def _calculate_diff_per_particle(df):
            x_ref = df['x']
            y_ref = df['y']
            df['x_s'] = x_ref
            df['y_s'] = y_ref
            df['x_smooth'] = x_ref * scale
            df['y_smooth'] = y_ref * scale

            df['x_diff'] = x_ref - x_ref.shift(1)
            df['y_diff'] = y_ref - y_ref.shift(1)
            df['distance'] = np.sqrt(df['x_diff'] ** 2 + df['y_diff'] ** 2)
            df['v'] = scale * df['distance'] / frame_interval
            df['vx'] = scale * df['x_diff'] / frame_interval
            df['vy'] = scale * df['y_diff'] / frame_interval

            df['theta'] = np.arctan2(df['y_diff'], df['x_diff'])
            df.loc[df['distance'] == 0, 'theta'] = np.nan

            df['dx'] = df['x_diff'] / df['distance']
            df['dy'] = df['y_diff'] / df['distance']
            df.loc[df['distance'] == 0, ['dx', 'dy']] = np.nan

            dx_prev = df['dx'].shift(1)
            dy_prev = df['dy'].shift(1)
            cross = dx_prev * df['dy'] - dy_prev * df['dx']
            dot = dx_prev * df['dx'] + dy_prev * df['dy']

            df['dtheta'] = np.arctan2(cross, dot)
            df.loc[df['distance'] == 0, 'dtheta'] = np.nan
            df['omega'] = df['dtheta'] / frame_interval

            df['ddx'] = df['dx'] - dx_prev
            df['ddy'] = df['dy'] - dy_prev
            df['alpha'] = np.arctan2(df['ddy'], df['ddx'])
            df.loc[df['distance'] == 0, 'alpha'] = np.nan

            df['t'] = df['frame'] * frame_interval
            return df

        tracking_df = tracking_df.groupby('particle', group_keys=False).apply(calculate_spline_per_particle)
        return tracking_df

    # 従来の移動平均 / 差分法
    def calculate_v_and_theta(df, frame_interval=frame_interval):
        if pos_window is not None and pos_window > 1:
            df['x_s'] = df['x'].rolling(window=pos_window, center=pos_center, min_periods=pos_min_periods).mean()
            df['y_s'] = df['y'].rolling(window=pos_window, center=pos_center, min_periods=pos_min_periods).mean()
            x_ref = df['x_s']
            y_ref = df['y_s']
        else:
            x_ref = df['x']
            y_ref = df['y']

        # 前のフレームとの座標差を計算
        df['x_diff'] = x_ref - x_ref.shift(1)
        df['y_diff'] = y_ref - y_ref.shift(1)

        # 座標差から距離（速さ）を計算
        df['distance'] = np.sqrt(df['x_diff'] ** 2 + df['y_diff'] ** 2)

        # 速さを計算
        df['v'] = scale * df['distance'] / frame_interval
        df['vx'] = scale * df['x_diff'] / frame_interval
        df['vy'] = scale * df['y_diff'] / frame_interval
        df['x_smooth'] = x_ref * scale
        df['y_smooth'] = y_ref * scale

        # 角度（ラジアン）を計算（np.arctan2）。距離が0の点はNaNにする
        df['theta'] = np.arctan2(df['y_diff'], df['x_diff'])
        df.loc[df['distance'] == 0, 'theta'] = np.nan

        # 角度方向の変化（符号付き回転量）
        df['dx'] = df['x_diff'] / df['distance']
        df['dy'] = df['y_diff'] / df['distance']
        df.loc[df['distance'] == 0, ['dx', 'dy']] = np.nan

        dx_prev = df['dx'].shift(1)
        dy_prev = df['dy'].shift(1)
        cross = dx_prev * df['dy'] - dy_prev * df['dx']
        dot = dx_prev * df['dx'] + dy_prev * df['dy']

        df['dtheta'] = np.arctan2(cross, dot)
        df.loc[df['distance'] == 0, 'dtheta'] = np.nan
        df['omega'] = df['dtheta'] / frame_interval

        # Additional alpha: direction of change vector
        df['ddx'] = df['dx'] - df['dx'].shift(1)
        df['ddy'] = df['dy'] - df['dy'].shift(1)
        df['alpha'] = np.arctan2(df['ddy'], df['ddx'])
        df.loc[df['distance'] == 0, 'alpha'] = np.nan

        df['t'] = df['frame'] * frame_interval

        cols = ['v', 'vx', 'vy', 'theta', 'alpha', 't', 'dtheta', 'omega', 'x_smooth', 'y_smooth']
        if 'x_s' in df.columns:
            cols += ['x_s', 'y_s']
        return df[cols]

    # パーティクルごとに速度と角度を一度に計算してデータフレームに追加
    cols_to_assign = ['v', 'vx', 'vy', 'theta', 'alpha', 't', 'dtheta', 'omega', 'x_smooth', 'y_smooth']
    if pos_window is not None and pos_window > 1:
        cols_to_assign += ['x_s', 'y_s']

    res = tracking_df.groupby('particle', group_keys=False).apply(calculate_v_and_theta)
    for c in cols_to_assign:
        tracking_df[c] = res[c]

    return tracking_df


def cal_advanced(
    tracking_df: pd.DataFrame,
    method: str = "kalman_rts",
    scale: float = 1.0,
    frame_interval: float = 1.0,
    **kwargs
) -> pd.DataFrame:
    """
    高度な平滑化・微分手法を用いてトラッキングデータから速度・角速度等を計算する関数。

    Parameters:
    - tracking_df: トラッキング結果の DataFrame ('x', 'y', 'particle', 'frame' を含む)
    - method: 'kalman_rts' (カルマン/RTS スムーザー), 'spline' (平滑化スプライン), 'diff' (差分法)
    - scale: μm/pixel 変換スケール
    - frame_interval: フレーム間時間隔 [s]
    - kwargs: 各推定器へのオプション引数 (例: obs_noise_std, process_noise_std, smoothing_factor 等)

    Returns:
    - tracking_df: vx, vy, v, theta, omega, x_smooth, y_smooth, v_std 等が追加された DataFrame
    """
    from libs.trajectory_velocity import estimate_particle_velocities

    df_res = estimate_particle_velocities(
        tracking_df=tracking_df,
        method=method,
        frame_interval=frame_interval,
        scale=scale,
        **kwargs
    )

    # 角度変化 dtheta, 角速度 omega も追加計算
    def _calc_angular(df_p):
        theta = df_p['theta'].values
        # 連続する角度差 (wrapped to [-pi, pi])
        dtheta = np.diff(theta, prepend=theta[0])
        dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi
        df_p['dtheta'] = dtheta
        df_p['omega'] = dtheta / frame_interval
        return df_p

    df_res = df_res.groupby('particle', group_keys=False).apply(_calc_angular)
    return df_res



def plot_v(tracking_df, color = "black", alpha = 0.2, time_interval = 1, xlabel = 'x', ylabel = 'y', title = 'title'):
    """
    全てのパーティクルの速さをフレームごとにプロットする関数。

    Parameters:
    - tracking_df: トラッキング結果のデータフレーム（'x', 'y', 'particle', 'frame', 'v'を含む）
    """
    # パーティクルごとにデータをグループ化
    grouped = tracking_df.groupby('particle')
    
    # プロットを作成
    plt.figure(figsize=(10, 6))

    for particle, group in grouped:
        # 各パーティクルの速さをプロット
        plt.plot(time_interval * group['frame'], group['v'], label=f'Particle {particle}', color = color, alpha = alpha)

    # グラフの詳細設定
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    #plt.legend()
    plt.grid(True)
    plt.show()

def low_area(df, v_column='v', threshold=1.0, min_low_duration=1, time_interval = 1,xlabel = 'x', ylabel = 'y', title = 'title', ylim = (0.0, 1.0), display = True):
    """
    低速領域をプロットする関数。

    Parameters:
    - df: トラッキング結果のデータフレーム（'particle', 'frame', 'v'などを含む）
    - v_column: 速さのカラム名（デフォルトは'v'）
    - threshold: 低速領域の開始・終了地点を決定するための速さの閾値
    - min_low_duration: 低速領域の持続時間として認める最小フレーム数（短すぎる低速領域を除外する）

    Returns:
    - low_area_df: 低速領域のパーティクルid, 低速領域の開始、終了フレームをまとめる
    """

    lowarea = []

    # パーティクルごとにプロット
    for particle_id, group in df.groupby('particle'):
        group = group.sort_values(by='frame')  # フレーム順に並べ替え
        v = group[v_column].values
        frames = group['frame'].values

        # 低速領域の開始・終了地点を検出
        low_starts = []  # パルスの開始地点のリスト
        low_ends = []    # パルスの終了地点のリスト
        in_low = False   # 現在パルスの中かどうかを示すフラグ


        for i in range(len(v)):
            if not in_low and v[i] <= threshold:
                low_starts.append(i)  # 低速区間の開始インデックスを記録
                in_low = True
            elif in_low and v[i] > threshold:
                low_ends.append(i-1)  # 低速区間の終了インデックスを記録
                in_low = False

        # 低速領域の開始・終了のペアをプロット
        for start, end in zip(low_starts, low_ends):
            if end - start >= min_low_duration:  # 持続時間のフィルタリング
                lowarea.append({
                    'particle': particle_id,
                    'start_frame': frames[start],
                    'end_frame': frames[end],
                    'interval': (frames[end] - frames[start]) * time_interval
                })
                if display ==True:
                    plt.plot(time_interval * frames[start:end+1], v[start:end+1], label=f'Particle {particle_id}')
                
    
    # プロットの設定
    if display ==True:
        plt.xlim((0, time_interval * df['frame'].max()))
        plt.ylim(ylim)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.show()



    # 結果をデータフレームに変換
    low_area_df = pd.DataFrame(lowarea)

    return low_area_df

def pulses(df, v_column='v', threshold=1.0, min_pulse_duration=1, time_interval = 1,xlabel = 'x', ylabel = 'y', title = 'title', display = True):
    """
    パルス領域をプロットする関数。calculate_pulse_intervals_v3で識別したパルスの区間をプロット。

    Parameters:
    - df: トラッキング結果のデータフレーム（'particle', 'frame', 'v'などを含む）
    - v_column: 速さのカラム名（デフォルトは'v'）
    - threshold: パルスの開始・終了地点を決定するための速さの閾値
    - min_pulse_duration: パルスの持続時間として認める最小フレーム数（短すぎるパルスを除外する）

    Returns:
    - pulses_df: パルスのパーティクルid, 低速領域の開始、終了フレームをまとめる
    """

    pulses = []

    # パーティクルごとにプロット
    for particle_id, group in df.groupby('particle'):
        group = group.sort_values(by='frame')  # フレーム順に並べ替え
        v = group[v_column].values
        frames = group['frame'].values

        # パルスの開始・終了地点を検出
        pulse_starts = []  # パルスの開始地点のリスト
        pulse_ends = []    # パルスの終了地点のリスト
        in_pulse = False   # 現在パルスの中かどうかを示すフラグ

        for i in range(len(v)):
            if not in_pulse and v[i] > threshold:
                pulse_starts.append(i)  # パルスの開始インデックスを記録
                in_pulse = True
            elif in_pulse and v[i] <= threshold:
                pulse_ends.append(i)  # パルスの終了インデックスを記録
                in_pulse = False

        # パルスの開始・終了のペアをプロット
        for start, end in zip(pulse_starts, pulse_ends):
            if end - start >= min_pulse_duration:  # 持続時間のフィルタリング
                pulses.append({
                    'particle': particle_id,
                    'start_frame': frames[start],
                    'end_frame': frames[end],
                    'interval': (frames[end] - frames[start]) * time_interval
                })
    
                if display == True:
                    plt.plot(time_interval*frames[start:end+1], v[start:end+1], label=f'Particle {particle_id}')
              
    # プロットの設定
    if display == True:
        plt.xlim((0, time_interval* df['frame'].max()))
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.show()



    # 結果をデータフレームに変換
    pulses_df = pd.DataFrame(pulses)

    return pulses_df


def taac(x, tau):
    """
    Time-averaged autocorrelation for sequence x at lag tau.
    - If elements of x are scalar angles (radians), compute autocorrelation of unit direction vectors: cos(theta(t+tau)-theta(t)).
    - If elements of x are vectors/arrays, compute dot product.
    """
    T = len(x)
    if T == 0 or tau >= T:
        return float('nan')
    As = 0.0
    count = 0
    for t in range(T - tau):
        xt = x[t]
        xtau = x[t + tau]
        # skip nan entries
        try:
            if (np.isscalar(xt) and np.isnan(xt)) or (np.isscalar(xtau) and np.isnan(xtau)):
                continue
        except Exception:
            pass
        if np.isscalar(xt):
            # xt and xtau are angles in radians
            A = np.cos(xtau - xt)
        else:
            A = np.dot(xtau, xt)
        As += A
        count += 1
    return As / count if count > 0 else float('nan')

def itaac(df, max_lag, dimension_2=False):
    """
    Compute per-particle time-averaged auto-correlation.
    If dimension_2 is True, uses velocity vectors (v * unit vector from theta).
    If theta is scalar (radians), unit vector is [cos(theta), sin(theta)].
    """
    ac_list = []
    for particle_id, group in df.groupby('particle'):
        group = group.sort_values(by='frame').dropna(subset=['v', 'theta'])
        v = group['v'].values
        theta = group['theta'].values

        if dimension_2:
            vectors = []
            for vi, th in zip(v, theta):
                if np.isscalar(th):
                    vectors.append(np.array([np.cos(th), np.sin(th)]) * vi)
                else:
                    vectors.append(np.array(th) * vi)
            x = np.array(vectors)
        else:
            x = v

        for tau in range(max_lag):
            ac = taac(x, tau)
            ac_list.append({'particle': particle_id, 'lag time': tau, 'auto correlation': ac})

    ac_df = pd.DataFrame(ac_list)

    return ac_df

# ---------------------- Moving average helpers ----------------------
def rolling_mean_series(s, window, center=False, min_periods=1):
    """Simple linear rolling mean using pandas."""
    return s.rolling(window=window, center=center, min_periods=min_periods).mean()


def rolling_circular_mean(s, window, center=False, min_periods=1):
    """Rolling circular mean for angles in radians.
    Uses atan2(mean(sin), mean(cos)) on the window while ignoring NaNs.
    """
    def circ_mean(arr):
        arr = np.asarray(arr)
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            return np.nan
        sin_mean = np.mean(np.sin(arr))
        cos_mean = np.mean(np.cos(arr))
        return np.arctan2(sin_mean, cos_mean)

    return s.rolling(window=window, center=center, min_periods=min_periods).apply(lambda x: circ_mean(np.array(x)), raw=False)


def moving_average(df, column, window, method='auto', new_column=None, center=False, min_periods=1):
    """Add a moving average column to a dataframe per-column.

    Parameters
    - df: pandas DataFrame
    - column: column name to smooth
    - window: window size (int)
    - method: 'linear', 'circular', or 'auto'
      - 'circular' is recommended for angle columns (radians)
      - 'auto' selects 'circular' for columns named 'theta'/'dtheta'/'alpha'
    - new_column: name of the output column. If None, defaults to f"{column}_ma{window}"
    - center: passed to pandas rolling
    - min_periods: minimum periods for rolling

    Returns: the new Series (and also attaches it to df)
    """
    if new_column is None:
        new_column = f"{column}_ma{window}"
    s = df[column]

    if method == 'auto':
        if column in ['theta', 'dtheta', 'alpha']:
            method = 'circular'
        else:
            method = 'linear'

    if method == 'linear':
        df[new_column] = rolling_mean_series(s, window, center=center, min_periods=min_periods)
    elif method == 'circular':
        df[new_column] = rolling_circular_mean(s, window, center=center, min_periods=min_periods)
    else:
        raise ValueError("method must be 'linear' or 'circular' or 'auto'")

    return df[new_column]

# ---------------------- End moving average helpers ----------------------

def calc_eac(iac,interval = 1, display=True, xscale_log=True):
    eac=iac.groupby('lag time').mean()['auto correlation']
    N = len(iac[iac['lag time']==0].index)
    eac_err = iac.groupby('lag time').std()['auto correlation']/np.sqrt(N)
    
    if display == True:
        times = eac.index * interval
        fig, ax = plt.subplots(figsize=(6,6))
        ax.errorbar(times, eac, yerr= eac_err, fmt ='o')
        #ax.plot(times, 10**popt[1] * 10 ** (np.array(times)*popt[0]) , lw =10)

        ax.set_xlabel('lag time $\\Delta t$')
        ax.set_ylabel('$<v(t)\\cdot v(t+\\tau)>$')
        if xscale_log == True:
            ax.set_xscale('log')

    return eac, eac_err

def fit_eac(eac):
    def line(x, a, b):
        return a * x + b
    def fit_line(eac, interval=10):
        times = eac.index * interval
        popt, pcov = curve_fit(line, times, np.log10(eac))
        return popt, pcov
    popt, pcov = fit_line(eac)

    return popt, pcov