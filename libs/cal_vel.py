import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def cal(tracking_df, scale=1, frame_interval=1):
    """
    データフレーム内の各particleごとに速さ（スカラー値）と単位ベクトルを計算し、
    新しいカラムとして追加する関数。

    Parameters:
    - tracking_df: トラッキング結果のデータフレーム（'x', 'y', 'particle', 'frame'を含む）
    - frame_interval: フレーム間の時間間隔（デフォルトは1）

    Returns:
    - tracking_df: 速さと単位ベクトルが追加されたデータフレーム
    """
    # 各パーティクルごとに速度と単位ベクトルを計算
    def calculate_v_and_theta(df, frame_interval=frame_interval):
        # 前のフレームとの座標差を計算
        df['x_diff'] = df['x'] - df['x'].shift(1)
        df['y_diff'] = df['y'] - df['y'].shift(1)
        
        # 座標差から距離（速さ）を計算
        df['distance'] = np.sqrt(df['x_diff']**2 + df['y_diff']**2)
        
        # 速さを計算
        df['v'] = scale * df['distance'] / frame_interval
        
        # 距離が0でない場合のみ単位ベクトルを計算
        df['theta'] = df.apply(
            lambda row: np.array([row['x_diff'] / row['distance'], row['y_diff'] / row['distance']])
            if row['distance'] != 0 else np.array([np.nan, np.nan]), axis=1
        )

        df['t'] = df['frame'] * frame_interval
        
        return df[['v', 'theta', 't']]
    
    # パーティクルごとに速度と単位ベクトルを一度に計算してデータフレームに追加
    tracking_df[['v', 'theta', 't']] = tracking_df.groupby('particle').apply(
        calculate_v_and_theta
    ).reset_index(level=0, drop=True)

    return tracking_df


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
    As = 0
    T = len(x)
    if T == 0:
        return float('nan')
    for t in range(T-tau):
        A = np.dot(x[t+tau], x[t])
        As += A

    return As/T

def itaac(df, max_lag, dimension_2 = bool):
    ac_list = []
    for tau in range(max_lag):
        for particle_id, group in df.groupby('particle'):
            group = group.sort_values(by='frame')  # フレーム順に並べ替え
            v = group['v']
            v = np.array(v[1:len(v)].tolist()).reshape(-1,1)
            if dimension_2 == True:
                theta = group['theta']
                theta = np.array(theta[1:len(theta)].tolist())
                theta = group['theta']
                theta = np.array(theta[1:len(theta)].tolist())
                v = v * theta
            ac = taac(theta, tau)
            ac_list.append({
                'particle':particle_id,
                'lag time':tau,
                'auto correlation':ac
            })

    ac_df = pd.DataFrame(ac_list)

    return ac_df

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