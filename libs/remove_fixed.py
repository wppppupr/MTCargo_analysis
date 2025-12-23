import pandas as pd
import numpy as np

def remove_fixed(tracking_df, distance_threshold):
    """
    パーティクルごとに最初と最後のフレーム間の距離を計算し、閾値より短いパーティクルを元のデータフレームから除外する関数。

    Parameters:
    - tracking_df: トラッキング結果のデータフレーム（'x', 'y', 'particle', 'frame'を含む）
    - distance_threshold: 閾値（これより短い距離のパーティクルを除外）

    Returns:
    - filtered_df: 閾値よりも距離が短いパーティクルを除いたデータフレーム
    """
    # データフレームをパーティクルごとにグループ化して距離を計算
    grouped = tracking_df.groupby('particle')
    distances = grouped.apply(lambda df: np.sqrt(
        (df.iloc[-1]['x'] - df.iloc[0]['x'])**2 +
        (df.iloc[-1]['y'] - df.iloc[0]['y'])**2
    ))

    # 閾値よりも短い距離のパーティクルを取得
    short_distance_particles = distances[distances < distance_threshold].index

    # 元のデータフレームからこれらのパーティクルを除外
    filtered_df = tracking_df[~tracking_df['particle'].isin(short_distance_particles)]

    return filtered_df

def cal_dis(tracking_df):
    """
    トラジェクトリごとに、最初のフレームと最後のフレームの距離を計算する関数。

    Parameters:
    - tracking_df: トラッキング結果のデータフレーム（'x', 'y', 'particle', 'frame'を含む）

    Returns:
    - distances_df: パーティクルごとの距離を含むデータフレーム
    """
    # データフレームをパーティクルごとにグループ化
    grouped = tracking_df.groupby('particle')

    # 各パーティクルに対して、最初のフレームと最後のフレームの距離を計算
    distances = grouped.apply(lambda df: np.sqrt(
        (df.iloc[-1]['x'] - df.iloc[0]['x'])**2 +
        (df.iloc[-1]['y'] - df.iloc[0]['y'])**2
    ))

    # 距離を含むデータフレームを作成
    distances_df = distances.reset_index(name='distance')

    return distances_df

# 使用例
# start_end_distances = calculate_start_end_distance(your_tracking_result_df)
# print(start_end_distances)
