import os


folder = '/Volumes/My Passport/Sasaki/MTsingleBeads'

track_path = 'MTtrack.csv'

path0 = os.path.join(folder, "20260121/beads_trans_crop_crop", track_path)
path1 = os.path.join(folder, "20260121/exp_crop1", track_path)
path2 = os.path.join(folder, "20260122/exp", track_path)
path3 = os.path.join(folder, "20260122/exp001", track_path)
path4 = os.path.join(folder, "20260122/exp002", track_path)

#####

paths = [path0, path1, path2, path3, path4]
labels = ['exp_0', 'exp_1', 'exp_2', 'exp_3', 'exp_4'] # 任意のラベル
threshold = 10 # 閾値を設定（例：粒子が10個未満のフレームは信頼しない）


#####

import numpy as np
import pandas as pd

def vec(df):
    # 1. 前処理: frameでソートしておく（groupby内での処理を減らすため）
    df = df.sort_values(['particle', 'frame'])

    # 2. 差分計算（各グループの最初の一行はNaNになる）
    # diff() は pandas のメソッドを使うとインデックスが維持されるので安全です
    dx = df.groupby('particle')['x'].diff()
    dy = df.groupby('particle')['y'].diff()

    # 3. 速度ベクトルと単位ベクトルの計算
    v = np.sqrt(dx**2 + dy**2)

    df['theta'] = dx/v + 1j * dy/v

    return df

def calculate_polar_order(df):
    """
    各フレームごとのポーラー度（Order Parameter）を算出する
    """
    # 1. 各フレームごとに複素ベクトルの平均を取る
    # thetaには既に単位ベクトル（dx/v + i*dy/v）が入っている前提
    group_avg = df.groupby('frame')['theta'].mean()
    
    # 2. 平均ベクトルの絶対値（長さ）を計算する
    # これが 1 に近いほど揃っており、0 に近いほどバラバラ
    polar_order = group_avg.abs()

    # 各フレームごとの粒子の数をカウント
    # 'theta' が NaN でないもの（速度が計算できているもの）だけを数えるのが実用的です
    counts = df.dropna(subset=['theta']).groupby('frame')['particle'].count()
    
    return polar_order, counts

polars_dict = {}
count_dict = {}

for path, label in zip(paths, labels):
    print(path)
    df_temp = pd.read_csv(path)
    df_temp = vec(df_temp)
    # calculate_polar_order の結果を辞書に格納
    polar_order, counts = calculate_polar_order(df_temp)
    polars_dict[label] = polar_order
    count_dict[label] = counts


# 辞書から一気にDataFrameを作成（frameがインデックスになります）
df_polar_all = pd.DataFrame(polars_dict)
df_count_all = pd.DataFrame(count_dict)

# df_polar_all と同じ形状の True/False マスクを作成
mask = df_count_all >= threshold

# マスクを適用：条件を満たさない（False）場所を NaN にする
df_polar_masked = df_polar_all.where(mask)

df_polar_all.to_csv("analysis_data/global_polar.csv", index=False)
df_polar_masked.to_csv("analysis_data/global_polar_masked.csv", index=False)
df_count_all.to_csv("analysis_data/global_count.csv", index=False)

print("done")