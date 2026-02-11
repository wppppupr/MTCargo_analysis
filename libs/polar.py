import os
import numpy as np
import zarr
import dask.array as da  # dask.arrayをインポート

####################################################

FILE_PATH = '/Volumes/My Passport/Sasaki/MTsingleBeads/20260122/exp002'
####################################################


def getP(path):
    flow_path = os.path.join(path, "MTs_red_flow_LK.zarr")
    flow_array = zarr.open_array(flow_path, mode='r')
    
    x = flow_array[:, :, :, 0]  # x成分
    y = flow_array[:, :, :, 1]  # y成分
    # ベクトルの大きさを計算
    magnitude = np.sqrt(x**2 + y**2)

    # --- 1. Polar Order Parameter (P) の計算 ---
    # 全ベクトルの平均を計算
    mean_x = np.mean(x, axis=(1,2))
    mean_y = np.mean(y, axis=(1,2))
        
    # 平均ベクトルの大きさを、全ベクトルの平均の大きさで割って正規化
    # P = |<v>| / <|v|>
    mean_magnitude_of_vectors = np.mean(magnitude, axis=(1,2))
    magnitude_of_mean_vector = np.sqrt(mean_x**2 + mean_y**2)

    P = magnitude_of_mean_vector / mean_magnitude_of_vectors
    n_nanind = np.where(~np.isnan(P))
    PNan = P[n_nanind[0]]

    polar_path = os.path.join(path, "MTs_red_polar_LK.zarr")
    Polar_parameter = zarr.open(polar_path, mode = 'w', shape = P.shape, dtype = P.dtype)
    Polar_parameter[:] = PNan

    return PNan

def getP_dask(path):
    flow_path = os.path.join(path, "green_flow.zarr")
    polar_path = os.path.join(path, "green_globalP.zarr")
    
    # 1. zarrをdask arrayとして開く（ここではデータはまだ読み込まれません）
    # chunks='auto' で最適なサイズに自動調整
    flow_array = da.from_zarr(flow_path)
    
    # --- 計算ロジック (Numpyと同じ記述でOK) ---
    # これらは「計算グラフ」を作るだけで、メモリは消費しません
    x = flow_array[:, :, :, 0]
    y = flow_array[:, :, :, 1]
    
    magnitude = da.sqrt(x**2 + y**2)
    
    # 平均計算
    mean_x = da.mean(x, axis=(1, 2))
    mean_y = da.mean(y, axis=(1, 2))
    
    mean_magnitude_of_vectors = da.mean(magnitude, axis=(1, 2))
    magnitude_of_mean_vector = da.sqrt(mean_x**2 + mean_y**2)
    
    # Pの計算（まだ実行されません）
    P = magnitude_of_mean_vector / mean_magnitude_of_vectors

    # --- 保存と実行 ---
    # 2. ここで初めて計算が走り、少しずつディスクに書き込まれます
    # overwrite=Trueにしておくと既存ファイルを上書きします
    print("Computing and saving to Zarr...")
    P.to_zarr(polar_path, overwrite=True)
    
    # 3. 必要であれば計算後の値をNumpy配列としてメモリに戻して返す
    # 注意: ここで全データをメモリに入れることになるので、返すデータが巨大なら注意
    # 今回はPは1次元(フレーム数)だけなので、メモリに乗るはずです
    P_computed = da.from_zarr(polar_path).compute()
    
    # NaNを除去する場合
    # ※時系列データの場合、NaNを除去するとフレーム番号がズレるので注意してください
    P_clean = P_computed[~np.isnan(P_computed)]
    
    return P_clean

if __name__ == "__main__":

    _ = getP_dask(FILE_PATH)