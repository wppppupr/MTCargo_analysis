import numpy as np
import zarr
import os

####################################################

FILE_PATH = "/Volumes/My Passport/Sasaki/MTsingleBeads/20260122/exp002"
####################################################


def getP(path):
    flow_path = os.path.join(path, "MTs_red_flow.zarr")
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

    polar_path = os.path.join(path, "MTs_red_polar.zarr")
    Polar_parameter = zarr.open(polar_path, mode = 'w', shape = P.shape, dtype = P.dtype)
    Polar_parameter[:] = PNan

    return PNan

if __name__ == "__main__":

    _ = getP(FILE_PATH)