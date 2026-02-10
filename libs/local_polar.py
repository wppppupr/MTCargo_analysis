import numpy as np
import pandas as pd
import zarr
import os
from tqdm import tqdm

####################################################

FILE_PATH = '/Volumes/My Passport/Sasaki/MTsingleBeads/20260122/exp002'

scale = 0.11

rectangle = 5

height = rectangle/scale
weight = rectangle/scale

####################################################

def local_polar(flow_array, tracks, height, weight):
    # --- パラメータ設定 ---
    h2 = int(height / 2)
    w2 = int(weight / 2)

    # トラックデータをNumpy配列化
    xc_all = tracks['x'].to_numpy().astype(np.int16)
    yc_all = tracks['y'].to_numpy().astype(np.int16)
    frames_all = tracks['frame'].to_numpy().astype(np.int16)

    # 結果を格納する配列（最初はNaNで埋めておく）
    Ps = np.full(len(tracks), np.nan)

    # グリッドの作成（相対座標）
    dy = np.arange(-h2, h2)
    dx = np.arange(-w2, w2)
    grid_y, grid_x = np.meshgrid(dy, dx, indexing='ij')

    # 画像サイズ取得（flow_arrayがリストの場合は最初の要素から形状を取得）
    if isinstance(flow_array, list):
        max_h, max_w = flow_array[0].shape[0], flow_array[0].shape[1]
    else:
        max_h, max_w = flow_array.shape[1], flow_array.shape[2]

    # --- フレームごとの高速ループ ---
    # ユニークなフレーム番号を取得してループ
    unique_frames = np.unique(frames_all)

    print(f"Processing {len(unique_frames)} frames...")

    for t in tqdm(unique_frames[1:]):
        # 1. このフレームにある点のインデックスを全て取得（ブールマスク）
        mask = (frames_all == t)
        
        # 該当する点の座標を抽出
        xs_t = xc_all[mask]
        ys_t = yc_all[mask]
        
        # 該当する点がなければスキップ
        if len(xs_t) == 0:
            continue

        # 2. 現在のフレーム画像を取得（ここがリストやh5pyでも動く理由）
        # flow_array全体を配列アクセスせず、t番目だけを取り出すのでエラーにならない
        current_flow = flow_array[t-1] 
        
        # 3. 座標計算（ブロードキャスト）
        # (N_points, window_h, window_w) のインデックスを作成
        idx_y = ys_t[:, None, None] + grid_y[None, :, :]
        idx_x = xs_t[:, None, None] + grid_x[None, :, :]
        
        # 境界クリップ
        idx_y = np.clip(idx_y, 0, max_h - 1)
        idx_x = np.clip(idx_x, 0, max_w - 1)
        
        # 4. ファンシーインデックスで一括抽出
        # current_flow は2次元(H,W,2)なので、単純な配列アクセスが可能
        batch_flows = current_flow[idx_y, idx_x] # shape: (N_points, h, w, 2)
        
        # 5. 行列計算（ここからは前回と同じ）
        batch_xf = batch_flows[..., 0]
        batch_yf = batch_flows[..., 1]
        
        magnitude = np.sqrt(batch_xf**2 + batch_yf**2)
        
        # 平均計算
        mean_xf = np.mean(batch_xf, axis=(1, 2))
        mean_yf = np.mean(batch_yf, axis=(1, 2))
        
        mean_mag_vectors = np.mean(magnitude, axis=(1, 2))
        mag_mean_vector = np.sqrt(mean_xf**2 + mean_yf**2)
        
        # 計算結果を代入（ゼロ除算対策付き）
        with np.errstate(divide='ignore', invalid='ignore'):
            P_vals = mag_mean_vector / mean_mag_vectors
        
        # NaNを0にするならここで（必要なければコメントアウト）
        #P_vals = np.nan_to_num(P_vals, nan=0.0)
        
        # 結果を元の配列の正しい位置に戻す
        Ps[mask] = P_vals

    print("Calculation complete.")
    return Ps

if __name__ == "__main__":

    print(f"Loading data from {FILE_PATH}...")

    flow_path = os.path.join(FILE_PATH, "green_flow.zarr")
    flow_array = zarr.open_array(flow_path, mode='r')

    tracks_path = os.path.join(FILE_PATH, "beads_tracks.csv")
    tracks = pd.read_csv(tracks_path)

    Ps = local_polar(flow_array, tracks, height, weight)

    tracks['local_P'] = Ps
    output_path = os.path.join(FILE_PATH, f"beads_tracks_with_local_P_rec{rectangle}_new.csv")
    tracks.to_csv(output_path, index=False)
    print(f"Saved tracks with local P to {output_path}")