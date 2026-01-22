import numpy as np
import zarr
import os
import pandas as pd
# =============================================================================
# 設定
# =============================================================================
TARGET_PATH = r"/Volumes/data/Sasaki/MTsingleBeads/20260122/exp"

# RDFの計算設定
MAX_RC = 50  # 粒子系の何倍まで計算するか

# =============================================================================
# 関数定義
# =============================================================================

def calculate_RDF(image, centers, max_r):
    """
    For文を一切使わず、かつ局所領域（ROI）のみを計算する最速・最適化版。
    (TypeError修正済み: max_rを強制的にintに変換します)
    """
    # 入力データの型を保証
    image = np.asarray(image)
    centers = np.asarray(centers)
    
    # 【修正箇所】np.padのためにmax_rを必ず整数型に変換
    max_r = int(max_r)
    
    n_centers = centers.shape[0]
    
    # 1. 画像のパディング
    # max_r が int になったのでエラーは起きません
    padded_image = np.pad(image, pad_width=max_r, mode='constant', constant_values=0)
    
    # 中心座標を整数化し、パディング分ずらす
    # ここも念のため round してから int にキャスト
    centers_int = np.round(centers).astype(int) + max_r
    
    # 2. 切り出し用インデックスの一括生成
    d_range = np.arange(-max_r, max_r + 1)
    
    # grid_y, grid_x: (window_size, window_size)
    grid_y, grid_x = np.meshgrid(d_range, d_range, indexing='ij')
    
    # 全中心点に対する絶対座標インデックス (Fancy Indexing用)
    absolute_y = centers_int[:, 1, None, None] + grid_y[None, :, :]
    absolute_x = centers_int[:, 0, None, None] + grid_x[None, :, :]
    
    # 3. 領域の一括抽出 (Extract Patches)
    patches = padded_image[absolute_y, absolute_x]
    
    # 4. 距離マスクの作成とフラット化
    local_dist = np.sqrt(grid_x**2 + grid_y**2)
    local_dist_int = local_dist.astype(np.int32)
    
    mask_valid = local_dist_int <= max_r
    
    # フラット化
    patches_flat = patches.reshape(n_centers, -1)
    dists_flat = local_dist_int.ravel()
    mask_flat = mask_valid.ravel()
    
    # 効率化のためタイル化
    tiled_dists = np.tile(dists_flat, (n_centers, 1))
    
    # マスク外の距離を「ゴミ箱ビン（max_r + 1）」に飛ばす
    tiled_dists[:, ~mask_flat] = max_r + 1
    
    # 5. リニアインデックス法による一括集計
    row_stride = max_r + 2
    offsets = np.arange(n_centers)[:, None] * row_stride
    
    global_bins = tiled_dists + offsets
    
    # 1次元に展開して集計
    global_bins_flat = global_bins.ravel()
    weights_flat = patches_flat.ravel()
    
    total_bins = n_centers * row_stride
    sums = np.bincount(global_bins_flat, weights=weights_flat, minlength=total_bins)
    counts = np.bincount(global_bins_flat, minlength=total_bins)
    
    # 6. 整形と後処理
    sums = sums.reshape(n_centers, row_stride)
    counts = counts.reshape(n_centers, row_stride)
    
    # ゴミ箱と余分な列を捨てる
    sums = sums[:, :max_r+1]
    counts = counts[:, :max_r+1]
    
    # 平均計算 (0除算回避)
    with np.errstate(invalid='ignore'):
        profiles = sums / counts
        
    profiles[counts == 0] = 0

    global_mean_intensity = np.mean(image)
    g_r = profiles/global_mean_intensity
    
    return g_r

def get_RDFs(image_seq, tracks, max_r):
    g_r_list = []
    frames = np.arange(len(image_seq))
    for frame in frames:
        image = image_seq[frame]
        track = tracks[tracks['frame'] == frame]
        x = track['x']
        y = track['y']
        pos = np.array([x, y]).T

        g_r = calculate_RDF(image, pos, max_r)
        g_r_list.extend(g_r)

    return np.array(g_r_list)

if __name__ == "__main__":

    g_r_path = os.path.join(TARGET_PATH, 'RDF.zarr')

    scale = 0.11 # um/px
    cargo_radius = 0.59 # um

    r_c = cargo_radius/scale

    MTs_path = os.path.join(TARGET_PATH, "MTs.zarr")
    MTszarr = zarr.open_array(MTs_path, mode = 'r')

    track_path = os.path.join(TARGET_PATH, "beads_tracks.csv")
    tracks = pd.read_csv(track_path)

    print('start calculate')
    g_r = get_RDFs(MTszarr[:], tracks, max_r=r_c*MAX_RC)

    print(f"save in {g_r_path}")
    g_r_zarr = zarr.open(g_r_path, mode='w', shape=g_r.shape, dtype=g_r.dtype)
    g_r_zarr[:] = g_r