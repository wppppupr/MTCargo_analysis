import numpy as np
import zarr
import os
import pandas as pd
import argparse
import cv2

from tqdm import tqdm

# =============================================================================
# 関数定義
# =============================================================================

def calculate_RDF(image, centers, max_r, mask_r=None):
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

    if mask_r is not None and mask_r > 0:
        mask_r_int = int(mask_r)
        mask_uint8 = np.ones(image.shape, dtype=np.uint8)
        for x_p, y_p in np.round(centers).astype(int):
            cv2.circle(mask_uint8, (int(x_p), int(y_p)), mask_r_int, 0, thickness=-1)
        valid_pixels = image[mask_uint8 == 1]
        if len(valid_pixels) > 0:
            mean_intensity = np.mean(valid_pixels)
        else:
            mean_intensity = np.mean(image)
    else:
        mean_intensity = np.mean(image)

    g_r = profiles / mean_intensity
    
    return g_r

def get_RDFs(image_seq, tracks, max_r, mask_r=None):
    g_r_list = []
    frames = np.arange(len(image_seq))
    for frame in tqdm(frames):
        image = image_seq[frame]
        track = tracks[tracks['frame'] == frame]
        x = track['x']
        y = track['y']
        pos = np.array([x, y]).T

        g_r = calculate_RDF(image, pos, max_r, mask_r)
        g_r_list.extend(g_r)

    return np.array(g_r_list)

def main():
    parser = argparse.ArgumentParser(description='Calculate RDF.')
    parser.add_argument('base_path', type=str, help='Path to the directory containing MTs.zarr and beads_tracks.csv')
    parser.add_argument('--max_rc', type=int, default=10, help='Maximum R_c to calculate (default: 10)')
    parser.add_argument('--scale', type=float, default=0.11, help='Scale um/px (default: 0.11)')
    parser.add_argument('--cargo_radius', type=float, default=0.59, help='Cargo radius um (default: 0.59)')
    parser.add_argument('--mask_radius', type=float, default=None, help='Radius (in px) around particles to mask out for bulk intensity. Defaults to cargo_radius/scale * 3.')
    args = parser.parse_args()

    target_path = args.base_path
    g_r_path = os.path.join(target_path, 'RDF.zarr')

    r_c = args.cargo_radius / args.scale
    mask_r = args.mask_radius if args.mask_radius is not None else r_c * 7

    print(f'calculate RDF for {target_path}')

    MTs_path = os.path.join(target_path, "GFP.zarr")
    if not os.path.exists(MTs_path):
        print(f"Error: {MTs_path} not found.")
        return

    track_path = os.path.join(target_path, "beads_tracks.csv")
    if not os.path.exists(track_path):
        print(f"Error: {track_path} not found.")
        return

    MTszarr = zarr.open_array(MTs_path, mode='r')
    tracks = pd.read_csv(track_path)

    g_r = get_RDFs(MTszarr[:], tracks, max_r=r_c*args.max_rc, mask_r=mask_r)
    g_r_zarr = zarr.open(g_r_path, mode='w', shape=g_r.shape, dtype=g_r.dtype)
    g_r_zarr[:] = g_r
    
    print('done')

if __name__ == "__main__":
    main()