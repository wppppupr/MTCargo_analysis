import nd2
import cv2
import zarr
import matplotlib.pyplot as plt
import numpy as np
import dask.array as da
from pathlib import Path
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

##########################################################################################
# 移動量 18px に最適化したパラメータ
params = dict(
    pyr_scale = 0.5, 
    levels = 7,          # 階層を増やす（18pxなら5層以上必要）
    winsize = 21,        # 移動量よりも大きいサイズを確保。2倍程度が安定
    iterations = 10, 
    poly_n = 7,          # 大きな動きにはより滑らかな近似（7）が有効
    poly_sigma = 1
)

p_dir = Path('/Volumes/My passport/Sasaki/MTsingleBeads/20260122/exp')
p_file = p_dir / 'exp.nd2'

##########################################################################################
scale = 0.11
interval = 4
sigma = (0.0, 1.0, 1.0)

print(p_file)

file = nd2.imread(p_file, dask = True)

images = file[0:272, 1, :, :]

images_blur = gaussian_filter(images, sigma)

def comp8bit(img_u16):
    # 1. 最小値と最大値を計算（ここが一番コストが高い）
    p_min, p_max = img_u16.min(), img_u16.max()
    
    # 2. 0-255に収めるための係数を計算
    diff = p_max - p_min
    if diff == 0:
        return np.zeros_like(img_u16, dtype=np.uint8)
    
    scale = 255.0 / diff
    
    # 3. 演算を一つにまとめて、最後にキャスト
    # (x - min) * scale を一気に行う
    return ((img_u16 - p_min) * scale).astype(np.uint8)

images_8bit = comp8bit(images_blur)

def compute_flow(images, params):
    t_len = images.shape[0]
    h, w = images.shape[1], images.shape[2]
    
    if t_len <= 1:
        return np.empty((0, h, w, 2), dtype=np.float32)
    
    flows = np.empty((t_len - 1, h, w, 2), dtype=np.float32)
    current_flow = np.zeros((h, w, 2), np.float32)

    # images が Dask 配列の場合に備え、ループ内で実体化させる
    for i in tqdm(range(t_len - 1), desc="Computing Optical Flow"):
        # 【修正ポイント】np.asarray() で NumPy 配列に変換（実体化）
        prev_img = np.asarray(images[i])
        next_img = np.asarray(images[i+1])
        
        # データ型を確実に uint8 にする（念のため）
        if prev_img.dtype != np.uint8:
            prev_img = prev_img.astype(np.uint8)
        if next_img.dtype != np.uint8:
            next_img = next_img.astype(np.uint8)

        if i == 0:
            flow_flags = 0 
        else:
            flow_flags = cv2.OPTFLOW_USE_INITIAL_FLOW
            
        current_flow = cv2.calcOpticalFlowFarneback(
            prev_img, 
            next_img, 
            current_flow, 
            **params, 
            flags=flow_flags
        )
        
        flows[i] = current_flow
        
    return flows

flows = compute_flow(images_8bit, params)

flow_p = p_dir / 'green_flow.zarr'
flows_zarr = zarr.open(flow_p, mode = 'w', dtype = flows.dtype, shape = flows.shape)
flows_zarr[:] = flows

print('done')