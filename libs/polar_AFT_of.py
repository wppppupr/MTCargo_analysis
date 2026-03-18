import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import zarr
import cv2
import sys
from pathlib import Path
from tqdm import tqdm
import AFT_tools_v2 as AFT

# pathlib によるパスの定義
base_path = Path('/Volumes/My Passport/Sasaki/MTsingleBeads/20260122/exp002')

green = zarr.open_array(str(base_path / "MTs.zarr"), read_only=True)

# AFT parameters
window_size_um = 10 # MTs length = 10um
frame = 100

scale = 0.11  # um/pixel

#### required parameters ####
window_size = int(window_size_um/scale)
overlap = 0.2
neighborhood_radius = 1

d = 30

print("Calculating AFT...")

x, y, u, v, im_theta, im_eccentricity = AFT.image_local_order(
    green[:,:,:], window_size, overlap, save_path='', eccentricity_thresh=0.2,
    plot_overlay=False, plot_angles=False, plot_eccentricity=False, save_figures=False
)

# --- 1. 定数とデータの準備 ---
num_frames = green.shape[0]
polar_order_list = []
interval = 4  # 時間間隔(秒)などはご自身の実験系に合わせてください

print("Calculating Polar Order...")

# numpy 配列のベクトル演算によるループ処理の高速化のため、グリッドを事前計算
target_x = np.unique(x).astype(int)  # 列の座標 (cpos)
target_y = np.unique(y).astype(int)  # 行の座標 (rpos)
target_X, target_Y = np.meshgrid(target_x, target_y)

for f in tqdm(range(1, num_frames)):
    # a. フレームの取得とオプティカルフロー計算
    prev_frame = cv2.normalize(green[f-1, :, :], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    curr_frame = cv2.normalize(green[f, :, :], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    flow = cv2.calcOpticalFlowFarneback(prev_frame, curr_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    # b. 現在のフレームの角度マップを取得
    curr_theta = im_theta[f]  # shape: (rows, cols)
    
    # c. im_theta から単位ベクトルを生成
    ui = np.cos(curr_theta)
    vi = np.sin(curr_theta)
    
    # d. 窓の中心に対応するオプティカルフローを取得
    move_x = flow[target_Y, target_X, 0]
    move_y = flow[target_Y, target_X, 1]
    
    # e. 内積による極性判定 (numpyを使用した一括計算)
    dot_product = ui * move_x + vi * move_y
    mask = dot_product < 0
    
    u_polar = np.where(mask, -ui, ui)
    v_polar = np.where(mask, -vi, vi)
    
    # f. ポーラー度の計算
    if np.any(~np.isnan(u_polar)):
        mean_u = np.nanmean(u_polar)
        mean_v = np.nanmean(v_polar)
        
        p_val = np.sqrt(mean_u**2 + mean_v**2)
        polar_order_list.append(p_val)
    else:
        polar_order_list.append(np.nan)

# --- 2. polar_order_list を zarr で保存 ---
polar_order_array = np.array(polar_order_list)
output_zarr_path = base_path / "polar_order.zarr"

# zarr 配列として上書きモードで保存
zarr.save(str(output_zarr_path), polar_order_array)
print(f"Saved polar order array to {output_zarr_path}")