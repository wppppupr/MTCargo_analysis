import argparse
import numpy as np
import h5py
import zarr
import cv2
import xarray as xr
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from scipy.ndimage import uniform_filter
from concurrent.futures import ThreadPoolExecutor
import os

def main():
    parser = argparse.ArgumentParser(description='Calculate local polar order over multiple window sizes.')
    parser.add_argument('base_path', type=str, help='Path to the base directory containing hdf5 files')
    parser.add_argument('--start', type=int, default=3, help='Start window size (default: 3)')
    parser.add_argument('--stop', type=int, default=15, help='Stop window size (inclusive, default: 15)')
    parser.add_argument('--step', type=int, default=2, help='Step size for window sizes (default: 2)')
    args = parser.parse_args()

    base_path = Path(args.base_path)
    print(f'Loading data from {base_path}...')

    try:
        with h5py.File(str(base_path / "GFP_flows.h5"), 'r') as f:
            dataset_key = list(f.keys())[0]
            print("Loading data into memory for vectorized calculation...")
            # データ容量を抑えるための float16 をそのまま保持し、計算時にフレームごとに float32 に変換します
            flow_data = f[dataset_key][:]
    except Exception as e:
        print(f"Error loading hdf5 files: {e}")
        return

    num_frames = flow_data.shape[0]
    rows, cols = flow_data.shape[1], flow_data.shape[2]
    
    # 範囲指定で探索するウィンドウサイズのリストを生成
    local_sizes = list(range(args.start, args.stop + 1, args.step))
    
    # 局所的なポーラー度を保存するための配列 (ウィンドウサイズごとに用意)
    local_sizes = list(range(args.start, args.stop + 1, args.step))
    
    print(f"flow shape: {flow_data.shape}")
    print(f"Calculating local polar order for window sizes: {local_sizes}...")

    move_x = flow_data[:, :, :, 0]
    move_y = flow_data[:, :, :, 1]
    
    print("Calculating spatial local average and streaming to zarr...")
    output_zarr_path = base_path / "polar_order_sweep.zarr"
    
    import shutil
    if output_zarr_path.exists():
        shutil.rmtree(output_zarr_path)

    workers = min(32, (os.cpu_count() or 1) + 4)
    
    for i, size in enumerate(tqdm(local_sizes, desc="Window Sizes")):
        local_p_val = np.empty((num_frames, rows, cols), dtype=np.float32)
        
        def process_frame(t):
            # 空間方向(Y, X)のみのフィルタなのでフレームごとに処理。中間生成配列のサイズを削減可能。
            # float16で格納されているデータを計算精度確保のため float32 にキャストして処理
            u_t = uniform_filter(move_x[t].astype(np.float32), size=size)
            v_t = uniform_filter(move_y[t].astype(np.float32), size=size)
            # np.hypot で sqrt(u^2 + v^2) を1パスで計算し、高速化・省メモリ化
            return t, np.hypot(u_t, v_t, dtype=np.float32)
            
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for t, p_val_t in executor.map(process_frame, range(num_frames)):
                local_p_val[t] = p_val_t
                
        # ウィンドウサイズ1つ分のDataArrayを作成
        da = xr.DataArray(
            local_p_val[np.newaxis, ...],  # shape: (1, num_frames, rows, cols)
            dims=['window_size', 'frame', 'y', 'x'],
            coords={
                'window_size': [size],
                'frame': np.arange(num_frames),
                'y': np.arange(rows),
                'x': np.arange(cols)
            },
            name='local_polar_order'
        )
        da.attrs['window_size_unit'] = 'pixels'
        da.attrs['description'] = 'Local polar order calculated via uniform filter'
        # スタックせずにそのまま Zarr へ size ごとに追記(または新規作成)
        if i == 0:
            da.to_zarr(str(output_zarr_path), mode='w')
        else:
            da.to_zarr(str(output_zarr_path), append_dim='window_size')
            
    print(f"\nSaved local polar order with axis information to {output_zarr_path}")

if __name__ == "__main__":
    main()