import argparse
import numpy as np
import pandas as pd
import h5py
import xarray as xr
import os
import shutil
from pathlib import Path
from tqdm import tqdm
import cv2
from concurrent.futures import ThreadPoolExecutor

def main():
    parser = argparse.ArgumentParser(description='Calculate local polar order for an arbitrary ROI.')
    parser.add_argument('base_path', type=str, help='Path to the directory containing hdf5 and csv')
    parser.add_argument('--h5_file', type=str, default='GFP_flows.h5', help='H5 file name')
    parser.add_argument('--roi_x', type=int, default=None, help='X coordinate of the ROI center (optional)')
    parser.add_argument('--roi_y', type=int, default=None, help='Y coordinate of the ROI center (optional)')
    parser.add_argument('--windows', type=str, nargs='+', default=['10:200:10', '200:2000:100'], 
                        help='Window sizes. Accepts space/comma separated numbers, or start:stop:step (e.g. 10 50 100, or 10:200:10)')
    args = parser.parse_args()

    base_path = Path(args.base_path)
    h5_path = base_path / args.h5_file

    if not h5_path.exists():
        print(f"Error: HDF5 file not found at {h5_path}")
        return

    # Parse window sizes
    sizes = set()
    for arg_w in args.windows:
        for p in arg_w.split(','):
            if not p.strip(): continue
            if ':' in p:
                parts = p.split(':')
                w_start = int(parts[0])
                w_stop = int(parts[1]) if len(parts) > 1 else w_start
                w_step = int(parts[2]) if len(parts) > 2 else 1
                sizes.update(range(w_start, w_stop + 1, w_step))
            else:
                sizes.add(int(p))
    local_sizes = sorted(list(sizes))

    # Read particle tracks if they exist for overlap checking
    df_tracks = None

    with h5py.File(str(h5_path), 'r') as f:
        dataset_key = list(f.keys())[0]
        flow_data = f[dataset_key]
        shape = flow_data.shape
        print(f"Flow data shape: {shape}")
        
        num_frames = shape[0]
        if shape[-1] == 2:
            rows, cols = shape[1], shape[2]
            channel_first = False
        else:
            rows, cols = shape[2], shape[3]
            channel_first = True

        max_size = max(local_sizes)
        half_w = max_size // 2

        polar_order_array = np.full((len(local_sizes), num_frames), np.nan, dtype=np.float32)

        def process_frame(t):
            if channel_first:
                m_x = flow_data[t, 0, ...].astype(np.float32)
                m_y = flow_data[t, 1, ...].astype(np.float32)
            else:
                m_x = flow_data[t, ..., 0].astype(np.float32)
                m_y = flow_data[t, ..., 1].astype(np.float32)
            
            v_mag = np.hypot(m_x, m_y)
            with np.errstate(divide='ignore', invalid='ignore'):
                m_ux = m_x / v_mag
                m_uy = m_y / v_mag
                # Fill NaNs with 0 for vector addition
                m_ux[~np.isfinite(m_ux)] = 0
                m_uy[~np.isfinite(m_uy)] = 0

            p_vals = np.empty((len(local_sizes),), dtype=np.float32)
            
            for w_idx, size in enumerate(local_sizes):
                # 1. 局所ポーラー度フィールド全体を計算
                kernel = np.zeros((size, size), dtype=np.float32)
                center = size / 2.0 - 0.5
                ky, kx = np.ogrid[:size, :size]
                kmask = (kx - center)**2 + (ky - center)**2 <= (size / 2.0)**2
                kernel[kmask] = 1.0
                kernel /= kernel.sum()

                u_avg = cv2.filter2D(m_ux, -1, kernel, borderType=cv2.BORDER_REFLECT)
                v_avg = cv2.filter2D(m_uy, -1, kernel, borderType=cv2.BORDER_REFLECT)
                p_field = np.hypot(u_avg, v_avg, dtype=np.float32)

                p_vals[w_idx] = np.mean(p_field)

            return t, p_vals

        print(f"Calculating with window sizes: {local_sizes}...")
        workers = min(32, (os.cpu_count() or 1) + 4)
        active_frames = list(range(num_frames))
        
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for t, p_vals in tqdm(executor.map(process_frame, active_frames), total=num_frames):
                polar_order_array[:, t] = p_vals

    print("\nConsolidating data into xarray...")
    
    ds_roi = xr.Dataset(
        data_vars={
            'polar_order': xr.DataArray(
                polar_order_array,
                dims=['window size', 'frame'],
                coords={'window size': local_sizes, 'frame': np.arange(num_frames)}
            )
        }
    )
    
    # Add metadata
    ds_roi.attrs['description'] = f'Local polar order analysis for arbitrary ROI'
    ds_roi.attrs['window sizes'] = local_sizes

    out_roi = base_path / f"local_polar_noCargo.zarr"

    if out_roi.exists():
        shutil.rmtree(out_roi, ignore_errors=True)
    
    ds_roi.to_zarr(str(out_roi), mode='w', consolidated=False)
    
    print(f"Success! Data saved to {out_roi}")

if __name__ == "__main__":
    main()
