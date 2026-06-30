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
    parser.add_argument('--tracks_csv', type=str, default='beads_tracks.csv', help='Trackpy csv file name')
    parser.add_argument('--sigmas', type=str, nargs='+', default=['5:100:5', '100:1000:50'], 
                        help='Gaussian sigmas. Accepts space/comma separated numbers, or start:stop:step (e.g. 10 50 100, or 10:200:10)')
    args = parser.parse_args()

    base_path = Path(args.base_path)
    csv_path = base_path / args.tracks_csv
    h5_path = base_path / args.h5_file

    if not h5_path.exists():
        print(f"Error: HDF5 file not found at {h5_path}")
        return

    # Parse window sizes
    sizes = set()
    for arg_w in args.sigmas:
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
    if csv_path.exists():
        print(f"Loading tracks from {csv_path} for overlap checking...")
        df_tracks = pd.read_csv(csv_path)
    else:
        print(f"Warning: Track file not found at {csv_path}. Particle overlap checking will be skipped.")

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
        half_w = int(np.ceil(3 * max_size))

        if args.roi_x is None or args.roi_y is None:
            if df_tracks is None:
                raise ValueError("Error: Track CSV is required to automatically find an empty ROI.")
            
            print("Automatically searching for the safest particle-free ROI...")
            # Compute a global 2D mask of all particle positions over time
            y_all = np.clip(np.round(df_tracks['y'].values).astype(int), 0, rows - 1)
            x_all = np.clip(np.round(df_tracks['x'].values).astype(int), 0, cols - 1)
            particle_mask = np.zeros((rows, cols), dtype=bool)
            particle_mask[y_all, x_all] = True
            
            from scipy.ndimage import distance_transform_edt
            dist = distance_transform_edt(~particle_mask)
            
            max_dist = dist.max()
            if max_dist <= half_w:
                raise ValueError(f"Could not find any empty space large enough for the max gaussian extent (3*sigma) (Radius required: {half_w}, Max available: {max_dist:.1f}).")
            
            y_idx, x_idx = np.unravel_index(dist.argmax(), dist.shape)
            print(f"Automatically selected ROI center at ({x_idx}, {y_idx}) with a safe radius of {max_dist:.1f} px to the nearest particle.")
        else:
            y_idx = np.clip(args.roi_y, 0, rows - 1)
            x_idx = np.clip(args.roi_x, 0, cols - 1)

            # Check for particle overlap for manual ROI
            if df_tracks is not None:
                in_roi = df_tracks[
                    (df_tracks['x'] >= x_idx - half_w) & (df_tracks['x'] <= x_idx + half_w) &
                    (df_tracks['y'] >= y_idx - half_w) & (df_tracks['y'] <= y_idx + half_w)
                ]
                if not in_roi.empty:
                    frames_with_particles = in_roi['frame'].unique()
                    raise ValueError(
                        f"Error: Particle(s) detected inside the manual ROI at ({x_idx}, {y_idx}) "
                        f"with the maximum window size ({max_size}).\n"
                        f"Frames with overlap: {frames_with_particles}"
                    )
                else:
                    print("No particles overlap with the specified ROI. Proceeding...")

        polar_order_array = np.full((len(local_sizes), num_frames), np.nan, dtype=np.float32)

        # --- 事前に距離マップを計算しておく（mainのループ外で1回実行） ---
        y_all = np.clip(np.round(df_tracks['y'].values).astype(int), 0, rows - 1)
        x_all = np.clip(np.round(df_tracks['x'].values).astype(int), 0, cols - 1)
        particle_mask = np.zeros((rows, cols), dtype=bool)
        particle_mask[y_all, x_all] = True
        from scipy.ndimage import distance_transform_edt
        dist_map = distance_transform_edt(~particle_mask)

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
                sigma = float(size)
                kernel_size = int(np.ceil(6 * sigma))
                if kernel_size % 2 == 0: kernel_size += 1
                k1d = cv2.getGaussianKernel(kernel_size, sigma)
                kernel = (k1d @ k1d.T).astype(np.float32)

                u_avg = cv2.filter2D(m_ux, -1, kernel, borderType=cv2.BORDER_REFLECT)
                v_avg = cv2.filter2D(m_uy, -1, kernel, borderType=cv2.BORDER_REFLECT)
                p_field = np.hypot(u_avg, v_avg, dtype=np.float32)

                # 2. 「半径 size/2 以内に粒子がいない」安全なピクセルをすべて特定
                safe_mask = dist_map > (int(np.ceil(3 * size)) + 5) # 5pxのマージン
                
                # 3. 安全な領域の平均値をとる（これがアンサンブル平均）
                if np.any(safe_mask):
                    p_vals[w_idx] = np.mean(p_field[safe_mask])
                else:
                    p_vals[w_idx] = np.nan

            return t, p_vals

        print(f"Calculating for ROI ({x_idx}, {y_idx}) with window sizes: {local_sizes}...")
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
                dims=['sigma', 'frame'],
                coords={'sigma': local_sizes, 'frame': np.arange(num_frames)}
            )
        }
    )
    
    # Add metadata
    ds_roi.attrs['description'] = f'Local polar order analysis for arbitrary ROI at ({x_idx}, {y_idx})'
    ds_roi.attrs['roi_x'] = int(x_idx)
    ds_roi.attrs['roi_y'] = int(y_idx)
    ds_roi.attrs['sigmas'] = local_sizes

    out_roi = base_path / f"local_polar_bg.zarr"

    if out_roi.exists():
        shutil.rmtree(out_roi, ignore_errors=True)
    
    ds_roi.to_zarr(str(out_roi), mode='w', consolidated=False)
    
    print(f"Success! Data saved to {out_roi}")

if __name__ == "__main__":
    main()
