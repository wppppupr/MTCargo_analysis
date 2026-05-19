import argparse
import numpy as np
import pandas as pd
import h5py
import xarray as xr
import os
import shutil
from pathlib import Path
from tqdm import tqdm
from scipy.ndimage import uniform_filter, binary_dilation
from concurrent.futures import ThreadPoolExecutor

def main():
    parser = argparse.ArgumentParser(description='Calculate local polar order for particles and background.')
    parser.add_argument('base_path', type=str, help='Path to the directory containing hdf5 and csv')
    parser.add_argument('--tracks_csv', type=str, default='beads_tracks.csv', help='Trackpy csv file name')
    parser.add_argument('--windows', type=str, nargs='+', default=['10:200:10', '200:2000:100'], 
                        help='Window sizes. Accepts space/comma separated numbers, or start:stop:step (e.g. 10 50 100, or 10:200:10)')
    parser.add_argument('--roi_bbox', type=int, nargs=4, default=None, metavar=('XMIN', 'XMAX', 'YMIN', 'YMAX'),
                        help='Bounding box for flow ROI (xmin xmax ymin ymax)')
    parser.add_argument('--particle_radius', type=int, default=0, help='Radius (in pixels) around particles to mask out (0 to disable masking).')
    args = parser.parse_args()

    base_path = Path(args.base_path)
    csv_path = base_path / args.tracks_csv
    h5_path = base_path / "GFP_flows.h5"

    if not csv_path.exists() or not h5_path.exists():
        print("Error: Required files (CSV or H5) not found.")
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

    # 1. データの読み込み
    print(f"Loading tracks from {csv_path}...")
    df_tracks = pd.read_csv(csv_path)
    
    # 2. 並列処理の設定
    workers = min(32, (os.cpu_count() or 1) + 4)
    grouped = df_tracks.groupby('frame')

    with h5py.File(str(h5_path), 'r') as f:
        dataset_key = list(f.keys())[0]
        # メモリ節約のため、参照のみ取得（スライシングで必要な時だけロード）
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

        active_frames = sorted([f for f in grouped.groups.keys() if f < num_frames])

        if args.roi_bbox is not None:
            roi_xmin, roi_xmax, roi_ymin, roi_ymax = args.roi_bbox
            roi_xmin = np.clip(roi_xmin, 0, cols - 1)
            roi_xmax = np.clip(roi_xmax, 0, cols - 1)
            roi_ymin = np.clip(roi_ymin, 0, rows - 1)
            roi_ymax = np.clip(roi_ymax, 0, rows - 1)
            
            # create slices for ROI to extract the field
            roi_y_slice = slice(roi_ymin, roi_ymax + 1)
            roi_x_slice = slice(roi_xmin, roi_xmax + 1)
            
            polar_order_roi_array = np.full((len(local_sizes), num_frames), np.nan, dtype=np.float32)
        else:
            polar_order_roi_array = None

        # ds_particles の初期化 (ここで df_tracks を変換してしまう)
        if 'particle' not in df_tracks.columns:
            df_tracks['particle'] = np.arange(len(df_tracks))
        
        ds_particles = df_tracks.set_index(['frame', 'particle']).to_xarray()
        
        polar_order_array = np.full((len(local_sizes), len(ds_particles.frame), len(ds_particles.particle)), np.nan, dtype=np.float32)

        frame_to_idx = {f: i for i, f in enumerate(ds_particles.frame.values)}
        particle_to_idx = {p: i for i, p in enumerate(ds_particles.particle.values)}

        def process_frame(t):
            frame_idx = grouped.groups[t]
            subset = df_tracks.loc[frame_idx]
            num_p = len(subset)
            p_ids = subset['particle'].values

            # 座標の準備
            y_idx = np.clip(np.round(subset['y'].values).astype(int), 0, rows - 1)
            x_idx = np.clip(np.round(subset['x'].values).astype(int), 0, cols - 1)

            # 流速データのロードと計算 (float32)
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
                m_ux[~np.isfinite(m_ux)] = 0
                m_uy[~np.isfinite(m_uy)] = 0

            if args.particle_radius > 0:
                particle_mask = np.zeros((rows, cols), dtype=bool)
                particle_mask[y_idx, x_idx] = True
                r = args.particle_radius
                y_grid, x_grid = np.ogrid[-r:r+1, -r:r+1]
                struct = x_grid**2 + y_grid**2 <= r**2
                particle_mask = binary_dilation(particle_mask, structure=struct)
                valid_mask = (~particle_mask).astype(np.float32)
                m_ux_masked = m_ux * valid_mask
                m_uy_masked = m_uy * valid_mask
            else:
                valid_mask = None

            p_vals = np.empty((len(local_sizes), num_p), dtype=np.float32)
            
            if args.roi_bbox is not None:
                roi_p_vals = np.empty((len(local_sizes),), dtype=np.float32)
            else:
                roi_p_vals = None
            
            for w_idx, size in enumerate(local_sizes):
                if args.particle_radius > 0:
                    valid_avg = uniform_filter(valid_mask, size=size)
                    with np.errstate(divide='ignore', invalid='ignore'):
                        u_avg = uniform_filter(m_ux_masked, size=size) / valid_avg
                        v_avg = uniform_filter(m_uy_masked, size=size) / valid_avg
                else:
                    u_avg = uniform_filter(m_ux, size=size)
                    v_avg = uniform_filter(m_uy, size=size)

                p_field = np.hypot(u_avg, v_avg, dtype=np.float32)

                p_vals[w_idx, :] = p_field[y_idx, x_idx]
                
                if args.roi_bbox is not None:
                    roi_p_vals[w_idx] = np.nanmean(p_field[roi_y_slice, roi_x_slice])

            return t, p_vals, p_ids, roi_p_vals

        # 3. 実行
        print(f"Calculating for window sizes: {local_sizes}...")
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for t, p_vals, p_ids, roi_p_vals in tqdm(executor.map(process_frame, active_frames), total=len(active_frames)):
                f_i = frame_to_idx[t]
                p_indices = [particle_to_idx[p] for p in p_ids]
                polar_order_array[:, f_i, p_indices] = p_vals
                
                if args.roi_bbox is not None:
                    polar_order_roi_array[:, t] = roi_p_vals

    # 4. xarray Dataset の構築と保存
    print("\nConsolidating data into xarray...")
    
    ds_particles['polar_order'] = xr.DataArray(
        polar_order_array,
        dims=['window size', 'frame', 'particle'],
        coords={'window size': local_sizes, 'frame': ds_particles.frame, 'particle': ds_particles.particle}
    )
    
    # メタデータの付与
    ds_particles.attrs['description'] = 'Local polar order analysis: Particles'
    ds_particles.attrs['window sizes'] = local_sizes
    ds_particles.attrs['particle_radius'] = args.particle_radius

    out_particle = base_path / "local_polar_w.zarr"

    # Zarr保存
    if out_particle.exists():
        shutil.rmtree(out_particle, ignore_errors=True)
    
    ds_particles.to_zarr(str(out_particle), mode='w', consolidated=False)
    
    print(f"Success! Data saved to {out_particle}")

    if args.roi_bbox is not None:
        print("\nConsolidating ROI data into xarray...")
        ds_roi = xr.Dataset(
            data_vars={
                'polar_order': xr.DataArray(
                    polar_order_roi_array,
                    dims=['window size', 'frame'],
                    coords={'window size': local_sizes, 'frame': np.arange(num_frames)}
                )
            }
        )
        ds_roi.attrs['description'] = f'Local polar order analysis for flow ROI (xmin={roi_xmin}, xmax={roi_xmax}, ymin={roi_ymin}, ymax={roi_ymax})'
        ds_roi.attrs['roi_bbox'] = [int(roi_xmin), int(roi_xmax), int(roi_ymin), int(roi_ymax)]
        ds_roi.attrs['particle_radius'] = args.particle_radius
        ds_roi.attrs['window sizes'] = local_sizes
        
        out_roi = base_path / "local_polar_flow_roi.zarr"
        if out_roi.exists():
            shutil.rmtree(out_roi, ignore_errors=True)
        ds_roi.to_zarr(str(out_roi), mode='w', consolidated=False)
        print(f"Success! ROI Data saved to {out_roi}")

if __name__ == "__main__":
    main()