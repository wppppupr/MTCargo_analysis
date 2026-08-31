import argparse
import numpy as np
import pandas as pd
import h5py
import xarray as xr
import os
import sys
import shutil
from pathlib import Path
from tqdm import tqdm

current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))
if str(current_dir.parent) not in sys.path:
    sys.path.insert(0, str(current_dir.parent))

try:
    from libs.fft_convolution import FFTConvolver
except ImportError:
    from fft_convolution import FFTConvolver


def main():
    parser = argparse.ArgumentParser(description='Calculate Gaussian local polar order for background ROI.')
    parser.add_argument('base_path', type=str, help='Path to the directory containing hdf5 and csv')
    parser.add_argument('--h5_file', type=str, default='GFP_flows.h5', help='H5 file name')
    parser.add_argument('--roi_x', type=int, default=None, help='X coordinate of the ROI center (optional)')
    parser.add_argument('--roi_y', type=int, default=None, help='Y coordinate of the ROI center (optional)')
    parser.add_argument('--tracks_csv', type=str, default='beads_tracks.csv', help='Trackpy csv file name')
    parser.add_argument('--sigmas', type=str, nargs='+', default=['5:100:5', '100:500:50'], 
                        help='Gaussian sigmas. Accepts space/comma separated numbers, or start:stop:step')
    parser.add_argument('--device', type=str, default=None, choices=['cuda', 'cpu', 'torch_cpu', 'scipy'],
                        help='Compute backend (cuda/cpu/scipy). Default: auto-detect GPU.')
    args = parser.parse_args()

    base_path = Path(args.base_path)
    csv_path = base_path / args.tracks_csv
    h5_path = base_path / args.h5_file

    if not h5_path.exists():
        print(f"Error: HDF5 file not found at {h5_path}")
        return

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

    df_tracks = None
    if csv_path.exists():
        print(f"Loading tracks from {csv_path} for overlap checking...")
        df_tracks = pd.read_csv(csv_path)

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
        half_w = max_size

        if args.roi_x is None or args.roi_y is None:
            if df_tracks is None:
                raise ValueError("Error: Track CSV is required to automatically find an empty ROI.")
            
            print("Automatically searching for the safest particle-free ROI...")
            y_all = np.clip(np.round(df_tracks['y'].values).astype(int), 0, rows - 1)
            x_all = np.clip(np.round(df_tracks['x'].values).astype(int), 0, cols - 1)
            particle_mask = np.zeros((rows, cols), dtype=bool)
            particle_mask[y_all, x_all] = True
            
            from scipy.ndimage import distance_transform_edt
            dist = distance_transform_edt(~particle_mask)
            
            max_dist = dist.max()
            if max_dist <= half_w:
                raise ValueError(f"Could not find any empty space large enough for the max sigma ({max_size}).")
            
            y_idx, x_idx = np.unravel_index(dist.argmax(), dist.shape)
            print(f"Automatically selected ROI center at ({x_idx}, {y_idx}) with safe radius {max_dist:.1f} px.")
        else:
            y_idx = np.clip(args.roi_y, 0, rows - 1)
            x_idx = np.clip(args.roi_x, 0, cols - 1)

        polar_order_array = np.full((len(local_sizes), num_frames), np.nan, dtype=np.float32)

        if df_tracks is not None:
            y_all = np.clip(np.round(df_tracks['y'].values).astype(int), 0, rows - 1)
            x_all = np.clip(np.round(df_tracks['x'].values).astype(int), 0, cols - 1)
            particle_mask = np.zeros((rows, cols), dtype=bool)
            particle_mask[y_all, x_all] = True
            from scipy.ndimage import distance_transform_edt
            dist_map = distance_transform_edt(~particle_mask)
            safe_masks = [(dist_map > (s * 3 + 5)) for s in local_sizes]
        else:
            safe_masks = [np.ones((rows, cols), dtype=bool) for _ in local_sizes]

        print(f"Initializing Fast Gaussian FFT Convolver ({len(local_sizes)} sigmas, device={args.device or 'auto'})...")
        convolver = FFTConvolver(shape=(rows, cols), sizes=local_sizes, kernel_type='gaussian', device=args.device)
        print(f"Using backend: {convolver.device_type}")

        print(f"Calculating Gaussian background polar order for {num_frames} frames...")
        for t in tqdm(range(num_frames)):
            if channel_first:
                m_x = flow_data[t, 0, ...].astype(np.float32)
                m_y = flow_data[t, 1, ...].astype(np.float32)
            else:
                m_x = flow_data[t, ..., 0].astype(np.float32)
                m_y = flow_data[t, ..., 1].astype(np.float32)
            
            v_mag = np.hypot(m_x, m_y)
            with np.errstate(divide='ignore', invalid='ignore'):
                m_ux = np.where(v_mag > 0, m_x / v_mag, 0.0).astype(np.float32)
                m_uy = np.where(v_mag > 0, m_y / v_mag, 0.0).astype(np.float32)

            p_vals = convolver.convolve_and_sample_bg_polar(m_ux=m_ux, m_uy=m_uy, safe_masks=safe_masks)
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
    
    ds_roi.attrs['description'] = f'Gaussian Local polar order analysis for arbitrary ROI at ({x_idx}, {y_idx})'
    ds_roi.attrs['roi_x'] = int(x_idx)
    ds_roi.attrs['roi_y'] = int(y_idx)
    ds_roi.attrs['window sizes'] = local_sizes

    out_roi = base_path / "local_polar_gaussian_bg.zarr"
    if out_roi.exists():
        shutil.rmtree(out_roi, ignore_errors=True)
    
    ds_roi.to_zarr(str(out_roi), mode='w', consolidated=False)
    print(f"Success! Data saved to {out_roi}")


if __name__ == "__main__":
    main()
