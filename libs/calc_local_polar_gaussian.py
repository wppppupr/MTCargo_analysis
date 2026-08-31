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
import cv2

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
    parser = argparse.ArgumentParser(description='Calculate local polar order for particles and background using Gaussian weights.')
    parser.add_argument('base_path', type=str, help='Path to the directory containing hdf5 and csv')
    parser.add_argument('--tracks_csv', type=str, default='beads_tracks.csv', help='Trackpy csv file name')
    parser.add_argument('--h5_file', type=str, default='GFP_flows.h5', help='H5 file name')
    parser.add_argument('--sigmas', type=str, nargs='+', default=['5:100:5', '100:1000:50'], 
                        help='Gaussian sigmas. Accepts space/comma separated numbers, or start:stop:step (e.g. 10 50 100, or 10:200:10)')
    parser.add_argument('--roi_bbox', type=int, nargs=4, default=None, metavar=('XMIN', 'XMAX', 'YMIN', 'YMAX'),
                        help='Bounding box for flow ROI (xmin xmax ymin ymax)')
    parser.add_argument('--particle_radius', type=int, default=0, help='Radius (in pixels) around particles to mask out (0 to disable masking).')
    parser.add_argument('--device', type=str, default=None, choices=['cuda', 'cpu', 'torch_cpu', 'scipy'],
                        help='Compute backend (cuda/cpu/scipy). Default: auto-detect GPU.')
    args = parser.parse_args()

    base_path = Path(args.base_path)
    csv_path = base_path / args.tracks_csv
    h5_path = base_path / args.h5_file

    if not csv_path.exists() or not h5_path.exists():
        print(f"Error: Required files not found.\n CSV: {csv_path} (exists: {csv_path.exists()})\n H5: {h5_path} (exists: {h5_path.exists()})")
        return

    # Parse window sizes (sigmas)
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

    print(f"Loading tracks from {csv_path}...")
    df_tracks = pd.read_csv(csv_path)
    grouped = df_tracks.groupby('frame')

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

        active_frames = sorted([f for f in grouped.groups.keys() if f < num_frames])

        if args.roi_bbox is not None:
            roi_xmin, roi_xmax, roi_ymin, roi_ymax = args.roi_bbox
            roi_xmin = int(np.clip(roi_xmin, 0, cols - 1))
            roi_xmax = int(np.clip(roi_xmax, 0, cols - 1))
            roi_ymin = int(np.clip(roi_ymin, 0, rows - 1))
            roi_ymax = int(np.clip(roi_ymax, 0, rows - 1))
            
            roi_slice = (slice(roi_ymin, roi_ymax + 1), slice(roi_xmin, roi_xmax + 1))
            polar_order_roi_array = np.full((len(local_sizes), num_frames), np.nan, dtype=np.float32)
        else:
            roi_slice = None
            polar_order_roi_array = None

        if 'particle' not in df_tracks.columns:
            df_tracks['particle'] = np.arange(len(df_tracks))
        
        ds_particles = df_tracks.set_index(['frame', 'particle']).to_xarray()
        polar_order_array = np.full((len(local_sizes), len(ds_particles.frame), len(ds_particles.particle)), np.nan, dtype=np.float32)

        frame_to_idx = {f: i for i, f in enumerate(ds_particles.frame.values)}
        particle_to_idx = {p: i for i, p in enumerate(ds_particles.particle.values)}

        print(f"Initializing Fast Gaussian FFT Convolver ({len(local_sizes)} sigmas, device={args.device or 'auto'})...")
        convolver = FFTConvolver(shape=(rows, cols), sizes=local_sizes, kernel_type='gaussian', device=args.device)
        print(f"Using backend: {convolver.device_type}")

        print(f"Calculating Gaussian local polar order for {len(active_frames)} frames...")
        for t in tqdm(active_frames):
            frame_idx = grouped.groups[t]
            subset = df_tracks.loc[frame_idx]
            p_ids = subset['particle'].values

            y_idx = np.clip(np.round(subset['y'].values).astype(int), 0, rows - 1)
            x_idx = np.clip(np.round(subset['x'].values).astype(int), 0, cols - 1)

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

            if args.particle_radius > 0:
                mask_uint8 = np.ones((rows, cols), dtype=np.uint8)
                r = int(args.particle_radius)
                for xp, yp in zip(x_idx, y_idx):
                    cv2.circle(mask_uint8, (int(xp), int(yp)), r, 0, thickness=-1)
                valid_mask = mask_uint8.astype(np.float32)
                m_ux = m_ux * valid_mask
                m_uy = m_uy * valid_mask
            else:
                valid_mask = None

            p_vals, roi_p_vals = convolver.convolve_and_sample_polar(
                m_ux=m_ux, m_uy=m_uy, valid_mask=valid_mask,
                y_idx=y_idx, x_idx=x_idx, roi_slice=roi_slice
            )

            f_i = frame_to_idx[t]
            p_indices = [particle_to_idx[p] for p in p_ids]
            polar_order_array[:, f_i, p_indices] = p_vals

            if roi_slice is not None:
                polar_order_roi_array[:, t] = roi_p_vals

    print("\nConsolidating data into xarray...")
    ds_particles['polar_order'] = xr.DataArray(
        polar_order_array,
        dims=['window size', 'frame', 'particle'],
        coords={'window size': local_sizes, 'frame': ds_particles.frame, 'particle': ds_particles.particle}
    )
    ds_particles.attrs['description'] = 'Gaussian Local polar order analysis: Particles'
    ds_particles.attrs['window sizes'] = local_sizes
    ds_particles.attrs['particle_radius'] = args.particle_radius

    out_particle = base_path / "local_polar_gaussian_w.zarr"
    if out_particle.exists():
        shutil.rmtree(out_particle, ignore_errors=True)
    
    ds_particles.to_zarr(str(out_particle), mode='w', consolidated=False)
    print(f"Success! Data saved to {out_particle}")

    if args.roi_bbox is not None:
        ds_roi = xr.Dataset(
            data_vars={
                'polar_order': xr.DataArray(
                    polar_order_roi_array,
                    dims=['window size', 'frame'],
                    coords={'window size': local_sizes, 'frame': np.arange(num_frames)}
                )
            }
        )
        ds_roi.attrs['description'] = f'Gaussian Local polar order analysis for flow ROI'
        ds_roi.attrs['roi_bbox'] = [int(roi_xmin), int(roi_xmax), int(roi_ymin), int(roi_ymax)]
        ds_roi.attrs['particle_radius'] = args.particle_radius
        ds_roi.attrs['window sizes'] = local_sizes
        
        out_roi = base_path / "local_polar_flow_gaussian_roi.zarr"
        if out_roi.exists():
            shutil.rmtree(out_roi, ignore_errors=True)
        ds_roi.to_zarr(str(out_roi), mode='w', consolidated=False)
        print(f"Success! ROI Data saved to {out_roi}")


if __name__ == "__main__":
    main()