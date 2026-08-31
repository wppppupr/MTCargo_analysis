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


def parse_distances(distance_args):
    """
    Parse distance / window arguments.
    Accepts space/comma separated numbers, or start:stop:step (e.g. 10 50 100, or 10:200:10).
    """
    sizes = set()
    for arg_w in distance_args:
        for p in arg_w.split(','):
            if not p.strip():
                continue
            if ':' in p:
                parts = p.split(':')
                w_start = int(parts[0])
                w_stop = int(parts[1]) if len(parts) > 1 else w_start
                w_step = int(parts[2]) if len(parts) > 2 else 1
                sizes.update(range(w_start, w_stop + 1, w_step))
            else:
                sizes.add(int(p))
    return sorted(list(sizes))


def compute_bead_unit_velocities(df_tracks):
    """
    Compute unit displacement vectors for particles from track coordinates.
    """
    df = df_tracks.copy()
    if 'particle' not in df.columns:
        df['particle'] = 0

    df = df.sort_values(by=['particle', 'frame']).reset_index(drop=True)
    
    df['dx'] = np.nan
    df['dy'] = np.nan
    df['bead_ux'] = np.nan
    df['bead_uy'] = np.nan

    grouped = df.groupby('particle')
    dx_list, dy_list = [], []
    
    for _, group in grouped:
        x = group['x'].values
        y = group['y'].values
        n = len(group)
        if n == 1:
            dx = np.zeros(1, dtype=np.float32)
            dy = np.zeros(1, dtype=np.float32)
        else:
            dx = np.zeros(n, dtype=np.float32)
            dy = np.zeros(n, dtype=np.float32)
            dx[0] = x[1] - x[0]
            dy[0] = y[1] - y[0]
            dx[-1] = x[-1] - x[-2]
            dy[-1] = y[-1] - y[-2]
            if n > 2:
                dx[1:-1] = (x[2:] - x[:-2]) / 2.0
                dy[1:-1] = (y[2:] - y[:-2]) / 2.0
                
        dx_list.append(dx)
        dy_list.append(dy)

    if len(dx_list) > 0:
        df['dx'] = np.concatenate(dx_list)
        df['dy'] = np.concatenate(dy_list)
        norm = np.hypot(df['dx'].values, df['dy'].values)
        with np.errstate(divide='ignore', invalid='ignore'):
            df['bead_ux'] = np.where(norm > 0, df['dx'].values / norm, 0.0).astype(np.float32)
            df['bead_uy'] = np.where(norm > 0, df['dy'].values / norm, 0.0).astype(np.float32)

    return df


def main():
    parser = argparse.ArgumentParser(description='Calculate angular spatial correlation of optical flow around particles.')
    parser.add_argument('base_path', type=str, help='Path to the directory containing hdf5 and csv')
    parser.add_argument('--tracks_csv', type=str, default='beads_tracks.csv', help='Trackpy csv file name')
    parser.add_argument('--h5_file', type=str, default='GFP_flows.h5', help='H5 file name')
    parser.add_argument('--distances', '--windows', dest='distances', type=str, nargs='+', default=['2:100:2', '100:500:20'], 
                        help='Distance ranges / radii in pixels. Accepts space/comma separated numbers, or start:stop:step (e.g. 5:100:5)')
    parser.add_argument('--shell_width', type=float, default=2.0, help='Width of the annular shell for ring kernel (pixels).')
    parser.add_argument('--kernel_type', type=str, default='ring', choices=['ring', 'disk', 'gaussian'],
                        help='Kernel type: "ring" (annular shell, default), "disk" (circular window), or "gaussian".')
    parser.add_argument('--roi_bbox', type=int, nargs=4, default=None, metavar=('XMIN', 'XMAX', 'YMIN', 'YMAX'),
                        help='Bounding box for flow ROI (xmin xmax ymin ymax)')
    parser.add_argument('--particle_radius', type=int, default=0, help='Radius (in pixels) around particles to mask out (0 to disable masking).')
    parser.add_argument('--out_name', type=str, default='angular_correlation_w.zarr', help='Output zarr directory name')
    parser.add_argument('--device', type=str, default=None, choices=['cuda', 'cpu', 'torch_cpu', 'scipy'],
                        help='Compute backend (cuda/cpu/scipy). Default: auto-detect GPU.')
    args = parser.parse_args()

    base_path = Path(args.base_path)
    csv_path = base_path / args.tracks_csv
    h5_path = base_path / args.h5_file

    if not csv_path.exists() or not h5_path.exists():
        print(f"Error: Required files not found.\n CSV: {csv_path} (exists: {csv_path.exists()})\n H5: {h5_path} (exists: {h5_path.exists()})")
        return

    distances = parse_distances(args.distances)
    print(f"Distances to compute: {distances}")

    # 1. Load tracking data and compute bead velocities
    print(f"Loading tracks from {csv_path}...")
    df_tracks = pd.read_csv(csv_path)
    df_tracks = compute_bead_unit_velocities(df_tracks)
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
            corr_roi_array = np.full((len(distances), num_frames), np.nan, dtype=np.float32)
        else:
            roi_slice = None
            corr_roi_array = None

        ds_particles = df_tracks.set_index(['frame', 'particle']).to_xarray()
        corr_flow_array = np.full((len(distances), len(ds_particles.frame), len(ds_particles.particle)), np.nan, dtype=np.float32)
        corr_bead_array = np.full((len(distances), len(ds_particles.frame), len(ds_particles.particle)), np.nan, dtype=np.float32)

        frame_to_idx = {f: i for i, f in enumerate(ds_particles.frame.values)}
        particle_to_idx = {p: i for i, p in enumerate(ds_particles.particle.values)}

        # 2. FFTConvolver の初期化
        print(f"Initializing Fast FFT Convolver ({len(distances)} distances, kernel={args.kernel_type}, device={args.device or 'auto'})...")
        convolver = FFTConvolver(shape=(rows, cols), sizes=distances, kernel_type=args.kernel_type, shell_width=args.shell_width, device=args.device)
        print(f"Using backend: {convolver.device_type}")

        # 3. 実行ループ
        print(f"Calculating angular spatial correlation for {len(active_frames)} frames...")
        for t in tqdm(active_frames):
            frame_indices = grouped.groups[t]
            subset = df_tracks.loc[frame_indices]
            p_ids = subset['particle'].values

            y_idx = np.clip(np.round(subset['y'].values).astype(int), 0, rows - 1)
            x_idx = np.clip(np.round(subset['x'].values).astype(int), 0, cols - 1)

            b_ux = subset['bead_ux'].values.astype(np.float32)
            b_uy = subset['bead_uy'].values.astype(np.float32)

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

            center_flow_ux = m_ux[y_idx, x_idx]
            center_flow_uy = m_uy[y_idx, x_idx]

            if args.particle_radius > 0:
                mask_uint8 = np.ones((rows, cols), dtype=np.uint8)
                r_mask = int(args.particle_radius)
                for xp, yp in zip(x_idx, y_idx):
                    cv2.circle(mask_uint8, (int(xp), int(yp)), r_mask, 0, thickness=-1)
                valid_mask = mask_uint8.astype(np.float32)
                m_ux = m_ux * valid_mask
                m_uy = m_uy * valid_mask
            else:
                valid_mask = None

            p_corr_flow, p_corr_bead, roi_corr_vals = convolver.convolve_and_sample_angular_correlation(
                m_ux=m_ux, m_uy=m_uy, valid_mask=valid_mask,
                y_idx=y_idx, x_idx=x_idx,
                center_flow_ux=center_flow_ux, center_flow_uy=center_flow_uy,
                b_ux=b_ux, b_uy=b_uy, roi_slice=roi_slice
            )

            f_i = frame_to_idx[t]
            p_indices = [particle_to_idx[p] for p in p_ids]
            corr_flow_array[:, f_i, p_indices] = p_corr_flow
            corr_bead_array[:, f_i, p_indices] = p_corr_bead

            if roi_slice is not None:
                corr_roi_array[:, t] = roi_corr_vals

    # 4. Consolidate into xarray and save to Zarr
    print("\nConsolidating particle correlation data into xarray...")
    ds_particles['angular_correlation'] = xr.DataArray(
        corr_flow_array,
        dims=['distance', 'frame', 'particle'],
        coords={'distance': distances, 'frame': ds_particles.frame, 'particle': ds_particles.particle}
    )
    ds_particles['bead_correlation'] = xr.DataArray(
        corr_bead_array,
        dims=['distance', 'frame', 'particle'],
        coords={'distance': distances, 'frame': ds_particles.frame, 'particle': ds_particles.particle}
    )

    ds_particles.attrs['description'] = 'Angular spatial correlation analysis: Particles'
    ds_particles.attrs['distances'] = distances
    ds_particles.attrs['kernel_type'] = args.kernel_type
    ds_particles.attrs['shell_width'] = args.shell_width
    ds_particles.attrs['particle_radius'] = args.particle_radius

    out_particle = base_path / args.out_name
    if out_particle.exists():
        shutil.rmtree(out_particle, ignore_errors=True)

    ds_particles.to_zarr(str(out_particle), mode='w', consolidated=False)
    print(f"Success! Particle angular correlation saved to {out_particle}")

    if args.roi_bbox is not None:
        print("\nConsolidating ROI correlation data into xarray...")
        ds_roi = xr.Dataset(
            data_vars={
                'angular_correlation': xr.DataArray(
                    corr_roi_array,
                    dims=['distance', 'frame'],
                    coords={'distance': distances, 'frame': np.arange(num_frames)}
                )
            }
        )
        ds_roi.attrs['description'] = f'Angular spatial correlation analysis for flow ROI (xmin={roi_xmin}, xmax={roi_xmax}, ymin={roi_ymin}, ymax={roi_ymax})'
        ds_roi.attrs['roi_bbox'] = [int(roi_xmin), int(roi_xmax), int(roi_ymin), int(roi_ymax)]
        ds_roi.attrs['distances'] = distances
        ds_roi.attrs['kernel_type'] = args.kernel_type
        ds_roi.attrs['shell_width'] = args.shell_width
        ds_roi.attrs['particle_radius'] = args.particle_radius

        out_roi = base_path / "angular_correlation_flow_roi.zarr"
        if out_roi.exists():
            shutil.rmtree(out_roi, ignore_errors=True)
        ds_roi.to_zarr(str(out_roi), mode='w', consolidated=False)
        print(f"Success! ROI angular correlation saved to {out_roi}")


if __name__ == "__main__":
    main()
