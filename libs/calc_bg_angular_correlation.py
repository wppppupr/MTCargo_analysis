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

def create_kernel(r, kernel_type='ring', shell_width=2.0):
    """
    Create a spatial convolution kernel for a given distance/radius r.
    """
    if r == 0:
        return np.ones((1, 1), dtype=np.float32)

    half_w = float(shell_width) / 2.0

    if kernel_type == 'ring':
        max_r = int(np.ceil(r + half_w))
        ksize = 2 * max_r + 1
        center = max_r
        ky, kx = np.ogrid[:ksize, :ksize]
        dist = np.hypot(kx - center, ky - center)
        kmask = (dist >= (r - half_w)) & (dist <= (r + half_w))
        kernel = np.zeros((ksize, ksize), dtype=np.float32)
        if np.any(kmask):
            kernel[kmask] = 1.0
        else:
            closest_idx = np.unravel_index(np.argmin(np.abs(dist - r)), dist.shape)
            kernel[closest_idx] = 1.0
    elif kernel_type == 'disk':
        max_r = int(np.ceil(r))
        ksize = 2 * max_r + 1
        center = max_r
        ky, kx = np.ogrid[:ksize, :ksize]
        dist = np.hypot(kx - center, ky - center)
        kmask = dist <= r
        kernel = np.zeros((ksize, ksize), dtype=np.float32)
        kernel[kmask] = 1.0
    elif kernel_type == 'gaussian':
        sigma = float(r)
        max_r = int(np.ceil(3 * sigma))
        ksize = 2 * max_r + 1
        center = max_r
        ky, kx = np.ogrid[:ksize, :ksize]
        dist_sq = (kx - center)**2 + (ky - center)**2
        kernel = np.exp(-dist_sq / (2.0 * sigma**2)).astype(np.float32)
    else:
        raise ValueError(f"Unknown kernel_type: {kernel_type}. Choose from 'ring', 'disk', 'gaussian'.")

    k_sum = kernel.sum()
    if k_sum > 0:
        kernel /= k_sum
    return kernel

def main():
    parser = argparse.ArgumentParser(description='Calculate angular spatial correlation for background / ROI.')
    parser.add_argument('base_path', type=str, help='Path to the directory containing hdf5 and csv')
    parser.add_argument('--h5_file', type=str, default='GFP_flows.h5', help='H5 file name')
    parser.add_argument('--roi_x', type=int, default=None, help='X coordinate of the ROI center (optional)')
    parser.add_argument('--roi_y', type=int, default=None, help='Y coordinate of the ROI center (optional)')
    parser.add_argument('--tracks_csv', type=str, default='beads_tracks.csv', help='Trackpy csv file name')
    parser.add_argument('--distances', '--windows', dest='distances', type=str, nargs='+', default=['2:100:2', '100:500:20'], 
                        help='Distance ranges / radii in pixels. Accepts space/comma separated numbers, or start:stop:step')
    parser.add_argument('--shell_width', type=float, default=2.0, help='Width of the annular shell for ring kernel (pixels).')
    parser.add_argument('--kernel_type', type=str, default='ring', choices=['ring', 'disk', 'gaussian'],
                        help='Kernel type: "ring" (annular shell, default), "disk" (circular window), or "gaussian".')
    parser.add_argument('--out_name', type=str, default='angular_correlation_bg.zarr', help='Output zarr directory name')
    args = parser.parse_args()

    base_path = Path(args.base_path)
    csv_path = base_path / args.tracks_csv
    h5_path = base_path / args.h5_file

    if not h5_path.exists():
        print(f"Error: HDF5 file not found at {h5_path}")
        return

    # Parse distance list
    distances = parse_distances(args.distances)
    print(f"Distances to compute: {distances}")

    # Read particle tracks if they exist for overlap checking
    df_tracks = None
    if csv_path.exists():
        print(f"Loading tracks from {csv_path} for overlap checking...")
        df_tracks = pd.read_csv(csv_path)
    else:
        print(f"Warning: Track file not found at {csv_path}. Particle overlap checking will be skipped.")

    kernels = [create_kernel(d, kernel_type=args.kernel_type, shell_width=args.shell_width) for d in distances]

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

        max_dist = max(distances)
        half_w = max_dist

        if args.roi_x is None or args.roi_y is None:
            if df_tracks is None:
                raise ValueError("Error: Track CSV is required to automatically find an empty ROI.")

            print("Automatically searching for the safest particle-free ROI...")
            y_all = np.clip(np.round(df_tracks['y'].values).astype(int), 0, rows - 1)
            x_all = np.clip(np.round(df_tracks['x'].values).astype(int), 0, cols - 1)
            particle_mask = np.zeros((rows, cols), dtype=bool)
            particle_mask[y_all, x_all] = True

            from scipy.ndimage import distance_transform_edt
            dist_map = distance_transform_edt(~particle_mask)

            max_available = dist_map.max()
            if max_available <= half_w:
                print(f"Warning: Safe radius to nearest particle is {max_available:.1f} px, which is smaller than max distance ({half_w}).")

            roi_y, roi_x = np.unravel_index(dist_map.argmax(), dist_map.shape)
            print(f"Automatically selected ROI center at ({roi_x}, {roi_y}) with a safe radius of {max_available:.1f} px to the nearest particle.")
        else:
            roi_y = np.clip(args.roi_y, 0, rows - 1)
            roi_x = np.clip(args.roi_x, 0, cols - 1)

            # Check for particle overlap for manual ROI
            if df_tracks is not None:
                in_roi = df_tracks[
                    (df_tracks['x'] >= roi_x - half_w) & (df_tracks['x'] <= roi_x + half_w) &
                    (df_tracks['y'] >= roi_y - half_w) & (df_tracks['y'] <= roi_y + half_w)
                ]
                if not in_roi.empty:
                    frames_with_particles = in_roi['frame'].unique()
                    print(
                        f"Warning: Particle(s) detected inside the manual ROI at ({roi_x}, {roi_y}) "
                        f"with the maximum window size ({max_dist}).\n"
                        f"Frames with overlap: {frames_with_particles}"
                    )
                else:
                    print("No particles overlap with the specified ROI. Proceeding...")

        corr_bg_array = np.full((len(distances), num_frames), np.nan, dtype=np.float32)

        def process_frame(t):
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

            # Center flow direction at ROI center
            center_ux = m_ux[roi_y, roi_x]
            center_uy = m_uy[roi_y, roi_x]

            c_vals = np.empty((len(distances),), dtype=np.float32)

            for d_idx, kernel in enumerate(kernels):
                u_avg = cv2.filter2D(m_ux, -1, kernel, borderType=cv2.BORDER_REFLECT)
                v_avg = cv2.filter2D(m_uy, -1, kernel, borderType=cv2.BORDER_REFLECT)

                # Correlation at ROI center: <u(roi) . u(roi + r)>
                avg_ux_at_roi = u_avg[roi_y, roi_x]
                avg_uy_at_roi = v_avg[roi_y, roi_x]
                c_vals[d_idx] = center_ux * avg_ux_at_roi + center_uy * avg_uy_at_roi

            return t, c_vals

        workers = min(32, (os.cpu_count() or 1) + 4)
        print(f"Calculating background angular spatial correlation across {num_frames} frames...")
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for t, c_vals in tqdm(executor.map(process_frame, range(num_frames)), total=num_frames):
                corr_bg_array[:, t] = c_vals

    print("\nConsolidating background data into xarray...")
    ds_bg = xr.Dataset(
        data_vars={
            'angular_correlation': xr.DataArray(
                corr_bg_array,
                dims=['distance', 'frame'],
                coords={'distance': distances, 'frame': np.arange(num_frames)}
            )
        }
    )

    ds_bg.attrs['description'] = f'Angular spatial correlation analysis for Background ROI at ({roi_x}, {roi_y})'
    ds_bg.attrs['roi_center'] = [int(roi_x), int(roi_y)]
    ds_bg.attrs['distances'] = distances
    ds_bg.attrs['kernel_type'] = args.kernel_type
    ds_bg.attrs['shell_width'] = args.shell_width

    out_bg = base_path / args.out_name
    if out_bg.exists():
        shutil.rmtree(out_bg, ignore_errors=True)

    ds_bg.to_zarr(str(out_bg), mode='w', consolidated=False)
    print(f"Success! Background angular correlation saved to {out_bg}")

if __name__ == "__main__":
    main()
