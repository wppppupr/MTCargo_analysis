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
    
    Parameters:
    -----------
    r : float or int
        Distance / radius in pixels.
    kernel_type : str
        'ring' (annular shell), 'disk' (circular disk), or 'gaussian'.
    shell_width : float
        Thickness of the annular shell for 'ring' kernel in pixels.
        
    Returns:
    --------
    kernel : np.ndarray (float32)
        Normalized 2D kernel.
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
            # Fallback for small r where no grid point falls strictly within range
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

def compute_bead_unit_velocities(df_tracks):
    """
    Compute unit displacement vectors for particles from track coordinates.
    """
    df = df_tracks.copy()
    if 'particle' not in df.columns:
        df['particle'] = 0

    df = df.sort_values(by=['particle', 'frame']).reset_index(drop=True)
    
    # Calculate displacement: central difference if possible, else forward/backward
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
            # Forward diff for first
            dx[0] = x[1] - x[0]
            dy[0] = y[1] - y[0]
            # Backward diff for last
            dx[-1] = x[-1] - x[-2]
            dy[-1] = y[-1] - y[-2]
            # Central diff for middle
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
    args = parser.parse_args()

    base_path = Path(args.base_path)
    csv_path = base_path / args.tracks_csv
    h5_path = base_path / args.h5_file

    if not csv_path.exists() or not h5_path.exists():
        print(f"Error: Required files not found.\n CSV: {csv_path} (exists: {csv_path.exists()})\n H5: {h5_path} (exists: {h5_path.exists()})")
        return

    # Parse distance list
    distances = parse_distances(args.distances)
    print(f"Distances to compute: {distances}")

    # 1. Load tracking data and compute bead velocities
    print(f"Loading tracks from {csv_path}...")
    df_tracks = pd.read_csv(csv_path)
    df_tracks = compute_bead_unit_velocities(df_tracks)

    # 2. Setup parallel processing and hdf5 data
    workers = min(32, (os.cpu_count() or 1) + 4)
    grouped = df_tracks.groupby('frame')

    # Pre-generate kernels for each distance
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

        active_frames = sorted([f for f in grouped.groups.keys() if f < num_frames])

        if args.roi_bbox is not None:
            roi_xmin, roi_xmax, roi_ymin, roi_ymax = args.roi_bbox
            roi_xmin = int(np.clip(roi_xmin, 0, cols - 1))
            roi_xmax = int(np.clip(roi_xmax, 0, cols - 1))
            roi_ymin = int(np.clip(roi_ymin, 0, rows - 1))
            roi_ymax = int(np.clip(roi_ymax, 0, rows - 1))

            roi_y_slice = slice(roi_ymin, roi_ymax + 1)
            roi_x_slice = slice(roi_xmin, roi_xmax + 1)
            corr_roi_array = np.full((len(distances), num_frames), np.nan, dtype=np.float32)
        else:
            corr_roi_array = None

        # ds_particles initialization
        ds_particles = df_tracks.set_index(['frame', 'particle']).to_xarray()

        corr_flow_array = np.full((len(distances), len(ds_particles.frame), len(ds_particles.particle)), np.nan, dtype=np.float32)
        corr_bead_array = np.full((len(distances), len(ds_particles.frame), len(ds_particles.particle)), np.nan, dtype=np.float32)

        frame_to_idx = {f: i for i, f in enumerate(ds_particles.frame.values)}
        particle_to_idx = {p: i for i, p in enumerate(ds_particles.particle.values)}

        def process_frame(t):
            frame_indices = grouped.groups[t]
            subset = df_tracks.loc[frame_indices]
            num_p = len(subset)
            p_ids = subset['particle'].values

            # Particle coordinates
            y_idx = np.clip(np.round(subset['y'].values).astype(int), 0, rows - 1)
            x_idx = np.clip(np.round(subset['x'].values).astype(int), 0, cols - 1)

            # Bead unit velocities
            b_ux = subset['bead_ux'].values.astype(np.float32)
            b_uy = subset['bead_uy'].values.astype(np.float32)

            # Load optical flow (float32)
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

            # Optical flow direction at particle centers
            center_flow_ux = m_ux[y_idx, x_idx]
            center_flow_uy = m_uy[y_idx, x_idx]

            # Particle masking
            if args.particle_radius > 0:
                mask_uint8 = np.ones((rows, cols), dtype=np.uint8)
                r_mask = int(args.particle_radius)
                for xp, yp in zip(x_idx, y_idx):
                    cv2.circle(mask_uint8, (int(xp), int(yp)), r_mask, 0, thickness=-1)
                valid_mask = mask_uint8.astype(np.float32)
                m_ux_masked = m_ux * valid_mask
                m_uy_masked = m_uy * valid_mask
            else:
                valid_mask = None
                m_ux_masked = m_ux
                m_uy_masked = m_uy

            p_corr_flow = np.empty((len(distances), num_p), dtype=np.float32)
            p_corr_bead = np.empty((len(distances), num_p), dtype=np.float32)
            roi_corr_vals = np.empty((len(distances),), dtype=np.float32) if args.roi_bbox is not None else None

            for d_idx, kernel in enumerate(kernels):
                if args.particle_radius > 0:
                    valid_avg = cv2.filter2D(valid_mask, -1, kernel, borderType=cv2.BORDER_REFLECT)
                    with np.errstate(divide='ignore', invalid='ignore'):
                        u_avg = np.where(valid_avg > 0, cv2.filter2D(m_ux_masked, -1, kernel, borderType=cv2.BORDER_REFLECT) / valid_avg, np.nan)
                        v_avg = np.where(valid_avg > 0, cv2.filter2D(m_uy_masked, -1, kernel, borderType=cv2.BORDER_REFLECT) / valid_avg, np.nan)
                else:
                    u_avg = cv2.filter2D(m_ux, -1, kernel, borderType=cv2.BORDER_REFLECT)
                    v_avg = cv2.filter2D(m_uy, -1, kernel, borderType=cv2.BORDER_REFLECT)

                # Sample average surrounding flow at particle positions
                avg_ux_at_p = u_avg[y_idx, x_idx]
                avg_uy_at_p = v_avg[y_idx, x_idx]

                # 1. Angular correlation with center optical flow: <u_flow(x_p) . u_flow(x_p + r)>
                p_corr_flow[d_idx, :] = center_flow_ux * avg_ux_at_p + center_flow_uy * avg_uy_at_p

                # 2. Angular correlation with bead movement direction: <u_bead . u_flow(x_p + r)>
                p_corr_bead[d_idx, :] = b_ux * avg_ux_at_p + b_uy * avg_uy_at_p

                # 3. ROI spatial correlation if requested
                if args.roi_bbox is not None:
                    # Pointwise correlation map across entire field: u(x) . u_avg(x)
                    corr_map = m_ux * u_avg + m_uy * v_avg
                    roi_corr_vals[d_idx] = np.nanmean(corr_map[roi_y_slice, roi_x_slice])

            return t, p_corr_flow, p_corr_bead, p_ids, roi_corr_vals

        # 3. Parallel Execution
        print(f"Calculating angular spatial correlation for {len(distances)} distances across {len(active_frames)} frames...")
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for t, p_corr_flow, p_corr_bead, p_ids, roi_corr_vals in tqdm(executor.map(process_frame, active_frames), total=len(active_frames)):
                f_i = frame_to_idx[t]
                p_indices = [particle_to_idx[p] for p in p_ids]
                corr_flow_array[:, f_i, p_indices] = p_corr_flow
                corr_bead_array[:, f_i, p_indices] = p_corr_bead

                if args.roi_bbox is not None:
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
