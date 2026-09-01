import argparse
import numpy as np
import pandas as pd
import h5py
import xarray as xr
import zarr
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

    for p_id, group in df.groupby('particle'):
        idxs = group.index.values
        if len(idxs) < 2:
            continue
        
        dx = np.diff(group['x'].values)
        dy = np.diff(group['y'].values)
        mag = np.hypot(dx, dy)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            ux = np.where(mag > 0, dx / mag, 0.0)
            uy = np.where(mag > 0, dy / mag, 0.0)
        
        # Forward diff for t to t+1
        df.loc[idxs[:-1], 'dx'] = dx.astype(np.float32)
        df.loc[idxs[:-1], 'dy'] = dy.astype(np.float32)
        df.loc[idxs[:-1], 'bead_ux'] = ux.astype(np.float32)
        df.loc[idxs[:-1], 'bead_uy'] = uy.astype(np.float32)

    return df


def load_nematic_thetas(base_path, num_frames, flow_data=None, channel_first=True):
    """
    Load or compute the global nematic director angle theta(t) for each frame.
    1. If MTs_im_theta.zarr exists, compute global nematic angle using tensor average:
       0.5 * arctan2(nanmean(sin(2*theta)), nanmean(cos(2*theta)))
    2. Fallback: compute from optical flow field orientation (2-tensor average).
    """
    theta_path = base_path / "MTs_im_theta.zarr"
    thetas = np.zeros(num_frames, dtype=np.float32)

    if theta_path.exists():
        try:
            im_theta_zarr = zarr.open_array(str(theta_path), mode='r')
            mts_frames = min(im_theta_zarr.shape[0], num_frames)
            for t in range(mts_frames):
                th_frame = im_theta_zarr[t]
                s2 = np.nanmean(np.sin(2.0 * th_frame))
                c2 = np.nanmean(np.cos(2.0 * th_frame))
                if np.isnan(s2) or np.isnan(c2) or (s2 == 0 and c2 == 0):
                    thetas[t] = float(np.nanmean(th_frame)) if not np.isnan(np.nanmean(th_frame)) else 0.0
                else:
                    thetas[t] = 0.5 * np.arctan2(s2, c2)
            if mts_frames < num_frames:
                thetas[mts_frames:] = thetas[mts_frames - 1]
            print(f"Loaded global nematic angles from {theta_path.name} (mean theta = {np.mean(thetas):.3f} rad)")
            return thetas
        except Exception as e:
            print(f"[WARNING] Failed to load MTs_im_theta.zarr: {e}. Falling back to flow nematic axis.")

    # Fallback to flow data
    if flow_data is not None:
        print("Computing nematic angles from optical flow field orientation...")
        for t in range(num_frames):
            if channel_first:
                mx = flow_data[t, 0, ...]
                my = flow_data[t, 1, ...]
            else:
                mx = flow_data[t, ..., 0]
                my = flow_data[t, ..., 1]
            mag = np.hypot(mx, my)
            valid = mag > 1e-4
            if np.any(valid):
                phi = np.arctan2(my[valid], mx[valid])
                s2 = np.mean(np.sin(2.0 * phi))
                c2 = np.mean(np.cos(2.0 * phi))
                thetas[t] = 0.5 * np.arctan2(s2, c2)
            else:
                thetas[t] = 0.0

    return thetas


def main():
    parser = argparse.ArgumentParser(description="Calculate GPU-accelerated 2D-FFT angular spatial correlation around beads.")
    parser.add_argument('base_path', type=str, help='Path to directory containing beads_tracks.csv and GFP_flows.h5')
    parser.add_argument('--distances', nargs='+', default=['2:100:2', '120:500:20'],
                        help='Shell distances r (pixels) (default: 2:100:2 120:500:20)')
    parser.add_argument('--kernel_type', type=str, default='ring', choices=['ring', 'disk', 'gaussian'],
                        help="Kernel type: 'ring' (concentric shell), 'disk', or 'gaussian'")
    parser.add_argument('--shell_width', type=float, default=2.0,
                        help="Radial shell width dr (pixels) for 'ring' kernel (default: 2.0)")
    parser.add_argument('--particle_radius', type=int, default=0,
                        help='Radius around bead center to mask out flow (pixels, default: 0)')
    parser.add_argument('--roi_bbox', type=int, nargs=4, default=None,
                        help='Optional ROI bbox [xmin, xmax, ymin, ymax] for background flow correlation')
    parser.add_argument('--device', type=str, default=None,
                        help="Execution device: 'cuda', 'scipy', or None (auto)")
    parser.add_argument('--out_name', type=str, default='angular_correlation_w.zarr',
                        help='Output Zarr dataset filename (default: angular_correlation_w.zarr)')
    args = parser.parse_args()

    base_path = Path(args.base_path)
    csv_path = base_path / 'beads_tracks.csv'
    flow_path = base_path / 'GFP_flows.h5'

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}")
    if not flow_path.exists():
        raise FileNotFoundError(f"Missing {flow_path}")

    distances = parse_distances(args.distances)
    print(f"Distances to compute: {distances}")

    # 1. データの読み込み
    print(f"Loading tracks from {csv_path}...")
    df_tracks = pd.read_csv(csv_path)
    df_tracks = compute_bead_unit_velocities(df_tracks)
    grouped = df_tracks.groupby('frame')

    with h5py.File(str(flow_path), 'r') as f:
        k = list(f.keys())[0]
        flow_data = f[k]
        shape = flow_data.shape
        num_frames = shape[0]
        print(f"Flow data shape: {shape}")

        if shape[1] == 2:
            channel_first = True
            rows, cols = shape[2], shape[3]
        elif shape[3] == 2:
            channel_first = False
            rows, cols = shape[1], shape[2]
        else:
            rows, cols = shape[2], shape[3]
            channel_first = True

        active_frames = sorted([f for f in grouped.groups.keys() if f < num_frames])

        # Load global nematic director angle theta(t) for 1st & 2nd principal component decomposition
        thetas = load_nematic_thetas(base_path, num_frames, flow_data=flow_data, channel_first=channel_first)

        if args.roi_bbox is not None:
            roi_xmin, roi_xmax, roi_ymin, roi_ymax = args.roi_bbox
            roi_xmin = int(np.clip(roi_xmin, 0, cols - 1))
            roi_xmax = int(np.clip(roi_xmax, 0, cols - 1))
            roi_ymin = int(np.clip(roi_ymin, 0, rows - 1))
            roi_ymax = int(np.clip(roi_ymax, 0, rows - 1))

            roi_slice = (slice(roi_ymin, roi_ymax + 1), slice(roi_xmin, roi_xmax + 1))
            corr_roi_array = np.full((len(distances), num_frames), np.nan, dtype=np.float32)
            corr_roi_par_array = np.full((len(distances), num_frames), np.nan, dtype=np.float32)
            corr_roi_perp_array = np.full((len(distances), num_frames), np.nan, dtype=np.float32)
        else:
            roi_slice = None
            corr_roi_array = None
            corr_roi_par_array = None
            corr_roi_perp_array = None

        ds_particles = df_tracks.set_index(['frame', 'particle']).to_xarray()
        
        num_d = len(distances)
        num_fr = len(ds_particles.frame)
        num_p = len(ds_particles.particle)

        # Flow angular correlation arrays (Total, 1st PC / Parallel, 2nd PC / Perpendicular)
        corr_flow_array = np.full((num_d, num_fr, num_p), np.nan, dtype=np.float32)
        corr_flow_par_array = np.full((num_d, num_fr, num_p), np.nan, dtype=np.float32)
        corr_flow_perp_array = np.full((num_d, num_fr, num_p), np.nan, dtype=np.float32)

        # Bead-flow correlation arrays (Total, 1st PC / Parallel, 2nd PC / Perpendicular)
        corr_bead_array = np.full((num_d, num_fr, num_p), np.nan, dtype=np.float32)
        corr_bead_par_array = np.full((num_d, num_fr, num_p), np.nan, dtype=np.float32)
        corr_bead_perp_array = np.full((num_d, num_fr, num_p), np.nan, dtype=np.float32)

        frame_to_idx = {f: i for i, f in enumerate(ds_particles.frame.values)}
        particle_to_idx = {p: i for i, p in enumerate(ds_particles.particle.values)}

        # 2. FFTConvolver の初期化
        print(f"Initializing Fast FFT Convolver ({len(distances)} distances, kernel={args.kernel_type}, device={args.device or 'auto'})...")
        convolver = FFTConvolver(shape=(rows, cols), sizes=distances, kernel_type=args.kernel_type, shell_width=args.shell_width, device=args.device)
        print(f"Using backend: {convolver.device_type}")

        # 3. 実行ループ
        print(f"Calculating angular spatial correlation (Total, 1st PC Parallel, 2nd PC Perpendicular) for {len(active_frames)} frames...")
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

            th_t = thetas[t] if t < len(thetas) else 0.0

            res = convolver.convolve_and_sample_angular_correlation(
                m_ux=m_ux, m_uy=m_uy, valid_mask=valid_mask,
                y_idx=y_idx, x_idx=x_idx,
                center_flow_ux=center_flow_ux, center_flow_uy=center_flow_uy,
                b_ux=b_ux, b_uy=b_uy, theta=th_t, roi_slice=roi_slice
            )

            f_i = frame_to_idx[t]
            p_indices = [particle_to_idx[p] for p in p_ids]

            corr_flow_array[:, f_i, p_indices] = res['flow_total']
            corr_flow_par_array[:, f_i, p_indices] = res['flow_par']
            corr_flow_perp_array[:, f_i, p_indices] = res['flow_perp']

            corr_bead_array[:, f_i, p_indices] = res['bead_total']
            corr_bead_par_array[:, f_i, p_indices] = res['bead_par']
            corr_bead_perp_array[:, f_i, p_indices] = res['bead_perp']

            if roi_slice is not None:
                corr_roi_array[:, t] = res['roi_total']
                corr_roi_par_array[:, t] = res['roi_par']
                corr_roi_perp_array[:, t] = res['roi_perp']

    # 4. Consolidate into xarray and save to Zarr
    print("\nConsolidating particle correlation data into xarray...")
    ds_particles['angular_correlation'] = xr.DataArray(
        corr_flow_array,
        dims=['distance', 'frame', 'particle'],
        coords={'distance': distances, 'frame': ds_particles.frame, 'particle': ds_particles.particle}
    )
    ds_particles['angular_correlation_parallel'] = xr.DataArray(
        corr_flow_par_array,
        dims=['distance', 'frame', 'particle'],
        coords={'distance': distances, 'frame': ds_particles.frame, 'particle': ds_particles.particle}
    )
    ds_particles['angular_correlation_perpendicular'] = xr.DataArray(
        corr_flow_perp_array,
        dims=['distance', 'frame', 'particle'],
        coords={'distance': distances, 'frame': ds_particles.frame, 'particle': ds_particles.particle}
    )

    ds_particles['bead_correlation'] = xr.DataArray(
        corr_bead_array,
        dims=['distance', 'frame', 'particle'],
        coords={'distance': distances, 'frame': ds_particles.frame, 'particle': ds_particles.particle}
    )
    ds_particles['bead_correlation_parallel'] = xr.DataArray(
        corr_bead_par_array,
        dims=['distance', 'frame', 'particle'],
        coords={'distance': distances, 'frame': ds_particles.frame, 'particle': ds_particles.particle}
    )
    ds_particles['bead_correlation_perpendicular'] = xr.DataArray(
        corr_bead_perp_array,
        dims=['distance', 'frame', 'particle'],
        coords={'distance': distances, 'frame': ds_particles.frame, 'particle': ds_particles.particle}
    )

    theta_nematic_padded = np.full(len(ds_particles.frame), np.nan, dtype=np.float32)
    n_copy = min(len(thetas), len(ds_particles.frame))
    theta_nematic_padded[:n_copy] = thetas[:n_copy]

    ds_particles['theta_nematic'] = xr.DataArray(
        theta_nematic_padded,
        dims=['frame'],
        coords={'frame': ds_particles.frame}
    )

    ds_particles.attrs['description'] = 'Angular spatial correlation analysis: Particles (Total, Parallel/1st PC, Perpendicular/2nd PC)'
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
                ),
                'angular_correlation_parallel': xr.DataArray(
                    corr_roi_par_array,
                    dims=['distance', 'frame'],
                    coords={'distance': distances, 'frame': np.arange(num_frames)}
                ),
                'angular_correlation_perpendicular': xr.DataArray(
                    corr_roi_perp_array,
                    dims=['distance', 'frame'],
                    coords={'distance': distances, 'frame': np.arange(num_frames)}
                ),
                'theta_nematic': xr.DataArray(
                    thetas,
                    dims=['frame'],
                    coords={'frame': np.arange(num_frames)}
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
