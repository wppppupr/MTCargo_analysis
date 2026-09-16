import argparse
import os
import shutil
import sys
import time
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
import xarray as xr
import zarr

current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))
if str(current_dir.parent) not in sys.path:
    sys.path.insert(0, str(current_dir.parent))

try:
    from libs.fft_convolution import FFTConvolver, FullField2DAutocorrelation
except ImportError:
    from fft_convolution import FFTConvolver, FullField2DAutocorrelation


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


def load_nematic_thetas(base_path, num_frames, flow_data=None, channel_first=True):
    """
    Load or compute the global nematic director angle theta(t) for each frame.
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
    parser = argparse.ArgumentParser(
        description='Calculate background / bulk MT flow angular spatial correlation (Random Points Control or Full-Field 2D FFT).'
    )
    parser.add_argument('base_path', type=str, help='Path to directory containing GFP_flows.h5 and beads_tracks.csv')
    parser.add_argument('--h5_file', type=str, default='GFP_flows.h5', help='H5 file name')
    parser.add_argument('--tracks_csv', type=str, default='beads_tracks.csv', help='Trackpy csv file name')
    parser.add_argument(
        '--method', type=str, default='random_points', choices=['random_points', 'full_field'],
        help="Calculation method: 'random_points' (Control points matching particle sampling, recommended) or 'full_field' (2D-FFT)."
    )
    parser.add_argument(
        '--n_random_points', type=int, default=20,
        help='Number of random virtual particle / control points per frame for random_points method (default: 20).'
    )
    parser.add_argument(
        '--distances', '--windows', dest='distances', type=str, nargs='+', default=['2:100:2', '120:500:20'],
        help='Distance ranges / radii in pixels. Accepts space/comma separated numbers, or start:stop:step'
    )
    parser.add_argument(
        '--kernel_type', type=str, default='ring', choices=['ring', 'disk', 'gaussian'],
        help="Kernel type for random_points: 'ring' (concentric shell), 'disk', or 'gaussian' (default: ring)."
    )
    parser.add_argument(
        '--shell_width', type=float, default=2.0,
        help='Width of annular shell for radial averaging in pixels (default: 2.0).'
    )
    parser.add_argument(
        '--particle_mask_radius', type=float, default=15.0,
        help='Mask radius around tracked particles in pixels to avoid sampling control points on particles (default: 15.0).'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducible virtual particle sampling (default: 42).'
    )
    parser.add_argument(
        '--out_name', type=str, default='angular_correlation_bg.zarr',
        help='Output zarr directory name (default: angular_correlation_bg.zarr)'
    )
    parser.add_argument(
        '--device', type=str, default=None, choices=['cuda', 'cpu', 'torch_cpu', 'scipy'],
        help='Compute backend (cuda/cpu/scipy). Default: auto-detect GPU.'
    )
    args = parser.parse_args()

    base_path = Path(args.base_path)
    csv_path = base_path / args.tracks_csv
    h5_path = base_path / args.h5_file

    if not h5_path.exists():
        print(f"Error: HDF5 file not found at {h5_path}")
        return

    distances = parse_distances(args.distances)
    num_d = len(distances)
    print(f"Distances to compute ({num_d} scales): {distances}")

    df_tracks = None
    if csv_path.exists():
        print(f"Loading tracks from {csv_path} for particle masking...")
        df_tracks = pd.read_csv(csv_path)

    rng = np.random.default_rng(args.seed)

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

        thetas = load_nematic_thetas(base_path, num_frames, flow_data=flow_data, channel_first=channel_first)

        if args.method == 'random_points':
            print(f"\n--- Method: Random Virtual Particle (Control Points) Sampling ---")
            print(f"Sampling {args.n_random_points} background points/frame avoiding particles within {args.particle_mask_radius} px...")

            # Initialize FFTConvolver identical to particle analysis
            convolver = FFTConvolver(
                shape=(rows, cols), sizes=distances,
                kernel_type=args.kernel_type, shell_width=args.shell_width, device=args.device
            )
            print(f"Using backend: {convolver.device_type}")

            n_pts = args.n_random_points
            corr_bg_array = np.full((num_d, num_frames, n_pts), np.nan, dtype=np.float32)
            corr_bg_par_array = np.full((num_d, num_frames, n_pts), np.nan, dtype=np.float32)
            corr_bg_perp_array = np.full((num_d, num_frames, n_pts), np.nan, dtype=np.float32)

            r_sq = args.particle_mask_radius**2
            yy, xx = np.ogrid[:rows, :cols]

            print(f"Calculating angular spatial correlation for {num_frames} frames...")
            for t in tqdm(range(num_frames)):
                if channel_first:
                    m_x = flow_data[t, 0, ...].astype(np.float32)
                    m_y = flow_data[t, 1, ...].astype(np.float32)
                else:
                    m_x = flow_data[t, ..., 0].astype(np.float32)
                    m_y = flow_data[t, ..., 1].astype(np.float32)

                v_mag = np.hypot(m_x, m_y)
                valid_mask = (v_mag > 1e-4)

                # Particle exclusion mask
                if df_tracks is not None:
                    df_f = df_tracks[df_tracks['frame'] == t]
                    if not df_f.empty:
                        y_pts = df_f['y'].values
                        x_pts = df_f['x'].values
                        for py, px in zip(y_pts, x_pts):
                            p_dist_sq = (yy - py)**2 + (xx - px)**2
                            valid_mask[p_dist_sq <= r_sq] = False

                with np.errstate(divide='ignore', invalid='ignore'):
                    m_ux = np.where(valid_mask, m_x / np.maximum(v_mag, 1e-6), 0.0).astype(np.float32)
                    m_uy = np.where(valid_mask, m_y / np.maximum(v_mag, 1e-6), 0.0).astype(np.float32)

                cand_y, cand_x = np.where(valid_mask)
                num_cand = len(cand_y)

                if num_cand > 0:
                    chosen_k = min(n_pts, num_cand)
                    chosen_idx = rng.choice(num_cand, size=chosen_k, replace=False)
                    sel_y = cand_y[chosen_idx]
                    sel_x = cand_x[chosen_idx]

                    c_ux = m_ux[sel_y, sel_x]
                    c_uy = m_uy[sel_y, sel_x]

                    th_t = thetas[t] if t < len(thetas) else 0.0

                    res = convolver.convolve_and_sample_angular_correlation(
                        m_ux=m_ux, m_uy=m_uy, valid_mask=None,
                        y_idx=sel_y, x_idx=sel_x,
                        center_flow_ux=c_ux, center_flow_uy=c_uy,
                        b_ux=c_ux, b_uy=c_uy,
                        theta=th_t, roi_slice=None
                    )

                    corr_bg_array[:, t, :chosen_k] = res['flow_total']
                    corr_bg_par_array[:, t, :chosen_k] = res['flow_par']
                    corr_bg_perp_array[:, t, :chosen_k] = res['flow_perp']

            print("\nConsolidating random control points data into xarray...")
            ds_bg = xr.Dataset(
                data_vars={
                    'angular_correlation': xr.DataArray(
                        corr_bg_array,
                        dims=['distance', 'frame', 'random_point'],
                        coords={'distance': distances, 'frame': np.arange(num_frames), 'random_point': np.arange(n_pts)}
                    ),
                    'angular_correlation_parallel': xr.DataArray(
                        corr_bg_par_array,
                        dims=['distance', 'frame', 'random_point'],
                        coords={'distance': distances, 'frame': np.arange(num_frames), 'random_point': np.arange(n_pts)}
                    ),
                    'angular_correlation_perpendicular': xr.DataArray(
                        corr_bg_perp_array,
                        dims=['distance', 'frame', 'random_point'],
                        coords={'distance': distances, 'frame': np.arange(num_frames), 'random_point': np.arange(n_pts)}
                    ),
                    'theta_nematic': xr.DataArray(
                        thetas,
                        dims=['frame'],
                        coords={'frame': np.arange(num_frames)}
                    )
                }
            )
            ds_bg.attrs['description'] = 'Random virtual particle (Control Points) angular spatial correlation analysis for Background MT flow'
            ds_bg.attrs['method'] = 'Random Points Control (FFT Convolver with Ring Kernel)'
            ds_bg.attrs['n_random_points'] = n_pts
            ds_bg.attrs['particle_mask_radius'] = args.particle_mask_radius
            ds_bg.attrs['kernel_type'] = args.kernel_type
            ds_bg.attrs['shell_width'] = args.shell_width
            ds_bg.attrs['distances'] = distances

        else:
            # Full-Field 2D FFT Autocorrelation (Legacy / Reference)
            print(f"\n--- Method: Full-Field 2D FFT Autocorrelation ---")
            corr_bg_array = np.full((num_d, num_frames), np.nan, dtype=np.float32)
            corr_bg_par_array = np.full((num_d, num_frames), np.nan, dtype=np.float32)
            corr_bg_perp_array = np.full((num_d, num_frames), np.nan, dtype=np.float32)

            correlator = FullField2DAutocorrelation(shape=(rows, cols), distances=distances, shell_width=args.shell_width, device=args.device)
            print(f"Using backend: {correlator.device_type}")

            print(f"Calculating full-field 2-point angular spatial correlation across {num_frames} frames...")
            for t in tqdm(range(num_frames)):
                if channel_first:
                    m_x = flow_data[t, 0, ...].astype(np.float32)
                    m_y = flow_data[t, 1, ...].astype(np.float32)
                else:
                    m_x = flow_data[t, ..., 0].astype(np.float32)
                    m_y = flow_data[t, ..., 1].astype(np.float32)

                v_mag = np.hypot(m_x, m_y)
                valid_mask = (v_mag > 1e-4)

                if df_tracks is not None:
                    df_f = df_tracks[df_tracks['frame'] == t]
                    if not df_f.empty:
                        y_pts = df_f['y'].values
                        x_pts = df_f['x'].values
                        r_sq = args.particle_mask_radius**2
                        yy, xx = np.ogrid[:rows, :cols]
                        for py, px in zip(y_pts, x_pts):
                            p_dist_sq = (yy - py)**2 + (xx - px)**2
                            valid_mask[p_dist_sq <= r_sq] = False

                with np.errstate(divide='ignore', invalid='ignore'):
                    m_ux = np.where(valid_mask, m_x / np.maximum(v_mag, 1e-6), 0.0).astype(np.float32)
                    m_uy = np.where(valid_mask, m_y / np.maximum(v_mag, 1e-6), 0.0).astype(np.float32)

                th_t = thetas[t] if t < len(thetas) else 0.0
                res = correlator.compute_frame(m_ux=m_ux, m_uy=m_uy, valid_mask=valid_mask.astype(np.float32), theta=th_t)

                corr_bg_array[:, t] = res['corr_total']
                corr_bg_par_array[:, t] = res['corr_par']
                corr_bg_perp_array[:, t] = res['corr_perp']

            print("\nConsolidating background data into xarray...")
            ds_bg = xr.Dataset(
                data_vars={
                    'angular_correlation': xr.DataArray(
                        corr_bg_array,
                        dims=['distance', 'frame'],
                        coords={'distance': distances, 'frame': np.arange(num_frames)}
                    ),
                    'angular_correlation_parallel': xr.DataArray(
                        corr_bg_par_array,
                        dims=['distance', 'frame'],
                        coords={'distance': distances, 'frame': np.arange(num_frames)}
                    ),
                    'angular_correlation_perpendicular': xr.DataArray(
                        corr_bg_perp_array,
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
            ds_bg.attrs['description'] = 'Full-field 2-point angular spatial correlation analysis for Background / Bulk MT flow'
            ds_bg.attrs['method'] = '2D FFT Wiener-Khinchin Autocorrelation with Radial Averaging'
            ds_bg.attrs['distances'] = distances
            ds_bg.attrs['shell_width'] = args.shell_width

    out_bg = base_path / args.out_name
    if out_bg.exists():
        shutil.rmtree(out_bg, ignore_errors=True)

    ds_bg.to_zarr(str(out_bg), mode='w', consolidated=False)
    print(f"Success! Background angular correlation saved to {out_bg}")


if __name__ == "__main__":
    main()
