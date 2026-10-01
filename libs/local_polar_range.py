import numpy as np
import pandas as pd
import zarr
import os
from tqdm import tqdm

####################################################
# Configuration (Example)
# You should change these paths to your data location
FILE_PATH = '/media/sasaki/myssd/Sasaki/MTsingleBeads/20260121/beads_trans_crop_crop'
scale = 0.11
####################################################

def local_polar_range(flow_array, tracks, ranges, scale):
    """
    Calculate local polar order parameter for multiple ranges (window sizes).

    Args:
        flow_array: Optical flow array (zarr or numpy), shape (T, H, W, 2)
        tracks: pandas DataFrame with columns 'x', 'y', 'frame'
        ranges: list of window sizes (in physical units, same units as scale determines)
        scale: conversion factor (physical units per pixel)

    Returns:
        tracks DataFrame with added columns for each range.
    """
    tracks_out = tracks.copy()

    # Pre-compute common data
    xc_all = tracks['x'].to_numpy().astype(np.int16)
    yc_all = tracks['y'].to_numpy().astype(np.int16)
    frames_all = tracks['frame'].to_numpy().astype(np.int16)
    unique_frames = np.unique(frames_all)

    # Image dimensions
    if isinstance(flow_array, list):
        max_h, max_w = flow_array[0].shape[0], flow_array[0].shape[1]
    else:
        max_h, max_w = flow_array.shape[1], flow_array.shape[2]

    # Prepare grids for all ranges
    grids = {}
    for r in ranges:
        window_size = r / scale
        h_half = int(window_size / 2)
        w_half = int(window_size / 2)
        dy = np.arange(-h_half, h_half)
        dx = np.arange(-w_half, w_half)
        grids[r] = np.meshgrid(dy, dx, indexing='ij')

    # Prepare result arrays
    Ps_dict = {r: np.full(len(tracks), np.nan) for r in ranges}

    print(f"Processing {len(unique_frames)} frames for {len(ranges)} ranges: {ranges}...")

    # Process only frames > 0 because we need flow from previous frame
    # Assuming flow_array is aligned such that flow_array[t-1] corresponds to flow at frame t
    # (or between t-1 and t)

    frames_to_process = [t for t in unique_frames if t > 0]

    for t in tqdm(frames_to_process):
        # 1. Get points in this frame
        mask = (frames_all == t)
        xs_t = xc_all[mask]
        ys_t = yc_all[mask]

        if len(xs_t) == 0:
            continue

        # 2. Get flow for this frame
        if t-1 >= flow_array.shape[0]:
             # This might happen if tracks go beyond flow calculation
             continue

        current_flow = flow_array[t-1]

        # 3. Process each range
        for r in ranges:
            grid_y, grid_x = grids[r]

            # (N_points, window_h, window_w)
            idx_y = ys_t[:, None, None] + grid_y[None, :, :]
            idx_x = xs_t[:, None, None] + grid_x[None, :, :]

            # Clip coordinates
            idx_y = np.clip(idx_y, 0, max_h - 1)
            idx_x = np.clip(idx_x, 0, max_w - 1)

            # Extract flow vectors
            batch_flows = current_flow[idx_y, idx_x] # shape: (N_points, h, w, 2)

            batch_xf = batch_flows[..., 0]
            batch_yf = batch_flows[..., 1]

            magnitude = np.sqrt(batch_xf**2 + batch_yf**2)

            # Mean calculation
            mean_xf = np.mean(batch_xf, axis=(1, 2))
            mean_yf = np.mean(batch_yf, axis=(1, 2))

            mean_mag_vectors = np.mean(magnitude, axis=(1, 2))
            mag_mean_vector = np.sqrt(mean_xf**2 + mean_yf**2)

            # Calculate P (P = |<v>| / <|v|>)
            with np.errstate(divide='ignore', invalid='ignore'):
                P_vals = mag_mean_vector / mean_mag_vectors

            # Store results
            Ps_dict[r][mask] = P_vals

    print("Calculation complete.")

    # Add to dataframe
    for r in ranges:
        tracks_out[f'local_P_{r}'] = Ps_dict[r]

    return tracks_out

if __name__ == "__main__":

    print(f"Loading data from {FILE_PATH}...")

    # Check if paths exist
    if os.path.exists(FILE_PATH):
        flow_path = os.path.join(FILE_PATH, "green_flow.zarr")
        tracks_path = os.path.join(FILE_PATH, "beads_tracks.csv")

        if os.path.exists(flow_path) and os.path.exists(tracks_path):
            try:
                flow_array = zarr.open_array(flow_path, mode='r')
                tracks = pd.read_csv(tracks_path)

                # Example ranges: You can modify this list as needed
                # These are the side lengths of the square window in physical units
                ranges = [2, 5, 10, 20]

                print(f"Calculating local polar parameters for ranges: {ranges}")
                tracks_new = local_polar_range(flow_array, tracks, ranges, scale)

                output_path = os.path.join(FILE_PATH, "beads_tracks_with_local_P_ranges.csv")
                tracks_new.to_csv(output_path, index=False)
                print(f"Saved tracks with local P to {output_path}")
            except Exception as e:
                print(f"An error occurred: {e}")
        else:
            if not os.path.exists(flow_path):
                print(f"Flow file not found: {flow_path}")
            if not os.path.exists(tracks_path):
                print(f"Tracks file not found: {tracks_path}")
    else:
        print(f"Directory not found: {FILE_PATH}")
