import argparse
import logging
from pathlib import Path

import numpy as np
import zarr
from skimage import exposure
import dask
from dask.diagnostics import ProgressBar

import AFT_tools_v2 as AFT

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_args():
    parser = argparse.ArgumentParser(description='Calculate Alignment by Fourier Transform (AFT) for a given image stack.')
    parser.add_argument('base_path', type=str, 
                        help='Path to the base directory containing MTs.zarr')
    parser.add_argument('--zarr_path', type=str, default="MTs.zarr",
                        help='Path to the Zarr file containing the image stack (default: MTs.zarr)')
    parser.add_argument('--window_size_um', type=float, default=10.0, 
                        help='Length of microtubules in um for window size calculation (default: 10.0)')
    parser.add_argument('--scale', type=float, default=0.11, 
                        help='Pixel scale in um/pixel (default: 0.11)')
    parser.add_argument('--overlap', type=float, default=0.2, 
                        help='Overlap fraction for window scanning (default: 0.2)')
    parser.add_argument('--eccentricity_thresh', type=float, default=0.2, 
                        help='Threshold for eccentricity to filter orientational order (default: 0.2)')
    parser.add_argument('--neighborhood_radius', type=int, default=2, 
                        help='Neighborhood radius for order calculation (default: 2)')
    parser.add_argument('--n_jobs', type=int, default=-1, 
                        help='Number of parallel jobs to run. -1 means using all cores (default: -1)')
    parser.add_argument('--chunk_size', type=int, default=20, 
                        help='Number of frames to process in memory at once (default: 20)')
    return parser.parse_args()

@dask.delayed
def process_and_save_chunk(start_idx, end_idx, zarr_path, window_size, overlap, eccentricity_thresh, neighborhood_radius, theta_path, ecc_path, order_path):
    """Read a chunk from input Zarr, process AFT, and write directly to output Zarr arrays."""
    # 1. Open arrays inside the worker process
    green_in = zarr.open_array(str(zarr_path), read_only=True)
    im_theta_out = zarr.open_array(str(theta_path), mode='a')
    im_ecc_out = zarr.open_array(str(ecc_path), mode='a')
    order_out = zarr.open_array(str(order_path), mode='a')
    
    # Read the chunk into memory
    chunk_frames = green_in[start_idx:end_idx]
    
    if chunk_frames.size == 0:
        return True
        
    # 2. Histogram equalization for the chunk
    eq_chunk = np.empty_like(chunk_frames, dtype=np.uint8)
    for i in range(chunk_frames.shape[0]):
        frame_eq = exposure.equalize_hist(chunk_frames[i])
        eq_chunk[i] = (frame_eq * 255.0).astype(np.uint8)
        
    # 3. Calculate AFT for chunk
    # Force n_jobs=1 because Dask is already parallelizing over chunks
    x, y, u, v, im_theta, im_ecc = AFT.image_local_order(
        eq_chunk, 
        window_size=window_size, 
        overlap=overlap, 
        save_path='', 
        eccentricity_thresh=eccentricity_thresh,
        plot_overlay=False, 
        plot_angles=False, 
        plot_eccentricity=False, 
        save_figures=False,
        n_jobs=1
    )
    
    im_theta = np.array(im_theta)
    im_ecc = np.array(im_ecc)
    
    # 4. Calculate order parameter for chunk
    order_parameter = AFT.calculate_order_parameter(im_theta, neighborhood_radius)
    order_parameter = np.array(order_parameter)
    
    # 5. Write to Zarr
    im_theta_out[start_idx:end_idx] = im_theta
    im_ecc_out[start_idx:end_idx] = im_ecc
    order_out[start_idx:end_idx] = order_parameter
    
    return True

def main():
    args = parse_args()
    base_path = Path(args.base_path)

    if not base_path.exists():
        logging.error(f"Base path does not exist: {base_path}")
        return

    # 1. Open input Zarr
    zarr_path = base_path / args.zarr_path
    logging.info(f"Loading data from {zarr_path}")
    try:
        green = zarr.open_array(str(zarr_path), read_only=True)
    except Exception as e:
        logging.error(f"Failed to open {zarr_path}: {e}")
        return

    N_frames, N_rows, N_cols = green.shape
    window_size = int(args.window_size_um / args.scale)
    
    # Ensure window_size is odd as done in AFT_tools_v2.py
    if window_size % 2 == 0:
        window_size += 1
        
    logging.info(f"AFT Parameters -> window_size: {window_size} px, overlap: {args.overlap}, eccentricity_thresh: {args.eccentricity_thresh}")

    # 2. Calculate output shapes and initialize Zarr arrays
    radius = int(np.floor((window_size) / 2))
    rpos = np.arange(radius, N_rows - radius, int(window_size * args.overlap))
    cpos = np.arange(radius, N_cols - radius, int(window_size * args.overlap))
    
    out_shape_2d = (len(rpos), len(cpos))
    out_shape_3d = (N_frames, len(rpos), len(cpos))
    
    theta_path = base_path / "MTs_im_theta.zarr"
    ecc_path = base_path / "MTs_im_eccentricity.zarr"
    order_path = base_path / "MTs_order_parameter.zarr"
    
    logging.info("Initializing output Zarr arrays...")
    zarr.open_array(str(theta_path), mode='w', shape=out_shape_3d, dtype=np.float64, chunks=(1, out_shape_2d[0], out_shape_2d[1]))
    zarr.open_array(str(ecc_path), mode='w', shape=out_shape_3d, dtype=np.float64, chunks=(1, out_shape_2d[0], out_shape_2d[1]))
    zarr.open_array(str(order_path), mode='w', shape=(N_frames,), dtype=np.float64, chunks=(N_frames,))

    # 3. Create Dask tasks for chunk processing
    chunk_size = args.chunk_size
    tasks = []
    
    logging.info(f"Processing {N_frames} frames in chunks of {chunk_size}...")
    for start_idx in range(0, N_frames, chunk_size):
        end_idx = min(start_idx + chunk_size, N_frames)
        task = process_and_save_chunk(
            start_idx, end_idx, zarr_path, window_size, args.overlap, 
            args.eccentricity_thresh, args.neighborhood_radius, 
            theta_path, ecc_path, order_path
        )
        tasks.append(task)
        
    # 4. Execute Tasks
    import dask.config
    num_workers = args.n_jobs if args.n_jobs > 0 else None
    
    # Configure multiprocessing with dask
    with dask.config.set(scheduler='processes', num_workers=num_workers):
        with ProgressBar():
            dask.compute(*tasks)

    logging.info("Calculation and save completed successfully.")

if __name__ == "__main__":
    main()