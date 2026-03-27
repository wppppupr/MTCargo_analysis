import argparse
import logging
import concurrent.futures
from pathlib import Path

import numpy as np
import zarr
from skimage import exposure

import AFT_tools_v2 as AFT

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_args():
    parser = argparse.ArgumentParser(description='Calculate Alignment by Fourier Transform (AFT) for a given image stack.')
    parser.add_argument('base_path', type=str, 
                        help='Path to the base directory containing MTs.zarr')
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
    return parser.parse_args()

def _equalize_frame(frame):
    """Apply histogram equalization to a single frame and scale to 8-bit [0, 255]."""
    frame_eq = exposure.equalize_hist(frame)
    return (frame_eq * 255.0).astype(np.uint8)

def process_equalization(green_zarr, max_workers=None):
    """Read frames from Zarr array and apply parallel histogram equalization."""
    logging.info("Equalizing histogram in parallel...")
    shape = green_zarr.shape
    eq_green = np.empty(shape, dtype=np.uint8)
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(_equalize_frame, green_zarr)
        for i, frame_proc in enumerate(results):
            eq_green[i] = frame_proc
            
    return eq_green

def main():
    args = parse_args()
    base_path = Path(args.base_path)

    if not base_path.exists():
        logging.error(f"Base path does not exist: {base_path}")
        return

    # 1. Load data from Zarr
    zarr_path = base_path / "MTs.zarr"
    logging.info(f"Loading data from {zarr_path}")
    try:
        green = zarr.open_array(str(zarr_path), read_only=True)
    except Exception as e:
        logging.error(f"Failed to open {zarr_path}: {e}")
        return

    # 2. Parallel histogram equalization
    max_workers = None if args.n_jobs <= 0 else args.n_jobs
    eq_green = process_equalization(green, max_workers)

    # 3. Calculate AFT parameters
    window_size = int(args.window_size_um / args.scale)
    logging.info(f"AFT Parameters -> window_size: {window_size} px, overlap: {args.overlap}, eccentricity_thresh: {args.eccentricity_thresh}")

    # 4. Compute AFT (image_local_order)
    logging.info("Calculating AFT (image_local_order)...")
    x, y, u, v, im_theta, im_eccentricity = AFT.image_local_order(
        eq_green, 
        window_size=window_size, 
        overlap=args.overlap, 
        save_path='', 
        eccentricity_thresh=args.eccentricity_thresh,
        plot_overlay=False, 
        plot_angles=False, 
        plot_eccentricity=False, 
        save_figures=False,
        n_jobs=args.n_jobs
    )

    im_theta = np.array(im_theta)
    im_eccentricity = np.array(im_eccentricity)

    order_parameter = AFT.calculate_order_parameter(im_theta, args.neighborhood_radius)
    order_parameter = np.array(order_parameter)

    # 5. Save results to Zarr
    logging.info("Saving results to Zarr...")
    
    theta_path = base_path / "MTs_im_theta.zarr"
    im_theta_zarr = zarr.open_array(str(theta_path), mode='w', shape=im_theta.shape, dtype=im_theta.dtype)
    im_theta_zarr[:] = im_theta

    ecc_path = base_path / "MTs_im_eccentricity.zarr"
    im_eccentricity_zarr = zarr.open_array(str(ecc_path), mode='w', shape=im_eccentricity.shape, dtype=im_eccentricity.dtype)
    im_eccentricity_zarr[:] = im_eccentricity

    order_path = base_path / "MTs_order_parameter.zarr"
    order_parameter_zarr = zarr.open_array(str(order_path), mode='w', shape=order_parameter.shape, dtype=order_parameter.dtype)
    order_parameter_zarr[:] = order_parameter

    logging.info("Calculation and save completed successfully.")

if __name__ == "__main__":
    main()