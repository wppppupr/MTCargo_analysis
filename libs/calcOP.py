import argparse
import logging
from pathlib import Path

import numpy as np
import zarr

import AFT_tools_v2 as AFT

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_args():
    parser = argparse.ArgumentParser(description='Calculate Alignment by Fourier Transform (AFT) order parameter from im_theta.')
    parser.add_argument('base_path', type=str, 
                        help='Path to the base directory containing MTs_im_theta.zarr')
    parser.add_argument('--neighborhood_radius', type=int, default=2, 
                        help='Neighborhood radius for order calculation (default: 2)')
    parser.add_argument('--eccentricity_thresh', type=float, default=0.2, 
                        help='Threshold for eccentricity to filter orientational order (default: 0.2)')
    return parser.parse_args()


def main():
    args = parse_args()
    base_path = Path(args.base_path)

    if not base_path.exists():
        logging.error(f"Base path does not exist: {base_path}")
        return

    # 1. Load data from Zarr
    theta_path = base_path / "MTs_im_theta.zarr"
    if not theta_path.exists():
        logging.error(f"Theta Zarr does not exist: {theta_path}")
        return
    logging.info(f"Loading theta from {theta_path}...")
    im_theta_zarr = zarr.open_array(str(theta_path), mode='r')
    im_theta = im_theta_zarr[:]

    # 2. Check and load eccentricity Zarr if available
    ecc_path = base_path / "MTs_im_eccentricity.zarr"
    im_ecc = None
    if ecc_path.exists():
        logging.info(f"Loading eccentricity from {ecc_path} (threshold={args.eccentricity_thresh})...")
        im_ecc_zarr = zarr.open_array(str(ecc_path), mode='r')
        im_ecc = im_ecc_zarr[:]
    else:
        logging.warning(f"Eccentricity Zarr not found at {ecc_path}. Proceeding without additional eccentricity mask.")

    # 3. Calculate order parameter with eccentricity filtering
    logging.info(f"Calculating order parameter (neighborhood_radius={args.neighborhood_radius}, eccentricity_thresh={args.eccentricity_thresh})...")
    order_parameter = AFT.calculate_order_parameter(
        im_theta, 
        args.neighborhood_radius,
        im_ecc_stack=im_ecc,
        eccentricity_thresh=args.eccentricity_thresh
    )
    order_parameter = np.array(order_parameter)

    valid_vals = order_parameter[~np.isnan(order_parameter)]
    if len(valid_vals) > 0:
        logging.info(f"Calculated Order Parameter: Mean = {np.mean(valid_vals):.4f}, Median = {np.median(valid_vals):.4f}, Std = {np.std(valid_vals):.4f} (N={len(valid_vals)})")
    else:
        logging.warning("All calculated order parameter values are NaN.")

    # 4. Save results to Zarr
    order_path = base_path / "MTs_order_parameter.zarr"
    logging.info(f"Saving results to {order_path}...")
    order_parameter_zarr = zarr.open_array(str(order_path), mode='w', shape=order_parameter.shape, dtype=order_parameter.dtype)
    order_parameter_zarr[:] = order_parameter

    logging.info("Calculation and save completed successfully.")

if __name__ == "__main__":
    main()