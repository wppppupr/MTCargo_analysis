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
    parser.add_argument('--neighborhood_radius', type=int, default=2, 
                        help='Neighborhood radius for order calculation (default: 2)')
    return parser.parse_args()


def main():
    args = parse_args()
    base_path = Path(args.base_path)

    if not base_path.exists():
        logging.error(f"Base path does not exist: {base_path}")
        return

    # 1. Load data from Zarr
    theta_path = base_path / "MTs_im_theta.zarr"
    im_theta_zarr = zarr.open_array(str(theta_path), mode='r')
    im_theta = im_theta_zarr[:]

    order_parameter = AFT.calculate_order_parameter(im_theta, args.neighborhood_radius)
    order_parameter = np.array(order_parameter)

    # 5. Save results to Zarr
    logging.info("Saving results to Zarr...")

    order_path = base_path / "MTs_order_parameter.zarr"
    order_parameter_zarr = zarr.open_array(str(order_path), mode='w', shape=order_parameter.shape, dtype=order_parameter.dtype)
    order_parameter_zarr[:] = order_parameter

    logging.info("Calculation and save completed successfully.")

if __name__ == "__main__":
    main()