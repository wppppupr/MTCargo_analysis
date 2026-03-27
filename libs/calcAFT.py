import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import zarr
from skimage import exposure
from pathlib import Path
from tqdm import tqdm
import AFT_tools_v2 as AFT
import concurrent.futures

def _equalize_frame(frame):
    frame_eq = exposure.equalize_hist(frame)
    return (frame_eq * 255.0).astype(np.uint8)

# コマンドライン引数の設定
parser = argparse.ArgumentParser(description='Calculate AFT.')
parser.add_argument('base_path', type=str, help='Path to the base directory containing MTs.zarr')
args = parser.parse_args()

# pathlib によるパスの定義
base_path = Path(args.base_path)

print('loading data from', base_path)

green = zarr.open_array(str(base_path / "MTs.zarr"), read_only=True)

print('equalizing histogram in parallel...')
# equalize hist
eq_green = np.empty(green.shape, dtype=np.uint8)
    
with concurrent.futures.ProcessPoolExecutor() as executor:
    results = executor.map(_equalize_frame, green)
    for i, frame_proc in enumerate(results):
        eq_green[i] = frame_proc

# AFT parameters
window_size_um = 10 # MTs length = 10um
frame = 100

scale = 0.11  # um/pixel

#### required parameters ####
window_size = int(window_size_um/scale)
overlap = 0.2
neighborhood_radius = 1

d = 30

print("Calculating AFT...")

x, y, u, v, im_theta, im_eccentricity = AFT.image_local_order(
    eq_green[:,:,:], window_size, overlap, save_path='', eccentricity_thresh=0.2,
    plot_overlay=False, plot_angles=False, plot_eccentricity=False, save_figures=False,
    n_jobs=-1
)

im_theta = np.array(im_theta)
im_eccentricity = np.array(im_eccentricity)

print('save data to zarr...')

im_theta_zarr = zarr.open_array(str(base_path / "im_theta.zarr"), mode='w', shape=im_theta.shape, dtype=im_theta.dtype)
im_theta_zarr[:] = im_theta

im_eccentricity_zarr = zarr.open_array(str(base_path / "im_eccentricity.zarr"), mode='w', shape=im_eccentricity.shape, dtype=im_eccentricity.dtype)
im_eccentricity_zarr[:] = im_eccentricity

print('done')