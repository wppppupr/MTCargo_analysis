import os
import glob
import argparse
import h5py
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from cv2 import equalizeHist
from tqdm import tqdm
from matplotlib.animation import FuncAnimation

def hex_to_bgr(hex_color):
    """Converts a hex color string to a BGR tuple for OpenCV."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0))

def main():
    parser = argparse.ArgumentParser(description="Visualize optical flow as a video overlay on original images.")
    parser.add_argument('--image_dir', type=str, required=True, help="Directory containing original TIF images.")
    parser.add_argument('--h5_path', type=str, required=True, help="Path to the HDF5 file containing dense flows.")
    parser.add_argument('--output_video', type=str, default='output_flow.mp4', help="Output video path (.mp4).")
    parser.add_argument('--step', type=int, default=32, help="Grid step size for drawing flow arrows.")
    parser.add_argument('--scale', type=float, default=4.0, help="Scaling factor for flow arrows.")
    parser.add_argument('--fps', type=int, default=10, help="Frames per second for output video.")
    parser.add_argument('--arrow_color', type=str, default='#FF00FF', help="Hex color for flow arrows.")
    parser.add_argument('--thickness', type=int, default=2, help="Thickness of flow arrows.")
    args = parser.parse_args()

    plt.style.use('libs/my_style.mplstyle')

    img_folder = Path(args.image_dir)

    image_paths = list(sorted(img_folder.glob('frame_*.tif')))
    if len(image_paths) == 0:
        raise ValueError(f"No TIF images found in {args.image_dir}")

    print(image_paths[0])

    # equalize histogram
    eqimgs = []
    for img_path in tqdm(image_paths, desc="Equalizing histograms"):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        eqimg = equalizeHist(img)
        eqimgs.append(eqimg)

    print(f"Opening HDF5 file: {args.h5_path}")
    with h5py.File(args.h5_path, 'r') as h5f:
        if 'flows' not in h5f:
            raise KeyError("Dataset 'flows' not found in HDF5 file.")
        
        flows = h5f['flows']
        num_pairs, C, H, W = flows.shape
        print(f"Flow data shape: {flows.shape}")

        actual_pairs = min(num_pairs, len(image_paths) - 1)
        if len(image_paths) < num_pairs + 1:
            print(f"Warning: Number of images ({len(image_paths)}) is less than necessary. Processing {actual_pairs} pairs.")

        # カスタムカラーマップの作成: 値がゼロ（何もない場所）で黒になる
        cmap_green = LinearSegmentedColormap.from_list('black_green', ['black', '#00FF00'])

        fig, ax = plt.subplots()
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax.axis('off')
        
        im = ax.imshow(eqimgs[0], cmap=cmap_green, vmin=0, vmax=255)
        
        step = args.step
        Y, X = np.mgrid[0:H:step, 0:W:step]
        
        # C=0 is usually U (horizontal), C=1 is V (vertical) flow
        U_init = flows[0, 0, 0:H:step, 0:W:step]
        V_init = flows[0, 1, 0:H:step, 0:W:step]
        
        quiv = ax.quiver(X, Y, U_init, V_init, color=args.arrow_color, scale_units='xy', angles='xy', scale=1.0/args.scale, alpha=0.6)

        def update(frame):
            im.set_data(eqimgs[frame])
            u = flows[frame, 0, 0:H:step, 0:W:step]
            v = flows[frame, 1, 0:H:step, 0:W:step]
            quiv.set_UVC(u, v)
            return [im, quiv]

        ani = FuncAnimation(fig, update, frames=tqdm(range(actual_pairs), desc="Creating animation"), interval=1000/args.fps, blit=True)
        ani.save(args.output_video, writer='ffmpeg', fps=args.fps)

    print(f"Saved video to {args.output_video}")

if __name__ == '__main__':
    main()
