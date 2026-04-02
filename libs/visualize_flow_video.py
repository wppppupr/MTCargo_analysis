import os
import glob
import argparse
import h5py
import cv2
import numpy as np
from tqdm import tqdm

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
    parser.add_argument('--scale', type=float, default=2.0, help="Scaling factor for flow arrows.")
    parser.add_argument('--fps', type=int, default=10, help="Frames per second for output video.")
    parser.add_argument('--bg_color', type=str, default='#117733', help="Hex color for tinting original images.")
    parser.add_argument('--arrow_color', type=str, default='#CC6677', help="Hex color for flow arrows.")
    parser.add_argument('--thickness', type=int, default=2, help="Thickness of flow arrows.")
    parser.add_argument('--vmax_prc', type=float, default=90, help="Percentile for max intensity normalization (e.g., 99.5).")
    parser.add_argument('--vmin_prc', type=float, default=5, help="Percentile for min intensity normalization (e.g., 0.1).")
    
    args = parser.parse_args()

    bg_bgr = np.array(hex_to_bgr(args.bg_color), dtype=np.float32)
    arrow_bgr = hex_to_bgr(args.arrow_color)

    image_paths = sorted(glob.glob(os.path.join(args.image_dir, '*.tif')))
    if len(image_paths) == 0:
        raise ValueError(f"No TIF images found in {args.image_dir}")

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

        W_even = (W // 2) * 2
        H_even = (H // 2) * 2

        # コーデック設定
        # Macでは 'avc1' がベストですが、失敗する場合は 'mp4v' に戻してください
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(args.output_video, fourcc, args.fps, (W_even, H_even))

        # --- Find global min and max for normalization ---
        print(f"Calculating global min ({args.vmin_prc}th prc) and max ({args.vmax_prc}th prc) intensity for normalization...")
        global_min = float('inf')
        global_max = float('-inf')
        for idx in tqdm(range(actual_pairs), desc="Finding min/max"):
            img = cv2.imread(image_paths[idx], cv2.IMREAD_UNCHANGED)
            if img is not None:
                # Use percentiles to ignore bright outliers (dust/noise) which makes the rest too faint
                global_min = min(global_min, np.percentile(img, args.vmin_prc))
                global_max = max(global_max, np.percentile(img, args.vmax_prc))
        
        print(f"Global min: {global_min}, Global max: {global_max}")
        if global_max <= global_min:
            global_max = global_min + 1

        # --- Optimize memory: Use a lookup table (LUT) for tinting instead of float arrays ---
        lut = np.zeros((256, 3), dtype=np.uint8)
        for val in range(256):
            lut[val] = (val / 255.0 * bg_bgr).astype(np.uint8)

        # Pre-allocate buffer for flow vector reading to avoid h5py memory leaks
        flow_buffer = np.empty((1, 2, H, W), dtype=flows.dtype)

        for i in tqdm(range(actual_pairs), desc="Generating Video"):
            img = cv2.imread(image_paths[i], cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"Warning: Could not read image {image_paths[i]}")
                continue
            
            # Normalize using global min/max
            img = img.astype(np.float32)
            img = (img - global_min) / (global_max - global_min)
            img = np.clip(img * 255, 0, 255).astype(np.uint8)

            if (img.shape[0] != H) or (img.shape[1] != W):
                img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)

            # Fast tinting using numpy advanced indexing (very low memory footprint)
            # This directly creates a (H, W, 3) uint8 array representing the tinted image
            tinted_img = lut[img].copy()

            # Read flow directly into pre-allocated buffer to avoid memory leaks
            flows.read_direct(flow_buffer, np.s_[i:i+1], np.s_[:])
            flow = flow_buffer[0] # shape: (2, H, W) -> [dx, dy]

            for y in range(0, H, args.step):
                for x in range(0, W, args.step):
                    dx = flow[0, y, x]
                    dy = flow[1, y, x]
                    
                    if abs(dx) > 0.1 or abs(dy) > 0.1:
                        end_px = int(x + dx * args.scale)
                        end_py = int(y + dy * args.scale)
                        cv2.arrowedLine(
                            tinted_img,
                            (x, y),
                            (end_px, end_py),
                            arrow_bgr,
                            args.thickness,
                            tipLength=0.3
                        )
            
            out.write(tinted_img)

            # Explicitly delete frame-bound references to encourage garbage collection
            del img, tinted_img, flow

        out.release()
    print(f"Saved video to {args.output_video}")

if __name__ == '__main__':
    main()
