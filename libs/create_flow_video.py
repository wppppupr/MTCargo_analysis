"""
Visualize optical flow on GFP images using color-wheel (twilight) angle representation.
Matches the style used in notebooks/snapshot.ipynb.
"""

import os
import argparse
from pathlib import Path
import numpy as np
import zarr
import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.font_manager as fm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib.animation import FuncAnimation
from tqdm import tqdm


def get_cud_gfp_cmap():
    """Returns the CUD-compliant GFP colormap."""
    gfp_base = mcolors.to_rgb("#00A960")  # (0.0, 0.663, 0.376)
    return mcolors.LinearSegmentedColormap.from_list(
        "cud_gfp",
        [
            (0.0, (0.0, 0.0, 0.0, 1.0)),              # 輝度0: 完全透明な黒
            (0.7, (gfp_base[0], gfp_base[1], gfp_base[2], 1.0)), # 輝度70%: 純粋なCUD緑（半透明）
            (1.0, (0.6, 1.0, 0.75, 1.0))              # 輝度100%: 芯が光るエメラルドホワイト（不透明）
        ]
    )


def add_color_wheel_inset(ax, position=[0.82, 0.82, 0.16, 0.16], dpi=100):
    """
    Adds a twilight color wheel inset to the given axes.
    position: [left, bottom, width, height] in axes coordinate (0 to 1).
    """
    fig = ax.figure
    # Inset axes using polar projection
    inset_ax = fig.add_axes(position, projection='polar')
    
    # Image coordinates (East is 0, clockwise / downward is positive)
    inset_ax.set_theta_zero_location('E')
    inset_ax.set_theta_direction(-1)
    inset_ax.grid(False)
    
    theta = np.linspace(0, 2 * np.pi, 300)
    r = np.linspace(0.4, 1.0, 30)  # Donut shape
    T, R = np.meshgrid(theta, r)
    alpha = np.where(theta <= np.pi, theta, theta - 2 * np.pi)
    Alpha, _ = np.meshgrid(alpha, r)
    
    norm_angle = mcolors.Normalize(vmin=-np.pi, vmax=np.pi)
    inset_ax.pcolormesh(T, R, Alpha, cmap='twilight', norm=norm_angle, shading='auto')
    inset_ax.axis('off')
    inset_ax.patch.set_alpha(0.0)
    return inset_ax


def create_flow_video(
    data_dir,
    output_path=None,
    flow_h5_name="GFP_flows.h5",
    gfp_zarr_name="GFP.zarr",
    step=64,
    scale=0.15,
    threshold_percentile=98.0,
    mts_max_factor=0.3,
    pixel_size_um=0.11,
    scale_bar_um=50,
    fps=10,
    dpi=100,
    start_frame=0,
    end_frame=None,
    add_wheel=True,
    wheel_pos=[0.84, 0.84, 0.14, 0.14]
):
    data_dir = Path(data_dir)
    flow_file = data_dir / flow_h5_name
    gfp_file = data_dir / gfp_zarr_name

    if not flow_file.exists():
        raise FileNotFoundError(f"Flow file not found: {flow_file}")
    if not gfp_file.exists():
        raise FileNotFoundError(f"GFP zarr not found: {gfp_file}")

    if output_path is None:
        output_path = data_dir / "optical_flow_twilight.mp4"
    else:
        output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading MT zarr: {gfp_file}")
    MTs = zarr.open(gfp_file, mode='r')
    
    print(f"Loading flows: {flow_file}")
    with h5py.File(flow_file, 'r') as h5f:
        if 'flows' not in h5f:
            raise KeyError(f"'flows' dataset not found in {flow_file}")
        flows = h5f['flows']
        num_flows, _, H, W = flows.shape
        print(f"Flows shape: {flows.shape}, MTs shape: {MTs.shape}")

        total_frames = min(num_flows, MTs.shape[0])
        if end_frame is None or end_frame > total_frames:
            end_frame = total_frames
        
        frames_to_process = list(range(start_frame, end_frame))
        n_frames = len(frames_to_process)
        print(f"Processing {n_frames} frames ({start_frame} to {end_frame - 1}) at {fps} fps...")

        # Setup colormaps & normalization
        cud_gfp_cmap = get_cud_gfp_cmap()
        norm_angle = mcolors.Normalize(vmin=-np.pi, vmax=np.pi)

        # Setup Grid for Quiver
        y_grid, x_grid = np.mgrid[0:H:step, 0:W:step]

        # Initial frame computation
        f0 = frames_to_process[0]
        flow0 = flows[f0].astype(np.float32)
        u0 = flow0[0, ::step, ::step]
        v0 = flow0[1, ::step, ::step]
        mag0 = np.sqrt(u0**2 + v0**2)
        if threshold_percentile is not None and threshold_percentile < 100:
            thresh0 = np.percentile(mag0[~np.isnan(mag0)], threshold_percentile)
            mask0 = mag0 > thresh0
            u0[mask0] = np.nan
            v0[mask0] = np.nan
        angles0 = np.arctan2(v0, u0)

        # Setup Figure with exact aspect ratio
        fig, ax = plt.subplots(figsize=(W / dpi, H / dpi), dpi=dpi)
        fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
        ax.set_position([0, 0, 1, 1])
        ax.axis('off')

        # Background image
        im = ax.imshow(
            MTs[f0],
            cmap=cud_gfp_cmap,
            origin='upper',
            vmin=MTs[f0].min(),
            vmax=MTs[f0].max() * mts_max_factor
        )

        # Quiver plot
        q = ax.quiver(
            x_grid, y_grid, u0, v0, angles0,
            cmap='twilight', norm=norm_angle,
            angles='xy', scale_units='xy', scale=scale,
            width=0.0035,
            headwidth=3,
            headlength=5,
            headaxislength=5.0,
            linewidth=0.4,
            alpha=1.0
        )

        # Scale bar (50 um default)
        fontprops = fm.FontProperties(size=24)
        size_bar_px = scale_bar_um / pixel_size_um
        size_bar = AnchoredSizeBar(
            ax.transData,
            size=size_bar_px,
            label='',
            loc=3,  # lower left
            pad=0.5,
            color='white',
            frameon=False,
            size_vertical=20,
            fontproperties=fontprops
        )
        ax.add_artist(size_bar)

        # Optional color wheel inset
        if add_wheel:
            add_color_wheel_inset(ax, position=wheel_pos, dpi=dpi)

        # Progress bar setup
        pbar = tqdm(total=n_frames, desc="Generating Optical Flow Video")

        def update(frame_idx):
            f_num = frames_to_process[frame_idx]
            im.set_data(MTs[f_num])
            
            fl = flows[f_num].astype(np.float32)
            u = fl[0, ::step, ::step]
            v = fl[1, ::step, ::step]
            mag = np.sqrt(u**2 + v**2)
            if threshold_percentile is not None and threshold_percentile < 100:
                thresh = np.percentile(mag[~np.isnan(mag)], threshold_percentile)
                mask = mag > thresh
                u[mask] = np.nan
                v[mask] = np.nan
            angles = np.arctan2(v, u)

            q.set_UVC(u, v)
            q.set_array(angles.ravel())
            pbar.update(1)
            return [im, q]

        ani = FuncAnimation(fig, update, frames=n_frames, blit=False)
        ani.save(
            str(output_path),
            writer='ffmpeg',
            fps=fps,
            dpi=dpi,
            codec='libx264',
            extra_args=['-pix_fmt', 'yuv420p', '-crf', '18']
        )
        pbar.close()
        plt.close(fig)

    print(f"\nSuccessfully generated Optical Flow video:\n  -> {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Create optical flow video with twilight angle coloring and CUD-GFP background.")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to folder containing GFP.zarr and GFP_flows.h5")
    parser.add_argument('--output_path', type=str, default=None, help="Path for output mp4 video")
    parser.add_argument('--step', type=int, default=64, help="Grid step size (default: 64)")
    parser.add_argument('--scale', type=float, default=0.15, help="Quiver scale factor (default: 0.15)")
    parser.add_argument('--threshold', type=float, default=98.0, help="Percentile threshold for outlier removal (default: 98.0)")
    parser.add_argument('--max_factor', type=float, default=0.3, help="Max factor for GFP image brightness (default: 0.3)")
    parser.add_argument('--fps', type=int, default=10, help="Frames per second (default: 10)")
    parser.add_argument('--dpi', type=int, default=100, help="Rendering DPI (default: 100)")
    parser.add_argument('--start_frame', type=int, default=0, help="Start frame index (default: 0)")
    parser.add_argument('--end_frame', type=int, default=None, help="End frame index (default: None for all)")
    parser.add_argument('--no_wheel', action='store_true', help="Disable color wheel inset")

    args = parser.parse_args()

    create_flow_video(
        data_dir=args.data_dir,
        output_path=args.output_path,
        step=args.step,
        scale=args.scale,
        threshold_percentile=args.threshold,
        mts_max_factor=args.max_factor,
        fps=args.fps,
        dpi=args.dpi,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        add_wheel=not args.no_wheel
    )


if __name__ == '__main__':
    main()
