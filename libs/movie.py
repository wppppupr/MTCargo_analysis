from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import os
import zarr
import numpy as np
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
from tqdm import tqdm  # 進捗バー用ライブラリ
import argparse

def create_movie(folder, output_name=None):
    # --- 設定 ---
    if output_name is None:
        output_name = os.path.join(folder, "tracking.mov")

    track_path = os.path.join(folder, "beads_tracks.csv")
    MTs_path = os.path.join(folder, "MTs.zarr")
    MTs_red_path = os.path.join(folder, "MTs_red.zarr")
    beads_path = os.path.join(folder, "beads.zarr")

    scale = 0.11

    # --- データ読み込み ---
    # メモリ節約のため Lazy Loading
    MTs = zarr.open_array(MTs_path, mode='r')
    MTs_red = zarr.open_array(MTs_red_path, mode='r')
    beads = zarr.open_array(beads_path, mode='r')

    # トラッキングデータ
    track = pd.read_csv(track_path)

    # --- 前処理: Pandas -> Numpy (高速化) ---
    print("Preprocessing tracking data...")
    particle_tracks = {}
    groups = track.groupby("particle")
    for p, group in groups:
        sorted_data = group.sort_values("frame")[["frame", "x", "y"]].values
        particle_tracks[p] = sorted_data

    total_frames = MTs.shape[0]

    # --- カラーマップ作成 ---
    colormap_data = {
        "MTs": [(0, 0, 0), (0, 1, 0)],
        "MTs_red": [(0, 0, 0, 0.0), (1, 0, 1)],
        "beads": [(0, 0, 0), (0, 1, 1)]
    }
    colormaps = {}
    for name, colors in colormap_data.items():
        colormaps[name] = LinearSegmentedColormap.from_list(name, colors)

    # --- プロット設定 ---
    width_px = 2560
    height_px = 2160
    dpi = 80
    figsize = (width_px / dpi, height_px / dpi)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # 背景初期化
    extent = [0, MTs.shape[2], 0, MTs.shape[1]]
    im1 = ax.imshow(MTs[0], cmap=colormaps['MTs'], interpolation='none',
                    aspect='auto', origin='lower', extent=extent)
    im2 = ax.imshow(MTs_red[0], cmap=colormaps['MTs_red'], interpolation='none',
                    aspect='auto', alpha=0.9, origin='lower', extent=extent)
    im3 = ax.imshow(beads[0], cmap=colormaps['beads'], interpolation='none',
                    aspect='auto', alpha=0.6, origin='lower', extent=extent)

    # Normalize設定
    norm = Normalize(vmin=0, vmax=total_frames)

    # LineCollectionとPoints初期化
    line_collections = {}
    points = {}

    for p in particle_tracks.keys():
        lc = LineCollection([], cmap="cubehelix", norm=norm, alpha=1.0, linewidths=15)
        line_collections[p] = lc
        ax.add_collection(lc)

        pt, = ax.plot([], [], 'x', color=(1.0, 75/255, 0, 1.0), alpha=0.7, markersize=10, markeredgewidth=3)
        points[p] = pt

    # 軸設定
    ax.set_xlim(0, MTs.shape[2])
    ax.set_ylim(0, MTs.shape[1])
    ax.axis('off')
    ax.set_position([0, 0, 1, 1])

    # スケールバー
    fontprops = fm.FontProperties(size=24)
    size_bar = AnchoredSizeBar(ax.transData,
                               size=50/scale,
                               label='',
                               loc=4,
                               pad=0.5,
                               color='white',
                               frameon=False,
                               size_vertical=20,
                               fontproperties=fontprops)
    ax.add_artist(size_bar)

    # --- アニメーション更新関数 ---
    def init():
        im1.set_data(MTs[0])
        im2.set_data(MTs_red[0])
        im3.set_data(beads[0])
        return [im1, im2, im3] + list(line_collections.values()) + list(points.values())

    def update(frame):
        # 画像更新 (Lazy Loading)
        im1.set_data(MTs[frame])
        im2.set_data(MTs_red[frame])
        im3.set_data(beads[frame])

        artists = [im1, im2, im3]

        # 粒子ごとの更新
        for p, data in particle_tracks.items():
            # 高速検索 (Numpy searchsorted)
            idx = np.searchsorted(data[:, 0], frame, side='right')
            history = data[:idx]

            if len(history) > 0:
                current_x, current_y = history[-1, 1], history[-1, 2]
                points[p].set_data([current_x], [current_y])
                artists.append(points[p])

            if len(history) > 1:
                pts = history[:, 1:3]
                # Vectorized segments creation
                segments = np.concatenate([pts[:-1, np.newaxis, :], pts[1:, np.newaxis, :]], axis=1)
                line_collections[p].set_segments(segments)
                line_collections[p].set_array(history[:-1, 0])
                artists.append(line_collections[p])

        return artists

    # --- 動画保存 (進捗バー付き) ---
    writer = FFMpegWriter(fps=10, metadata=dict(artist='Me'), bitrate=-1,
                          extra_args=['-vcodec', 'libx264', '-crf', '28', '-preset', 'superfast', '-pix_fmt', 'yuv420p'])

    ani = FuncAnimation(fig, update, frames=range(0, total_frames), init_func=init, blit=True)

    print(f"Start saving animation to: {output_name}")
    print(f"Total frames: {total_frames}")

    # tqdmを使った進捗バーの設定
    with tqdm(total=total_frames, unit="frame") as pbar:
        # コールバック関数: フレームが1つ処理されるたびに呼ばれる
        def progress_callback(current_frame, total_frames_in_save):
            pbar.update(1)

        ani.save(output_name, writer=writer, dpi=dpi, progress_callback=progress_callback)

    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create tracking movie.")
    parser.add_argument("--folder", type=str, default="/Volumes/data/Sasaki/MTsingleBeads/20260121/beads_trans_crop_crop",
                        help="Path to the data folder.")
    args = parser.parse_args()

    create_movie(args.folder)
