from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import os
import zarr
import numpy as np
import argparse
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
from tqdm import tqdm  # 進捗バー用ライブラリ

# --- 設定 ---
parser = argparse.ArgumentParser(description='Create movie from tracking data.')
parser.add_argument('folder', type=str, help='Path to the directory containing hdf5 and csv')
parser.add_argument('--tracks_csv', type=str, default='beads_tracks.csv', help='Trackpy csv file name')
parser.add_argument('--MTs_zarr', type=str, default='MTs.zarr', help='MTs zarr file name')
parser.add_argument('--MTs_red_zarr', type=str, default='MTs_red.zarr', help='MTs_red zarr file name')
parser.add_argument('--beads_zarr', type=str, default='beads.zarr', help='beads zarr file name')
parser.add_argument('--output_name', type=str, default='tracking.mov', help='Output movie file name')
parser.add_argument('--scale', type=float, default=0.11, help='Scale factor')
parser.add_argument('--vmax_prc', type=float, default=99.5, help='Global percentile for max intensity normalization (e.g., 99.5).')
parser.add_argument('--vmin_prc', type=float, default=0.5, help='Global percentile for min intensity normalization (e.g., 0.5).')
parser.add_argument('--vmax_prc_MTs', type=float, default=None, help='Percentile for max intensity of MTs.')
parser.add_argument('--vmin_prc_MTs', type=float, default=None, help='Percentile for min intensity of MTs.')
parser.add_argument('--vmax_prc_red', type=float, default=None, help='Percentile for max intensity of MTs_red.')
parser.add_argument('--vmin_prc_red', type=float, default=None, help='Percentile for min intensity of MTs_red.')
parser.add_argument('--vmax_prc_beads', type=float, default=None, help='Percentile for max intensity of beads.')
parser.add_argument('--vmin_prc_beads', type=float, default=None, help='Percentile for min intensity of beads.')
args = parser.parse_args()

folder = args.folder
output_name = os.path.join(folder, args.output_name)
track_path = os.path.join(folder, args.tracks_csv)
MTs_path = os.path.join(folder, args.MTs_zarr)
MTs_red_path = os.path.join(folder, args.MTs_red_zarr)
beads_path = os.path.join(folder, args.beads_zarr)

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

# --- 輝度の正規化のためのグローバル最小・最大値の計算 ---
print(f"Calculating global intensity percentiles...")
# サンプルフレームでパーセンタイルを推定 (全フレームだと時間がかかるため、最大50フレームで計算)
sample_frames = np.linspace(0, total_frames - 1, min(total_frames, 50), dtype=int)

def get_vmin_vmax(zarr_array, vmin_prc, vmax_prc):
    if zarr_array.shape[0] == 0:
        return 0, 1
    vmin = float('inf')
    vmax = float('-inf')
    for f in sample_frames:
        img = zarr_array[f]
        vmin = min(vmin, np.percentile(img, vmin_prc))
        vmax = max(vmax, np.percentile(img, vmax_prc))
    if vmax <= vmin:
        vmax = vmin + 1
    return vmin, vmax

MTs_vmin_prc = args.vmin_prc_MTs if args.vmin_prc_MTs is not None else args.vmin_prc
MTs_vmax_prc = args.vmax_prc_MTs if args.vmax_prc_MTs is not None else args.vmax_prc
MTs_vmin, MTs_vmax = get_vmin_vmax(MTs, MTs_vmin_prc, MTs_vmax_prc)

MTs_red_vmin_prc = args.vmin_prc_red if args.vmin_prc_red is not None else args.vmin_prc
MTs_red_vmax_prc = args.vmax_prc_red if args.vmax_prc_red is not None else args.vmax_prc
MTs_red_vmin, MTs_red_vmax = get_vmin_vmax(MTs_red, MTs_red_vmin_prc, MTs_red_vmax_prc)

beads_vmin_prc = args.vmin_prc_beads if args.vmin_prc_beads is not None else args.vmin_prc
beads_vmax_prc = args.vmax_prc_beads if args.vmax_prc_beads is not None else args.vmax_prc
beads_vmin, beads_vmax = get_vmin_vmax(beads, beads_vmin_prc, beads_vmax_prc)

print(f"MTs norm (vmin_prc={MTs_vmin_prc}, vmax_prc={MTs_vmax_prc}): min={MTs_vmin:.2f}, max={MTs_vmax:.2f}")
print(f"MTs_red norm (vmin_prc={MTs_red_vmin_prc}, vmax_prc={MTs_red_vmax_prc}): min={MTs_red_vmin:.2f}, max={MTs_red_vmax:.2f}")
print(f"beads norm (vmin_prc={beads_vmin_prc}, vmax_prc={beads_vmax_prc}): min={beads_vmin:.2f}, max={beads_vmax:.2f}")

# --- カラーマップ作成 ---
colormap_data = {
    "MTs": [(0, 0, 0), (119/255, 217/255, 168/255)],
    "MTs_red": [(0, 0, 0, 0.0), (255/255, 75/255, 0)],
    "beads": [(0, 0, 0), (136/255, 204/255, 238/255)]
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
                aspect='auto', origin='lower', extent=extent,
                vmin=MTs_vmin, vmax=MTs_vmax)
im2 = ax.imshow(MTs_red[0], cmap=colormaps['MTs_red'], interpolation='none', 
                aspect='auto', alpha=0.9, origin='lower', extent=extent,
                vmin=MTs_red_vmin, vmax=MTs_red_vmax)
im3 = ax.imshow(beads[0], cmap=colormaps['beads'], interpolation='none', 
                aspect='auto', alpha=0.6, origin='lower', extent=extent,
                vmin=beads_vmin, vmax=beads_vmax)

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