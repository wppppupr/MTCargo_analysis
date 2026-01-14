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
from tqdm import tqdm

# ★追加: 大津の二値化用ライブラリ
from skimage.filters import threshold_otsu

# --- 設定 ---
folder = '/Volumes/My Passport/Sasaki/MTsingleBeads/20260107'
output_name = os.path.join(folder, "tracking_otsu.mov") # ファイル名変更
track_path = os.path.join(folder, "beads_tracks.csv")
MTs_path = os.path.join(folder, "MTs.zarr")
beads_path = os.path.join(folder, "beads.zarr")

scale = 0.11

# --- データ読み込み ---
MTs = zarr.open_array(MTs_path, mode='r')
beads = zarr.open_array(beads_path, mode='r')
track = pd.read_csv(track_path)

# --- 前処理 ---
print("Preprocessing tracking data...")
particle_tracks = {}
groups = track.groupby("particle")
for p, group in groups:
    sorted_data = group.sort_values("frame")[["frame", "x", "y"]].values
    particle_tracks[p] = sorted_data

total_frames = MTs.shape[0]

# --- カラーマップ作成 ---
colormap_data = {
    "MTs": [(0, 0, 0), (119/255, 217/255, 168/255)],
    # beads: 0(False)は透明、1(True)は赤
    "beads": [(0, 0, 0, 0.0), (255/255, 75/255, 0, 1.0)]
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
# beadsは二値化されるので vmin=0, vmax=1 に固定すると安定します
im2 = ax.imshow(beads[0], cmap=colormaps['beads'], interpolation='none', 
                aspect='auto', alpha=0.4, origin='lower', extent=extent, vmin=0, vmax=1)

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
size_bar = AnchoredSizeBar(ax.transData, size=50/scale, label='', loc=4, pad=0.5, 
                           color='white', frameon=False, size_vertical=20, fontproperties=fontprops)
ax.add_artist(size_bar)

# --- アニメーション更新関数 ---

def apply_otsu(image_data):
    """大津の二値化を適用するヘルパー関数"""
    try:
        # 【高速化】画像を間引いて(1/16のサイズで)閾値を計算
        # 画像全体のヒストグラム形状は間引いても変わらないため、精度を保ったまま高速化できます
        thresh = threshold_otsu(image_data[::4, ::4])
        
        # 閾値以上をTrue(1), 以下をFalse(0)にする
        binary_img = image_data > thresh
        return binary_img
    except ValueError:
        # 画像が真っ黒などで閾値が計算できない場合のエラー回避
        return np.zeros_like(image_data, dtype=bool)

def init():
    im1.set_data(MTs[0])
    # 初期フレームも二値化
    im2.set_data(apply_otsu(beads[0]))
    return [im1, im2] + list(line_collections.values()) + list(points.values())

def update(frame):
    # 画像更新
    im1.set_data(MTs[frame])
    
    # ★ここで大津の二値化を適用
    raw_beads = beads[frame]
    binary_beads = apply_otsu(raw_beads)
    im2.set_data(binary_beads)

    artists = [im1, im2]

    # 粒子ごとの更新 (変更なし)
    for p, data in particle_tracks.items():
        idx = np.searchsorted(data[:, 0], frame, side='right')
        history = data[:idx]

        if len(history) > 0:
            current_x, current_y = history[-1, 1], history[-1, 2]
            points[p].set_data([current_x], [current_y])
            artists.append(points[p])
        
        if len(history) > 1:
            pts = history[:, 1:3]
            segments = np.concatenate([pts[:-1, np.newaxis, :], pts[1:, np.newaxis, :]], axis=1)
            line_collections[p].set_segments(segments)
            line_collections[p].set_array(history[:-1, 0])
            artists.append(line_collections[p])

    return artists

# --- 動画保存 ---
writer = FFMpegWriter(fps=10, metadata=dict(artist='Me'), bitrate=-1, 
                      extra_args=['-vcodec', 'libx264', '-crf', '28', '-preset', 'superfast', '-pix_fmt', 'yuv420p'])

ani = FuncAnimation(fig, update, frames=range(0, total_frames), init_func=init, blit=True)

print(f"Start saving animation to: {output_name}")
with tqdm(total=total_frames, unit="frame") as pbar:
    def progress_callback(current_frame, total_frames_in_save):
        pbar.update(1)
    ani.save(output_name, writer=writer, dpi=dpi, progress_callback=progress_callback)

print("Done.")