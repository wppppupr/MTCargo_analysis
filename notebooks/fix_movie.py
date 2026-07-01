import json
import os

file_path = "/Users/sasakinozomu/code/MTCargo_analysis/notebooks/movie.ipynb"
with open(file_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

old_code = """# 軌跡
scatter = ax.scatter([], [], c=[], cmap=plasma_cmap,
                     vmin=0, vmax=1, s=20, animated=True)

# ==========================================
# 修正ポイント2: 軸と余白の完全な無効化（固定）
# ==========================================
ax.axis('off')
ax.set_position([0, 0, 1, 1]) # AxesをFigure全体に限界まで広げる
fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)

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

def update_frame(frame):
    im1.set_array(MTs[frame])
    im2.set_array(beads[frame])

    current_tracks = tracks_df[tracks_df['frame'] <= frame]

    if len(current_tracks) > 0:
        x = current_tracks['x'].values
        y = current_tracks['y'].values
        t = current_tracks['frame'].values / n_frames
        scatter.set_offsets(np.c_[x, y])
        scatter.set_array(t)

    return im1, im2, scatter"""

new_code = """from matplotlib.collections import LineCollection

# 軌跡を線で描画するためのLineCollection
lc = LineCollection([], cmap=plasma_cmap, norm=plt.Normalize(0, 1), animated=True, linewidths=2)
ax.add_collection(lc)

# ==========================================
# 修正ポイント2: 軸と余白の完全な無効化（固定）
# ==========================================
ax.axis('off')
ax.set_position([0, 0, 1, 1]) # AxesをFigure全体に限界まで広げる
fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)

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

def update_frame(frame):
    im1.set_array(MTs[frame])
    im2.set_array(beads[frame])

    current_tracks = tracks_df[tracks_df['frame'] <= frame]

    if len(current_tracks) > 0:
        segments = []
        colors = []
        for particle_id, group in current_tracks.groupby('particle'):
            x = group['x'].values
            y = group['y'].values
            t = group['frame'].values / n_frames
            
            pts = np.array([x, y]).T.reshape(-1, 1, 2)
            if len(pts) > 1:
                segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
                segments.extend(segs)
                colors.extend(t[1:])
                
        lc.set_segments(segments)
        lc.set_array(np.array(colors))

    return im1, im2, lc"""

modified = False
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        if "scatter = ax.scatter([], [], c=[], cmap=plasma_cmap," in source:
            if old_code in source:
                source = source.replace(old_code, new_code)
                cell["source"] = [line + "\n" for line in source.split("\n")]
                if cell["source"]:
                    cell["source"][-1] = cell["source"][-1].rstrip("\n")
                modified = True
                print("Replaced successfully.")
            else:
                print("Found scatter but old_code did not match exactly.")

if modified:
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print("Notebook saved.")
else:
    print("No modifications made.")
