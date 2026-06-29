import nbformat

nb_path = '/home/sasaki/MTCargo_analysis/notebooks/snap_of_beads.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

new_cell_source = """\
# GFP_flows.h5からフレーム0の流れ場を取得
with h5py.File(flow, 'r') as f:
    # f['flows'] のshapeは (T, 2, H, W) 
    flow_data = f['flows'][0].astype(np.float32)

# 流れ場の大きさ(magnitude)を計算
# チャネル0と1がそれぞれ y(v), x(u) 方向のフローと仮定
v = flow_data[0]
u = flow_data[1]
magnitude = np.sqrt(u**2 + v**2)

# ベクトルを描画する場合はダウンサンプリングする
step = 64
y_grid, x_grid = np.mgrid[0:magnitude.shape[0]:step, 0:magnitude.shape[1]:step]
u_down = u[::step, ::step]
v_down = v[::step, ::step]

fig, ax = plt.subplots(figsize=(12, 10))

# 背景に流れ場の大きさをヒートマップとしてプロット
im = ax.imshow(magnitude, cmap='viridis', origin='upper', alpha=0.8)
plt.colorbar(im, ax=ax, label='Flow Magnitude')

# 流れ場の向きを矢印でプロット（Quiver）
ax.quiver(x_grid, y_grid, u_down, v_down, color='white', alpha=0.5, scale_units='xy')

# ビーズの軌跡をプロット
for p in beads['particle'].unique():
    track = beads[beads['particle'] == p]
    ax.plot(track['x'], track['y'], linewidth=1.5, alpha=0.8)

# 最初のフレームにおけるビーズの位置を点でプロット
frame0_beads = beads[beads['frame'] == 0]
ax.scatter(frame0_beads['x'], frame0_beads['y'], color='red', s=10, zorder=5, label='Beads at Frame 0')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Optical Flow (Frame 0) and Bead Tracks')
ax.legend()

plt.tight_layout()
plt.show()
"""

if len(nb.cells) > 1 and nb.cells[-1].cell_type == 'code' and not nb.cells[-1].source.strip():
    nb.cells[-1].source = new_cell_source
else:
    new_cell = nbformat.v4.new_code_cell(source=new_cell_source)
    nb.cells.append(new_cell)

with open(nb_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)
