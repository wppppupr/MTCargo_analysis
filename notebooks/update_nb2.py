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

# 描画する矢印のスケール（値が小さいほど矢印が長く描画される）
# scale_units='xy' とすることで、流れ場とビーズの変位の長さを比較可能にする
shared_scale = 0.5 

# 流れ場の向きを矢印でプロット（Quiver、白色）
ax.quiver(x_grid, y_grid, u_down, v_down, color='white', alpha=0.5, 
          angles='xy', scale_units='xy', scale=shared_scale, label='Optical Flow')

# ビーズの運動方向（フレーム0からフレームdtへの変位）を計算
dt = 1 # 何フレーム先との差分をとるか
bx, by, b_dx, b_dy = [], [], [], []

# フレーム0に存在する全パーティクルについて
for p in beads[beads['frame'] == 0]['particle'].unique():
    track = beads[beads['particle'] == p]
    f0_data = track[track['frame'] == 0]
    ft_data = track[track['frame'] == dt]
    
    if not f0_data.empty and not ft_data.empty:
        x0, y0 = f0_data['x'].values[0], f0_data['y'].values[0]
        xt, yt = ft_data['x'].values[0], ft_data['y'].values[0]
        
        bx.append(x0)
        by.append(y0)
        b_dx.append(xt - x0)
        b_dy.append(yt - y0)

# ビーズの運動方向をマゼンタの矢印で描画（速度場と区別）
if len(bx) > 0:
    ax.quiver(bx, by, b_dx, b_dy, color='magenta', width=0.005,
              angles='xy', scale_units='xy', scale=shared_scale, label=f'Bead Motion (dt={dt})')
    # 初期位置を点で強調
    ax.scatter(bx, by, color='magenta', s=15, zorder=5)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Optical Flow and Bead Motion Direction (Frame 0)')
ax.legend()

plt.tight_layout()
plt.show()
"""

# notebookの最後のセルを更新、または追加
if len(nb.cells) > 1 and nb.cells[-1].cell_type == 'code':
    nb.cells[-1].source = new_cell_source
else:
    new_cell = nbformat.v4.new_code_cell(source=new_cell_source)
    nb.cells.append(new_cell)

with open(nb_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)
