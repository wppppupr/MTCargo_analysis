import re
import sys

# 1. calc_local_polar_gaussian.py
f = 'calc_local_polar_gaussian.py'
with open(f, 'r') as file:
    content = file.read()

content = content.replace("parser.add_argument('--windows', type=str", "parser.add_argument('--sigmas', type=str")
content = content.replace("help='Window sizes.", "help='Gaussian sigmas.")
content = content.replace("for arg_w in args.windows:", "for arg_w in args.sigmas:")
content = content.replace("out_particle = base_path / \"local_polar_w.zarr\"", "out_particle = base_path / \"local_polar_w_gaussian.zarr\"")
content = content.replace("out_roi = base_path / \"local_polar_flow_roi.zarr\"", "out_roi = base_path / \"local_polar_flow_roi_gaussian.zarr\"")

kernel_old = """                kernel_size = 2 * size + 1
                kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
                center = size
                ky, kx = np.ogrid[:kernel_size, :kernel_size]
                kmask = (kx - center)**2 + (ky - center)**2 <= size**2
                kernel[kmask] = 1.0
                kernel /= kernel.sum()"""

kernel_new = """                sigma = float(size)
                kernel_size = int(np.ceil(6 * sigma))
                if kernel_size % 2 == 0: kernel_size += 1
                k1d = cv2.getGaussianKernel(kernel_size, sigma)
                kernel = (k1d @ k1d.T).astype(np.float32)"""
content = content.replace(kernel_old, kernel_new)
content = content.replace("'window size'", "'sigma'")
content = content.replace("'window sizes'", "'sigmas'")

with open(f, 'w') as file:
    file.write(content)

# 2. calc_bg_polar_gaussian.py
f = 'calc_bg_polar_gaussian.py'
with open(f, 'r') as file:
    content = file.read()

content = content.replace("parser.add_argument('--windows', type=str", "parser.add_argument('--sigmas', type=str")
content = content.replace("help='Window sizes.", "help='Gaussian sigmas.")
content = content.replace("for arg_w in args.windows:", "for arg_w in args.sigmas:")
content = content.replace("half_w = max_size", "half_w = int(np.ceil(3 * max_size))")
content = content.replace("max window size", "max gaussian extent (3*sigma)")
content = content.replace("out_roi = base_path / \"bg_polar_flow_roi.zarr\"", "out_roi = base_path / \"bg_polar_flow_roi_gaussian.zarr\"")
content = content.replace(kernel_old, kernel_new)
content = content.replace("dist_map > (size + 5)", "dist_map > (int(np.ceil(3 * size)) + 5)")
content = content.replace("'window size'", "'sigma'")
content = content.replace("'window sizes'", "'sigmas'")

with open(f, 'w') as file:
    file.write(content)

# 3. calc_local_polar_noCargo_gaussian.py
f = 'calc_local_polar_noCargo_gaussian.py'
with open(f, 'r') as file:
    content = file.read()

content = content.replace("parser.add_argument('--windows', type=str", "parser.add_argument('--sigmas', type=str")
content = content.replace("help='Window sizes.", "help='Gaussian sigmas.")
content = content.replace("for arg_w in args.windows:", "for arg_w in args.sigmas:")
content = content.replace("half_w = max_size", "half_w = int(np.ceil(3 * max_size))")
content = content.replace("out_roi = base_path / f\"local_polar_noCargo.zarr\"", "out_roi = base_path / f\"local_polar_noCargo_gaussian.zarr\"")
content = content.replace(kernel_old, kernel_new)
content = content.replace("'window size'", "'sigma'")
content = content.replace("'window sizes'", "'sigmas'")

with open(f, 'w') as file:
    file.write(content)

# 4. bead_flow_interaction_gaussian.py
f = 'bead_flow_interaction_gaussian.py'
with open(f, 'r') as file:
    content = file.read()

old_func1 = """    # 中心(bx, by)からの距離の二乗を計算し、円内のマスクを作成
    dist_sq = (x_idx - bx)**2 + (y_idx - by)**2
    mask = (dist_sq <= radius**2) & (dist_sq > particle_radius**2)
    
    # マスク内のピクセルが存在しない場合(例えば完全に画像外)は0を返す
    if not np.any(mask):
        return 0.0, 0.0, 0.0, 0.0, 0.0
        
    # 円内のフローを抽出
    local_u = flow_u[y_min:y_max, x_min:x_max][mask]
    local_v = flow_v[y_min:y_max, x_min:x_max][mask]
    
    # ビーズの速度のノルム
    b_norm = np.sqrt(b_dx**2 + b_dy**2)
    
    # 各ピクセルでの内積を計算
    local_dot = b_dx * local_u + b_dy * local_v
    
    # 各ピクセルでのcos類似度を計算
    if b_norm > 0:
        local_flow_norm = np.sqrt(local_u**2 + local_v**2)
        # flow_normが0のピクセルでは0除算を避ける
        local_cos = np.divide(
            local_dot, 
            b_norm * local_flow_norm, 
            out=np.zeros_like(local_dot), 
            where=local_flow_norm != 0
        )
    else:
        # ビーズが動いていない場合はcos類似度を0とする
        local_cos = np.zeros_like(local_dot)
        
    # それぞれの平均を計算（無効な値・NaNなどがあれば除く）
    mean_dot = np.nanmean(local_dot)
    mean_cos = np.nanmean(local_cos)
    mean_u = np.nanmean(local_u)
    mean_v = np.nanmean(local_v)
    cos_std = np.nanstd(local_cos)"""

new_func1 = """    sigma = float(radius)
    # 中心(bx, by)からの距離の二乗を計算
    dist_sq = (x_idx - bx)**2 + (y_idx - by)**2
    
    weights = np.exp(-dist_sq / (2 * sigma**2))
    if particle_radius > 0:
        weights[dist_sq <= particle_radius**2] = 0
    
    weights_sum = np.sum(weights)
    if weights_sum == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
        
    local_u = flow_u[y_min:y_max, x_min:x_max]
    local_v = flow_v[y_min:y_max, x_min:x_max]
    
    b_norm = np.sqrt(b_dx**2 + b_dy**2)
    local_dot = b_dx * local_u + b_dy * local_v
    
    if b_norm > 0:
        local_flow_norm = np.sqrt(local_u**2 + local_v**2)
        local_cos = np.divide(
            local_dot, 
            b_norm * local_flow_norm, 
            out=np.zeros_like(local_dot), 
            where=local_flow_norm != 0
        )
    else:
        local_cos = np.zeros_like(local_dot)
        
    # Calculate weighted means, excluding NaNs
    valid = ~np.isnan(local_dot) & ~np.isnan(local_cos) & ~np.isnan(local_u) & ~np.isnan(local_v)
    w_valid = weights[valid]
    w_sum = np.sum(w_valid)
    
    if w_sum == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
        
    mean_dot = np.sum(w_valid * local_dot[valid]) / w_sum
    mean_cos = np.sum(w_valid * local_cos[valid]) / w_sum
    mean_u = np.sum(w_valid * local_u[valid]) / w_sum
    mean_v = np.sum(w_valid * local_v[valid]) / w_sum
    
    cos_var = np.sum(w_valid * (local_cos[valid] - mean_cos)**2) / w_sum
    cos_std = np.sqrt(cos_var)"""

content = content.replace(old_func1, new_func1)

content = content.replace("x_min = max(0, int(np.floor(bx - radius)))", "x_min = max(0, int(np.floor(bx - 3*radius)))")
content = content.replace("x_max = min(W, int(np.ceil(bx + radius)) + 1)", "x_max = min(W, int(np.ceil(bx + 3*radius)) + 1)")
content = content.replace("y_min = max(0, int(np.floor(by - radius)))", "y_min = max(0, int(np.floor(by - 3*radius)))")
content = content.replace("y_max = min(H, int(np.ceil(by + radius)) + 1)", "y_max = min(H, int(np.ceil(by + 3*radius)) + 1)")

old_func2 = """    # 3. 各粒子に対する円内マスクを作成 (shape: (N, H, W))
    mask = (dist_sq <= radius**2) & (dist_sq > particle_radius**2)
    
    # 4. フローデータの次元を拡張して粒子次元に対応させる (shape: (1, H, W))
    flow_u_ext = flow_u[np.newaxis, :, :]
    flow_v_ext = flow_v[np.newaxis, :, :]
    
    # 5. 各ピクセルでの内積を一括計算 (shape: (N, H, W))
    # ベクトル演算: b_dx * flow_u + b_dy * flow_v
    dot_grid = b_dx_arr * flow_u_ext + b_dy_arr * flow_v_ext
    
    # 6. 各ピクセルでのcos類似度を一括計算 (shape: (N, H, W))
    b_norm = np.sqrt(b_dx_arr**2 + b_dy_arr**2)  # (N, 1, 1)
    flow_norm = np.sqrt(flow_u_ext**2 + flow_v_ext**2)  # (1, H, W)
    denom = b_norm * flow_norm  # (N, H, W)
    
    # 0除算を回避してcos類似度を計算
    cos_grid = np.divide(dot_grid, denom, out=np.zeros_like(dot_grid), where=denom > 0)
    
    # 7. マスクを適用して平均・標準偏差を計算 (ここがポイント)
    # マスク外の値を NaN に置換することで、np.nanmean などの恩恵を受ける
    dot_masked = np.where(mask, dot_grid, np.nan)
    cos_masked = np.where(mask, cos_grid, np.nan)
    u_masked = np.where(mask, flow_u_ext, np.nan)
    v_masked = np.where(mask, flow_v_ext, np.nan)
    
    # 粒子ごと（axis=(1,2) つまり H, W 方向）に統計量を計算
    # 警告（すべてNaNの粒子がある場合）を一時的に無視
    with np.errstate(all='ignore'):
        mean_dot = np.nanmean(dot_masked, axis=(1, 2))
        mean_cos = np.nanmean(cos_masked, axis=(1, 2))
        mean_u = np.nanmean(u_masked, axis=(1, 2))
        mean_v = np.nanmean(v_masked, axis=(1, 2))
        cos_std = np.nanstd(cos_masked, axis=(1, 2))"""

new_func2 = """    sigma = float(radius)
    # 3. Gaussian weights instead of mask
    weights = np.exp(-dist_sq / (2 * sigma**2))
    if particle_radius > 0:
        weights = np.where(dist_sq <= particle_radius**2, 0, weights)
        
    flow_u_ext = flow_u[np.newaxis, :, :]
    flow_v_ext = flow_v[np.newaxis, :, :]
    
    dot_grid = b_dx_arr * flow_u_ext + b_dy_arr * flow_v_ext
    
    b_norm = np.sqrt(b_dx_arr**2 + b_dy_arr**2)
    flow_norm = np.sqrt(flow_u_ext**2 + flow_v_ext**2)
    denom = b_norm * flow_norm
    cos_grid = np.divide(dot_grid, denom, out=np.zeros_like(dot_grid), where=denom > 0)
    
    # Create mask for NaNs in flow data
    valid_mask = ~np.isnan(flow_u_ext) & ~np.isnan(flow_v_ext)
    weights = np.where(valid_mask, weights, 0.0)
    weights_sum = np.sum(weights, axis=(1, 2))
    
    with np.errstate(all='ignore'):
        mean_dot = np.sum(weights * dot_grid, axis=(1, 2)) / weights_sum
        mean_cos = np.sum(weights * cos_grid, axis=(1, 2)) / weights_sum
        mean_u = np.sum(weights * flow_u_ext, axis=(1, 2)) / weights_sum
        mean_v = np.sum(weights * flow_v_ext, axis=(1, 2)) / weights_sum
        
        cos_var = np.sum(weights * (cos_grid - mean_cos[:, np.newaxis, np.newaxis])**2, axis=(1, 2)) / weights_sum
        cos_std = np.sqrt(cos_var)"""

content = content.replace(old_func2, new_func2)

content = content.replace("parser.add_argument('--radii', type=str", "parser.add_argument('--sigmas', type=str")
content = content.replace("help='Radii sizes.", "help='Gaussian sigmas.")
content = content.replace("for arg_w in args.radii:", "for arg_w in args.sigmas:")
content = content.replace("'beads_flow_interaction.csv'", "'beads_flow_interaction_gaussian.csv'")
content = content.replace("'radius': rad", "'sigma': rad")

with open(f, 'w') as file:
    file.write(content)

print("Modification complete.")
