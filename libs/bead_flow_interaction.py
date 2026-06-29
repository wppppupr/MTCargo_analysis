import argparse
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from tqdm import tqdm

def calculate_flow_bead_dot_product(bx, by, b_dx, b_dy, flow_u, flow_v, radius, particle_radius=0):
    """
    ビーズの運動ベクトルと、その近傍の円形領域内のオプティカルフローの内積およびcos類似度を計算する。
    （内積・cos類似度をピクセルごとに計算してから平均をとる）

    Parameters:
    -----------
    bx, by : float
        ビーズの中心座標 (ピクセル単位)
    b_dx, b_dy : float
        ビーズの変位ベクトル (または速度ベクトル)
    flow_u : 2D np.ndarray
        オプティカルフローのX成分 (水平方向のフロー, shape: [H, W])
    flow_v : 2D np.ndarray
        オプティカルフローのY成分 (垂直方向のフロー, shape: [H, W])
    radius : float
        計算対象とする近傍の半径 (ピクセル単位)
    particle_radius : float
        除外する粒子自身の半径。この半径以内のピクセルは計算から除外される (ピクセル単位)

    Returns:
    --------
    mean_dot : float
        ピクセルごとの内積の平均値
    mean_cos : float
        ピクセルごとのcos類似度の平均値 (-1.0 〜 1.0)
    mean_u, mean_v : float
        円内での平均オプティカルフローのX, Y成分
    cos_std : float
        cos類似度の標準偏差
    """
    H, W = flow_u.shape
    
    # 円を囲むバウンディングボックスを計算 (画像外に出ないようにクリップ)
    x_min = max(0, int(np.floor(bx - radius)))
    x_max = min(W, int(np.ceil(bx + radius)) + 1)
    y_min = max(0, int(np.floor(by - radius)))
    y_max = min(H, int(np.ceil(by + radius)) + 1)
    
    # バウンディングボックス内のグリッドを生成
    y_idx, x_idx = np.ogrid[y_min:y_max, x_min:x_max]
    
    # 中心(bx, by)からの距離の二乗を計算し、円内のマスクを作成
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
    cos_std = np.nanstd(local_cos)

    # 全てがNaNだった場合のフォールバック
    if np.isnan(mean_dot): mean_dot = 0.0
    if np.isnan(mean_cos): mean_cos = 0.0
    if np.isnan(mean_u): mean_u = 0.0
    if np.isnan(mean_v): mean_v = 0.0
    
    return mean_dot, mean_cos, mean_u, mean_v, cos_std

def calculate_flow_bead_dot_product_broadcast(bx_arr, by_arr, b_dx_arr, b_dy_arr, flow_u, flow_v, radius, particle_radius=0):
    """
    ブロードキャストを活用し、ループを一切使わずに全粒子のフロー相互作用を一括計算する。
    
    Shapeの動き:
    - N: 粒子数, H: 画像の高さ, W: 画像の幅
    """
    bx_arr = np.asarray(bx_arr)[:, np.newaxis, np.newaxis]   # shape: (N, 1, 1)
    by_arr = np.asarray(by_arr)[:, np.newaxis, np.newaxis]   # shape: (N, 1, 1)
    b_dx_arr = np.asarray(b_dx_arr)[:, np.newaxis, np.newaxis] # shape: (N, 1, 1)
    b_dy_arr = np.asarray(b_dy_arr)[:, np.newaxis, np.newaxis] # shape: (N, 1, 1)
    
    H, W = flow_u.shape
    
    # 1. 全ピクセルのXY座標グリッドを作成
    # flow_u と同じ形状の座標行列 (shape: (H, W))
    y_grid, x_grid = np.ogrid[0:H, 0:W]
    x_grid = x_grid.astype(np.float64)
    y_grid = y_grid.astype(np.float64)
    
    # 2. 全粒子から全ピクセルへの距離の二乗を一括計算 (ブロードキャスト)
    # (N, 1, 1) と (1, H, W) の演算になり、結果は (N, H, W)
    dist_sq = (x_grid[np.newaxis, :, :] - bx_arr)**2 + (y_grid[np.newaxis, :, :] - by_arr)**2
    
    # 3. 各粒子に対する円内マスクを作成 (shape: (N, H, W))
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
        cos_std = np.nanstd(cos_masked, axis=(1, 2))
        
    # 全てがNaN（画像外など）だった粒子のフォールバック処理
    mean_dot = np.isnan(mean_dot, out=mean_dot, where=np.isnan(mean_dot)) # 0にする場合は後述
    mean_dot = np.where(np.isnan(mean_dot), 0.0, mean_dot)
    mean_cos = np.where(np.isnan(mean_cos), 0.0, mean_cos)
    mean_u = np.where(np.isnan(mean_u), 0.0, mean_u)
    mean_v = np.where(np.isnan(mean_v), 0.0, mean_v)
    cos_stds = np.where(np.isnan(cos_std), 0.0, cos_std)
    
    return mean_dot, mean_cos, mean_u, mean_v, cos_stds

def calculate_flow_bead_dot_product_batch(bx_arr, by_arr, b_dx_arr, b_dy_arr, flow_u, flow_v, radius, particle_radius=0):
    """
    複数のビーズについて一括で内積とcos類似度を計算する関数。
    
    Parameters:
    -----------
    bx_arr, by_arr : array-like
        ビーズの中心座標配列
    b_dx_arr, b_dy_arr : array-like
        ビーズの変位ベクトル配列
    flow_u, flow_v : 2D np.ndarray
        オプティカルフローのX成分, Y成分
    radius : float
        近傍の半径
    particle_radius : float
        除外する粒子自身の半径
        
    Returns:
    --------
    dot_products : np.ndarray
        各ビーズにおけるピクセルごとの内積平均の配列
    cos_sims : np.ndarray
        各ビーズにおけるピクセルごとのcos類似度平均の配列
    mean_us, mean_vs : np.ndarray
        各ビーズ近傍の平均オプティカルフロー成分の配列
    """
    bx_arr = np.asarray(bx_arr)
    by_arr = np.asarray(by_arr)
    b_dx_arr = np.asarray(b_dx_arr)
    b_dy_arr = np.asarray(b_dy_arr)
    
    n_beads = len(bx_arr)
    dot_products = np.zeros(n_beads)
    cos_sims = np.zeros(n_beads)
    mean_us = np.zeros(n_beads)
    mean_vs = np.zeros(n_beads)
    cos_stds = np.zeros(n_beads)
    
    for i in range(n_beads):
        dot, cos, mu, mv, cos_std = calculate_flow_bead_dot_product(
            bx_arr[i], by_arr[i], b_dx_arr[i], b_dy_arr[i], 
            flow_u, flow_v, radius, particle_radius
        )
        dot_products[i] = dot
        cos_sims[i] = cos
        mean_us[i] = mu
        mean_vs[i] = mv
        cos_stds[i] = cos_std
        
    return dot_products, cos_sims, mean_us, mean_vs, cos_stds


def main():
    parser = argparse.ArgumentParser(description='Calculate bead-flow interaction (dot product and cos sim).')
    parser.add_argument('base_path', type=str, help='Path to the directory containing hdf5 and csv')
    parser.add_argument('--tracks_csv', type=str, default='beads_tracks.csv', help='Trackpy csv file name')
    parser.add_argument('--h5_file', type=str, default='GFP_flows.h5', help='H5 file name')
    parser.add_argument('--radii', type=str, nargs='+', default=['10:200:10', '200:2000:100'], 
                        help='Radii sizes. Accepts space/comma separated numbers, or start:stop:step (e.g. 10 50 100, or 10:200:10)')
    parser.add_argument('--particle_radius', type=float, default=0.0, help='Inner radius to exclude particle itself')
    parser.add_argument('--dt', type=int, default=1, help='Time difference (frames) to calculate bead velocity')
    parser.add_argument('--output', type=str, default='beads_flow_interaction.csv', help='Output CSV file name')
    
    args = parser.parse_args()
    
    base_path = Path(args.base_path)
    csv_path = base_path / args.tracks_csv
    h5_path = base_path / args.h5_file
    out_path = base_path / args.output
    
    if not csv_path.exists() or not h5_path.exists():
        print("Error: Required files (CSV or H5) not found.")
        return
        
    sizes = set()
    for arg_w in args.radii:
        for p in arg_w.split(','):
            if not p.strip(): continue
            if ':' in p:
                parts = p.split(':')
                w_start = int(parts[0])
                w_stop = int(parts[1]) if len(parts) > 1 else w_start
                w_step = int(parts[2]) if len(parts) > 2 else 1
                sizes.update(range(w_start, w_stop + 1, w_step))
            else:
                sizes.add(int(p))
    radii_list = sorted(list(sizes))
        
    print(f"Loading tracks from {csv_path}...")
    df_tracks = pd.read_csv(csv_path)
    
    if 'particle' not in df_tracks.columns:
        df_tracks['particle'] = np.arange(len(df_tracks))
        
    results = []
    
    with h5py.File(str(h5_path), 'r') as f:
        dataset_key = list(f.keys())[0]
        flow_data = f[dataset_key]
        shape = flow_data.shape
        print(f"Flow data shape: {shape}")
        
        num_frames = shape[0]
        if len(shape) == 4 and shape[-1] == 2:
            channel_first = False
        else:
            channel_first = True
            
        grouped = df_tracks.groupby('frame')
        active_frames = sorted([fr for fr in grouped.groups.keys() if fr < num_frames])
        
        for t in tqdm(active_frames, desc="Processing frames"):
            if t + args.dt not in grouped.groups:
                continue
                
            subset_t0 = df_tracks.loc[grouped.groups[t]]
            subset_tdt = df_tracks.loc[grouped.groups[t + args.dt]]
            
            merged = pd.merge(subset_t0, subset_tdt, on='particle', suffixes=('_0', '_t'))
            if merged.empty:
                continue
                
            bx = merged['x_0'].values
            by = merged['y_0'].values
            b_dx = merged['x_t'].values - bx
            b_dy = merged['y_t'].values - by
            p_ids = merged['particle'].values
            
            if channel_first:
                m_x = flow_data[t, 0, ...].astype(np.float32)
                m_y = flow_data[t, 1, ...].astype(np.float32)
            else:
                m_x = flow_data[t, ..., 0].astype(np.float32)
                m_y = flow_data[t, ..., 1].astype(np.float32)
                
            for rad in radii_list:
                dot, cos, mu, mv, cos_std = calculate_flow_bead_dot_product_broadcast(
                    bx, by, b_dx, b_dy, m_x, m_y, rad, args.particle_radius
                )
                
                for i, p_id in enumerate(p_ids):
                    results.append({
                        'frame': t,
                        'particle': p_id,
                        'radius': rad,
                        'dot_product': dot[i],
                        'cos_sim': cos[i],
                        'mean_flow_u': mu[i],
                        'mean_flow_v': mv[i],
                        'cos_std': cos_std[i]
                    })

    if len(results) > 0:
        res_df = pd.DataFrame(results)
        print(f"Saving results to {out_path}...")
        res_df.to_csv(out_path, index=False)
        print("Done!")
    else:
        print("No interaction data computed (maybe no particles matched dt).")

if __name__ == "__main__":
    main()
