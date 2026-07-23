import argparse
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from tqdm import tqdm
from numba import njit, prange

def calculate_flow_bead_dot_product(bx, by, b_dx, b_dy, flow_u, flow_v, radius, particle_radius=0):
    """
    ビーズの運動ベクトルと、その近傍の円形領域内のオプティカルフローの内積およびcos類似度を計算する（単体テスト用）。
    """
    H, W = flow_u.shape
    
    x_min = max(0, int(np.floor(bx - 3*radius)))
    x_max = min(W, int(np.ceil(bx + 3*radius)) + 1)
    y_min = max(0, int(np.floor(by - 3*radius)))
    y_max = min(H, int(np.ceil(by + 3*radius)) + 1)
    
    y_idx, x_idx = np.ogrid[y_min:y_max, x_min:x_max]
    
    sigma = float(radius)
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
    cos_std = np.sqrt(max(0.0, cos_var))

    if np.isnan(mean_dot): mean_dot = 0.0
    if np.isnan(mean_cos): mean_cos = 0.0
    if np.isnan(mean_u): mean_u = 0.0
    if np.isnan(mean_v): mean_v = 0.0
    
    return mean_dot, mean_cos, mean_u, mean_v, cos_std


@njit(parallel=True)
def calculate_flow_bead_dot_product_numba(bx_arr, by_arr, b_dx_arr, b_dy_arr, flow_u, flow_v, radius, particle_radius=0.0):
    """
    Numbaを用いた超高速かつ省メモリなフロー相互作用の一括計算。
    バウンディングボックス内のみアクセスし、マルチスレッド処理で計算を並列化。
    """
    N = len(bx_arr)
    H, W = flow_u.shape
    
    mean_dot = np.zeros(N, dtype=np.float32)
    mean_cos = np.zeros(N, dtype=np.float32)
    mean_u = np.zeros(N, dtype=np.float32)
    mean_v = np.zeros(N, dtype=np.float32)
    cos_stds = np.zeros(N, dtype=np.float32)
    
    sigma = float(radius)
    sigma2 = 2.0 * sigma * sigma
    pr2 = float(particle_radius * particle_radius)
    
    x_mins = np.maximum(0, np.floor(bx_arr - 3*radius)).astype(np.int32)
    x_maxs = np.minimum(W, np.ceil(bx_arr + 3*radius) + 1).astype(np.int32)
    y_mins = np.maximum(0, np.floor(by_arr - 3*radius)).astype(np.int32)
    y_maxs = np.minimum(H, np.ceil(by_arr + 3*radius) + 1).astype(np.int32)
    
    for i in prange(N):
        x = bx_arr[i]
        y = by_arr[i]
        dx = b_dx_arr[i]
        dy = b_dy_arr[i]
        
        x_min = x_mins[i]
        x_max = x_maxs[i]
        y_min = y_mins[i]
        y_max = y_maxs[i]
        
        b_norm = np.sqrt(dx*dx + dy*dy)
        
        w_sum = 0.0
        dot_sum = 0.0
        cos_sum = 0.0
        u_sum = 0.0
        v_sum = 0.0
        cos_sq_sum = 0.0
        
        for yy in range(y_min, y_max):
            dy_sq = (yy - y) * (yy - y)
            for xx in range(x_min, x_max):
                u = flow_u[yy, xx]
                v = flow_v[yy, xx]
                
                if np.isnan(u) or np.isnan(v):
                    continue
                    
                dist_sq = (xx - x)*(xx - x) + dy_sq
                
                if particle_radius > 0.0 and dist_sq <= pr2:
                    continue
                    
                w = np.exp(-dist_sq / sigma2)
                
                w_sum += w
                u_sum += w * u
                v_sum += w * v
                
                dot = dx * u + dy * v
                dot_sum += w * dot
                
                if b_norm > 0.0:
                    flow_norm = np.sqrt(u*u + v*v)
                    if flow_norm > 0.0:
                        cos_val = dot / (b_norm * flow_norm)
                        cos_sum += w * cos_val
                        cos_sq_sum += w * cos_val * cos_val
                        
        if w_sum > 0.0:
            mean_dot[i] = dot_sum / w_sum
            mean_u[i] = u_sum / w_sum
            mean_v[i] = v_sum / w_sum
            m_cos = cos_sum / w_sum
            mean_cos[i] = m_cos
            
            # 浮動小数点誤差による 0 未満の数値をクリップして NaN を防止
            cos_var = (cos_sq_sum / w_sum) - (m_cos * m_cos)
            if cos_var < 0.0:
                cos_var = 0.0
            cos_stds[i] = np.sqrt(cos_var)

    return mean_dot, mean_cos, mean_u, mean_v, cos_stds


def calculate_flow_bead_dot_product_batch(bx_arr, by_arr, b_dx_arr, b_dy_arr, flow_u, flow_v, radius, particle_radius=0):
    """
    複数のビーズについて一括で内積とcos類似度を計算する関数。
    """
    bx_arr = np.asarray(bx_arr, dtype=np.float64)
    by_arr = np.asarray(by_arr, dtype=np.float64)
    b_dx_arr = np.asarray(b_dx_arr, dtype=np.float64)
    b_dy_arr = np.asarray(b_dy_arr, dtype=np.float64)
    
    return calculate_flow_bead_dot_product_numba(
        bx_arr, by_arr, b_dx_arr, b_dy_arr, 
        flow_u, flow_v, float(radius), float(particle_radius)
    )


def main():
    parser = argparse.ArgumentParser(description='Calculate bead-flow interaction (dot product and cos sim).')
    parser.add_argument('base_path', type=str, help='Path to the directory containing hdf5 and csv')
    parser.add_argument('--tracks_csv', type=str, default='beads_tracks.csv', help='Trackpy csv file name')
    parser.add_argument('--h5_file', type=str, default='GFP_flows.h5', help='H5 file name')
    parser.add_argument('--sigmas', type=str, nargs='+', default=['5:100:5', '100:1000:50'], 
                        help='Gaussian sigmas. Accepts space/comma separated numbers, or start:stop:step (e.g. 10 50 100, or 10:200:10)')
    parser.add_argument('--particle_radius', type=float, default=0.0, help='Inner radius to exclude particle itself')
    parser.add_argument('--dt', type=int, default=1, help='Time difference (frames) to calculate bead velocity')
    parser.add_argument('--output', type=str, default='beads_flow_interaction_gaussian.csv', help='Output CSV file name')
    
    args = parser.parse_args()
    
    base_path = Path(args.base_path)
    csv_path = base_path / args.tracks_csv
    h5_path = base_path / args.h5_file
    out_path = base_path / args.output
    
    if not csv_path.exists() or not h5_path.exists():
        print("Error: Required files (CSV or H5) not found.")
        return
        
    sizes = set()
    for arg_w in args.sigmas:
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
    
    # JITコンパイルのウォームアップ
    dummy_arr = np.array([10.0], dtype=np.float64)
    dummy_flow = np.zeros((10, 10), dtype=np.float32)
    calculate_flow_bead_dot_product_numba(
        dummy_arr, dummy_arr, dummy_arr, dummy_arr, dummy_flow, dummy_flow, 5.0, 0.0
    )

    if out_path.exists():
        out_path.unlink()

    first_write = True
    
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
                
            bx = merged['x_0'].to_numpy(dtype=np.float32)
            by = merged['y_0'].to_numpy(dtype=np.float32)
            b_dx = (merged['x_t'] - merged['x_0']).to_numpy(dtype=np.float32)
            b_dy = (merged['y_t'] - merged['y_0']).to_numpy(dtype=np.float32)
            p_ids = merged['particle'].to_numpy()
            
            if channel_first:
                m_x = np.ascontiguousarray(flow_data[t, 0, ...], dtype=np.float32)
                m_y = np.ascontiguousarray(flow_data[t, 1, ...], dtype=np.float32)
            else:
                m_x = np.ascontiguousarray(flow_data[t, ..., 0], dtype=np.float32)
                m_y = np.ascontiguousarray(flow_data[t, ..., 1], dtype=np.float32)
                
            n_particles = len(p_ids)
            
            # 1フレーム分の全 sigma 結果を一時的に保持するリスト
            frame_results = []
            
            for rad in radii_list:
                dot, cos, mu, mv, cos_std = calculate_flow_bead_dot_product_numba(
                    bx, by, b_dx, b_dy, m_x, m_y, float(rad), float(args.particle_radius)
                )
                
                frame_df = pd.DataFrame({
                    'frame': np.full(n_particles, t, dtype=np.int32),
                    'particle': p_ids,
                    'sigma': np.full(n_particles, rad, dtype=np.float32),
                    'dot_product': dot,
                    'cos_sim': cos,
                    'mean_flow_u': mu,
                    'mean_flow_v': mv,
                    'cos_std': cos_std
                })
                frame_results.append(frame_df)

            # 1フレーム分の全 sigma をまとめて1回の to_csv で書き出し
            if frame_results:
                one_frame_df = pd.concat(frame_results, ignore_index=True)
                one_frame_df.to_csv(
                    out_path,
                    mode='a',
                    header=first_write,
                    index=False
                )
                first_write = False
        
if __name__ == "__main__":
    main()