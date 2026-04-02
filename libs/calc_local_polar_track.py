import argparse
import numpy as np
import pandas as pd
import h5py
import xarray as xr
import os
import shutil
from pathlib import Path
from tqdm import tqdm
from scipy.ndimage import uniform_filter, binary_dilation
from concurrent.futures import ThreadPoolExecutor

def main():
    parser = argparse.ArgumentParser(description='Calculate local polar order for particles and background.')
    parser.add_argument('base_path', type=str, help='Path to the directory containing hdf5 and csv')
    parser.add_argument('--tracks_csv', type=str, default='beads_tracks.csv', help='Trackpy csv file name')
    parser.add_argument('--start', type=int, default=10, help='Start window size')
    parser.add_argument('--stop', type=int, default=201, help='Stop window size')
    parser.add_argument('--step', type=int, default=10, help='Step size')
    parser.add_argument('--dilation', type=int, default=50, help='Margin to exclude around particles for background')
    args = parser.parse_args()

    base_path = Path(args.base_path)
    csv_path = base_path / args.tracks_csv
    h5_path = base_path / "GFP_flows.h5"

    if not csv_path.exists() or not h5_path.exists():
        print("Error: Required files (CSV or H5) not found.")
        return

    # 1. データの読み込み
    print(f"Loading tracks from {csv_path}...")
    df_tracks = pd.read_csv(csv_path)
    
    local_sizes = list(range(args.start, args.stop + 1, args.step))
    
    # 2. 並列処理の設定
    workers = min(32, (os.cpu_count() or 1) + 4)
    grouped = df_tracks.groupby('frame')

    with h5py.File(str(h5_path), 'r') as f:
        dataset_key = list(f.keys())[0]
        # メモリ節約のため、参照のみ取得（スライシングで必要な時だけロード）
        flow_data = f[dataset_key]
        shape = flow_data.shape
        print(f"Flow data shape: {shape}")
        
        num_frames = shape[0]
        if shape[-1] == 2:
            rows, cols = shape[1], shape[2]
            channel_first = False
        else:
            rows, cols = shape[2], shape[3]
            channel_first = True

        active_frames = sorted([f for f in grouped.groups.keys() if f < num_frames])

        # ds_particles の初期化 (ここで df_tracks を変換してしまう)
        if 'particle' not in df_tracks.columns:
            df_tracks['particle'] = np.arange(len(df_tracks))
        
        ds_particles = df_tracks.set_index(['frame', 'particle']).to_xarray()
        
        polar_order_array = np.full((len(local_sizes), len(ds_particles.frame), len(ds_particles.particle)), np.nan, dtype=np.float32)
        
        max_num_p = max((len(group) for group in grouped.groups.values()), default=1)
        bg_polar_array = np.full((len(local_sizes), len(ds_particles.frame), max_num_p), np.nan, dtype=np.float32)
        
        frame_to_idx = {f: i for i, f in enumerate(ds_particles.frame.values)}
        particle_to_idx = {p: i for i, p in enumerate(ds_particles.particle.values)}

        def process_frame(t):
            frame_idx = grouped.groups[t]
            subset = df_tracks.loc[frame_idx]
            num_p = len(subset)
            p_ids = subset['particle'].values

            # 座標の準備
            y_idx = np.clip(np.round(subset['y'].values).astype(int), 0, rows - 1)
            x_idx = np.clip(np.round(subset['x'].values).astype(int), 0, cols - 1)

            # 背景マスクの作成（粒子周辺を除外）
            mask = np.zeros((rows, cols), dtype=bool)
            mask[y_idx, x_idx] = True
            # 窓サイズの半分程度を目安に膨らませる
            exclude_mask = binary_dilation(mask, iterations=args.dilation)
            empty_y, empty_x = np.where(~exclude_mask)

            # 背景サンプリング（粒子数と同じ数を抽出）
            rng = np.random.default_rng(seed=t)
            if len(empty_y) > num_p:
                sample_idx = rng.choice(len(empty_y), size=num_p, replace=False)
                bg_y, bg_x = empty_y[sample_idx], empty_x[sample_idx]
            else:
                bg_y, bg_x = empty_y, empty_x

            # 流速データのロードと計算 (float32)
            if channel_first:
                m_x = flow_data[t, 0, ...].astype(np.float32)
                m_y = flow_data[t, 1, ...].astype(np.float32)
            else:
                m_x = flow_data[t, ..., 0].astype(np.float32)
                m_y = flow_data[t, ..., 1].astype(np.float32)
            
            v_mag = np.hypot(m_x, m_y)
            m_ux = m_x / v_mag
            m_uy = m_y / v_mag

            p_vals = np.empty((len(local_sizes), num_p), dtype=np.float32)
            bg_vals = np.full((len(local_sizes), num_p), np.nan, dtype=np.float32)
            
            for w_idx, size in enumerate(local_sizes):
                u_avg = uniform_filter(m_ux, size=size)
                v_avg = uniform_filter(m_uy, size=size)
                p_field = np.hypot(u_avg, v_avg, dtype=np.float32)

                p_vals[w_idx, :] = p_field[y_idx, x_idx]
                if len(bg_y) > 0:
                    bg_vals[w_idx, :len(bg_y)] = p_field[bg_y, bg_x]

            return t, p_vals, bg_vals, p_ids

        # 3. 実行
        print(f"Calculating for window sizes: {local_sizes}...")
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for t, p_vals, bg_vals, p_ids in tqdm(executor.map(process_frame, active_frames), total=len(active_frames)):
                f_i = frame_to_idx[t]
                p_indices = [particle_to_idx[p] for p in p_ids]
                polar_order_array[:, f_i, p_indices] = p_vals
                
                if bg_vals.shape[1] > 0:
                    bg_polar_array[:, f_i, :bg_vals.shape[1]] = bg_vals

    # 4. xarray Dataset の構築と保存
    print("\nConsolidating data into xarray...")
    
    ds_particles['polar_order'] = xr.DataArray(
        polar_order_array,
        dims=['window_size', 'frame', 'particle'],
        coords={'window_size': local_sizes, 'frame': ds_particles.frame, 'particle': ds_particles.particle}
    )

    ds_bg = xr.Dataset(
        data_vars={
            'polar_order': xr.DataArray(
                bg_polar_array,
                dims=['window_size', 'frame', 'particle'],
                coords={'window_size': local_sizes, 'frame': ds_particles.frame, 'particle': np.arange(max_num_p)}
            )
        }
    )
    
    # メタデータの付与
    ds_particles.attrs['description'] = 'Local polar order analysis: Particles'
    ds_particles.attrs['window_sizes'] = local_sizes
    ds_particles.attrs['dilation_margin'] = args.dilation

    ds_bg.attrs['description'] = 'Local polar order analysis: Background'
    ds_bg.attrs['window_sizes'] = local_sizes
    ds_bg.attrs['dilation_margin'] = args.dilation

    out_particle = base_path / "local_polar_particle.zarr"
    out_bg = base_path / "local_polar_bg.zarr"

    # Zarr保存
    if out_particle.exists():
        shutil.rmtree(out_particle, ignore_errors=True)
    if out_bg.exists():
        shutil.rmtree(out_bg, ignore_errors=True)
    
    ds_particles.to_zarr(str(out_particle), mode='w', consolidated=False)
    ds_bg.to_zarr(str(out_bg), mode='w', consolidated=False)
    
    print(f"Success! Data saved to {out_particle} and {out_bg}")

if __name__ == "__main__":
    main()