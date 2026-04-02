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
    parser.add_argument('--windows', type=str, nargs='+', default=['10:200:10', '200:2000:100'], 
                        help='Window sizes. Accepts space/comma separated numbers, or start:stop:step (e.g. 10 50 100, or 10:200:10)')
    args = parser.parse_args()

    base_path = Path(args.base_path)
    csv_path = base_path / args.tracks_csv
    h5_path = base_path / "GFP_flows.h5"

    if not csv_path.exists() or not h5_path.exists():
        print("Error: Required files (CSV or H5) not found.")
        return

    # Parse window sizes
    sizes = set()
    for arg_w in args.windows:
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
    local_sizes = sorted(list(sizes))

    # 1. データの読み込み
    print(f"Loading tracks from {csv_path}...")
    df_tracks = pd.read_csv(csv_path)
    
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
            
            for w_idx, size in enumerate(local_sizes):
                u_avg = uniform_filter(m_ux, size=size)
                v_avg = uniform_filter(m_uy, size=size)
                p_field = np.hypot(u_avg, v_avg, dtype=np.float32)

                p_vals[w_idx, :] = p_field[y_idx, x_idx]

            return t, p_vals, p_ids

        # 3. 実行
        print(f"Calculating for window sizes: {local_sizes}...")
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for t, p_vals, p_ids in tqdm(executor.map(process_frame, active_frames), total=len(active_frames)):
                f_i = frame_to_idx[t]
                p_indices = [particle_to_idx[p] for p in p_ids]
                polar_order_array[:, f_i, p_indices] = p_vals

    # 4. xarray Dataset の構築と保存
    print("\nConsolidating data into xarray...")
    
    ds_particles['polar_order'] = xr.DataArray(
        polar_order_array,
        dims=['window size', 'frame', 'particle'],
        coords={'window size': local_sizes, 'frame': ds_particles.frame, 'particle': ds_particles.particle}
    )
    
    # メタデータの付与
    ds_particles.attrs['description'] = 'Local polar order analysis: Particles'
    ds_particles.attrs['window sizes'] = local_sizes

    out_particle = base_path / "local_polar_w.zarr"

    # Zarr保存
    if out_particle.exists():
        shutil.rmtree(out_particle, ignore_errors=True)
    
    ds_particles.to_zarr(str(out_particle), mode='w', consolidated=False)
    
    print(f"Success! Data saved to {out_particle}")

if __name__ == "__main__":
    main()