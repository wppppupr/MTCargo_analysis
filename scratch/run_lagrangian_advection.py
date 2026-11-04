#!/usr/bin/env python3
"""
scratch/run_lagrangian_advection.py
全実験・全アクティブ粒子を対象に、
「本来の仮想貨物 (Lagrangian Advection Tracer)」の自律移流軌道を並列計算する。
- 仮想円盤を自分自身の位置における足印平均流速に乗せて移流させる。
"""

import sys
import time
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import h5py
from multiprocessing import Pool
import verify_cos_decomposition as vcd

ROOT_DIR = vcd.ROOT_DIR
WORKSPACE_DIR = vcd.CURRENT_DIR
BEADS_INFO = vcd.BEADS_INFO
SCALE = vcd.SCALE_UM_PER_PX  # 0.11 um/px
DT = 4.0  # s / frame


def simulate_experiment(exp_info):
    exp_dir_path, bname, d_um, r_um, target_pids = exp_info
    exp_dir = Path(exp_dir_path)
    flow_path = exp_dir / "GFP_flows.h5"
    tracks_path = exp_dir / "beads_tracks.csv"
    
    if not flow_path.exists() or not tracks_path.exists():
        return []
        
    try:
        df_tracks = pd.read_csv(tracks_path)
    except Exception as e:
        print(f"Error loading tracks {exp_dir.name}: {e}")
        return []
        
    r_px = r_um / SCALE
    rad_ceil = int(np.ceil(r_px))
    y_g, x_g = np.ogrid[-rad_ceil:rad_ceil+1, -rad_ceil:rad_ceil+1]
    disk_mask = (x_g**2 + y_g**2) <= r_px**2
    
    # 対象粒子のメタデータ
    particle_meta = {}
    for pid in target_pids:
        sub = df_tracks[df_tracks['particle'] == pid].sort_values('frame')
        if len(sub) < 3:
            continue
        particle_meta[pid] = {
            'start_frame': int(sub['frame'].min()),
            'end_frame': int(sub['frame'].max()),
            'start_x': float(sub['x'].iloc[0]),
            'start_y': float(sub['y'].iloc[0]),
            'real_sub': sub,
        }
        
    if not particle_meta:
        return []
        
    t0 = time.time()
    results = []
    
    with h5py.File(flow_path, 'r') as f:
        flows_ds = f['flows']
        n_frames, _, H, W = flows_ds.shape
        
        cur_pos = {pid: [particle_meta[pid]['start_x'], particle_meta[pid]['start_y']] for pid in particle_meta}
        virt_records = {pid: [{
            'bead_name': bname,
            'diameter_um': d_um,
            'radius_um': r_um,
            'exp_dir': exp_dir.name,
            'particle': pid,
            'frame': particle_meta[pid]['start_frame'],
            'x_virt_px': particle_meta[pid]['start_x'],
            'y_virt_px': particle_meta[pid]['start_y'],
            'u_virt': 0.0,
            'v_virt': 0.0,
        }] for pid in particle_meta}
        
        min_frm = min(m['start_frame'] for m in particle_meta.values())
        max_frm = max(m['end_frame'] for m in particle_meta.values())
        
        for frm in range(min_frm, min(max_frm, n_frames)):
            flow_frame = flows_ds[frm]  # (2, H, W)
            flow_u = flow_frame[0].astype(np.float32)
            flow_v = flow_frame[1].astype(np.float32)
            
            for pid, meta in particle_meta.items():
                if frm < meta['start_frame'] or frm >= meta['end_frame']:
                    continue
                
                cx, cy = cur_pos[pid]
                x_int = int(round(cx))
                y_int = int(round(cy))
                
                y_min = y_int - rad_ceil
                y_max = y_int + rad_ceil + 1
                x_min = x_int - rad_ceil
                x_max = x_int + rad_ceil + 1
                
                if y_min < 0 or y_max > H or x_min < 0 or x_max > W:
                    y_idx, x_idx = np.ogrid[max(0, y_min):min(H, y_max), max(0, x_min):min(W, x_max)]
                    sub_mask = ((x_idx - cx)**2 + (y_idx - cy)**2) <= r_px**2
                    if np.any(sub_mask):
                        u_mean = float(np.mean(flow_u[y_idx, x_idx][sub_mask]))
                        v_mean = float(np.mean(flow_v[y_idx, x_idx][sub_mask]))
                    else:
                        u_mean, v_mean = 0.0, 0.0
                else:
                    patch_u = flow_u[y_min:y_max, x_min:x_max]
                    patch_v = flow_v[y_min:y_max, x_min:x_max]
                    u_mean = float(np.mean(patch_u[disk_mask]))
                    v_mean = float(np.mean(patch_v[disk_mask]))
                    
                new_x = cx + u_mean
                new_y = cy + v_mean
                cur_pos[pid] = [new_x, new_y]
                
                virt_records[pid].append({
                    'bead_name': bname,
                    'diameter_um': d_um,
                    'radius_um': r_um,
                    'exp_dir': exp_dir.name,
                    'particle': pid,
                    'frame': frm + 1,
                    'x_virt_px': new_x,
                    'y_virt_px': new_y,
                    'u_virt': u_mean,
                    'v_virt': v_mean,
                })
                
        for pid, recs in virt_records.items():
            results.extend(recs)
            
    t1 = time.time()
    print(f"Done {exp_dir.name} ({len(particle_meta)} particles, {min_frm}-{max_frm} frames) in {t1-t0:.1f}s")
    return results


def main():
    active_set = vcd.load_active_particles(WORKSPACE_DIR)
    
    tasks = []
    for b_item in BEADS_INFO:
        bname = b_item['name']
        d_um = b_item['diameter_um']
        r_um = b_item['radius_um']
        exp_dirs = vcd.find_experiment_dirs(ROOT_DIR, bname)
        
        for exp_dir in exp_dirs:
            # アクティブ粒子のみ
            pids = [pid for (exp, pid) in active_set if exp == exp_dir.name]
            if not pids:
                # active_set が空なら全粒子
                try:
                    tdf = pd.read_csv(exp_dir / "beads_tracks.csv")
                    pids = tdf['particle'].unique().tolist()
                except:
                    continue
            if pids:
                tasks.append((str(exp_dir), bname, d_um, r_um, pids))
                
    print(f"Total experiment tasks to process: {len(tasks)}")
    
    # 6プロセスで並列処理
    t_start = time.time()
    all_results = []
    with Pool(processes=6) as pool:
        results_nested = pool.map(simulate_experiment, tasks)
        for res in results_nested:
            all_results.extend(res)
            
    t_end = time.time()
    print(f"\nAll simulations finished in {t_end - t_start:.1f} seconds! Total records: {len(all_results)}")
    
    df_lagrangian = pd.DataFrame(all_results)
    out_path = Path("scratch/lagrangian_virt_tracks.pkl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_lagrangian.to_pickle(out_path)
    print(f"Saved Lagrangian virtual tracks to {out_path}")


if __name__ == '__main__':
    main()

