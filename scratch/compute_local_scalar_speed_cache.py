#!/usr/bin/env python3
"""
scratch/compute_local_scalar_speed_cache.py

全22実験・全アクティブ粒子の瞬時フレームにおいて、
貨物の足元アニュラス（環状シェル）領域内のピクセルスカラー速さの平均:
  <|u|>_local = mean( sqrt(u_i^2 + v_i^2) )
を GFP_flows.h5 から直接抽出し、キャッシュとして保存する。
"""

import sys
import time
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm
import verify_cos_decomposition as vcd

ROOT_DIR = vcd.ROOT_DIR
WORKSPACE_DIR = vcd.CURRENT_DIR
OUT_CACHE = WORKSPACE_DIR / "scratch" / "local_scalar_speed_cache.pkl"

RAD_MAP = {
    'beads06um': 3,
    'beads1um': 6,
    'beads3um': 15,
    'beads5um': 25,
    'beads7um': 35,
    'beads20um': 95
}
BEST_MAP = {
    'beads06um': 5,
    'beads1um': 10,
    'beads3um': 20,
    'beads5um': 30,
    'beads7um': 40,
    'beads20um': 100
}

def main():
    df_inst = pd.read_pickle(WORKSPACE_DIR / "scratch" / "contrast_inst_cache.pkl")
    print(f"Loaded df_inst with {len(df_inst)} rows.")
    
    local_speeds = []
    
    # Process experiment by experiment
    grouped_exp = df_inst.groupby('exp_dir')
    print(f"Processing {len(grouped_exp)} experiments...")
    
    for exp_name, df_exp in tqdm(grouped_exp, desc="Experiments"):
        bname = df_exp['bead_name'].iloc[0]
        p_rad = RAD_MAP[bname]
        best_rad = BEST_MAP[bname]
        r_ceil = best_rad
        
        # Find exp dir
        exp_dirs = vcd.find_experiment_dirs(ROOT_DIR, bname)
        ed_list = [d for d in exp_dirs if d.name == exp_name]
        if not ed_list:
            print(f"Warning: exp_dir not found for {exp_name}")
            local_speeds.extend([np.nan] * len(df_exp))
            continue
        ed = ed_list[0]
        flow_p = ed / "GFP_flows.h5"
        
        if not flow_p.exists():
            print(f"Warning: GFP_flows.h5 not found in {ed}")
            local_speeds.extend([np.nan] * len(df_exp))
            continue
            
        with h5py.File(flow_p, 'r') as f:
            flow = f[list(f.keys())[0]]
            h, w = flow.shape[2], flow.shape[3]
            
            # Group by frame to minimize HDF5 reads
            exp_results = {}
            for frm, grp_frm in df_exp.groupby('frame'):
                frm = int(frm)
                if frm >= flow.shape[0]:
                    for idx in grp_frm.index:
                        exp_results[idx] = np.nan
                    continue
                
                # Read flow at frame once
                fu = flow[frm, 0, :, :].astype(np.float32)
                fv = flow[frm, 1, :, :].astype(np.float32)
                
                for idx, r in grp_frm.iterrows():
                    bx = int(round(r['x_px']))
                    by = int(round(r['y_px']))
                    
                    y_min = max(0, by - r_ceil)
                    y_max = min(h, by + r_ceil + 1)
                    x_min = max(0, bx - r_ceil)
                    x_max = min(w, bx + r_ceil + 1)
                    
                    sub_u = fu[y_min:y_max, x_min:x_max]
                    sub_v = fv[y_min:y_max, x_min:x_max]
                    
                    yg, xg = np.ogrid[y_min-by:y_max-by, x_min-bx:x_max-bx]
                    dsq = xg**2 + yg**2
                    m = (dsq <= best_rad**2) & (dsq > p_rad**2)
                    
                    if np.any(m):
                        spd = float(np.mean(np.hypot(sub_u[m], sub_v[m])))
                    else:
                        spd = np.nan
                    exp_results[idx] = spd
                    
            # Put back in original order
            for idx in df_exp.index:
                local_speeds.append(exp_results[idx])
                
    df_inst['local_flow_spd_px'] = local_speeds
    df_inst.to_pickle(OUT_CACHE)
    print(f"Saved local scalar speed cache to {OUT_CACHE}")
    print(df_inst[['bead_name', 'flow_spd', 'local_flow_spd_px']].describe())

if __name__ == '__main__':
    main()

