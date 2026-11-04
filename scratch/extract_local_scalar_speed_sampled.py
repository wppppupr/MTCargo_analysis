#!/usr/bin/env python3
"""
scratch/extract_local_scalar_speed_sampled.py

全71粒子について、各粒子の追跡フレームから最大50フレームを等間隔サンプリングし、
足元アニュラス（環状シェル）領域内のピクセルスカラー速さの平均 <|u|>_local を
HDF5 から直接抽出して粒子ごとの平均局所スカラー流速 <|u|>_local,i を算出する。
"""

import sys
import time
from pathlib import Path
from multiprocessing import Pool
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import h5py
import verify_cos_decomposition as vcd

ROOT_DIR = vcd.ROOT_DIR
WORKSPACE_DIR = vcd.CURRENT_DIR
OUT_CACHE = WORKSPACE_DIR / "scratch" / "particle_local_scalar_speed_summary.pkl"

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


def process_particle(args):
    unique_pid, exp_name, pid, bname, sub_rows = args
    p_rad = RAD_MAP[bname]
    best_rad = BEST_MAP[bname]
    r_ceil = best_rad

    exp_dirs = vcd.find_experiment_dirs(ROOT_DIR, bname)
    ed_list = [d for d in exp_dirs if d.name == exp_name]
    if not ed_list:
        return unique_pid, np.nan, np.nan
    ed = ed_list[0]
    flow_p = ed / "GFP_flows.h5"
    if not flow_p.exists():
        return unique_pid, np.nan, np.nan

    # Sample up to 50 frames uniformly
    n_pts = len(sub_rows)
    if n_pts > 50:
        step = n_pts // 50
        sample_rows = sub_rows.iloc[::step].head(50)
    else:
        sample_rows = sub_rows

    speeds = []
    flow_vec_speeds = []
    
    with h5py.File(flow_p, 'r') as f:
        flow = f[list(f.keys())[0]]
        h, w = flow.shape[2], flow.shape[3]

        for _, r in sample_rows.iterrows():
            frm = int(r['frame'])
            if frm >= flow.shape[0]:
                continue
            bx = int(round(r['x_px']))
            by = int(round(r['y_px']))

            y_min = max(0, by - r_ceil)
            y_max = min(h, by + r_ceil + 1)
            x_min = max(0, bx - r_ceil)
            x_max = min(w, bx + r_ceil + 1)

            fu = flow[frm, 0, y_min:y_max, x_min:x_max].astype(np.float32)
            fv = flow[frm, 1, y_min:y_max, x_min:x_max].astype(np.float32)

            yg, xg = np.ogrid[y_min-by:y_max-by, x_min-bx:x_max-bx]
            dsq = xg**2 + yg**2
            m = (dsq <= best_rad**2) & (dsq > p_rad**2)

            if np.any(m):
                u_in, v_in = fu[m], fv[m]
                spd_scalar = float(np.mean(np.hypot(u_in, v_in)))
                spd_vec = float(np.hypot(np.mean(u_in), np.mean(v_in)))
                speeds.append(spd_scalar)
                flow_vec_speeds.append(spd_vec)

    mean_scalar = float(np.mean(speeds)) if speeds else np.nan
    mean_vec = float(np.mean(flow_vec_speeds)) if flow_vec_speeds else np.nan
    return unique_pid, mean_scalar, mean_vec


def main():
    t0 = time.time()
    df_inst = pd.read_pickle(WORKSPACE_DIR / "scratch" / "contrast_inst_cache.pkl")
    df_inst['unique_pid'] = df_inst['exp_dir'] + '_' + df_inst['particle'].astype(str)

    tasks = []
    for (exp_name, pid), grp in df_inst.groupby(['exp_dir', 'particle']):
        unique_pid = f"{exp_name}_{pid}"
        bname = grp['bead_name'].iloc[0]
        tasks.append((unique_pid, exp_name, pid, bname, grp[['frame', 'x_px', 'y_px']].copy()))

    print(f"Extracting local scalar speeds for {len(tasks)} particles using 8 workers...")
    with Pool(processes=8) as pool:
        results = pool.map(process_particle, tasks)

    df_p_speeds = pd.DataFrame(results, columns=['unique_pid', 'local_scalar_spd_px', 'local_vec_spd_px'])
    df_p_speeds.to_pickle(OUT_CACHE)
    t1 = time.time()
    print(f"Extraction completed in {t1-t0:.2f} s. Saved to {OUT_CACHE}")

    # Merge with particle metadata
    p_meta = df_inst.groupby('unique_pid')[['bead_name', 'diameter_um', 'radius_um', 'rc_over_xi', 'flow_spd']].mean().reset_index()
    m = pd.merge(p_meta, df_p_speeds, on='unique_pid')
    m['A_local'] = m['local_vec_spd_px'] / m['local_scalar_spd_px']

    print("\n=== Particle-level Local A Summary ===")
    for b, grp in m.groupby('bead_name'):
        print(f"{b:10s} (d={grp['diameter_um'].iloc[0]:5.2f} um, Rc/xi={grp['rc_over_xi'].mean():.3f}): A_local = {grp['A_local'].mean():.4f} +/- {grp['A_local'].std():.4f} (N={len(grp)})")


if __name__ == '__main__':
    main()

