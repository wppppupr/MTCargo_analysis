#!/usr/bin/env python3
"""
scratch/compute_local_scalar_speed_parallel.py

マルチプロセス並列（6ワーカー）で全22実験の足元アニュラス（環状シェル）内の
ピクセルスカラー速さの平均 <|u|>_local を HDF5 から直接抽出し、キャッシュ化する。
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


def process_single_experiment(args):
    exp_name, sub_indices, sub_data = args
    bname = sub_data['bead_name'].iloc[0]
    p_rad = RAD_MAP[bname]
    best_rad = BEST_MAP[bname]
    r_ceil = best_rad

    exp_dirs = vcd.find_experiment_dirs(ROOT_DIR, bname)
    ed_list = [d for d in exp_dirs if d.name == exp_name]
    if not ed_list:
        return {idx: np.nan for idx in sub_indices}
    ed = ed_list[0]
    flow_p = ed / "GFP_flows.h5"
    if not flow_p.exists():
        return {idx: np.nan for idx in sub_indices}

    results = {}
    with h5py.File(flow_p, 'r') as f:
        flow = f[list(f.keys())[0]]
        h, w = flow.shape[2], flow.shape[3]

        # Group by frame
        for frm, grp_frm in sub_data.groupby('frame'):
            frm = int(frm)
            if frm >= flow.shape[0]:
                for idx in grp_frm.index:
                    results[idx] = np.nan
                continue

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
                    results[idx] = float(np.mean(np.hypot(sub_u[m], sub_v[m])))
                else:
                    results[idx] = np.nan

    return results


def main():
    t0 = time.time()
    df_inst = pd.read_pickle(WORKSPACE_DIR / "scratch" / "contrast_inst_cache.pkl")
    print(f"Loaded df_inst with {len(df_inst)} rows.")

    tasks = []
    for exp_name, grp in df_inst.groupby('exp_dir'):
        tasks.append((exp_name, grp.index.tolist(), grp[['bead_name', 'frame', 'x_px', 'y_px']].copy()))

    print(f"Running parallel extraction across {len(tasks)} experiments using 6 workers...")
    all_results = {}
    with Pool(processes=6) as pool:
        for res_dict in pool.imap_unordered(process_single_experiment, tasks):
            all_results.update(res_dict)

    # Assign to df_inst
    df_inst['local_flow_spd_px'] = [all_results.get(idx, np.nan) for idx in df_inst.index]
    df_inst.to_pickle(OUT_CACHE)
    t1 = time.time()
    print(f"Completed in {t1-t0:.1f} s. Saved to {OUT_CACHE}")

    # Summary
    sub_valid = df_inst.dropna(subset=['local_flow_spd_px']).copy()
    sub_valid['ratio_A'] = sub_valid['flow_spd'] / sub_valid['local_flow_spd_px']
    for b, grp in sub_valid.groupby('bead_name'):
        print(f"{b}: local_flow_spd_px mean={grp['local_flow_spd_px'].mean():.4f}, flow_spd mean={grp['flow_spd'].mean():.4f}, ratio A = {grp['ratio_A'].mean():.4f} +/- {grp['ratio_A'].std():.4f}")


if __name__ == '__main__':
    main()

