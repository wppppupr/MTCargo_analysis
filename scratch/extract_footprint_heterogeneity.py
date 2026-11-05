#!/usr/bin/env python3
"""
scratch/extract_footprint_heterogeneity.py

全22実験・全アクティブ粒子の瞬時フレーム（27,969行）について、
各フレームにおける足印（アニュラス）内のオプティカルフローから以下を高速抽出・保存する：
1. 局所極性秩序 Phi_t = || <u / |u|>_{Rc} || (方向の揃い具合: 0〜1)
2. 逆向きレーン占有率 f_opp = fraction( u . <u_Rc> < 0 ) (対向流の混ざり度: 0〜0.5)
3. 足印平均流速 |<u_Rc>| および ピクセルスカラー流速 <|u|>_{Rc}
4. 貨物速度の足元流向成分 v_parallel = v_c . u_hat_fp
"""

import sys
import time
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import pandas as pd
import h5py

CURRENT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(CURRENT_DIR))
import verify_cos_decomposition as vcd

ROOT_DIR = vcd.ROOT_DIR
WORKSPACE_DIR = CURRENT_DIR
OUT_CACHE = WORKSPACE_DIR / "scratch" / "footprint_heterogeneity_cache.pkl"

SCALE_UM_PER_PX = 0.11
DT_SEC = 4.0
V_UNIT = SCALE_UM_PER_PX / DT_SEC

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


def process_experiment(exp_name_and_bname):
    exp_name, bname, exp_rows = exp_name_and_bname
    t0 = time.time()
    
    p_rad = RAD_MAP[bname]
    best_rad = BEST_MAP[bname]
    r_ceil = best_rad

    exp_dirs = vcd.find_experiment_dirs(ROOT_DIR, bname)
    ed_list = [d for d in exp_dirs if d.name == exp_name]
    if not ed_list:
        print(f"[{exp_name}] Dir not found.")
        return []
    ed = ed_list[0]
    flow_p = ed / "GFP_flows.h5"
    if not flow_p.exists():
        print(f"[{exp_name}] Flow file not found: {flow_p}")
        return []

    # Group by frame to minimize HDF5 chunk reads
    frame_groups = exp_rows.groupby('frame')
    results = []

    # Enable 128MB chunk cache so the 44MB frame chunk stays in RAM for all particles in this frame
    with h5py.File(flow_p, 'r', rdcc_nbytes=128 * 1024 * 1024) as f:
        flow = f[list(f.keys())[0]]
        total_frms = flow.shape[0]
        h, w = flow.shape[2], flow.shape[3]

        for frm, grp in frame_groups:
            frm_int = int(frm)
            if frm_int >= total_frms:
                continue

            for _, r in grp.iterrows():
                bx = int(round(r['x_px']))
                by = int(round(r['y_px']))

                y_min = max(0, by - r_ceil)
                y_max = min(h, by + r_ceil + 1)
                x_min = max(0, bx - r_ceil)
                x_max = min(w, bx + r_ceil + 1)

                patch = flow[frm_int, :, y_min:y_max, x_min:x_max]  # shape: (2, dy, dx)
                u_box = patch[0]
                v_box = patch[1]

                yg, xg = np.ogrid[y_min - by:y_max - by, x_min - bx:x_max - bx]
                dsq = xg**2 + yg**2
                m = (dsq <= best_rad**2) & (dsq > p_rad**2)

                if not np.any(m):
                    continue

                u_in = u_box[m].astype(np.float32)
                v_in = v_box[m].astype(np.float32)
                norm_in = np.hypot(u_in, v_in)
                valid = norm_in > 0

                if not np.any(valid):
                    continue

                # 1. Local polar order Phi_t (direction only)
                u_hat = u_in[valid] / norm_in[valid]
                v_hat = v_in[valid] / norm_in[valid]
                phi_t = float(np.hypot(np.mean(u_hat), np.mean(v_hat)))

                # 2. Footprint mean velocity vector
                u_bar = float(np.mean(u_in))
                v_bar = float(np.mean(v_in))
                u_fp_px = float(np.hypot(u_bar, v_bar))

                # 3. Opposite flow fraction f_opp
                if u_fp_px > 0:
                    dot = u_in * u_bar + v_in * v_bar
                    f_opp = float(np.mean(dot < 0))
                    # v_parallel in um/s
                    u_hat_fp = u_bar / u_fp_px
                    v_hat_fp = v_bar / u_fp_px
                    v_par_um = float((r['cargo_dx'] * u_hat_fp + r['cargo_dy'] * v_hat_fp) * V_UNIT)
                else:
                    f_opp = 0.0
                    v_par_um = 0.0

                # 4. Local scalar speed in um/s
                u_local_scalar_um = float(np.mean(norm_in) * V_UNIT)
                u_fp_um = float(u_fp_px * V_UNIT)
                v_cargo_um = float(np.hypot(r['cargo_dx'], r['cargo_dy']) * V_UNIT)

                results.append({
                    'bead_name': bname,
                    'diameter_um': r['diameter_um'],
                    'radius_um': r['radius_um'],
                    'rc_over_xi': r['rc_over_xi'],
                    'exp_dir': exp_name,
                    'particle': r['particle'],
                    'unique_pid': f"{exp_name}_{r['particle']}",
                    'frame': frm_int,
                    'phi_t': phi_t,
                    'f_opp': f_opp,
                    'v_parallel': v_par_um,
                    'u_fp': u_fp_um,
                    'u_local_scalar': u_local_scalar_um,
                    'v_cargo': v_cargo_um,
                })

    t1 = time.time()
    print(f"[{exp_name}] Finished {len(results)} records in {t1 - t0:.1f} s")
    return results


def main():
    t_start = time.time()
    inst_cache = WORKSPACE_DIR / "scratch" / "contrast_inst_cache.pkl"
    if not inst_cache.exists():
        print(f"Error: {inst_cache} not found")
        sys.exit(1)

    df_inst = pd.read_pickle(inst_cache)
    print(f"Loaded {len(df_inst)} instantaneous records from cache.")

    exp_tasks = []
    for exp_name, grp in df_inst.groupby('exp_dir'):
        bname = grp['bead_name'].iloc[0]
        exp_tasks.append((exp_name, bname, grp.copy()))

    print(f"Processing {len(exp_tasks)} experiments using 6 parallel workers...")
    with Pool(processes=6) as pool:
        nested_results = pool.map(process_experiment, exp_tasks)

    flat_results = [item for sublist in nested_results for item in sublist]
    df_res = pd.DataFrame(flat_results)
    df_res.to_pickle(OUT_CACHE)

    t_end = time.time()
    print(f"\nExtraction complete in {t_end - t_start:.1f} s!")
    print(f"Saved {len(df_res)} records to {OUT_CACHE}")
    print("\nSummary per bead size:")
    print(df_res.groupby('bead_name')[['phi_t', 'f_opp', 'u_fp', 'v_parallel']].mean())


if __name__ == '__main__':
    main()
