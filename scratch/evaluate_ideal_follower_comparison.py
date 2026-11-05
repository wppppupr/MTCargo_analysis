#!/usr/bin/env python3
"""
scratch/evaluate_ideal_follower_comparison.py

実測貨物の運動と「同じ流れの中を、同じ方法で測った理想的な追従者」の伝達効率比 eta の厳密検証。

1. 幾何学的領域の定義:
   - 点追従 (Point): 粒子中心位置の局所流速 u(r=0)
   - 円盤領域 (Disc): 粒子直下 r <= Rc (Footprint)
   - 円環領域 (Annulus): 粒子直下をマスクした外周 Rc < r <= Rout (実際の測定幾何)

2. 算出する物理量:
   - 幾何学的相殺係数 A = <|<u|_{geom}>| / <|u|>_{geom}
   - 実測貨物の伝達効率 eta_real = <v_parallel> / <|<u|_{geom}>|
   - 理想追従者の伝達効率 eta_virt = <v_virt_parallel> / <|<u|_{geom}>| = 1.000
   - 速さ比: |v_real| / |u_geom| および |v_virt| / |u_geom|
   - 粒子単位のブートストラップ (N=2000) による統計誤差 (SEM, 95% CI)
"""

import sys
import time
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

CURRENT_DIR = Path("/home/sasaki/MTCargo_analysis")
sys.path.append(str(CURRENT_DIR))
import verify_cos_decomposition as vcd

ROOT_DIR = vcd.ROOT_DIR
SCALE = 0.11
DT = 4.0
V_UNIT = SCALE / DT

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

def process_experiment(args):
    exp_name, df_exp = args
    bname = df_exp['bead_name'].iloc[0]
    p_rad = RAD_MAP[bname]
    best_rad = BEST_MAP[bname]
    r_ceil = best_rad
    
    exp_dirs = vcd.find_experiment_dirs(ROOT_DIR, bname)
    ed_list = [d for d in exp_dirs if d.name == exp_name]
    if not ed_list:
        return []
    ed = ed_list[0]
    flow_p = ed / "GFP_flows.h5"
    if not flow_p.exists():
        return []
        
    records = []
    with h5py.File(flow_p, 'r') as f:
        flow = f[list(f.keys())[0]]
        h, w = flow.shape[2], flow.shape[3]
        
        for frm, grp_frm in df_exp.groupby('frame'):
            frm = int(frm)
            if frm >= flow.shape[0]:
                continue
            fu = flow[frm, 0, :, :].astype(np.float32)
            fv = flow[frm, 1, :, :].astype(np.float32)
            
            for idx, r in grp_frm.iterrows():
                bx = float(r['x_px'])
                by = float(r['y_px'])
                ix = int(round(bx))
                iy = int(round(by))
                
                # 1. Point at center
                if 0 <= ix < w and 0 <= iy < h:
                    pt_u = float(fu[iy, ix])
                    pt_v = float(fv[iy, ix])
                else:
                    pt_u, pt_v = 0.0, 0.0
                    
                # Bounding box
                y_min = max(0, int(np.floor(by - r_ceil)))
                y_max = min(h, int(np.ceil(by + r_ceil)) + 1)
                x_min = max(0, int(np.floor(bx - r_ceil)))
                x_max = min(w, int(np.ceil(bx + r_ceil)) + 1)
                
                sub_u = fu[y_min:y_max, x_min:x_max]
                sub_v = fv[y_min:y_max, x_min:x_max]
                sub_spd = np.hypot(sub_u, sub_v)
                
                yg, xg = np.ogrid[y_min:y_max, x_min:x_max]
                dsq = (xg - bx)**2 + (yg - by)**2
                
                # 2. Disc (r <= p_rad)
                m_disc = dsq <= p_rad**2
                if np.any(m_disc):
                    disc_u = float(np.mean(sub_u[m_disc]))
                    disc_v = float(np.mean(sub_v[m_disc]))
                    disc_sc = float(np.mean(sub_spd[m_disc]))
                else:
                    disc_u, disc_v, disc_sc = pt_u, pt_v, np.hypot(pt_u, pt_v)
                    
                # 3. Annulus (p_rad < r <= best_rad)
                m_ann = (dsq <= best_rad**2) & (dsq > p_rad**2)
                if np.any(m_ann):
                    ann_u = float(np.mean(sub_u[m_ann]))
                    ann_v = float(np.mean(sub_v[m_ann]))
                    ann_sc = float(np.mean(sub_spd[m_ann]))
                else:
                    ann_u, ann_v, ann_sc = disc_u, disc_v, disc_sc
                    
                records.append({
                    'index': idx,
                    'pt_u': pt_u,
                    'pt_v': pt_v,
                    'disc_u': disc_u,
                    'disc_v': disc_v,
                    'disc_sc': disc_sc,
                    'ann_u': ann_u,
                    'ann_v': ann_v,
                    'ann_sc': ann_sc
                })
    return records

def main():
    inst_cache = CURRENT_DIR / "scratch" / "contrast_inst_cache.pkl"
    df_inst = pd.read_pickle(inst_cache)
    df_inst['unique_pid'] = df_inst['exp_dir'] + '_' + df_inst['particle'].astype(str)
    
    print(f"Extracting Point, Disc, and Annulus flow fields for {len(df_inst)} rows across {df_inst['exp_dir'].nunique()} experiments...")
    
    tasks = [(exp_name, df_exp) for exp_name, df_exp in df_inst.groupby('exp_dir')]
    with Pool(processes=min(10, cpu_count())) as pool:
        results_nested = pool.map(process_experiment, tasks)
        
    flat_results = [r for sublist in results_nested for r in sublist]
    df_extracted = pd.DataFrame(flat_results).set_index('index')
    
    df_merged = df_inst.join(df_extracted, how='inner')
    print(f"Successfully processed {len(df_merged)} rows.")
    
    # 速度・平行成分・相殺係数の計算
    # 貨物速度 (um/s)
    c_dx_um = df_merged['cargo_dx'] * V_UNIT
    c_dy_um = df_merged['cargo_dy'] * V_UNIT
    c_spd_um = np.hypot(c_dx_um, c_dy_um)
    df_merged['cargo_spd_um'] = c_spd_um
    
    # 幾何ごとの量
    geoms = ['pt', 'disc', 'ann']
    for g in geoms:
        gu_um = df_merged[f'{g}_u'] * V_UNIT
        gv_um = df_merged[f'{g}_v'] * V_UNIT
        g_vec_spd = np.hypot(gu_um, gv_um)
        df_merged[f'{g}_vec_spd_um'] = g_vec_spd
        
        # 平行成分 v_par = v_c . u_hat_g
        # ゼロ割りを安全に処理
        valid = g_vec_spd > 0
        u_hat_x = np.zeros_like(gu_um)
        u_hat_y = np.zeros_like(gv_um)
        u_hat_x[valid] = gu_um[valid] / g_vec_spd[valid]
        u_hat_y[valid] = gv_um[valid] / g_vec_spd[valid]
        
        df_merged[f'{g}_v_par_um'] = c_dx_um * u_hat_x + c_dy_um * u_hat_y
        
        if g in ['disc', 'ann']:
            df_merged[f'{g}_sc_spd_um'] = df_merged[f'{g}_sc'] * V_UNIT
            
    # ブートストラップによるサイズ別集計
    np.random.seed(42)
    N_BOOT = 2000
    
    summary_list = []
    
    print("\n" + "="*120)
    print("=== IDEAL FOLLOWER COMPARISON: Real Cargo vs Ideal Tracer under Point, Disc, and Annulus Measurement ===")
    print("="*120)
    
    for b_item in vcd.BEADS_INFO:
        bname = b_item['name']
        d_um = b_item['diameter_um']
        r_um = b_item['radius_um']
        sub = df_merged[df_merged['bead_name'] == bname]
        if len(sub) == 0:
            continue
            
        p_means = sub.groupby('unique_pid', numeric_only=True).mean()
        n_p = len(p_means)
        p_idx = np.arange(n_p)
        
        # 観測量
        # 1. Point
        pt_v_par = p_means['pt_v_par_um'].values
        pt_u_vec = p_means['pt_vec_spd_um'].values
        
        # 2. Disc
        disc_v_par = p_means['disc_v_par_um'].values
        disc_u_vec = p_means['disc_vec_spd_um'].values
        disc_u_sc = p_means['disc_sc_spd_um'].values
        
        # 3. Annulus
        ann_v_par = p_means['ann_v_par_um'].values
        ann_u_vec = p_means['ann_vec_spd_um'].values
        ann_u_sc = p_means['ann_sc_spd_um'].values
        
        # Cargo speed
        c_spd = p_means['cargo_spd_um'].values
        
        # Point estimates
        eta_pt = np.mean(pt_v_par) / np.mean(pt_u_vec)
        spd_ratio_pt = np.mean(c_spd) / np.mean(pt_u_vec)
        
        eta_disc = np.mean(disc_v_par) / np.mean(disc_u_vec)
        A_disc = np.mean(disc_u_vec) / np.mean(disc_u_sc)
        spd_ratio_disc = np.mean(c_spd) / np.mean(disc_u_vec)
        
        eta_ann = np.mean(ann_v_par) / np.mean(ann_u_vec)
        A_ann = np.mean(ann_u_vec) / np.mean(ann_u_sc)
        spd_ratio_ann = np.mean(c_spd) / np.mean(ann_u_vec)
        
        # Bootstrap
        b_eta_pt, b_eta_disc, b_eta_ann = [], [], []
        b_A_disc, b_A_ann = [], []
        b_spd_disc, b_spd_ann = [], []
        
        for _ in range(N_BOOT):
            idx = np.random.choice(p_idx, size=n_p, replace=True)
            
            b_eta_pt.append(np.mean(pt_v_par[idx]) / np.mean(pt_u_vec[idx]))
            
            m_dv = np.mean(disc_v_par[idx])
            m_du = np.mean(disc_u_vec[idx])
            m_ds = np.mean(disc_u_sc[idx])
            b_eta_disc.append(m_dv / m_du)
            b_A_disc.append(m_du / m_ds)
            b_spd_disc.append(np.mean(c_spd[idx]) / m_du)
            
            m_av = np.mean(ann_v_par[idx])
            m_au = np.mean(ann_u_vec[idx])
            m_as = np.mean(ann_u_sc[idx])
            b_eta_ann.append(m_av / m_au)
            b_A_ann.append(m_au / m_as)
            b_spd_ann.append(np.mean(c_spd[idx]) / m_au)
            
        summary_list.append({
            'bead_name': bname,
            'diameter_um': d_um,
            'radius_um': r_um,
            'n_particles': n_p,
            # Point
            'eta_pt': eta_pt,
            'eta_pt_sem': np.std(b_eta_pt),
            'spd_ratio_pt': spd_ratio_pt,
            # Disc
            'A_disc': A_disc,
            'A_disc_sem': np.std(b_A_disc),
            'eta_disc': eta_disc,
            'eta_disc_sem': np.std(b_eta_disc),
            'spd_ratio_disc': spd_ratio_disc,
            'spd_ratio_disc_sem': np.std(b_spd_disc),
            # Annulus
            'A_ann': A_ann,
            'A_ann_sem': np.std(b_A_ann),
            'eta_ann': eta_ann,
            'eta_ann_sem': np.std(b_eta_ann),
            'spd_ratio_ann': spd_ratio_ann,
            'spd_ratio_ann_sem': np.std(b_spd_ann),
        })
        
    df_sum = pd.DataFrame(summary_list)
    out_path = CURRENT_DIR / "scratch" / "ideal_follower_comparison_summary.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(df_sum, f)
    df_sum.to_csv(CURRENT_DIR / "figure" / "contrast_analysis" / "ideal_follower_comparison_summary.csv", index=False)
    
    print("\nSummary Results Table:")
    cols_show = ['bead_name', 'diameter_um', 'A_disc', 'eta_disc', 'A_ann', 'eta_ann', 'spd_ratio_ann']
    print(df_sum[cols_show].to_string(index=False))

if __name__ == '__main__':
    main()

