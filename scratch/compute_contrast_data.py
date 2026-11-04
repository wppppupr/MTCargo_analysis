#!/usr/bin/env python3
"""
scratch/compute_contrast_data.py
全ビーズサイズ・全アクティブ粒子を対象に:
1. 実測軌道と仮想軌道 (Footprint Advection) の抽出・ペア生成
2. 各粒子の iMSD (実測 & 仮想)
3. 変位分布 PDF (tau = 1, 4, 16 frames: 4s, 16s, 64s)
4. 正規化結合度 <cos psi> と従来 <cos theta_i>
5. 時間シャッフルヌルモデル & 粒子スワップヌルモデル
6. 速度伝達係数 kappa (線形回帰)
を一括計算し、結果を pickle に保存する。
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from tqdm import tqdm
import verify_cos_decomposition as vcd

ROOT_DIR = vcd.ROOT_DIR
WORKSPACE_DIR = vcd.CURRENT_DIR
BEADS_INFO = vcd.BEADS_INFO
SCALE = vcd.SCALE_UM_PER_PX  # 0.11 um/px
DT = 4.0  # s / frame

def calc_segment_msd(pos: np.ndarray, max_tau: int = 150) -> Tuple[np.ndarray, np.ndarray]:
    """1本の連続軌道 pos (N, 2) から時間平均 MSD を計算する。"""
    n_pts = len(pos)
    if n_pts < 3:
        return np.array([]), np.array([])
    taus = np.arange(1, min(n_pts, max_tau + 1))
    msd_vals = []
    for tau in taus:
        disp = pos[tau:] - pos[:-tau]
        sq_disp = np.sum(disp**2, axis=1)
        msd_vals.append(np.mean(sq_disp))
    return taus, np.array(msd_vals)

def run_extraction():
    active_set = vcd.load_active_particles(WORKSPACE_DIR)
    xi_dict = vcd.load_experiment_correlation_lengths(ROOT_DIR, WORKSPACE_DIR)
    
    print(f"Loaded {len(active_set)} active particles, {len(xi_dict)} xi entries.")
    
    # 全瞬時データ (速度、角度、流速)
    instant_records = []
    # 軌道ペアデータ (粒子ごとの実測 & 仮想 iMSD)
    imsd_records = []
    # 変位サンプリング (PDF 用)
    disp_records = {'real': {b['name']: {1: [], 4: [], 16: []} for b in BEADS_INFO},
                    'virt': {b['name']: {1: [], 4: [], 16: []} for b in BEADS_INFO}}
    
    for b_item in BEADS_INFO:
        bname = b_item['name']
        d_um = b_item['diameter_um']
        r_um = b_item['radius_um']
        target_rad_px = r_um / SCALE
        
        exp_dirs = vcd.find_experiment_dirs(ROOT_DIR, bname)
        print(f"\nProcessing {bname} ({len(exp_dirs)} exps)...")
        
        for exp_dir in exp_dirs:
            exp_name = exp_dir.name
            xi_val = xi_dict.get(exp_name, np.nan)
            
            csv_std = exp_dir / "beads_flow_interaction.csv"
            csv_gau = exp_dir / "beads_flow_interaction_gaussian.csv"
            tracks_p = exp_dir / "beads_tracks.csv"
            
            if csv_std.exists():
                df_flow = pd.read_csv(csv_std)
                rad_col = "radius"
            elif csv_gau.exists():
                df_flow = pd.read_csv(csv_gau)
                rad_col = "sigma"
            else:
                continue
                
            if not tracks_p.exists():
                continue
            df_tracks = pd.read_csv(tracks_p)
            
            # 貨物速度マップ (verify_cos_decomposition と同様に dot_product から復元)
            cargo_velocity_map = {}
            for (frm, pid), grp_fp in df_flow.groupby(['frame', 'particle']):
                grp_v = grp_fp[(grp_fp['mean_flow_u'] != 0) | (grp_fp['mean_flow_v'] != 0)]
                if len(grp_v) >= 2:
                    u_vals = grp_v['mean_flow_u'].values
                    v_vals = grp_v['mean_flow_v'].values
                    d_vals = grp_v['dot_product'].values
                    A = np.column_stack([u_vals, v_vals])
                    res, _, _, _ = np.linalg.lstsq(A, d_vals, rcond=None)
                    bdx, bdy = float(res[0]), float(res[1])
                    b_spd = float(np.hypot(bdx, bdy))
                    if b_spd > 0:
                        cargo_velocity_map[(int(pid), int(frm))] = (bdx, bdy, b_spd)
                        
            # 最寄り有効半径
            df_flow['rad_px'] = df_flow[rad_col].astype(float)
            valid_mask = (df_flow['mean_flow_u'] != 0) | (df_flow['mean_flow_v'] != 0) | (df_flow['cos_sim'] != 0)
            valid_rads = np.sort(df_flow.loc[valid_mask, 'rad_px'].unique())
            if len(valid_rads) == 0:
                continue
            best_idx = int(np.argmin(np.abs(valid_rads - target_rad_px)))
            best_rad = float(valid_rads[best_idx])
            
            sub_best = df_flow[(df_flow['rad_px'] == best_rad) & valid_mask].copy()
            
            # 各粒子について処理
            for pid, p_flow in sub_best.groupby('particle'):
                pid = int(pid)
                if active_set and (exp_name, pid) not in active_set:
                    continue
                p_flow = p_flow.sort_values('frame')
                p_track = df_tracks[df_tracks['particle'] == pid].sort_values('frame')
                if len(p_track) < 3:
                    continue
                    
                # tracks と flow をマージ
                merged = pd.merge(p_track[['frame', 'x', 'y']], 
                                  p_flow[['frame', 'mean_flow_u', 'mean_flow_v', 'cos_sim', 'dot_product']], 
                                  on='frame', how='inner').sort_values('frame')
                if len(merged) < 3:
                    continue
                    
                # 速度ベクトル復元
                c_dx_list = []
                c_dy_list = []
                c_spd_list = []
                for _, r in merged.iterrows():
                    frm = int(r['frame'])
                    if (pid, frm) in cargo_velocity_map:
                        cv = cargo_velocity_map[(pid, frm)]
                    else:
                        cv = (np.nan, np.nan, np.nan)
                    c_dx_list.append(cv[0])
                    c_dy_list.append(cv[1])
                    c_spd_list.append(cv[2])
                    
                merged['cargo_dx'] = c_dx_list
                merged['cargo_dy'] = c_dy_list
                merged['cargo_spd'] = c_spd_list
                
                # 有効行フィルタ
                merged_valid = merged.dropna(subset=['cargo_dx', 'cargo_dy', 'cargo_spd']).copy()
                if len(merged_valid) < 3:
                    continue
                    
                # 瞬時レコード追加
                for _, r in merged_valid.iterrows():
                    cdx = r['cargo_dx']
                    cdy = r['cargo_dy']
                    cspd = r['cargo_spd']
                    u = r['mean_flow_u']
                    v = r['mean_flow_v']
                    v_fn = np.hypot(u, v)
                    if cspd <= 0 or v_fn <= 0:
                        continue
                    cb = (cdx * u + cdy * v) / (cspd * v_fn)
                    ci = r['cos_sim']
                    instant_records.append({
                        'bead_name': bname,
                        'diameter_um': d_um,
                        'radius_um': r_um,
                        'exp_dir': exp_name,
                        'particle': pid,
                        'frame': int(r['frame']),
                        'x_px': r['x'],
                        'y_px': r['y'],
                        'cargo_dx': cdx,
                        'cargo_dy': cdy,
                        'cargo_spd': cspd,
                        'flow_u': u,
                        'flow_v': v,
                        'flow_spd': v_fn,
                        'cos_i': ci,
                        'cos_bulk': cb,
                        'xi': xi_val,
                        'rc_over_xi': r_um / xi_val if np.isfinite(xi_val) else np.nan,
                    })
                    
                # 連続セグメントに分割して仮想軌道 & 実測軌道から MSD / PDF 計算
                merged_valid['f_diff'] = merged_valid['frame'].diff()
                seg_id = (merged_valid['f_diff'] != 1).cumsum()
                
                for s_id, seg in merged_valid.groupby(seg_id):
                    if len(seg) < 4:
                        continue
                    x_r = seg['x'].values * SCALE
                    y_r = seg['y'].values * SCALE
                    pos_r = np.column_stack([x_r, y_r])
                    
                    # 仮想変位 (各時刻の足印流速)
                    u_v = seg['mean_flow_u'].values * SCALE
                    v_v = seg['mean_flow_v'].values * SCALE
                    # 仮想位置 (実測初期位置からの積算)
                    # 注: 各フレームの変位が u_v, v_v なので、
                    # x_virt[k] = x_r[0] + sum_{j=0}^{k-1} u_v[j]
                    x_virt = np.empty(len(seg))
                    y_virt = np.empty(len(seg))
                    x_virt[0] = x_r[0]
                    y_virt[0] = y_r[0]
                    x_virt[1:] = x_r[0] + np.cumsum(u_v[:-1])
                    y_virt[1:] = y_r[0] + np.cumsum(v_v[:-1])
                    pos_v = np.column_stack([x_virt, y_virt])
                    
                    # MSD 計算
                    taus_r, msd_r = calc_segment_msd(pos_r, max_tau=100)
                    taus_v, msd_v = calc_segment_msd(pos_v, max_tau=100)
                    
                    if len(taus_r) > 0:
                        for tau_idx, tau in enumerate(taus_r):
                            imsd_records.append({
                                'bead_name': bname,
                                'diameter_um': d_um,
                                'radius_um': r_um,
                                'exp_dir': exp_name,
                                'particle': pid,
                                'segment': s_id,
                                'tau_frames': tau,
                                'lag_time_s': tau * DT,
                                'msd_real': msd_r[tau_idx],
                                'msd_virt': msd_v[tau_idx],
                                'xi': xi_val,
                                'rc_over_xi': r_um / xi_val if np.isfinite(xi_val) else np.nan,
                            })
                            
                    # 変位サンプリング (PDF 用: tau = 1, 4, 16)
                    for tau_target in [1, 4, 16]:
                        if len(pos_r) > tau_target:
                            dx_r = pos_r[tau_target:, 0] - pos_r[:-tau_target, 0]
                            dy_r = pos_r[tau_target:, 1] - pos_r[:-tau_target, 1]
                            dx_v = pos_v[tau_target:, 0] - pos_v[:-tau_target, 0]
                            dy_v = pos_v[tau_target:, 1] - pos_v[:-tau_target, 1]
                            
                            disp_records['real'][bname][tau_target].extend(dx_r.tolist())
                            disp_records['real'][bname][tau_target].extend(dy_r.tolist())
                            disp_records['virt'][bname][tau_target].extend(dx_v.tolist())
                            disp_records['virt'][bname][tau_target].extend(dy_v.tolist())

    df_inst_full = pd.DataFrame(instant_records)
    df_imsd_full = pd.DataFrame(imsd_records)
    
    # 保存先・形式は contrast_analysis.py 側と共通化 (実行ディレクトリに依存しない)
    from contrast_analysis import save_cached_data
    save_cached_data(df_inst_full, df_imsd_full, disp_records)
    print("\nExtraction complete!")
    print(f"Total instant records: {len(df_inst_full)}")
    print(f"Total iMSD points: {len(df_imsd_full)}")

if __name__ == '__main__':
    run_extraction()
