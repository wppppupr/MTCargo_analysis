#!/usr/bin/env python3
"""
scratch/compute_three_models_msd.py
3つの運動モード:
1. Real Cargo (実測貨物)
2. Eulerian Footprint Flow (累積通過流 / 滑り量モデル)
3. Lagrangian Virtual Cargo (本来の仮想貨物: 自律移流トレーサー)
の iMSD および変位統計を統合計算し、キャッシュする。
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy import stats
import verify_cos_decomposition as vcd

ROOT_DIR = vcd.ROOT_DIR
WORKSPACE_DIR = vcd.CURRENT_DIR
BEADS_INFO = vcd.BEADS_INFO
SCALE = vcd.SCALE_UM_PER_PX  # 0.11 um/px
DT = 4.0                     # s/frame

def calc_msd(pos: np.ndarray, max_tau: int = 150):
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

def main():
    print("Loading data...")
    # 既存の実測 & 累積通過流 iMSD キャッシュ
    df_imsd_pass = pd.read_pickle("scratch/contrast_imsd_cache.pkl")
    # 今回計算完了したラグランジュ仮想貨物軌道
    df_lag = pd.read_pickle("scratch/lagrangian_virt_tracks.pkl")
    # 相関長辞書
    xi_dict = vcd.load_experiment_correlation_lengths(ROOT_DIR, WORKSPACE_DIR)

    print(f"Lagrangian records: {len(df_lag)}")

    # ラグランジュ仮想貨物の iMSD を計算
    lag_imsd_records = []
    for (bname, exp_name, pid), grp in df_lag.groupby(['bead_name', 'exp_dir', 'particle']):
        grp = grp.sort_values('frame')
        if len(grp) < 4:
            continue
        d_um = grp['diameter_um'].iloc[0]
        r_um = grp['radius_um'].iloc[0]
        xi_val = xi_dict.get(exp_name, np.nan)
        
        pos_lag = np.column_stack([grp['x_virt_px'].values * SCALE, grp['y_virt_px'].values * SCALE])
        taus, msd_lag = calc_msd(pos_lag, max_tau=100)
        
        for tau_idx, tau in enumerate(taus):
            lag_imsd_records.append({
                'bead_name': bname,
                'diameter_um': d_um,
                'radius_um': r_um,
                'exp_dir': exp_name,
                'particle': pid,
                'tau_frames': tau,
                'lag_time_s': tau * DT,
                'msd_lagrangian_virt': msd_lag[tau_idx],
                'xi': xi_val,
                'rc_over_xi': r_um / xi_val if np.isfinite(xi_val) else np.nan,
            })

    df_imsd_lag = pd.DataFrame(lag_imsd_records)
    print(f"Lagrangian iMSD points: {len(df_imsd_lag)}")

    # 3者の統合比較テーブル作成
    # df_imsd_pass の msd_real, msd_virt (これは累積通過流 msd_pass) とマージ
    df_imsd_pass = df_imsd_pass.rename(columns={'msd_virt': 'msd_eulerian_pass'})
    
    # 粒子・ラグタイムごとにマージ
    merged_imsd = pd.merge(
        df_imsd_pass[['bead_name', 'diameter_um', 'radius_um', 'exp_dir', 'particle', 'segment', 'tau_frames', 'lag_time_s', 'msd_real', 'msd_eulerian_pass', 'rc_over_xi']],
        df_imsd_lag[['bead_name', 'exp_dir', 'particle', 'tau_frames', 'msd_lagrangian_virt']],
        on=['bead_name', 'exp_dir', 'particle', 'tau_frames'],
        how='inner'
    )
    print(f"Merged iMSD points: {len(merged_imsd)}")

    out_p = Path("scratch/three_models_imsd_cache.pkl")
    merged_imsd.to_pickle(out_p)
    print(f"Saved merged iMSD to {out_p}")

    # 集計と表示 (tau = 4s, 16s, 64s)
    beads_order = ['beads06um', 'beads1um', 'beads3um', 'beads7um', 'beads5um', 'beads20um']
    print("\n================================================================================")
    print("3者 MSD 比較: 実測 (Real) vs 累積通過流 (Eulerian Pass) vs 本来の仮想貨物 (Lagrangian Virt)")
    print("================================================================================")
    for bname in beads_order:
        sub = merged_imsd[merged_imsd['bead_name'] == bname]
        d_um = sub['diameter_um'].iloc[0]
        rc_xi = sub['rc_over_xi'].mean()
        print(f"\n[{bname}] d={d_um} um, R_c/xi = {rc_xi:.3f} (N={sub['particle'].nunique()} particles):")
        for t_s in [4.0, 16.0, 64.0]:
            sub_t = sub[sub['lag_time_s'] == t_s]
            if len(sub_t) == 0:
                continue
            m_real = sub_t['msd_real'].mean()
            m_pass = sub_t['msd_eulerian_pass'].mean()
            m_lag = sub_t['msd_lagrangian_virt'].mean()
            
            ratio_real_to_lag = m_real / m_lag if m_lag > 0 else np.nan
            ratio_real_to_pass = m_real / m_pass if m_pass > 0 else np.nan
            slip_ratio = 1.0 - ratio_real_to_pass
            
            print(f"  tau = {t_s:4.0f} s:")
            print(f"     実測貨物 (Real)              : {m_real:7.3f} um^2")
            print(f"     本来の仮想貨物 (Lagrangian Virt): {m_lag:7.3f} um^2  --> 実測/仮想 (追従度) = {ratio_real_to_lag:6.3f}")
            print(f"     累積通過流 (Eulerian Pass)    : {m_pass:7.3f} um^2  --> 滑り率 (Slip)     = {slip_ratio*100:5.1f}%")

if __name__ == '__main__':
    main()

