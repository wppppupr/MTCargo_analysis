#!/usr/bin/env python3
"""
scratch/analyze_contrast_results.py
抽出した対照解析データの統計量計算と検証
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

df_inst = pd.read_pickle("scratch/contrast_inst_cache.pkl")
df_imsd = pd.read_pickle("scratch/contrast_imsd_cache.pkl")
disp_data = np.load("scratch/contrast_disp_cache.npy", allow_pickle=True).item()

beads_order = [
    ('beads06um', 0.63, 0.315),
    ('beads1um',  1.18, 0.590),
    ('beads3um',  3.37, 1.685),
    ('beads7um',  7.24, 3.620),
    ('beads5um',  5.00, 2.500),
    ('beads20um', 20.0, 10.00),
]

print("================================================================================")
print("1. 結合度解析: 実測 <cos psi> vs 従来 <cos theta_i> vs 下限ヌルモデル")
print("================================================================================")

np.random.seed(42)

summary_rows = []

for bname, d_um, r_um in beads_order:
    sub = df_inst[df_inst['bead_name'] == bname].copy()
    rc_xi = sub['rc_over_xi'].mean()
    xi_m = sub['xi'].mean()
    
    ci_vals = sub['cos_i'].values
    cb_vals = sub['cos_bulk'].values
    
    # ヌルモデル 1: 時間シャッフル (粒子ごとに時間シャッフルを100回試行)
    shuf_means = []
    for _ in range(100):
        # 粒子ごとに cargo 速度ベクトルをシャッフル
        shuf_cbs = []
        for (exp, pid), grp in sub.groupby(['exp_dir', 'particle']):
            if len(grp) < 2:
                continue
            idx = np.random.permutation(len(grp))
            # シャッフル後の貨物速度
            cdx_s = grp['cargo_dx'].values[idx]
            cdy_s = grp['cargo_dy'].values[idx]
            cspd_s = grp['cargo_spd'].values[idx]
            # 流速はそのまま
            u = grp['flow_u'].values
            v = grp['flow_v'].values
            v_fn = grp['flow_spd'].values
            
            cb_shuf = (cdx_s * u + cdy_s * v) / (cspd_s * v_fn)
            shuf_cbs.extend(cb_shuf)
        if shuf_cbs:
            shuf_means.append(np.mean(shuf_cbs))
            
    null_time_mean = np.mean(shuf_means)
    null_time_std = np.std(shuf_means)
    
    # ヌルモデル 2: 粒子スワップ (同一実験内の別粒子とペアリング)
    swap_means = []
    exp_grps = sub.groupby('exp_dir')
    for _ in range(100):
        swap_cbs = []
        for exp_name, e_grp in exp_grps:
            pids = e_grp['particle'].unique()
            if len(pids) < 2:
                # 粒子が1つしかない場合は時間シャッフルで代用
                idx = np.random.permutation(len(e_grp))
                cdx_s = e_grp['cargo_dx'].values[idx]
                cdy_s = e_grp['cargo_dy'].values[idx]
                cspd_s = e_grp['cargo_spd'].values[idx]
                cb_sw = (cdx_s * e_grp['flow_u'].values + cdy_s * e_grp['flow_v'].values) / (cspd_s * e_grp['flow_spd'].values)
                swap_cbs.extend(cb_sw)
                continue
            # 粒子ラベルをシャッフル
            pids_perm = np.random.permutation(pids)
            p_map = dict(zip(pids, pids_perm))
            for pid in pids:
                target_pid = p_map[pid]
                grp_a = e_grp[e_grp['particle'] == pid]
                grp_b = e_grp[e_grp['particle'] == target_pid]
                min_len = min(len(grp_a), len(grp_b))
                if min_len < 2:
                    continue
                cdx_b = grp_b['cargo_dx'].values[:min_len]
                cdy_b = grp_b['cargo_dy'].values[:min_len]
                cspd_b = grp_b['cargo_spd'].values[:min_len]
                u_a = grp_a['flow_u'].values[:min_len]
                v_a = grp_a['flow_v'].values[:min_len]
                vfn_a = grp_a['flow_spd'].values[:min_len]
                cb_sw = (cdx_b * u_a + cdy_b * v_a) / (cspd_b * vfn_a)
                swap_cbs.extend(cb_sw)
        if swap_cbs:
            swap_means.append(np.mean(swap_cbs))
    null_swap_mean = np.mean(swap_means) if swap_means else np.nan
    null_swap_std = np.std(swap_means) if swap_means else np.nan
    
    # 有意性 (Z-score: 実測値がシャッフル分布から何シグマ離れているか)
    z_score = (np.mean(cb_vals) - null_time_mean) / (np.std(cb_vals) / np.sqrt(len(cb_vals)))
    
    # 速度伝達係数 kappa (直交回帰 or 最小二乗回帰: v_cargo = kappa * v_flow)
    # v_cargo_vec . v_flow_vec / |v_flow_vec|^2
    kappa_local = np.sum(sub['cargo_dx'] * sub['flow_u'] + sub['cargo_dy'] * sub['flow_v']) / np.sum(sub['flow_spd']**2)
    speed_ratio = np.mean(sub['cargo_spd']) / np.mean(sub['flow_spd'])
    
    summary_rows.append({
        'bead_name': bname,
        'diameter_um': d_um,
        'rc_over_xi': rc_xi,
        'xi_um': xi_m,
        'cos_i_mean': np.mean(ci_vals),
        'cos_psi_real': np.mean(cb_vals),
        'cos_psi_sem': stats.sem(cb_vals),
        'null_time_mean': null_time_mean,
        'null_time_std': null_time_std,
        'null_swap_mean': null_swap_mean,
        'z_score': z_score,
        'kappa': kappa_local,
        'speed_ratio': speed_ratio,
        'n_points': len(sub),
        'n_particles': sub['particle'].nunique(),
    })
    
    print(f"\n[{bname}] d={d_um} um, R_c/xi = {rc_xi:.3f} (xi = {xi_m:.2f} um):")
    print(f"  実測 <cos psi> (正規化結合度) : {np.mean(cb_vals):.4f} +/- {stats.sem(cb_vals):.4f}")
    print(f"  従来 <cos theta_i> (点平均)   : {np.mean(ci_vals):.4f} +/- {stats.sem(ci_vals):.4f}")
    print(f"  下限 時間シャッフル ヌル       : {null_time_mean:.4f} +/- {null_time_std:.4f}")
    print(f"  下限 粒子スワップ ヌル         : {null_swap_mean:.4f} +/- {null_swap_std:.4f}")
    print(f"  上限 仮想貨物 (理論)          : 1.0000")
    print(f"  シグナル有意性 (Z-score)       : Z = {z_score:.2f} (p < 1e-10 if Z > 6)")
    print(f"  伝達係数 kappa               : {kappa_local:.4f}, 速度比: {speed_ratio:.4f}")

df_sum = pd.DataFrame(summary_rows)

print("\n================================================================================")
print("2. 仮想貨物 vs 実測貨物: MSD 比較 (tau = 4s, 16s, 64s)")
print("================================================================================")

msd_summary = []
for bname, d_um, r_um in beads_order:
    sub_m = df_imsd[df_imsd['bead_name'] == bname]
    rc_xi = sub_m['rc_over_xi'].mean()
    
    print(f"\n[{bname}] (d={d_um} um, R_c/xi={rc_xi:.3f}):")
    for t_s in [4.0, 16.0, 64.0]:
        sub_t = sub_m[sub_m['lag_time_s'] == t_s]
        if len(sub_t) == 0:
            continue
        m_real = sub_t['msd_real'].mean()
        m_virt = sub_t['msd_virt'].mean()
        ratio = m_real / m_virt if m_virt > 0 else np.nan
        print(f"  tau = {t_s:4.0f} s: 実測 MSD = {m_real:7.3f} um^2 | 仮想 MSD = {m_virt:7.3f} um^2 | 実測/仮想 = {ratio:6.3f}")

print("\n================================================================================")
print("3. 仮想貨物 vs 実測貨物: 変位 PDF 統計 (tau = 1 frame = 4s)")
print("================================================================================")
for bname, d_um, r_um in beads_order:
    r_disp = np.array(disp_data['real'][bname][1])
    v_disp = np.array(disp_data['virt'][bname][1])
    
    r_std = np.std(r_disp)
    v_std = np.std(v_disp)
    r_kurt = stats.kurtosis(r_disp) # 正規分布で 0
    v_kurt = stats.kurtosis(v_disp)
    
    print(f"[{bname}] (d={d_um} um):")
    print(f"  実測: sigma = {r_std:.3f} um, Excess Kurtosis = {r_kurt:.2f}, N = {len(r_disp)}")
    print(f"  仮想: sigma = {v_std:.3f} um, Excess Kurtosis = {v_kurt:.2f}, N = {len(v_disp)}")
    print(f"  sigma比 (実測/仮想) = {r_std/v_std:.3f}")

