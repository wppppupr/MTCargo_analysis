#!/usr/bin/env python3
"""
scratch/test_particle_bootstrap_fit.py
個別粒子ごとに線形回帰を行ってから、ブートストラップ法でアンサンブル平均・SEM・95% CIを算出する。
"""

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

df_inst = pd.read_pickle("scratch/contrast_inst_cache.pkl")
SCALE = 0.11
DT = 4.0
V_UNIT = SCALE / DT

beads_order = ['beads06um', 'beads1um', 'beads3um', 'beads7um', 'beads5um', 'beads20um']

np.random.seed(42)

def bootstrap_stats(data, n_boot=2000):
    arr = np.asarray(data, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n == 0:
        return np.nan, np.nan, np.nan, np.nan
    if n == 1:
        return arr[0], 0.0, arr[0], arr[0]
    
    boot_means = [np.mean(np.random.choice(arr, size=n, replace=True)) for _ in range(n_boot)]
    mean_val = np.mean(arr)
    sem_val = np.std(boot_means)
    ci_low = np.percentile(boot_means, 2.5)
    ci_high = np.percentile(boot_means, 97.5)
    return mean_val, sem_val, ci_low, ci_high

print("================================================================================")
print("粒子個別フィット -> ブートストラップ平均 (Bootstrap N=2000)")
print("================================================================================")

summary_results = []

for bname in beads_order:
    sub = df_inst[df_inst['bead_name'] == bname].copy()
    d_um = sub['diameter_um'].iloc[0]
    rc_xi = sub['rc_over_xi'].mean()
    
    particle_fits = []
    
    for (exp, pid), grp in sub.groupby(['exp_dir', 'particle']):
        if len(grp) < 10:
            continue
        vc = grp['cargo_spd'].values * V_UNIT
        uf = grp['flow_spd'].values * V_UNIT
        
        # 粒子ごとの線形回帰
        reg = stats.linregress(uf, vc)
        r_val, _ = stats.pearsonr(uf, vc)
        
        # 速度比
        ratio = np.mean(vc) / np.mean(uf) if np.mean(uf) > 0 else np.nan
        
        particle_fits.append({
            'exp_dir': exp,
            'particle': pid,
            'n_frames': len(grp),
            'slope': reg.slope,
            'intercept': reg.intercept,
            'r_value': r_val,
            'r_squared': reg.rvalue**2,
            'speed_ratio': ratio,
            'mean_vc': np.mean(vc),
            'mean_uf': np.mean(uf),
        })
        
    df_p = pd.DataFrame(particle_fits)
    n_p = len(df_p)
    
    # 各指標のブートストラップ統計
    slope_m, slope_sem, slope_l, slope_h = bootstrap_stats(df_p['slope'])
    int_m, int_sem, int_l, int_h = bootstrap_stats(df_p['intercept'])
    r_m, r_sem, r_l, r_h = bootstrap_stats(df_p['r_value'])
    ratio_m, ratio_sem, ratio_l, ratio_h = bootstrap_stats(df_p['speed_ratio'])
    
    summary_results.append({
        'bead_name': bname,
        'diameter_um': d_um,
        'rc_over_xi': rc_xi,
        'n_particles': n_p,
        'slope_mean': slope_m,
        'slope_sem': slope_sem,
        'slope_ci': (slope_l, slope_h),
        'intercept_mean': int_m,
        'intercept_sem': int_sem,
        'intercept_ci': (int_l, int_h),
        'r_mean': r_m,
        'r_sem': r_sem,
        'r_ci': (r_l, r_h),
        'ratio_mean': ratio_m,
        'ratio_sem': ratio_sem,
        'ratio_ci': (ratio_l, ratio_h),
        'particle_df': df_p,
    })
    
    print(f"\n[{bname}] d={d_um} um, R_c/xi = {rc_xi:.3f} (N={n_p} particles):")
    print(f"  傾き alpha (応答感度) : {slope_m:.4f} +/- {slope_sem:.4f}  [95% CI: {slope_l:.4f} ~ {slope_h:.4f}]")
    print(f"  切片 v0              : {int_m:.4f} +/- {int_sem:.4f} um/s  [95% CI: {int_l:.4f} ~ {int_h:.4f}]")
    print(f"  相関係数 Pearson r    : {r_m:.4f} +/- {r_sem:.4f}  [95% CI: {r_l:.4f} ~ {r_h:.4f}]")
    print(f"  速度比 <vc>/<uf>     : {ratio_m:.4f} +/- {ratio_sem:.4f}  [95% CI: {ratio_l:.4f} ~ {ratio_h:.4f}]")
    print("  個別粒子の傾き list:", [round(x, 3) for x in df_p['slope'].values])

