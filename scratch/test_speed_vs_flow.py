#!/usr/bin/env python3
"""
scratch/test_speed_vs_flow.py
|v_c| vs |<u_fp>| の関係を全ビーズサイズで検証する。
"""

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

df_inst = pd.read_pickle("scratch/contrast_inst_cache.pkl")
print(f"Loaded {len(df_inst)} records.")

SCALE = 0.11
DT = 4.0
V_UNIT = SCALE / DT  # um/s

beads_order = ['beads06um', 'beads1um', 'beads3um', 'beads7um', 'beads5um', 'beads20um']

print("================================================================================")
print("|v_c| vs |<u_fp>| の回帰・相関統計 (単位: um/s)")
print("================================================================================")

for bname in beads_order:
    sub = df_inst[df_inst['bead_name'] == bname].copy()
    d_um = sub['diameter_um'].iloc[0]
    rc_xi = sub['rc_over_xi'].mean()
    
    vc = sub['cargo_spd'].values * V_UNIT
    uf = sub['flow_spd'].values * V_UNIT
    
    # ピアソン相関係数
    r_val, p_val = stats.pearsonr(uf, vc)
    # スピアマン順位相関係数
    rho_val, p_rho = stats.spearmanr(uf, vc)
    
    # 線形回帰 vc = slope * uf + intercept
    res = stats.linregress(uf, vc)
    slope = res.slope
    intercept = res.intercept
    r_squared = res.rvalue**2
    
    # 平均比率
    ratio = np.mean(vc) / np.mean(uf)
    
    print(f"\n[{bname}] d={d_um} um, R_c/xi={rc_xi:.3f} (N={len(sub)} points, {sub['particle'].nunique()} particles):")
    print(f"  平均流速 <|u_fp|> = {np.mean(uf):.4f} um/s, 平均貨物速さ <|v_c|> = {np.mean(vc):.4f} um/s (比率={ratio:.3f})")
    print(f"  線形回帰: |v_c| = {slope:.4f} * |<u_fp>| + {intercept:.4f} um/s")
    print(f"  相関係数: Pearson r = {r_val:.4f} (R^2 = {r_squared:.4f}, p = {p_val:.2e}), Spearman rho = {rho_val:.4f}")

