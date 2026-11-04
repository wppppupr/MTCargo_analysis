#!/usr/bin/env python3
"""
scratch/test_binned_speed.py
ビン平均による |v_c| vs |<u_fp>| の応答プロファイル計算
"""

import numpy as np
import pandas as pd
from pathlib import Path

df_inst = pd.read_pickle("scratch/contrast_inst_cache.pkl")
SCALE = 0.11
DT = 4.0
V_UNIT = SCALE / DT

beads_order = ['beads06um', 'beads1um', 'beads3um', 'beads7um', 'beads5um', 'beads20um']

bins = np.linspace(0.0, 0.35, 15)  # 14 bins from 0 to 0.35 um/s
bin_centers = 0.5 * (bins[:-1] + bins[1:])

for bname in beads_order:
    sub = df_inst[df_inst['bead_name'] == bname].copy()
    vc = sub['cargo_spd'].values * V_UNIT
    uf = sub['flow_spd'].values * V_UNIT
    
    bin_idx = np.digitize(uf, bins) - 1
    
    means = []
    sems = []
    counts = []
    
    for b_i in range(len(bin_centers)):
        mask = (bin_idx == b_i)
        cnt = np.sum(mask)
        counts.append(cnt)
        if cnt >= 10:
            means.append(np.mean(vc[mask]))
            sems.append(np.std(vc[mask]) / np.sqrt(cnt))
        else:
            means.append(np.nan)
            sems.append(np.nan)
            
    print(f"\n[{bname}] Binned Speed Profile:")
    for bc, m, s, c in zip(bin_centers[:6], means[:6], sems[:6], counts[:6]):
        print(f"  |u_fp| = {bc:.3f} um/s (N={c:4d}): <|v_c|> = {m:.4f} +/- {s:.4f} um/s")

