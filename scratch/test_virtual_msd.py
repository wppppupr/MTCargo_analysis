#!/usr/bin/env python3
"""
scratch/test_virtual_msd.py
仮想貨物の軌道構築とMSD・PDF計算のテスト
"""

import numpy as np
import pandas as pd
from pathlib import Path

# キャッシュの読み込み
df_inst = pd.read_pickle("scratch/df_inst_cache.pkl")

# 例として beads1um の粒子を取り出す
sub1 = df_inst[df_inst['bead_name'] == 'beads1um'].copy()
print(f"beads1um points: {len(sub1)}, particles: {sub1['particle'].nunique()}, exps: {sub1['exp_dir'].nunique()}")

# 1つの粒子を選択
first_exp = sub1['exp_dir'].iloc[0]
pids = sub1[sub1['exp_dir'] == first_exp]['particle'].unique()
print(f"Exp: {first_exp}, particles: {pids}")

p_data = sub1[(sub1['exp_dir'] == first_exp) & (sub1['particle'] == pids[0])].sort_values('frame')
print(f"Frames: {len(p_data)}, min_f: {p_data['frame'].min()}, max_f: {p_data['frame'].max()}")
print(p_data[['frame', 'v_flow_norm', 'v_cargo_norm', 'cos_i', 'cos_bulk']].head())

