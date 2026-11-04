#!/usr/bin/env python3
"""
scratch/test_null_and_virtual.py
対照解析のテストスクリプト:
1. 正規化結合度 <cos psi>
2. 下限ヌルモデル: 時間シャッフル、軌道スワップ
3. 上限モデル: 仮想貨物 (足印平均流速での移流) の MSD, PDF, <v_hat . u_hat>
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# キャッシュ読み込み
df_inst = pd.read_pickle("scratch/df_inst_cache.pkl")
print(f"Loaded df_inst: {len(df_inst)} rows")

beads_order = ['beads06um', 'beads1um', 'beads3um', 'beads7um', 'beads5um', 'beads20um']

# 1. ヌルモデルの計算 (時間シャッフル & 粒子スワップ)
np.random.seed(42)

results = []

for bname in beads_order:
    sub = df_inst[df_inst['bead_name'] == bname].copy()
    d_um = sub['diameter_um'].iloc[0]
    rc_xi = sub['rc_over_xi'].mean()
    xi_m = sub['xi'].mean()
    
    # 実測値
    cos_i_real = sub['cos_i'].values
    cos_psi_real = sub['cos_bulk'].values
    phi_real = sub['Phi_mag'].values
    
    # 時間シャッフルヌルモデル (粒子ごとに時間シャッフル)
    cos_psi_time_shuf = []
    cos_i_time_shuf = []
    
    for (exp, pid), grp in sub.groupby(['exp_dir', 'particle']):
        if len(grp) < 5:
            continue
        cb = grp['cos_bulk'].values
        ci = grp['cos_i'].values
        # 貨物速度の方向をシャッフルしたのと同等に、cos_bulkをシャッフル
        # より厳密には、貨物の単位速度ベクトルと流速単位ベクトルの内積を、時刻をランダムに入れ替えて計算
        # 各フレームで dot_p = cdx*u + cdy*v, v_c = cspd, v_f = vn
        # cdx, cdy と u, v をランダムシャッフル
        # ここでは、grp['cos_bulk'] や内積を直接シャッフルして再計算
        n_pts = len(grp)
        # 速度ベクトルを取り出す（verify_cos_decompositionの定義に基づく）
        # 速度ベクトル復元
        # u, v, cdx, cdy をシャッフル
        # 簡単なシャッフル: grp内の行順序をシャッフルして別時刻の流れと内積
        # 10回試行
        for _ in range(10):
            shuf_idx = np.random.permutation(n_pts)
            # 貨物速度はgrpのまま、流れベクトルをshuf_idxで入れ替え
            # cos_psi_shuf = cos(theta_c - theta_f_shuf)
            # theta_c と theta_f を直接計算可能:
            # cos_bulk = cos(theta_c - theta_f)
            # シャッフル時の内積:
            # 実測の flow_u, flow_v と cargo_dx, cargo_dy
            pass
            
    # 実測粒子の速度ベクトルと足印流速の復元
    # sub には dot_product, cos_bulk, v_flow_norm, v_cargo_norm がある
    # cos_bulk = (v_c . u_f) / (|v_c| |u_f|)
    # シャッフルヌルモデル:
    # 粒子ごとに速度ベクトルの角度 theta_c と 流速の角度 theta_f を独立にシャッフル
    # cos_bulk_shuf = cos(theta_c(t) - theta_f(t_rand))
    
    # 100回の時間シャッフル
    n_samples = len(sub)
    shuf_cos_psi_means = []
    shuf_cos_i_means = []
    for _ in range(50):
        # 粒子ごと、または実験ごとにシャッフル
        shuf_cos_psi_parts = []
        for (exp, pid), grp in sub.groupby(['exp_dir', 'particle']):
            if len(grp) < 2:
                continue
            # grp内のインデックスをランダム置換
            idx = np.random.permutation(len(grp))
            # ランダムに組み合わせた場合の cos_bulk の期待値
            # 等方的乱数ではなく、実験内の貨物方向分布と流れ方向分布の積
            # ランダムペアの角度差の余弦
            # ここで、粒子ごとの方向ベクトルの内積:
            # v_c と u_f の角度をそれぞれ求めておくのが一番綺麗
            pass

print("Test script initialized.")

