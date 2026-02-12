import pandas as pd
import numpy as np
from pathlib import Path
import os

SCALE, INTERVAL = 0.11, 4

mypass = '/Volumes/My Passport/Sasaki/MTsingleBeads'

path0 = os.path.join(mypass,"20260121/beads_trans_crop_crop/beads_tracks.csv")
path1 = os.path.join(mypass,"20260121/exp_crop1/beads_tracks.csv")
path2 = os.path.join(mypass,"20260122/exp/beads_tracks.csv")
path3 = os.path.join(mypass,"20260122/exp001/beads_tracks.csv")
path4 = os.path.join(mypass,"20260122/exp002/beads_tracks.csv")

paths = [path0, path1, path2, path3, path4]

max_lag=1000

def calcMRL(df, max_lag, interval=1):
    # (中身は提供いただいたコードと同じですが、効率化のために少しだけ整理しています)
    df = df.sort_values(['particle', 'frame'])
    results = []
    
    for lag in range(1, max_lag + 1):
        g = df.groupby('particle')
        dx = g['x'].diff(periods=lag)
        dy = g['y'].diff(periods=lag)
        dr = np.sqrt(dx**2 + dy**2)
        
        mask = dr > 0
        if not mask.any():
            continue
            
        temp_df = pd.DataFrame({
            'particle': df.loc[mask, 'particle'],
            'du': dx[mask] / dr[mask],
            'dv': dy[mask] / dr[mask]
        })

        particle_stats = temp_df.groupby('particle').mean()
        mrl_per_particle = np.sqrt(particle_stats['du']**2 + particle_stats['dv']**2)
        
        results.append({
            'lag': lag * interval,
            'mrl': mrl_per_particle.mean(),
            'std': mrl_per_particle.std(),
            'n_particles': len(mrl_per_particle) # 統計用に粒子数も保持
        })
    
    return pd.DataFrame(results)

# --- 実験ごとの平均を出すメイン処理 ---

all_exp_results = []

for path in paths:
    # ファイル読み込み（pathがPathオブジェクトでも文字列でも対応）
    df = pd.read_csv(path)
    
    # 個別の実験のMRLを計算
    exp_mrl = calcMRL(df, max_lag, INTERVAL)
    
    # どの実験データか識別できるようにパスを保存（オプション）
    exp_mrl['source_file'] = Path(path).name
    
    all_exp_results.append(exp_mrl)

# 1. 全実験のデータを一つの大きなDataFrameに結合
combined_df = pd.concat(all_exp_results, ignore_index=True)

# 2. 'lag' ごとに全実験の平均と標準偏差を算出
# ここでの「mean」が実験間平均（Grand Average）になります
final_summary = combined_df.groupby('lag').agg({
    'mrl': ['mean', 'std', 'count'], # countは実験数
    'std': 'mean'                   # 実験内のばらつきの平均
}).reset_index()

# カラム名を見やすく整理
final_summary.columns = ['lag', 'mrl_mean', 'mrl_exp_std', 'n_experiments', 'avg_intra_std']

import matplotlib.pyplot as plt

plt.style.use('libs/my_style.mplstyle')

fig, ax = plt.subplots()

ax.plot(final_summary['lag'], final_summary['mrl_mean'])
ax.fill_between(final_summary['lag'], final_summary['mrl_mean']+final_summary['mrl_exp_std']/np.sqrt(final_summary['n_experiments']), final_summary['mrl_mean']-final_summary['mrl_exp_std']/np.sqrt(final_summary['n_experiments']), alpha = 0.5)

ax.set(
    xlim=(0, 200),
    ylim=(0, 1),
    xscale='linear',
    yscale='linear',
    xlabel='lag time $\\Delta t$ [s]',
    ylabel='MRL $R_1(\\Delta t)$'
    )

fig.savefig("/Users/sasakinozomu/code/MTCargo_analysis/notebooks/figures/MRL.png")
fig.savefig("/Users/sasakinozomu/code/MTCargo_analysis/notebooks/figures/MRL.pdf")