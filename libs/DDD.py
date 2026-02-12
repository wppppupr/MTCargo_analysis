TAU = 200
SCALE = 0.11
INTERVAL = 4

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os
import sys

sys.path.append(os.path.abspath(".."))

p = Path(os.getcwd())
output_png = p / f'figures/DDD_dt{TAU}s.png'
output_pdf = p / f'figures/DDD_dt{TAU}s.pdf'


plt.style.use('libs/my_style.mplstyle')

path0 = Path("/Volumes/My Passport/Sasaki/MTsingleBeads/20260122/exp")
path1 = Path("/Volumes/My Passport/Sasaki/MTsingleBeads/20260122/exp001")
path2 = Path("/Volumes/My Passport/Sasaki/MTsingleBeads/20260122/exp002")
path3 = Path('/Volumes/My Passport/Sasaki/MTsingleBeads/20260121/beads_trans_crop_crop')
path4 = Path('/Volumes/My Passport/Sasaki/MTsingleBeads/20260121/exp_crop1')

paths = [path0, path1, path2, path3, path4]
tau_frame = int(TAU/INTERVAL)


def plot_displacement_radar(paths, tau, bins=36):
    """
    変位の方向分布をレーダーチャート（極座標ヒストグラム）として表示する
    """
    all_angles = []

    for path in paths:
        dfp = Path(path) / "beads_tracks.csv"
        if not dfp.exists():
            continue
            
        df = pd.read_csv(dfp)
        
        # 粒子ごとに計算（dpm.PDFの内部処理を参考に変位を抽出）
        # 各粒子IDでグルーピングして、tauステップ後の差分をとる
        for pid, group in df.groupby('particle'):
            # インデックスではなく実際の時間やフレームに基づいてdiffをとる必要がある
            # ここでは簡易的に、tauステップ離れたデータとの差分を計算
            dx = group['x'].diff(periods=tau).dropna()
            dy = group['y'].diff(periods=tau).dropna()
            
            # 角度（ラジアン）を計算
            angles = np.arctan2(dy, dx)
            all_angles.extend(angles)

    if not all_angles:
        print("No data to plot.")
        return

    # ヒストグラムの計算
    counts, bin_edges = np.histogram(all_angles, bins=bins, range=(-np.pi, np.pi))
    
    # プロット用にデータを整形（円を閉じるために終点を追加）
    angles_plot = np.linspace(-np.pi, np.pi, bins, endpoint=False)
    # 中心を合わせるためにbinの幅の半分ずらす
    width = (2 * np.pi) / bins
    angles_plot += width / 2

    # プロット作成
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
    
    # 棒グラフによるレーダーチャート
    bars = ax.bar(angles_plot, counts, width=width, bottom=0.0, 
                  alpha=0.6)

    # 見栄えの調整
    ax.set_theta_zero_location('E')  # 0度を東（右）に設定
    ax.set_theta_direction(1)        # 反時計回り
    #ax.set_title(f"Displacement Direction Distribution ($\\Delta$t={TAU} s)", va='bottom')
    
    fig.savefig(output_png)
    fig.savefig(output_pdf)

plot_displacement_radar(paths, tau=tau_frame)