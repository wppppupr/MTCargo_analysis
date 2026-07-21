"""
VACF (Velocity Autocorrelation Function) の解析およびプロットを行うスクリプトです。
notebooks/vacf.ipynb の内容を整理し、再利用しやすい形にまとめました。
"""

import os
import sys
import glob
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# libsディレクトリから親ディレクトリ(MTCargo_analysis)をパスに追加
current_dir = Path(__file__).parent.resolve()
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

# 同じ libs 内の vacf モジュールをインポート
from libs import vacf


def vacf_plot(track_path, max_timeshift_frames=50, frame_interval=4, scale=0.11, display=False, ax=None):
    """
    1つの実験データ(beads_tracks.csv)からVACFを計算する。
    """
    track_df = pd.read_csv(track_path)

    if display and ax is None:
        fig, ax = plt.subplots()
        ax.set(ylabel=r'$\langle \vec{v}(t) \cdot \vec{v}(t+\tau) \rangle / \langle |\vec{v}|^2 \rangle$',
               xlabel=r'lag time $\Delta t$ [s]')

    # 個別のVACFを計算
    IVACF = vacf.ivacf(track_df, max_timeshift_frames=max_timeshift_frames, 
                       frame_interval=frame_interval, scale=scale, display=False, ax=ax)

    # アンサンブル平均を計算
    EVACF, EVACF_err = vacf.evacf(IVACF, display=False, ax=ax)    

    return IVACF, EVACF, EVACF_err


def vacf_dir(directory, max_timeshift_frames=50, frame_interval=4, scale=0.11):
    """
    指定ディレクトリ以下の全実験データについてVACFを計算し、まとめたDataFrameを返す。
    """
    evacf_list = []
    directory = Path(directory)
    
    # 2階層下のディレクトリを探索 (例: directory / "date" / "exp_name")
    exp_dirs = [Path(p) for p in glob.glob(str(directory / "*" / "*"))]
    exp_dirs = [d for d in exp_dirs if d.is_dir()]
    
    for i, input_dir in enumerate(exp_dirs):
        track_path = input_dir / "beads_tracks.csv"
        if not track_path.exists():
            continue
            
        ivacf, evacf, evacf_err = vacf_plot(
            track_path, 
            max_timeshift_frames=max_timeshift_frames, 
            frame_interval=frame_interval, 
            scale=scale, 
            display=False
        )
        
        df = pd.DataFrame({
            "exp": i,
            "VACF": evacf
        })
        evacf_list.append(df)

    if not evacf_list:
        return pd.DataFrame()

    evacf_df = pd.concat(evacf_list)
    return evacf_df


def plot_all_beads_vacf(root_dir, out_dir):
    """
    各サイズのビーズのVACFを計算してプロット・保存する。
    """
    root = Path(root_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    # 各ビーズサイズのディレクトリ名とプロット用設定
    # (ディレクトリ名, マーカー, ラベル)
    beads_info = [
        ("beads06um", '^', '0.63 μm'),
        ("beads1um", 'o', '1.18 μm'),
        ("beads3um", 'd', '3.37 μm'),
        ("beads5um", 10,  '5 μm'),
        ("beads7um", 11,  '7.24 μm'),
        ("beads20um", 's', '20 μm'),
    ]

    fig, ax = plt.subplots()

    for dname, marker, label in beads_info:
        target_dir = root / dname
        if not target_dir.exists():
            print(f"ディレクトリが見つかりません: {target_dir}")
            continue
            
        print(f"{dname} を処理中...")
        evacf_df = vacf_dir(target_dir)
        if evacf_df.empty:
            print(f"データがありません: {dname}")
            continue
            
        evacf_mean = evacf_df.groupby('lag time').mean()
        evacf_std = evacf_df.groupby('lag time').std()

        ax.errorbar(
            evacf_mean.index, 
            evacf_mean['VACF'], 
            yerr=evacf_std['VACF'],  
            marker=marker, 
            label=label
        )

    ax.legend()

    ax.hlines(0, 0, 30, colors='#333333', linestyles='dashed', alpha = 0.6, zorder = -1)

    ax.set(
        xlim=(0, 30),
        ylim=(-1, 1),
        xlabel='lag time $\Delta t$ [s]',
        ylabel='ACF $\langle \\Delta\\hat{r}_i(t)\\cdot \\Delta\\hat{r}_i(t+\\Delta t) \\rangle$'
    )

    save_path = out / "VACF.svg"
    fig.savefig(save_path, bbox_inches='tight')
    print(f"プロットを保存しました: {save_path}")


def main():
    # スタイルの適用
    style_path = current_dir / 'my_style.mplstyle'
    if style_path.exists():
        plt.style.use(str(style_path))
    else:
        print(f"警告: スタイルファイルが見つかりません: {style_path}")

    # NASのルートディレクトリ
    # 必要に応じてパスを変更してください
    root_dir = '/Volumes/data/Sasaki/MTsingleBeads'
    #root_dir = '/mnt/NAS-Ebanaru/sasaki/MTsingleBeads'
    
    out_dir = Path(root_dir) / 'figure'
    
    plot_all_beads_vacf(root_dir, out_dir)


if __name__ == "__main__":
    main()
