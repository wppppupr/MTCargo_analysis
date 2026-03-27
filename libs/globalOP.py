import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import zarr
import os
import sys
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Calculate and plot global Order Parameter for different experiments.")
    parser.add_argument('-e', '--experiments', type=str, nargs='+', required=True,
                        help="List of experiment paths.")
    parser.add_argument('-o', '--output', type=Path, default=Path("globalOP_boxplot.png"), 
                        help="Output path for the boxplot.")
    
    args = parser.parse_args()
    
    sys.path.append(os.path.abspath(".."))
    try:
        plt.style.use('libs/my_style.mplstyle')
    except OSError:
        pass  # ignore if style is not found

    op_data = []
    labels = []

    i=0
    
    # データの読み込み
    for exp_subpath in args.experiments:
        target_path = Path(exp_subpath) / "MTs_order_parameter.zarr"
        
        if target_path.exists():
            try:
                op_zarr = zarr.open(str(target_path), mode='r')
                op = op_zarr[:]
                
                # NaNなどの無効な値を除外して1次元配列にする
                op = np.array(op).flatten()
                op = op[~np.isnan(op)]
                op_data.append(op)
                
                # ラベルに実験のサブパスを使用
                labels.append(f'exp{i}')
                
                num = len(op)
                op_mean = np.mean(op) if num > 0 else np.nan
                op_sem = np.std(op, ddof=1)/np.sqrt(num) if num > 1 else 0
                print(f"{exp_subpath}: Mean = {op_mean:.4f}, SEM = {op_sem:.4f}, N = {num}")
                i+=1
                
            except Exception as e:
                print(f"Error loading {target_path}: {e}")
        else:
            print(f"Warning: {target_path} does not exist.")

    # 箱ひげ図の作成
    if op_data:
        fig, ax = plt.subplots()
        
        # ボックスプロットを描画
        bp = ax.boxplot(op_data, labels=labels, patch_artist=True)
        
        # 見やすくするために色を付ける（オプション）
        for box in bp['boxes']:
            box.set(facecolor='#88CCEE', alpha=0.7)
            
        ax.set_ylabel("Nematic Order Parameter $S$")
        plt.xticks(rotation=45, ha='right')
        
        # 画像の保存
        plt.savefig(args.output, dpi=300)
        print(f"Saved boxplot to {args.output}")
    else:
        print("No valid data available to plot.")

if __name__ == "__main__":
    main()
