import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import zarr
import os
import sys
import glob

from pathlib import Path

sys.path.append(os.path.abspath(".."))

plt.style.use('libs/my_style.mplstyle')

mypass = Path('/mnt/NAS-Ebanaru/sasaki/MTSingleBeads')
#mypass = Path('/Volumes/data/Sasaki/MTsingleBeads')

input_dir = mypass /'control'

def load(input_dir, folder):
    op_data = []

    # データの読み込み
    for exp_subpath in sorted(glob.glob(str(input_dir/ folder/'*'/'*'))):

        if os.path.isdir(exp_subpath):
            print(exp_subpath)

            target_path = Path(exp_subpath) / "MTs_order_parameter.zarr"
            
            if target_path.exists():
                try:
                    op_zarr = zarr.open(str(target_path), mode='r')
                    op = op_zarr[:]
                    
                    # NaNなどの無効な値を除外して1次元配列にする
                    op = np.array(op).flatten()
                    op = op[~np.isnan(op)]
                    
                    num = len(op)
                    op_mean = np.mean(op) if num > 0 else np.nan
                    print(target_path)

                    if not np.isnan(op_mean):
                        op_data.append(op_mean)
                    
                except Exception as e:
                    print(f"Error loading {target_path}: {e}")

            else:
                raise ValueError(f'{exp_subpath}')
    op_data = np.array(op_data)

    op_mean = np.mean(op_data)
    op_err = np.std(op_data)#/np.sqrt(op_data.shape[0])

    return op_mean, op_err

mean_1uM, err_1uM = load(input_dir, 'MTs1uM')
mean_4uM, err_4uM = load(input_dir, 'MTs4uM')
mean_6uM, err_6uM = load(input_dir, 'MTs6uM')
mean_8uM, err_8uM = load(input_dir, 'MTs8uM')
mean_10uM, err_10uM = load(input_dir, 'MTs10uM')

means = [mean_1uM, mean_4uM, mean_6uM, mean_8uM, mean_10uM]
errs = [err_1uM, err_4uM, err_6uM, err_8uM, err_10uM]

fig, ax = plt.subplots()
ax.errorbar([1, 4, 6, 8, 10], means, errs, marker='o')
ax.set_xlabel('tub concentration [\u03bcM]')
ax.set_ylabel('Nematic order parameter')
ax.set_xlim(0,11)
ax.set_ylim(0,1)

fig.savefig(mypass / "figure" / "control_order.pdf", bbox_inches = 'tight')

fig2, ax = plt.subplots()
ax.errorbar([28, 7, 4.67, 3.5, 2.8], means, errs, marker='o')
ax.set_xlabel('MTs dilution')
ax.set_ylabel('Nematic order parameter')
ax.set_xlim(0,30)
ax.set_ylim(0,1)

fig2.savefig(mypass / "figure" / "control_order2.pdf", bbox_inches = 'tight')