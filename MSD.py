import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
import os
import sys
import glob
from pathlib import Path

#sys.path.append(os.path.abspath(".."))

from libs import fit_model as fm
from libs import displacement as dpm

plt.style.use('libs/my_style.mplstyle')
style_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

mypass = Path('/mnt/NAS-Ebanaru/sasaki/MTSingleBeads')

def concatMSD(folder, interval_list, scale = 0.11):
    MSD_list = []
    num = 0
    for i in glob.glob(str(Path(folder) / "*" / "*" /"beads_tracks.csv")):
        MSD_df = {"exp":[], "MSD":[]} # Initialize an empty DataFrame to store the results
        IMSD, _ = dpm.get_msd(i, scale, interval_list[num], threshold = 0.0)
        emsd = IMSD.groupby('lag time').mean()['MSD']
        
        MSD_df["MSD"] = emsd
        MSD_df["exp"] = num

        MSD_list.append(pd.DataFrame(MSD_df))

        num +=1
    
    return pd.concat(MSD_list)

I_concat_1um = concatMSD(mypass / "beads1um", [4, 4, 4, 4])
I_concat_3um = concatMSD(mypass / "beads3um", [4, 4, 4])
I_concat_7um = concatMSD(mypass / "beads7um", [4, 4, 4])
I_concat_8um = concatMSD(mypass / "beads8um", [4])
I_concat_20um = concatMSD(mypass / "beads20um", [4, 4, 4, 4])

scale = 0.11

alpha = 1
marker_size = 10

min_t = 20
max_t = 144

minimum=min_t/4
maximum=max_t/4

emsd_combined_1um = I_concat_1um.groupby('lag time').mean()['MSD']
emsd_err_combined_1um = I_concat_1um.groupby('lag time').std()['MSD'] / np.sqrt(1+max(I_concat_1um["exp"]))
emsd_combined_3um = I_concat_3um.groupby('lag time').mean()['MSD']
emsd_err_combined_3um = I_concat_3um.groupby('lag time').std()['MSD'] / np.sqrt(1+max(I_concat_3um["exp"]))
emsd_combined_7um = I_concat_7um.groupby('lag time').mean()['MSD']
emsd_err_combined_7um = I_concat_7um.groupby('lag time').std()['MSD'] / np.sqrt(1+max(I_concat_7um["exp"]))
emsd_combined_8um = I_concat_8um.groupby('lag time').mean()['MSD']
emsd_err_combined_8um = I_concat_8um.groupby('lag time').std()['MSD'] / np.sqrt(1+max(I_concat_8um["exp"]))
emsd_combined_20um = I_concat_20um.groupby('lag time').mean()['MSD']
emsd_err_combined_20um = I_concat_20um.groupby('lag time').std()['MSD'] / np.sqrt(1+max(I_concat_20um["exp"]))

mask = (emsd_combined_1um.index > minimum) & (emsd_combined_1um.index < maximum)

# take log10 of selected x and y values and convert to numpy arrays
x = np.log10(emsd_combined_1um.index[mask].to_numpy())
y = np.log10(np.float32(emsd_combined_1um[mask].to_numpy()))
 # remove non-finite entries (e.g., log of non-positive values)
finite = np.isfinite(x) & np.isfinite(y)
x = x[finite]
y = y[finite]
# fit
popt, pcov = curve_fit(fm.ln_pl, x, y)

fig, ax = plt.subplots()

ax.errorbar(emsd_combined_1um.index, emsd_combined_1um, yerr=emsd_err_combined_1um, marker='o', label=f'1.18 \u03bcm, N={1+max(I_concat_1um["exp"])}', alpha = alpha, markersize = marker_size)
ax.errorbar(emsd_combined_3um.index, emsd_combined_3um, yerr=emsd_err_combined_3um, marker='d', label=f'3.37 \u03bcm, N={1+max(I_concat_3um["exp"])}', alpha = alpha, markersize = marker_size)
ax.errorbar(emsd_combined_7um.index, emsd_combined_7um, yerr=emsd_err_combined_7um, marker='^', label=f'7.24 \u03bcm, N={1+max(I_concat_7um["exp"])}', alpha = alpha, markersize = marker_size)  
ax.errorbar(emsd_combined_8um.index, emsd_combined_8um, yerr=emsd_err_combined_8um, marker='v', label=f'8.66 \u03bcm, N={1+max(I_concat_8um["exp"])}', alpha = alpha, markersize = marker_size)
ax.errorbar(emsd_combined_20um.index, emsd_combined_20um, yerr=emsd_err_combined_20um, marker='s', label=f'20.0 \u03bcm, N={1+max(I_concat_20um["exp"])}', alpha = alpha, markersize = marker_size)

ax.legend()

xrange = [min_t, max_t]

ax.plot(xrange, fm.power_law(xrange, *popt) * 5, label = f'{popt[1]:.2f}$\\times\\Delta t^{{{popt[0]:.2f}}}$', color='#333333')
ax.text(25, 3e1, f'$\propto \Delta t^{{{popt[0]:.1f}}}$')

ax.plot([80,200], fm.power_law([80,200], 1 ,popt[1]*1e-1) * 8, label = f'{popt[1]:.2f}$\\times\\Delta t^{{{popt[0]:.2f}}}$', color='#333333')
ax.text(70, 2, f'$\propto \Delta t^{{{1.0}}}$')

ax.set(
    xlim=(3.5e-0, 150),
    ylim=(1e-1, 1e2),
    xscale='log',
    yscale='log',
    xlabel='lag time $\Delta t$ [s]',
    ylabel='MSD $\\langle \\Delta r^2 \\rangle$ [\u03bcm$^2$]'
    )


fig.savefig(mypass / "figure" / "MSD.png", bbox_inches = 'tight')
fig.savefig(mypass / "figure" / "MSD.pdf", bbox_inches = 'tight')