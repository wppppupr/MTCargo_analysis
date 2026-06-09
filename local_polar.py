import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import zarr
from scipy.optimize import curve_fit
import os
import sys
import glob
from pathlib import Path

from libs import fit_model as fm
from libs import cal_vel as cv
from libs import displacement as dpm

plt.style.use('libs/my_style.mplstyle')
style_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

root = Path('/Volumes/data/Sasaki/MTsingleBeads')

beads1um = "beads1um"
beads3um = "beads3um"
beads7um = "beads7um"

def func(root, folder):

    particles = []
    bgs = []

    for input_dir in glob.glob(str(root / folder / "*"/ "*")):
        input_dir = Path(input_dir)
        if os.path.isfile(input_dir):
            continue
        #print(input_dir)
        particle= xr.open_zarr(input_dir / "local_polar_w.zarr")
        bg = xr.open_zarr(input_dir / "local_polar_bg.zarr")
        p_mean = particle.polar_order.mean(dim=['frame', 'particle'])
        bg_mean = bg.polar_order.mean(dim=['frame'])

        particles.append(p_mean)
        bgs.append(bg_mean)

    return particles, bgs

def make_df(particles, bgs):
    p_list = []
    bg_list = []

    i = 0
    for particle in particles:
        p_df = {'window_size':[], 'polar_order':[], 'exp':[]}
        p_mean = np.array(particle)
        p_df['window_size'] = particles[0]['window size']
        p_df['polar_order'] = p_mean
        p_df['exp'] = i
        i += 1
        p_df = pd.DataFrame(p_df)
        p_list.append(p_df)
    i = 0
    for bg in bgs:
        bg_df = {'window_size':[], 'polar_order':[], 'exp':[]}
        bg_mean = np.array(bg)
        bg_df['window_size'] = bgs[0]['window size']
        bg_df['polar_order'] = bg_mean
        bg_df['exp'] = i
        i += 1
        bg_df = pd.DataFrame(bg_df)
        bg_list.append(bg_df)

    p_concat = pd.concat(p_list, ignore_index=True)
    bg_concat = pd.concat(bg_list, ignore_index=True)

    return p_concat, bg_concat

def plot_polar(df, ax=None, label=None, marker='o', linestyle='-', scale = 0.11):
    if ax is None:
        ax = plt.gca()

    window_size = df['window_size'].unique() * scale
    polar_order = df.groupby('window_size').mean()['polar_order']
    sem = df.groupby('window_size').sem()['polar_order']

    ax.errorbar(window_size, polar_order, yerr=sem, label=label, marker=marker, linestyle=linestyle)

    return window_size, polar_order, sem

def main():
    particles1um, bgs1um = func(root, "beads1um")
    particles3um, bgs3um = func(root, "beads3um")
    particles7um, bgs7um = func(root, "beads7um")
    df_p1, df_bg1 = make_df(particles1um, bgs1um)
    df_p3, df_bg3 = make_df(particles3um, bgs3um)
    df_p7, df_bg7 = make_df(particles7um, bgs7um)

    fig, ax = plt.subplots()
    plot_polar(df_p1, ax=ax, label='Particle', marker='o', linestyle='-')
    plot_polar(df_bg1, ax=ax, label='Background', marker='s', linestyle='--')
    plot_polar(df_p3, ax=ax, label='3um Beads', marker='^', linestyle='-')
    plot_polar(df_bg3, ax=ax, label='3um Beads Background', marker='v', linestyle='--')
    plot_polar(df_p7, ax=ax, label='7um Beads', marker='D', linestyle='-')
    plot_polar(df_bg7, ax=ax, label='7um Beads Background', marker='d', linestyle='--')
    ax.set_xlabel('Window Size (pixels)')
    ax.set_ylabel('Polar Order')
    ax.set_title('Polar Order vs Window Size')
    ax.legend()

if __name__ == "__main__":
    main()