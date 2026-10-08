#!/usr/bin/env python3
"""
Test script to inspect displacement tail behavior and fitting.
"""
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
current_dir = Path(__file__).resolve().parent.parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from libs import cal_vel as cv
from libs import displacement as dpm

root = Path('/mnt/NAS-Ebanaru/sasaki/MTsingleBeads')
beads = [
    {"name": "beads06um", "diameter_um": 0.63, "marker": "^"},
    {"name": "beads1um",  "diameter_um": 1.18, "marker": "o"},
    {"name": "beads3um",  "diameter_um": 3.37, "marker": "d"},
    {"name": "beads5um",  "diameter_um": 5.00, "marker": "p"},
    {"name": "beads7um",  "diameter_um": 7.24, "marker": "h"},
    {"name": "beads20um", "diameter_um": 20.0, "marker": "s"},
]

for b in beads:
    files = sorted(glob.glob(str(root / b["name"] / "*" / "*" / "beads_tracks.csv")))
    all_disps = []
    for f in files:
        df = pd.read_csv(f)
        disp = dpm.calc_displacement_magnitudes(df, tau=1, scale=0.11, component='norm')
        all_disps.extend(disp)
    all_disps = np.array(all_disps)
    all_disps = all_disps[all_disps > 0]
    print(f"{b['name']}: {len(all_disps)} displacements, min={all_disps.min():.3f}, max={all_disps.max():.3f}, median={np.median(all_disps):.3f}")
