import h5py
import pandas as pd
from pathlib import Path

root = Path('/mnt/NAS-Ebanaru/Sasaki/MTsingleBeads')
folder = root / "beads1um" / "20260122" / "exp001"
try:
    beads = pd.read_csv(folder / "beads_tracks.csv")
    print("Beads columns:", beads.columns)
    print(beads.head())
except Exception as e:
    print(e)

print("---------------------------------")
try:
    with h5py.File(folder / "GFP_flows.h5", 'r') as f:
        print("H5 keys:", list(f.keys()))
        for key in f.keys():
            if isinstance(f[key], h5py.Dataset):
                print(f"{key}: shape={f[key].shape}, dtype={f[key].dtype}")
            elif isinstance(f[key], h5py.Group):
                print(f"Group {key}:", list(f[key].keys()))
                for subkey in f[key].keys():
                    print(f"  {subkey}: shape={f[key][subkey].shape}, dtype={f[key][subkey].dtype}")
except Exception as e:
    print(e)

