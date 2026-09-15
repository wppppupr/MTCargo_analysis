"""
scratch/test_batch_integration.py
"""

import h5py
import numpy as np
from pathlib import Path
import subprocess

scratch_dir = Path(__file__).resolve().parent
mock_root = scratch_dir / "mock_root"
mock_root.mkdir(parents=True, exist_ok=True)

conditions = ["beads06um", "beads1um", "beads3um"]
domain_sizes = [8, 16, 24]
T, H, W = 6, 128, 128
y, x = np.mgrid[:H, :W]

for cond, d_size in zip(conditions, domain_sizes):
    cond_dir = mock_root / cond / "exp_01"
    cond_dir.mkdir(parents=True, exist_ok=True)
    flow_h5 = cond_dir / "GFP_flows.h5"

    stripe = ((y // d_size) % 2) * 2 - 1
    flow_data = np.zeros((T, 2, H, W), dtype=np.float32)
    for t in range(T):
        flow_data[t, 0, :, :] = stripe + np.random.normal(0, 0.05, (H, W))
        flow_data[t, 1, :, :] = np.random.normal(0, 0.05, (H, W))

    with h5py.File(str(flow_h5), 'w') as f:
        f.create_dataset('flow', data=flow_data)

print(f"Created mock batch structure under {mock_root}")

# Batch run
cmd = [
    "pixi", "run", "python", "spatial_heterogeneity_analysis.py",
    "--root_dir", str(mock_root),
    "--batch",
    "--distances", "2:30:2",
    "--grid_step", "8",
    "--out_dir", str(scratch_dir / "batch_figure_out")
]

print("Running command:", " ".join(cmd))
res = subprocess.run(cmd, capture_output=True, text=True)
print("Return code:", res.returncode)
print("Stdout:", res.stdout)
print("Stderr:", res.stderr)
