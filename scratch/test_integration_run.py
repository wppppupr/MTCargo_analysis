"""
scratch/test_integration_run.py
"""

import h5py
import numpy as np
from pathlib import Path
import subprocess

scratch_dir = Path(__file__).resolve().parent
mock_exp_dir = scratch_dir / "mock_condition" / "mock_exp1"
mock_exp_dir.mkdir(parents=True, exist_ok=True)

# 模擬 GFP_flows.h5 を作成 (T=10, Channels=2, H=128, W=128)
flow_h5 = mock_exp_dir / "GFP_flows.h5"
T, H, W = 10, 128, 128
y, x = np.mgrid[:H, :W]

# 逆並行ドメイン場
stripe = ((y // 16) % 2) * 2 - 1
flow_data = np.zeros((T, 2, H, W), dtype=np.float32)
for t in range(T):
    flow_data[t, 0, :, :] = stripe + np.random.normal(0, 0.05, (H, W))
    flow_data[t, 1, :, :] = np.random.normal(0, 0.05, (H, W))

with h5py.File(str(flow_h5), 'w') as f:
    f.create_dataset('flow', data=flow_data)

print(f"Created mock flow data at {flow_h5}")

# spatial_heterogeneity_analysis.py を単一実験モードで実行
cmd = [
    "pixi", "run", "python", "spatial_heterogeneity_analysis.py",
    "--input_dir", str(mock_exp_dir),
    "--distances", "2:30:2",
    "--grid_step", "8",
    "--out_dir", str(scratch_dir / "figure_out")
]

print("Running command:", " ".join(cmd))
res = subprocess.run(cmd, capture_output=True, text=True)
print("Return code:", res.returncode)
print("Stdout:", res.stdout)
print("Stderr:", res.stderr)
