import pandas as pd
import numpy as np

df = pd.read_csv('figure/polar_vs_normalized_velocity/polar_vs_vpar_norm_binned.csv')
print(df.columns.tolist())
df['eta'] = df['vpar_mean'] / df['vvirt_mean']
# Approx SEM for ratio: eta * sqrt((vpar_sem/vpar_mean)^2 + (vvirt_sem/vvirt_mean)^2)
df['eta_sem'] = np.abs(df['eta']) * np.sqrt(
    (df['vpar_sem'] / np.maximum(np.abs(df['vpar_mean']), 1e-4))**2 +
    (df['vvirt_sem'] / np.maximum(df['vvirt_mean'], 1e-4))**2
)

for bname, grp in df.groupby('bead_name'):
    print(f"\n=== {bname} ===")
    for _, r in grp.iterrows():
        print(f"1-Phi = {r['x_mean_one_minus_phi']:.4f} | eta = {r['eta']:.3f} +- {r['eta_sem']:.3f} (vpar={r['vpar_mean']:.3f}, vvirt={r['vvirt_mean']:.3f})")
