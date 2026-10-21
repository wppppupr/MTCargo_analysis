import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('figure/msd/msd_pooled_mean_iqr_summary.csv')

fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
axes = axes.flatten()

for idx, b in enumerate(df['bead_name'].unique()):
    sub = df[df['bead_name'] == b].sort_values('lag_time_s')
    t = sub['lag_time_s'].values
    msd = sub['mean_msd_um2'].values
    
    log_t = np.log10(t)
    log_msd = np.log10(msd)
    
    # 局所微分
    alpha_local = np.gradient(log_msd, log_t)
    
    ax = axes[idx]
    ax.plot(t, alpha_local, label='Raw gradient', color='tab:blue', alpha=0.5, lw=1)
    ax.set_xscale('log')
    ax.set_title(b)
    ax.axhline(2.0, color='r', linestyle='--', alpha=0.5, label=r'$\Delta t^2$ (ballistic)')
    ax.axhline(1.0, color='g', linestyle='--', alpha=0.5, label=r'$\Delta t^1$ (diffusive)')
    ax.axhline(1.5, color='orange', linestyle=':', alpha=0.7, label=r'$\alpha=1.5$')
    ax.set_ylim(0, 2.5)
    ax.set_xlabel(r'$\Delta t$ [s]')
    ax.set_ylabel(r'$\alpha(t) = d\ln\mathrm{MSD}/d\ln t$')
    ax.grid(True, which='both', ls='--', alpha=0.3)
    if idx == 0:
        ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('scratch/raw_local_alpha.png', dpi=150)
print("Saved scratch/raw_local_alpha.png")
