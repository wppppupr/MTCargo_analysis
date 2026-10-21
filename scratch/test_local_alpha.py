import pandas as pd
import numpy as np

df = pd.read_csv('figure/msd/msd_pooled_mean_iqr_summary.csv')
for b in df['bead_name'].unique():
    sub = df[df['bead_name'] == b].sort_values('lag_time_s')
    t = sub['lag_time_s'].values
    msd = sub['mean_msd_um2'].values
    log_t = np.log10(t)
    log_msd = np.log10(msd)
    dlog = np.gradient(log_msd, log_t)
    print(f"{b}: min lag={t[0]}, max lag={t[-1]}, dlog[0:15]={np.round(dlog[:15], 3)}")
