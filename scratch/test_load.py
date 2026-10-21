import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter

# データ読み込み
df = pd.read_csv('figure/msd/msd_pooled_mean_iqr_summary.csv')

print("Loaded msd_pooled_mean_iqr_summary.csv successfully.")
print(df.head())
