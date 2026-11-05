import pandas as pd
df = pd.read_pickle('scratch/footprint_heterogeneity_cache.pkl')
print(df.columns.tolist())
print(df[['phi_t', 'u_fp', 'u_local_scalar', 'v_parallel', 'v_cargo']].head(10))
