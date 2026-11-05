import pandas as pd
import numpy as np

def build_merged_binned_stats(sub: pd.DataFrame, bin_edges: np.ndarray, min_points: int = 50):
    """
    sub のデータを bin_edges で分割し、points < min_points のビンは隣と統合する
    """
    # 各行の bin index (0 to len(bin_edges)-2)
    b_indices = np.digitize(sub['one_minus_phi'].values, bin_edges) - 1
    n_orig_bins = len(bin_edges) - 1
    
    # 各ビンのインデックスリストを構築
    bin_groups = [[i] for i in range(n_orig_bins)]
    
    # カウントを計算
    counts = [np.sum(b_indices == i) for i in range(n_orig_bins)]
    
    # 右端から左へ、または左から右へ min_points 未満のものをマージ
    merged_groups = []
    curr_group = []
    curr_count = 0
    for i in range(n_orig_bins):
        curr_group.append(i)
        curr_count += counts[i]
        if curr_count >= min_points or i == n_orig_bins - 1:
            merged_groups.append(curr_group)
            curr_group = []
            curr_count = 0
    # もし最後のグループが min_points 未満で残ったら、前のグループと統合
    if len(merged_groups) > 1 and curr_count > 0:
        merged_groups[-1].extend(curr_group)
    elif len(merged_groups) > 1 and sum(counts[idx] for idx in merged_groups[-1]) < min_points:
        last = merged_groups.pop()
        merged_groups[-1].extend(last)
        
    return merged_groups

df_fp = pd.read_pickle('scratch/footprint_heterogeneity_cache.pkl')
df_fp['one_minus_phi'] = np.clip(1.0 - df_fp['phi_t'], 1e-5, 1.0)
n_bins = 10
_, bin_edges = pd.qcut(df_fp['one_minus_phi'], q=n_bins, retbins=True)

for bname in ['beads06um', 'beads1um', 'beads20um']:
    sub = df_fp[df_fp['bead_name'] == bname]
    mg = build_merged_binned_stats(sub, bin_edges, min_points=50)
    print(f"{bname} merged bins (min 50 pts): {mg}")
    for g in mg:
        sub_pts = sub[np.isin(np.digitize(sub['one_minus_phi'].values, bin_edges) - 1, g)]
        print(f"   group {g}: N={len(sub_pts)}, particles={sub_pts['unique_pid'].nunique()}, 1-Phi={sub_pts['one_minus_phi'].mean():.4f}")
