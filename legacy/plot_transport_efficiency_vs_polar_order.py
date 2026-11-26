#!/usr/bin/env python3
"""
plot_transport_efficiency_vs_polar_order.py
===========================================

微小管ポーラーオーダー Phi (または 1 - Phi) に対する
貨物前進速度 v_parallel および足印流速 |<u_fp>| の解析・描画スクリプト。
ユーザー指定により、毎フレームごとの遠方流速 u_inf(t) で割って規格化。

【主要な仕様】
- 規格化:
  - 貨物速度: v_parallel / u_inf(t) (毎フレームごとの遠方微小管流速で無次元化)
  - 足印流速: |<u_fp>| / u_inf(t) (貨物周囲アニュラスの微小管正味流速)
- 横軸:
  1. 線形 Phi（0.0 〜 1.0、右肩上がりのすっきりしたグラフ、デフォルトでスタージェス則等の等間隔分割）
  2. 対数 1 - Phi（左が完全極性 Phi ~ 1、右が打ち消し合い Phi ~ 0）
- ビンの選び方: スタージェス則 (Sturges' formula, k = 16 bins)、FD則、または等幅分割。
  サンプル数が少ないビン（< min_points 点、または < min_particles 粒子）は隣接ビンと適応的に統合。
- 誤差棒: 粒子単位のクラスター・ブートストラップ（N=2000）による標準誤差 (SEM)。
- 基準線:
  - v_norm = 1.0 (Far-field flow speed u_inf)
  - v_norm = 0.0 (Zero Velocity / Stall)
- 出力図:
  1. 線形 Phi 版:
     - vpar_norm_vs_phi_linear_overlay.png / .svg (全ビーズ重ね描き図)
     - vpar_norm_vs_phi_linear_6panels.png / .svg (各ビーズ個別図：貨物速度 vs 足印の流れ)
     - vpar_norm_vs_phi_linear_binned.csv
  2. 従来の対数 1 - Phi 版も併せて出力可能
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = Path("/mnt/NAS-Ebanaru/sasaki/MTsingleBeads")
if not ROOT_DIR.exists():
    ROOT_DIR = CURRENT_DIR

OUT_DIRS = [
    CURRENT_DIR / "figure" / "polar_vs_normalized_velocity",
    ROOT_DIR / "figure" / "polar_vs_normalized_velocity"
]

# スタイルの適用
style_path = CURRENT_DIR / 'libs' / 'my_style.mplstyle'
if style_path.exists():
    try:
        plt.style.use(str(style_path))
        style_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    except Exception:
        style_colors = ['#882255', '#CC6677', '#DDCC77', '#999933', '#117733', '#44AA99']
else:
    style_colors = ['#882255', '#CC6677', '#DDCC77', '#999933', '#117733', '#44AA99']

BEADS_INFO = [
    {"name": "beads06um", "diameter_um": 0.63, "radius_um": 0.315, "marker": "^", "color": style_colors[0], "label": r"$0.63\,\mu\mathrm{m}$"},
    {"name": "beads1um",  "diameter_um": 1.18, "radius_um": 0.590, "marker": "o", "color": style_colors[1], "label": r"$1.18\,\mu\mathrm{m}$"},
    {"name": "beads3um",  "diameter_um": 3.37, "radius_um": 1.685, "marker": "d", "color": style_colors[2], "label": r"$3.37\,\mu\mathrm{m}$"},
    {"name": "beads5um",  "diameter_um": 5.00, "radius_um": 2.500, "marker": "p", "color": style_colors[3], "label": r"$5.00\,\mu\mathrm{m}$"},
    {"name": "beads7um",  "diameter_um": 7.24, "radius_um": 3.620, "marker": "h", "color": style_colors[4], "label": r"$7.24\,\mu\mathrm{m}$"},
    {"name": "beads20um", "diameter_um": 20.0, "radius_um": 10.00, "marker": "s", "color": style_colors[5], "label": r"$20.0\,\mu\mathrm{m}$"},
]

def sampled_frames_for_exp(exp_dir: Path, n_u: int) -> Optional[np.ndarray]:
    """
    calc_and_plot_cargo_flow_profile.py が u_inf(t) を取ったフレーム番号を再構成する
    (available_frames 上の linspace サンプリングを同じ手順で再現)。
    """
    import h5py
    csv_path = exp_dir / "beads_tracks.csv"
    h5_path = exp_dir / "GFP_flows.h5"
    if not (csv_path.exists() and h5_path.exists()):
        return None
    with h5py.File(str(h5_path), 'r') as hf:
        num_frames = hf[list(hf.keys())[0]].shape[0]
    frames = pd.read_csv(csv_path, usecols=['frame'])['frame'].unique()
    available = np.array([f for f in sorted(frames) if f < num_frames])
    idx = np.linspace(0, len(available) - 1, min(len(available), n_u), dtype=int)
    return available[idx]

def load_data(u_inf_mode: str = 'frame') -> pd.DataFrame:
    """
    足元アニュラス（貨物真下マスク）の局所ポーラーオーダーと無次元化速度データをロード・結合する。
    u_inf_mode='frame': 遠方プラトー (r>=20um) のフレームごとの u_inf(t) を時間補間して各観測フレームに割り当て。
    """
    cache_fp = CURRENT_DIR / 'scratch' / 'footprint_heterogeneity_cache.pkl'
    cache_ext = CURRENT_DIR / 'scratch' / 'cargo_flow_profile_extended_cache.pkl'

    if not cache_fp.exists():
        raise FileNotFoundError(f"Missing required cache: {cache_fp}")
    if not cache_ext.exists():
        raise FileNotFoundError(f"Missing required cache: {cache_ext}")

    df_fp = pd.read_pickle(cache_fp)
    df_ext = pd.read_pickle(cache_ext)

    ext_map = {Path(p).name: float(u) for p, u in zip(df_ext['exp_dir'], df_ext['u_inf'])}
    df_fp['u_inf_movie'] = df_fp['exp_dir'].map(ext_map)
    df_fp['u_inf'] = df_fp['u_inf_movie']

    if u_inf_mode == 'frame':
        if 'u_inf_frames' not in df_ext.columns:
            raise SystemExit("Extended cache lacks u_inf_frames. Re-run calc_and_plot_cargo_flow_profile.py --force_recompute.")
        for p, uf in zip(df_ext['exp_dir'], df_ext['u_inf_frames']):
            name = Path(p).name
            uf = np.asarray(uf, dtype=float)
            fr_s = sampled_frames_for_exp(Path(p), len(uf)) if len(uf) else None
            if fr_s is None or len(fr_s) != len(uf):
                continue
            m = df_fp['exp_dir'] == name
            df_fp.loc[m, 'u_inf'] = np.interp(df_fp.loc[m, 'frame'].values, fr_s, uf)

    # 毎フレームごとの u_inf(t) で規格化
    df_fp['vpar_norm'] = df_fp['v_parallel'] / df_fp['u_inf']
    df_fp['vvirt_norm'] = df_fp['u_fp'] / df_fp['u_inf']
    
    # 1 - Phi（対数軸用、下限 1e-5 クリップ）
    df_fp['one_minus_phi'] = np.clip(1.0 - df_fp['phi_t'], 1e-5, 1.0)

    return df_fp

def save_fig_all(fig: plt.Figure, base_name: str):
    """PNG & SVG で保存"""
    for out_dir in OUT_DIRS:
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_dir / f"{base_name}.png", dpi=300, bbox_inches='tight')
            fig.savefig(out_dir / f"{base_name}.svg", bbox_inches='tight')
            print(f"Saved: {out_dir / base_name}.png / .svg")
        except Exception as e:
            print(f"Warning: Failed to save {base_name} to {out_dir}: {e}")

def hierarchical_bootstrap_velocity(
    sub_bin: pd.DataFrame,
    x_col: str = "phi_t",
    n_boot: int = 2000,
    seed: int = 42
) -> Dict[str, Any]:
    """
    階層ブートストラップ (Hierarchical Bootstrap: movie -> cargo) による
    vpar_norm = v_parallel / u_inf(t) および
    vvirt_norm = |<u_fp>| / u_inf(t) の平均値と標準誤差 (SEM) の算出。

    階層構造:
      Level 1: 実験動画 (exp_dir) を復元抽出
      Level 2: 選択された動画内の各貨物粒子 (unique_pid) を復元抽出
    """
    unique_pids = sub_bin['unique_pid'].unique()
    unique_movies = sub_bin['exp_dir'].unique()
    n_p = len(unique_pids)
    n_m = len(unique_movies)

    if n_p == 0:
        return {
            'x_mean': np.nan,
            'vpar_mean': np.nan, 'vpar_sem': np.nan,
            'vvirt_mean': np.nan, 'vvirt_sem': np.nan,
            'n_particles': 0, 'n_movies': 0
        }

    orig_vpar = float(sub_bin['vpar_norm'].mean())
    orig_vvirt = float(sub_bin['vvirt_norm'].mean())
    orig_x = float(sub_bin[x_col].mean())

    if n_p == 1 or n_m == 0:
        return {
            'x_mean': orig_x,
            'vpar_mean': orig_vpar, 'vpar_sem': 0.0,
            'vvirt_mean': orig_vvirt, 'vvirt_sem': 0.0,
            'n_particles': n_p, 'n_movies': n_m
        }

    # 各 movie ごとに、所属する各 cargo の平均値 [vpar_mean, vvirt_mean] と重み (サンプル数) を事前計算
    mov_dict = {}
    for exp_k, grp in sub_bin.groupby('exp_dir'):
        p_stats = grp.groupby('unique_pid')[['vpar_norm', 'vvirt_norm']].mean().values
        p_weights = grp.groupby('unique_pid').size().values
        mov_dict[exp_k] = (p_stats, p_weights)

    exp_keys = list(mov_dict.keys())
    n_keys = len(exp_keys)

    rng = np.random.default_rng(seed)
    boot_vpar = []
    boot_vvirt = []

    for _ in range(n_boot):
        # Level 1: 動画の復元抽出
        sampled_keys = rng.choice(exp_keys, size=n_keys, replace=True)
        tot_weighted_vpar = 0.0
        tot_weighted_vvirt = 0.0
        tot_weight = 0.0

        for k in sampled_keys:
            p_stats, p_weights = mov_dict[k]
            n_c = len(p_weights)
            if n_c > 0:
                # Level 2: 動画内の貨物粒子の復元抽出
                c_idx = rng.integers(0, n_c, size=n_c)
                w_s = p_weights[c_idx]
                tot_weight += np.sum(w_s)
                tot_weighted_vpar += np.sum(p_stats[c_idx, 0] * w_s)
                tot_weighted_vvirt += np.sum(p_stats[c_idx, 1] * w_s)

        if tot_weight > 0:
            boot_vpar.append(tot_weighted_vpar / tot_weight)
            boot_vvirt.append(tot_weighted_vvirt / tot_weight)

    if len(boot_vpar) > 10:
        vpar_sem = float(np.std(boot_vpar, ddof=1))
        vpar_ci_lo, vpar_ci_hi = np.percentile(boot_vpar, [2.5, 97.5])
    else:
        vpar_sem = 0.0
        vpar_ci_lo, vpar_ci_hi = orig_vpar, orig_vpar

    if len(boot_vvirt) > 10:
        vvirt_sem = float(np.std(boot_vvirt, ddof=1))
        vvirt_ci_lo, vvirt_ci_hi = np.percentile(boot_vvirt, [2.5, 97.5])
    else:
        vvirt_sem = 0.0
        vvirt_ci_lo, vvirt_ci_hi = orig_vvirt, orig_vvirt

    return {
        'x_mean': orig_x,
        'vpar_mean': orig_vpar,
        'vpar_sem': vpar_sem,
        'vpar_ci_lo': float(vpar_ci_lo),
        'vpar_ci_hi': float(vpar_ci_hi),
        'vvirt_mean': orig_vvirt,
        'vvirt_sem': vvirt_sem,
        'vvirt_ci_lo': float(vvirt_ci_lo),
        'vvirt_ci_hi': float(vvirt_ci_hi),
        'n_particles': n_p,
        'n_movies': n_m
    }

def particle_cluster_bootstrap_velocity(
    sub_bin: pd.DataFrame,
    x_col: str = "phi_t",
    n_boot: int = 2000,
    seed: int = 42
) -> Dict[str, Any]:
    """
    粒子単位のクラスター・ブートストラップによる
    vpar_norm = v_parallel / u_inf(t) および
    vvirt_norm = |<u_fp>| / u_inf(t) の平均値と標準誤差 (SEM) の算出。
    """
    unique_pids = sub_bin['unique_pid'].unique()
    unique_movies = sub_bin['exp_dir'].unique()
    n_p = len(unique_pids)
    n_m = len(unique_movies)

    if n_p == 0:
        return {
            'x_mean': np.nan,
            'vpar_mean': np.nan, 'vpar_sem': np.nan,
            'vvirt_mean': np.nan, 'vvirt_sem': np.nan,
            'n_particles': 0, 'n_movies': 0
        }

    orig_vpar = float(sub_bin['vpar_norm'].mean())
    orig_vvirt = float(sub_bin['vvirt_norm'].mean())
    orig_x = float(sub_bin[x_col].mean())

    if n_p == 1:
        return {
            'x_mean': orig_x,
            'vpar_mean': orig_vpar, 'vpar_sem': 0.0,
            'vvirt_mean': orig_vvirt, 'vvirt_sem': 0.0,
            'n_particles': n_p, 'n_movies': n_m
        }

    # 各粒子ごとの平均値と重み（サンプル数）
    part_stats = sub_bin.groupby('unique_pid')[['vpar_norm', 'vvirt_norm']].mean().values
    part_weights = sub_bin.groupby('unique_pid').size().values
    w_sum = np.sum(part_weights)

    rng = np.random.default_rng(seed)
    boot_vpar = []
    boot_vvirt = []

    for _ in range(n_boot):
        c_idx = rng.choice(n_p, size=n_p, replace=True)
        w_sample = part_weights[c_idx]
        sw = np.sum(w_sample)
        if sw == 0:
            continue
        v_p = np.sum(part_stats[c_idx, 0] * w_sample) / sw
        v_v = np.sum(part_stats[c_idx, 1] * w_sample) / sw
        boot_vpar.append(v_p)
        boot_vvirt.append(v_v)

    if len(boot_vpar) > 10:
        vpar_sem = float(np.std(boot_vpar, ddof=1))
        vpar_ci_lo, vpar_ci_hi = np.percentile(boot_vpar, [2.5, 97.5])
    else:
        vpar_sem = 0.0
        vpar_ci_lo, vpar_ci_hi = orig_vpar, orig_vpar

    if len(boot_vvirt) > 10:
        vvirt_sem = float(np.std(boot_vvirt, ddof=1))
        vvirt_ci_lo, vvirt_ci_hi = np.percentile(boot_vvirt, [2.5, 97.5])
    else:
        vvirt_sem = 0.0
        vvirt_ci_lo, vvirt_ci_hi = orig_vvirt, orig_vvirt

    return {
        'x_mean': orig_x,
        'vpar_mean': orig_vpar,
        'vpar_sem': vpar_sem,
        'vpar_ci_lo': float(vpar_ci_lo),
        'vpar_ci_hi': float(vpar_ci_hi),
        'vvirt_mean': orig_vvirt,
        'vvirt_sem': vvirt_sem,
        'vvirt_ci_lo': float(vvirt_ci_lo),
        'vvirt_ci_hi': float(vvirt_ci_hi),
        'n_particles': n_p,
        'n_movies': n_m
    }

def get_bin_edges(
    series: pd.Series,
    x_mode: str = "phi",
    bin_rule: str = "sturges",
    n_bins: Optional[int] = None
) -> np.ndarray:
    """
    指定されたルールに基づいてビンの境界（エッジ）を決定する。
    """
    if x_mode == "phi":
        if bin_rule == "sturges":
            edges = np.histogram_bin_edges(series.values, bins='sturges', range=(0.0, 1.0))
        elif bin_rule == "fd":
            edges = np.histogram_bin_edges(series.values, bins='fd', range=(0.0, 1.0))
        elif bin_rule == "equal10":
            edges = np.linspace(0.0, 1.0, 11)
        elif bin_rule == "equal20":
            edges = np.linspace(0.0, 1.0, 21)
        elif bin_rule == "quantile":
            k = n_bins if n_bins is not None else 10
            edges = np.quantile(series.values, np.linspace(0, 1, k + 1))
        else:
            k = n_bins if n_bins is not None else 16
            edges = np.linspace(0.0, 1.0, k + 1)
    else:
        # 1 - Phi (対数軸用)
        k = n_bins if n_bins is not None else 10
        _, edges = pd.qcut(series, q=k, retbins=True)
        
    return edges

def compute_velocity_binned_stats(
    df: pd.DataFrame,
    x_mode: str = "phi",
    bin_rule: str = "sturges",
    n_bins: Optional[int] = None,
    min_points: int = 50,
    min_particles: int = 2,
    bootstrap_method: str = "hierarchical",
    n_boot: int = 2000,
    seed: int = 42
) -> pd.DataFrame:
    """
    全サイズ共通のエッジを基準とし、データ数が min_points 未満または
    粒子数が min_particles 未満のビンは隣接ビンと適応的に統合して
    規格化速度の階層/粒子ブートストラップ統計量を算出する。
    """
    df_work = df.copy()
    x_col = 'phi_t' if x_mode == 'phi' else 'one_minus_phi'
    
    bin_edges = get_bin_edges(df_work[x_col], x_mode=x_mode, bin_rule=bin_rule, n_bins=n_bins)
    n_orig_bins = len(bin_edges) - 1

    binned_rows = []

    for item in BEADS_INFO:
        bname = item['name']
        sub = df_work[df_work['bead_name'] == bname].copy()
        if sub.empty:
            continue

        b_indices = np.digitize(sub[x_col].values, bin_edges) - 1
        b_indices = np.clip(b_indices, 0, n_orig_bins - 1)
        counts = [int(np.sum(b_indices == i)) for i in range(n_orig_bins)]

        # ビン統合グループの決定（左から走査し、min_points または min_particles 未満は統合）
        merged_groups = []
        curr_group = []
        curr_count = 0
        curr_pids = set()
        for i in range(n_orig_bins):
            curr_group.append(i)
            curr_count += counts[i]
            pids_in_bin = sub.loc[b_indices == i, 'unique_pid'].unique()
            curr_pids.update(pids_in_bin)

            if (curr_count >= min_points and len(curr_pids) >= min_particles) or i == n_orig_bins - 1:
                merged_groups.append(curr_group)
                curr_group = []
                curr_count = 0
                curr_pids = set()

        if len(merged_groups) > 1:
            last_mask = np.isin(b_indices, merged_groups[-1])
            last_cnt = int(np.sum(last_mask))
            last_pids = sub.loc[last_mask, 'unique_pid'].nunique()
            if last_cnt < min_points or last_pids < min_particles:
                last_g = merged_groups.pop()
                merged_groups[-1].extend(last_g)

        # 各統合ビンについて統計量を計算
        for g_idx, g_bins in enumerate(merged_groups):
            mask = np.isin(b_indices, g_bins)
            sub_bin = sub[mask]
            n_pts = len(sub_bin)
            if n_pts == 0:
                continue

            if bootstrap_method == "hierarchical":
                res = hierarchical_bootstrap_velocity(sub_bin, x_col=x_col, n_boot=n_boot, seed=seed)
            elif bootstrap_method == "particle":
                res = particle_cluster_bootstrap_velocity(sub_bin, x_col=x_col, n_boot=n_boot, seed=seed)
            else:
                raise ValueError(f"Unknown bootstrap_method: {bootstrap_method}")

            b_left = bin_edges[g_bins[0]]
            b_right = bin_edges[g_bins[-1] + 1]

            binned_rows.append({
                'bead_name': bname,
                'diameter_um': item['diameter_um'],
                'label': item['label'],
                'group_idx': g_idx,
                'merged_bins': str(g_bins),
                'bin_range': f"[{b_left:.5f}, {b_right:.5f}]",
                'n_points': n_pts,
                'n_particles': res['n_particles'],
                'n_movies': res['n_movies'],
                'x_mean': res['x_mean'],
                'phi_mean': float(sub_bin['phi_t'].mean()),
                'one_minus_phi_mean': float(sub_bin['one_minus_phi'].mean()),
                'vpar_mean': res['vpar_mean'],
                'vpar_sem': res['vpar_sem'],
                'vpar_ci_lo': res['vpar_ci_lo'],
                'vpar_ci_hi': res['vpar_ci_hi'],
                'vvirt_mean': res['vvirt_mean'],
                'vvirt_sem': res['vvirt_sem'],
                'vvirt_ci_lo': res['vvirt_ci_lo'],
                'vvirt_ci_hi': res['vvirt_ci_hi'],
                'u_inf_mean': float(sub_bin['u_inf'].mean()),
            })

    return pd.DataFrame(binned_rows)

def plot_vpar_norm_linear_overlay(
    df_binned: pd.DataFrame,
    bootstrap_method: str = "hierarchical",
    error_bar: str = "ci95"
):
    """
    全ビーズ重ね描き図（毎フレームごとの u_inf(t) で規格化、線形 Phi、右肩上がり、すっきりしたデザイン）:
    横軸: Microtubule Polar Order Phi (0.0 〜 1.0)
    縦軸: Normalized Cargo Velocity v_parallel / u_inf(t)
    """
    fig, ax = plt.subplots(figsize=(9.8, 6.5))

    method_str = "Hierarchical" if bootstrap_method == "hierarchical" else "Particle"
    err_str = f"{method_str} 95% CI" if error_bar == "ci95" else f"{method_str} SEM"

    # 基準線: 完全停止 v=0
    ax.axhline(0.0, color='#888888', linestyle='-', linewidth=0.9, alpha=0.8, zorder=1)
    ax.text(0.98, -0.035, r"Zero Velocity / Stall ($v_\parallel = 0$)", ha='right', fontsize=11.0, color='#666666')

    for item in BEADS_INFO:
        bname = item['name']
        color = item['color']
        marker = item['marker']
        lbl = item['label']

        sub = df_binned[df_binned['bead_name'] == bname].sort_values('x_mean')
        if sub.empty:
            continue

        total_pts = sub['n_points'].sum()

        if error_bar == "ci95" and 'vpar_ci_lo' in sub.columns and 'vpar_ci_hi' in sub.columns:
            yerr_lo = np.maximum(0.0, sub['vpar_mean'] - sub['vpar_ci_lo'])
            yerr_hi = np.maximum(0.0, sub['vpar_ci_hi'] - sub['vpar_mean'])
            yerr = [yerr_lo.values, yerr_hi.values]
        else:
            yerr = sub['vpar_sem'].values

        ax.errorbar(
            sub['x_mean'], sub['vpar_mean'], yerr=yerr,
            color=color, linestyle='-', linewidth=2.4,
            marker=marker, markersize=8.0, markerfacecolor=color, markeredgecolor='#111111',
            markeredgewidth=1.1, capsize=3.5, capthick=1.2,
            label=f"{lbl} ($N={total_pts:,}$)", zorder=4
        )

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.08, 0.48)
    ax.set_xticks(np.linspace(0.0, 1.0, 6))

    ax.set_xlabel(r"Microtubule Polar Order $\Phi$ (Opposing $\rightarrow$ Fully Polar)", fontsize=14)
    ax.set_ylabel(r"Normalized Cargo Velocity $v_\parallel / u_\infty(t)$", fontsize=14)
    ax.set_title(
        r"Cargo Velocity $v_\parallel / u_\infty(t)$ vs Polar Order $\Phi$"
        + f" ({err_str}, Frame-wise $u_\infty(t)$)",
        fontsize=15.0, fontweight='bold', pad=12
    )

    ax.legend(
        fontsize=11.0,
        loc='upper left',
        bbox_to_anchor=(0.02, 0.96),
        framealpha=0.92,
        borderpad=0.45,
        labelspacing=0.35,
        handletextpad=0.4
    )

    fig.tight_layout()
    save_fig_all(fig, "vpar_norm_vs_phi_linear_overlay")
    plt.close(fig)

def plot_vpar_norm_linear_6panels(
    df_binned: pd.DataFrame,
    bootstrap_method: str = "hierarchical",
    error_bar: str = "ci95"
):
    """
    ビーズ径別 6パネル個別図（線形 Phi、実測貨物速度 vs 足印の流れ）:
    横軸: Microtubule Polar Order Phi (0.0 〜 1.0)
    縦軸: Normalized Velocity (v / u_inf(t))
    """
    fig, axes = plt.subplots(2, 3, figsize=(15.6, 9.5), sharex=True, sharey=True)
    axes_flat = axes.flatten()
    method_str = "Hierarchical" if bootstrap_method == "hierarchical" else "Particle"
    err_str = f"{method_str} 95% CI" if error_bar == "ci95" else f"{method_str} SEM"

    for idx, item in enumerate(BEADS_INFO):
        ax = axes_flat[idx]
        bname = item['name']
        color = item['color']
        marker = item['marker']
        lbl = item['label']
        d_um = item['diameter_um']

        sub = df_binned[df_binned['bead_name'] == bname].sort_values('x_mean')

        # 基準線
        ax.axhline(0.0, color='#888888', linestyle='-', linewidth=0.9, alpha=0.8, zorder=1)
        ax.axhline(1.0, color='#222222', linestyle='--', linewidth=1.2, alpha=0.7, zorder=1)

        if sub.empty:
            ax.text(0.5, 0.5, "No Data", ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title(f"{lbl}", fontsize=15, fontweight='bold')
            continue

        # エラーバーの選定
        if error_bar == "ci95" and 'vpar_ci_lo' in sub.columns and 'vpar_ci_hi' in sub.columns:
            yerr_vpar_lo = np.maximum(0.0, sub['vpar_mean'] - sub['vpar_ci_lo'])
            yerr_vpar_hi = np.maximum(0.0, sub['vpar_ci_hi'] - sub['vpar_mean'])
            yerr_vpar = [yerr_vpar_lo.values, yerr_vpar_hi.values]

            yerr_vvirt_lo = np.maximum(0.0, sub['vvirt_mean'] - sub['vvirt_ci_lo'])
            yerr_vvirt_hi = np.maximum(0.0, sub['vvirt_ci_hi'] - sub['vvirt_mean'])
            yerr_vvirt = [yerr_vvirt_lo.values, yerr_vvirt_hi.values]
        else:
            yerr_vpar = sub['vpar_sem'].values
            yerr_vvirt = sub['vvirt_sem'].values

        # 1. 足印の流れ |<u_fp>| / u_inf(t)
        ax.errorbar(
            sub['x_mean'], sub['vvirt_mean'], yerr=yerr_vvirt,
            color='#555555', linestyle=':', linewidth=2.0,
            marker='o', markersize=6.0, markerfacecolor='white', markeredgecolor='#555555',
            markeredgewidth=1.2, capsize=3.0, capthick=1.0,
            label=r"Footprint Flow $|\langle\mathbf{u}\rangle_{\mathrm{fp}}| / u_\infty(t)$", zorder=3
        )

        # 2. 実測貨物速度 v_parallel / u_inf(t)
        ax.errorbar(
            sub['x_mean'], sub['vpar_mean'], yerr=yerr_vpar,
            color=color, linestyle='-', linewidth=2.4,
            marker=marker, markersize=8.0, markerfacecolor=color, markeredgecolor='#111111',
            markeredgewidth=1.1, capsize=3.5, capthick=1.3,
            label=r"Cargo $v_\parallel / u_\infty(t)$", zorder=5
        )

        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.15, 1.0)
        ax.set_xticks(np.linspace(0.0, 1.0, 6))

        total_pts = sub['n_points'].sum()
        max_p = sub['n_particles'].max()
        max_m = sub['n_movies'].max() if 'n_movies' in sub.columns else None
        m_info = f", movies: {max_m}" if max_m is not None else ""
        ax.text(
            0.05, 0.94,
            f"{lbl} ($2R_c = {d_um:.2f}\\,\\mu\\mathrm{{m}}$)\n$N = {total_pts:,}$ (particles: {max_p}{m_info})",
            transform=ax.transAxes,
            fontsize=11.5,
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.35', facecolor='white', alpha=0.90, edgecolor='#cccccc')
        )

        ax.set_title(f"{lbl} ($2R_c = {d_um:.2f}\\,\\mu\\mathrm{{m}}$)", fontsize=14.5, fontweight='bold')

        if idx >= 3:
            ax.set_xlabel(r"Microtubule Polar Order $\Phi$ (Opposing $\rightarrow$ Fully Polar)", fontsize=12.5)

        if idx % 3 == 0:
            ax.set_ylabel(r"Normalized Velocity ($v_{\parallel}/ u_\infty(t)$)", fontsize=13.0)

        if idx == 0:
            ax.legend(loc='center left', bbox_to_anchor=(0.04, 0.65), fontsize=10.0, framealpha=0.90)

    fig.suptitle(
        r"Cargo Velocity $v_\parallel / u_\infty(t)$ and Footprint Flow $|\langle\mathbf{u}\rangle_{\mathrm{fp}}| / u_\infty(t)$ vs Polar Order $\Phi$"
        + f" ({err_str})",
        fontsize=16.0, fontweight='bold', y=0.995
    )
    fig.tight_layout()
    save_fig_all(fig, "vpar_norm_vs_phi_linear_6panels")
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(
        description="Plot cargo normalized velocity vs polar order with frame-wise u_inf normalization."
    )
    parser.add_argument(
        '--bin_rule', type=str, default='sturges', choices=['sturges', 'fd', 'equal10', 'equal20', 'quantile'],
        help="Binning rule for linear Phi: 'sturges' (default, 16 bins), 'fd', 'equal10', 'equal20', or 'quantile'."
    )
    parser.add_argument(
        '--bootstrap_method', type=str, default='hierarchical', choices=['hierarchical', 'particle'],
        help="Bootstrap method: 'hierarchical' (movie -> cargo, default) or 'particle' (cargo cluster)."
    )
    parser.add_argument(
        '--error_bar', type=str, default='ci95', choices=['ci95', 'sem'],
        help="Error bar representation: 'ci95' (95%% bootstrap CI, default) or 'sem' (bootstrap SEM)."
    )
    parser.add_argument(
        '--min_points', type=int, default=50,
        help="Minimum data points per bin before adaptive merging (default: 50)."
    )
    parser.add_argument(
        '--min_particles', type=int, default=2,
        help="Minimum number of unique particles per bin (default: 2)."
    )
    parser.add_argument(
        '--n_boot', type=int, default=2000,
        help="Number of bootstrap iterations (default: 2000)."
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help="Random seed for bootstrap reproducibility (default: 42)."
    )
    args = parser.parse_args()

    print("Loading footprint heterogeneity data with frame-wise u_inf(t) normalization...")
    df = load_data(u_inf_mode='frame')
    print(f"Loaded {len(df):,} instantaneous observations.")

    method_str = "Hierarchical bootstrap (movie -> cargo)" if args.bootstrap_method == "hierarchical" else "Particle cluster bootstrap"
    print(f"\nComputing linear Phi bins (rule: {args.bin_rule}, min_points: {args.min_points}, min_particles: {args.min_particles})...")
    print(f"Evaluating {method_str} for v_parallel / u_inf(t) and |<u_fp>| / u_inf(t) (N={args.n_boot})...")
    df_binned = compute_velocity_binned_stats(
        df,
        x_mode="phi",
        bin_rule=args.bin_rule,
        min_points=args.min_points,
        min_particles=args.min_particles,
        bootstrap_method=args.bootstrap_method,
        n_boot=args.n_boot,
        seed=args.seed
    )

    for out_dir in OUT_DIRS:
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            csv_path = out_dir / "vpar_norm_vs_phi_linear_binned.csv"
            df_binned.to_csv(csv_path, index=False)
            print(f"Saved binned CSV to {csv_path}")
        except Exception as e:
            print(f"Warning: Failed to save CSV to {out_dir}: {e}")

    print("\nGenerating clean linear Phi figures with frame-wise u_inf(t) normalization...")
    plot_vpar_norm_linear_overlay(df_binned, bootstrap_method=args.bootstrap_method, error_bar=args.error_bar)
    plot_vpar_norm_linear_6panels(df_binned, bootstrap_method=args.bootstrap_method, error_bar=args.error_bar)

    print("\n[Complete] All velocity plots generated successfully.")

if __name__ == '__main__':
    main()
