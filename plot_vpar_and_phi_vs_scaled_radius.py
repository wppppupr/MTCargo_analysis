#!/usr/bin/env python3
"""
plot_vpar_and_phi_vs_scaled_radius.py
======================================

横軸: 貨物半径の無次元化スケール R_c / xi (対数軸)
縦軸 (第1軸・左): 規格化貨物前進速度 v_parallel / u_inf(t)
縦軸 (第2軸・右): 微小管ポーラーオーダーパラメータ Phi

【主要な仕様】
- 横軸: R_c / xi (xi は各動画ごとの背景配向相関長)
- 第1軸 (左): v_parallel / u_inf(t) (毎フレームごとの遠方流速 u_inf(t) で規格化)
- 第2軸 (右): 足元アニュラスの微小管ポーラーオーダー Phi
- 統計・誤差棒: 階層ブートストラップ (movie -> cargo, N=2000) による 95% 信頼区間 (および SEM)
- 小マーカー: 各実験動画単位のプロット (データ分散の可視化)
- 大マーカー: サイズ別平均値 (階層 95% CI エラーバー付き)
- 基準線:
  - R_c / xi = 1.0 (相関長との幾何学的クロスオーバー境界)
  - v_parallel / u_inf = 0.0 (完全停止・スタック境界)
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
    {"name": "beads7um",  "diameter_um": 7.24, "radius_um": 3.620, "marker": "h", "color": style_colors[4], "label": r"$7.24\,\mu\mathrm{m}$"},
    {"name": "beads5um",  "diameter_um": 5.00, "radius_um": 2.500, "marker": "p", "color": style_colors[3], "label": r"$5.00\,\mu\mathrm{m}$"},
    {"name": "beads20um", "diameter_um": 20.0, "radius_um": 10.00, "marker": "s", "color": style_colors[5], "label": r"$20.0\,\mu\mathrm{m}$"},
]

def sampled_frames_for_exp(exp_dir: Path, n_u: int) -> Optional[np.ndarray]:
    """
    calc_and_plot_cargo_flow_profile.py が u_inf(t) を取ったフレーム番号を再構成する。
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

    df_fp['vpar_norm'] = df_fp['v_parallel'] / df_fp['u_inf']
    df_fp['vvirt_norm'] = df_fp['u_fp'] / df_fp['u_inf']
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

def hierarchical_bootstrap_stats(
    sub_df: pd.DataFrame,
    n_boot: int = 2000,
    seed: int = 42
) -> Dict[str, Any]:
    """
    階層ブートストラップ (movie -> cargo) による
    vpar_norm, phi_t, rc_over_xi の平均値、SEM、95%信頼区間の算出。
    """
    unique_pids = sub_df['unique_pid'].unique()
    unique_movies = sub_df['exp_dir'].unique()
    n_p = len(unique_pids)
    n_m = len(unique_movies)

    orig_vpar = float(sub_df['vpar_norm'].mean())
    orig_phi = float(sub_df['phi_t'].mean())
    orig_rc_xi = float(sub_df['rc_over_xi'].mean())

    if n_p <= 1 or n_m == 0:
        return {
            'rc_xi_mean': orig_rc_xi, 'rc_xi_sem': 0.0,
            'vpar_mean': orig_vpar, 'vpar_sem': 0.0, 'vpar_ci_lo': orig_vpar, 'vpar_ci_hi': orig_vpar,
            'phi_mean': orig_phi, 'phi_sem': 0.0, 'phi_ci_lo': orig_phi, 'phi_ci_hi': orig_phi,
            'n_particles': n_p, 'n_movies': n_m
        }

    # 各 movie ごとに、所属する各 cargo の平均値 [vpar_mean, phi_mean, rc_xi_mean] と重み (サンプル数) を事前計算
    mov_dict = {}
    for exp_k, grp in sub_df.groupby('exp_dir'):
        p_stats = grp.groupby('unique_pid')[['vpar_norm', 'phi_t', 'rc_over_xi']].mean().values
        p_weights = grp.groupby('unique_pid').size().values
        mov_dict[exp_k] = (p_stats, p_weights)

    exp_keys = list(mov_dict.keys())
    n_keys = len(exp_keys)

    rng = np.random.default_rng(seed)
    boot_vpar = []
    boot_phi = []
    boot_rc_xi = []

    for _ in range(n_boot):
        sampled_keys = rng.choice(exp_keys, size=n_keys, replace=True)
        tot_w = 0.0
        tot_v = 0.0
        tot_p = 0.0
        tot_r = 0.0

        for k in sampled_keys:
            p_stats, p_weights = mov_dict[k]
            n_c = len(p_weights)
            if n_c > 0:
                c_idx = rng.integers(0, n_c, size=n_c)
                w_s = p_weights[c_idx]
                tot_w += np.sum(w_s)
                tot_v += np.sum(p_stats[c_idx, 0] * w_s)
                tot_p += np.sum(p_stats[c_idx, 1] * w_s)
                tot_r += np.sum(p_stats[c_idx, 2] * w_s)

        if tot_w > 0:
            boot_vpar.append(tot_v / tot_w)
            boot_phi.append(tot_p / tot_w)
            boot_rc_xi.append(tot_r / tot_w)

    vpar_sem = float(np.std(boot_vpar, ddof=1)) if len(boot_vpar) > 10 else 0.0
    vpar_ci_lo, vpar_ci_hi = np.percentile(boot_vpar, [2.5, 97.5]) if len(boot_vpar) > 10 else (orig_vpar, orig_vpar)

    phi_sem = float(np.std(boot_phi, ddof=1)) if len(boot_phi) > 10 else 0.0
    phi_ci_lo, phi_ci_hi = np.percentile(boot_phi, [2.5, 97.5]) if len(boot_phi) > 10 else (orig_phi, orig_phi)

    rc_xi_sem = float(np.std(boot_rc_xi, ddof=1)) if len(boot_rc_xi) > 10 else 0.0

    return {
        'rc_xi_mean': orig_rc_xi,
        'rc_xi_sem': rc_xi_sem,
        'vpar_mean': orig_vpar,
        'vpar_sem': vpar_sem,
        'vpar_ci_lo': float(vpar_ci_lo),
        'vpar_ci_hi': float(vpar_ci_hi),
        'phi_mean': orig_phi,
        'phi_sem': phi_sem,
        'phi_ci_lo': float(phi_ci_lo),
        'phi_ci_hi': float(phi_ci_hi),
        'n_particles': n_p,
        'n_movies': n_m
    }

def main():
    parser = argparse.ArgumentParser(
        description="Plot normalized velocity (left) and polar order (right) vs scaled radius Rc/xi."
    )
    parser.add_argument(
        '--n_boot', type=int, default=2000,
        help="Number of bootstrap iterations (default: 2000)."
    )
    parser.add_argument(
        '--error_bar', type=str, default='ci95', choices=['ci95', 'sem'],
        help="Error bar representation: 'ci95' (95%% bootstrap CI, default) or 'sem' (bootstrap SEM)."
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help="Random seed for bootstrap reproducibility (default: 42)."
    )
    args = parser.parse_args()

    print("Loading footprint heterogeneity data with frame-wise u_inf(t) normalization...")
    df = load_data(u_inf_mode='frame')
    print(f"Loaded {len(df):,} instantaneous observations.")

    # 1. 動画ごとの統計量 (per-movie points)
    movie_rows = []
    for exp_k, grp in df.groupby('exp_dir'):
        bname = grp['bead_name'].iloc[0]
        d_um = grp['diameter_um'].iloc[0]
        rc_um = grp['radius_um'].iloc[0]
        rc_xi = float(grp['rc_over_xi'].mean())
        vp = float(grp['vpar_norm'].mean())
        phi = float(grp['phi_t'].mean())
        n_p = grp['unique_pid'].nunique()
        n_pts = len(grp)
        movie_rows.append({
            'exp_dir': exp_k,
            'bead_name': bname,
            'diameter_um': d_um,
            'radius_um': rc_um,
            'rc_over_xi': rc_xi,
            'vpar_norm': vp,
            'phi': phi,
            'n_particles': n_p,
            'n_points': n_pts
        })
    df_movie = pd.DataFrame(movie_rows)

    # 2. サイズ別統計量 (per-size statistics with hierarchical bootstrap)
    print(f"Evaluating hierarchical bootstrap (movie -> cargo, N={args.n_boot}) across bead sizes...")
    size_rows = []
    for item in BEADS_INFO:
        bname = item['name']
        sub = df[df['bead_name'] == bname]
        if sub.empty:
            continue

        res = hierarchical_bootstrap_stats(sub, n_boot=args.n_boot, seed=args.seed)
        m_sub = df_movie[df_movie['bead_name'] == bname]
        rc_xi_min = float(m_sub['rc_over_xi'].min())
        rc_xi_max = float(m_sub['rc_over_xi'].max())

        size_rows.append({
            'bead_name': bname,
            'diameter_um': item['diameter_um'],
            'radius_um': item['radius_um'],
            'label': item['label'],
            'color': item['color'],
            'marker': item['marker'],
            'rc_xi_mean': res['rc_xi_mean'],
            'rc_xi_sem': res['rc_xi_sem'],
            'rc_xi_min': rc_xi_min,
            'rc_xi_max': rc_xi_max,
            'vpar_mean': res['vpar_mean'],
            'vpar_sem': res['vpar_sem'],
            'vpar_ci_lo': res['vpar_ci_lo'],
            'vpar_ci_hi': res['vpar_ci_hi'],
            'phi_mean': res['phi_mean'],
            'phi_sem': res['phi_sem'],
            'phi_ci_lo': res['phi_ci_lo'],
            'phi_ci_hi': res['phi_ci_hi'],
            'n_particles': res['n_particles'],
            'n_movies': res['n_movies'],
            'n_points': len(sub)
        })
    df_size = pd.DataFrame(size_rows).sort_values('rc_xi_mean').reset_index(drop=True)

    # CSV 出力
    for out_dir in OUT_DIRS:
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            csv_path = out_dir / "vpar_and_phi_vs_rc_over_xi_stats.csv"
            df_size.to_csv(csv_path, index=False)
            print(f"Saved stats CSV to {csv_path}")
        except Exception as e:
            print(f"Warning: Failed to save CSV to {out_dir}: {e}")

    # ==========================================
    # プロット描画: 第1軸 (左) vs 第2軸 (右)
    # ==========================================
    fig, ax1 = plt.subplots(figsize=(8.2, 5.8))
    ax2 = ax1.twinx()

    # テーマカラー
    c_vpar = '#1f77b4'  # 左軸: 速度 (ブルー系)
    c_phi = '#d62728'   # 右軸: ポーラーオーダー (レッド/クリムゾン系)

    # 基準線
    ax1.axvline(1.0, color='#888888', linestyle='--', linewidth=1.2, alpha=0.75, zorder=1)
    ax1.text(1.04, 0.52, r"$R_c = \xi$", color='#555555', fontsize=11.0, fontweight='bold', va='top')
    ax1.axhline(0.0, color='#999999', linestyle='-', linewidth=0.8, alpha=0.7, zorder=1)

    # 1. 各実験動画の個別点 (散布点: 背景に薄く描画)
    for _, r_m in df_movie.iterrows():
        b_info = next(item for item in BEADS_INFO if item['name'] == r_m['bead_name'])
        # 左軸: vpar_norm
        ax1.scatter(
            r_m['rc_over_xi'], r_m['vpar_norm'],
            marker=b_info['marker'], s=45, color=c_vpar, alpha=0.25,
            edgecolors='none', zorder=2
        )
        # 右軸: phi
        ax2.scatter(
            r_m['rc_over_xi'], r_m['phi'],
            marker=b_info['marker'], s=45, color=c_phi, alpha=0.25,
            edgecolors='none', zorder=2
        )

    # 2. サイズ別平均曲線の描画
    # エラーバーの選定
    if args.error_bar == 'ci95':
        yerr_vpar_lo = np.maximum(0.0, df_size['vpar_mean'] - df_size['vpar_ci_lo'])
        yerr_vpar_hi = np.maximum(0.0, df_size['vpar_ci_hi'] - df_size['vpar_mean'])
        yerr_vpar = [yerr_vpar_lo.values, yerr_vpar_hi.values]

        yerr_phi_lo = np.maximum(0.0, df_size['phi_mean'] - df_size['phi_ci_lo'])
        yerr_phi_hi = np.maximum(0.0, df_size['phi_ci_hi'] - df_size['phi_mean'])
        yerr_phi = [yerr_phi_lo.values, yerr_phi_hi.values]
        err_label = "Hierarchical 95% CI"
    else:
        yerr_vpar = df_size['vpar_sem'].values
        yerr_phi = df_size['phi_sem'].values
        err_label = "Hierarchical SEM"

    # 横軸エラーバー (動画間の R_c/xi 範囲)
    xerr_rc = [
        np.maximum(0.0, df_size['rc_xi_mean'] - df_size['rc_xi_min']).values,
        np.maximum(0.0, df_size['rc_xi_max'] - df_size['rc_xi_mean']).values
    ]

    # 左軸: v_parallel / u_inf(t)
    line1 = ax1.errorbar(
        df_size['rc_xi_mean'], df_size['vpar_mean'],
        xerr=xerr_rc, yerr=yerr_vpar,
        color=c_vpar, linestyle='-', linewidth=2.4,
        marker='o', markersize=8.5, markerfacecolor=c_vpar, markeredgecolor='#111111',
        markeredgewidth=1.1, capsize=3.5, capthick=1.2,
        label=r"Cargo Velocity $v_\parallel / u_\infty(t)$", zorder=4
    )

    # 右軸: Polar Order Phi
    line2 = ax2.errorbar(
        df_size['rc_xi_mean'], df_size['phi_mean'],
        xerr=xerr_rc, yerr=yerr_phi,
        color=c_phi, linestyle='--', linewidth=2.2,
        marker='s', markersize=8.0, markerfacecolor='white', markeredgecolor=c_phi,
        markeredgewidth=1.6, capsize=3.5, capthick=1.2,
        label=r"Polar Order Parameter $\Phi$", zorder=5
    )

    # 各サイズのビーズ径注記
    for _, r in df_size.iterrows():
        # 重なり防止のためサイズごとにオフセット位置を微調整
        d_val = r['diameter_um']
        if d_val == 0.63:
            offset = (-12, 10)
        elif d_val == 1.18:
            offset = (0, 12)
        elif d_val == 3.37:
            offset = (-26, -5)
        elif d_val == 7.24:
            offset = (16, -18)
        elif d_val == 5.0:
            offset = (22, 6)
        elif d_val == 20.0:
            offset = (0, 10)
        else:
            offset = (0, 10)

        ax1.annotate(
            r['label'],
            xy=(r['rc_xi_mean'], r['vpar_mean']),
            xytext=offset, textcoords='offset points',
            ha='center', fontsize=9.2, color='#222222', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.85, edgecolor='#cccccc', linewidth=0.6)
        )

    # 軸設定
    ax1.set_xscale('log')
    ax1.set_xlim(0.045, 4.0)
    ax1.set_xticks([0.05, 0.1, 0.2, 0.5, 1.0, 2.0])
    ax1.get_xaxis().set_major_formatter(ticker.ScalarFormatter())

    ax1.set_ylim(-0.06, 0.58)
    ax2.set_ylim(0.38, 1.05)

    ax1.set_xlabel(r"Scaled Cargo Radius $R_c / \xi$  ($\xi$: Correlation Length)", fontsize=13.5)
    ax1.set_ylabel(r"Normalized Cargo Velocity $v_\parallel / u_\infty(t)$", fontsize=13.5, color=c_vpar)
    ax2.set_ylabel(r"Microtubule Polar Order Parameter $\Phi$", fontsize=13.5, color=c_phi)

    ax1.tick_params(axis='y', labelcolor=c_vpar)
    ax2.tick_params(axis='y', labelcolor=c_phi)

    # 凡例の統合
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='lower left', bbox_to_anchor=(0.03, 0.04), fontsize=10.5, framealpha=0.92)

    ax1.set_title(
        r"Cargo Velocity $v_\parallel/u_\infty$ and Polar Order $\Phi$ vs Scaled Radius $R_c/\xi$"
        + f" ({err_label})",
        fontsize=14.0, fontweight='bold', pad=12
    )

    fig.tight_layout()
    save_fig_all(fig, "vpar_and_phi_vs_rc_over_xi")
    plt.close(fig)
    print("\n[Complete] Figure generated successfully.")

if __name__ == '__main__':
    main()
