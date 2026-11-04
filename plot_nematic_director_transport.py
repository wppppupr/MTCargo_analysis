#!/usr/bin/env python3
"""
plot_nematic_director_transport.py

ネマチック軸（ディレクター）n̂ を基準としたカーゴ粒子輸送特性の解析・作図スクリプト:
    phi_i = angle(Delta r_i, n_hat)
    Q_v   = <cos 2 phi>

- Q_v > 0: ネマチック軸に沿った移動（平行輸送, parallel transport）
- Q_v < 0: ネマチック軸に垂直な移動（垂直輸送, perpendicular transport）
- Q_v = 0: 等方的移動（isotropic）

【2通りのディレクター n_hat】
1. 大域ディレクター n_hat_global:
   動画全体の微小管配向主軸（レーダー図の軸 theta_nem, mt_orientation_per_experiment.csv 準拠）
2. 局所ディレクター n_hat_local:
   粒子周囲の環状領域 (annular region: R_c <= r <= R_c + Delta R) の微小管配向テンソル平均
   （粒子直下の接触・光学フローアーティファクトを完全に排除）

【対照解析とロバスト性】
- 位置決定誤差 (|Delta r| > 3 sigma_loc) による静止ノイズ除去
- 時間刻み Delta t 依存性 Q_v(Delta t) の評価（短時間揺らぎ vs 長時間輸送の分離）
- 同一視野内粒子間の相対変位 (dr_i - dr_j) および重心ドリフト補正によるドリフト対照解析
- Rc / xi に対するスケーリングプロット（平行輸送から垂直輸送へのスイッチングの実証）
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import zarr
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

# ワークスペースおよびNASのパス設定
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR_CANDIDATES = [
    Path("/Volumes/data/Sasaki/MTsingleBeads"),
    Path("/Volumes/data-1/Sasaki/MTsingleBeads"),
    Path("/mnt/NAS-Ebanaru/sasaki/MTsingleBeads"),
    Path("/mnt/NAS-Ebanaru/Sasaki/MTsingleBeads"),
    CURRENT_DIR,
]
ROOT_DIR = next((p for p in ROOT_DIR_CANDIDATES if p.exists()), CURRENT_DIR)

OUT_DIRS = [
    CURRENT_DIR / "figure" / "nematic_transport",
    ROOT_DIR / "figure" / "nematic_transport",
]
CACHE_DIR = CURRENT_DIR / "scratch"

# Tol's Muted パレット準拠のビーズ定義 (unified_notation.md 準拠)
BEADS_INFO = [
    {'name': 'beads06um', 'label': r'0.63 $\mu\mathrm{m}$', 'diameter_um': 0.63, 'radius_um': 0.315,
     'color': '#882255', 'marker': '^'},
    {'name': 'beads1um',  'label': r'1.18 $\mu\mathrm{m}$', 'diameter_um': 1.18, 'radius_um': 0.590,
     'color': '#CC6677', 'marker': 'o'},
    {'name': 'beads3um',  'label': r'3.37 $\mu\mathrm{m}$', 'diameter_um': 3.37, 'radius_um': 1.685,
     'color': '#DDCC77', 'marker': 'd'},
    {'name': 'beads7um',  'label': r'7.24 $\mu\mathrm{m}$', 'diameter_um': 7.24, 'radius_um': 3.620,
     'color': '#117733', 'marker': 'h'},
    {'name': 'beads5um',  'label': r'5.00 $\mu\mathrm{m}$', 'diameter_um': 5.00, 'radius_um': 2.500,
     'color': '#999933', 'marker': 'p'},
    {'name': 'beads20um', 'label': r'20.0 $\mu\mathrm{m}$', 'diameter_um': 20.0, 'radius_um': 10.000,
     'color': '#44AA99', 'marker': 's'},
]

SCALE_UM_PER_PX = 0.11
DT_SEC_PER_FRAME = 4.0
SIGMA_LOC_UM = 0.033  # 推定位置決定誤差 (~33 nm)
DEFAULT_DR_MIN = 3.0 * SIGMA_LOC_UM  # ~0.10 um


def setup_matplotlib():
    """統一論文フォント・スタイル設定"""
    plt.rcParams.update({
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.family': 'sans-serif',
        'mathtext.fontset': 'dejavusans',
        'font.size': 12,
        'axes.labelsize': 13,
        'axes.titlesize': 14,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 10.5,
        'figure.titlesize': 15,
        'axes.linewidth': 1.2,
        'xtick.major.width': 1.2,
        'ytick.major.width': 1.2,
        'xtick.minor.width': 0.8,
        'ytick.minor.width': 0.8,
        'lines.linewidth': 2.0,
        'grid.linewidth': 0.6,
        'grid.alpha': 0.4,
        'grid.linestyle': '--',
    })


def save_figure(fig, base_name: str, out_dirs: List[Path] = OUT_DIRS):
    """PNG と SVG の両形式で各出力ディレクトリに図を保存"""
    for out_dir in out_dirs:
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f"{base_name}.png", dpi=300, bbox_inches='tight')
        fig.savefig(out_dir / f"{base_name}.svg", format='svg', bbox_inches='tight')
    print(f"[Saved] {base_name}.png and .svg")


def load_experiment_metadata(root_dir: Path) -> Tuple[Dict[str, float], Dict[str, dict]]:
    """
    各実験の相関長 xi [um] および大域ネマチック主軸角 theta_nem [rad]・ミラー符号を読み込む。
    """
    # 1. 相関長 xi
    xi_candidates = [
        root_dir / "figure" / "bg_angular_correlation" / "bg_angular_correlation_length_per_experiment.csv",
        CURRENT_DIR / "figure" / "bg_angular_correlation" / "bg_angular_correlation_length_per_experiment.csv",
    ]
    xi_map = {}
    for p in xi_candidates:
        if p.exists():
            df_xi = pd.read_csv(p)
            for _, r in df_xi.iterrows():
                xi_map[str(r['exp_dir']).strip()] = float(r['xi_um'])
            break

    # 2. 大域ネマチック主軸角 theta_nem & mirror sign
    nem_candidates = [
        root_dir / "figure" / "mt_orientation" / "mt_orientation_per_experiment.csv",
        CURRENT_DIR / "figure" / "mt_orientation" / "mt_orientation_per_experiment.csv",
    ]
    nem_map = {}
    for p in nem_candidates:
        if p.exists():
            df_nem = pd.read_csv(p)
            for _, r in df_nem.iterrows():
                exp_path = Path(str(r['exp_dir']).strip())
                exp_key = exp_path.name
                nem_map[exp_key] = {
                    'theta_nem_rad': np.deg2rad(float(r['theta_nem_mean_deg'])),
                    'theta_sign': int(r.get('theta_sign', -1)),
                    'axis_order': float(r.get('theta_nem_axis_order', np.nan)),
                }
            break

    return xi_map, nem_map


def compute_or_load_data(root_dir: Path,
                         recompute: bool = False,
                         annulus_width_um: float = 3.0,
                         lags: List[int] = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]) -> pd.DataFrame:
    """
    全実験・全粒子・全フレームの変位と、大域・局所ディレクターとのなす角 phi を一括抽出しキャッシュする。
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / "nematic_transport_cache.pkl"

    if cache_path.exists() and not recompute:
        print(f"[INFO] Loading cached dataset from {cache_path}")
        return pd.read_pickle(cache_path)

    print("[INFO] Computing nematic transport dataset from raw experiments...")
    xi_map, nem_meta_map = load_experiment_metadata(root_dir)

    all_records = []

    for b_item in BEADS_INFO:
        bname = b_item['name']
        d_um = b_item['diameter_um']
        r_um = b_item['radius_um']

        # 実験ディレクトリの探索
        base = root_dir / bname
        if not base.exists():
            continue
        exp_dirs = [p for p in base.glob("*/*") if p.is_dir() and (p / "beads_tracks.csv").exists()]
        if not exp_dirs:
            exp_dirs = [p for p in base.glob("*") if p.is_dir() and (p / "beads_tracks.csv").exists()]

        print(f"--- Processing {bname} ({len(exp_dirs)} experiments) ---")

        for exp_dir in exp_dirs:
            exp_name = exp_dir.name
            tracks_p = exp_dir / "beads_tracks.csv"
            zarr_p = exp_dir / "MTs_im_theta.zarr"

            if not tracks_p.exists():
                continue

            tdf = pd.read_csv(tracks_p)
            if tdf.empty:
                continue

            # 相関長 xi
            xi_val = xi_map.get(exp_name, np.nan)

            # 大域ネマチック主軸
            n_meta = nem_meta_map.get(exp_name, {'theta_nem_rad': 0.0, 'theta_sign': -1, 'axis_order': np.nan})
            theta_global = n_meta['theta_nem_rad']
            theta_sign = n_meta['theta_sign']

            # MTs_im_theta.zarr の読み込み（局所ディレクター用）
            arr_th = None
            if zarr_p.exists():
                try:
                    z = zarr.open_array(str(zarr_p), mode='r')
                    # ミラー符号補正 (-1) を適用して光学フロー/追跡座標系と一致させる
                    arr_th = np.asarray(z[:], dtype=np.float64) * theta_sign
                except Exception as e:
                    print(f"  Warning: Failed to load {zarr_p}: {e}")

            # 局所ディレクター計算用グリッド設定
            H_orig, W_orig = 2160, 2560
            if arr_th is not None:
                H_grid, W_grid = arr_th.shape[1], arr_th.shape[2]
                sy = H_grid / H_orig
                sx = W_grid / W_orig
                R_in_um = r_um
                R_out_um = r_um + annulus_width_um
                r_in_grid = (R_in_um / SCALE_UM_PER_PX) * sx
                r_out_grid = (R_out_um / SCALE_UM_PER_PX) * sx
            else:
                H_grid = W_grid = sy = sx = 0
                r_in_grid = r_out_grid = 0

            # 局所ディレクターのフレーム・粒子別キャッシュ
            local_director_dict = {}
            if arr_th is not None:
                for _, row in tdf.iterrows():
                    frm = int(row['frame'])
                    pid = int(row['particle'])
                    if frm >= arr_th.shape[0]:
                        continue
                    cx_grid = row['x'] * sx
                    cy_grid = row['y'] * sy
                    gx_min = max(0, int(np.floor(cx_grid - r_out_grid)))
                    gx_max = min(W_grid, int(np.ceil(cx_grid + r_out_grid + 1)))
                    gy_min = max(0, int(np.floor(cy_grid - r_out_grid)))
                    gy_max = min(H_grid, int(np.ceil(cy_grid + r_out_grid + 1)))

                    patch = arr_th[frm, gy_min:gy_max, gx_min:gx_max]
                    yy, xx = np.meshgrid(np.arange(gy_min, gy_max), np.arange(gx_min, gx_max), indexing='ij')
                    dist_um = np.hypot((xx - cx_grid) / sx, (yy - cy_grid) / sy) * SCALE_UM_PER_PX
                    mask = (dist_um >= R_in_um) & (dist_um <= R_out_um)

                    vals = patch[mask]
                    vals = vals[np.isfinite(vals)]
                    if len(vals) >= 2:
                        s2 = np.mean(np.sin(2.0 * vals))
                        c2 = np.mean(np.cos(2.0 * vals))
                        th_loc = 0.5 * np.arctan2(s2, c2)
                        s_loc = np.hypot(c2, s2)
                        local_director_dict[(frm, pid)] = (th_loc, s_loc)
                    else:
                        local_director_dict[(frm, pid)] = (np.nan, np.nan)

            # 軌道変位の計算（各ラグ lag について）
            for pid, grp in tdf.groupby('particle'):
                grp = grp.sort_values('frame')
                fs = grp['frame'].values
                xs = grp['x'].values * SCALE_UM_PER_PX
                ys = grp['y'].values * SCALE_UM_PER_PX
                f_map = {f: (x, y) for f, x, y in zip(fs, xs, ys)}

                for lag in lags:
                    dt_sec = lag * DT_SEC_PER_FRAME
                    for f, x, y in zip(fs, xs, ys):
                        f_next = f + lag
                        if f_next in f_map:
                            x_next, y_next = f_map[f_next]
                            dx = x_next - x
                            dy = y_next - y
                            dr = float(np.hypot(dx, dy))
                            theta_v = float(np.arctan2(dy, dx))

                            # 大域ディレクターとの角度差
                            phi_global = theta_v - theta_global
                            cos2_global = float(np.cos(2.0 * phi_global))

                            # 局所ディレクターとの角度差
                            loc_info = local_director_dict.get((f, pid), (np.nan, np.nan))
                            theta_local, s_local = loc_info
                            if np.isfinite(theta_local):
                                phi_local = theta_v - theta_local
                                cos2_local = float(np.cos(2.0 * phi_local))
                            else:
                                phi_local = np.nan
                                cos2_local = np.nan

                            all_records.append({
                                'bead_name': bname,
                                'diameter_um': d_um,
                                'radius_um': r_um,
                                'exp_dir': exp_name,
                                'xi_um': xi_val,
                                'scaled_radius': r_um / xi_val if np.isfinite(xi_val) and xi_val > 0 else np.nan,
                                'particle': pid,
                                'frame': f,
                                'lag_frames': lag,
                                'dt_sec': dt_sec,
                                'dx_um': dx,
                                'dy_um': dy,
                                'dr_um': dr,
                                'theta_v': theta_v,
                                'theta_global': theta_global,
                                'phi_global': (phi_global + np.pi / 2.0) % np.pi - np.pi / 2.0,  # [-pi/2, pi/2]
                                'cos2_global': cos2_global,
                                'theta_local': theta_local,
                                'phi_local': (phi_local + np.pi / 2.0) % np.pi - np.pi / 2.0 if np.isfinite(phi_local) else np.nan,
                                'cos2_local': cos2_local,
                                's_local': s_local,
                            })

    df = pd.DataFrame(all_records)
    df.to_pickle(cache_path)
    print(f"[INFO] Cached {len(df)} displacement records to {cache_path}")
    return df


def compute_relative_displacements(df: pd.DataFrame, lag_target: int = 1) -> pd.DataFrame:
    """
    同一動画・同一フレームに存在する粒子ペア間の相対変位 (dr_i - dr_j) を抽出し、
    大域並進ドリフトを完全相殺した Q_v_rel を計算する。
    """
    sub = df[df['lag_frames'] == lag_target].copy()
    pair_records = []

    for (exp_name, frm), grp in sub.groupby(['exp_dir', 'frame']):
        if len(grp) < 2:
            continue
        pids = grp['particle'].values
        dxs = grp['dx_um'].values
        dys = grp['dy_um'].values
        bname = grp['bead_name'].iloc[0]
        r_um = grp['radius_um'].iloc[0]
        xi_val = grp['xi_um'].iloc[0]
        sc_r = grp['scaled_radius'].iloc[0]
        th_g = grp['theta_global'].iloc[0]

        for i in range(len(pids)):
            for j in range(i + 1, len(pids)):
                ddx = dxs[i] - dxs[j]
                ddy = dys[i] - dys[j]
                ddr = float(np.hypot(ddx, ddy))
                th_rel = float(np.arctan2(ddy, ddx))
                phi_rel = th_rel - th_g
                c2_rel = float(np.cos(2.0 * phi_rel))

                pair_records.append({
                    'bead_name': bname,
                    'radius_um': r_um,
                    'exp_dir': exp_name,
                    'xi_um': xi_val,
                    'scaled_radius': sc_r,
                    'frame': frm,
                    'p_i': pids[i],
                    'p_j': pids[j],
                    'dr_rel': ddr,
                    'phi_rel': (phi_rel + np.pi / 2.0) % np.pi - np.pi / 2.0,
                    'cos2_rel': c2_rel,
                })

    return pd.DataFrame(pair_records)


def bootstrap_ci(arr: np.ndarray, n_boot: int = 1000, ci: float = 0.95) -> Tuple[float, float, float]:
    """ブートストラップ法による 95% 信頼区間と平均値を算出"""
    clean = arr[np.isfinite(arr)]
    if len(clean) == 0:
        return np.nan, np.nan, np.nan
    mean_val = float(np.mean(clean))
    if len(clean) == 1:
        return mean_val, mean_val, mean_val
    boot_means = np.empty(n_boot, dtype=np.float64)
    n = len(clean)
    for i in range(n_boot):
        sample = np.random.choice(clean, size=n, replace=True)
        boot_means[i] = np.mean(sample)
    alpha = (1.0 - ci) / 2.0
    low = float(np.percentile(boot_means, alpha * 100.0))
    high = float(np.percentile(boot_means, (1.0 - alpha) * 100.0))
    return mean_val, low, high


def plot_qv_vs_scaled_radius(df: pd.DataFrame, lag_target: int = 1, dr_min: float = DEFAULT_DR_MIN):
    """
    【主要結果図】Q_v vs Rc / xi
    小粒子 (Q_v > 0: 平行輸送) から大粒子 (Q_v < 0: 垂直輸送) への切り替わりを明快に示す 2 パネル図。
    - パネル (a): 大域ディレクター（レーダー図の軸）基準
    - パネル (b): 局所ディレクター（環状領域）基準
    """
    sub = df[(df['lag_frames'] == lag_target) & (df['dr_um'] >= dr_min)].copy()

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2), sharey=True)

    titles = [
        r"(a) Global Nematic Director $\hat{\mathbf{n}}_{\mathrm{global}}$ (Radar Axis)",
        r"(b) Local Director $\hat{\mathbf{n}}_{\mathrm{local}}$ (Annulus $R_c \leq r \leq R_c + 3\,\mu\mathrm{m}$)",
    ]
    col_names = ['cos2_global', 'cos2_local']

    summary_rows = []

    for ax, col, title in zip(axes, col_names, titles):
        # ゼロライン（等方性の境界）
        ax.axhline(0, color='gray', linestyle='--', linewidth=1.2, alpha=0.8, zorder=1)

        # 領域注記 (Parallel vs Perpendicular)
        ax.text(0.04, 0.93, r"$\mathbf{Parallel\ Transport}\ (Q_v > 0)$",
                transform=ax.transAxes, color='#117733', fontsize=11, fontweight='bold', va='top')
        ax.text(0.04, 0.07, r"$\mathbf{Perpendicular\ Transport}\ (Q_v < 0)$",
                transform=ax.transAxes, color='#882255', fontsize=11, fontweight='bold', va='bottom')

        # 背景領域の淡い色分け
        ax.axhspan(0, 0.35, color='#117733', alpha=0.04, zorder=0)
        ax.axhspan(-0.35, 0, color='#882255', alpha=0.04, zorder=0)

        for b_item in BEADS_INFO:
            bname = b_item['name']
            lbl = b_item['label']
            color = b_item['color']
            marker = b_item['marker']

            b_sub = sub[sub['bead_name'] == bname].copy()
            if b_sub.empty:
                continue

            # 1. 各実験ごとの散布プロット (個別実験データ)
            for exp_name, e_grp in b_sub.groupby('exp_dir'):
                c2_vals = e_grp[col].values
                clean = c2_vals[np.isfinite(c2_vals)]
                if len(clean) >= 5:
                    exp_sc_r = e_grp['scaled_radius'].iloc[0]
                    exp_qv = np.mean(clean)
                    ax.scatter(exp_sc_r, exp_qv, color=color, marker=marker,
                               alpha=0.35, s=45, edgecolor='none', zorder=2)

            # 2. 条件アンサンブル平均 & ブートストラップ 95% 信頼区間
            c2_all = b_sub[col].values
            mean_sc_r = b_sub['scaled_radius'].mean()
            m_qv, qv_low, qv_high = bootstrap_ci(c2_all)

            yerr_low = m_qv - qv_low
            yerr_high = qv_high - m_qv

            ax.errorbar(mean_sc_r, m_qv,
                        yerr=[[yerr_low], [yerr_high]],
                        fmt=marker, color=color, ecolor=color,
                        markersize=9, capsize=4, elinewidth=1.8, markeredgecolor='black',
                        markeredgewidth=0.8, label=lbl, zorder=4)

            summary_rows.append({
                'director_type': col.replace('cos2_', ''),
                'bead_name': bname,
                'diameter_um': b_item['diameter_um'],
                'radius_um': b_item['radius_um'],
                'scaled_radius_mean': mean_sc_r,
                'N_displacements': len(c2_all[np.isfinite(c2_all)]),
                'Q_v_mean': m_qv,
                'Q_v_ci_low': qv_low,
                'Q_v_ci_high': qv_high,
                'SEM': np.std(c2_all[np.isfinite(c2_all)]) / np.sqrt(len(c2_all[np.isfinite(c2_all)])),
            })

        ax.set_xscale('log')
        ax.set_xlim(0.04, 3.0)
        ax.set_ylim(-0.25, 0.25)
        ax.set_xlabel(r"Scaled Cargo Radius $R_c / \xi$", fontsize=13)
        ax.set_title(title, fontsize=12.5, pad=10)
        ax.grid(True, which='both', linestyle='--', alpha=0.4)

    axes[0].set_ylabel(r"Nematic Transport Order $Q_v \equiv \langle \cos 2\phi \rangle$", fontsize=13)

    # 凡例の追加 (Tol's Muted)
    handles, labels = axes[0].get_legend_handles_labels()
    axes[1].legend(handles, labels, title="Cargo Diameter", loc='upper right',
                   frameon=True, framealpha=0.92, fontsize=10.5)

    fig.suptitle(rf"Cargo Transport Crossover: Parallel to Perpendicular Transport ($\Delta t = {lag_target * DT_SEC_PER_FRAME:.0f}\,\mathrm{{s}}$)",
                 fontsize=14.5, fontweight='bold', y=0.99)
    fig.tight_layout()
    save_figure(fig, "nematic_transport_Qv_vs_scaled_radius")
    plt.close(fig)

    # CSV 出力
    df_sum = pd.DataFrame(summary_rows)
    for out_dir in OUT_DIRS:
        out_dir.mkdir(parents=True, exist_ok=True)
        df_sum.to_csv(out_dir / "nematic_transport_summary.csv", index=False)
    print("[Saved] nematic_transport_summary.csv")


def plot_qv_vs_radius(df: pd.DataFrame, lag_target: int = 1, dr_min: float = DEFAULT_DR_MIN):
    """Q_v vs 物理半径 Rc [um]（線形横軸版）"""
    sub = df[(df['lag_frames'] == lag_target) & (df['dr_um'] >= dr_min)].copy()

    fig, ax = plt.subplots(figsize=(7.5, 5.0))

    ax.axhline(0, color='gray', linestyle='--', linewidth=1.2, alpha=0.8, zorder=1)
    ax.axhspan(0, 0.35, color='#117733', alpha=0.04, zorder=0)
    ax.axhspan(-0.35, 0, color='#882255', alpha=0.04, zorder=0)

    for b_item in BEADS_INFO:
        bname = b_item['name']
        lbl = b_item['label']
        color = b_item['color']
        marker = b_item['marker']
        rc = b_item['radius_um']

        b_sub = sub[sub['bead_name'] == bname].copy()
        if b_sub.empty:
            continue

        c2_all = b_sub['cos2_global'].values
        m_qv, qv_low, qv_high = bootstrap_ci(c2_all)

        ax.errorbar(rc, m_qv,
                    yerr=[[m_qv - qv_low], [qv_high - m_qv]],
                    fmt=marker, color=color, ecolor=color,
                    markersize=9, capsize=4, elinewidth=1.8, markeredgecolor='black',
                    markeredgewidth=0.8, label=lbl, zorder=4)

    ax.set_xscale('log')
    ax.set_xlim(0.2, 15.0)
    ax.set_ylim(-0.25, 0.25)
    ax.set_xlabel(r"Cargo Radius $R_c\ [\mu\mathrm{m}]$", fontsize=13)
    ax.set_ylabel(r"Nematic Transport Order $Q_v \equiv \langle \cos 2\phi \rangle$", fontsize=13)
    ax.set_title(rf"Transport Order vs Cargo Radius ($\Delta t = {lag_target * DT_SEC_PER_FRAME:.0f}\,\mathrm{{s}}$)", fontsize=13.5, pad=10)
    ax.legend(title="Cargo Diameter", loc='upper right', frameon=True, framealpha=0.92)
    ax.grid(True, which='both', linestyle='--', alpha=0.4)

    fig.tight_layout()
    save_figure(fig, "nematic_transport_Qv_vs_radius")
    plt.close(fig)


def plot_qv_vs_deltat(df: pd.DataFrame, dr_min: float = DEFAULT_DR_MIN):
    """
    【時間刻み依存性】Q_v(Delta t) vs Delta t [s]
    短時間の揺らぎと長時間の輸送を分離し、サイズ別の持続性を可視化する。
    """
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2), sharey=True)
    col_names = ['cos2_global', 'cos2_local']
    titles = [
        r"(a) Global Nematic Director $\hat{\mathbf{n}}_{\mathrm{global}}$",
        r"(b) Local Director $\hat{\mathbf{n}}_{\mathrm{local}}$",
    ]

    for ax, col, title in zip(axes, col_names, titles):
        ax.axhline(0, color='gray', linestyle='--', linewidth=1.2, alpha=0.8, zorder=1)
        ax.axhspan(0, 0.35, color='#117733', alpha=0.03, zorder=0)
        ax.axhspan(-0.35, 0, color='#882255', alpha=0.03, zorder=0)

        for b_item in BEADS_INFO:
            bname = b_item['name']
            lbl = b_item['label']
            color = b_item['color']
            marker = b_item['marker']

            b_df = df[(df['bead_name'] == bname) & (df['dr_um'] >= dr_min)].copy()
            if b_df.empty:
                continue

            dt_vals = []
            qv_vals = []
            qv_lows = []
            qv_highs = []

            for lag, grp in b_df.groupby('lag_frames'):
                dt = lag * DT_SEC_PER_FRAME
                c2_vals = grp[col].values
                clean = c2_vals[np.isfinite(c2_vals)]
                if len(clean) >= 10:
                    m, low, high = bootstrap_ci(clean)
                    dt_vals.append(dt)
                    qv_vals.append(m)
                    qv_lows.append(low)
                    qv_highs.append(high)

            if dt_vals:
                dt_arr = np.array(dt_vals)
                qv_arr = np.array(qv_vals)
                y_err_l = qv_arr - np.array(qv_lows)
                y_err_h = np.array(qv_highs) - qv_arr

                ax.errorbar(dt_arr, qv_arr, yerr=[y_err_l, y_err_h],
                            fmt=f"-{marker}", color=color, ecolor=color,
                            markersize=7, capsize=3, elinewidth=1.5, linewidth=1.8,
                            markeredgecolor='black', markeredgewidth=0.6,
                            label=lbl, zorder=3)

        ax.set_xscale('log')
        ax.set_xlim(3.0, 300.0)
        ax.set_ylim(-0.25, 0.25)
        ax.set_xlabel(r"Lag Time $\Delta t\ [\mathrm{s}]$", fontsize=13)
        ax.set_title(title, fontsize=12.5, pad=10)
        ax.grid(True, which='both', linestyle='--', alpha=0.4)

    axes[0].set_ylabel(r"Nematic Transport Order $Q_v(\Delta t) \equiv \langle \cos 2\phi \rangle$", fontsize=13)
    axes[1].legend(title="Cargo Diameter", loc='upper right', frameon=True, framealpha=0.92, fontsize=10)

    fig.suptitle(r"Timescale Dependence: Separation of Short-time Fluctuations and Long-time Transport",
                 fontsize=14.0, fontweight='bold', y=0.99)
    fig.tight_layout()
    save_figure(fig, "nematic_transport_Qv_vs_deltat")
    plt.close(fig)


def plot_drift_contrast(df: pd.DataFrame, df_rel: pd.DataFrame, lag_target: int = 1, dr_min: float = DEFAULT_DR_MIN):
    """
    【ドリフト対照解析】
    通常の変位 vs 粒子間相対変位 (dr_i - dr_j) の比較。
    大域並進ドリフトを除去しても、平行輸送 (Q_v > 0) の結論が堅牢に成立することを実証する。
    """
    sub_ind = df[(df['lag_frames'] == lag_target) & (df['dr_um'] >= dr_min)].copy()
    sub_rel = df_rel[df_rel['dr_rel'] >= dr_min * np.sqrt(2)].copy()

    fig, ax = plt.subplots(figsize=(8.0, 5.2))

    ax.axhline(0, color='gray', linestyle='--', linewidth=1.2, alpha=0.8, zorder=1)
    ax.axhspan(0, 0.35, color='#117733', alpha=0.03, zorder=0)
    ax.axhspan(-0.35, 0, color='#882255', alpha=0.03, zorder=0)

    # 複数粒子が存在する条件 (0.63, 1.18, 3.37, 7.24, 5.00 um)
    for b_item in BEADS_INFO:
        bname = b_item['name']
        lbl = b_item['label']
        color = b_item['color']
        marker = b_item['marker']

        # 通常の変位
        b_ind = sub_ind[sub_ind['bead_name'] == bname]
        # 相対変位
        b_rel = sub_rel[sub_rel['bead_name'] == bname]

        if not b_ind.empty:
            sc_r = b_ind['scaled_radius'].mean()
            m_ind, l_ind, h_ind = bootstrap_ci(b_ind['cos2_global'].values)
            # 個別変位（塗りつぶしマーカー）
            ax.errorbar(sc_r * 0.96, m_ind, yerr=[[m_ind - l_ind], [h_ind - m_ind]],
                        fmt=marker, color=color, markersize=8.5, capsize=3.5, elinewidth=1.6,
                        markeredgecolor='black', markeredgewidth=0.7, zorder=4)

        if not b_rel.empty:
            sc_r = b_rel['scaled_radius'].mean()
            m_rel, l_rel, h_rel = bootstrap_ci(b_rel['cos2_rel'].values)
            # 相対変位（中空マーカー）
            ax.errorbar(sc_r * 1.04, m_rel, yerr=[[m_rel - l_rel], [h_rel - m_rel]],
                        fmt=marker, color=color, mfc='white', mew=2.0,
                        markersize=8.5, capsize=3.5, elinewidth=1.6, zorder=4)

    ax.set_xscale('log')
    ax.set_xlim(0.04, 3.0)
    ax.set_ylim(-0.25, 0.25)
    ax.set_xlabel(r"Scaled Cargo Radius $R_c / \xi$", fontsize=13)
    ax.set_ylabel(r"Nematic Order $\langle \cos 2\phi \rangle$", fontsize=13)
    ax.set_title(rf"Robustness against Drift: Individual vs Pairwise Relative Displacements ($\Delta t = {lag_target * DT_SEC_PER_FRAME:.0f}\,\mathrm{{s}}$)",
                 fontsize=12.5, pad=10)
    ax.grid(True, which='both', linestyle='--', alpha=0.4)

    # カスタム凡例
    legend_elements = [
        Line2D([0], [0], marker='o', color='black', label=r'Individual Displacements $\Delta \mathbf{r}_i$',
               markerfacecolor='black', markersize=8),
        Line2D([0], [0], marker='o', color='black', label=r'Relative Displacements $\Delta \mathbf{r}_i - \Delta \mathbf{r}_j$ (Drift-free)',
               markerfacecolor='white', markeredgewidth=2.0, markersize=8),
    ]
    for b in BEADS_INFO:
        legend_elements.append(Line2D([0], [0], marker=b['marker'], color=b['color'],
                                      label=b['label'], linestyle='None', markersize=7))

    ax.legend(handles=legend_elements, loc='upper right', frameon=True, framealpha=0.92, fontsize=9.5)
    fig.tight_layout()
    save_figure(fig, "nematic_transport_drift_contrast")
    plt.close(fig)


def plot_angle_distributions(df: pd.DataFrame, lag_target: int = 1, dr_min: float = DEFAULT_DR_MIN):
    """
    【角度分布】phi_i = angle(Delta r, n_hat) の確率密度分布 P(phi)
    - 小粒子 (0.63, 1.18 um): phi = 0, pi（軸方向）にピーク（平行輸送）
    - 大粒子 (20 um): phi = +-pi/2（垂直方向）にピーク（垂直輸送）
    """
    sub = df[(df['lag_frames'] == lag_target) & (df['dr_um'] >= dr_min)].copy()

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.5), sharex=True, sharey=True)
    axes = axes.flatten()

    bins = np.linspace(-np.pi / 2.0, np.pi / 2.0, 25)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    deg_centers = np.rad2deg(bin_centers)

    for i, b_item in enumerate(BEADS_INFO):
        ax = axes[i]
        bname = b_item['name']
        lbl = b_item['label']
        color = b_item['color']
        marker = b_item['marker']

        b_sub = sub[sub['bead_name'] == bname]
        phi_g = b_sub['phi_global'].dropna().values

        # 等方分布の一様密度ライン
        ax.axhline(1.0 / np.pi, color='gray', linestyle='--', linewidth=1.2, label='Isotropic')

        if len(phi_g) > 0:
            hist, _ = np.histogram(phi_g, bins=bins, density=True)
            ax.plot(deg_centers, hist, f"-{marker}", color=color, linewidth=2.0,
                    markersize=6, label=rf"{lbl} ($N={len(phi_g)}$)")
            ax.fill_between(deg_centers, hist, 1.0 / np.pi, color=color, alpha=0.15)

        ax.set_title(rf"{lbl} ($R_c/\xi \approx {b_sub['scaled_radius'].mean():.2f}$)", fontsize=12, pad=6)
        ax.set_xlim(-90, 90)
        ax.set_xticks([-90, -45, 0, 45, 90])
        ax.set_xticklabels([r'$-90^\circ$', r'$-45^\circ$', r'$0^\circ$', r'$45^\circ$', r'$90^\circ$'])
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend(loc='upper right', fontsize=9.5)

        if i % 3 == 0:
            ax.set_ylabel(r"Probability Density $P(\phi)$", fontsize=11.5)
        if i >= 3:
            ax.set_xlabel(r"Angle to Director $\phi = \angle(\Delta\mathbf{r}, \hat{\mathbf{n}})\ [^\circ]$", fontsize=11.5)

    fig.suptitle(rf"Angular Distribution Relative to Nematic Axis ($\Delta t = {lag_target * DT_SEC_PER_FRAME:.0f}\,\mathrm{{s}}$): Parallel ($0^\circ$) vs Perpendicular ($\pm 90^\circ$)",
                 fontsize=13.5, fontweight='bold', y=0.99)
    fig.tight_layout()
    save_figure(fig, "nematic_transport_angle_distribution")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Analyze cargo transport relative to nematic director.")
    parser.add_argument("--recompute", action="store_true", help="Force recomputation of dataset cache.")
    parser.add_argument("--dr_min", type=float, default=DEFAULT_DR_MIN, help="Minimum displacement threshold (um).")
    parser.add_argument("--lag", type=int, default=1, help="Lag frame for main Q_v plot (default: 1 = 4s).")
    args = parser.parse_args()

    setup_matplotlib()

    print("=" * 78)
    print(" NEMATIC DIRECTOR CARGO TRANSPORT ANALYSIS")
    print("=" * 78)
    print(f"Data Root         : {ROOT_DIR}")
    print(f"Threshold |dr|    : {args.dr_min:.3f} um (>= 3 sigma_loc)")
    print(f"Main Lag Frame    : {args.lag} ({args.lag * DT_SEC_PER_FRAME:.1f} s)")
    print("=" * 78)

    # 1. データの抽出 & キャッシュロード
    df = compute_or_load_data(ROOT_DIR, recompute=args.recompute)

    # 2. ドリフト対照用の相対変位データ抽出
    print("[INFO] Computing pairwise relative displacements for drift contrast...")
    df_rel = compute_relative_displacements(df, lag_target=args.lag)

    # 3. 作図
    print("[INFO] Generating publication figures...")
    plot_qv_vs_scaled_radius(df, lag_target=args.lag, dr_min=args.dr_min)
    plot_qv_vs_radius(df, lag_target=args.lag, dr_min=args.dr_min)
    plot_qv_vs_deltat(df, dr_min=args.dr_min)
    plot_drift_contrast(df, df_rel, lag_target=args.lag, dr_min=args.dr_min)
    plot_angle_distributions(df, lag_target=args.lag, dr_min=args.dr_min)

    print("=" * 78)
    print(" ANALYSIS COMPLETED SUCCESSFULLY")
    print(f" Figures saved to: {OUT_DIRS[0]}")
    print("=" * 78)


if __name__ == "__main__":
    main()

