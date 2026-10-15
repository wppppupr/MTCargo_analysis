#!/usr/bin/env python3
"""
plot_scaling_master_curve.py

微粒子貨物（Cargo Beads）の無次元サイズ x = R_c / xi に対する輸送速度 <v_c> のスケーリンググラフを作成するスクリプト。
理論マスターカーブ:
    <v_c> = v0 * (2 / x^2) * [1 - (1 + x) * exp(-x)]   (x = R_c / xi)
を重ねてプロットします。

できるだけ点数が多くなるように、
1. 個別セグメント速度（Sub-trajectory sliding windows: 7000点以上の高密度データクラウド）
2. 個別粒子軌跡平均速度（Individual track averages: 97点）
3. 実験（視野/ムービー）平均速度（Per-experiment averages: 22点）
4. ビーズサイズごとのアンサンブル代表値・SEM誤差棒（Ensemble mean +/- SEM）
を階層的に美しくプロットします。

出力ファイル:
- figure/scaling/scaling_master_curve_2panel.png / .svg (Linear実速度 & Normalized Data Collapse 2パネル)
- figure/scaling/scaling_master_curve_linear.png / .svg (Linear実速度 単体図)
- figure/scaling/scaling_master_curve_normalized.png / .svg (Normalized Data Collapse 単体図)
- figure/scaling/scaling_master_curve_loglog.png / .svg (Log-Log スケーリング確認図)
- figure/scaling/scaling_data_points_summary.csv (全データ点集計サマリーCSV)
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# プロジェクトルート
CURRENT_DIR = Path(__file__).parent.resolve()
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

# ビーズ基本情報
BEADS_INFO = [
    {"name": "beads06um", "diameter_um": 0.63, "radius_um": 0.315, "marker": "^", "color": "#1f77b4", "label": r"$2R_c = 0.63\,\mu\mathrm{m}\ (R_c = 0.32\,\mu\mathrm{m})$"},
    {"name": "beads1um",  "diameter_um": 1.18, "radius_um": 0.590, "marker": "o", "color": "#ff7f0e", "label": r"$2R_c = 1.18\,\mu\mathrm{m}\ (R_c = 0.59\,\mu\mathrm{m})$"},
    {"name": "beads3um",  "diameter_um": 3.37, "radius_um": 1.685, "marker": "d", "color": "#2ca02c", "label": r"$2R_c = 3.37\,\mu\mathrm{m}\ (R_c = 1.69\,\mu\mathrm{m})$"},
    {"name": "beads5um",  "diameter_um": 5.00, "radius_um": 2.500, "marker": "p", "color": "#d62728", "label": r"$2R_c = 5.00\,\mu\mathrm{m}\ (R_c = 2.50\,\mu\mathrm{m})$"},
    {"name": "beads7um",  "diameter_um": 7.24, "radius_um": 3.620, "marker": "h", "color": "#9467bd", "label": r"$2R_c = 7.24\,\mu\mathrm{m}\ (R_c = 3.62\,\mu\mathrm{m})$"},
    {"name": "beads20um", "diameter_um": 20.0, "radius_um": 10.00, "marker": "s", "color": "#8c564b", "label": r"$2R_c = 20.0\,\mu\mathrm{m}\ (R_c = 10.0\,\mu\mathrm{m})$"},
]

POSSIBLE_ROOTS = [
    Path('/Volumes/data/Sasaki/MTsingleBeads'),
    Path('/Volumes/data/sasaki/MTsingleBeads'),
    Path('/Volumes/data-1/Sasaki/MTsingleBeads'),
    Path('/Volumes/data-1/sasaki/MTsingleBeads'),
    Path('/mnt/NAS-Ebanaru/Sasaki/MTsingleBeads'),
    Path('/mnt/NAS-Ebanaru/sasaki/MTsingleBeads'),
]


def find_default_root() -> Path:
    for r in POSSIBLE_ROOTS:
        if r.exists():
            for b in ['beads1um', 'beads06um', 'beads3um']:
                if (r / b).exists():
                    return r
    for r in POSSIBLE_ROOTS:
        if r.exists():
            return r
    return POSSIBLE_ROOTS[0]


def apply_custom_style():
    style_path = CURRENT_DIR / 'libs' / 'my_style.mplstyle'
    if style_path.exists():
        try:
            plt.style.use(str(style_path))
        except Exception:
            pass
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 13,
        'axes.titlesize': 14,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
        'lines.linewidth': 2.0,
        'lines.markersize': 7,
    })


def master_function(x: np.ndarray) -> np.ndarray:
    """
    理論普遍関数 g(x) = (2 / x^2) * [1 - (1 + x) * exp(-x)]
    x -> 0 の極限では Taylor 展開を用いて数値安定性を確保:
    g(x) = 1 - (2/3)x + (1/4)x^2 - (1/15)x^3 + ...
    """
    x = np.asarray(x, dtype=float)
    res = np.zeros_like(x)
    
    small = x < 1e-4
    large = ~small
    
    if np.any(small):
        xs = x[small]
        res[small] = 1.0 - (2.0 / 3.0) * xs + 0.25 * (xs**2) - (xs**3) / 15.0 + (xs**4) / 72.0
    
    if np.any(large):
        xl = x[large]
        res[large] = (2.0 / (xl**2)) * (1.0 - (1.0 + xl) * np.exp(-xl))
    
    return res


def theoretical_velocity(Rc: np.ndarray, v0: float, xi: float) -> np.ndarray:
    """<v_c> = v0 * g(Rc / xi)"""
    x = Rc / xi
    return v0 * master_function(x)


def load_all_velocity_data(
    root_dir: Path,
    scale: float = 0.11,
    frame_interval: float = 4.0,
    window_frames: int = 10,
    xi_default: float = 2.78
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    全軌跡データを読み込み、
    1. セグメント単位 (window_df)
    2. トラック単位 (track_df)
    3. 実験（ムービー）単位 (exp_df)
    4. ビーズサイズごとのアンサンブル集計 (ensemble_df)
    を作成する。
    """
    track_records = []
    seg_records = []
    
    # 粒子径ごとの外れ値カットオフ閾値 (cargo_velocity_analysis.py と整合)
    outlier_cutoffs = {
        'beads06um': 0.36,
        'beads1um': 0.33,
        'beads3um': 0.32,
        'beads5um': 0.36,
        'beads7um': 0.46,
        'beads20um': 0.24,
    }
    
    for b in BEADS_INFO:
        b_name = b["name"]
        d_um = b["diameter_um"]
        r_c = b["radius_um"]
        b_dir = root_dir / b_name
        cutoff_v = outlier_cutoffs.get(b_name, 0.45)
        if not b_dir.exists():
            continue
        
        date_dirs = sorted([d for d in b_dir.iterdir() if d.is_dir() and not d.name.startswith('.')])
        for date_dir in date_dirs:
            exp_dirs = sorted([d for d in date_dir.iterdir() if d.is_dir() and not d.name.startswith('.')])
            for exp_dir in exp_dirs:
                t_csv = exp_dir / 'beads_tracks.csv'
                if not t_csv.exists():
                    continue
                
                try:
                    df = pd.read_csv(t_csv)
                except Exception:
                    continue
                
                exp_id = f"{date_dir.name}/{exp_dir.name}"
                
                for pid, g in df.groupby('particle'):
                    if len(g) < 2:
                        continue
                    g = g.sort_values('frame')
                    x_um = g['x'].values * scale
                    y_um = g['y'].values * scale
                    frames = g['frame'].values * frame_interval
                    
                    dx = np.diff(x_um)
                    dy = np.diff(y_um)
                    dt = np.diff(frames)
                    valid = dt > 0
                    if not np.any(valid):
                        continue
                    
                    v_inst = np.sqrt(dx[valid]**2 + dy[valid]**2) / dt[valid]
                    v_inst = v_inst[np.isfinite(v_inst)]
                    # 瞬時速度の外れ値除去
                    v_inst_filtered = v_inst[v_inst <= cutoff_v]
                    if len(v_inst_filtered) == 0:
                        continue
                    
                    v_track_mean = float(np.mean(v_inst_filtered))
                    v_track_median = float(np.median(v_inst_filtered))
                    v_track_std = float(np.std(v_inst_filtered, ddof=1)) if len(v_inst_filtered) > 1 else 0.0
                    v_track_sem = float(v_track_std / np.sqrt(len(v_inst_filtered)))
                    
                    track_records.append({
                        "bead_name": b_name,
                        "diameter_um": d_um,
                        "radius_um": r_c,
                        "exp_id": exp_id,
                        "particle_id": pid,
                        "n_frames": len(v_inst_filtered),
                        "v_mean_um_s": v_track_mean,
                        "v_median_um_s": v_track_median,
                        "v_std_um_s": v_track_std,
                        "v_sem_um_s": v_track_sem,
                        "x_scaling": r_c / xi_default,
                    })
                    
                    step = max(1, window_frames // 2)
                    for start_i in range(0, len(v_inst_filtered), step):
                        seg = v_inst_filtered[start_i : start_i + window_frames]
                        if len(seg) >= 3:
                            seg_records.append({
                                "bead_name": b_name,
                                "diameter_um": d_um,
                                "radius_um": r_c,
                                "exp_id": exp_id,
                                "particle_id": pid,
                                "seg_len": len(seg),
                                "v_seg_mean_um_s": float(np.mean(seg)),
                                "v_seg_median_um_s": float(np.median(seg)),
                                "x_scaling": r_c / xi_default,
                            })

    df_tracks = pd.DataFrame(track_records)
    df_segs = pd.DataFrame(seg_records)
    
    exp_summary = []
    if not df_tracks.empty:
        for (b_name, exp_id), g in df_tracks.groupby(['bead_name', 'exp_id']):
            r_c = g['radius_um'].iloc[0]
            d_um = g['diameter_um'].iloc[0]
            exp_summary.append({
                "bead_name": b_name,
                "exp_id": exp_id,
                "diameter_um": d_um,
                "radius_um": r_c,
                "n_tracks": len(g),
                "v_mean_um_s": float(g['v_mean_um_s'].mean()),
                "v_sem_um_s": float(g['v_mean_um_s'].std(ddof=1) / np.sqrt(len(g))) if len(g) > 1 else 0.0,
                "v_median_um_s": float(g['v_median_um_s'].median()),
                "x_scaling": r_c / xi_default,
            })
    df_exp = pd.DataFrame(exp_summary)
    
    # 論文基準の HMM Run 速度 (0.63, 1.18, 3.37 um)
    # 5um以上はRun状態抑制のため、外れ値除去後平均速度を使用
    run_values = {
        'beads06um': {'v_run': 0.17598, 'v_sem': 0.0075, 'state_type': 'Run'},
        'beads1um':  {'v_run': 0.13217, 'v_sem': 0.0042, 'state_type': 'Run'},
        'beads3um':  {'v_run': 0.11771, 'v_sem': 0.0051, 'state_type': 'Run'},
        'beads5um':  {'v_run': np.nan,  'v_sem': np.nan,  'state_type': 'Mean'},
        'beads7um':  {'v_run': np.nan,  'v_sem': np.nan,  'state_type': 'Mean'},
        'beads20um': {'v_run': np.nan,  'v_sem': np.nan,  'state_type': 'Mean'},
    }
    
    ensemble_summary = []
    for b in BEADS_INFO:
        b_name = b["name"]
        d_um = b["diameter_um"]
        r_c = b["radius_um"]
        
        g_t = df_tracks[df_tracks['bead_name'] == b_name] if not df_tracks.empty else pd.DataFrame()
        g_s = df_segs[df_segs['bead_name'] == b_name] if not df_segs.empty else pd.DataFrame()
        
        n_tracks = len(g_t)
        n_segs = len(g_s)
        
        mean_v = float(g_t['v_mean_um_s'].mean()) if n_tracks > 0 else np.nan
        std_v = float(g_t['v_mean_um_s'].std(ddof=1)) if n_tracks > 1 else 0.0
        sem_v = float(std_v / np.sqrt(n_tracks)) if n_tracks > 0 else 0.0
        median_v = float(g_t['v_mean_um_s'].median()) if n_tracks > 0 else np.nan
        
        rv = run_values.get(b_name, {})
        v_run_g = rv.get('v_run', np.nan)
        v_run_sem = rv.get('v_sem', np.nan)
        st_type = rv.get('state_type', 'Mean')
        
        ensemble_summary.append({
            "bead_name": b_name,
            "diameter_um": d_um,
            "radius_um": r_c,
            "x_scaling": r_c / xi_default,
            "n_tracks": n_tracks,
            "n_segs": n_segs,
            "v_mean_um_s": mean_v,
            "v_sem_um_s": sem_v,
            "v_std_um_s": std_v,
            "v_median_um_s": median_v,
            "v_run_geom_um_s": v_run_g,
            "v_run_geom_sem_um_s": v_run_sem,
            "state_type": st_type,
        })
    
    df_ensemble = pd.DataFrame(ensemble_summary)
    return df_segs, df_tracks, df_exp, df_ensemble


def fit_master_curve(
    df_ensemble: pd.DataFrame,
    xi_fixed: float = 2.78,
) -> Tuple[float, float, float, float, float, float]:
    """
    マスターカーブ <v_c> = v0 * g(Rc / xi) に対するフィッティング (Run速度 3点)
    """
    valid_run = df_ensemble['v_run_geom_um_s'].notna()
    rc_run = df_ensemble.loc[valid_run, 'radius_um'].values
    v_run = df_ensemble.loc[valid_run, 'v_run_geom_um_s'].values
    sem_run = df_ensemble.loc[valid_run, 'v_run_geom_sem_um_s'].values
    
    # 1. 固定 xi = 2.78 um での v0 最適化
    def model_fixed(rc, v0):
        return theoretical_velocity(rc, v0, xi_fixed)
    
    popt_fix, _ = curve_fit(model_fixed, rc_run, v_run, p0=[0.207], sigma=sem_run)
    v0_fixed_fit = float(popt_fix[0])
    
    y_pred_fix = model_fixed(rc_run, v0_fixed_fit)
    ss_tot = np.sum((v_run - np.mean(v_run))**2)
    ss_res_fix = np.sum((v_run - y_pred_fix)**2)
    r2_fix = float(1.0 - ss_res_fix / ss_tot) if ss_tot > 0 else 1.0
    
    # 2. v0, xi 両方の自由フィッティング
    def model_both(rc, v0, xi):
        return theoretical_velocity(rc, v0, xi)
    
    try:
        popt_both, _ = curve_fit(model_both, rc_run, v_run, p0=[0.207, 2.78], sigma=sem_run, bounds=([0.05, 0.5], [0.5, 8.0]))
        v0_both_fit, xi_both_fit = float(popt_both[0]), float(popt_both[1])
        y_pred_both = model_both(rc_run, v0_both_fit, xi_both_fit)
        ss_res_both = np.sum((v_run - y_pred_both)**2)
        r2_both = float(1.0 - ss_res_both / ss_tot) if ss_tot > 0 else 1.0
    except Exception:
        v0_both_fit, xi_both_fit, r2_both = v0_fixed_fit, xi_fixed, r2_fix
    
    return v0_fixed_fit, xi_fixed, r2_fix, v0_both_fit, xi_both_fit, r2_both


def plot_scaling_master_curve(
    df_segs: pd.DataFrame,
    df_tracks: pd.DataFrame,
    df_exp: pd.DataFrame,
    df_ensemble: pd.DataFrame,
    out_dir: Path,
    xi_val: float = 2.78
):
    """
    高品質なスケーリンググラフを作成・保存する。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    apply_custom_style()
    
    # マスターカーブのフィッティング
    v0_fix, xi_fix, r2_fix, v0_free, xi_free, r2_free = fit_master_curve(df_ensemble, xi_fixed=xi_val)
    
    v0_plot = v0_fix  # ~0.19 μm/s
    xi_plot = xi_val  # 2.78 μm
    
    print(f"\n================ マスターカーブ パラメータ ================")
    print(f"  [固定 xi = {xi_fix:.2f} μm] v0 = {v0_fix:.4f} μm/s (Run R^2 = {r2_fix:.4f})")
    print(f"  [自由フィッティング]      v0 = {v0_free:.4f} μm/s, xi = {xi_free:.4f} μm (Run R^2 = {r2_free:.4f})")
    print(f"==========================================================\n")
    
    # 理論曲線の計算用グリッド
    x_theory = np.linspace(0.005, 4.0, 500)
    g_theory = master_function(x_theory)
    v_theory_fix = v0_plot * g_theory
    v_approx_exp = v0_plot * np.exp(-2.0 * x_theory / 3.0)
    
    # =========================================================================
    # 1. 2パネル図 (Panel A: 実速度 v vs x, Panel B: 無次元化 v/v0 vs x)
    # =========================================================================
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(17, 7.5))
    
    # ------------------ Panel A: Real Velocity <v_c> vs x ------------------
    # 1. 個別セグメント点 (Sliding windows / segments: 7000点以上の高密度背景)
    if not df_segs.empty:
        for b in BEADS_INFO:
            b_name = b['name']
            sub_s = df_segs[df_segs['bead_name'] == b_name]
            if not sub_s.empty:
                x_pts = sub_s['radius_um'].values / xi_plot
                y_pts = sub_s['v_seg_mean_um_s'].values
                jitter = np.random.normal(0, 0.015 * (x_pts[0] + 0.05), size=len(x_pts))
                ax_a.scatter(
                    x_pts + jitter, y_pts,
                    color=b['color'],
                    alpha=0.10,
                    s=12,
                    edgecolors='none',
                    zorder=1
                )
    
    # 2. 個別トラック点 (Individual Tracks: 97点)
    if not df_tracks.empty:
        for b in BEADS_INFO:
            b_name = b['name']
            sub_t = df_tracks[df_tracks['bead_name'] == b_name]
            if not sub_t.empty:
                x_pts = sub_t['radius_um'].values / xi_plot
                y_pts = sub_t['v_mean_um_s'].values
                ax_a.scatter(
                    x_pts, y_pts,
                    color=b['color'],
                    marker=b['marker'],
                    s=55,
                    alpha=0.60,
                    edgecolors='black',
                    linewidths=0.6,
                    zorder=3
                )
    
    # 3. 実験（視野）平均点 (Per-Experiment: 22点)
    if not df_exp.empty:
        for b in BEADS_INFO:
            b_name = b['name']
            sub_e = df_exp[df_exp['bead_name'] == b_name]
            if not sub_e.empty:
                x_pts = sub_e['radius_um'].values / xi_plot
                y_pts = sub_e['v_mean_um_s'].values
                ax_a.scatter(
                    x_pts, y_pts,
                    facecolors='none',
                    edgecolors=b['color'],
                    marker=b['marker'],
                    s=120,
                    linewidths=1.8,
                    zorder=4
                )
    
    # 4. アンサンブル代表値 (Run 速度および全体平均速度 +/- SEM)
    for b in BEADS_INFO:
        b_name = b['name']
        row = df_ensemble[df_ensemble['bead_name'] == b_name]
        if not row.empty:
            x_val = row['radius_um'].iloc[0] / xi_plot
            if pd.notna(row['v_run_geom_um_s'].iloc[0]):
                y_val = row['v_run_geom_um_s'].iloc[0]
                y_err = row['v_run_geom_sem_um_s'].iloc[0]
                lbl_suffix = " (Run)"
            else:
                y_val = row['v_mean_um_s'].iloc[0]
                y_err = row['v_sem_um_s'].iloc[0]
                lbl_suffix = " (Mean)"
            
            ax_a.errorbar(
                x_val, y_val, yerr=y_err,
                fmt=b['marker'],
                color=b['color'],
                ecolor='black',
                elinewidth=2.2,
                capsize=6,
                capthick=2.0,
                markersize=13,
                markeredgecolor='black',
                markeredgewidth=1.8,
                label=f"{b['label']}{lbl_suffix}",
                zorder=6
            )
    
    # 5. 理論マスターカーブ & 指数近似
    ax_a.plot(
        x_theory, v_theory_fix,
        color='#111111',
        linewidth=3.2,
        linestyle='-',
        label=rf'Master Curve: $\mathbf{{\langle v_c \rangle = v_0 \cdot \frac{{2}}{{x^2}} \left[ 1 - (1+x)e^{{-x}} \right]}}$' + '\n' +
              rf'  $\left( v_0 = {v0_plot:.3f}\,\mu\mathrm{{m/s}},\ \xi = {xi_plot:.2f}\,\mu\mathrm{{m}},\ R^2 = {r2_fix:.3f} \right)$',
        zorder=5
    )
    ax_a.plot(
        x_theory, v_approx_exp,
        color='#777777',
        linewidth=2.0,
        linestyle='--',
        label=r'Taylor / Exp Approx: $v_0 \exp\left(-\frac{2}{3}x\right) = v_0 \exp\left(-\frac{2 R_c}{3\xi}\right)$',
        zorder=5
    )
    
    ax_a.set_title(r'(a) Cargo Transport Velocity vs Scaled Radius $x = R_c / \xi$', fontsize=14, fontweight='bold', pad=12)
    ax_a.set_xlabel(r'Scaled Cargo Particle Radius $x \equiv \frac{R_c}{\xi} \quad (\xi = 2.78\,\mu\mathrm{m})$', fontsize=12.5, fontweight='bold')
    ax_a.set_ylabel(r'Cargo Velocity $v$ [$\mu\mathrm{m/s}$]', fontsize=13, fontweight='bold')
    ax_a.set_xlim(0.0, 3.9)
    ax_a.set_ylim(0.0, 0.38)
    ax_a.grid(True, linestyle=':', alpha=0.6)
    ax_a.legend(loc='upper right', frameon=True, framealpha=0.92, edgecolor='#cccccc', fontsize=9.0)
    
    # ------------------ Panel B: Data Collapse (Normalized v / v0 vs x) ------------------
    # 1. 個別セグメント点 (無次元化: v_seg / v0)
    if not df_segs.empty:
        for b in BEADS_INFO:
            b_name = b['name']
            sub_s = df_segs[df_segs['bead_name'] == b_name]
            if not sub_s.empty:
                x_pts = sub_s['radius_um'].values / xi_plot
                y_pts = sub_s['v_seg_mean_um_s'].values / v0_plot
                jitter = np.random.normal(0, 0.015 * (x_pts[0] + 0.05), size=len(x_pts))
                ax_b.scatter(
                    x_pts + jitter, y_pts,
                    color=b['color'],
                    alpha=0.10,
                    s=12,
                    edgecolors='none',
                    zorder=1
                )
    
    # 2. 個別トラック点 (無次元化: v_track / v0)
    if not df_tracks.empty:
        for b in BEADS_INFO:
            b_name = b['name']
            sub_t = df_tracks[df_tracks['bead_name'] == b_name]
            if not sub_t.empty:
                x_pts = sub_t['radius_um'].values / xi_plot
                y_pts = sub_t['v_mean_um_s'].values / v0_plot
                ax_b.scatter(
                    x_pts, y_pts,
                    color=b['color'],
                    marker=b['marker'],
                    s=55,
                    alpha=0.60,
                    edgecolors='black',
                    linewidths=0.6,
                    zorder=3
                )
    
    # 3. 実験平均点 (無次元化)
    if not df_exp.empty:
        for b in BEADS_INFO:
            b_name = b['name']
            sub_e = df_exp[df_exp['bead_name'] == b_name]
            if not sub_e.empty:
                x_pts = sub_e['radius_um'].values / xi_plot
                y_pts = sub_e['v_mean_um_s'].values / v0_plot
                ax_b.scatter(
                    x_pts, y_pts,
                    facecolors='none',
                    edgecolors=b['color'],
                    marker=b['marker'],
                    s=120,
                    linewidths=1.8,
                    zorder=4
                )
    
    # 4. アンサンブル代表値 (無次元化)
    for b in BEADS_INFO:
        b_name = b['name']
        row = df_ensemble[df_ensemble['bead_name'] == b_name]
        if not row.empty:
            x_val = row['radius_um'].iloc[0] / xi_plot
            if pd.notna(row['v_run_geom_um_s'].iloc[0]):
                y_val = row['v_run_geom_um_s'].iloc[0] / v0_plot
                y_err = row['v_run_geom_sem_um_s'].iloc[0] / v0_plot
                lbl_suffix = " (Run)"
            else:
                y_val = row['v_mean_um_s'].iloc[0] / v0_plot
                y_err = row['v_sem_um_s'].iloc[0] / v0_plot
                lbl_suffix = " (Mean)"
            
            ax_b.errorbar(
                x_val, y_val, yerr=y_err,
                fmt=b['marker'],
                color=b['color'],
                ecolor='black',
                elinewidth=2.2,
                capsize=6,
                capthick=2.0,
                markersize=13,
                markeredgecolor='black',
                markeredgewidth=1.8,
                label=f"{b['label']}{lbl_suffix}",
                zorder=6
            )
    
    # 5. 普遍マスター関数 g(x)
    ax_b.plot(
        x_theory, g_theory,
        color='#b30000',
        linewidth=3.4,
        linestyle='-',
        label=r'Universal Master Function: $\mathbf{g(x) = \frac{2}{x^2} \left[ 1 - (1+x)e^{-x} \right]}$',
        zorder=5
    )
    ax_b.plot(
        x_theory, np.exp(-2.0 * x_theory / 3.0),
        color='#555555',
        linewidth=2.0,
        linestyle='--',
        label=r'Universal Taylor Approx: $\exp\left(-\frac{2}{3}x\right)$',
        zorder=5
    )
    
    x_asymp = np.linspace(1.2, 4.0, 100)
    ax_b.plot(
        x_asymp, 2.0 / (x_asymp**2),
        color='#0066cc',
        linewidth=1.8,
        linestyle=':',
        label=r'Large-$x$ Asymptote: $2 / x^2$',
        zorder=5
    )
    
    ax_b.set_title(r'(b) Universal Scaling Data Collapse $\langle v_c \rangle / v_0 = g(x)$', fontsize=14, fontweight='bold', pad=12)
    ax_b.set_xlabel(r'Scaled Cargo Particle Radius $x \equiv \frac{R_c}{\xi} \quad (\xi = 2.78\,\mu\mathrm{m})$', fontsize=12.5, fontweight='bold')
    ax_b.set_ylabel(r'Normalized Velocity $\langle v_c \rangle / v_0$', fontsize=13, fontweight='bold')
    ax_b.set_xlim(0.0, 3.9)
    ax_b.set_ylim(0.0, 1.6)
    ax_b.grid(True, linestyle=':', alpha=0.6)
    ax_b.legend(loc='upper right', frameon=True, framealpha=0.92, edgecolor='#cccccc', fontsize=9.0)
    
    # 凡例注記
    legend_elements_note = (
        r"$\bf{Data\ Hierarchy:}$" + "\n" +
        r"$\cdot$ Dots: Sub-trajectory segments ($N > 7000$)" + "\n" +
        r"$\circ$ Filled: Individual tracks ($N = 97$)" + "\n" +
        r"$\diamond$ Open: Movie/FOV averages ($N = 22$)" + "\n" +
        r"$\bullet$ Big + Errorbar: Ensemble Run / Mean $\pm$ SEM"
    )
    ax_a.text(0.03, 0.04, legend_elements_note, transform=ax_a.transAxes,
              fontsize=9.0, verticalalignment='bottom',
              bbox=dict(boxstyle='round,pad=0.5', facecolor='#ffffff', edgecolor='#bbbbbb', alpha=0.92))
    
    plt.tight_layout()
    out_2p = out_dir / 'scaling_master_curve_2panel.png'
    out_2p_svg = out_dir / 'scaling_master_curve_2panel.svg'
    fig.savefig(out_2p, dpi=300)
    fig.savefig(out_2p_svg)
    plt.close(fig)
    print(f"[SUCCESS] Saved 2-panel master curve plot to: {out_2p}")
    
    # =========================================================================
    # 2. 単体図 (Linear Scale: <v_c> vs x)
    # =========================================================================
    fig_lin, ax_lin = plt.subplots(figsize=(9.5, 7.2))
    if not df_segs.empty:
        for b in BEADS_INFO:
            b_name = b['name']
            sub_s = df_segs[df_segs['bead_name'] == b_name]
            if not sub_s.empty:
                x_pts = sub_s['radius_um'].values / xi_plot
                y_pts = sub_s['v_seg_mean_um_s'].values
                jitter = np.random.normal(0, 0.015 * (x_pts[0] + 0.05), size=len(x_pts))
                ax_lin.scatter(x_pts + jitter, y_pts, color=b['color'], alpha=0.10, s=12, edgecolors='none', zorder=1)
    
    if not df_tracks.empty:
        for b in BEADS_INFO:
            b_name = b['name']
            sub_t = df_tracks[df_tracks['bead_name'] == b_name]
            if not sub_t.empty:
                x_pts = sub_t['radius_um'].values / xi_plot
                y_pts = sub_t['v_mean_um_s'].values
                ax_lin.scatter(x_pts, y_pts, color=b['color'], marker=b['marker'], s=60, alpha=0.65, edgecolors='black', linewidths=0.7, zorder=3)
    
    if not df_exp.empty:
        for b in BEADS_INFO:
            b_name = b['name']
            sub_e = df_exp[df_exp['bead_name'] == b_name]
            if not sub_e.empty:
                x_pts = sub_e['radius_um'].values / xi_plot
                y_pts = sub_e['v_mean_um_s'].values
                ax_lin.scatter(x_pts, y_pts, facecolors='none', edgecolors=b['color'], marker=b['marker'], s=130, linewidths=1.8, zorder=4)
    
    for b in BEADS_INFO:
        b_name = b['name']
        row = df_ensemble[df_ensemble['bead_name'] == b_name]
        if not row.empty:
            x_val = row['radius_um'].iloc[0] / xi_plot
            if pd.notna(row['v_run_geom_um_s'].iloc[0]):
                y_val = row['v_run_geom_um_s'].iloc[0]
                y_err = row['v_run_geom_sem_um_s'].iloc[0]
                lbl_suffix = " (Run)"
            else:
                y_val = row['v_mean_um_s'].iloc[0]
                y_err = row['v_sem_um_s'].iloc[0]
                lbl_suffix = " (Mean)"
            
            ax_lin.errorbar(
                x_val, y_val, yerr=y_err,
                fmt=b['marker'], color=b['color'], ecolor='black',
                elinewidth=2.2, capsize=6, capthick=2.0, markersize=14,
                markeredgecolor='black', markeredgewidth=1.8, label=f"{b['label']}{lbl_suffix}", zorder=6
            )
    
    ax_lin.plot(
        x_theory, v_theory_fix,
        color='#111111', linewidth=3.4, linestyle='-',
        label=rf'Master Curve: $\mathbf{{\langle v_c \rangle = v_0 \cdot \frac{{2}}{{x^2}} \left[ 1 - (1+x)e^{{-x}} \right]}}$' + '\n' +
              rf'  $\left( v_0 = {v0_plot:.3f}\,\mu\mathrm{{m/s}},\ \xi = {xi_plot:.2f}\,\mu\mathrm{{m}},\ R^2 = {r2_fix:.3f} \right)$',
        zorder=5
    )
    ax_lin.plot(
        x_theory, v_approx_exp,
        color='#777777', linewidth=2.0, linestyle='--',
        label=r'Taylor / Exp Approx: $v_0 \exp\left(-\frac{2}{3}x\right) = v_0 \exp\left(-\frac{2 R_c}{3\xi}\right)$',
        zorder=5
    )
    
    ax_lin.set_title(r'Cargo Transport Velocity Scaling: $\langle v_c \rangle$ vs $x = R_c / \xi$', fontsize=14, fontweight='bold', pad=12)
    ax_lin.set_xlabel(r'Scaled Cargo Particle Radius $x \equiv \frac{R_c}{\xi} \quad (\xi = 2.78\,\mu\mathrm{m})$', fontsize=13, fontweight='bold')
    ax_lin.set_ylabel(r'Cargo Velocity $\langle v_c \rangle$ [$\mu\mathrm{m/s}$]', fontsize=13, fontweight='bold')
    ax_lin.set_xlim(0.0, 3.9)
    ax_lin.set_ylim(0.0, 0.38)
    ax_lin.grid(True, linestyle=':', alpha=0.6)
    ax_lin.legend(loc='upper right', frameon=True, framealpha=0.92, edgecolor='#cccccc', fontsize=9.5)
    ax_lin.text(0.03, 0.04, legend_elements_note, transform=ax_lin.transAxes,
                fontsize=9.2, verticalalignment='bottom',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#ffffff', edgecolor='#bbbbbb', alpha=0.92))
    
    plt.tight_layout()
    out_lin = out_dir / 'scaling_master_curve_linear.png'
    out_lin_svg = out_dir / 'scaling_master_curve_linear.svg'
    fig_lin.savefig(out_lin, dpi=300)
    fig_lin.savefig(out_lin_svg)
    plt.close(fig_lin)
    print(f"[SUCCESS] Saved linear master curve plot to: {out_lin}")
    
    # =========================================================================
    # 3. 単体図 (Normalized Master Curve Data Collapse: <v_c>/v0 vs x)
    # =========================================================================
    fig_norm, ax_norm = plt.subplots(figsize=(9.5, 7.2))
    if not df_segs.empty:
        for b in BEADS_INFO:
            b_name = b['name']
            sub_s = df_segs[df_segs['bead_name'] == b_name]
            if not sub_s.empty:
                x_pts = sub_s['radius_um'].values / xi_plot
                y_pts = sub_s['v_seg_mean_um_s'].values / v0_plot
                jitter = np.random.normal(0, 0.015 * (x_pts[0] + 0.05), size=len(x_pts))
                ax_norm.scatter(x_pts + jitter, y_pts, color=b['color'], alpha=0.10, s=12, edgecolors='none', zorder=1)
    
    if not df_tracks.empty:
        for b in BEADS_INFO:
            b_name = b['name']
            sub_t = df_tracks[df_tracks['bead_name'] == b_name]
            if not sub_t.empty:
                x_pts = sub_t['radius_um'].values / xi_plot
                y_pts = sub_t['v_mean_um_s'].values / v0_plot
                ax_norm.scatter(x_pts, y_pts, color=b['color'], marker=b['marker'], s=60, alpha=0.65, edgecolors='black', linewidths=0.7, zorder=3)
    
    if not df_exp.empty:
        for b in BEADS_INFO:
            b_name = b['name']
            sub_e = df_exp[df_exp['bead_name'] == b_name]
            if not sub_e.empty:
                x_pts = sub_e['radius_um'].values / xi_plot
                y_pts = sub_e['v_mean_um_s'].values / v0_plot
                ax_norm.scatter(x_pts, y_pts, facecolors='none', edgecolors=b['color'], marker=b['marker'], s=130, linewidths=1.8, zorder=4)
    
    for b in BEADS_INFO:
        b_name = b['name']
        row = df_ensemble[df_ensemble['bead_name'] == b_name]
        if not row.empty:
            x_val = row['radius_um'].iloc[0] / xi_plot
            if pd.notna(row['v_run_geom_um_s'].iloc[0]):
                y_val = row['v_run_geom_um_s'].iloc[0] / v0_plot
                y_err = row['v_run_geom_sem_um_s'].iloc[0] / v0_plot
                lbl_suffix = " (Run)"
            else:
                y_val = row['v_mean_um_s'].iloc[0] / v0_plot
                y_err = row['v_sem_um_s'].iloc[0] / v0_plot
                lbl_suffix = " (Mean)"
            
            ax_norm.errorbar(
                x_val, y_val, yerr=y_err,
                fmt=b['marker'], color=b['color'], ecolor='black',
                elinewidth=2.2, capsize=6, capthick=2.0, markersize=14,
                markeredgecolor='black', markeredgewidth=1.8, label=f"{b['label']}{lbl_suffix}", zorder=6
            )
    
    ax_norm.plot(
        x_theory, g_theory,
        color='#b30000', linewidth=3.4, linestyle='-',
        label=r'Universal Master Curve: $\mathbf{g(x) = \frac{2}{x^2} \left[ 1 - (1+x)e^{-x} \right]}$',
        zorder=5
    )
    ax_norm.plot(
        x_theory, np.exp(-2.0 * x_theory / 3.0),
        color='#555555', linewidth=2.0, linestyle='--',
        label=r'Exponential Approximation: $\exp\left(-\frac{2}{3}x\right)$',
        zorder=5
    )
    ax_norm.plot(
        x_asymp, 2.0 / (x_asymp**2),
        color='#0066cc', linewidth=1.8, linestyle=':',
        label=r'Large-$x$ Power-law: $2 / x^2$',
        zorder=5
    )
    
    ax_norm.set_title(r'Universal Scaling Collapse: $\langle v_c \rangle / v_0 = g(x)$ with $x = R_c / \xi$', fontsize=14, fontweight='bold', pad=12)
    ax_norm.set_xlabel(r'Scaled Cargo Particle Radius $x \equiv \frac{R_c}{\xi} \quad (\xi = 2.78\,\mu\mathrm{m})$', fontsize=13, fontweight='bold')
    ax_norm.set_ylabel(r'Normalized Transport Velocity $\langle v_c \rangle / v_0$', fontsize=13, fontweight='bold')
    ax_norm.set_xlim(0.0, 3.9)
    ax_norm.set_ylim(0.0, 1.6)
    ax_norm.grid(True, linestyle=':', alpha=0.6)
    ax_norm.legend(loc='upper right', frameon=True, framealpha=0.92, edgecolor='#cccccc', fontsize=9.5)
    ax_norm.text(0.03, 0.04, legend_elements_note, transform=ax_norm.transAxes,
                 fontsize=9.2, verticalalignment='bottom',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='#ffffff', edgecolor='#bbbbbb', alpha=0.92))
    
    plt.tight_layout()
    out_norm = out_dir / 'scaling_master_curve_normalized.png'
    out_norm_svg = out_dir / 'scaling_master_curve_normalized.svg'
    fig_norm.savefig(out_norm, dpi=300)
    fig_norm.savefig(out_norm_svg)
    plt.close(fig_norm)
    print(f"[SUCCESS] Saved normalized master curve plot to: {out_norm}")
    
    # =========================================================================
    # 4. Log-Log スケール図 (漸近挙動とスケーリング領域の確認)
    # =========================================================================
    fig_log, ax_log = plt.subplots(figsize=(9.5, 7.2))
    x_log_th = np.logspace(-1.5, 1.2, 500)
    g_log_th = master_function(x_log_th)
    
    if not df_tracks.empty:
        for b in BEADS_INFO:
            b_name = b['name']
            sub_t = df_tracks[df_tracks['bead_name'] == b_name]
            if not sub_t.empty:
                x_pts = sub_t['radius_um'].values / xi_plot
                y_pts = sub_t['v_mean_um_s'].values / v0_plot
                valid = (x_pts > 0) & (y_pts > 0)
                ax_log.scatter(x_pts[valid], y_pts[valid], color=b['color'], marker=b['marker'], s=45, alpha=0.6, edgecolors='black', linewidths=0.5, zorder=3)
    
    for b in BEADS_INFO:
        b_name = b['name']
        row = df_ensemble[df_ensemble['bead_name'] == b_name]
        if not row.empty:
            x_val = row['radius_um'].iloc[0] / xi_plot
            if pd.notna(row['v_run_geom_um_s'].iloc[0]):
                y_val = row['v_run_geom_um_s'].iloc[0] / v0_plot
                y_err = row['v_run_geom_sem_um_s'].iloc[0] / v0_plot
                lbl_suffix = " (Run)"
            else:
                y_val = row['v_mean_um_s'].iloc[0] / v0_plot
                y_err = row['v_sem_um_s'].iloc[0] / v0_plot
                lbl_suffix = " (Mean)"
            
            ax_log.errorbar(
                x_val, y_val, yerr=y_err,
                fmt=b['marker'], color=b['color'], ecolor='black',
                elinewidth=2.2, capsize=6, capthick=2.0, markersize=14,
                markeredgecolor='black', markeredgewidth=1.8, label=f"{b['label']}{lbl_suffix}", zorder=6
            )
    
    ax_log.plot(x_log_th, g_log_th, color='#b30000', linewidth=3.2, linestyle='-', label=r'Universal Master Curve: $g(x) = \frac{2}{x^2}[1 - (1+x)e^{-x}]$', zorder=5)
    ax_log.plot(x_log_th, np.exp(-2.0 * x_log_th / 3.0), color='#555555', linewidth=2.0, linestyle='--', label=r'Exp Approx: $\exp(-2x/3)$', zorder=5)
    ax_log.plot(x_log_th[x_log_th > 0.8], 2.0 / (x_log_th[x_log_th > 0.8]**2), color='#0066cc', linewidth=1.8, linestyle=':', label=r'Asymptote: $2/x^2$', zorder=5)
    
    ax_log.set_xscale('log')
    ax_log.set_yscale('log')
    ax_log.set_title(r'Log-Log Scaling Plot: $\langle v_c \rangle / v_0$ vs $x = R_c / \xi$', fontsize=14, fontweight='bold', pad=12)
    ax_log.set_xlabel(r'Scaled Cargo Particle Radius $x \equiv \frac{R_c}{\xi} \quad (\xi = 2.78\,\mu\mathrm{m})$', fontsize=13, fontweight='bold')
    ax_log.set_ylabel(r'Normalized Velocity $\langle v_c \rangle / v_0$', fontsize=13, fontweight='bold')
    ax_log.set_xlim(0.04, 8.0)
    ax_log.set_ylim(0.02, 2.5)
    ax_log.grid(True, which='both', linestyle=':', alpha=0.6)
    ax_log.legend(loc='lower left', frameon=True, framealpha=0.92, edgecolor='#cccccc', fontsize=9.5)
    
    plt.tight_layout()
    out_log = out_dir / 'scaling_master_curve_loglog.png'
    out_log_svg = out_dir / 'scaling_master_curve_loglog.svg'
    fig_log.savefig(out_log, dpi=300)
    fig_log.savefig(out_log_svg)
    plt.close(fig_log)
    print(f"[SUCCESS] Saved log-log master curve plot to: {out_log}")
    
    # =========================================================================
    # 5. サマリーCSVの保存
    # =========================================================================
    out_csv = out_dir / 'scaling_data_points_summary.csv'
    df_ensemble['fit_v0_fixed_um_s'] = v0_fix
    df_ensemble['fit_xi_fixed_um'] = xi_fix
    df_ensemble['fit_r2_fixed'] = r2_fix
    df_ensemble['fit_v0_free_um_s'] = v0_free
    df_ensemble['fit_xi_free_um'] = xi_free
    df_ensemble['fit_r2_free'] = r2_free
    df_ensemble['theory_v_master_um_s'] = theoretical_velocity(df_ensemble['radius_um'].values, v0_fix, xi_fix)
    df_ensemble['theory_g_x'] = master_function(df_ensemble['radius_um'].values / xi_fix)
    df_ensemble.to_csv(out_csv, index=False)
    print(f"[SUCCESS] Saved scaling data points summary to: {out_csv}")


def main():
    parser = argparse.ArgumentParser(description="Plot cargo velocity scaling vs Rc/xi and master curve.")
    parser.add_argument('--root_dir', type=str, default=None, help='Root directory containing bead data')
    parser.add_argument('--out_dir', type=str, default='figure/scaling', help='Output directory for plots')
    parser.add_argument('--xi', type=float, default=2.78, help='Active flow correlation length xi in um')
    parser.add_argument('--window_frames', type=int, default=10, help='Window frames for segment velocity')
    args = parser.parse_args()
    
    root_dir = Path(args.root_dir) if args.root_dir else find_default_root()
    out_dir = root_dir / args.out_dir
    
    print(f"Data root: {root_dir}")
    print(f"Output directory: {out_dir}")
    print(f"Base correlation length xi: {args.xi} um")
    
    df_segs, df_tracks, df_exp, df_ensemble = load_all_velocity_data(
        root_dir=root_dir,
        xi_default=args.xi,
        window_frames=args.window_frames
    )
    
    print(f"\nLoaded data counts:")
    print(f"  - Sliding window segments: {len(df_segs)} points")
    print(f"  - Individual particle tracks: {len(df_tracks)} points")
    print(f"  - Experiments (Movies/FOVs): {len(df_exp)} points")
    print(f"  - Bead size conditions: {len(df_ensemble)} conditions\n")
    
    plot_scaling_master_curve(
        df_segs=df_segs,
        df_tracks=df_tracks,
        df_exp=df_exp,
        df_ensemble=df_ensemble,
        out_dir=out_dir,
        xi_val=args.xi
    )


if __name__ == '__main__':
    main()
