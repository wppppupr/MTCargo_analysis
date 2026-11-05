#!/usr/bin/env python3
r"""
calc_and_plot_time_lagged_cargo_flow.py

貨物による微小管流速低下の起源を検証する新規解析スクリプト:
【仮説検証】
(a) 集積仮説 (Accumulation): 貨物が、もともと遅い領域に行き着く。
(b) 撹乱仮説 (Disturbance): 貨物が障害物になって、周りの微小管を乱し遅い領域を作る。

【判定基準】
- 時刻 t の貨物の位置 x_c(t) のまわりで、少し前の時刻 t - \Delta t の流れ場を使って |u|(r/\xi) を計算。
- 貨物が来る前から低下があり、\Delta t を大きくするにつれて徐々に浅くなる -> (a) 集積。
- 低下が到着後にしか現れない -> (b) 撹乱。

【時間差 \Delta t の設定】
- \Delta t = 0 s (同時刻 / 貨物滞在時)
- \Delta t = 12 s (3 frames)
- \Delta t = 32 s (8 frames)
- \Delta t = 64 s (16 frames) ~ \xi / |v_c|
- \Delta t = 128 s (32 frames) ~ 流れ場の寿命 \tau_life
- --delta_t で任意指定可。負の \Delta t は貨物通過「後」の流れ場 (t_flow = t + |\Delta t|)
  を表し、撹乱の緩和 (回復) を見る対照として使える。例: --delta_t -64 -32 0 32 64 128
"""

import os
import sys
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ワークスペースおよび保存先設定
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = Path("/mnt/NAS-Ebanaru/sasaki/MTsingleBeads")
if not ROOT_DIR.exists():
    ROOT_DIR = CURRENT_DIR

OUT_DIRS = [
    CURRENT_DIR / "figure" / "cargo_lagged_flow_analysis",
    ROOT_DIR / "figure" / "cargo_lagged_flow_analysis"
]
CACHE_DIR = CURRENT_DIR / "scratch"

# スタイル適用
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
    {"name": "beads06um", "diameter_um": 0.63, "radius_um": 0.315, "marker": "^", "color": style_colors[0], "label": r"$0.63\,\mu\mathrm{m}$", "rc_px": 3.0},
    {"name": "beads1um",  "diameter_um": 1.18, "radius_um": 0.590, "marker": "o", "color": style_colors[1], "label": r"$1.18\,\mu\mathrm{m}$", "rc_px": 6.0},
    {"name": "beads3um",  "diameter_um": 3.37, "radius_um": 1.685, "marker": "d", "color": style_colors[2], "label": r"$3.37\,\mu\mathrm{m}$", "rc_px": 15.0},
    {"name": "beads5um",  "diameter_um": 5.00, "radius_um": 2.500, "marker": "p", "color": style_colors[3], "label": r"$5.00\,\mu\mathrm{m}$", "rc_px": 25.0},
    {"name": "beads7um",  "diameter_um": 7.24, "radius_um": 3.620, "marker": "h", "color": style_colors[4], "label": r"$7.24\,\mu\mathrm{m}$", "rc_px": 35.0},
    {"name": "beads20um", "diameter_um": 20.0, "radius_um": 10.00, "marker": "s", "color": style_colors[5], "label": r"$20.0\,\mu\mathrm{m}$", "rc_px": 95.0},
]

TARGET_DIRS = [
    "/mnt/NAS-Ebanaru/sasaki/MTsingleBeads/beads06um/20260716/beads06um",
    "/mnt/NAS-Ebanaru/sasaki/MTsingleBeads/beads06um/20260716/beads06um001",
    "/mnt/NAS-Ebanaru/sasaki/MTsingleBeads/beads06um/20260716/beads06um002",
    "/mnt/NAS-Ebanaru/sasaki/MTsingleBeads/beads06um/20260716/beads06um003",
    "/mnt/NAS-Ebanaru/sasaki/MTsingleBeads/beads1um/20260717/beads1um001_crop",
    "/mnt/NAS-Ebanaru/sasaki/MTsingleBeads/beads1um/20260717/beads1um002",
    "/mnt/NAS-Ebanaru/sasaki/MTsingleBeads/beads1um/20260717/beads1um003",
    "/mnt/NAS-Ebanaru/sasaki/MTsingleBeads/beads1um/20260717/beads1um_crop",
    "/mnt/NAS-Ebanaru/sasaki/MTsingleBeads/beads3um/20260708/beads3um",
    "/mnt/NAS-Ebanaru/sasaki/MTsingleBeads/beads3um/20260708/beads3um001",
    "/mnt/NAS-Ebanaru/sasaki/MTsingleBeads/beads3um/20260708/beads3um002",
    "/mnt/NAS-Ebanaru/sasaki/MTsingleBeads/beads3um/20260708/beads3um003",
    "/mnt/NAS-Ebanaru/sasaki/MTsingleBeads/beads5um/20260715/beads5um",
    "/mnt/NAS-Ebanaru/sasaki/MTsingleBeads/beads5um/20260715/beads5um001",
    "/mnt/NAS-Ebanaru/sasaki/MTsingleBeads/beads5um/20260715/beads5um002",
    "/mnt/NAS-Ebanaru/sasaki/MTsingleBeads/beads5um/20260715/beads5um003",
    "/mnt/NAS-Ebanaru/sasaki/MTsingleBeads/beads7um/20260608/beads7um",
    "/mnt/NAS-Ebanaru/sasaki/MTsingleBeads/beads7um/20260608/beads7um002",
    "/mnt/NAS-Ebanaru/sasaki/MTsingleBeads/beads7um/20260608/beads7um003",
    "/mnt/NAS-Ebanaru/sasaki/MTsingleBeads/beads20um/20260723/beads20um001",
    "/mnt/NAS-Ebanaru/sasaki/MTsingleBeads/beads20um/20260723/beads20um002",
    "/mnt/NAS-Ebanaru/sasaki/MTsingleBeads/beads20um/20260723/beads20um_crop"
]

SCALE_UM_PER_PX = 0.11
DT_SEC = 4.0
V_UNIT = SCALE_UM_PER_PX / DT_SEC # um/s

# 評価する時間遅れ \Delta t (秒単位)
DELTA_T_SEC_LIST = [0, 12, 32, 64, 128]

def setup_matplotlib():
    plt.rcParams.update({
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.family': 'sans-serif',
        'mathtext.fontset': 'dejavusans',
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 15,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 10.0,
        'figure.titlesize': 16,
        'axes.linewidth': 1.2,
        'lines.linewidth': 2.0,
        'grid.linewidth': 0.6,
        'grid.alpha': 0.4,
        'grid.linestyle': '--',
    })

def save_fig(fig, base_name):
    for out_dir in OUT_DIRS:
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f"{base_name}.png", dpi=300, bbox_inches='tight')
        fig.savefig(out_dir / f"{base_name}.svg", format='svg', bbox_inches='tight')
    print(f"Saved figure: {base_name}.png and .svg")

def load_xi_dict():
    candidates = [
        CURRENT_DIR / "figure" / "bg_angular_correlation" / "bg_angular_correlation_length_per_experiment.csv",
        ROOT_DIR / "figure" / "bg_angular_correlation" / "bg_angular_correlation_length_per_experiment.csv",
    ]
    xi_map = {}
    for p in candidates:
        if p.exists():
            try:
                df = pd.read_csv(p)
                for _, r in df.iterrows():
                    exp_name = str(r['exp_dir']).strip()
                    xi_val = float(r['xi_um'])
                    xi_map[exp_name] = xi_val
                print(f"Loaded {len(xi_map)} xi values from {p.name}")
                break
            except Exception as e:
                print(f"Warning: Failed to load {p}: {e}")
    return xi_map

def process_experiment_timelag_fast(exp_dir_str: str, b_info: dict, max_r_edge_um: float = 20.0, bin_width_um: float = 0.50, n_samples: int = 12,
                                    delta_t_sec_list=None):
    """
    メモリキャッシングを活用した高速タイムラグ解析
    delta_t_sec_list: 秒単位の時間差リスト。負値は貨物通過後の流れ場 (t_flow = t - \\Delta t > t)。
    """
    if delta_t_sec_list is None:
        delta_t_sec_list = DELTA_T_SEC_LIST
    import h5py  # 遅延 import: キャッシュから作図するだけなら h5py 不要
    exp_dir = Path(exp_dir_str)
    h5_path = exp_dir / "GFP_flows.h5"
    csv_path = exp_dir / "beads_tracks.csv"
    
    if not h5_path.exists() or not csv_path.exists():
        return None
        
    df_tracks = pd.read_csv(csv_path)
    if 'particle' not in df_tracks.columns:
        df_tracks['particle'] = 0
        
    Rc_px = b_info['rc_px']
    max_r_edge_px = max_r_edge_um / SCALE_UM_PER_PX
    bin_width_px = bin_width_um / SCALE_UM_PER_PX
    bins_px = np.arange(0, max_r_edge_px + bin_width_px, bin_width_px)
    n_bins = len(bins_px) - 1
    bin_centers_um = 0.5 * (bins_px[:-1] + bins_px[1:]) * SCALE_UM_PER_PX

    delta_frames = [int(round(s / DT_SEC)) for s in delta_t_sec_list]
    max_dt_fr = max(max(delta_frames), 0)
    min_dt_fr = min(min(delta_frames), 0)

    with h5py.File(str(h5_path), 'r') as hf:
        dataset_key = list(hf.keys())[0]
        flows = hf[dataset_key]
        num_frames = flows.shape[0]
        H, W = flows.shape[2], flows.shape[3]
        
        valid_frames = [f for f in sorted(df_tracks['frame'].unique()) if f - max_dt_fr >= 0 and f - min_dt_fr < num_frames]
        if len(valid_frames) == 0:
            return None
            
        sample_indices = np.linspace(0, len(valid_frames) - 1, min(len(valid_frames), n_samples), dtype=int)
        sampled_t_beads = [valid_frames[i] for i in sample_indices]
        
        # 必要なフローフレームのみを一括取得して速度行列をメモリに保持
        needed_flow_frames = set()
        for t_bead in sampled_t_beads:
            for dt_fr in delta_frames:
                needed_flow_frames.add(t_bead - dt_fr)
        sorted_flow_frames = sorted(list(needed_flow_frames))
        
        speed_cache = {}
        for f_idx in sorted_flow_frames:
            fl = flows[f_idx]
            speed_cache[f_idx] = np.sqrt(fl[0].astype(np.float32)**2 + fl[1].astype(np.float32)**2).astype(np.float16)

        profiles_by_dt = {dt_sec: {'sums': np.zeros(n_bins, dtype=np.float64), 'counts': np.zeros(n_bins, dtype=np.int64)}
                          for dt_sec in delta_t_sec_list}
                          
        for t_bead in sampled_t_beads:
            sub_tracks = df_tracks[df_tracks['frame'] == t_bead]
            if len(sub_tracks) == 0:
                continue
                
            # ビーズの位置・マスクをこの時刻で一括計算
            bead_geoms = []
            for _, row in sub_tracks.iterrows():
                bx, by = row['x'], row['y']
                max_d = Rc_px + max_r_edge_px
                x_min = max(0, int(np.floor(bx - max_d)))
                x_max = min(W, int(np.ceil(bx + max_d)) + 1)
                y_min = max(0, int(np.floor(by - max_d)))
                y_max = min(H, int(np.ceil(by + max_d)) + 1)
                
                y_idx, x_idx = np.ogrid[y_min:y_max, x_min:x_max]
                dist = np.sqrt((x_idx - bx)**2 + (y_idx - by)**2)
                r_edge = dist - Rc_px
                valid = (r_edge >= 0) & (r_edge < max_r_edge_px)
                if np.any(valid):
                    b_idx = np.digitize(r_edge[valid], bins_px) - 1
                    b_c = np.bincount(b_idx, minlength=n_bins)
                    bead_geoms.append((x_min, x_max, y_min, y_max, valid, b_idx, b_c))
                    
            # 各時間差 \Delta t の速度場を積算
            for dt_sec, dt_fr in zip(delta_t_sec_list, delta_frames):
                spd = speed_cache[t_bead - dt_fr]
                for (x_min, x_max, y_min, y_max, valid, b_idx, b_c) in bead_geoms:
                    sub_spd = spd[y_min:y_max, x_min:x_max][valid].astype(np.float32)
                    b_s = np.bincount(b_idx, weights=sub_spd, minlength=n_bins)
                    profiles_by_dt[dt_sec]['counts'] += b_c
                    profiles_by_dt[dt_sec]['sums'] += b_s

    far_mask = bin_centers_um >= 12.0
    norm_profiles_by_dt = {}
    u_inf_by_dt = {}
    
    for dt_sec in delta_t_sec_list:
        sums = profiles_by_dt[dt_sec]['sums']
        counts = profiles_by_dt[dt_sec]['counts']
        spd = (sums / np.maximum(counts, 1)) * V_UNIT
        if np.sum(counts[far_mask]) > 50:
            u_inf = np.average(spd[far_mask], weights=counts[far_mask])
        else:
            u_inf = np.mean(spd[int(n_bins * 0.7):])
        u_inf_by_dt[dt_sec] = u_inf
        norm_profiles_by_dt[dt_sec] = spd / u_inf if u_inf > 0 else np.zeros_like(spd)

    return {
        'exp_dir': exp_dir_str,
        'bead_name': b_info['name'],
        'diameter_um': b_info['diameter_um'],
        'radius_um': b_info['radius_um'],
        'r_edge_um': bin_centers_um,
        'norm_profiles_by_dt': norm_profiles_by_dt,
        'u_inf_by_dt': u_inf_by_dt
    }

def fit_relaxation_time(dt_vals, depth_vals, depth_sems):
    """
    \Delta t >= 0 の低下深さを D(\Delta t) = D_0 exp(-\Delta t / \tau) でフィットし (D_0, \tau, \tau_err) を返す。
    集積仮説 (a) では \tau ~ 流れ場の寿命 (有限・大), 撹乱仮説 (b) では \tau -> 0 (到着時のみ低下)。
    点数不足・D_0 <= 0・収束失敗時は NaN。
    """
    dt = np.asarray(dt_vals, float); d = np.asarray(depth_vals, float); e = np.asarray(depth_sems, float)
    m = (dt >= 0) & np.isfinite(d)
    dt, d, e = dt[m], d[m], e[m]
    if len(dt) < 3 or d[dt == 0].size == 0 or d[dt == 0][0] <= 0:
        return np.nan, np.nan, np.nan
    sigma = np.where(e > 0, e, np.nan)
    sigma = np.where(np.isfinite(sigma), sigma, np.nanmean(sigma) if np.any(np.isfinite(sigma)) else 1.0)
    try:
        popt, pcov = curve_fit(lambda x, d0, tau: d0 * np.exp(-x / tau), dt, d,
                               p0=[d[dt == 0][0], 60.0], sigma=sigma, absolute_sigma=True,
                               bounds=([0.0, 1.0], [1.0, 1e4]), maxfev=5000)
        return float(popt[0]), float(popt[1]), float(np.sqrt(np.diag(pcov))[1])
    except (RuntimeError, ValueError):
        return np.nan, np.nan, np.nan

def main():
    global DELTA_T_SEC_LIST
    parser = argparse.ArgumentParser(description="Time-lagged cargo flow profile analysis: accumulation vs disturbance.")
    parser.add_argument('--force_recompute', action='store_true', help="Force recomputation ignoring cache")
    parser.add_argument('--n_samples', type=int, default=12, help="Number of bead time frames sampled (default 12)")
    parser.add_argument('--rep_bead', type=str, default='beads3um', choices=[b['name'] for b in BEADS_INFO],
                        help="Bead used for the single-panel Figure 1 (default beads3um)")
    parser.add_argument('--edge_rxi', type=float, default=0.0,
                        help="Average edge speed over r/xi <= edge_rxi for slowdown depth (0 = first grid point only, legacy)")
    parser.add_argument('--max_scaled_r', type=float, default=2.5, help="Upper limit of r/xi axis (default 2.5)")
    parser.add_argument('--delta_t', type=int, nargs='+', default=None,
                        help="Lag times in seconds (multiples of 4 s). Negative = flow AFTER cargo passage. "
                             f"Default: {DELTA_T_SEC_LIST}")
    args = parser.parse_args()

    if args.delta_t is not None:
        bad = [d for d in args.delta_t if d % int(DT_SEC) != 0]
        if bad:
            parser.error(f"--delta_t values must be multiples of {DT_SEC:g} s: {bad}")
        DELTA_T_SEC_LIST = sorted(set(args.delta_t))
    is_default_dt = DELTA_T_SEC_LIST == [0, 12, 32, 64, 128]

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    # パラメータごとにキャッシュを分離 (デフォルト設定は従来のファイル名を維持)
    if is_default_dt and args.n_samples == 12:
        cache_file = CACHE_DIR / "time_lagged_flow_fast_cache.pkl"
    else:
        dt_tag = "_".join(f"m{-d}" if d < 0 else str(d) for d in DELTA_T_SEC_LIST)
        cache_file = CACHE_DIR / f"time_lagged_flow_fast_cache_n{args.n_samples}_dt{dt_tag}.pkl"
    suffix = "" if is_default_dt else "_custom_dt"

    df_all = None
    if cache_file.exists() and not args.force_recompute:
        print(f"Loading cached time-lagged profiles from {cache_file}...")
        df_all = pd.read_pickle(cache_file)
        # キャッシュに必要な \Delta t が揃っているか検証 (不足なら再計算)
        if len(df_all) == 0 or not all(
                set(DELTA_T_SEC_LIST) <= set(d.keys()) for d in df_all['norm_profiles_by_dt']):
            print("  Cache lacks some requested Delta t values -> recomputing.")
            df_all = None
    if df_all is None:
        print(f"Computing time-lagged flow profiles across {len(TARGET_DIRS)} experiments (fast cached)...")
        t0 = time.time()
        results = []
        # 実験ごとの途中結果キャッシュ: 長時間計算が中断 (timeout 等) されても再実行時に続きから再開できる。
        # サンプリングされるフレームは \Delta t の集合と n_samples に依存するため、最終キャッシュと同じキーで分離する。
        partial_dir = CACHE_DIR / "time_lagged_partial" / cache_file.stem
        if args.force_recompute and partial_dir.exists():
            for f in partial_dir.glob("*.pkl"):
                f.unlink()
        partial_dir.mkdir(parents=True, exist_ok=True)
        for exp_idx, exp_dir in enumerate(TARGET_DIRS):
            b_name = None
            for b in BEADS_INFO:
                if f"/{b['name']}/" in exp_dir:
                    b_name = b['name']
                    b_info = b
                    break
            if not b_name:
                continue
                
            exp_path = Path(exp_dir)
            part_file = partial_dir / f"{exp_path.parent.name}__{exp_path.name}.pkl"
            if part_file.exists():
                res = pd.read_pickle(part_file)
                print(f"[{exp_idx+1}/{len(TARGET_DIRS)}] {b_name}: {exp_path.name} -> resumed from partial cache.")
                if res is not None:
                    results.append(res)
                continue

            print(f"[{exp_idx+1}/{len(TARGET_DIRS)}] Processing {b_name}: {exp_path.name}...")
            t_exp = time.time()
            try:
                res = process_experiment_timelag_fast(exp_dir, b_info, max_r_edge_um=20.0, bin_width_um=0.50, n_samples=args.n_samples,
                                                      delta_t_sec_list=DELTA_T_SEC_LIST)
            except (OSError, KeyError, ValueError) as e:
                # 1 実験の HDF5 破損・NAS 読み込み失敗で全体を落とさない (部分キャッシュは書かず次回再試行)
                print(f"    Warning: skipped due to {type(e).__name__}: {e}")
                continue
            pd.to_pickle(res, part_file)  # None (データ欠損) も記録して再探索を避ける
            if res is not None:
                results.append(res)
                print(f"    Done in {time.time() - t_exp:.1f} s.")
            else:
                print("    Skipped (missing GFP_flows.h5 / beads_tracks.csv or no valid frames).")
                
        df_all = pd.DataFrame(results)
        df_all.to_pickle(cache_file)
        print(f"All experiments processed in {time.time() - t0:.1f} s. Cached to {cache_file}.")

    xi_map = load_xi_dict()
    setup_matplotlib()

    max_scaled_r = args.max_scaled_r
    scaled_r_grid = np.linspace(0.04, max_scaled_r, 50)
    ensemble_by_bead = {}
    
    for b in BEADS_INFO:
        b_name = b['name']
        sub = df_all[df_all['bead_name'] == b_name]
        if len(sub) == 0:
            continue
            
        ensemble_by_bead[b_name] = {'info': b, 'profiles': {}}
        
        for dt_sec in DELTA_T_SEC_LIST:
            curves = []
            for _, row in sub.iterrows():
                exp_n = Path(row['exp_dir']).name
                xi_val = xi_map.get(exp_n, np.nan)
                if not np.isfinite(xi_val) or xi_val <= 0:
                    continue
                r_um = row['r_edge_um']
                r_over_xi = r_um / xi_val
                norm_spd = row['norm_profiles_by_dt'][dt_sec]
                f_interp = interp1d(r_over_xi, norm_spd, bounds_error=False, fill_value=(norm_spd[0], norm_spd[-1]))
                curves.append(f_interp(scaled_r_grid))
                
            if len(curves) > 0:
                mat = np.vstack(curves)
                mean_c = np.mean(mat, axis=0)
                std_c = np.std(mat, axis=0)
                sem_c = std_c / np.sqrt(len(curves)) if len(curves) > 1 else np.zeros_like(std_c)
                # 端の速度: edge_rxi 以下の r/xi で各実験ごとに平均 → 実験間で mean/SEM
                edge_mask = scaled_r_grid <= args.edge_rxi
                if not np.any(edge_mask):
                    edge_mask = np.zeros_like(scaled_r_grid, dtype=bool)
                    edge_mask[0] = True
                edge_per_exp = mat[:, edge_mask].mean(axis=1)
                edge_val = float(np.mean(edge_per_exp))
                edge_sem = float(np.std(edge_per_exp) / np.sqrt(len(curves))) if len(curves) > 1 else 0.0
                ensemble_by_bead[b_name]['profiles'][dt_sec] = {
                    'mean': mean_c,
                    'sem': sem_c,
                    'edge_val': edge_val,
                    'edge_sem': edge_sem,
                    'depth': 1.0 - edge_val,
                    'n_exps': len(curves)
                }

    if not ensemble_by_bead:
        print("Error: no experiment had a valid xi value; check bg_angular_correlation_length_per_experiment.csv")
        sys.exit(1)

    # =========================================================================
    # Figure 1: 代表的なビーズ径 (3.37 um) における \Delta t 依存性プロファイル
    # =========================================================================
    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    dt_colors = plt.cm.viridis(np.linspace(0.05, 0.90, len(DELTA_T_SEC_LIST)))
    
    rep_bead = args.rep_bead
    if rep_bead in ensemble_by_bead:
        p_data = ensemble_by_bead[rep_bead]['profiles']
        for idx, dt_sec in enumerate(DELTA_T_SEC_LIST):
            if dt_sec not in p_data:
                continue
            p = p_data[dt_sec]
            phase = " (Arrival / Present)" if dt_sec == 0 else (" (Before arrival)" if dt_sec > 0 else " (After passage)")
            lbl = f"$\\Delta t = {dt_sec}\\,\\mathrm{{s}}$" + phase
            ax.plot(scaled_r_grid, p['mean'], color=dt_colors[idx], lw=2.2, label=lbl)
            ax.fill_between(scaled_r_grid, p['mean'] - p['sem'], p['mean'] + p['sem'],
                            color=dt_colors[idx], alpha=0.15)
            
        ax.axhline(1.0, color='gray', linestyle=':', lw=1.5, label=r'Bulk $u_{\infty}$')
        ax.axvline(1.0, color='black', linestyle='--', alpha=0.5, lw=1.2, label=r'$r = \xi$')
        ax.set_xlabel(r'Scaled distance from edge $r \,/\, \xi \equiv (\mathrm{dist} - R_c) \,/\, \xi$ [-]')
        ax.set_ylabel(r'Normalized flow speed $|\mathbf{u}|(r/\xi) \,/\, u_{\infty}$ [-]')
        ax.set_title('Time-Lagged Flow Profile Around Cargo: Beads ' + ensemble_by_bead[rep_bead]['info']['label'])
        ax.set_xlim(0, max_scaled_r)
        ax.set_ylim(0.55, 1.05)
        ax.legend(loc='lower right', frameon=True, framealpha=0.9, fontsize=9.5)
        ax.grid(True)
        save_fig(fig, f"cargo_timelagged_flow_profile_{rep_bead}{suffix}")
    else:
        print(f"Warning: {rep_bead} has no valid profiles; Figure 1 skipped.")
    plt.close(fig)

    # =========================================================================
    # Figure 2: 全ビーズ径における 6 パネル \Delta t 依存性プロファイル
    # =========================================================================
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.0), sharex=True, sharey=True)
    axes = axes.flatten()
    
    for idx, b in enumerate(BEADS_INFO):
        ax = axes[idx]
        b_name = b['name']
        if b_name not in ensemble_by_bead:
            ax.set_visible(False)
            continue
            
        p_data = ensemble_by_bead[b_name]['profiles']
        for dt_idx, dt_sec in enumerate(DELTA_T_SEC_LIST):
            if dt_sec not in p_data:
                continue
            p = p_data[dt_sec]
            ax.plot(scaled_r_grid, p['mean'], color=dt_colors[dt_idx], lw=2.0,
                    label=f"$\\Delta t = {dt_sec}\\,\\mathrm{{s}}$")
            ax.fill_between(scaled_r_grid, p['mean'] - p['sem'], p['mean'] + p['sem'],
                            color=dt_colors[dt_idx], alpha=0.12)
            
        ax.axhline(1.0, color='gray', linestyle=':', lw=1.2)
        ax.axvline(1.0, color='black', linestyle='--', alpha=0.4, lw=1.0)
        n_exps = max((p['n_exps'] for p in p_data.values()), default=0)
        ax.set_title(f"{b['label']} (N={n_exps})", fontsize=13, fontweight='bold')
        ax.grid(True)
        if idx in [0, 3]:
            ax.set_ylabel(r'$|\mathbf{u}|(r/\xi) \,/\, u_{\infty}$ [-]')
        if idx >= 3:
            ax.set_xlabel(r'Scaled distance $r \,/\, \xi$ [-]')
        ax.set_xlim(0, max_scaled_r)
        ax.set_ylim(0.45, 1.05)
        if idx == 0:
            ax.legend(loc='lower right', fontsize=8.5)
            
    plt.suptitle(r'Evolution of Flow Deficit with Pre-Arrival Lag Time $\Delta t$ ($t_{\mathrm{flow}} = t - \Delta t$)',
                 fontsize=15, y=0.98)
    plt.tight_layout()
    save_fig(fig, f"cargo_timelagged_flow_profile_6panels{suffix}")
    plt.close(fig)

    # =========================================================================
    # Figure 3: 【判定プロット】低下の深さ (Depth = 1 - |u_edge|/u_inf) vs \Delta t
    # =========================================================================
    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    summary_records = []
    
    for b in BEADS_INFO:
        b_name = b['name']
        if b_name not in ensemble_by_bead:
            continue
        p_data = ensemble_by_bead[b_name]['profiles']
        
        dt_vals = []
        depth_vals = []
        depth_sems = []
        
        for dt_sec in DELTA_T_SEC_LIST:
            if dt_sec not in p_data:
                continue
            p = p_data[dt_sec]
            dt_vals.append(dt_sec)
            depth_vals.append(p['depth'])
            depth_sems.append(p['edge_sem'])
            
            summary_records.append({
                'bead_name': b_name,
                'diameter_um': b['diameter_um'],
                'delta_t_sec': dt_sec,
                'edge_norm_speed': p['edge_val'],
                'edge_sem': p['edge_sem'],
                'slowdown_depth': p['depth'],
                'slowdown_pct': p['depth'] * 100,
                'edge_rxi_window': args.edge_rxi,
                'n_exps': p['n_exps'],
            })
            
        ax.errorbar(dt_vals, depth_vals, yerr=depth_sems, fmt='-', color=b['color'],
                    marker=b['marker'], markersize=7.0, lw=2.2, capsize=3.5, label=b['label'])
        
    ax.axhline(0.0, color='gray', linestyle='--', lw=1.2, label=r'No slowdown ($|\mathbf{u}| = u_{\infty}$)')
    ax.set_xlabel(r'Pre-arrival lag time $\Delta t \equiv t_{\mathrm{cargo}} - t_{\mathrm{flow}}$ [$\mathrm{s}$]')
    ax.set_ylabel(r'Flow slowdown depth at edge $1 - |\mathbf{u}|(r\to 0) \,/\, u_{\infty}$ [-]')
    ax.set_title(r'Slowdown Depth vs $\Delta t$: Accumulation (a) vs Disturbance (b)')
    ax.set_ylim(-0.10, 0.55)
    ax.set_xlim(min(min(DELTA_T_SEC_LIST), 0) - 5, max(DELTA_T_SEC_LIST) + 10)
    if min(DELTA_T_SEC_LIST) < 0:
        # 貨物通過後 (\Delta t < 0) の領域を網掛けして区別
        ax.axvspan(ax.get_xlim()[0], 0, color='lightgray', alpha=0.35, zorder=0, lw=0)
        ax.text(0.01, 0.02, 'After passage', transform=ax.transAxes, fontsize=9, color='dimgray')
    ax.legend(loc='upper right', frameon=True, framealpha=0.9, fontsize=10.0)
    ax.grid(True)
    
    # 判定の物理的ガイドボックス
    ax.text(0.04, 0.82, r"Criterion for Hypothesis (a) Accumulation:" "\n" r"Slowdown persists before arrival ($\Delta t > 0$)" "\n" r"and gradually relaxes with flow lifetime." "\n\n" r"Criterion for Hypothesis (b) Disturbance:" "\n" r"Slowdown appears only upon arrival ($\Delta t = 0$).",
            transform=ax.transAxes, fontsize=9.2, bbox=dict(boxstyle='round,pad=0.4', facecolor='azure', alpha=0.9, edgecolor='steelblue'))
    
    save_fig(fig, f"cargo_slowdown_depth_vs_delta_t{suffix}")
    plt.close(fig)

    df_sum = pd.DataFrame(summary_records)
    for out_dir in OUT_DIRS:
        df_sum.to_csv(out_dir / f"cargo_timelagged_slowdown_summary{suffix}.csv", index=False)

    # =========================================================================
    # 定量判定指標:
    #  (i)  緩和時間 \tau: D(\Delta t>=0) = D_0 exp(-\Delta t/\tau)
    #  (ii) 前後非対称性 A(|\Delta t|) = [D(-|\Delta t|) - D(+|\Delta t|)] / D(0)
    #       A > 0: 通過後の方が遅い (撹乱の航跡 → (b)), A ~ 0: 時間対称 (既存の遅い領域 → (a))
    # =========================================================================
    metric_records = []
    asym_by_bead = {}
    for b_name, g in df_sum.groupby('bead_name', sort=False):
        d_map = dict(zip(g['delta_t_sec'], g['slowdown_depth']))
        e_map = dict(zip(g['delta_t_sec'], g['edge_sem']))
        d0_fit, tau, tau_err = fit_relaxation_time(g['delta_t_sec'], g['slowdown_depth'], g['edge_sem'])
        rec = {'bead_name': b_name, 'diameter_um': g['diameter_um'].iloc[0],
               'D0_fit': d0_fit, 'tau_relax_s': tau, 'tau_relax_err_s': tau_err}
        d0 = d_map.get(0, np.nan)
        pairs = sorted(dt for dt in d_map if dt > 0 and -dt in d_map)
        if pairs and np.isfinite(d0) and d0 > 0:
            a_vals = [(d_map[-dt] - d_map[dt]) / d0 for dt in pairs]
            a_errs = [np.hypot(e_map[-dt], e_map[dt]) / d0 for dt in pairs]
            asym_by_bead[b_name] = (pairs, a_vals, a_errs)
            for dt, a, ae in zip(pairs, a_vals, a_errs):
                rec[f'asym_{dt}s'] = a
                rec[f'asym_{dt}s_err'] = ae
        metric_records.append(rec)
    df_metric = pd.DataFrame(metric_records)
    for out_dir in OUT_DIRS:
        df_metric.to_csv(out_dir / f"cargo_timelagged_criterion_metrics{suffix}.csv", index=False)
    print("\n=== Criterion metrics (tau: relaxation time, asym: after-before asymmetry / D(0)) ===")
    print(df_metric.round(3).to_string(index=False))

    if asym_by_bead:
        fig, ax = plt.subplots(figsize=(7.0, 5.0))
        for b in BEADS_INFO:
            if b['name'] not in asym_by_bead:
                continue
            pairs, a_vals, a_errs = asym_by_bead[b['name']]
            ax.errorbar(pairs, a_vals, yerr=a_errs, fmt='-', color=b['color'], marker=b['marker'],
                        markersize=7.0, lw=2.0, capsize=3.5, label=b['label'])
        ax.axhline(0.0, color='gray', linestyle='--', lw=1.2)
        ax.set_xlabel(r'Lag magnitude $|\Delta t|$ [$\mathrm{s}$]')
        ax.set_ylabel(r'Asymmetry $[D(-|\Delta t|) - D(+|\Delta t|)] \,/\, D(0)$ [-]')
        ax.set_title('After-passage vs pre-arrival slowdown asymmetry')
        ax.text(0.02, 0.97, r'$A>0$: wake after passage $\rightarrow$ (b) Disturbance' '\n'
                r'$A\approx 0$: time-symmetric $\rightarrow$ (a) Accumulation',
                transform=ax.transAxes, va='top', fontsize=9.2,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='azure', alpha=0.9, edgecolor='steelblue'))
        ax.legend(loc='lower right', fontsize=9.5)
        ax.grid(True)
        save_fig(fig, f"cargo_slowdown_asymmetry_vs_abs_delta_t{suffix}")
        plt.close(fig)
        
    print("\n=== Analysis Summary: Slowdown Depth vs Delta t ===")
    piv = df_sum.pivot(index='bead_name', columns='delta_t_sec', values='slowdown_depth')
    print(piv.round(3).to_string())
    print("\nAll time-lagged analyses completed successfully!")

if __name__ == "__main__":
    main()
