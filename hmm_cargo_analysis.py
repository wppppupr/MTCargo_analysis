"""
hmm_cargo_analysis.py

1次元対数速力 O_t = [ ln(v_t + \epsilon) ] を用いたガウス放出隠れマルコフモデル（Gaussian HMM）により、
貨物微粒子（蛍光ビーズ）の運動モード（Run / Tumble / 停滞等）を一括解析・可視化するスクリプトです。

全ビーズサイズ（0.63μm, 1.18μm, 3.37μm, 5.0μm, 7.24μm, 20μm）において、
1. 1次元観測量空間 ln(v+eps) での放出確率密度ヒストグラム & 混合ガウスフィット
2. 粒子軌跡の運動モード色分けプロット (Trajectory segmentation)
3. 瞬時速度・推定状態・事後確率の時系列同期プロット
4. 各運動モード（Run / Tumble）の持続時間分布 (PDF & CCDF)
5. 粒子径 vs 運動パラメータ（平均速度、速度幅、平均持続時間、状態占有率）のサマリープロット
6. 状態遷移確率行列のヒートマップ
7. 事後確率分布と確信度指標
8. 状態別 MSD 曲線 & べき乗則フィッティング (alpha, D)
9. 状態分離度 S_v = |mu_fast - mu_slow| / sqrt((sigma_fast^2 + sigma_slow^2)/2) vs 粒子径プロット
10. モデル選択基準 (BIC / AIC vs 状態数 K)
11. 統計サマリー CSV の出力
を行います。
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# 親ディレクトリのパス設定
current_dir = Path(__file__).parent.resolve()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from libs import hmm_cargo as hc

# スタイルの適用
style_path = current_dir / 'libs' / 'my_style.mplstyle'
if style_path.exists():
    try:
        plt.style.use(str(style_path))
        style_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    except Exception:
        style_colors = ['#882255', '#CC6677', '#DDCC77', '#999933', '#117733', '#44AA99']
else:
    style_colors = ['#882255', '#CC6677', '#DDCC77', '#999933', '#117733', '#44AA99']

# ビーズ条件設定
BEADS_INFO = [
    {"name": "beads06um", "diameter_um": 0.63, "marker": "^", "color": style_colors[0]},
    {"name": "beads1um",  "diameter_um": 1.18, "marker": "o", "color": style_colors[1]},
    {"name": "beads3um",  "diameter_um": 3.37, "marker": "d", "color": style_colors[2]},
    {"name": "beads5um",  "diameter_um": 5.00, "marker": "p", "color": style_colors[3]},
    {"name": "beads7um",  "diameter_um": 7.24, "marker": "h", "color": style_colors[4]},
    {"name": "beads20um", "diameter_um": 20.0, "marker": "s", "color": style_colors[5]},
]

POSSIBLE_ROOTS = [
    Path('/Volumes/data-1/Sasaki/MTsingleBeads'),
    Path('/Volumes/data-1/sasaki/MTsingleBeads'),
    Path('/Volumes/data/Sasaki/MTsingleBeads'),
    Path('/Volumes/data/sasaki/MTsingleBeads'),
]

# モード用の配色 (State 0: Tumble, State 1: Run)
STATE_COLORS = {
    0: '#d95f02',  # Tumble: オレンジ系
    1: '#1b9e77',  # Run: 青緑系
    2: '#7570b3',  # K=3の時の第3状態: 紫系
}
STATE_NAMES = {
    2: {0: 'Tumble / Pause', 1: 'Run'},
    3: {0: 'Tumble / Pause', 1: 'Intermediate', 2: 'Fast Run'}
}


def find_default_root() -> Path:
    for r in POSSIBLE_ROOTS:
        if r.exists():
            for b in ['beads1um', 'beads06um', 'beads3um', 'beads5um', 'beads7um', 'beads20um']:
                if (r / b).exists() and len(list((r / b).glob('*/*beads_tracks.csv'))) > 0:
                    return r
    for r in POSSIBLE_ROOTS:
        if r.exists():
            return r
    return POSSIBLE_ROOTS[0]


def find_experiment_dirs(root_dir: Path, bead_name: str) -> List[Path]:
    base = Path(root_dir) / bead_name
    if not base.exists():
        return []
    exp_dirs = []
    for p in sorted(base.glob("*/*")):
        if p.is_dir() and (p / "beads_tracks.csv").exists():
            exp_dirs.append(p)
    if not exp_dirs:
        for p in sorted(base.glob("*")):
            if p.is_dir() and (p / "beads_tracks.csv").exists():
                exp_dirs.append(p)
    return exp_dirs


def safe_save_csv(df: pd.DataFrame, target_path: Path, max_retries: int = 5):
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(max_retries):
        try:
            df.to_csv(target_path, index=False)
            break
        except Exception as e:
            if attempt == max_retries - 1:
                try:
                    csv_text = df.to_csv(index=False)
                    with open(str(target_path), 'w', encoding='utf-8') as f:
                        f.write(csv_text)
                    break
                except Exception:
                    print(f"[WARNING] Could not save {target_path}: {e}", flush=True)
                    break
            time.sleep(0.5)

    # Mirror to local workspace figure/hmm_1d
    local_dir = Path("/Users/sasakinozomu/code/MTCargo_analysis/figure/hmm_1d")
    local_dir.mkdir(parents=True, exist_ok=True)
    local_path = local_dir / target_path.name
    if local_path != target_path:
        try:
            df.to_csv(local_path, index=False)
        except Exception:
            pass


_TRACKS_CACHE = {}


def load_cached_tracks(csv_path: Path) -> Optional[pd.DataFrame]:
    str_path = str(csv_path)
    if str_path in _TRACKS_CACHE:
        return _TRACKS_CACHE[str_path]
    try:
        df = pd.read_csv(csv_path)
        if {'particle', 'frame', 'x', 'y'}.issubset(df.columns):
            _TRACKS_CACHE[str_path] = df[['particle', 'frame', 'x', 'y']].copy()
            return _TRACKS_CACHE[str_path]
    except Exception as e:
        print(f"[WARNING] Failed to load {csv_path}: {e}", flush=True)
    return None


def collect_bead_hmm_data(
    exp_dirs: List[Path],
    tau: int = 1,
    scale: float = 0.11,
    frame_interval: float = 4.0,
    epsilon: float = 1e-3,
) -> Tuple[np.ndarray, List[int], pd.DataFrame]:
    all_dfs = []
    particle_offset = 0

    for edir in exp_dirs:
        tracks_csv = edir / "beads_tracks.csv"
        if not tracks_csv.exists():
            continue
        df_tracks = load_cached_tracks(tracks_csv)
        if df_tracks is None or df_tracks.empty:
            continue

        df_copy = df_tracks.copy()
        if 'particle' not in df_copy.columns or 'frame' not in df_copy.columns:
            continue

        # 実験ディレクトリごとに粒子IDをユニーク化
        df_copy['particle'] = df_copy['particle'] + particle_offset
        particle_offset += int(df_copy['particle'].max()) + 1
        df_copy['exp_dir'] = edir.name
        all_dfs.append(df_copy)

    if not all_dfs:
        return np.empty((0, 1)), [], pd.DataFrame()

    df_combined = pd.concat(all_dfs, ignore_index=True)
    X, lengths, df_obs = hc.extract_hmm_features(
        df_combined,
        tau=tau,
        scale=scale,
        frame_interval=frame_interval,
        epsilon=epsilon,
    )
    return X, lengths, df_obs


def plot_trajectory_segmentation(
    fitted_results: Dict[str, dict],
    output_path: Path,
    n_components: int = 2,
):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for idx, binfo in enumerate(BEADS_INFO):
        ax = axes[idx]
        bname = binfo['name']
        dia = binfo['diameter_um']

        if bname not in fitted_results:
            ax.set_title(f"{dia:.2f} $\\mu$m (No data)")
            continue

        res = fitted_results[bname]
        df_obs = res['df_obs']
        pred_states = res['pred_states']

        if df_obs.empty:
            ax.set_title(f"{dia:.2f} $\\mu$m (No data)")
            continue

        part_counts = df_obs['particle'].value_counts()
        top_particles = part_counts.head(4).index.tolist()

        for pid in top_particles:
            p_mask = (df_obs['particle'] == pid).to_numpy()
            sub_p = df_obs[p_mask].sort_values(by='frame')
            p_states = pred_states[p_mask]

            x_pts = sub_p['x_um'].to_numpy()
            y_pts = sub_p['y_um'].to_numpy()
            frames = sub_p['frame'].to_numpy()

            if len(x_pts) < 3:
                continue

            for i in range(len(x_pts) - 1):
                if frames[i+1] != frames[i] + 1:
                    continue
                st = p_states[i]
                scolor = STATE_COLORS.get(st, f"C{st}")
                ax.plot(
                    [x_pts[i], x_pts[i+1]],
                    [y_pts[i], y_pts[i+1]],
                    color=scolor,
                    lw=2.0,
                    alpha=0.85,
                    solid_capstyle='round',
                )
            ax.plot(x_pts[0], y_pts[0], marker='o', markersize=4, color='black', alpha=0.7)

        ax.set_title(f"$d = {dia:.2f}\\,\\mu\\mathrm{{m}}$", fontsize=12, fontweight='bold')
        ax.set_aspect('equal', adjustable='datalim')
        ax.grid(True, linestyle='--', alpha=0.4)
        if idx >= 3:
            ax.set_xlabel(r"$x$ [$\mu\mathrm{m}$]", fontsize=11)
        if idx % 3 == 0:
            ax.set_ylabel(r"$y$ [$\mu\mathrm{m}$]", fontsize=11)

    legend_elements = [
        plt.Line2D([0], [0], color=STATE_COLORS.get(s, f"C{s}"), lw=3, label=STATE_NAMES.get(n_components, {}).get(s, f"State {s}"))
        for s in range(n_components)
    ]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=n_components, frameon=True, fontsize=11)
    fig.suptitle("HMM Decoded Trajectory Segmentation (Representative Tracks)", fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVED] {output_path}", flush=True)


def plot_timeseries_sync(
    fitted_results: Dict[str, dict],
    output_path: Path,
    target_bead: str = 'beads1um',
    n_components: int = 2,
    frame_interval: float = 4.0,
):
    if target_bead not in fitted_results:
        for b in BEADS_INFO:
            if b['name'] in fitted_results:
                target_bead = b['name']
                break
    if target_bead not in fitted_results:
        return

    res = fitted_results[target_bead]
    df_obs = res['df_obs']
    hmm_model = res['model']
    X = res['X']

    if df_obs.empty:
        return

    part_lens = df_obs['particle'].value_counts()
    best_pid = part_lens.index[0]
    df_p = df_obs[df_obs['particle'] == best_pid].sort_values(by='frame').copy()

    if len(df_p) < 15:
        return

    X_p = df_p[['obs_0']].to_numpy()
    if 'pred_state' in df_p.columns:
        pred_p = df_p['pred_state'].to_numpy()
    else:
        pred_p = hmm_model.predict(X_p)
    proba_p = hmm_model.predict_proba(X_p)

    time_sec = (df_p['frame'].to_numpy() - df_p['frame'].iloc[0]) * frame_interval

    fig, axes = plt.subplots(3, 1, figsize=(12, 7.5), sharex=True)

    # 1. 速度 v(t)
    axes[0].plot(time_sec, df_p['v'], color='#1f77b4', lw=1.8, marker='o', markersize=3, label='Instantaneous Speed $v(t)$')
    axes[0].set_ylabel(r"$v(t)$ [$\mu\mathrm{m/s}$]", fontsize=11)
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].legend(loc='upper right')

    # 2. 推定状態系列 S_t
    ax2 = axes[1]
    ax2.step(time_sec, pred_p, where='mid', color='black', lw=2, label='Decoded State (Viterbi)')
    ax2.set_ylabel("State", fontsize=11)
    ax2.set_yticks(range(n_components))
    ax2.set_yticklabels([STATE_NAMES.get(n_components, {}).get(s, f"S{s}") for s in range(n_components)])
    ax2.set_ylim(-0.3, n_components - 0.7)
    ax2.grid(True, linestyle='--', alpha=0.5)

    for t in range(len(time_sec) - 1):
        st = pred_p[t]
        ax2.axvspan(time_sec[t], time_sec[t+1], color=STATE_COLORS.get(st, f"C{st}"), alpha=0.25)

    # 3. 状態事後確率 P(S_t = k | O)
    ax3 = axes[2]
    for s in range(n_components):
        lbl = STATE_NAMES.get(n_components, {}).get(s, f"State {s}")
        ax3.plot(time_sec, proba_p[:, s], color=STATE_COLORS.get(s, f"C{s}"), lw=1.8, label=r"$P(S_t = \text{" + lbl + r"})$")
    ax3.set_ylabel("Posterior Prob.", fontsize=11)
    ax3.set_xlabel("Time [s]", fontsize=11)
    ax3.set_ylim(-0.05, 1.05)
    ax3.grid(True, linestyle='--', alpha=0.5)
    ax3.legend(loc='center right')

    fig.suptitle(f"Synchronized Time Series & 1D HMM State Decoding ({target_bead}, Particle #{best_pid})", fontsize=14, fontweight='bold')
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVED] {output_path}", flush=True)


def plot_dwell_time_distributions(
    fitted_results: Dict[str, dict],
    output_path: Path,
    n_components: int = 2,
    frame_interval: float = 4.0,
):
    fig, axes = plt.subplots(n_components, 2, figsize=(12, 4.5 * n_components))
    if n_components == 1:
        axes = np.array([axes])

    for s in range(n_components):
        s_lbl = STATE_NAMES.get(n_components, {}).get(s, f"State {s}")
        scolor = STATE_COLORS.get(s, f"C{s}")
        ax_pdf = axes[s, 0]
        ax_ccdf = axes[s, 1]

        for binfo in BEADS_INFO:
            bname = binfo['name']
            dia = binfo['diameter_um']
            col = binfo['color']
            mrk = binfo['marker']

            if bname not in fitted_results:
                continue

            dwell_fits = fitted_results[bname].get('dwell_fits', {})
            fit_res = dwell_fits.get(s, {})
            ccdf_res = fitted_results[bname].get('dwell_ccdf_fits', {}).get(s, {})
            if not fit_res:
                continue

            # PDF プロット
            t_pts = fit_res.get('t_centers', [])
            p_vals = fit_res.get('p_vals', [])
            p_errs = fit_res.get('p_errs', [])
            t_fit = fit_res.get('t_fit', [])
            p_fit = fit_res.get('p_fit', [])
            tau_fit = fit_res.get('tau_fit_s', np.nan)

            if len(t_pts) > 0:
                lbl_pdf = f"$d={dia:.2f}\\,\\mu\\mathrm{{m}}$"
                ax_pdf.errorbar(t_pts, p_vals, yerr=p_errs, marker=mrk, color=col, linestyle='none', label=lbl_pdf, markersize=4.5, capsize=2)
                if len(t_fit) > 0 and not np.isnan(tau_fit):
                    ax_pdf.plot(t_fit, p_fit, color=col, linestyle=':', lw=1.0, alpha=0.6)

            # CCDF プロット (実測点 + CCDF 指数フィッティング線)
            if ccdf_res and len(ccdf_res.get('t_unique', [])) > 0:
                t_u = ccdf_res['t_unique']
                c_u = ccdf_res['ccdf_unique']
                t_cfit = ccdf_res['t_fit']
                c_cfit = ccdf_res['ccdf_fit']
                tau_ccdf = ccdf_res['tau_fit_s']
                r2_ccdf = ccdf_res.get('r2_log', np.nan)

                r2_str = f", $R^2={r2_ccdf:.2f}$" if not np.isnan(r2_ccdf) else ""
                lbl_ccdf = f"$d={dia:.2f}\\,\\mu\\mathrm{{m}}$ ($\\tau_{{\\mathrm{{CCDF}}}}={tau_ccdf:.1f}\\,\\mathrm{{s}}${r2_str})"
                ax_ccdf.plot(t_u, c_u, marker=mrk, color=col, linestyle='none', label=lbl_ccdf, markersize=4.0, alpha=0.85)
                if len(t_cfit) > 0 and not np.isnan(tau_ccdf):
                    ax_ccdf.plot(t_cfit, c_cfit, color=col, linestyle='--', lw=1.4, alpha=0.85)
            else:
                dwells = fitted_results[bname]['dwell_times'].get(s, [])
                if len(dwells) > 0:
                    arr = np.asarray(dwells)
                    sorted_d = np.sort(arr)
                    ccdf = 1.0 - (np.arange(1, len(sorted_d) + 1) - 0.5) / len(sorted_d)
                    ax_ccdf.plot(sorted_d, ccdf, marker=mrk, color=col, label=f"$d={dia:.2f}\\,\\mu\\mathrm{{m}}$", lw=1.5, markersize=3.5)

        ax_pdf.set_yscale('log')
        ax_pdf.set_xlabel("Dwell Time $t$ [s]", fontsize=11)
        ax_pdf.set_ylabel(f"PDF $P(t)$ ({s_lbl})", fontsize=11)
        ax_pdf.set_title(f"(a) Dwell Time PDF: {s_lbl}", fontsize=12, fontweight='bold')
        ax_pdf.grid(True, linestyle='--', alpha=0.5)
        ax_pdf.legend(loc='best', fontsize=8.0, frameon=True)

        ax_ccdf.set_yscale('log')
        ax_ccdf.set_xlabel("Dwell Time $t$ [s]", fontsize=11)
        ax_ccdf.set_ylabel(f"CCDF $P(T \geq t)$ ({s_lbl})", fontsize=11)
        ax_ccdf.set_title(f"(b) Dwell Time CCDF with Exponential Fits: {s_lbl}", fontsize=12, fontweight='bold')
        ax_ccdf.grid(True, linestyle='--', alpha=0.5)
        ax_ccdf.legend(loc='best', fontsize=7.8, frameon=True)

    fig.suptitle(r"Dwell Time Distributions & CCDF Exponential Fits ($P(T \geq t) = \exp(-t/\tau_{\mathrm{CCDF}})$)", fontsize=14, fontweight='bold')
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVED] {output_path}", flush=True)


def plot_dwell_time_ccdf_fit_6panel(
    fitted_results: Dict[str, dict],
    output_path: Path,
    n_components: int = 2,
    frame_interval: float = 4.0,
):
    """
    各粒子径（6パネル）における Run と Tumble の Dwell time CCDF P(T >= t) の指数分布フィッティングプロット。
    C(t) = exp(-t / tau) のゼロ切片線形回帰。
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5))
    axes = axes.flatten()

    for idx, binfo in enumerate(BEADS_INFO):
        ax = axes[idx]
        bname = binfo['name']
        dia = binfo['diameter_um']

        if bname not in fitted_results:
            ax.set_title(f"$d = {dia:.2f}\\,\\mu\\mathrm{{m}}$ (No data)", fontsize=12)
            ax.axis('off')
            continue

        res = fitted_results[bname]
        dwell_ccdf_fits = res.get('dwell_ccdf_fits', {})

        has_data = False
        max_t_plot = 0.0

        for s in range(n_components):
            s_lbl = STATE_NAMES.get(n_components, {}).get(s, f"State {s}")
            scolor = STATE_COLORS.get(s, f"C{s}")
            mrk = 'o' if s == 0 else 's'
            fit_res = dwell_ccdf_fits.get(s, {})

            if not fit_res or len(fit_res.get('t_unique', [])) == 0:
                continue

            has_data = True
            t_pts = fit_res['t_unique']
            c_pts = fit_res['ccdf_unique']
            t_fit = fit_res['t_fit']
            c_fit = fit_res['ccdf_fit']
            tau_val = fit_res['tau_fit_s']
            tau_err = fit_res['tau_err_s']
            r2_val = fit_res.get('r2_log', np.nan)

            max_t_plot = max(max_t_plot, np.max(t_pts) if len(t_pts) > 0 else 0.0)

            # データ点（経験的 CCDF）
            ax.plot(
                t_pts, c_pts,
                marker=mrk, color=scolor, linestyle='none',
                markersize=5.0, alpha=0.85, zorder=3,
                label=f"{s_lbl}: Data ($N={fit_res['count']}$)",
            )

            # フィッティング曲線 C(t) = exp(-t/tau)
            if len(t_fit) > 0 and not np.isnan(tau_val):
                err_str = f"\\pm {tau_err:.1f}" if not np.isnan(tau_err) else ""
                r2_str = f", $R^2={r2_val:.2f}$" if not np.isnan(r2_val) else ""
                fit_label = f"{s_lbl}: Fit ($\\tau={tau_val:.1f}{err_str}\\,\\mathrm{{s}}${r2_str})"
                ax.plot(
                    t_fit, c_fit,
                    color=scolor, linestyle='--' if s == 0 else '-',
                    linewidth=2.0, alpha=0.9, zorder=2,
                    label=fit_label,
                )

        if not has_data:
            ax.set_title(f"$d = {dia:.2f}\\,\\mu\\mathrm{{m}}$ (Insufficient data)", fontsize=12)
            continue

        ax.set_yscale('log')
        ax.set_ylim(1e-3, 1.5)
        #ax.set_xlim(0, 200)
        ax.set_title(f"$d = {dia:.2f}\\,\\mu\\mathrm{{m}}$", fontsize=12, fontweight='bold')
        ax.grid(True, which="both", linestyle='--', alpha=0.4)
        ax.legend(loc='upper right', fontsize=8.5, frameon=True, framealpha=0.92)

        if idx >= 3:
            ax.set_xlabel(r"Dwell Time $t$ [s]", fontsize=11)
        if idx % 3 == 0:
            ax.set_ylabel(r"CCDF $P(T \geq t)$", fontsize=11)

    fig.suptitle(
        r"Dwell Time Complementary Cumulative Distribution Functions (CCDF) with Exponential Fits: $P(T \geq t) = A e^{-t/\tau}$",
        fontsize=13.5,
        fontweight='bold',
    )
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    png_path = output_path.with_suffix('.png')
    if png_path != output_path:
        fig.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVED] {output_path} (and {png_path})", flush=True)


def plot_dwell_time_pdf_fit_6panel(
    fitted_results: Dict[str, dict],
    output_path: Path,
    n_components: int = 2,
    frame_interval: float = 4.0,
):
    """
    各粒子径（6パネル）における Run と Tumble の Dwell time PDF P(t) の指数分布フィッティングプロット。
    y 軸を対数（ln P(t)）にとることで、指数減衰 P(t) = A exp(-t/tau) が直線として可視化される。
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5))
    axes = axes.flatten()

    for idx, binfo in enumerate(BEADS_INFO):
        ax = axes[idx]
        bname = binfo['name']
        dia = binfo['diameter_um']

        if bname not in fitted_results:
            ax.set_title(f"$d = {dia:.2f}\\,\\mu\\mathrm{{m}}$ (No data)", fontsize=12)
            ax.axis('off')
            continue

        res = fitted_results[bname]
        dwell_fits = res.get('dwell_fits', {})

        has_data = False
        max_t_plot = 0.0

        for s in range(n_components):
            s_lbl = STATE_NAMES.get(n_components, {}).get(s, f"State {s}")
            scolor = STATE_COLORS.get(s, f"C{s}")
            mrk = 'o' if s == 0 else 's'
            fit_res = dwell_fits.get(s, {})

            if not fit_res or len(fit_res.get('t_centers', [])) == 0:
                continue

            has_data = True
            t_pts = fit_res['t_centers']
            p_pts = fit_res['p_vals']
            p_errs = fit_res['p_errs']
            t_fit = fit_res['t_fit']
            p_fit = fit_res['p_fit']
            tau_val = fit_res['tau_fit_s']
            tau_err = fit_res['tau_err_s']
            r2_val = fit_res.get('r2_log', np.nan)

            max_t_plot = max(max_t_plot, np.max(t_pts) if len(t_pts) > 0 else 0.0)

            # データ点（エラーバー付き）
            ax.errorbar(
                t_pts, p_pts, yerr=p_errs,
                fmt=mrk, color=scolor, ecolor=scolor, elinewidth=1.2,
                capsize=3, markersize=5.5, alpha=0.9, zorder=3,
                label=f"{s_lbl}: Data ($N={fit_res['count']}$)",
            )

            # フィッティング直線 P(t) = A exp(-t/tau)
            if len(t_fit) > 0 and not np.isnan(tau_val):
                err_str = f"\\pm {tau_err:.1f}" if not np.isnan(tau_err) else ""
                r2_str = f", $R^2_{{\\ln}}={r2_val:.2f}$" if not np.isnan(r2_val) else ""
                fit_label = f"{s_lbl}: Fit ($\\tau={tau_val:.1f}{err_str}\\,\\mathrm{{s}}${r2_str})"
                ax.plot(
                    t_fit, p_fit,
                    color=scolor, linestyle='--' if s == 0 else '-',
                    linewidth=2.0, alpha=0.85, zorder=2,
                    label=fit_label,
                )

        if not has_data:
            ax.set_title(f"$d = {dia:.2f}\\,\\mu\\mathrm{{m}}$ (Insufficient data)", fontsize=12)
            continue

        ax.set_yscale('log')
        #ax.set_xlim(0, 200)
        ax.set_title(f"$d = {dia:.2f}\\,\\mu\\mathrm{{m}}$", fontsize=12, fontweight='bold')
        ax.grid(True, which="both", linestyle='--', alpha=0.4)
        ax.legend(loc='upper right', fontsize=8.5, frameon=True, framealpha=0.92)

        if idx >= 3:
            ax.set_xlabel(r"Dwell Time $t$ [s]", fontsize=11)
        if idx % 3 == 0:
            ax.set_ylabel(r"Probability Density $P(t)$ [$\mathrm{s}^{-1}$]", fontsize=11)

    fig.suptitle(
        r"Dwell Time Probability Density Functions $P(t)$ with Exponential Fits: $\ln P(t) = \ln A - t/\tau$",
        fontsize=13.5,
        fontweight='bold',
    )
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    png_path = output_path.with_suffix('.png')
    if png_path != output_path:
        fig.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVED] {output_path} (and {png_path})", flush=True)
    print(f"[SAVED] {output_path} (and {png_path})", flush=True)


def plot_summary_vs_diameter(
    df_summary: pd.DataFrame,
    output_path: Path,
    n_components: int = 2,
):
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # (0, 0): 平均速度 vs 粒子径
    ax = axes[0, 0]
    for s in range(n_components):
        df_s = df_summary[df_summary['state'] == s].sort_values(by='diameter_um')
        lbl = STATE_NAMES.get(n_components, {}).get(s, f"State {s}")
        ax.plot(
            df_s['diameter_um'],
            df_s['mean_speed_model_um_s'],
            marker='o',
            lw=2.0,
            color=STATE_COLORS.get(s, f"C{s}"),
            label=lbl,
        )
    ax.set_xscale('log')
    ax.set_xlabel(r"Cargo Diameter $d$ [$\mu\mathrm{m}$]", fontsize=11)
    ax.set_ylabel(r"Mean Speed $\langle v \rangle$ [$\mu\mathrm{m/s}$]", fontsize=11)
    ax.set_title("(a) State Speed vs Particle Size", fontsize=12, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='best', frameon=True)

    # (0, 1): 対数速度標準偏差 σ_ln_v
    ax = axes[0, 1]
    for s in range(n_components):
        df_s = df_summary[df_summary['state'] == s].sort_values(by='diameter_um')
        lbl = STATE_NAMES.get(n_components, {}).get(s, f"State {s}")
        ax.plot(
            df_s['diameter_um'],
            df_s['std_log_v'],
            marker='s',
            lw=2.0,
            color=STATE_COLORS.get(s, f"C{s}"),
            label=lbl,
        )
    ax.set_xscale('log')
    ax.set_xlabel(r"Cargo Diameter $d$ [$\mu\mathrm{m}$]", fontsize=11)
    ax.set_ylabel(r"Log Speed Std Dev $\sigma_{\ln v}$", fontsize=11)
    ax.set_title(r"(b) State Log-Speed Width $\sigma_{\ln v}$ vs Particle Size", fontsize=12, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='best', frameon=True)

    # (1, 0): 平均持続時間 vs 粒子径
    ax = axes[1, 0]
    for s in range(n_components):
        df_s = df_summary[df_summary['state'] == s].sort_values(by='diameter_um')
        lbl = STATE_NAMES.get(n_components, {}).get(s, f"State {s}")
        scolor = STATE_COLORS.get(s, f"C{s}")
        tau_c = df_s['tau_ccdf_s'] if 'tau_ccdf_s' in df_s.columns else df_s['mean_dwell_emp_s']
        err_c = df_s['tau_ccdf_err_s'] if 'tau_ccdf_err_s' in df_s.columns else df_s['dwell_err_s']
        ax.errorbar(
            df_s['diameter_um'],
            tau_c,
            yerr=err_c,
            marker='^',
            lw=2.0,
            capsize=4,
            color=scolor,
            label=f"{lbl} (CCDF Fit $\\tau$)",
        )
        ax.plot(
            df_s['diameter_um'],
            df_s['theoretical_dwell_time_s'],
            linestyle='--',
            color=scolor,
            alpha=0.6,
            label=f"{lbl} (Markov $\\tau_{{\\mathrm{{theo}}}}$)",
        )
    ax.set_xscale('log')
    ax.set_xlabel(r"Cargo Diameter $d$ [$\mu\mathrm{m}$]", fontsize=11)
    ax.set_ylabel(r"Relaxation Time $\tau$ [s]", fontsize=11)
    ax.set_title(r"(c) State Relaxation Time $\tau_{\mathrm{CCDF}}$ vs Particle Size", fontsize=12, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='best', frameon=True, fontsize=8.5)

    # (1, 1): 定常占有率 vs 粒子径
    ax = axes[1, 1]
    for s in range(n_components):
        df_s = df_summary[df_summary['state'] == s].sort_values(by='diameter_um')
        lbl = STATE_NAMES.get(n_components, {}).get(s, f"State {s}")
        ax.plot(
            df_s['diameter_um'],
            df_s['stationary_prob'] * 100.0,
            marker='D',
            lw=2.0,
            color=STATE_COLORS.get(s, f"C{s}"),
            label=lbl,
        )
    ax.set_xscale('log')
    ax.set_ylim(0, 100)
    ax.set_xlabel(r"Cargo Diameter $d$ [$\mu\mathrm{m}$]", fontsize=11)
    ax.set_ylabel(r"Stationary Population $\pi_i$ [%]", fontsize=11)
    ax.set_title("(d) Stationary State Population vs Particle Size", fontsize=12, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='best', frameon=True)

    fig.suptitle("Summary of 1D Motion State Parameters vs Particle Diameter", fontsize=14, fontweight='bold')
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVED] {output_path}", flush=True)


def plot_transition_matrices_6panel(
    fitted_results: Dict[str, dict],
    output_path: Path,
    n_components: int = 2,
):
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    axes = axes.flatten()

    for idx, binfo in enumerate(BEADS_INFO):
        ax = axes[idx]
        bname = binfo['name']
        dia = binfo['diameter_um']

        if bname not in fitted_results:
            ax.set_title(f"{dia:.2f} $\\mu$m (No data)")
            continue

        hmm_model = fitted_results[bname]['model']
        A = hmm_model.model.transmat_

        im = ax.imshow(A, vmin=0.0, vmax=1.0, cmap='Blues')
        ax.set_title(f"$d = {dia:.2f}\\,\\mu\\mathrm{{m}}$", fontsize=12, fontweight='bold')
        ax.set_xticks(range(n_components))
        ax.set_yticks(range(n_components))
        ax.set_xticklabels([STATE_NAMES.get(n_components, {}).get(s, f"S{s}") for s in range(n_components)])
        ax.set_yticklabels([STATE_NAMES.get(n_components, {}).get(s, f"S{s}") for s in range(n_components)])

        for i in range(n_components):
            for j in range(n_components):
                val = A[i, j]
                color = 'white' if val > 0.55 else 'black'
                ax.text(j, i, f"{val:.3f}", ha='center', va='center', color=color, fontweight='bold', fontsize=11)

        if idx >= 3:
            ax.set_xlabel("To State $S_{t+1}$", fontsize=11)
        if idx % 3 == 0:
            ax.set_ylabel("From State $S_t$", fontsize=11)

    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Transition Probability $A_{ij}$")
    fig.suptitle("HMM State Transition Probability Matrices $A_{ij}$", fontsize=14, fontweight='bold')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVED] {output_path}", flush=True)


def plot_posterior_distributions_6panel(
    fitted_results: Dict[str, dict],
    output_path: Path,
    n_components: int = 2,
):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), sharex=True, sharey=True)
    axes = axes.flatten()

    for idx, binfo in enumerate(BEADS_INFO):
        ax = axes[idx]
        bname = binfo['name']
        dia = binfo['diameter_um']

        if bname not in fitted_results:
            ax.set_visible(False)
            continue

        proba = fitted_results[bname]['proba']
        if len(proba) == 0:
            ax.set_visible(False)
            continue

        bins = np.linspace(0.0, 1.0, 31)
        for s in range(n_components):
            lbl = STATE_NAMES.get(n_components, {}).get(s, f"State {s}")
            col = STATE_COLORS.get(s, f"C{s}")
            ax.hist(
                proba[:, s],
                bins=bins,
                density=True,
                alpha=0.45,
                color=col,
                edgecolor=col,
                label=f"$P(S_t = \\text{{{lbl}}})$",
                lw=1.5,
            )

        conf = fitted_results[bname]['conf_stats']
        mean_c = conf.get('mean_confidence', np.nan)
        high_r = conf.get('high_conf_ratio_80', np.nan)
        ax.text(
            0.5, 0.85,
            f"Mean Conf: {mean_c:.2f}\n$P\\geq 0.8$: {high_r*100:.1f}%",
            transform=ax.transAxes,
            ha='center',
            fontsize=9.5,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85, edgecolor='gray')
        )

        ax.set_title(f"$d = {dia:.2f}\\,\\mu\\mathrm{{m}}$ (N={len(proba):,})", fontsize=12, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.set_xlim(-0.02, 1.02)
        if idx >= 3:
            ax.set_xlabel("Posterior Probability $P(S_t = k \\mid \\mathbf{O})$", fontsize=11)
        if idx % 3 == 0:
            ax.set_ylabel("Probability Density", fontsize=11)

    axes[0].legend(loc='upper right', fontsize=9, frameon=True)
    fig.suptitle("HMM Posterior Probability Distributions $P(S_t = k \\mid \\mathbf{O})$", fontsize=14, fontweight='bold')
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVED] {output_path}", flush=True)


def plot_confidence_vs_diameter(
    df_conf: pd.DataFrame,
    output_path: Path,
):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # 1. Mean Confidence
    axes[0].plot(df_conf['diameter_um'], df_conf['mean_confidence'], marker='o', lw=2.2, color='#1f77b4')
    axes[0].set_xscale('log')
    axes[0].set_ylim(0.7, 1.02)
    axes[0].set_xlabel(r"Cargo Particle Diameter $d$ [$\mu\mathrm{m}$]", fontsize=11)
    axes[0].set_ylabel(r"Mean Confidence $\langle \max_k P(S_t=k) \rangle$", fontsize=11)
    axes[0].set_title("(a) Mean Classification Confidence", fontsize=12, fontweight='bold')
    axes[0].grid(True, linestyle='--', alpha=0.5)

    # 2. High Confidence Ratio
    axes[1].plot(df_conf['diameter_um'], df_conf['high_conf_ratio_80'] * 100, marker='s', lw=2.2, color='#2ca02c')
    axes[1].set_xscale('log')
    axes[1].set_ylim(60, 102)
    axes[1].set_xlabel(r"Cargo Particle Diameter $d$ [$\mu\mathrm{m}$]", fontsize=11)
    axes[1].set_ylabel(r"High Confidence Steps ($P \geq 0.8$) [%]", fontsize=11)
    axes[1].set_title("(b) Unambiguous State Ratio", fontsize=12, fontweight='bold')
    axes[1].grid(True, linestyle='--', alpha=0.5)

    # 3. Normalized Entropy
    axes[2].plot(df_conf['diameter_um'], df_conf['norm_entropy'], marker='^', lw=2.2, color='#d62728')
    axes[2].set_xscale('log')
    axes[2].set_ylim(0.0, 0.5)
    axes[2].set_xlabel(r"Cargo Particle Diameter $d$ [$\mu\mathrm{m}$]", fontsize=11)
    axes[2].set_ylabel(r"Normalized Entropy $H / \ln K$", fontsize=11)
    axes[2].set_title("(c) Classification Uncertainty (Entropy)", fontsize=12, fontweight='bold')
    axes[2].grid(True, linestyle='--', alpha=0.5)

    fig.suptitle("HMM State Classification Confidence & Entropy vs Cargo Diameter", fontsize=14, fontweight='bold')
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVED] {output_path}", flush=True)


def plot_state_dependent_msd_6panel(
    fitted_results: Dict[str, dict],
    output_path: Path,
    n_components: int = 2,
):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True, sharey=True)
    axes = axes.flatten()

    for idx, binfo in enumerate(BEADS_INFO):
        ax = axes[idx]
        bname = binfo['name']
        dia = binfo['diameter_um']

        if bname not in fitted_results:
            ax.set_visible(False)
            continue

        res = fitted_results[bname]
        df_msd = res['df_msd']
        df_fits = res['df_msd_fits']

        if df_msd.empty:
            ax.set_visible(False)
            continue

        for s in range(n_components):
            sub_m = df_msd[df_msd['state'] == s].sort_values(by='lag_time_s')
            if sub_m.empty:
                continue

            lbl = STATE_NAMES.get(n_components, {}).get(s, f"State {s}")
            col = STATE_COLORS.get(s, f"C{s}")
            mrk = 'o' if s == 0 else 's'

            fit_row = df_fits[df_fits['state'] == s]
            if not fit_row.empty and not np.isnan(fit_row.iloc[0]['alpha']):
                alpha_val = fit_row.iloc[0]['alpha']
                lbl_with_alpha = f"{lbl} ($\\alpha={alpha_val:.2f}$)"
            else:
                lbl_with_alpha = lbl

            valid_pts = sub_m[sub_m['msd_um2'] > 0]
            ax.errorbar(
                valid_pts['lag_time_s'],
                valid_pts['msd_um2'],
                yerr=valid_pts['msd_sem_um2'],
                marker=mrk,
                markersize=4,
                lw=1.8,
                color=col,
                capsize=2,
                label=lbl_with_alpha,
            )

            # フィット線
            if not fit_row.empty and not np.isnan(fit_row.iloc[0]['alpha']):
                alpha = fit_row.iloc[0]['alpha']
                D_app = fit_row.iloc[0]['D_apparent_um2_s']
                t_fit = np.logspace(np.log10(valid_pts['lag_time_s'].min()), np.log10(valid_pts['lag_time_s'].max()), 50)
                msd_fit = 4.0 * D_app * (t_fit ** alpha)
                ax.plot(t_fit, msd_fit, color=col, linestyle='--', lw=1.2, alpha=0.8)

        # All (全体のMSD)
        sub_all = df_msd[df_msd['state'] == -1].sort_values(by='lag_time_s')
        if not sub_all.empty:
            valid_all = sub_all[sub_all['msd_um2'] > 0]
            fit_all = df_fits[df_fits['state'] == -1]
            if not fit_all.empty and not np.isnan(fit_all.iloc[0]['alpha']):
                alpha_all = fit_all.iloc[0]['alpha']
                lbl_all = f"All ($\\alpha={alpha_all:.2f}$)"
            else:
                lbl_all = "All"
            ax.plot(valid_all['lag_time_s'], valid_all['msd_um2'], color='gray', linestyle=':', lw=1.5, label=lbl_all)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title(f"$d = {dia:.2f}\\,\\mu\\mathrm{{m}}$", fontsize=12, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend(loc='lower right', fontsize=8.5, frameon=True)

        if idx >= 3:
            ax.set_xlabel(r"Lag Time $\Delta t$ [s]", fontsize=11)
        if idx % 3 == 0:
            ax.set_ylabel(r"MSD $\langle \Delta r^2 \rangle$ [$\mu\mathrm{m}^2$]", fontsize=11)

    fig.suptitle("State-Dependent Mean Squared Displacement (MSD) of HMM Motion Modes", fontsize=14, fontweight='bold')
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVED] {output_path}", flush=True)


def plot_state_msd_params_vs_diameter(
    df_fits: pd.DataFrame,
    output_path: Path,
    n_components: int = 2,
):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # 1. 異常拡散指数 alpha
    ax0 = axes[0]
    ax0.axhline(1.0, color='gray', linestyle=':', lw=1.2, label='Normal Diffusion ($\\alpha=1$)')
    ax0.axhline(2.0, color='silver', linestyle=':', lw=1.2, label='Ballistic ($\\alpha=2$)')

    for s in range(n_components):
        sub_f = df_fits[df_fits['state'] == s].sort_values(by='diameter_um')
        lbl = STATE_NAMES.get(n_components, {}).get(s, f"State {s}")
        col = STATE_COLORS.get(s, f"C{s}")
        ax0.errorbar(
            sub_f['diameter_um'],
            sub_f['alpha'],
            yerr=sub_f['alpha_err'],
            marker='o' if s == 0 else 's',
            lw=2.0,
            capsize=4,
            color=col,
            label=lbl,
        )

    sub_all = df_fits[df_fits['state'] == -1].sort_values(by='diameter_um')
    if not sub_all.empty:
        ax0.plot(sub_all['diameter_um'], sub_all['alpha'], marker='^', lw=1.8, color='gray', linestyle='--', label='All Tracks')

    ax0.set_xscale('log')
    ax0.set_xlabel(r"Cargo Particle Diameter $d$ [$\mu\mathrm{m}$]", fontsize=11)
    ax0.set_ylabel(r"Anomalous Diffusion Exponent $\alpha$", fontsize=11)
    ax0.set_title(r"(a) Motion Mode $\alpha$ vs Particle Diameter", fontsize=12, fontweight='bold')
    ax0.set_ylim(-0.1, 2.2)
    ax0.grid(True, linestyle='--', alpha=0.5)
    ax0.legend(loc='lower left', fontsize=9, frameon=True)

    # 2. 見かけの拡散係数 D
    ax1 = axes[1]
    for s in range(n_components):
        sub_f = df_fits[df_fits['state'] == s].sort_values(by='diameter_um')
        lbl = STATE_NAMES.get(n_components, {}).get(s, f"State {s}")
        col = STATE_COLORS.get(s, f"C{s}")
        ax1.plot(
            sub_f['diameter_um'],
            sub_f['D_apparent_um2_s'],
            marker='o' if s == 0 else 's',
            lw=2.0,
            color=col,
            label=lbl,
        )

    if not sub_all.empty:
        ax1.plot(sub_all['diameter_um'], sub_all['D_apparent_um2_s'], marker='^', lw=1.8, color='gray', linestyle='--', label='All Tracks')

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel(r"Cargo Particle Diameter $d$ [$\mu\mathrm{m}$]", fontsize=11)
    ax1.set_ylabel(r"Apparent Diffusion Coeff. $D$ [$\mu\mathrm{m}^2/\mathrm{s}^\alpha$]", fontsize=11)
    ax1.set_title(r"(b) Apparent Diffusion Coefficient $D$ vs Diameter", fontsize=12, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='upper right', fontsize=9, frameon=True)

    fig.suptitle("HMM State-Dependent Transport Dynamics vs Cargo Diameter", fontsize=14, fontweight='bold')
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVED] {output_path}", flush=True)


def plot_run_abp_msd_fit_6panel(
    fitted_results: Dict[str, dict],
    output_path: Path,
):
    """
    Run 状態の MSD 曲線に対してアクティブブラウニアン粒子 (ABP) 理論式
        <Δr^2(t)> = 4 D_t t + 2 v_0^2 \tau_r [ t - \tau_r (1 - e^{-t/\tau_r}) ]
    をフィッティングした結果の6パネルプロット。
    """
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.5), sharex=True, sharey=True)
    axes = axes.flatten()

    for idx, binfo in enumerate(BEADS_INFO):
        ax = axes[idx]
        bname = binfo['name']
        dia = binfo['diameter_um']

        if bname not in fitted_results or 'abp_fit' not in fitted_results[bname]:
            ax.set_visible(False)
            continue

        res = fitted_results[bname]
        df_msd = res.get('df_msd', pd.DataFrame())
        abp_res = res.get('abp_fit', {})

        if df_msd.empty or not abp_res:
            ax.set_visible(False)
            continue

        sub_run = df_msd[df_msd['state'] == 1].sort_values(by='lag_time_s')
        valid_pts = sub_run[(sub_run['msd_um2'] > 0) & np.isfinite(sub_run['msd_um2'])]

        if valid_pts.empty:
            ax.set_visible(False)
            continue

        # 実測データ点
        ax.errorbar(
            valid_pts['lag_time_s'],
            valid_pts['msd_um2'],
            yerr=valid_pts['msd_sem_um2'],
            fmt='s',
            markersize=5,
            lw=2.0,
            color='#1b9e77',
            ecolor='#1b9e77',
            capsize=3,
            label=r'Run MSD Data $\langle \Delta r^2 \rangle_{\mathrm{Run}}$',
            zorder=4,
        )

        # ABP フィッティング曲線
        fit_t = abp_res.get('fit_t', np.array([]))
        fit_msd = abp_res.get('fit_msd', np.array([]))
        Dt = abp_res.get('Dt_um2_s', np.nan)
        tau_r = abp_res.get('tau_r_s', np.nan)
        v0 = abp_res.get('v0_um_s', np.nan)
        Deff = abp_res.get('D_eff_um2_s', np.nan)
        r2 = abp_res.get('r_squared', np.nan)

        if len(fit_t) > 0 and not np.isnan(Dt) and not np.isnan(tau_r):
            ax.plot(
                fit_t, fit_msd,
                color='#d95f02', lw=2.2, linestyle='-',
                label=r'Active Brownian Fit',
                zorder=3,
            )

            # 短時間弾道漸近線: 4 D_t t + v_0^2 t^2
            msd_short = 4.0 * Dt * fit_t + (v0**2) * (fit_t**2)
            ax.plot(fit_t, msd_short, color='#7570b3', linestyle=':', lw=1.3, alpha=0.75, label=r'Ballistic ($v_0^2 t^2$)')

            # 長時間拡散漸近線: 4 D_eff t
            msd_long = 4.0 * Deff * fit_t
            ax.plot(fit_t, msd_long, color='#666666', linestyle='--', lw=1.3, alpha=0.75, label=r'Diffusive ($4 D_{\mathrm{eff}} t$)')

            # パラメータテキスト
            r2_str = f"$R^2 = {r2:.3f}$" if not np.isnan(r2) else ""
            txt = (
                f"$v_0 = {v0:.3f}\\,\\mu\\mathrm{{m/s}}$\n"
                f"$D_t = {Dt:.4f}\\,\\mu\\mathrm{{m^2/s}}$\n"
                f"$\\tau_r = {tau_r:.1f}\\,\\mathrm{{s}}$\n"
                f"$D_{{\\mathrm{{eff}}}} = {Deff:.3f}\\,\\mu\\mathrm{{m^2/s}}$\n"
                f"{r2_str}"
            )
            ax.text(
                0.05, 0.95, txt,
                transform=ax.transAxes,
                va='top', ha='left',
                fontsize=8.5,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.88, edgecolor='#cccccc'),
                zorder=5,
            )

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title(f"$d = {dia:.2f}\\,\\mu\\mathrm{{m}}$", fontsize=12, fontweight='bold')
        ax.grid(True, which="both", linestyle='--', alpha=0.4)
        ax.legend(loc='lower right', fontsize=7.8, frameon=True, framealpha=0.9)

        if idx >= 3:
            ax.set_xlabel(r"Lag Time $\Delta t$ [s]", fontsize=11)
        if idx % 3 == 0:
            ax.set_ylabel(r"MSD $\langle \Delta r^2 \rangle$ [$\mu\mathrm{m}^2$]", fontsize=11)

    fig.suptitle(
        r"Run Mode MSD Fitted by Active Brownian Particle Model: $\langle \Delta r^2(t) \rangle = 4 D_t t + 2 v_0^2 \tau_r [ t - \tau_r (1 - e^{-t/\tau_r}) ]$",
        fontsize=13.0,
        fontweight='bold',
    )
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    png_path = output_path.with_suffix('.png')
    if png_path != output_path:
        fig.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVED] {output_path} (and {png_path})", flush=True)


def plot_abp_parameters_vs_diameter(
    df_abp_summary: pd.DataFrame,
    df_all_summary: pd.DataFrame,
    output_path: Path,
):
    """
    アクティブブラウニアンモデルから推定されたパラメータ (tau_r, D_t, D_eff, lambda_p) vs 粒子径のサマリープロット。
    """
    if df_abp_summary.empty:
        return

    df_abp = df_abp_summary.sort_values(by='diameter_um').copy()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))

    # (a) 回転・配向持続時間 tau_r vs Dwell CCDF 緩和時間 tau_dwell
    ax0 = axes[0]
    ax0.errorbar(
        df_abp['diameter_um'],
        df_abp['tau_r_s'],
        yerr=df_abp['tau_r_err_s'],
        fmt='-o',
        color='#d95f02',
        lw=2.2,
        markersize=7,
        capsize=3.5,
        label=r'ABP Orientation Persistence $\tau_r$',
        zorder=3,
    )

    df_run_state = df_all_summary[df_all_summary['state'] == 1].sort_values(by='diameter_um')
    if not df_run_state.empty:
        tau_col = 'tau_ccdf_s' if 'tau_ccdf_s' in df_run_state.columns else 'tau_fit_pdf_s'
        err_col = 'tau_ccdf_err_s' if 'tau_ccdf_err_s' in df_run_state.columns else 'dwell_err_s'
        ax0.errorbar(
            df_run_state['diameter_um'],
            df_run_state[tau_col],
            yerr=df_run_state[err_col],
            fmt='--s',
            color='#1b9e77',
            lw=1.8,
            markersize=6,
            capsize=3.5,
            label=r'Run Dwell Time $\tau_{\mathrm{Run}}^{\mathrm{dwell}}$',
            zorder=2,
        )

    for _, r in df_abp.iterrows():
        y_val = r['tau_r_s']
        d_val = r['diameter_um']
        if np.isfinite(y_val):
            ax0.annotate(
                f"{y_val:.1f} s",
                (d_val, y_val),
                textcoords="offset points",
                xytext=(0, 10 if d_val in [0.63, 3.37, 20.0] else -16),
                ha='center',
                fontsize=8.5,
                fontweight='bold',
                color='#d95f02',
            )

    ax0.set_xscale('log')
    ax0.set_yscale('log')
    ax0.set_xlabel(r"Cargo Particle Diameter $d$ [$\mu\mathrm{m}$]", fontsize=11)
    ax0.set_ylabel(r"Characteristic Time $\tau$ [s]", fontsize=11)
    ax0.set_title(r"(a) ABP Persistence $\tau_r$ vs Dwell Time $\tau_{\mathrm{dwell}}$", fontsize=12, fontweight='bold')
    ax0.set_xticks([0.63, 1.18, 3.37, 5.0, 7.24, 20.0])
    ax0.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax0.grid(True, which="both", linestyle='--', alpha=0.4)
    ax0.legend(loc='upper right', fontsize=8.5, frameon=True, framealpha=0.92)

    # (b) 並進拡散係数 D_t vs 有効拡散係数 D_eff
    ax1 = axes[1]
    ax1.plot(
        df_abp['diameter_um'],
        df_abp['Dt_um2_s'],
        marker='o',
        color='#7570b3',
        lw=2.0,
        markersize=7,
        label=r'Translational Diffusion $D_t$',
        zorder=3,
    )
    ax1.plot(
        df_abp['diameter_um'],
        df_abp['D_eff_um2_s'],
        marker='^',
        color='#e7298a',
        lw=2.2,
        markersize=7,
        label=r'Effective Diffusion $D_{\mathrm{eff}} = D_t + \frac{1}{2}v_0^2 \tau_r$',
        zorder=4,
    )

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel(r"Cargo Particle Diameter $d$ [$\mu\mathrm{m}$]", fontsize=11)
    ax1.set_ylabel(r"Diffusion Coefficient [$\mu\mathrm{m}^2/\mathrm{s}$]", fontsize=11)
    ax1.set_title(r"(b) Thermal Diffusion $D_t$ vs Effective Diffusion $D_{\mathrm{eff}}$", fontsize=12, fontweight='bold')
    ax1.set_xticks([0.63, 1.18, 3.37, 5.0, 7.24, 20.0])
    ax1.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax1.grid(True, which="both", linestyle='--', alpha=0.4)
    ax1.legend(loc='lower left', fontsize=8.5, frameon=True, framealpha=0.92)

    # (c) 持続走行長 lambda_p = v_0 * tau_r vs lambda_dwell = v_0 * tau_dwell
    ax2 = axes[2]
    ax2.plot(
        df_abp['diameter_um'],
        df_abp['lambda_p_um'],
        marker='D',
        color='#e6ab02',
        lw=2.2,
        markersize=7,
        label=r'ABP Persistence Length $\lambda_p = v_0 \tau_r$',
        zorder=3,
    )

    if not df_run_state.empty:
        tau_col = 'tau_ccdf_s' if 'tau_ccdf_s' in df_run_state.columns else 'tau_fit_pdf_s'
        lambda_dwell = df_run_state['mean_speed_geom_um_s'] * df_run_state[tau_col]
        ax2.plot(
            df_run_state['diameter_um'],
            lambda_dwell,
            marker='s',
            color='#66a61e',
            linestyle='--',
            lw=1.8,
            markersize=6,
            label=r'Dwell Burst Length $\lambda_{\mathrm{dwell}} = v_0 \tau_{\mathrm{dwell}}$',
            zorder=2,
        )

    for _, r in df_abp.iterrows():
        y_val_len = r['lambda_p_um']
        d_val = r['diameter_um']
        if np.isfinite(y_val_len):
            ax2.annotate(
                f"{y_val_len:.1f} $\\mu$m",
                (d_val, y_val_len),
                textcoords="offset points",
                xytext=(0, 10 if d_val in [0.63, 3.37, 20.0] else -16),
                ha='center',
                fontsize=8.5,
                fontweight='bold',
                color='#e6ab02',
            )

    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel(r"Cargo Particle Diameter $d$ [$\mu\mathrm{m}$]", fontsize=11)
    ax2.set_ylabel(r"Processivity / Length [$\mu\mathrm{m}$]", fontsize=11)
    ax2.set_title(r"(c) Persistence Run Length $\lambda_p$ vs Diameter", fontsize=12, fontweight='bold')
    ax2.set_xticks([0.63, 1.18, 3.37, 5.0, 7.24, 20.0])
    ax2.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax2.grid(True, which="both", linestyle='--', alpha=0.4)
    ax2.legend(loc='lower left', fontsize=8.5, frameon=True, framealpha=0.92)

    fig.suptitle(
        r"Active Brownian Transport Parameters vs Cargo Particle Diameter",
        fontsize=13.5,
        fontweight='bold',
    )
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    png_path = output_path.with_suffix('.png')
    if png_path != output_path:
        fig.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVED] {output_path} (and {png_path})", flush=True)


def plot_separation_index_vs_diameter(
    df_summary: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    """
    状態分離度 (State Separation Metric)
        S_v = |mu_fast - mu_slow| / sqrt((sigma_fast^2 + sigma_slow^2) / 2)
    を各粒子サイズごとに算出し、粒子径 vs 分離度の2パネルプロットを作成・保存する。
    """
    records = []
    for binfo in BEADS_INFO:
        bname = binfo['name']
        dia = binfo['diameter_um']
        df_b = df_summary[df_summary['bead_name'] == bname]
        if df_b.empty:
            continue
        row_s0 = df_b[df_b['state'] == 0]
        row_s1 = df_b[df_b['state'] == 1]
        if row_s0.empty or row_s1.empty:
            continue

        mu_slow = float(row_s0.iloc[0]['mean_log_v'])
        mu_fast = float(row_s1.iloc[0]['mean_log_v'])
        sig_slow = float(row_s0.iloc[0]['std_log_v'])
        sig_fast = float(row_s1.iloc[0]['std_log_v'])

        delta_mu = abs(mu_fast - mu_slow)
        sig_pooled = np.sqrt(0.5 * (sig_fast**2 + sig_slow**2))
        S_v = delta_mu / (sig_pooled + 1e-12)

        v_slow = float(row_s0.iloc[0]['mean_speed_geom_um_s'])
        v_fast = float(row_s1.iloc[0]['mean_speed_geom_um_s'])

        records.append({
            'bead_name': bname,
            'diameter_um': dia,
            'mu_slow': mu_slow,
            'mu_fast': mu_fast,
            'delta_mu_log_v': delta_mu,
            'sig_slow': sig_slow,
            'sig_fast': sig_fast,
            'sig_pooled': sig_pooled,
            'S_v': S_v,
            'v_slow_geom_um_s': v_slow,
            'v_fast_geom_um_s': v_fast,
            'speed_ratio': v_fast / (v_slow + 1e-12),
            'color': binfo['color'],
            'marker': binfo['marker'],
        })

    if not records:
        return pd.DataFrame()

    df_sep = pd.DataFrame(records).sort_values(by='diameter_um')

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # --- Panel (a): S_v vs 粒子径 ---
    ax0 = axes[0]
    ax0.axhspan(2.0, 3.5, color='#1b9e77', alpha=0.10, label=r"Very Large Separation ($S_v \geq 2.0$)")
    ax0.axhspan(1.2, 2.0, color='#377eb8', alpha=0.08, label=r"Large Separation ($1.2 \leq S_v < 2.0$)")
    ax0.axhspan(0.5, 1.2, color='#ff7f0e', alpha=0.06, label=r"Moderate Separation ($0.5 \leq S_v < 1.2$)")
    ax0.axhspan(0.0, 0.5, color='#e41a1c', alpha=0.06, label=r"Poor Separation / Single Mode ($S_v < 0.5$)")

    ax0.plot(df_sep['diameter_um'], df_sep['S_v'], color='#2b5c8f', lw=2.2, zorder=2)

    for _, row in df_sep.iterrows():
        ax0.scatter(
            row['diameter_um'],
            row['S_v'],
            s=110,
            color=row['color'],
            marker=row['marker'],
            edgecolors='black',
            linewidths=1.2,
            zorder=3,
            label=f"$d={row['diameter_um']:.2f}\\,\\mu\\mathrm{{m}}$",
        )
        if row['diameter_um'] == 7.24:
            y_offset = 14
        elif row['diameter_um'] in [0.63, 3.37, 20.0]:
            y_offset = 12
        else:
            y_offset = -18

        ax0.annotate(
            f"{row['S_v']:.3f}",
            (row['diameter_um'], row['S_v']),
            textcoords="offset points",
            xytext=(0, y_offset),
            ha='center',
            fontsize=9.5,
            fontweight='bold',
            color='#1a2a3a',
        )

    ax0.set_xscale('log')
    ax0.set_xlabel(r"Cargo Particle Diameter $d$ [$\mu\mathrm{m}$]", fontsize=11)
    ax0.set_ylabel(r"Velocity State Separation $S_v$", fontsize=11)
    ax0.set_title(r"(a) Velocity State Separation $S_v$ vs Diameter", fontsize=12, fontweight='bold')
    ax0.set_xticks([0.63, 1.18, 3.37, 5.0, 7.24, 20.0])
    ax0.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax0.set_ylim(-0.15, 3.2)
    ax0.grid(True, linestyle='--', alpha=0.5)
    ax0.legend(loc='lower left', fontsize=8.2, frameon=True, framealpha=0.92)

    # --- Panel (b): 分子（平均差 Δμ） vs 分母（合成標準偏差 σ_pooled） ---
    ax1 = axes[1]
    ax1.plot(
        df_sep['diameter_um'],
        df_sep['delta_mu_log_v'],
        color='#e41a1c',
        lw=2.2,
        marker='o',
        markersize=6,
        label=r"Mean Distance $|\mu_{\rm fast} - \mu_{\rm slow}|$",
        zorder=3,
    )
    ax1.plot(
        df_sep['diameter_um'],
        df_sep['sig_pooled'],
        color='#4daf4a',
        lw=2.0,
        linestyle='--',
        marker='s',
        markersize=5.5,
        label=r"Pooled Std Dev $\sqrt{(\sigma_{\rm fast}^2 + \sigma_{\rm slow}^2)/2}$",
        zorder=2,
    )

    for _, row in df_sep.iterrows():
        y_off_b = 10 if row['diameter_um'] != 7.24 else 12
        ax1.annotate(
            f"{row['delta_mu_log_v']:.2f}",
            (row['diameter_um'], row['delta_mu_log_v']),
            textcoords="offset points",
            xytext=(0, y_off_b),
            ha='center',
            fontsize=8.5,
            color='#e41a1c',
            fontweight='bold',
        )

    ax1.set_xscale('log')
    ax1.set_xlabel(r"Cargo Particle Diameter $d$ [$\mu\mathrm{m}$]", fontsize=11)
    ax1.set_ylabel("Log-Speed Difference / Width", fontsize=11)
    ax1.set_title(r"(b) Decomposition: Mean Distance vs Pooled Width", fontsize=12, fontweight='bold')
    ax1.set_xticks([0.63, 1.18, 3.37, 5.0, 7.24, 20.0])
    ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax1.set_ylim(0.0, 3.0)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='upper right', fontsize=9.2, frameon=True, framealpha=0.92)

    fig.suptitle(
        r"Motion Mode Separation Metric $S_v = \frac{|\mu_{\rm fast}-\mu_{\rm slow}|}{\sqrt{(\sigma_{\rm fast}^2+\sigma_{\rm slow}^2)/2}}$ (1D Speed-Only Model)",
        fontsize=13,
        fontweight='bold',
    )
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    png_path = output_path.with_suffix('.png')
    if png_path != output_path:
        fig.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVED] {output_path} (and {png_path})", flush=True)

    return df_sep


def plot_run_relaxation_time_vs_diameter(
    df_summary: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    """
    横軸が粒子径 d [um]、縦軸が Run 状態の緩和時間・持続時間 tau_Run [s] のグラフを作成・保存する。
    - Panel (a): Run 特性緩和時間 tau_Run vs 粒子径 (CCDF 指数フィッティング値 tau_CCDF ± err, 実測平均値, 理論マルコフ持続時間)
    - Panel (b): Run 平均走行距離 lambda_Run = v_geom * tau_Run [um] (Run 1回あたりの平均滑走距離) vs 粒子径
    """
    df_run = df_summary[df_summary['state'] == 1].copy()
    if df_run.empty:
        return pd.DataFrame()

    df_run = df_run.sort_values(by='diameter_um')
    tau_col = 'tau_ccdf_s' if 'tau_ccdf_s' in df_run.columns else 'tau_fit_pdf_s'
    err_col = 'tau_ccdf_err_s' if 'tau_ccdf_err_s' in df_run.columns else 'dwell_err_s'

    df_run['run_length_um'] = df_run['mean_speed_geom_um_s'] * df_run[tau_col]
    df_run['run_length_emp_um'] = df_run['mean_speed_geom_um_s'] * df_run['mean_dwell_emp_s']

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5))

    # --- Panel (a): Run 緩和時間 tau_Run vs 粒子径 ---
    ax0 = axes[0]
    ax0.errorbar(
        df_run['diameter_um'],
        df_run[tau_col],
        yerr=df_run[err_col],
        fmt='-o',
        color='#EE6677',
        ecolor='#EE6677',
        elinewidth=1.5,
        capsize=4,
        capthick=1.2,
        lw=2.2,
        markersize=7,
        label=r'Fitted Relaxation Time $\tau_{\mathrm{Run}}$ (CCDF Fit: $P(T \geq t) = A e^{-t/\tau}$)',
        zorder=3,
    )
    ax0.plot(
        df_run['diameter_um'],
        df_run['mean_dwell_emp_s'],
        marker='s',
        color='#CC3311',
        linestyle='--',
        lw=1.6,
        markersize=5.5,
        alpha=0.8,
        label=r'Empirical Mean Duration $\langle t_{\mathrm{Run}} \rangle$',
        zorder=2,
    )
    ax0.plot(
        df_run['diameter_um'],
        df_run['theoretical_dwell_time_s'],
        marker='^',
        color='gray',
        linestyle=':',
        lw=1.4,
        markersize=5,
        alpha=0.7,
        label=r'Theoretical Markov Duration $\Delta t / (1 - A_{11})$',
        zorder=1,
    )

    # 各粒子径のデータ点にアノテーション
    for _, row in df_run.iterrows():
        y_val = row[tau_col]
        d_val = row['diameter_um']
        ax0.annotate(
            f"{y_val:.1f} s",
            (d_val, y_val),
            textcoords="offset points",
            xytext=(0, 10 if d_val in [0.63, 3.37, 20.0] else -16),
            ha='center',
            fontsize=9.0,
            fontweight='bold',
            color='#AA3377',
        )

    ax0.set_xscale('log')
    ax0.set_yscale('log')
    ax0.set_xlabel(r"Cargo Particle Diameter $d$ [$\mu\mathrm{m}$]", fontsize=11)
    ax0.set_ylabel(r"Run Relaxation Time $\tau_{\mathrm{Run}}$ [s]", fontsize=11)
    ax0.set_title(r"(a) Run Mode Characteristic Relaxation Time $\tau_{\mathrm{Run}}$ vs Diameter", fontsize=12, fontweight='bold')
    ax0.set_xticks([0.63, 1.18, 3.37, 5.0, 7.24, 20.0])
    ax0.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax0.grid(True, which="both", linestyle='--', alpha=0.4)
    ax0.legend(loc='upper right', fontsize=8.5, frameon=True, framealpha=0.92)

    # --- Panel (b): Run 滑走長 lambda_Run vs 粒子径 ---
    ax1 = axes[1]
    ax1.plot(
        df_run['diameter_um'],
        df_run['run_length_um'],
        marker='o',
        color='#4477AA',
        lw=2.2,
        markersize=7,
        label=r'Mean Run Length $\lambda_{\mathrm{Run}} = v_{\mathrm{geom}} \times \tau_{\mathrm{fit}}$',
        zorder=3,
    )
    ax1.plot(
        df_run['diameter_um'],
        df_run['run_length_emp_um'],
        marker='s',
        color='#66CCEE',
        linestyle='--',
        lw=1.6,
        markersize=5.5,
        alpha=0.8,
        label=r'Empirical Run Length $v_{\mathrm{geom}} \times \langle t_{\mathrm{Run}} \rangle$',
        zorder=2,
    )

    for _, row in df_run.iterrows():
        y_val_len = row['run_length_um']
        d_val = row['diameter_um']
        ax1.annotate(
            f"{y_val_len:.1f} $\\mu$m",
            (d_val, y_val_len),
            textcoords="offset points",
            xytext=(0, 10 if d_val in [0.63, 3.37, 7.24] else -16),
            ha='center',
            fontsize=9.0,
            fontweight='bold',
            color='#225588',
        )

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel(r"Cargo Particle Diameter $d$ [$\mu\mathrm{m}$]", fontsize=11)
    ax1.set_ylabel(r"Run Processivity / Run Length $\lambda_{\mathrm{Run}}$ [$\mu\mathrm{m}$]", fontsize=11)
    ax1.set_title(r"(b) Mean Run Length per Burst $\lambda_{\mathrm{Run}}$ vs Diameter", fontsize=12, fontweight='bold')
    ax1.set_xticks([0.63, 1.18, 3.37, 5.0, 7.24, 20.0])
    ax1.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax1.grid(True, which="both", linestyle='--', alpha=0.4)
    ax1.legend(loc='lower left', fontsize=8.5, frameon=True, framealpha=0.92)

    fig.suptitle(
        r"Cargo Particle Run Relaxation Time $\tau_{\mathrm{Run}}$ and Run Length $\lambda_{\mathrm{Run}}$ vs Diameter",
        fontsize=13.5,
        fontweight='bold',
    )
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    png_path = output_path.with_suffix('.png')
    if png_path != output_path:
        fig.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVED] {output_path} (and {png_path})", flush=True)

    return df_run


def plot_duty_cycle_vs_diameter(
    df_summary: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    """
    粒子径 vs デューティ比 (Duty Cycle) f_run の比較解析プロット。
    
    1. 観測占有率 (Empirical Duty Cycle):
       f_run_emp = N_run / N_total
    2. マルコフ定常分布理論値 (Markov Stationary Duty Cycle):
       f_run_markov = \\pi_run = A_01 / (A_01 + A_10)
    3. Dwell time 指数フィット理論値 (Dwell-Fit Duty Cycle):
       f_run_dwell = \\tau_run_fit / (\\tau_run_fit + \\tau_tumble_fit)
    """
    if df_summary.empty:
        return pd.DataFrame()

    records = []
    for binfo in BEADS_INFO:
        bname = binfo['name']
        dia = binfo['diameter_um']
        sub = df_summary[df_summary['bead_name'] == bname]
        if sub.empty:
            continue
        row0 = sub[sub['state'] == 0]
        row1 = sub[sub['state'] == 1]
        if row0.empty or row1.empty:
            continue

        n_tot = int(row1.iloc[0].get('n_observations', 0))
        n_run = int(row1.iloc[0].get('n_state_obs', 0))
        f_emp = float(row1.iloc[0].get('empirical_prob', n_run / (n_tot + 1e-12)))

        pi_stat = float(row1.iloc[0]['stationary_prob'])

        tau_col = 'tau_ccdf_s' if 'tau_ccdf_s' in row0.columns else 'tau_fit_pdf_s'
        tau_tumble_fit = float(row0.iloc[0][tau_col])
        tau_run_fit = float(row1.iloc[0][tau_col])
        f_dwell_fit = tau_run_fit / (tau_run_fit + tau_tumble_fit + 1e-12)

        tau_tumble_theo = float(row0.iloc[0]['theoretical_dwell_time_s'])
        tau_run_theo = float(row1.iloc[0]['theoretical_dwell_time_s'])

        records.append({
            'bead_name': bname,
            'diameter_um': dia,
            'n_total_obs': n_tot,
            'n_run_obs': n_run,
            'n_tumble_obs': n_tot - n_run,
            'f_run_empirical': f_emp,
            'f_run_markov_stationary': pi_stat,
            'f_run_dwell_fit': f_dwell_fit,
            'tau_run_fit_s': tau_run_fit,
            'tau_tumble_fit_s': tau_tumble_fit,
            'tau_run_theo_s': tau_run_theo,
            'tau_tumble_theo_s': tau_tumble_theo,
            'consistency_ratio': f_emp / (pi_stat + 1e-12),
        })

    if not records:
        return pd.DataFrame()

    df_duty = pd.DataFrame(records).sort_values(by='diameter_um')

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    # --- Panel (a): デューティ比 vs 粒子径 ---
    ax0 = axes[0]
    ax0.plot(
        df_duty['diameter_um'],
        df_duty['f_run_empirical'] * 100.0,
        marker='o',
        color='#1f77b4',
        lw=2.2,
        markersize=7.5,
        label=r"Empirical: $f_{\mathrm{run}}^{\mathrm{emp}} = N_{\mathrm{run}} / N_{\mathrm{total}}$",
        zorder=4,
    )
    ax0.plot(
        df_duty['diameter_um'],
        df_duty['f_run_markov_stationary'] * 100.0,
        marker='^',
        color='#d62728',
        linestyle='--',
        lw=2.0,
        markersize=7.5,
        label=r"Markov Stationary: $f_{\mathrm{run}}^{\mathrm{Markov}} = \pi_{\mathrm{run}}$",
        zorder=3,
    )
    ax0.plot(
        df_duty['diameter_um'],
        df_duty['f_run_dwell_fit'] * 100.0,
        marker='s',
        color='#2ca02c',
        linestyle=':',
        lw=2.0,
        markersize=6.5,
        label=r"Dwell Fit: $\frac{\tau_{\mathrm{run}}}{\tau_{\mathrm{run}} + \tau_{\mathrm{tumble}}}$",
        zorder=2,
    )

    # 各点に数値ラベルを追加
    for _, r in df_duty.iterrows():
        y_off = -14 if r['diameter_um'] == 5.0 else 8
        ax0.annotate(
            f"{r['f_run_empirical']*100:.1f}%",
            xy=(r['diameter_um'], r['f_run_empirical']*100),
            xytext=(0, y_off),
            textcoords="offset points",
            ha='center',
            fontsize=8.5,
            color='#1f77b4',
            weight='bold',
        )

    ax0.set_xscale('log')
    ax0.set_ylim(0, 80)
    ax0.set_xlabel(r"Cargo Particle Diameter $d$ [$\mu\mathrm{m}$]", fontsize=11)
    ax0.set_ylabel(r"Run Duty Ratio $f_{\mathrm{run}}$ [%]", fontsize=11)
    ax0.set_title(r"(a) Run Duty Ratio $f_{\mathrm{run}}$ vs Particle Diameter", fontsize=12, fontweight='bold')
    ax0.set_xticks([0.63, 1.18, 3.37, 5.0, 7.24, 20.0])
    ax0.set_xticklabels(['0.63', '1.18', '3.37', '5.00', '7.24', '20.0'], fontsize=9.5)
    ax0.tick_params(axis='x', which='minor', bottom=False)
    ax0.grid(True, which="both", linestyle='--', alpha=0.4)
    ax0.legend(loc='upper right', fontsize=9.0, frameon=True, framealpha=0.92)

    # --- Panel (b): 観測占有率 / マルコフ定常理論値の比率 ---
    ax1 = axes[1]
    ax1.axhline(1.0, color='gray', linestyle='--', lw=1.5, label="Perfect Markov Consistency (Ratio = 1.0)", zorder=1)
    ax1.plot(
        df_duty['diameter_um'],
        df_duty['consistency_ratio'],
        marker='D',
        color='#9467bd',
        lw=2.0,
        markersize=7,
        label=r"Consistency Ratio: $f_{\mathrm{run}}^{\mathrm{emp}} / f_{\mathrm{run}}^{\mathrm{Markov}}$",
        zorder=3,
    )

    for _, r in df_duty.iterrows():
        y_off = -15 if r['diameter_um'] == 7.24 else 8
        ax1.annotate(
            f"{r['consistency_ratio']:.2f}",
            xy=(r['diameter_um'], r['consistency_ratio']),
            xytext=(0, y_off),
            textcoords="offset points",
            ha='center',
            fontsize=8.5,
            color='#9467bd',
            weight='bold',
        )

    ax1.set_xscale('log')
    ax1.set_ylim(0.7, 1.3)
    ax1.set_xlabel(r"Cargo Particle Diameter $d$ [$\mu\mathrm{m}$]", fontsize=11)
    ax1.set_ylabel(r"Consistency Ratio $f_{\mathrm{run}}^{\mathrm{emp}} / f_{\mathrm{run}}^{\mathrm{Markov}}$", fontsize=11)
    ax1.set_title(r"(b) Empirical vs Theoretical Markov Consistency", fontsize=12, fontweight='bold')
    ax1.set_xticks([0.63, 1.18, 3.37, 5.0, 7.24, 20.0])
    ax1.set_xticklabels(['0.63', '1.18', '3.37', '5.00', '7.24', '20.0'], fontsize=9.5)
    ax1.tick_params(axis='x', which='minor', bottom=False)
    ax1.grid(True, which="both", linestyle='--', alpha=0.4)
    ax1.legend(loc='lower left', fontsize=9.0, frameon=True, framealpha=0.92)

    fig.suptitle(
        r"Active Cargo Duty Ratio Analysis: Empirical Observation vs Markov Stationary Theory",
        fontsize=13.5,
        fontweight='bold',
    )
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    png_path = output_path.with_suffix('.png')
    if png_path != output_path:
        fig.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVED] {output_path} (and {png_path})", flush=True)

    return df_duty


def plot_msd_data_collapse(
    df_msd: pd.DataFrame,
    df_summary: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    """
    HMM固有の特性スケール (tau_run, lambda_run) および走行デューティ比 f_run を用いた
    MSD の完全データコラプス（Master Curve）プロット。
    
    各粒径 D_c に対し:
        走行デューティ比: f_{\\mathrm{run}} = \\frac{\\tau_{\\mathrm{run}}}{\\tau_{\\mathrm{run}} + \\tau_{\\mathrm{tumble}}}
        無次元ラグ時間:   \\widetilde{\\Delta t} = \\frac{\\Delta t}{\\tau_{\\mathrm{run}}(D_c)}
        無次元補正 MSD:   \\widetilde{\\mathrm{MSD}}_{\\mathrm{corrected}} = \\frac{\\langle \\Delta r^2(\\Delta t) \\rangle_{\\mathrm{all}}}{f_{\\mathrm{run}} \\cdot \\lambda_{\\mathrm{run}}(D_c)^2}
    
    期待される理論マスターカーブ (Persistent Random Walk):
        \\widetilde{\\mathrm{MSD}}(\\widetilde{\\Delta t}) = 2 [ \\widetilde{\\Delta t} - 1 + e^{-\\widetilde{\\Delta t}} ]
    """
    if df_msd.empty or df_summary.empty:
        return pd.DataFrame()

    # 各粒子径のスケール・デューティ比 f_run の集計
    scale_records = []
    for binfo in BEADS_INFO:
        bname = binfo['name']
        dia = binfo['diameter_um']
        sub_s = df_summary[df_summary['bead_name'] == bname]
        if sub_s.empty:
            continue
        row0 = sub_s[sub_s['state'] == 0]
        row1 = sub_s[sub_s['state'] == 1]
        if row0.empty or row1.empty:
            continue

        tau_col = 'tau_ccdf_s' if 'tau_ccdf_s' in row0.columns else 'tau_fit_pdf_s'
        tau_tumble = float(row0.iloc[0][tau_col])
        tau_run = float(row1.iloc[0][tau_col])
        v_run = float(row1.iloc[0]['mean_speed_geom_um_s'])
        f_run = tau_run / (tau_run + tau_tumble + 1e-12)
        lam_run = v_run * tau_run

        scale_records.append({
            'bead_name': bname,
            'diameter_um': dia,
            'tau_tumble_s': tau_tumble,
            'tau_run_s': tau_run,
            'f_run': f_run,
            'v_run_geom_um_s': v_run,
            'run_length_um': lam_run,
            'color': binfo['color'],
            'marker': binfo['marker'],
        })

    if not scale_records:
        return pd.DataFrame()

    df_scales = pd.DataFrame(scale_records)

    # 全体 MSD (state == -1) と Run 状態 MSD (state == 1) の結合
    df_collapse_all = pd.merge(df_msd[df_msd['state'] == -1], df_scales, on=['bead_name', 'diameter_um'])
    df_collapse_all['dimless_lag'] = df_collapse_all['lag_time_s'] / df_collapse_all['tau_run_s']
    df_collapse_all['dimless_msd_raw'] = df_collapse_all['msd_um2'] / (df_collapse_all['run_length_um']**2)
    df_collapse_all['dimless_msd_corrected'] = df_collapse_all['msd_um2'] / (df_collapse_all['f_run'] * (df_collapse_all['run_length_um']**2))
    df_collapse_all['dimless_msd_corrected_sem'] = df_collapse_all['msd_sem_um2'] / (df_collapse_all['f_run'] * (df_collapse_all['run_length_um']**2))

    df_collapse_run = pd.merge(df_msd[df_msd['state'] == 1], df_scales, on=['bead_name', 'diameter_um'])
    df_collapse_run['dimless_lag'] = df_collapse_run['lag_time_s'] / df_collapse_run['tau_run_s']
    df_collapse_run['dimless_msd_run'] = df_collapse_run['msd_um2'] / (df_collapse_run['run_length_um']**2)
    df_collapse_run['dimless_msd_run_sem'] = df_collapse_run['msd_sem_um2'] / (df_collapse_run['run_length_um']**2)

    # 3パネル図の作成
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # 理論曲線のグリッド
    t_grid = np.logspace(-2.5, 1.5, 300)
    msd_master_theory = 2.0 * (t_grid - 1.0 + np.exp(-t_grid))
    msd_ballistic_asymptote = t_grid**2
    msd_diffusive_asymptote = 2.0 * t_grid

    # --- Panel (a): 小粒子群 (0.63, 1.18, 3.37 um) の f_run 補正データコラプス ---
    ax0 = axes[0]
    ax0.plot(t_grid, msd_master_theory, color='black', lw=2.5, label=r'Master Curve: $2 [\tilde{t} - 1 + e^{-\tilde{t}}]$', zorder=4)
    ax0.plot(t_grid, msd_ballistic_asymptote, color='gray', linestyle='--', lw=1.3, label=r'Ballistic: $\tilde{t}^2$ ($\alpha=2$)', zorder=2)
    ax0.plot(t_grid, msd_diffusive_asymptote, color='silver', linestyle=':', lw=1.5, label=r'Diffusive: $2\tilde{t}$ ($\alpha=1$)', zorder=1)

    small_bead_names = ['beads06um', 'beads1um', 'beads3um']
    for binfo in BEADS_INFO:
        bname = binfo['name']
        if bname not in small_bead_names:
            continue
        sub = df_collapse_all[df_collapse_all['bead_name'] == bname].sort_values(by='dimless_lag')
        if sub.empty:
            continue
        dia = sub['diameter_um'].iloc[0]
        f_r = sub['f_run'].iloc[0]
        ax0.errorbar(
            sub['dimless_lag'],
            sub['dimless_msd_corrected'],
            yerr=sub['dimless_msd_corrected_sem'],
            fmt=binfo['marker'],
            color=binfo['color'],
            ecolor=binfo['color'],
            markersize=6,
            capsize=3,
            label=f"$d = {dia:.2f}\\,\\mu\\mathrm{{m}}$ ($f_{{\\mathrm{{run}}}}={f_r:.2f}$)",
            zorder=5,
        )

    ax0.set_xscale('log')
    ax0.set_yscale('log')
    ax0.set_xlim(5e-3, 5e0)
    ax0.set_ylim(2e-4, 5e1)
    ax0.set_xlabel(r"Dimensionless Lag Time $\widetilde{\Delta t} = \Delta t / \tau_{\mathrm{run}}$", fontsize=11)
    ax0.set_ylabel(r"Duty-Corrected MSD $\widetilde{\mathrm{MSD}}_{\mathrm{corrected}} = \frac{\langle \Delta r^2 \rangle_{\mathrm{all}}}{f_{\mathrm{run}} \cdot \lambda_{\mathrm{run}}^2}$", fontsize=11)
    ax0.set_title(r"(a) Duty-Corrected Master Curve Collapse ($d \leq 3.37\,\mu\mathrm{m}$)", fontsize=12, fontweight='bold')
    ax0.grid(True, which="both", linestyle='--', alpha=0.4)
    ax0.legend(loc='lower right', fontsize=8.5, frameon=True, framealpha=0.92)

    # --- Panel (b): 全粒子径 (0.63 - 20 um) の f_run 補正比較 ---
    ax1 = axes[1]
    ax1.plot(t_grid, msd_master_theory, color='black', lw=2.5, label=r'Master Curve $2 [\tilde{t} - 1 + e^{-\tilde{t}}]$', zorder=4)
    ax1.plot(t_grid, msd_ballistic_asymptote, color='gray', linestyle='--', lw=1.3, label=r'Ballistic ($\alpha=2$)', zorder=2)
    ax1.plot(t_grid, msd_diffusive_asymptote, color='silver', linestyle=':', lw=1.5, label=r'Diffusive ($\alpha=1$)', zorder=1)

    for binfo in BEADS_INFO:
        bname = binfo['name']
        sub = df_collapse_all[df_collapse_all['bead_name'] == bname].sort_values(by='dimless_lag')
        if sub.empty:
            continue
        dia = sub['diameter_um'].iloc[0]
        f_r = sub['f_run'].iloc[0]
        ax1.plot(
            sub['dimless_lag'],
            sub['dimless_msd_corrected'],
            marker=binfo['marker'],
            color=binfo['color'],
            lw=1.6,
            markersize=5,
            label=f"$d = {dia:.2f}\\,\\mu\\mathrm{{m}}$ ($f_{{\\mathrm{{run}}}}={f_r:.2f}$)",
            zorder=5,
        )

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlim(3e-3, 2e1)
    ax1.set_ylim(1e-4, 1e2)
    ax1.set_xlabel(r"Dimensionless Lag Time $\widetilde{\Delta t} = \Delta t / \tau_{\mathrm{run}}$", fontsize=11)
    ax1.set_ylabel(r"Duty-Corrected MSD $\widetilde{\mathrm{MSD}}_{\mathrm{corrected}} = \frac{\langle \Delta r^2 \rangle_{\mathrm{all}}}{f_{\mathrm{run}} \cdot \lambda_{\mathrm{run}}^2}$", fontsize=11)
    ax1.set_title(r"(b) Full Size Duty-Corrected Comparison ($d = 0.63 - 20\,\mu\mathrm{m}$)", fontsize=12, fontweight='bold')
    ax1.grid(True, which="both", linestyle='--', alpha=0.4)
    ax1.legend(loc='lower right', fontsize=8.2, frameon=True, framealpha=0.92)

    # --- Panel (c): 純粋な Run モード MSD の完全弾道コラプス (MSD_run = dt^2) ---
    ax2 = axes[2]
    ax2.plot(t_grid, msd_ballistic_asymptote, color='crimson', lw=2.5, label=r'Pure Ballistic Line: $\widetilde{\mathrm{MSD}}_{\mathrm{Run}} = \widetilde{\Delta t}^2$', zorder=4)

    for binfo in BEADS_INFO:
        bname = binfo['name']
        dia = binfo['diameter_um']
        sub_r = df_collapse_run[df_collapse_run['bead_name'] == bname].sort_values(by='dimless_lag')
        if sub_r.empty:
            continue
        ax2.plot(
            sub_r['dimless_lag'],
            sub_r['dimless_msd_run'],
            marker=binfo['marker'],
            color=binfo['color'],
            lw=1.6,
            markersize=5,
            label=f"$d = {dia:.2f}\\,\\mu\\mathrm{{m}}$ (Run)",
            zorder=5,
        )

    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlim(3e-3, 2e1)
    ax2.set_ylim(1e-5, 5e2)
    ax2.set_xlabel(r"Dimensionless Lag Time $\widetilde{\Delta t} = \Delta t / \tau_{\mathrm{run}}$", fontsize=11)
    ax2.set_ylabel(r"Dimensionless Run MSD $\widetilde{\mathrm{MSD}}_{\mathrm{Run}} = \langle \Delta r^2 \rangle_{\mathrm{Run}} / \lambda_{\mathrm{run}}^2$", fontsize=11)
    ax2.set_title(r"(c) HMM Segmented Run Mode Ballistic Collapse ($\alpha = 2$)", fontsize=12, fontweight='bold')
    ax2.grid(True, which="both", linestyle='--', alpha=0.4)
    ax2.legend(loc='lower right', fontsize=8.2, frameon=True, framealpha=0.92)

    fig.suptitle(
        r"Duty-Corrected Universal MSD Data Collapse onto Master Curve $\widetilde{\mathrm{MSD}}_{\mathrm{corrected}}(\widetilde{\Delta t}) = 2 [\widetilde{\Delta t} - 1 + e^{-\widetilde{\Delta t}}]$",
        fontsize=13.5,
        fontweight='bold',
    )
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    png_path = output_path.with_suffix('.png')
    if png_path != output_path:
        fig.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVED] {output_path} (and {png_path})", flush=True)

    return df_collapse_all


def plot_state_turning_angle_distributions_6panel(
    fitted_results: Dict[str, dict],
    output_path: Path,
    n_components: int = 2,
    bins: int = 36,
) -> pd.DataFrame:
    """
    各粒子径における Run と Tumble の方向転換角分布 P(Δθ) の 6 パネルプロット。
    一様分布基準線 P_uniform = 1/(2π) ≈ 0.159 rad^-1 と von Mises フィッティング曲線を描画。
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5))
    axes = axes.flatten()
    summary_records = []

    for idx, binfo in enumerate(BEADS_INFO):
        ax = axes[idx]
        bname = binfo['name']
        dia = binfo['diameter_um']

        if bname not in fitted_results:
            ax.set_title(f"$d = {dia:.2f}\\,\\mu\\mathrm{{m}}$ (No data)", fontsize=12)
            ax.axis('off')
            continue

        res = fitted_results[bname]
        angles_dict = res.get('turning_angles', {})
        if not angles_dict:
            ax.set_title(f"$d = {dia:.2f}\\,\\mu\\mathrm{{m}}$ (No angles)", fontsize=12)
            continue

        # 一様分布基準線
        ax.axhline(1.0 / (2.0 * np.pi), color='gray', linestyle=':', lw=1.5, label=r'Uniform: $1/(2\pi) \approx 0.159$', zorder=1)

        for s in range(n_components):
            s_lbl = STATE_NAMES.get(n_components, {}).get(s, f"State {s}")
            scolor = STATE_COLORS.get(s, f"C{s}")
            arr_s = angles_dict.get(s, np.array([]))

            if len(arr_s) < 10:
                continue

            c_stats = hc.calc_circular_stats_dict(arr_s)
            vm_fit = hc.fit_von_mises_distribution(arr_s, bins=bins)

            # ヒストグラム
            centers = vm_fit['bin_centers']
            pdf_vals = vm_fit['pdf_data']
            ax.step(centers, pdf_vals, where='mid', color=scolor, alpha=0.45, lw=1.2)
            ax.plot(centers, pdf_vals, 'o', color=scolor, markersize=4, alpha=0.7)

            # von Mises フィッティング線
            if len(vm_fit['fit_x']) > 0 and not np.isnan(vm_fit['kappa']):
                k_val = vm_fit['kappa']
                cos_val = c_stats['mean_cos']
                fit_lbl = f"{s_lbl}: Fit ($\\kappa={k_val:.2f}, \\langle\\cos\\Delta\\theta\\rangle={cos_val:.2f}$, $N={len(arr_s):,}$)"
                ax.plot(vm_fit['fit_x'], vm_fit['fit_y'], color=scolor, lw=2.2, label=fit_lbl, zorder=3)

            summary_records.append({
                'bead_name': bname,
                'diameter_um': dia,
                'state': s,
                'state_label': s_lbl,
                'count': len(arr_s),
                'kappa': vm_fit['kappa'],
                'kappa_err': vm_fit['kappa_err'],
                'r_squared': vm_fit['r_squared'],
                'mean_cos': c_stats['mean_cos'],
                'mean_resultant_length': c_stats['mean_resultant_length_R'],
                'circular_variance': c_stats['circular_variance'],
                'mean_abs_angle_deg': c_stats['mean_abs_angle_deg'],
            })

        ax.set_xlim(-np.pi, np.pi)
        ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        ax.set_xticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"], fontsize=10)
        ax.set_title(f"$d = {dia:.2f}\\,\\mu\\mathrm{{m}}$", fontsize=12, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend(loc='upper right', fontsize=8.0, frameon=True, framealpha=0.92)

        if idx >= 3:
            ax.set_xlabel(r"Turning Angle $\Delta\theta$ [rad]", fontsize=11)
        if idx % 3 == 0:
            ax.set_ylabel(r"Probability Density $P(\Delta\theta)$ [$\mathrm{rad}^{-1}$]", fontsize=11)

    fig.suptitle(
        r"State-Dependent Turning Angle Probability Density Distributions $P(\Delta\theta)$ with von Mises Fits",
        fontsize=13.5,
        fontweight='bold',
    )
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    png_path = output_path.with_suffix('.png')
    if png_path != output_path:
        fig.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVED] {output_path} (and {png_path})", flush=True)

    return pd.DataFrame(summary_records)


def plot_turning_angle_polar_6panel(
    fitted_results: Dict[str, dict],
    output_path: Path,
    n_components: int = 2,
    bins: int = 24,
):
    """
    極座標系（Polar Rose Plot）における Run と Tumble の方向転換角分布の 6 パネルプロット。
    0 rad (前方) への指向性と等方性を直感的に可視化。
    """
    fig, axes = plt.subplots(2, 3, figsize=(14, 9.5), subplot_kw=dict(polar=True))
    axes = axes.flatten()

    for idx, binfo in enumerate(BEADS_INFO):
        ax = axes[idx]
        bname = binfo['name']
        dia = binfo['diameter_um']

        if bname not in fitted_results:
            ax.set_title(f"$d = {dia:.2f}\\,\\mu\\mathrm{{m}}$ (No data)", fontsize=11)
            ax.axis('off')
            continue

        res = fitted_results[bname]
        angles_dict = res.get('turning_angles', {})
        if not angles_dict:
            continue

        bin_edges = np.linspace(-np.pi, np.pi, bins + 1)
        centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

        for s in range(n_components):
            s_lbl = STATE_NAMES.get(n_components, {}).get(s, f"State {s}")
            scolor = STATE_COLORS.get(s, f"C{s}")
            arr_s = angles_dict.get(s, np.array([]))

            if len(arr_s) < 10:
                continue

            counts, _ = np.histogram(arr_s, bins=bin_edges, density=True)
            theta_closed = np.concatenate([centers, [centers[0]]])
            r_closed = np.concatenate([counts, [counts[0]]])

            ax.plot(theta_closed, r_closed, color=scolor, lw=2.0, label=f"{s_lbl} ($N={len(arr_s):,}$)", zorder=3)
            ax.fill(theta_closed, r_closed, color=scolor, alpha=0.18, zorder=2)

        # 一様分布円
        u_val = 1.0 / (2.0 * np.pi)
        th_circ = np.linspace(-np.pi, np.pi, 200)
        ax.plot(th_circ, np.full_like(th_circ, u_val), color='gray', linestyle=':', lw=1.2, label=r'Uniform $1/(2\pi)$', zorder=1)

        ax.set_theta_zero_location('E')
        ax.set_title(f"$d = {dia:.2f}\\,\\mu\\mathrm{{m}}$", fontsize=11, fontweight='bold', pad=12)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='lower left', bbox_to_anchor=(-0.15, -0.2), fontsize=7.5, frameon=True, framealpha=0.9)

    fig.suptitle(
        r"Polar Distributions of Turning Angles $P(\Delta\theta)$: Forward Persistence vs Isotropy",
        fontsize=13.5,
        fontweight='bold',
    )
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    png_path = output_path.with_suffix('.png')
    if png_path != output_path:
        fig.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVED] {output_path} (and {png_path})", flush=True)


def plot_turning_angle_summary_vs_diameter(
    df_angle_summary: pd.DataFrame,
    output_path: Path,
    n_components: int = 2,
):
    """
    粒子径 vs 方向転換角パラメータ (von Mises 集中度 kappa & 平均コサイン <cosΔθ>) プロット。
    """
    if df_angle_summary.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    # (a) von Mises 集中度 kappa
    ax0 = axes[0]
    for s in range(n_components):
        sub = df_angle_summary[df_angle_summary['state'] == s].sort_values(by='diameter_um')
        lbl = STATE_NAMES.get(n_components, {}).get(s, f"State {s}")
        col = STATE_COLORS.get(s, f"C{s}")
        mrk = 'o' if s == 0 else 's'
        ax0.plot(
            sub['diameter_um'],
            sub['kappa'],
            marker=mrk,
            color=col,
            lw=2.0,
            markersize=7,
            label=f"{lbl} ($\kappa$)",
        )
    ax0.set_xscale('log')
    ax0.set_xlabel(r"Cargo Particle Diameter $d$ [$\mu\mathrm{m}$]", fontsize=11)
    ax0.set_ylabel(r"von Mises Concentration Parameter $\kappa$", fontsize=11)
    ax0.set_title(r"(a) Turning Angle Concentration $\kappa$ vs Diameter", fontsize=12, fontweight='bold')
    ax0.set_xticks([0.63, 1.18, 3.37, 5.0, 7.24, 20.0])
    ax0.set_xticklabels(['0.63', '1.18', '3.37', '5.00', '7.24', '20.0'], fontsize=9.5)
    ax0.tick_params(axis='x', which='minor', bottom=False)
    ax0.grid(True, which="both", linestyle='--', alpha=0.4)
    ax0.legend(loc='best', fontsize=9.2, frameon=True, framealpha=0.92)

    # (b) 方向持続性 <cosΔθ>
    ax1 = axes[1]
    ax1.axhline(0.0, color='gray', linestyle=':', lw=1.2, label="Isotropic / Random (0.0)")
    ax1.axhline(1.0, color='silver', linestyle=':', lw=1.2, label="Perfect Straight (1.0)")
    for s in range(n_components):
        sub = df_angle_summary[df_angle_summary['state'] == s].sort_values(by='diameter_um')
        lbl = STATE_NAMES.get(n_components, {}).get(s, f"State {s}")
        col = STATE_COLORS.get(s, f"C{s}")
        mrk = 'o' if s == 0 else 's'
        ax1.plot(
            sub['diameter_um'],
            sub['mean_cos'],
            marker=mrk,
            color=col,
            lw=2.0,
            markersize=7,
            label=f"{lbl} ($\langle\\cos\\Delta\\theta\\rangle$)",
        )
    ax1.set_xscale('log')
    ax1.set_ylim(-0.1, 1.05)
    ax1.set_xlabel(r"Cargo Particle Diameter $d$ [$\mu\mathrm{m}$]", fontsize=11)
    ax1.set_ylabel(r"Directional Persistence $\langle \cos\Delta\theta \rangle$", fontsize=11)
    ax1.set_title(r"(b) Directional Persistence $\langle \cos\Delta\theta \rangle$ vs Diameter", fontsize=12, fontweight='bold')
    ax1.set_xticks([0.63, 1.18, 3.37, 5.0, 7.24, 20.0])
    ax1.set_xticklabels(['0.63', '1.18', '3.37', '5.00', '7.24', '20.0'], fontsize=9.5)
    ax1.tick_params(axis='x', which='minor', bottom=False)
    ax1.grid(True, which="both", linestyle='--', alpha=0.4)
    ax1.legend(loc='best', fontsize=9.2, frameon=True, framealpha=0.92)

    fig.suptitle(
        r"Directional Persistence of Motion Modes: von Mises Concentration $\kappa$ & $\langle\cos\Delta\theta\rangle$",
        fontsize=13.5,
        fontweight='bold',
    )
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    png_path = output_path.with_suffix('.png')
    if png_path != output_path:
        fig.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVED] {output_path} (and {png_path})", flush=True)


def plot_state_autocorrelations_grid(
    fitted_results: Dict[str, dict],
    output_path: Path,
    n_components: int = 2,
    max_lag_s: float = 60.0,
):
    """
    3行6列グリッドで各粒径における運動モード（Run vs Tumble）の
    行0: 速度ベクトル自己相関 (VACF: Velocity Autocorrelation Function)
    行1: 配向方向自己相関 (OACF: Orientation Autocorrelation Function)
    行2: 速さスカラー自己相関 (SACF: Speed Autocorrelation Function)
    および指数減衰フィッティング曲線 C(tau) = exp(-tau / tau_corr) を描画する。
    """
    fig, axes = plt.subplots(3, 6, figsize=(22, 10.5), sharex=True)
    corr_types = [
        ('vacf', r'VACF $\langle \mathbf{v}(t)\cdot\mathbf{v}(t+\tau) \rangle / \langle v^2 \rangle$', 'Velocity Vector'),
        ('oacf', r'OACF $\langle \hat{\mathbf{e}}(t)\cdot\hat{\mathbf{e}}(t+\tau) \rangle$', 'Orientation Unit Vector'),
        ('sacf', r'SACF $\langle \delta v(t)\delta v(t+\tau) \rangle / \langle \delta v^2 \rangle$', 'Speed Fluctuation'),
    ]

    for col_idx, binfo in enumerate(BEADS_INFO):
        bname = binfo['name']
        dia = binfo['diameter_um']

        if bname not in fitted_results or 'autocorrelations' not in fitted_results[bname]:
            for row_idx in range(3):
                axes[row_idx, col_idx].set_visible(False)
            continue

        ac_dict = fitted_results[bname]['autocorrelations']
        fits_dict = fitted_results[bname].get('autocorr_fits', {})

        for row_idx, (ctype, ctitle, clbl) in enumerate(corr_types):
            ax = axes[row_idx, col_idx]
            ax.axhline(0.0, color='gray', linestyle=':', lw=1.0, alpha=0.7)

            for s in range(n_components):
                if s not in ac_dict or ctype not in ac_dict[s]:
                    continue
                df_c = ac_dict[s][ctype]
                if df_c.empty:
                    continue

                sub_c = df_c[df_c['lag_time_s'] <= max_lag_s]
                s_lbl = STATE_NAMES.get(n_components, {}).get(s, f"State {s}")
                col = STATE_COLORS.get(s, f"C{s}")
                mrk = 'o' if s == 0 else 's'
                lsty = '--' if s == 0 else '-'

                fit_res = fits_dict.get(s, {}).get(ctype, {})
                tau_c = fit_res.get('tau_corr_s', np.nan)
                off_A = fit_res.get('offset_A', np.nan)

                label_str = f"{s_lbl}"
                if not np.isnan(tau_c):
                    if not np.isnan(off_A) and abs(off_A) > 1e-4:
                        label_str += f" ($\\tau={tau_c:.1f}\\,\\mathrm{{s}}, A={off_A:.2f}$)"
                    else:
                        label_str += f" ($\\tau={tau_c:.1f}\\,\\mathrm{{s}}$)"

                # データ点 + エラーバー
                if 'sem' in sub_c.columns and not sub_c['sem'].isna().all():
                    ax.errorbar(
                        sub_c['lag_time_s'], sub_c['corr'], yerr=sub_c['sem'],
                        fmt=mrk, color=col, ecolor=col, markersize=4, capsize=2,
                        label=label_str, zorder=3, alpha=0.85
                    )
                else:
                    ax.plot(
                        sub_c['lag_time_s'], sub_c['corr'],
                        marker=mrk, color=col, linestyle='none', markersize=4,
                        label=label_str, zorder=3, alpha=0.85
                    )

                # フィッティング曲線
                fit_t = fit_res.get('fit_t', np.array([]))
                fit_y = fit_res.get('fit_corr', np.array([]))
                if len(fit_t) > 0 and len(fit_y) > 0:
                    mask_t = fit_t <= max_lag_s
                    ax.plot(fit_t[mask_t], fit_y[mask_t], color=col, linestyle=lsty, lw=1.8, zorder=2, alpha=0.9)

            ax.grid(True, linestyle='--', alpha=0.4)
            ax.set_ylim(-0.25, 1.05)

            if row_idx == 0:
                ax.set_title(f"$d = {dia:.2f}\\,\\mu\\mathrm{{m}}$", fontsize=12, fontweight='bold')
            if col_idx == 0:
                ax.set_ylabel(ctitle, fontsize=10, fontweight='bold')
            if row_idx == 2:
                ax.set_xlabel(r"Lag Time $\tau$ [s]", fontsize=11)

            ax.legend(loc='upper right', fontsize=7.2, frameon=True, framealpha=0.9)

    fig.suptitle(
        r"State-Dependent Autocorrelation Functions & Offset Exponential Fits ($C(\tau) = (1-A)e^{-\tau/\tau_{\mathrm{corr}}} + A$)",
        fontsize=14,
        fontweight='bold',
    )
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    png_path = output_path.with_suffix('.png')
    if png_path != output_path:
        fig.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVED] {output_path} (and {png_path})", flush=True)


def plot_autocorrelation_timescales_vs_diameter(
    df_autocorr_summary: pd.DataFrame,
    df_state_summary: pd.DataFrame,
    output_path: Path,
    n_components: int = 2,
):
    """
    粒子径 vs 各自己相関緩和時間 (tau_VACF, tau_OACF, tau_SACF) および Dwell time 緩和時間 (tau_dwell) の比較プロット。
    """
    if df_autocorr_summary.empty:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))

    mode_colors = {
        'vacf': '#1f77b4',  # 青
        'oacf': '#2ca02c',  # 緑
        'sacf': '#9467bd',  # 紫
    }
    mode_labels = {
        'vacf': r'Velocity Vector $\tau_{\mathrm{VACF}}$',
        'oacf': r'Orientation $\tau_{\mathrm{OACF}}$',
        'sacf': r'Speed Fluctuation $\tau_{\mathrm{SACF}}$',
    }
    mode_markers = {
        'vacf': 'o',
        'oacf': '^',
        'sacf': 's',
    }

    # (a) Run モードの各緩和時間 vs 粒子径
    ax0 = axes[0]
    for mode in ['vacf', 'oacf', 'sacf']:
        sub = df_autocorr_summary[(df_autocorr_summary['state'] == 1) & (df_autocorr_summary['mode'] == mode)].sort_values(by='diameter_um')
        if sub.empty:
            continue
        ax0.errorbar(
            sub['diameter_um'], sub['tau_corr_s'], yerr=sub['tau_err_s'],
            marker=mode_markers[mode], color=mode_colors[mode], lw=2.0, capsize=3.5,
            markersize=6.5, label=mode_labels[mode], zorder=3
        )

    # Run Dwell time 緩和時間も併記
    df_run_state = df_state_summary[df_state_summary['state'] == 1].sort_values(by='diameter_um')
    if not df_run_state.empty:
        tau_col = 'tau_ccdf_s' if 'tau_ccdf_s' in df_run_state.columns else 'tau_fit_pdf_s'
        err_col = 'tau_ccdf_err_s' if 'tau_ccdf_err_s' in df_run_state.columns else 'dwell_err_s'
        ax0.errorbar(
            df_run_state['diameter_um'], df_run_state[tau_col], yerr=df_run_state[err_col],
            fmt='--d', color='#e41a1c', lw=2.2, capsize=4, markersize=7,
            label=r'Run Dwell CCDF $\tau_{\mathrm{Run}}^{\mathrm{dwell}}$', zorder=4
        )

    ax0.set_xscale('log')
    ax0.set_yscale('log')
    ax0.set_xlabel(r"Cargo Particle Diameter $d$ [$\mu\mathrm{m}$]", fontsize=11)
    ax0.set_ylabel(r"Relaxation Time $\tau$ [s]", fontsize=11)
    ax0.set_title(r"(a) Run State Relaxation Timescales", fontsize=12, fontweight='bold')
    ax0.set_xticks([0.63, 1.18, 3.37, 5.0, 7.24, 20.0])
    ax0.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax0.grid(True, which="both", linestyle='--', alpha=0.4)
    ax0.legend(loc='best', fontsize=8.5, frameon=True, framealpha=0.92)

    # (b) Tumble モードの各緩和時間 vs 粒子径
    ax1 = axes[1]
    for mode in ['vacf', 'oacf', 'sacf']:
        sub = df_autocorr_summary[(df_autocorr_summary['state'] == 0) & (df_autocorr_summary['mode'] == mode)].sort_values(by='diameter_um')
        if sub.empty:
            continue
        ax1.errorbar(
            sub['diameter_um'], sub['tau_corr_s'], yerr=sub['tau_err_s'],
            marker=mode_markers[mode], color=mode_colors[mode], lw=2.0, capsize=3.5,
            markersize=6.5, label=mode_labels[mode], zorder=3
        )

    df_tumble_state = df_state_summary[df_state_summary['state'] == 0].sort_values(by='diameter_um')
    if not df_tumble_state.empty:
        tau_col = 'tau_ccdf_s' if 'tau_ccdf_s' in df_tumble_state.columns else 'tau_fit_pdf_s'
        err_col = 'tau_ccdf_err_s' if 'tau_ccdf_err_s' in df_tumble_state.columns else 'dwell_err_s'
        ax1.errorbar(
            df_tumble_state['diameter_um'], df_tumble_state[tau_col], yerr=df_tumble_state[err_col],
            fmt='--d', color='#e41a1c', lw=2.2, capsize=4, markersize=7,
            label=r'Tumble Dwell CCDF $\tau_{\mathrm{Tumble}}^{\mathrm{dwell}}$', zorder=4
        )

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel(r"Cargo Particle Diameter $d$ [$\mu\mathrm{m}$]", fontsize=11)
    ax1.set_ylabel(r"Relaxation Time $\tau$ [s]", fontsize=11)
    ax1.set_title(r"(b) Tumble State Relaxation Timescales", fontsize=12, fontweight='bold')
    ax1.set_xticks([0.63, 1.18, 3.37, 5.0, 7.24, 20.0])
    ax1.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax1.grid(True, which="both", linestyle='--', alpha=0.4)
    ax1.legend(loc='best', fontsize=8.5, frameon=True, framealpha=0.92)

    # (c) Run vs Tumble の 速度・配向 緩和時間比
    ax2 = axes[2]
    sub_run_v = df_autocorr_summary[(df_autocorr_summary['state'] == 1) & (df_autocorr_summary['mode'] == 'vacf')].set_index('bead_name')
    sub_tum_v = df_autocorr_summary[(df_autocorr_summary['state'] == 0) & (df_autocorr_summary['mode'] == 'vacf')].set_index('bead_name')
    sub_run_o = df_autocorr_summary[(df_autocorr_summary['state'] == 1) & (df_autocorr_summary['mode'] == 'oacf')].set_index('bead_name')
    sub_tum_o = df_autocorr_summary[(df_autocorr_summary['state'] == 0) & (df_autocorr_summary['mode'] == 'oacf')].set_index('bead_name')

    dias = []
    ratio_vacf = []
    ratio_oacf = []
    for binfo in BEADS_INFO:
        bn = binfo['name']
        if bn in sub_run_v.index and bn in sub_tum_v.index:
            tv_r = sub_run_v.loc[bn, 'tau_corr_s']
            tv_t = sub_tum_v.loc[bn, 'tau_corr_s']
            to_r = sub_run_o.loc[bn, 'tau_corr_s'] if bn in sub_run_o.index else np.nan
            to_t = sub_tum_o.loc[bn, 'tau_corr_s'] if bn in sub_tum_o.index else np.nan

            dias.append(binfo['diameter_um'])
            ratio_vacf.append(tv_r / (tv_t + 1e-12) if not np.isnan(tv_r) and not np.isnan(tv_t) else np.nan)
            ratio_oacf.append(to_r / (to_t + 1e-12) if not np.isnan(to_r) and not np.isnan(to_t) else np.nan)

    ax2.axhline(1.0, color='gray', linestyle=':', lw=1.2, label='Equal Ratio (1.0)')
    if dias:
        ax2.plot(dias, ratio_vacf, marker='o', color='#1f77b4', lw=2.0, markersize=7, label=r'VACF Ratio $\tau_{\mathrm{Run}} / \tau_{\mathrm{Tumble}}$')
        ax2.plot(dias, ratio_oacf, marker='^', color='#2ca02c', lw=2.0, markersize=7, label=r'OACF Ratio $\tau_{\mathrm{Run}} / \tau_{\mathrm{Tumble}}$')

    ax2.set_xscale('log')
    ax2.set_xlabel(r"Cargo Particle Diameter $d$ [$\mu\mathrm{m}$]", fontsize=11)
    ax2.set_ylabel(r"Relaxation Time Ratio $\tau_{\mathrm{Run}} / \tau_{\mathrm{Tumble}}$", fontsize=11)
    ax2.set_title(r"(c) Run / Tumble Persistence Ratio", fontsize=12, fontweight='bold')
    ax2.set_xticks([0.63, 1.18, 3.37, 5.0, 7.24, 20.0])
    ax2.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax2.grid(True, which="both", linestyle='--', alpha=0.4)
    ax2.legend(loc='best', fontsize=8.5, frameon=True, framealpha=0.92)

    fig.suptitle(
        r"Characteristic Relaxation Timescales of Velocity, Speed and Orientation Autocorrelations vs Diameter",
        fontsize=13.5,
        fontweight='bold',
    )
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    png_path = output_path.with_suffix('.png')
    if png_path != output_path:
        fig.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVED] {output_path} (and {png_path})", flush=True)


def plot_markov_property_validation_4panel(
    fitted_results: Dict[str, dict],
    output_path: Path,
    n_components: int = 2,
):
    """
    マルコフ性（1次マルコフ連鎖仮定）の包括的検証プロット（4パネル）：
    - Panel (a): Chapman-Kolmogorov 固有緩和時間 (Implied Timescale) tau_implied vs Lag Time
    - Panel (b): 状態系列の自己相関関数 C_S(Delta t) vs マルコフ理論減衰
    - Panel (c): Chapman-Kolmogorov 遷移確率行列誤差 ||A_emp - A^n||_F vs Lag Time
    - Panel (d): 高次記憶効果の独立性検定 p 値 (Chi-Square) & Dwell Time 指数性検定 p 値 (KS-Test) vs 粒子径
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10.5))

    # Panel (a): Implied Timescales vs Lag Time
    ax0 = axes[0, 0]
    for binfo in BEADS_INFO:
        bname = binfo['name']
        dia = binfo['diameter_um']
        if bname not in fitted_results or 'markov_check' not in fitted_results[bname]:
            continue
        mres = fitted_results[bname]['markov_check']
        df_ck = mres.get('ck_df', pd.DataFrame())
        tau_relax = mres.get('tau_relax_s', np.nan)
        if df_ck.empty:
            continue

        valid_ck = df_ck.dropna(subset=['tau_implied_s'])
        ax0.plot(
            valid_ck['lag_time_s'], valid_ck['tau_implied_s'],
            marker=binfo['marker'], color=binfo['color'], lw=1.8,
            label=f"$d={dia:.2f}\\,\\mu\\mathrm{{m}}$ ($\\tau_0={tau_relax:.1f}\\,\\mathrm{{s}}$)",
        )
        if not np.isnan(tau_relax):
            ax0.axhline(tau_relax, color=binfo['color'], linestyle=':', alpha=0.5, lw=1.0)

    ax0.set_xlabel(r"Lag Time $\Delta t$ [s]", fontsize=11)
    ax0.set_ylabel(r"Implied Relaxation Timescale $\tau_{\mathrm{implied}}$ [s]", fontsize=11)
    ax0.set_title(r"(a) Chapman-Kolmogorov: Implied Timescale vs Lag $\Delta t$", fontsize=12, fontweight='bold')
    ax0.grid(True, linestyle='--', alpha=0.5)
    ax0.legend(loc='upper right', fontsize=8.5, frameon=True)

    # Panel (b): State Autocorrelation C_S(Delta t)
    ax1 = axes[0, 1]
    for binfo in BEADS_INFO:
        bname = binfo['name']
        dia = binfo['diameter_um']
        if bname not in fitted_results or 'markov_check' not in fitted_results[bname]:
            continue
        mres = fitted_results[bname]['markov_check']
        df_ac = mres.get('autocorr_df', pd.DataFrame())
        if df_ac.empty:
            continue

        ax1.plot(
            df_ac['lag_time_s'], df_ac['autocorr_emp'],
            marker=binfo['marker'], color=binfo['color'], linestyle='none',
            label=f"$d={dia:.2f}\\,\\mu\\mathrm{{m}}$ (Data)", markersize=5,
        )
        ax1.plot(
            df_ac['lag_time_s'], df_ac['autocorr_theo'],
            color=binfo['color'], linestyle='--', lw=1.3, alpha=0.75,
        )

    ax1.set_xlabel(r"Lag Time $\Delta t$ [s]", fontsize=11)
    ax1.set_ylabel(r"State Autocorrelation $C_S(\Delta t)$", fontsize=11)
    ax1.set_title(r"(b) State Autocorrelation vs Theoretical Markov Decay", fontsize=12, fontweight='bold')
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='upper right', fontsize=8.5, frameon=True)

    # Panel (c): Chapman-Kolmogorov Frobenius Error
    ax2 = axes[1, 0]
    for binfo in BEADS_INFO:
        bname = binfo['name']
        dia = binfo['diameter_um']
        if bname not in fitted_results or 'markov_check' not in fitted_results[bname]:
            continue
        mres = fitted_results[bname]['markov_check']
        df_ck = mres.get('ck_df', pd.DataFrame())
        if df_ck.empty:
            continue

        ax2.plot(
            df_ck['lag_time_s'], df_ck['frobenius_error'],
            marker=binfo['marker'], color=binfo['color'], lw=1.8,
            label=f"$d={dia:.2f}\\,\\mu\\mathrm{{m}}$",
        )

    ax2.set_xlabel(r"Lag Time $\Delta t$ [s]", fontsize=11)
    ax2.set_ylabel(r"CK Transition Matrix Error $\|\hat{A}(\Delta t) - A^{\Delta t/\tau}\|_F$", fontsize=11)
    ax2.set_title(r"(c) Chapman-Kolmogorov Matrix Discrepancy", fontsize=12, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='upper left', fontsize=8.5, frameon=True)

    # Panel (d): Statistical Independence p-values vs Diameter
    ax3 = axes[1, 1]
    ax3.axhspan(0.05, 1.05, color='#1b9e77', alpha=0.08, label=r"Markovian (Fail to reject $H_0$, $p \geq 0.05$)")
    ax3.axhline(0.05, color='crimson', linestyle=':', lw=1.5, label=r"Significance Threshold $\alpha = 0.05$")

    dia_list = []
    p_tumble_list = []
    p_run_list = []
    p_ks_tumble = []
    p_ks_run = []

    for binfo in BEADS_INFO:
        bname = binfo['name']
        dia = binfo['diameter_um']
        if bname not in fitted_results or 'markov_check' not in fitted_results[bname]:
            continue
        mres = fitted_results[bname]['markov_check']
        chi2_res = mres.get('chi2_results', {})
        ks_res = mres.get('ks_results', {})

        p_t0 = chi2_res.get(0, {}).get('p_value', np.nan)
        p_t1 = chi2_res.get(1, {}).get('p_value', np.nan)
        p_k0 = ks_res.get(0, {}).get('p_value', np.nan)
        p_k1 = ks_res.get(1, {}).get('p_value', np.nan)

        dia_list.append(dia)
        p_tumble_list.append(p_t0)
        p_run_list.append(p_t1)
        p_ks_tumble.append(p_k0)
        p_ks_run.append(p_k1)

    if dia_list:
        ax3.plot(dia_list, p_tumble_list, marker='o', color='#d95f02', lw=1.8, label=r"$\chi^2$ Memory Test: Tumble ($S_{t+1} \perp S_{t-1} \mid S_t$)")
        ax3.plot(dia_list, p_run_list, marker='s', color='#1b9e77', lw=1.8, label=r"$\chi^2$ Memory Test: Run ($S_{t+1} \perp S_{t-1} \mid S_t$)")
        ax3.plot(dia_list, p_ks_tumble, marker='^', color='#d95f02', linestyle='--', lw=1.3, alpha=0.7, label=r"KS Dwell Test: Tumble")
        ax3.plot(dia_list, p_ks_run, marker='v', color='#1b9e77', linestyle='--', lw=1.3, alpha=0.7, label=r"KS Dwell Test: Run")

    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.set_xlabel(r"Cargo Particle Diameter $d$ [$\mu\mathrm{m}$]", fontsize=11)
    ax3.set_ylabel(r"Statistical Test $p$-value", fontsize=11)
    ax3.set_title(r"(d) Markov Memory Independence & Dwell Tests ($p$-value)", fontsize=12, fontweight='bold')
    ax3.set_xticks([0.63, 1.18, 3.37, 5.0, 7.24, 20.0])
    ax3.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax3.grid(True, which="both", linestyle='--', alpha=0.4)
    ax3.legend(loc='lower left', fontsize=8.0, frameon=True, framealpha=0.92)

    fig.suptitle(
        r"Validation of Markovian Dynamics: Chapman-Kolmogorov, Implied Timescales & Memory Independence",
        fontsize=13.5,
        fontweight='bold',
    )
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    png_path = output_path.with_suffix('.png')
    if png_path != output_path:
        fig.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVED] {output_path} (and {png_path})", flush=True)


def evaluate_model_selection_bic(
    exp_dirs_by_bead: Dict[str, List[Path]],
    output_path: Path,
    tau: int = 1,
    scale: float = 0.11,
    frame_interval: float = 4.0,
    epsilon: float = 1e-3,
    max_k: int = 4,
) -> pd.DataFrame:
    records = []
    print(f"\n=== Evaluating Model Selection (BIC / AIC) for K=1..{max_k} (1D Speed-Only) ===", flush=True)

    for binfo in BEADS_INFO:
        bname = binfo['name']
        dia = binfo['diameter_um']
        edirs = exp_dirs_by_bead.get(bname, [])
        if not edirs:
            continue

        X, lengths, _ = collect_bead_hmm_data(edirs, tau=tau, scale=scale, frame_interval=frame_interval, epsilon=epsilon)
        if len(X) < 30:
            continue

        for k in range(1, max_k + 1):
            try:
                if k == 2:
                    if bname == 'beads5um':
                        im = np.array([[-3.0], [0.0]])
                        ic = np.array([[[1.0]], [[0.1]]])
                        isp = np.array([0.8, 0.2])
                        it = np.array([[0.95, 0.05], [0.1, 0.9]])
                    else:
                        im = np.array([[-3.5], [0.0]])
                        ic, isp, it = None, None, None
                    model = hc.CargoGaussianHMM(
                        n_components=k, epsilon=epsilon, random_state=42,
                        init_means=im, init_covars=ic, init_transmat=it, init_startprob=isp,
                    )
                else:
                    model = hc.CargoGaussianHMM(n_components=k, epsilon=epsilon, random_state=42)

                model.fit(X, lengths=lengths)
                bic_val, aic_val = model.compute_bic_aic(X, lengths=lengths)
                log_lik = model.score(X, lengths=lengths)

                records.append({
                    'bead_name': bname,
                    'diameter_um': dia,
                    'k_components': k,
                    'n_samples': len(X),
                    'log_likelihood': log_lik,
                    'bic': bic_val,
                    'aic': aic_val,
                    'bic_per_sample': bic_val / len(X),
                    'aic_per_sample': aic_val / len(X),
                })
                print(f"  {bname} (d={dia:.2f}um) K={k}: logLik={log_lik:.1f}, BIC={bic_val:.1f}, AIC={aic_val:.1f}", flush=True)
            except Exception as e:
                print(f"  [ERROR] {bname} K={k} fitting failed: {e}", flush=True)

    if not records:
        return pd.DataFrame()

    df_bic = pd.DataFrame(records)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for binfo in BEADS_INFO:
        bname = binfo['name']
        dia = binfo['diameter_um']
        df_b = df_bic[df_bic['bead_name'] == bname].sort_values(by='k_components')
        if df_b.empty:
            continue

        df_k2 = df_b[df_b['k_components'] == 2]
        if df_k2.empty:
            bic_base = df_b['bic'].iloc[0]
            aic_base = df_b['aic'].iloc[0]
        else:
            bic_base = df_k2.iloc[0]['bic']
            aic_base = df_k2.iloc[0]['aic']

        axes[0].plot(df_b['k_components'], df_b['bic'] - bic_base, marker=binfo['marker'], color=binfo['color'], label=f"$d={dia:.2f}\\,\\mu\\mathrm{{m}}$", lw=1.8)
        axes[1].plot(df_b['k_components'], df_b['aic'] - aic_base, marker=binfo['marker'], color=binfo['color'], label=f"$d={dia:.2f}\\,\\mu\\mathrm{{m}}$", lw=1.8)

    axes[0].set_title(r"$\Delta$BIC vs Number of States $K$ (1D Speed-Only)", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Number of Hidden States $K$", fontsize=11)
    axes[0].set_ylabel(r"$\Delta\mathrm{BIC} = \mathrm{BIC}(K) - \mathrm{BIC}(2)$", fontsize=11)
    axes[0].set_xticks(range(1, max_k + 1))
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].legend(loc='best', fontsize=9, frameon=True)

    axes[1].set_title(r"$\Delta$AIC vs Number of States $K$ (1D Speed-Only)", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Number of Hidden States $K$", fontsize=11)
    axes[1].set_ylabel(r"$\Delta\mathrm{AIC} = \mathrm{AIC}(K) - \mathrm{AIC}(2)$", fontsize=11)
    axes[1].set_xticks(range(1, max_k + 1))
    axes[1].grid(True, linestyle='--', alpha=0.5)

    fig.suptitle("HMM Model Selection via Information Criteria (1D Speed-Only Model)", fontsize=14, fontweight='bold')
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    png_path = output_path.with_suffix('.png')
    if png_path != output_path:
        fig.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVED] {output_path} (and {png_path})", flush=True)

    return df_bic


def main():
    parser = argparse.ArgumentParser(
        description="Cargo particle 1D Gaussian HMM motion mode analysis using O_t = [ln(v+eps)]"
    )
    parser.add_argument("--root_dir", type=str, default=None, help="Root directory containing bead experiment folders")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (default: figure/hmm_1d)")
    parser.add_argument("--beads", type=str, default="all", help="Target beads (all, beads06um, beads1um, ...)")
    parser.add_argument("--n_components", type=int, default=2, help="Number of hidden states K (default: 2)")
    parser.add_argument("--covariance_type", type=str, default="full", choices=["full", "diag", "spherical", "tied"], help="Covariance type")
    parser.add_argument("--tau", type=int, default=1, help="Lag time in frames (default: 1)")
    parser.add_argument("--scale", type=float, default=0.11, help="Spatial scale in um/pixel (default: 0.11)")
    parser.add_argument("--frame_interval", type=float, default=4.0, help="Frame interval in seconds (default: 4.0)")
    parser.add_argument("--epsilon", type=float, default=1e-3, help="Epsilon for ln(v + epsilon) in um/s (default: 1e-3)")
    parser.add_argument("--init_means_tumble", type=float, default=-3.5, help="Initial log-speed mean for Tumble state in K=2 (default: -3.5)")
    parser.add_argument("--init_means_run", type=float, default=0.0, help="Initial log-speed mean for Run state in K=2 (default: 0.0)")
    parser.add_argument("--min_dwell_frames", type=int, default=2, help="Minimum dwell duration in frames for isolated glitch filter (default: 2 = 8s, set <=1 to disable)")
    parser.add_argument("--eval_bic", action="store_true", help="Evaluate BIC/AIC for K=1..4 and generate ΔBIC(2->1) vs diameter plot")
    parser.add_argument("--save_csv", action="store_true", default=True, help="Save summary CSV tables")

    args = parser.parse_args()

    root_dir = Path(args.root_dir) if args.root_dir else find_default_root()
    output_dir = root_dir / "figure" / "hmm_1d"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=================================================================")
    print("      Cargo Particle 1D Speed Gaussian HMM Motion Analysis       ")
    print("=================================================================")
    print(f"Root dir:         {root_dir}")
    print(f"Output dir:       {output_dir}")
    print(f"Hidden states K:  {args.n_components}")
    print(f"Covariance type:  {args.covariance_type}")
    print(f"Lag tau:          {args.tau} ({args.tau * args.frame_interval:.1f} s)")
    print(f"Epsilon:          {args.epsilon} um/s")
    print(f"Observation O_t:  [ln(v + {args.epsilon})]")
    if args.n_components == 2:
        print(f"Init means (K=2): Tumble={args.init_means_tumble}, Run={args.init_means_run}")
    if args.min_dwell_frames >= 2:
        print(f"Glitch filter:    Enabled (min_dwell >= {args.min_dwell_frames} frames = {args.min_dwell_frames * args.frame_interval:.1f} s)")
    print("=================================================================\n")

    if args.beads == "all":
        target_bead_infos = BEADS_INFO
    else:
        target_bead_infos = [b for b in BEADS_INFO if b['name'] == args.beads]
        if not target_bead_infos:
            print(f"[ERROR] Unknown bead name '{args.beads}'. Available: {[b['name'] for b in BEADS_INFO]}")
            return

    exp_dirs_by_bead = {}
    fitted_results = {}
    all_summaries = []
    all_trans_records = []
    all_conf_records = []
    all_msd_curves = []
    all_msd_fits = []
    all_autocorr_records = []
    all_abp_records = []

    for binfo in target_bead_infos:
        bname = binfo['name']
        dia = binfo['diameter_um']
        print(f"--- Processing {bname} (diameter: {dia:.2f} um) ---")

        edirs = find_experiment_dirs(root_dir, bname)
        exp_dirs_by_bead[bname] = edirs
        if not edirs:
            print(f"[WARNING] No experiment directories found for {bname} in {root_dir}")
            continue

        print(f"  Found {len(edirs)} experiment directories.")

        X, lengths, df_obs = collect_bead_hmm_data(
            edirs,
            tau=args.tau,
            scale=args.scale,
            frame_interval=args.frame_interval,
            epsilon=args.epsilon,
        )

        if len(X) < 20:
            print(f"[WARNING] Insufficient data points ({len(X)}) for {bname}. Skipping.")
            continue

        print(f"  Extracted {len(X):,} observation points across {len(lengths):,} tracks.")

        # 2状態モデル (K=2) の初期値設定:
        init_means = None
        init_covars = None
        init_transmat = None
        init_startprob = None

        if args.n_components == 2:
            if bname == 'beads5um':
                init_means = np.array([[-3.0], [args.init_means_run]])
                init_covars = np.array([[[1.0]], [[0.1]]])
                init_startprob = np.array([0.8, 0.2])
                init_transmat = np.array([[0.95, 0.05], [0.1, 0.9]])
            else:
                init_means = np.array([[args.init_means_tumble], [args.init_means_run]])

        hmm_model = hc.CargoGaussianHMM(
            n_components=args.n_components,
            covariance_type=args.covariance_type,
            epsilon=args.epsilon,
            random_state=42,
            init_means=init_means,
            init_covars=init_covars,
            init_transmat=init_transmat,
            init_startprob=init_startprob,
        )
        hmm_model.fit(X, lengths=lengths)

        raw_pred_states = hmm_model.predict(X, lengths=lengths)
        proba = hmm_model.predict_proba(X, lengths=lengths)

        if args.min_dwell_frames >= 2:
            pred_states = hc.filter_state_glitches(raw_pred_states, lengths, min_duration_frames=args.min_dwell_frames)
            n_glitches = int(np.sum(raw_pred_states != pred_states))
            if n_glitches > 0:
                print(f"  Applied {args.min_dwell_frames}-frame glitch filter: smoothed {n_glitches:,} isolated 1-frame spikes ({n_glitches/len(pred_states)*100:.2f}%).", flush=True)
        else:
            pred_states = raw_pred_states

        df_obs['pred_state'] = pred_states

        dwell_times = hc.calc_state_dwell_times(pred_states, lengths, frame_interval=args.frame_interval)
        conf_stats = hc.calc_posterior_statistics(proba, n_components=args.n_components)
        conf_rec = {'bead_name': bname, 'diameter_um': dia, **conf_stats}
        all_conf_records.append(conf_rec)

        df_msd, df_fits = hc.calc_state_dependent_msd(
            df_obs,
            max_tau=25,
            frame_interval=args.frame_interval,
            n_components=args.n_components,
            fit_min_tau=1,
            fit_max_tau=10,
        )
        df_msd['bead_name'] = bname
        df_msd['diameter_um'] = dia
        df_fits['bead_name'] = bname
        df_fits['diameter_um'] = dia
        all_msd_curves.append(df_msd)
        all_msd_fits.append(df_fits)

        df_state_sum = hmm_model.get_state_summary(frame_interval=args.frame_interval)
        df_state_sum['bead_name'] = bname
        df_state_sum['diameter_um'] = dia
        df_state_sum['n_observations'] = len(X)
        df_state_sum['n_tracks'] = len(lengths)

        state_counts = [int(np.sum(pred_states == s)) for s in range(args.n_components)]
        df_state_sum['n_state_obs'] = state_counts
        df_state_sum['empirical_prob'] = [cnt / (len(pred_states) + 1e-12) for cnt in state_counts]

        dwell_fits_dict = {}
        dwell_ccdf_fits_dict = {}
        empirical_dwells = []
        tau_ccdf_fits = []
        tau_ccdf_errors = []
        r2_ccdf_fits = []
        amplitudes_ccdf = []
        tau_pdf_fits = []
        dwell_errors_pdf = []
        r2_pdf_fits = []
        for s in range(args.n_components):
            t_list = dwell_times.get(s, [])
            fit_pdf = hc.fit_exponential_pdf(t_list, frame_interval=args.frame_interval)
            fit_ccdf = hc.fit_exponential_ccdf(t_list)
            dwell_fits_dict[s] = fit_pdf
            dwell_ccdf_fits_dict[s] = fit_ccdf
            empirical_dwells.append(fit_ccdf['mean_empirical_s'])
            tau_ccdf_fits.append(fit_ccdf['tau_fit_s'])
            tau_ccdf_errors.append(fit_ccdf['tau_err_s'])
            r2_ccdf_fits.append(fit_ccdf['r2_log'])
            amplitudes_ccdf.append(fit_ccdf['amplitude_A'])
            tau_pdf_fits.append(fit_pdf['tau_fit_s'])
            dwell_errors_pdf.append(fit_pdf['tau_err_s'])
            r2_pdf_fits.append(fit_pdf['r2_log'])

        df_state_sum['mean_dwell_emp_s'] = empirical_dwells
        df_state_sum['tau_fit_s'] = tau_ccdf_fits
        df_state_sum['tau_ccdf_s'] = tau_ccdf_fits
        df_state_sum['tau_ccdf_err_s'] = tau_ccdf_errors
        df_state_sum['dwell_err_s'] = tau_ccdf_errors
        df_state_sum['ccdf_exp_r2_log'] = r2_ccdf_fits
        df_state_sum['tau_fit_pdf_s'] = tau_pdf_fits
        df_state_sum['dwell_err_pdf_s'] = dwell_errors_pdf
        df_state_sum['dwell_exp_r2_log'] = r2_pdf_fits
        df_state_sum['amplitude_A'] = amplitudes_ccdf
        all_summaries.append(df_state_sum)

        A = hmm_model.model.transmat_
        markov_res = hc.check_markov_property(
            pred_states,
            lengths,
            A,
            frame_interval=args.frame_interval,
            max_lag_steps=15,
            dwell_times=dwell_times,
        )

        for i in range(args.n_components):
            for j in range(args.n_components):
                all_trans_records.append({
                    'bead_name': bname,
                    'diameter_um': dia,
                    'from_state': i,
                    'to_state': j,
                    'trans_prob': A[i, j],
                })

        # 状態別方向転換角 Δθ の抽出
        turning_angles = hc.calc_state_dependent_turning_angles(
            df_obs,
            n_components=args.n_components,
        )

        # 状態別自己相関（VACF, OACF, SACF）の算出 & 指数減衰フィッティング
        autocorr_data = hc.calc_state_dependent_autocorrelations(
            df_obs,
            n_components=args.n_components,
            frame_interval=args.frame_interval,
            max_lag_frames=25,
        )
        autocorr_fits = {}
        for s in range(args.n_components):
            autocorr_fits[s] = {}
            s_label = "Tumble / Pause" if s == 0 else ("Run" if s == 1 else f"State {s}")
            for ctype in ['vacf', 'oacf', 'sacf']:
                if s in autocorr_data and ctype in autocorr_data[s]:
                    df_c = autocorr_data[s][ctype]
                    f_res = hc.fit_autocorrelation_exponential(df_c, max_lag_s=60.0)
                    autocorr_fits[s][ctype] = f_res
                    all_autocorr_records.append({
                        'bead_name': bname,
                        'diameter_um': dia,
                        'state': s,
                        'state_label': s_label,
                        'mode': ctype,
                        'tau_corr_s': f_res.get('tau_corr_s', np.nan),
                        'tau_err_s': f_res.get('tau_err_s', np.nan),
                        'offset_A': f_res.get('offset_A', np.nan),
                        'offset_A_err': f_res.get('offset_A_err', np.nan),
                        'r_squared': f_res.get('r_squared', np.nan),
                        'count': f_res.get('count', 0),
                    })

        # Run 状態の MSD に対するアクティブブラウニアン粒子 (ABP) モデルフィッティング
        sub_run_msd = df_msd[df_msd['state'] == 1].sort_values(by='lag_time_s')
        v0_run = float(df_state_sum.loc[df_state_sum['state'] == 1, 'mean_speed_geom_um_s'].iloc[0]) if np.sum(df_state_sum['state'] == 1) > 0 else 0.0
        if not sub_run_msd.empty and v0_run > 0:
            abp_fit_res = hc.fit_active_brownian_msd(
                sub_run_msd['lag_time_s'].values,
                sub_run_msd['msd_um2'].values,
                v0=v0_run,
                sem_vals=sub_run_msd['msd_sem_um2'].values if 'msd_sem_um2' in sub_run_msd.columns else None,
            )
            all_abp_records.append({
                'bead_name': bname,
                'diameter_um': dia,
                'v0_geom_um_s': v0_run,
                'Dt_um2_s': abp_fit_res.get('Dt_um2_s', np.nan),
                'Dt_err_um2_s': abp_fit_res.get('Dt_err_um2_s', np.nan),
                'tau_r_s': abp_fit_res.get('tau_r_s', np.nan),
                'tau_r_err_s': abp_fit_res.get('tau_r_err_s', np.nan),
                'D_eff_um2_s': abp_fit_res.get('D_eff_um2_s', np.nan),
                'lambda_p_um': abp_fit_res.get('lambda_p_um', np.nan),
                'r_squared': abp_fit_res.get('r_squared', np.nan),
                'fit_points': abp_fit_res.get('count', 0),
            })
        else:
            abp_fit_res = {}

        fitted_results[bname] = {
            'X': X,
            'lengths': lengths,
            'df_obs': df_obs,
            'model': hmm_model,
            'pred_states': pred_states,
            'proba': proba,
            'conf_stats': conf_stats,
            'dwell_times': dwell_times,
            'dwell_fits': dwell_fits_dict,
            'dwell_ccdf_fits': dwell_ccdf_fits_dict,
            'turning_angles': turning_angles,
            'autocorrelations': autocorr_data,
            'autocorr_fits': autocorr_fits,
            'abp_fit': abp_fit_res,
            'markov_check': markov_res,
            'summary': df_state_sum,
            'df_msd': df_msd,
            'df_msd_fits': df_fits,
        }

        print("  State summary:")
        for _, srow in df_state_sum.iterrows():
            print(
                f"    State {int(srow['state'])} ({srow['label']}): "
                f"v_geom={srow['mean_speed_geom_um_s']:.3f} um/s, "
                f"frac={srow['stationary_prob']*100:.1f}%, "
                f"tau_fit={srow['tau_fit_pdf_s']:.1f}s (mean={srow['mean_dwell_emp_s']:.1f}s, R2_log={srow['dwell_exp_r2_log']:.2f})"
            )
        print(f"  Confidence: mean={conf_stats.get('mean_confidence', 0):.3f}, high_conf_frac={conf_stats.get('high_conf_ratio_80', 0)*100:.1f}%")

    if not fitted_results:
        print("[ERROR] No models were fitted.")
        return

    df_all_summary = pd.concat(all_summaries, ignore_index=True)
    df_trans = pd.DataFrame(all_trans_records)
    df_conf_all = pd.DataFrame(all_conf_records)
    df_msd_curves_all = pd.concat(all_msd_curves, ignore_index=True)
    df_msd_fits_all = pd.concat(all_msd_fits, ignore_index=True)

    print("\n=== Generating Figures ===")

    # 1. 1D 放出確率密度分布 (6パネル)
    fig1_path = output_dir / f"hmm_emission_density_k{args.n_components}.svg"
    hc.plot_emission_1d_distribution_6panel(
        fitted_results,
        BEADS_INFO,
        fig1_path,
        n_components=args.n_components,
        epsilon=args.epsilon,
        state_names=STATE_NAMES.get(args.n_components, {0: "Tumble / Pause", 1: "Run"}),
        state_colors=STATE_COLORS,
    )

    # 2. 軌跡のセグメンテーション描画 (6パネル)
    fig2_path = output_dir / f"hmm_trajectories_k{args.n_components}.svg"
    plot_trajectory_segmentation(fitted_results, fig2_path, n_components=args.n_components)

    # 3. 代表粒子の時系列同期プロット
    fig3_path = output_dir / f"hmm_timeseries_sync_k{args.n_components}.svg"
    plot_timeseries_sync(
        fitted_results,
        fig3_path,
        n_components=args.n_components,
        frame_interval=args.frame_interval,
    )

    # 4. 持続時間分布 (PDF & CCDF)
    fig4_path = output_dir / f"hmm_dwell_time_distributions_k{args.n_components}.svg"
    plot_dwell_time_distributions(fitted_results, fig4_path, n_components=args.n_components, frame_interval=args.frame_interval)

    # 4b. 状態別 Dwell time CCDF 指数分布フィッティング (6パネル)
    fig4b_path = output_dir / f"hmm_dwell_time_ccdf_fit_6panel_k{args.n_components}.svg"
    plot_dwell_time_ccdf_fit_6panel(fitted_results, fig4b_path, n_components=args.n_components, frame_interval=args.frame_interval)

    # 4b2. 状態別 Dwell time PDF 指数分布フィッティング (6パネル)
    fig4b2_path = output_dir / f"hmm_dwell_time_pdf_fit_6panel_k{args.n_components}.svg"
    plot_dwell_time_pdf_fit_6panel(fitted_results, fig4b2_path, n_components=args.n_components, frame_interval=args.frame_interval)

    # 4c. マルコフ性（1次マルコフ連鎖仮定）の包括的検証 (4パネル)
    if len(fitted_results) > 1 and args.n_components == 2:
        fig4c_path = output_dir / f"hmm_markov_property_validation_4panel_k{args.n_components}.svg"
        plot_markov_property_validation_4panel(fitted_results, fig4c_path, n_components=args.n_components)

    # 5. 粒子径 vs パラメータ サマリー (4パネル)
    if len(fitted_results) > 1:
        fig5_path = output_dir / f"hmm_summary_vs_diameter_k{args.n_components}.svg"
        plot_summary_vs_diameter(df_all_summary, fig5_path, n_components=args.n_components)

    # 6. 遷移確率行列ヒートマップ (6パネル)
    fig6_path = output_dir / f"hmm_transition_matrices_k{args.n_components}.svg"
    plot_transition_matrices_6panel(fitted_results, fig6_path, n_components=args.n_components)

    # 7. 事後確率分布 (6パネル)
    fig7_path = output_dir / f"hmm_posterior_distributions_k{args.n_components}.svg"
    plot_posterior_distributions_6panel(fitted_results, fig7_path, n_components=args.n_components)

    # 8. 確信度指標 vs 粒子径 (3パネル)
    if len(fitted_results) > 1:
        fig8_path = output_dir / f"hmm_confidence_vs_diameter_k{args.n_components}.svg"
        plot_confidence_vs_diameter(df_conf_all, fig8_path)

    # 9. 状態別 MSD 曲線 (6パネル)
    fig9_path = output_dir / f"hmm_state_msd_k{args.n_components}.svg"
    plot_state_dependent_msd_6panel(fitted_results, fig9_path, n_components=args.n_components)

    # 10. 状態別 MSD パラメータ (alpha, D) vs 粒子径
    if len(fitted_results) > 1:
        fig10_path = output_dir / f"hmm_state_msd_params_vs_diameter_k{args.n_components}.png"
        plot_state_msd_params_vs_diameter(df_msd_fits_all, fig10_path, n_components=args.n_components)

    # 11. 状態分離度 S_v vs 粒子径 (2パネル)
    if len(fitted_results) > 1 and args.n_components == 2:
        fig_sep_path = output_dir / "hmm_separation_index_vs_diameter.svg"
        df_sep = plot_separation_index_vs_diameter(df_all_summary, fig_sep_path)
        if not df_sep.empty and args.save_csv:
            safe_save_csv(df_sep, output_dir / "hmm_separation_index_summary.csv")

    # 11b. Run 緩和時間 tau_Run vs 粒子径 (2パネル: tau_Run & Run Length lambda_Run)
    df_run_tau = pd.DataFrame()
    if len(fitted_results) > 1 and args.n_components == 2:
        fig_run_tau_path = output_dir / "hmm_run_relaxation_time_vs_diameter.svg"
        df_run_tau = plot_run_relaxation_time_vs_diameter(df_all_summary, fig_run_tau_path)
        if not df_run_tau.empty and args.save_csv:
            safe_save_csv(df_run_tau, output_dir / "hmm_run_relaxation_time_summary.csv")

    # 11c. デューティ比 f_run vs 粒子径 (2パネル: 観測占有率 vs マルコフ定常理論値 & 整合比)
    df_duty = pd.DataFrame()
    if len(fitted_results) > 1 and args.n_components == 2:
        fig_duty_path = output_dir / "hmm_duty_cycle_vs_diameter.svg"
        df_duty = plot_duty_cycle_vs_diameter(df_all_summary, fig_duty_path)
        if not df_duty.empty and args.save_csv:
            safe_save_csv(df_duty, output_dir / "hmm_duty_cycle_summary_k2.csv")

    # 11d. MSD 完全データコラプス (Master Curve: \widetilde{MSD} = 2 [\widetilde{\Delta t} - 1 + e^{-\widetilde{\Delta t}}])
    if len(fitted_results) > 1 and args.n_components == 2:
        fig_collapse_path = output_dir / "hmm_msd_data_collapse_master_curve.svg"
        df_collapse = plot_msd_data_collapse(df_msd_curves_all, df_all_summary, fig_collapse_path)
        if not df_collapse.empty and args.save_csv:
            safe_save_csv(df_collapse, output_dir / "hmm_msd_data_collapse_summary.csv")

    # 11e. 状態別方向転換角分布 P(Δθ) (6パネル直交座標, 6パネル極座標, 集中度κ & <cosΔθ> vs 粒子径)
    df_angle_sum = pd.DataFrame()
    if len(fitted_results) > 1 and args.n_components == 2:
        fig_angle_dist_path = output_dir / "hmm_turning_angle_distributions_k2.svg"
        df_angle_sum = plot_state_turning_angle_distributions_6panel(fitted_results, fig_angle_dist_path, n_components=args.n_components)

        fig_angle_polar_path = output_dir / "hmm_turning_angle_polar_k2.svg"
        plot_turning_angle_polar_6panel(fitted_results, fig_angle_polar_path, n_components=args.n_components)

        fig_angle_sum_path = output_dir / "hmm_turning_angle_summary_vs_diameter.svg"
        plot_turning_angle_summary_vs_diameter(df_angle_sum, fig_angle_sum_path, n_components=args.n_components)

        if not df_angle_sum.empty and args.save_csv:
            safe_save_csv(df_angle_sum, output_dir / "hmm_turning_angle_summary_k2.csv")

    # 11f. 状態別自己相関（VACF, OACF, SACF）グリッドプロット & 緩和時間 vs 粒子径
    df_autocorr_summary = pd.DataFrame(all_autocorr_records)
    if len(fitted_results) > 1 and args.n_components == 2:
        fig_ac_grid_path = output_dir / "hmm_autocorrelation_grid_k2.svg"
        plot_state_autocorrelations_grid(fitted_results, fig_ac_grid_path, n_components=args.n_components)

        fig_ac_tau_path = output_dir / "hmm_autocorrelation_timescales_vs_diameter_k2.svg"
        plot_autocorrelation_timescales_vs_diameter(df_autocorr_summary, df_all_summary, fig_ac_tau_path, n_components=args.n_components)

        if not df_autocorr_summary.empty and args.save_csv:
            safe_save_csv(df_autocorr_summary, output_dir / "hmm_autocorrelation_summary_k2.csv")

    # 11g. Run 状態のアクティブブラウニアン (ABP) MSD フィッティング (6パネル) & パラメータ vs 粒子径
    df_abp_summary = pd.DataFrame(all_abp_records)
    if len(fitted_results) > 1 and args.n_components == 2:
        fig_abp_msd_path = output_dir / "hmm_run_abp_msd_fit_6panel_k2.svg"
        plot_run_abp_msd_fit_6panel(fitted_results, fig_abp_msd_path)

        fig_abp_params_path = output_dir / "hmm_abp_parameters_vs_diameter_k2.svg"
        plot_abp_parameters_vs_diameter(df_abp_summary, df_all_summary, fig_abp_params_path)

        if not df_abp_summary.empty and args.save_csv:
            safe_save_csv(df_abp_summary, output_dir / "hmm_run_abp_fits_summary_k2.csv")

    # 12. BIC / AIC モデル選択評価 & ΔBIC_{2->1} 解析
    if args.eval_bic or (len(fitted_results) > 1 and args.beads == "all"):
        fig11_path = output_dir / "hmm_model_selection_bic.svg"
        df_bic = evaluate_model_selection_bic(
            exp_dirs_by_bead,
            fig11_path,
            tau=args.tau,
            scale=args.scale,
            frame_interval=args.frame_interval,
            epsilon=args.epsilon,
        )
        if not df_bic.empty and args.save_csv:
            safe_save_csv(df_bic, output_dir / "hmm_model_selection_bic.csv")

    # CSV 保存
    if args.save_csv:
        csv1_path = output_dir / f"hmm_state_parameters_summary_k{args.n_components}.csv"
        csv2_path = output_dir / f"hmm_transition_matrices_k{args.n_components}.csv"
        csv3_path = output_dir / f"hmm_posterior_confidence_summary_k{args.n_components}.csv"
        csv4_path = output_dir / f"hmm_state_msd_curves_k{args.n_components}.csv"
        csv5_path = output_dir / f"hmm_state_msd_fits_k{args.n_components}.csv"
        csv6_path = output_dir / f"hmm_dwell_time_fits_summary_k{args.n_components}.csv"
        csv7_path = output_dir / f"hmm_markov_property_summary_k{args.n_components}.csv"
        csv8_path = output_dir / f"hmm_autocorrelation_summary_k{args.n_components}.csv"
        csv9_path = output_dir / f"hmm_run_abp_fits_summary_k{args.n_components}.csv"

        safe_save_csv(df_all_summary, csv1_path)
        safe_save_csv(df_trans, csv2_path)
        safe_save_csv(df_conf_all, csv3_path)
        safe_save_csv(df_msd_curves_all, csv4_path)
        safe_save_csv(df_msd_fits_all, csv5_path)
        safe_save_csv(df_all_summary[['bead_name', 'diameter_um', 'state', 'label', 'tau_fit_pdf_s', 'dwell_err_s', 'amplitude_A', 'mean_dwell_emp_s', 'theoretical_dwell_time_s', 'dwell_exp_r2_log']], csv6_path)
        if not df_autocorr_summary.empty:
            safe_save_csv(df_autocorr_summary, csv8_path)
        if not df_abp_summary.empty:
            safe_save_csv(df_abp_summary, csv9_path)

        # マルコフ性サマリーテーブル
        markov_records = []
        for binfo in BEADS_INFO:
            bname = binfo['name']
            if bname in fitted_results and 'markov_check' in fitted_results[bname]:
                mres = fitted_results[bname]['markov_check']
                c2 = mres.get('chi2_results', {})
                ks = mres.get('ks_results', {})
                p_chi2_0 = c2.get(0, {}).get('p_value', np.nan)
                p_chi2_1 = c2.get(1, {}).get('p_value', np.nan)
                p_ks_0 = ks.get(0, {}).get('p_value', np.nan)
                p_ks_1 = ks.get(1, {}).get('p_value', np.nan)
                markov_records.append({
                    'bead_name': bname,
                    'diameter_um': binfo['diameter_um'],
                    'tau_relax_s': mres.get('tau_relax_s', np.nan),
                    'mean_ck_frobenius_error': mres.get('mean_ck_error', np.nan),
                    'chi2_p_tumble': p_chi2_0,
                    'chi2_p_run': p_chi2_1,
                    'chi2_markovian_p05': (p_chi2_0 >= 0.05) and (p_chi2_1 >= 0.05),
                    'ks_p_tumble': p_ks_0,
                    'ks_p_run': p_ks_1,
                })
        if markov_records:
            df_markov_sum = pd.DataFrame(markov_records)
            safe_save_csv(df_markov_sum, csv7_path)
            print(f"[SAVED] {csv7_path}")

        print(f"[SAVED] {csv1_path}")
        print(f"[SAVED] {csv2_path}")
        print(f"[SAVED] {csv3_path}")
        print(f"[SAVED] {csv4_path}")
        print(f"[SAVED] {csv5_path}")
        print(f"[SAVED] {csv6_path}")

    print("\n=================================================================")
    print("      HMM Cargo Motion Mode Analysis Completed Successfully!     ")
    print("=================================================================")


if __name__ == '__main__':
    main()
