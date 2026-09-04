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
10. モデル選択基準 (BIC / AIC vs 状態数 K) および ΔBIC_{2->1} 評価
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
            return
        except Exception as e:
            if attempt == max_retries - 1:
                try:
                    csv_text = df.to_csv(index=False)
                    with open(str(target_path), 'w', encoding='utf-8') as f:
                        f.write(csv_text)
                    return
                except Exception:
                    print(f"[WARNING] Could not save {target_path}: {e}", flush=True)
                    return
            time.sleep(0.5)


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

            dwells = fitted_results[bname]['dwell_times'].get(s, [])
            if len(dwells) < 5:
                continue

            arr = np.asarray(dwells)
            bins = np.logspace(np.log10(frame_interval), np.log10(max(arr) * 1.2), 20)
            hist, bin_edges = np.histogram(arr, bins=bins, density=True)
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            valid = (hist > 0)
            ax_pdf.plot(bin_centers[valid], hist[valid], marker=mrk, color=col, label=f"$d={dia:.2f}\\,\\mu\\mathrm{{m}}$", lw=1.5)

            sorted_d = np.sort(arr)
            ccdf = 1.0 - (np.arange(1, len(sorted_d) + 1) - 0.5) / len(sorted_d)
            ax_ccdf.plot(sorted_d, ccdf, marker=mrk, color=col, label=f"$d={dia:.2f}\\,\\mu\\mathrm{{m}}$", lw=1.5)

        ax_pdf.set_xscale('log')
        ax_pdf.set_yscale('log')
        ax_pdf.set_xlabel("Dwell Time $t$ [s]", fontsize=11)
        ax_pdf.set_ylabel(f"PDF $P(t)$ ({s_lbl})", fontsize=11)
        ax_pdf.set_title(f"(a) Dwell Time PDF: {s_lbl}", fontsize=12, fontweight='bold')
        ax_pdf.grid(True, linestyle='--', alpha=0.5)
        ax_pdf.legend(loc='best', fontsize=8.5, frameon=True)

        ax_ccdf.set_xscale('log')
        ax_ccdf.set_yscale('log')
        ax_ccdf.set_xlabel("Dwell Time $t$ [s]", fontsize=11)
        ax_ccdf.set_ylabel(f"CCDF $P(T \\geq t)$ ({s_lbl})", fontsize=11)
        ax_ccdf.set_title(f"(b) Dwell Time CCDF: {s_lbl}", fontsize=12, fontweight='bold')
        ax_ccdf.grid(True, linestyle='--', alpha=0.5)
        ax_ccdf.legend(loc='best', fontsize=8.5, frameon=True)

    fig.suptitle("Dwell Time Distributions of HMM Inferred Motion States", fontsize=14, fontweight='bold')
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVED] {output_path}", flush=True)


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
        ax.errorbar(
            df_s['diameter_um'],
            df_s['mean_dwell_emp_s'],
            yerr=df_s['dwell_err_s'],
            marker='^',
            lw=2.0,
            capsize=4,
            color=STATE_COLORS.get(s, f"C{s}"),
            label=f"{lbl} (Observed)",
        )
        ax.plot(
            df_s['diameter_um'],
            df_s['theoretical_dwell_time_s'],
            linestyle='--',
            color=STATE_COLORS.get(s, f"C{s}"),
            alpha=0.6,
            label=f"{lbl} (Theoretical $A_{{ii}}$)",
        )
    ax.set_xscale('log')
    ax.set_xlabel(r"Cargo Diameter $d$ [$\mu\mathrm{m}$]", fontsize=11)
    ax.set_ylabel("Mean Dwell Time $\\tau_\\text{dwell}$ [s]", fontsize=11)
    ax.set_title("(c) State Dwell Time vs Particle Size", fontsize=12, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='best', frameon=True)

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


def plot_delta_bic_2to1_vs_diameter(
    df_bic: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    records = []
    for binfo in BEADS_INFO:
        bname = binfo['name']
        dia = binfo['diameter_um']
        df_b = df_bic[df_bic['bead_name'] == bname]
        if df_b.empty:
            continue

        row_k1 = df_b[df_b['k_components'] == 1]
        row_k2 = df_b[df_b['k_components'] == 2]
        if row_k1.empty or row_k2.empty:
            continue

        bic1 = float(row_k1.iloc[0]['bic'])
        bic2 = float(row_k2.iloc[0]['bic'])
        aic1 = float(row_k1.iloc[0]['aic'])
        aic2 = float(row_k2.iloc[0]['aic'])
        n_samples = int(row_k2.iloc[0]['n_samples'])

        delta_bic = bic2 - bic1
        delta_aic = aic2 - aic1

        records.append({
            'bead_name': bname,
            'diameter_um': dia,
            'n_samples': n_samples,
            'bic_k1': bic1,
            'bic_k2': bic2,
            'delta_bic_2to1': delta_bic,
            'delta_bic_per_sample': delta_bic / n_samples,
            'aic_k1': aic1,
            'aic_k2': aic2,
            'delta_aic_2to1': delta_aic,
            'delta_aic_per_sample': delta_aic / n_samples,
            'color': binfo['color'],
            'marker': binfo['marker'],
        })

    if not records:
        return pd.DataFrame()

    df_delta = pd.DataFrame(records).sort_values(by='diameter_um')
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Panel (a): 生の ΔBIC
    ax0 = axes[0]
    y_min_val = min(df_delta['delta_bic_2to1'].min() * 1.18, -4500)
    ax0.axhspan(y_min_val, -10, color='#1b9e77', alpha=0.08, label="Decisive Support for $K=2$ ($\\Delta\\mathrm{BIC} < -10$)")
    ax0.axhline(0, color='gray', linestyle='--', lw=1.2, alpha=0.7, label="Null Model Reference ($K=1$, $\\Delta\\mathrm{BIC}=0$)")
    ax0.axhline(-10, color='crimson', linestyle=':', lw=1.2, alpha=0.8)

    ax0.plot(df_delta['diameter_um'], df_delta['delta_bic_2to1'], color='#2b5c8f', lw=2.2, zorder=2)

    for _, row in df_delta.iterrows():
        ax0.scatter(
            row['diameter_um'],
            row['delta_bic_2to1'],
            s=100,
            color=row['color'],
            marker=row['marker'],
            edgecolors='black',
            linewidths=1.2,
            zorder=3,
            label=f"$d={row['diameter_um']:.2f}\\,\\mu\\mathrm{{m}}$ ($N={int(row['n_samples'])}$)",
        )
        y_offset = 12 if row['diameter_um'] in [0.63, 3.37, 20.0] else -18
        ax0.annotate(
            f"{row['delta_bic_2to1']:.0f}",
            (row['diameter_um'], row['delta_bic_2to1']),
            textcoords="offset points",
            xytext=(0, y_offset),
            ha='center',
            fontsize=9.5,
            fontweight='bold',
            color='#1a2a3a',
        )

    ax0.set_xscale('log')
    ax0.set_xlabel(r"Cargo Particle Diameter $d$ [$\mu\mathrm{m}$]", fontsize=11)
    ax0.set_ylabel(r"$\Delta\mathrm{BIC}_{2\rightarrow 1} = \mathrm{BIC}(K=2) - \mathrm{BIC}(K=1)$", fontsize=11)
    ax0.set_title(r"(a) Model Selection $\Delta\mathrm{BIC}_{2\rightarrow 1}$ vs Diameter (1D Speed-Only)", fontsize=12, fontweight='bold')
    ax0.set_xticks([0.63, 1.18, 3.37, 5.0, 7.24, 20.0])
    ax0.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax0.set_ylim(y_min_val, 400)
    ax0.grid(True, linestyle='--', alpha=0.5)
    ax0.legend(loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=2, fontsize=8.2, frameon=True, framealpha=0.92)

    # Panel (b): サンプル正規化 ΔBIC / N
    ax1 = axes[1]
    ax1.axhline(0, color='gray', linestyle='--', lw=1.2, alpha=0.7)
    y_min_norm = min(df_delta['delta_bic_per_sample'].min(), df_delta['delta_aic_per_sample'].min()) * 1.2
    ax1.axhspan(y_min_norm, 0, color='#1b9e77', alpha=0.08)

    ax1.plot(
        df_delta['diameter_um'],
        df_delta['delta_bic_per_sample'],
        color='#1f77b4',
        lw=2.2,
        marker='o',
        markersize=6,
        label=r"$\Delta\mathrm{BIC}_{2\rightarrow 1} / N$ (Per-step info gain)",
        zorder=3,
    )
    ax1.plot(
        df_delta['diameter_um'],
        df_delta['delta_aic_per_sample'],
        color='#ff7f0e',
        lw=1.8,
        linestyle='--',
        marker='s',
        markersize=5,
        label=r"$\Delta\mathrm{AIC}_{2\rightarrow 1} / N$",
        zorder=2,
    )

    for _, row in df_delta.iterrows():
        y_offset_norm = 10 if row['diameter_um'] in [3.37, 20.0] else -16
        ax1.annotate(
            f"{row['delta_bic_per_sample']:.3f}",
            (row['diameter_um'], row['delta_bic_per_sample']),
            textcoords="offset points",
            xytext=(0, y_offset_norm),
            ha='center',
            fontsize=9.5,
            fontweight='bold',
            color='#1f77b4',
        )

    ax1.set_xscale('log')
    ax1.set_xlabel(r"Cargo Particle Diameter $d$ [$\mu\mathrm{m}$]", fontsize=11)
    ax1.set_ylabel("Normalized Criterion Difference per Step", fontsize=11)
    ax1.set_title(r"(b) Sample-Normalized Difference vs Diameter (1D Speed-Only)", fontsize=12, fontweight='bold')
    ax1.set_xticks([0.63, 1.18, 3.37, 5.0, 7.24, 20.0])
    ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax1.set_ylim(y_min_norm, 0.1)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='lower right', fontsize=9.5, frameon=True, framealpha=0.92)

    fig.suptitle(
        r"Statistical Evidence for 2-State HMM over 1-State Model: $\Delta\mathrm{BIC}_{2\rightarrow 1}$ (1D Speed-Only Model)",
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

    return df_delta


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

    delta_bic_fig_path = output_path.parent / "hmm_delta_bic_2to1_vs_diameter.svg"
    df_delta = plot_delta_bic_2to1_vs_diameter(df_bic, delta_bic_fig_path)
    if not df_delta.empty:
        delta_csv_path = output_path.parent / "hmm_delta_bic_2to1_summary.csv"
        safe_save_csv(df_delta, delta_csv_path)
        print(f"[SAVED] {delta_csv_path}", flush=True)

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
    parser.add_argument("--eval_bic", action="store_true", help="Evaluate BIC/AIC for K=1..4 and generate ΔBIC(2->1) vs diameter plot")
    parser.add_argument("--save_csv", action="store_true", default=True, help="Save summary CSV tables")

    args = parser.parse_args()

    root_dir = Path(args.root_dir) if args.root_dir else find_default_root()
    output_dir = Path(args.output_dir) if args.output_dir else root_dir / "figure" / "hmm_1d"
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

        hmm_model = hc.CargoGaussianHMM(
            n_components=args.n_components,
            covariance_type=args.covariance_type,
            epsilon=args.epsilon,
            random_state=42,
        )
        hmm_model.fit(X, lengths=lengths)

        pred_states = hmm_model.predict(X, lengths=lengths)
        proba = hmm_model.predict_proba(X, lengths=lengths)
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

        empirical_dwells = []
        dwell_errors = []
        r2_fits = []
        for s in range(args.n_components):
            t_list = dwell_times.get(s, [])
            tau_mle, tau_err, r2 = hc.fit_exponential_distribution(t_list)
            empirical_dwells.append(tau_mle)
            dwell_errors.append(tau_err)
            r2_fits.append(r2)

        df_state_sum['mean_dwell_emp_s'] = empirical_dwells
        df_state_sum['dwell_err_s'] = dwell_errors
        df_state_sum['dwell_exp_r2'] = r2_fits
        all_summaries.append(df_state_sum)

        A = hmm_model.model.transmat_
        for i in range(args.n_components):
            for j in range(args.n_components):
                all_trans_records.append({
                    'bead_name': bname,
                    'diameter_um': dia,
                    'from_state': i,
                    'to_state': j,
                    'trans_prob': A[i, j],
                })

        fitted_results[bname] = {
            'X': X,
            'lengths': lengths,
            'df_obs': df_obs,
            'model': hmm_model,
            'pred_states': pred_states,
            'proba': proba,
            'conf_stats': conf_stats,
            'dwell_times': dwell_times,
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
                f"dwell={srow['mean_dwell_emp_s']:.1f}s"
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

        safe_save_csv(df_all_summary, csv1_path)
        safe_save_csv(df_trans, csv2_path)
        safe_save_csv(df_conf_all, csv3_path)
        safe_save_csv(df_msd_curves_all, csv4_path)
        safe_save_csv(df_msd_fits_all, csv5_path)

        print(f"[SAVED] {csv1_path}")
        print(f"[SAVED] {csv2_path}")
        print(f"[SAVED] {csv3_path}")
        print(f"[SAVED] {csv4_path}")
        print(f"[SAVED] {csv5_path}")

    print("\n=================================================================")
    print("      HMM Cargo Motion Mode Analysis Completed Successfully!     ")
    print("=================================================================")


if __name__ == '__main__':
    main()
