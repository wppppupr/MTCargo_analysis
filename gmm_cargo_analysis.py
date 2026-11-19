#!/usr/bin/env python3
"""
gmm_cargo_analysis.py

小貨物（0.63 µm, 1.18 µm）の運動モード（Run / Tumble）を、
時間変化する遠方微小管流速 u_inf(t) で規格化した瞬時速力の対数
    O_t = ln(v(t) / u_inf(t))
に対する 1次元ガウス混合モデル（1D Gaussian Mixture Model: GMM）で解析・可視化するスクリプトです。

データ抽出パイプライン:
  - experiment_cutoff (DEFAULT_EXPERIMENT_CUTOFFS) 適用
  - strict_active_particles (厳格な活性粒子セット) 適用
  - 時間ごとの遠方流速 u_inf(t) (scratch/u_inf_per_frame_cache.pkl) で速度を規格化

出力先:
  - figure/gmm_1d/
  - /mnt/NAS-Ebanaru/sasaki/MTsingleBeads/figure/gmm_1d/
"""

import argparse
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from libs import gmm_cargo as gc
from libs.experiment_cutoff import filter_by_cutoff, load_strict_active_particles

ROOT_CANDIDATES = [
    Path("/mnt/NAS-Ebanaru/sasaki/MTsingleBeads"),
    Path("/Volumes/data-1/Sasaki/MTsingleBeads"),
    Path("/Volumes/data/Sasaki/MTsingleBeads"),
]

# スタイルの適用
style_path = CURRENT_DIR / "libs" / "my_style.mplstyle"
if style_path.exists():
    try:
        plt.style.use(str(style_path))
        style_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    except Exception:
        style_colors = ["#882255", "#CC6677", "#DDCC77", "#999933", "#117733", "#44AA99"]
else:
    style_colors = ["#882255", "#CC6677", "#DDCC77", "#999933", "#117733", "#44AA99"]

SMALL_BEADS_INFO = [
    {"name": "beads06um", "diameter_um": 0.63, "radius_um": 0.315, "marker": "^", "color": style_colors[0]},
    {"name": "beads1um",  "diameter_um": 1.18, "radius_um": 0.590, "marker": "o", "color": style_colors[1]},
]


def find_default_root() -> Path:
    for r in ROOT_CANDIDATES:
        if r.exists():
            return r
    return CURRENT_DIR


def find_experiment_dirs(root_dir: Path, bead_name: str) -> List[Path]:
    base = Path(root_dir) / bead_name
    if not base.exists():
        return []
    exp_dirs = [p for p in sorted(base.glob("*/*")) if p.is_dir() and (p / "beads_tracks.csv").exists()]
    if not exp_dirs:
        exp_dirs = [p for p in sorted(base.glob("*")) if p.is_dir() and (p / "beads_tracks.csv").exists()]
    return exp_dirs


def save_fig_to_dirs(fig: plt.Figure, out_dirs: List[Path], stem: str, dpi: int = 300):
    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / f"{stem}.png", dpi=dpi, bbox_inches="tight")
        fig.savefig(d / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {stem}.png / .svg", flush=True)


def save_csv_to_dirs(df: pd.DataFrame, out_dirs: List[Path], filename: str):
    if df is None or df.empty:
        return
    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)
        df.to_csv(d / filename, index=False)
    print(f"[SAVED] {filename}", flush=True)


def collect_bead_gmm_data(
    exp_dirs: List[Path],
    bead_name: str,
    df_uinf: pd.DataFrame,
    strict_particles: set,
    tau: int = 1,
    scale: float = 0.11,
    frame_interval: float = 4.0,
    epsilon: float = 0.0,
) -> Tuple[np.ndarray, List[int], pd.DataFrame]:
    """
    指定ビーズサイズの実験群から、cutoff・strict_active_particles を適用して軌跡を読み込み、
    u_inf(t) で規格化した対数速力特徴量を抽出する。
    """
    all_dfs = []
    uid_offset = 0

    for edir in exp_dirs:
        tracks_csv = edir / "beads_tracks.csv"
        if not tracks_csv.exists():
            continue
        df_tracks = pd.read_csv(tracks_csv)
        if not {"particle", "frame", "x", "y"}.issubset(df_tracks.columns):
            continue

        exp_name = edir.name
        df_sub = df_tracks[["particle", "frame", "x", "y"]].copy()
        df_sub["exp_dir"] = exp_name
        df_sub["bead_name"] = bead_name
        df_sub = filter_by_cutoff(df_sub, exp_col="exp_dir", frame_col="frame")

        if strict_particles:
            keep_pids = {pid for (e, pid) in strict_particles if e == exp_name}
            df_sub = df_sub[df_sub["particle"].isin(keep_pids)].copy()

        if df_sub.empty:
            continue

        df_sub["orig_particle"] = df_sub["particle"].astype(int)
        uniq_p = sorted(df_sub["orig_particle"].unique())
        pmap = {p: uid_offset + i for i, p in enumerate(uniq_p)}
        df_sub["particle"] = df_sub["orig_particle"].map(pmap)
        uid_offset += len(uniq_p)
        all_dfs.append(df_sub)

    if not all_dfs:
        return np.empty((0, 1)), [], pd.DataFrame()

    df_combined = pd.concat(all_dfs, ignore_index=True)
    return gc.extract_gmm_features(
        df_combined,
        df_uinf=df_uinf,
        tau=tau,
        scale=scale,
        frame_interval=frame_interval,
        epsilon=epsilon,
    )


def plot_emission_density(
    fitted_results: Dict[str, dict],
    beads_info: List[dict],
    out_dirs: List[Path],
    n_components: int = 2,
    epsilon: float = 0.0,
):
    """ln(v / u_inf(t)) の確率密度ヒストグラムと GMM コンポーネントの比較プロット。"""
    n_beads = len(beads_info)
    fig, axes = plt.subplots(1, n_beads, figsize=(6.5 * n_beads, 4.8), sharex=True, sharey=True)
    if n_beads == 1:
        axes = [axes]

    state_names = gc.STATE_NAMES.get(n_components, {0: "Tumble", 1: "Run"})

    for ax, binfo in zip(axes, beads_info):
        bname = binfo["name"]
        dia = binfo["diameter_um"]
        if bname not in fitted_results:
            ax.set_visible(False)
            continue

        res = fitted_results[bname]
        log_v = res["X"][:, 0]
        model: gc.CargoGaussianGMM = res["model"]

        ax.hist(
            log_v,
            bins=38,
            density=True,
            color="#999999",
            alpha=0.35,
            edgecolor="#666666",
            label="Observed data",
            zorder=1,
        )

        x_grid = np.linspace(np.min(log_v) - 0.5, np.max(log_v) + 0.5, 400)
        total_pdf = np.zeros_like(x_grid)

        means = model.means_[:, 0]
        stds = model.stds_
        weights = model.weights_

        for s in range(n_components):
            mu, sd, w = float(means[s]), float(stds[s]), float(weights[s])
            comp_pdf = w * stats.norm.pdf(x_grid, loc=mu, scale=sd)
            total_pdf += comp_pdf
            v_geom = max(0.0, float(np.exp(mu) - epsilon))
            col = gc.STATE_COLORS.get(s, f"C{s}")
            lbl = state_names.get(s, f"State {s}")

            ax.plot(
                x_grid,
                comp_pdf,
                color=col,
                lw=2.2,
                label=rf"{lbl}: $\tilde{{v}}_{{\mathrm{{geom}}}}={v_geom:.2f}$ ({w*100:.1f}%)",
                zorder=3,
            )
            ax.fill_between(x_grid, 0, comp_pdf, color=col, alpha=0.18, zorder=2)
            ax.axvline(mu, color=col, ls=":", lw=1.4, alpha=0.85, zorder=3)

        ax.plot(x_grid, total_pdf, color="#111111", lw=1.9, ls="--", label="GMM total", zorder=4)

        boundary = model.get_decision_boundary()
        if boundary is not None and np.isfinite(boundary):
            v_star = max(0.0, float(np.exp(boundary) - epsilon))
            ax.axvline(
                boundary,
                color="#333333",
                ls="-.",
                lw=1.3,
                label=rf"Threshold $\tilde{{v}}^*={v_star:.2f}$",
                zorder=4,
            )

        sv = model.get_separation_index()
        dbic = res.get("delta_bic_1to2", np.nan)
        info_txt = rf"$S_v = {sv:.2f}$" + "\n" + rf"$\Delta\mathrm{{BIC}}_{{1\to 2}} = {dbic:.1f}$"
        ax.text(
            0.03,
            0.96,
            info_txt,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.88, edgecolor="#cccccc"),
        )

        ax.set_title(rf"$d = {dia:.2f}\,\mu\mathrm{{m}}$ ($N={len(log_v):,}$, {res['n_particles']} cargos)", fontsize=12, fontweight="bold")
        ax.set_xlabel(r"$\ln(v(t)\,/\,u_\infty(t))$", fontsize=11.5)
        ax.grid(True, ls="--", alpha=0.35)
        ax.legend(loc="upper right", fontsize=8.2, framealpha=0.92)

    axes[0].set_ylabel("Probability Density", fontsize=11.5)
    fig.suptitle(r"1D GMM of Log Normalized Cargo Speed $\ln(v(t)/u_\infty(t))$", fontsize=13.5, fontweight="bold")
    fig.tight_layout()
    save_fig_to_dirs(fig, out_dirs, f"gmm_emission_density_k{n_components}")


def evaluate_model_selection(
    fitted_results: Dict[str, dict],
    beads_info: List[dict],
    out_dirs: List[Path],
    k_range: Tuple[int, ...] = (1, 2, 3, 4),
) -> pd.DataFrame:
    """K=1..4 の BIC, AIC, ICL を評価してプロットおよび CSV 保存を行う。"""
    records = []
    for binfo in beads_info:
        bname = binfo["name"]
        dia = binfo["diameter_um"]
        if bname not in fitted_results:
            continue
        X = fitted_results[bname]["X"]
        for k in k_range:
            gm = gc.CargoGaussianGMM(n_components=k, random_state=42).fit(X)
            bic, aic, icl = gm.compute_bic_aic_icl(X)
            records.append({
                "bead_name": bname,
                "diameter_um": dia,
                "K": k,
                "n_samples": len(X),
                "log_likelihood": gm.score(X),
                "bic": bic,
                "aic": aic,
                "icl": icl,
            })

    df_bic = pd.DataFrame(records)
    if df_bic.empty:
        return df_bic

    # Delta from K=1
    for bname, grp in df_bic.groupby("bead_name"):
        b1 = float(grp.loc[grp["K"] == 1, "bic"].iloc[0])
        a1 = float(grp.loc[grp["K"] == 1, "aic"].iloc[0])
        i1 = float(grp.loc[grp["K"] == 1, "icl"].iloc[0])
        mask = df_bic["bead_name"] == bname
        df_bic.loc[mask, "delta_bic_from_k1"] = b1 - df_bic.loc[mask, "bic"]
        df_bic.loc[mask, "delta_aic_from_k1"] = a1 - df_bic.loc[mask, "aic"]
        df_bic.loc[mask, "delta_icl_from_k1"] = i1 - df_bic.loc[mask, "icl"]

    n_beads = len(beads_info)
    fig, axes = plt.subplots(1, n_beads, figsize=(5.8 * n_beads, 4.4))
    if n_beads == 1:
        axes = [axes]

    for ax, binfo in zip(axes, beads_info):
        bname = binfo["name"]
        sub = df_bic[df_bic["bead_name"] == bname].sort_values("K")
        if sub.empty:
            ax.set_visible(False)
            continue
        ks = sub["K"].to_numpy()
        ax.plot(ks, sub["bic"] - sub["bic"].min(), "o-", color="#1b9e77", lw=2.2, ms=7, label="BIC - min(BIC)")
        ax.plot(ks, sub["aic"] - sub["aic"].min(), "s--", color="#d95f02", lw=1.8, ms=6, label="AIC - min(AIC)")
        ax.plot(ks, sub["icl"] - sub["icl"].min(), "^:", color="#7570b3", lw=1.8, ms=6, label="ICL - min(ICL)")
        ax.axvline(2, color="#444444", ls="--", lw=1.1, alpha=0.6)
        ax.set_xticks(ks)
        ax.set_xlabel("Number of GMM Components $K$", fontsize=11)
        ax.set_ylabel("Information Criterion Offset (lower is better)", fontsize=10.5)
        ax.set_title(rf"$d = {binfo['diameter_um']:.2f}\,\mu\mathrm{{m}}$", fontsize=12, fontweight="bold")
        ax.grid(True, ls="--", alpha=0.35)
        ax.legend(fontsize=8.5, framealpha=0.9)

    fig.suptitle("GMM Model Selection (BIC / AIC / ICL)", fontsize=13.5, fontweight="bold")
    fig.tight_layout()
    save_fig_to_dirs(fig, out_dirs, "gmm_model_selection_bic")
    save_csv_to_dirs(df_bic, out_dirs, "gmm_model_selection_bic.csv")
    return df_bic


def plot_trajectory_segmentation(
    fitted_results: Dict[str, dict],
    beads_info: List[dict],
    out_dirs: List[Path],
    n_components: int = 2,
    n_examples: int = 3,
):
    """代表粒子の軌跡を Run (緑) / Tumble (橙) に色分けして描画する。"""
    n_beads = len(beads_info)
    fig, axes = plt.subplots(n_beads, n_examples, figsize=(4.5 * n_examples, 4.2 * n_beads))
    if n_beads == 1:
        axes = np.expand_dims(axes, axis=0)

    state_names = gc.STATE_NAMES.get(n_components, {0: "Tumble", 1: "Run"})

    for row_idx, binfo in enumerate(beads_info):
        bname = binfo["name"]
        dia = binfo["diameter_um"]
        if bname not in fitted_results:
            for c_idx in range(n_examples):
                axes[row_idx, c_idx].set_visible(False)
            continue

        df_obs = fitted_results[bname]["df_obs"]
        # 遷移があり軌跡が十分長い粒子を優先選択
        p_stats = []
        for pid, grp in df_obs.groupby("particle"):
            st = grp["pred_state"].to_numpy()
            n_trans = int(np.sum(st[1:] != st[:-1])) if len(st) > 1 else 0
            p_stats.append((pid, len(grp), n_trans))
        p_stats.sort(key=lambda x: (x[2] > 0, x[1]), reverse=True)
        chosen = [x[0] for x in p_stats[:n_examples]]

        for col_idx in range(n_examples):
            ax = axes[row_idx, col_idx]
            if col_idx >= len(chosen):
                ax.set_visible(False)
                continue
            pid = chosen[col_idx]
            grp = df_obs[df_obs["particle"] == pid].sort_values("frame")
            x = grp["x_um"].to_numpy()
            y = grp["y_um"].to_numpy()
            dx = grp["dx_um"].to_numpy()
            dy = grp["dy_um"].to_numpy()
            st = grp["pred_state"].to_numpy()
            frames = grp["frame"].to_numpy()

            # 0始まりに相対化
            x0, y0 = x[0], y[0]
            xr, yr = x - x0, y - y0

            for i in range(len(xr)):
                col = gc.STATE_COLORS.get(int(st[i]), "#333333")
                ax.plot([xr[i], xr[i] + dx[i]], [yr[i], yr[i] + dy[i]], color=col, lw=1.8, alpha=0.9)

            for s in range(n_components):
                m = st == s
                if np.any(m):
                    ax.scatter(
                        xr[m],
                        yr[m],
                        s=12,
                        color=gc.STATE_COLORS.get(s, "#333333"),
                        label=state_names.get(s, f"State {s}"),
                        zorder=4,
                    )

            ax.scatter([0], [0], color="black", s=35, marker="o", zorder=5)
            f_run = float(np.mean(st == 1)) if n_components == 2 else np.nan
            exp_name = grp["exp_dir"].iloc[0]
            orig_pid = int(grp["orig_particle"].iloc[0]) if "orig_particle" in grp.columns else int(pid)
            ax.set_title(
                rf"$d={dia:.2f}\,\mu\mathrm{{m}}$ ({exp_name} #{orig_pid})" + "\n" + rf"$N={len(frames)}$, $f_{{\mathrm{{Run}}}}={f_run:.2f}$",
                fontsize=9.8,
                fontweight="bold",
            )
            ax.set_xlabel(r"$\Delta x$ [$\mu\mathrm{m}$]", fontsize=9.5)
            if col_idx == 0:
                ax.set_ylabel(r"$\Delta y$ [$\mu\mathrm{m}$]", fontsize=9.5)
            ax.set_aspect("equal", adjustable="datalim")
            ax.grid(True, ls="--", alpha=0.3)
            if row_idx == 0 and col_idx == 0:
                ax.legend(loc="best", fontsize=8.5, framealpha=0.9)

    fig.suptitle("GMM Run / Tumble Trajectory Segmentation", fontsize=13.5, fontweight="bold")
    fig.tight_layout()
    save_fig_to_dirs(fig, out_dirs, f"gmm_trajectories_k{n_components}")


def plot_timeseries_sync(
    fitted_results: Dict[str, dict],
    beads_info: List[dict],
    out_dirs: List[Path],
    n_components: int = 2,
    frame_interval: float = 4.0,
):
    """代表粒子の v(t), u_inf(t), v(t)/u_inf(t), および事後確率 P(Run|x_t) の時系列同期プロット。"""
    n_beads = len(beads_info)
    fig, axes = plt.subplots(3, n_beads, figsize=(6.8 * n_beads, 8.2), sharex="col")
    if n_beads == 1:
        axes = np.expand_dims(axes, axis=1)

    for col_idx, binfo in enumerate(beads_info):
        bname = binfo["name"]
        dia = binfo["diameter_um"]
        if bname not in fitted_results:
            for r in range(3):
                axes[r, col_idx].set_visible(False)
            continue

        res = fitted_results[bname]
        df_obs = res["df_obs"]
        model: gc.CargoGaussianGMM = res["model"]

        # 遷移が多く長い代表粒子を選択
        best_pid = None
        best_score = -1
        for pid, grp in df_obs.groupby("particle"):
            st = grp["pred_state"].to_numpy()
            n_tr = int(np.sum(st[1:] != st[:-1])) if len(st) > 1 else 0
            f_run = float(np.mean(st == 1))
            bal = 1.0 - abs(f_run - 0.5)
            sc = len(grp) * (1 + n_tr) * (0.2 + bal)
            if sc > best_score:
                best_score = sc
                best_pid = pid

        grp = df_obs[df_obs["particle"] == best_pid].sort_values("frame")
        t_s = (grp["frame"].to_numpy() - grp["frame"].iloc[0]) * frame_interval
        v = grp["v"].to_numpy()
        u_inf = grp["u_inf_t"].to_numpy()
        v_norm = grp["v_norm"].to_numpy()
        st = grp["pred_state"].to_numpy()
        p_run = grp["proba_run"].to_numpy()

        exp_name = grp["exp_dir"].iloc[0]
        orig_pid = int(grp["orig_particle"].iloc[0]) if "orig_particle" in grp.columns else int(best_pid)

        ax0, ax1, ax2 = axes[0, col_idx], axes[1, col_idx], axes[2, col_idx]

        # Row 0: v(t) & u_inf(t)
        ax0.plot(t_s, v, color="#222222", lw=1.2, label=r"Cargo speed $v(t)$")
        ax0.plot(t_s, u_inf, color="#3366aa", lw=1.8, ls="--", label=r"Far-field flow $u_\infty(t)$")
        ax0.set_ylabel(r"Speed [$\mu\mathrm{m/s}$]", fontsize=10.5)
        ax0.set_title(rf"$d={dia:.2f}\,\mu\mathrm{{m}}$ ({exp_name}, particle #{orig_pid})", fontsize=11.5, fontweight="bold")
        ax0.grid(True, ls="--", alpha=0.35)
        ax0.legend(loc="upper right", fontsize=8.5, framealpha=0.9)

        # Row 1: v(t) / u_inf(t)
        ax1.plot(t_s, v_norm, color="#111111", lw=1.1, alpha=0.85)
        for s_id, lbl in [(0, "Tumble"), (1, "Run")]:
            v_g = float(np.exp(model.means_[s_id, 0]))
            ax1.axhline(v_g, color=gc.STATE_COLORS[s_id], ls=":", lw=1.5, label=rf"{lbl} geom. mean ({v_g:.2f})")
        boundary = model.get_decision_boundary()
        if boundary is not None:
            v_star = float(np.exp(boundary))
            ax1.axhline(v_star, color="#444444", ls="-.", lw=1.3, label=rf"Threshold ({v_star:.2f})")
        ax1.set_yscale("log")
        ax1.set_ylabel(r"$v(t)\,/\,u_\infty(t)$", fontsize=10.5)
        ax1.grid(True, which="both", ls="--", alpha=0.3)
        ax1.legend(loc="upper right", fontsize=8.0, framealpha=0.9)

        # Row 2: Posterior & state shading
        ax2.plot(t_s, p_run, color="#1b9e77", lw=1.1, label=r"$P(\mathrm{Run}\mid x_t)$")
        ax2.axhline(0.5, color="#555555", ls="--", lw=1.0)
        for i in range(len(t_s)):
            dt_half = 0.5 * frame_interval
            col = gc.STATE_COLORS.get(int(st[i]), "#999999")
            ax2.axvspan(t_s[i] - dt_half, t_s[i] + dt_half, color=col, alpha=0.16, lw=0)
        ax2.set_ylim(-0.05, 1.05)
        ax2.set_xlabel("Time $t$ [s]", fontsize=10.5)
        ax2.set_ylabel(r"$P(\mathrm{Run}\mid x_t)$", fontsize=10.5)
        ax2.grid(True, ls="--", alpha=0.35)
        ax2.legend(loc="upper right", fontsize=8.5, framealpha=0.9)

    fig.suptitle(r"Time-Series Synchronization with $u_\infty(t)$ Normalization and GMM Classification", fontsize=13.5, fontweight="bold")
    fig.tight_layout()
    save_fig_to_dirs(fig, out_dirs, f"gmm_timeseries_sync_k{n_components}")


def plot_posterior_distributions(
    fitted_results: Dict[str, dict],
    beads_info: List[dict],
    out_dirs: List[Path],
    n_components: int = 2,
):
    """事後確率 P(Run | x_t) と最大確信度の分布プロット。"""
    n_beads = len(beads_info)
    fig, axes = plt.subplots(1, n_beads, figsize=(5.8 * n_beads, 4.2), sharey=True)
    if n_beads == 1:
        axes = [axes]

    for ax, binfo in zip(axes, beads_info):
        bname = binfo["name"]
        if bname not in fitted_results:
            ax.set_visible(False)
            continue
        res = fitted_results[bname]
        p_run = res["proba"][:, 1] if n_components >= 2 else res["proba"][:, 0]
        conf = res["conf_stats"]
        ax.hist(p_run, bins=25, range=(0, 1), density=True, color="#1b9e77", alpha=0.65, edgecolor="white")
        ax.axvline(0.5, color="#333333", ls="--", lw=1.3)
        ax.set_title(
            rf"$d={binfo['diameter_um']:.2f}\,\mu\mathrm{{m}}$" + "\n" +
            rf"Mean conf={conf.get('mean_confidence', 0):.3f}, $\geq 0.8$: {conf.get('high_conf_ratio_80', 0)*100:.1f}%",
            fontsize=11,
            fontweight="bold",
        )
        ax.set_xlabel(r"Posterior Probability $P(\mathrm{Run} \mid x_t)$", fontsize=10.5)
        ax.grid(True, ls="--", alpha=0.35)

    axes[0].set_ylabel("Density", fontsize=10.5)
    fig.suptitle("GMM Posterior Probability Distributions", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_fig_to_dirs(fig, out_dirs, f"gmm_posterior_distributions_k{n_components}")


def plot_dwell_time_distributions(
    fitted_results: Dict[str, dict],
    beads_info: List[dict],
    out_dirs: List[Path],
    n_components: int = 2,
):
    """Run / Tumble 持続時間分布（PDF & CCDF + 指数フィット）のプロット。"""
    n_beads = len(beads_info)
    fig, axes = plt.subplots(2, n_beads, figsize=(6.0 * n_beads, 8.0))
    if n_beads == 1:
        axes = np.expand_dims(axes, axis=1)

    state_names = gc.STATE_NAMES.get(n_components, {0: "Tumble", 1: "Run"})

    for col_idx, binfo in enumerate(beads_info):
        bname = binfo["name"]
        dia = binfo["diameter_um"]
        if bname not in fitted_results:
            axes[0, col_idx].set_visible(False)
            axes[1, col_idx].set_visible(False)
            continue

        res = fitted_results[bname]
        dwell_times = res["dwell_times"]
        pdf_fits = res["dwell_pdf_fits"]
        ccdf_fits = res["dwell_ccdf_fits"]

        ax_pdf = axes[0, col_idx]
        ax_ccdf = axes[1, col_idx]

        for s in range(n_components):
            col = gc.STATE_COLORS.get(s, f"C{s}")
            lbl = state_names.get(s, f"State {s}")
            t_arr = np.asarray(dwell_times.get(s, []), dtype=float)
            if len(t_arr) == 0:
                continue

            pf = pdf_fits.get(s, {})
            if len(pf.get("t_centers", [])) > 0:
                ax_pdf.plot(pf["t_centers"], pf["p_vals"], "o", color=col, ms=5, alpha=0.8)
            if len(pf.get("t_fit", [])) > 0:
                ax_pdf.plot(
                    pf["t_fit"],
                    pf["p_fit"],
                    "-",
                    color=col,
                    lw=1.8,
                    label=rf"{lbl}: $\tau_{{\mathrm{{PDF}}}}={pf['tau_fit_s']:.1f}\,\mathrm{{s}}$ (mean {np.mean(t_arr):.1f} s)",
                )

            cf = ccdf_fits.get(s, {})
            if len(cf.get("t_unique", [])) > 0:
                ax_ccdf.step(cf["t_unique"], cf["ccdf_unique"], where="post", color=col, lw=1.6, alpha=0.85)
            if len(cf.get("t_fit", [])) > 0:
                t_fit_ccdf = cf["t_fit"]
                y_fit_ccdf = cf["ccdf_fit"]
                pos = t_fit_ccdf > 0
                ax_ccdf.plot(
                    t_fit_ccdf[pos],
                    y_fit_ccdf[pos],
                    "--",
                    color=col,
                    lw=1.8,
                    label=rf"{lbl}: $\tau_{{\mathrm{{CCDF}}}}={cf['tau_fit_s']:.1f}\,\mathrm{{s}}$ ($n={len(t_arr)}$)",
                )

        ax_pdf.set_yscale("log")
        ax_pdf.set_xlabel("Dwell Time $\\tau$ [s]", fontsize=10.5)
        ax_pdf.set_ylabel("PDF $P(\\tau)$ [1/s]", fontsize=10.5)
        ax_pdf.set_title(rf"$d={dia:.2f}\,\mu\mathrm{{m}}$: Dwell Time PDF", fontsize=11.5, fontweight="bold")
        ax_pdf.grid(True, which="both", ls="--", alpha=0.3)
        ax_pdf.legend(fontsize=8.5, framealpha=0.9)

        ax_ccdf.set_yscale("log")
        ax_ccdf.set_xscale("log")
        ax_ccdf.set_xlabel("Dwell Time $\\tau$ [s]", fontsize=10.5)
        ax_ccdf.set_ylabel(r"CCDF $P(T \geq \tau)$", fontsize=10.5)
        ax_ccdf.set_title(rf"$d={dia:.2f}\,\mu\mathrm{{m}}$: Dwell Time CCDF", fontsize=11.5, fontweight="bold")
        ax_ccdf.grid(True, which="both", ls="--", alpha=0.3)
        ax_ccdf.legend(fontsize=8.5, framealpha=0.9)

    fig.suptitle("Run / Tumble Dwell Time Distributions", fontsize=13.5, fontweight="bold")
    fig.tight_layout()
    save_fig_to_dirs(fig, out_dirs, f"gmm_dwell_time_distributions_k{n_components}")


def plot_state_msd(
    fitted_results: Dict[str, dict],
    beads_info: List[dict],
    out_dirs: List[Path],
    n_components: int = 2,
):
    """状態別 MSD 曲線（Run, Tumble, All）とべき乗則・ABP/ブラウン運動フィットの描画。"""
    n_beads = len(beads_info)
    fig, axes = plt.subplots(1, n_beads, figsize=(6.2 * n_beads, 5.0), sharex=True, sharey=True)
    if n_beads == 1:
        axes = [axes]

    state_names = gc.STATE_NAMES.get(n_components, {0: "Tumble", 1: "Run"})

    for ax, binfo in zip(axes, beads_info):
        bname = binfo["name"]
        dia = binfo["diameter_um"]
        if bname not in fitted_results:
            ax.set_visible(False)
            continue

        res = fitted_results[bname]
        df_msd = res["df_msd"]
        df_fits = res["df_msd_fits"]
        abp = res.get("abp_fit", {})
        t_diff = res.get("tumble_diff_fit", {})

        for s, col, mk, lbl in [
            (1, gc.STATE_COLORS[1], "o", state_names.get(1, "Run")),
            (0, gc.STATE_COLORS[0], "s", state_names.get(0, "Tumble")),
            (-1, "#222222", "^", "All"),
        ]:
            sub = df_msd[(df_msd["state"] == s) & (df_msd["msd_um2"] > 0)].sort_values("lag_time_s")
            if sub.empty:
                continue
            fit_row = df_fits[df_fits["state"] == s]
            alpha = float(fit_row["alpha"].iloc[0]) if not fit_row.empty else np.nan
            ax.errorbar(
                sub["lag_time_s"],
                sub["msd_um2"],
                yerr=sub["msd_sem_um2"],
                fmt=mk,
                color=col,
                ms=5,
                capsize=2.5,
                lw=1.4,
                ls="-",
                label=rf"{lbl} ($\alpha={alpha:.2f}$)",
                zorder=4 if s != -1 else 3,
            )

        if len(abp.get("fit_t", [])) > 0:
            tau_r = abp.get("tau_r_s", np.nan)
            abp_lbl = "Run ABP (ballistic limit)" if (np.isfinite(tau_r) and tau_r >= 9000) else rf"Run ABP ($\tau_r={tau_r:.1f}\,\mathrm{{s}}$)"
            ax.plot(
                abp["fit_t"],
                abp["fit_msd"],
                color=gc.STATE_COLORS[1],
                ls="--",
                lw=1.6,
                alpha=0.85,
                label=abp_lbl,
            )
        if len(t_diff.get("fit_t", [])) > 0:
            ax.plot(
                t_diff["fit_t"],
                t_diff["fit_msd"],
                color=gc.STATE_COLORS[0],
                ls=":",
                lw=1.6,
                alpha=0.85,
                label=rf"Tumble Brownian ($D_t={t_diff['Dt_um2_s']:.4f}\,\mu\mathrm{{m}}^2/\mathrm{{s}}$)",
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"Lag Time $\Delta t$ [s]", fontsize=11)
        ax.set_title(rf"$d={dia:.2f}\,\mu\mathrm{{m}}$", fontsize=12, fontweight="bold")
        ax.grid(True, which="both", ls="--", alpha=0.35)
        ax.legend(loc="upper left", fontsize=8.3, framealpha=0.92)

    axes[0].set_ylabel(r"MSD $\langle \Delta r^2(\Delta t) \rangle$ [$\mu\mathrm{m}^2$]", fontsize=11)
    fig.suptitle("State-Dependent MSD (GMM Run vs Tumble)", fontsize=13.5, fontweight="bold")
    fig.tight_layout()
    save_fig_to_dirs(fig, out_dirs, f"gmm_state_msd_k{n_components}")


def plot_turning_angles(
    fitted_results: Dict[str, dict],
    beads_info: List[dict],
    out_dirs: List[Path],
    n_components: int = 2,
) -> pd.DataFrame:
    """状態別方向転換角 Δθ の直交座標・極座標プロットとサマリー CSV を作成する。"""
    n_beads = len(beads_info)
    state_names = gc.STATE_NAMES.get(n_components, {0: "Tumble", 1: "Run"})

    fig_cart, axes_cart = plt.subplots(1, n_beads, figsize=(6.0 * n_beads, 4.4), sharex=True, sharey=True)
    fig_pol, axes_pol = plt.subplots(1, n_beads, figsize=(5.8 * n_beads, 4.8), subplot_kw={"projection": "polar"})
    if n_beads == 1:
        axes_cart = [axes_cart]
        axes_pol = [axes_pol]

    summary_rows = []
    bins = np.linspace(-np.pi, np.pi, 37)
    centers = 0.5 * (bins[:-1] + bins[1:])

    for idx, binfo in enumerate(beads_info):
        bname = binfo["name"]
        dia = binfo["diameter_um"]
        if bname not in fitted_results:
            axes_cart[idx].set_visible(False)
            axes_pol[idx].set_visible(False)
            continue

        ta = fitted_results[bname]["turning_angles"]
        ax_c = axes_cart[idx]
        ax_p = axes_pol[idx]

        for s, col, lbl in [
            (1, gc.STATE_COLORS[1], state_names.get(1, "Run")),
            (0, gc.STATE_COLORS[0], state_names.get(0, "Tumble")),
            (-1, "#333333", "All"),
        ]:
            ang = ta.get(s, np.array([]))
            if len(ang) < 5:
                continue
            c_stats = gc.calc_circular_stats_dict(ang)
            vm = gc.fit_von_mises_distribution(ang, bins=36)
            hist_vals, _ = np.histogram(ang, bins=bins, density=True)

            ls = "--" if s == -1 else "-"
            lw = 1.5 if s == -1 else 2.1
            ax_c.plot(
                np.rad2deg(centers),
                hist_vals,
                ls=ls,
                lw=lw,
                color=col,
                label=rf"{lbl} ($\langle\cos\Delta\theta\rangle={c_stats['mean_cos']:.2f}, \kappa={vm['kappa']:.2f}$)",
            )
            c_closed = np.append(centers, centers[0])
            h_closed = np.append(hist_vals, hist_vals[0])
            ax_p.plot(c_closed, h_closed, ls=ls, lw=lw, color=col, label=lbl)

            summary_rows.append({
                "bead_name": bname,
                "diameter_um": dia,
                "state": s,
                "state_label": lbl,
                "count": c_stats["count"],
                "mean_cos_dtheta": c_stats["mean_cos"],
                "mean_resultant_R": c_stats["mean_resultant_length_R"],
                "mean_abs_deg": c_stats["mean_abs_angle_deg"],
                "vm_kappa": vm["kappa"],
                "vm_kappa_err": vm["kappa_err"],
                "vm_r2": vm["r_squared"],
            })

        ax_c.set_xlim(-180, 180)
        ax_c.set_xlabel(r"Turning Angle $\Delta\theta$ [deg]", fontsize=10.5)
        ax_c.set_title(rf"$d={dia:.2f}\,\mu\mathrm{{m}}$", fontsize=12, fontweight="bold")
        ax_c.grid(True, ls="--", alpha=0.35)
        ax_c.legend(loc="upper right", fontsize=8.0, framealpha=0.9)

        ax_p.set_title(rf"$d={dia:.2f}\,\mu\mathrm{{m}}$", fontsize=11.5, fontweight="bold", pad=12)
        ax_p.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1), fontsize=8.2)

    axes_cart[0].set_ylabel(r"PDF $P(\Delta\theta)$ [rad$^{-1}$]", fontsize=10.5)
    fig_cart.suptitle("State-Dependent Turning Angle Distributions", fontsize=13.5, fontweight="bold")
    fig_cart.tight_layout()
    save_fig_to_dirs(fig_cart, out_dirs, f"gmm_turning_angle_distributions_k{n_components}")

    fig_pol.suptitle("State-Dependent Turning Angle Polar Distributions", fontsize=13.5, fontweight="bold")
    fig_pol.tight_layout()
    save_fig_to_dirs(fig_pol, out_dirs, f"gmm_turning_angle_polar_k{n_components}")

    df_angle_sum = pd.DataFrame(summary_rows)
    save_csv_to_dirs(df_angle_sum, out_dirs, f"gmm_turning_angle_summary_k{n_components}.csv")
    return df_angle_sum


def plot_state_msad(
    fitted_results: Dict[str, dict],
    beads_info: List[dict],
    out_dirs: List[Path],
    n_components: int = 2,
):
    """状態別 MSAD 曲線の可視化。"""
    n_beads = len(beads_info)
    fig, axes = plt.subplots(1, n_beads, figsize=(6.0 * n_beads, 4.4), sharex=True, sharey=True)
    if n_beads == 1:
        axes = [axes]

    state_names = gc.STATE_NAMES.get(n_components, {0: "Tumble", 1: "Run"})

    for ax, binfo in zip(axes, beads_info):
        bname = binfo["name"]
        dia = binfo["diameter_um"]
        if bname not in fitted_results:
            ax.set_visible(False)
            continue
        df_msad = fitted_results[bname]["df_msad"]
        df_fits = fitted_results[bname]["df_msad_fits"]
        if df_msad.empty:
            ax.set_visible(False)
            continue

        for s, col, mk, lbl in [
            (1, gc.STATE_COLORS[1], "o", state_names.get(1, "Run")),
            (0, gc.STATE_COLORS[0], "s", state_names.get(0, "Tumble")),
            (-1, "#222222", "^", "All"),
        ]:
            sub = df_msad[(df_msad["state"] == s) & df_msad["msad_rad2"].notna()].sort_values("lag_time_s")
            if sub.empty:
                continue
            fr = df_fits[df_fits["state"] == s] if not df_fits.empty else pd.DataFrame()
            dr_val = float(fr["Dr_rad2_s"].iloc[0]) if (not fr.empty and "Dr_rad2_s" in fr.columns) else np.nan
            ax.errorbar(
                sub["lag_time_s"],
                sub["msad_rad2"],
                yerr=sub["msad_sem_rad2"],
                fmt=mk,
                ms=4.5,
                capsize=2,
                lw=1.3,
                ls="-",
                color=col,
                label=rf"{lbl} ($D_r={dr_val:.3f}\,\mathrm{{rad}}^2/\mathrm{{s}}$)",
            )

        ax.set_xlabel(r"Lag Time $\Delta t$ [s]", fontsize=10.5)
        ax.set_title(rf"$d={dia:.2f}\,\mu\mathrm{{m}}$", fontsize=12, fontweight="bold")
        ax.grid(True, ls="--", alpha=0.35)
        ax.legend(loc="upper left", fontsize=8.3, framealpha=0.9)

    axes[0].set_ylabel(r"MSAD $\langle (\Delta\theta)^2 \rangle$ [rad$^2$]", fontsize=10.5)
    fig.suptitle("State-Dependent Mean Squared Angular Displacement (MSAD)", fontsize=13.5, fontweight="bold")
    fig.tight_layout()
    save_fig_to_dirs(fig, out_dirs, f"gmm_msad_k{n_components}")


def plot_autocorrelation_grid(
    fitted_results: Dict[str, dict],
    beads_info: List[dict],
    out_dirs: List[Path],
    n_components: int = 2,
):
    """状態別 VACF, OACF, SACF のグリッドプロット。"""
    n_beads = len(beads_info)
    ctypes = [("vacf", "VACF"), ("oacf", "OACF"), ("sacf", "SACF (v/u_inf)")]
    fig, axes = plt.subplots(n_beads, 3, figsize=(15.0, 4.0 * n_beads), sharex=True)
    if n_beads == 1:
        axes = np.expand_dims(axes, axis=0)

    state_names = gc.STATE_NAMES.get(n_components, {0: "Tumble", 1: "Run"})

    for r_idx, binfo in enumerate(beads_info):
        bname = binfo["name"]
        dia = binfo["diameter_um"]
        if bname not in fitted_results:
            for c_idx in range(3):
                axes[r_idx, c_idx].set_visible(False)
            continue

        ac_data = fitted_results[bname]["autocorrelations"]
        ac_fits = fitted_results[bname]["autocorr_fits"]

        for c_idx, (ckey, ctitle) in enumerate(ctypes):
            ax = axes[r_idx, c_idx]
            for s in range(n_components):
                if s not in ac_data or ckey not in ac_data[s]:
                    continue
                df_c = ac_data[s][ckey]
                if df_c.empty:
                    continue
                col = gc.STATE_COLORS.get(s, f"C{s}")
                lbl = state_names.get(s, f"State {s}")
                tau_c = ac_fits.get(s, {}).get(ckey, {}).get("tau_corr_s", np.nan)
                tau_str = rf"$\tau={tau_c:.1f}\,\mathrm{{s}}$" if np.isfinite(tau_c) else ""
                ax.errorbar(
                    df_c["lag_time_s"],
                    df_c["corr"],
                    yerr=df_c["sem"],
                    fmt="o-",
                    ms=4,
                    capsize=2,
                    lw=1.4,
                    color=col,
                    label=f"{lbl} ({tau_str})" if tau_str else lbl,
                )
            ax.axhline(0, color="#666666", ls="--", lw=1.0)
            ax.set_title(rf"$d={dia:.2f}\,\mu\mathrm{{m}}$: {ctitle}", fontsize=11, fontweight="bold")
            if r_idx == n_beads - 1:
                ax.set_xlabel(r"Lag Time $\Delta t$ [s]", fontsize=10.5)
            if c_idx == 0:
                ax.set_ylabel("Autocorrelation $C(\\Delta t)$", fontsize=10.5)
            ax.grid(True, ls="--", alpha=0.35)
            ax.legend(loc="upper right", fontsize=8.3, framealpha=0.9)

    fig.suptitle("State-Dependent Velocity, Orientation & Speed Autocorrelations", fontsize=13.5, fontweight="bold")
    fig.tight_layout()
    save_fig_to_dirs(fig, out_dirs, f"gmm_autocorrelation_grid_k{n_components}")


def plot_summary_comparison(
    df_sum: pd.DataFrame,
    df_msd_fits: pd.DataFrame,
    df_angle_sum: pd.DataFrame,
    df_particle_sum: pd.DataFrame,
    beads_info: List[dict],
    out_dirs: List[Path],
    n_components: int = 2,
):
    """小貨物2サイズ (0.63 µm vs 1.18 µm) の Run / Tumble 比較サマリー（6パネル）。"""
    if df_sum.empty or n_components != 2:
        return

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.0))
    bnames = [b["name"] for b in beads_info if b["name"] in df_sum["bead_name"].unique()]
    x = np.arange(len(bnames))
    x_labels = [rf"${next(b['diameter_um'] for b in beads_info if b['name'] == bn):.2f}\,\mu\mathrm{{m}}$" for bn in bnames]
    w = 0.34

    # (A) Normalized speeds
    ax = axes[0, 0]
    v_tum = [float(df_sum[(df_sum["bead_name"] == b) & (df_sum["state"] == 0)]["v_norm_geom"].iloc[0]) for b in bnames]
    v_run = [float(df_sum[(df_sum["bead_name"] == b) & (df_sum["state"] == 1)]["v_norm_geom"].iloc[0]) for b in bnames]
    v_thr = [float(df_sum[(df_sum["bead_name"] == b) & (df_sum["state"] == 1)]["decision_threshold_v_norm"].iloc[0]) for b in bnames]
    ax.bar(x - w / 2, v_tum, width=w, color=gc.STATE_COLORS[0], label="Tumble")
    ax.bar(x + w / 2, v_run, width=w, color=gc.STATE_COLORS[1], label="Run")
    ax.scatter(x, v_thr, color="#111111", marker="_", s=320, lw=2.2, zorder=5, label=r"Threshold $\tilde{v}^*$")
    ax.axhline(1.0, color="#3366aa", ls="--", lw=1.2, label=r"$v = u_\infty(t)$")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel(r"Geometric Mean $v\,/\,u_\infty(t)$")
    ax.set_title("(A) Normalized State Speeds", fontweight="bold")
    ax.legend(fontsize=8.2, framealpha=0.9)
    ax.grid(True, axis="y", ls="--", alpha=0.35)

    # (B) Run fraction (mixture weight, empirical, per-particle scatter)
    ax = axes[0, 1]
    w_run = [float(df_sum[(df_sum["bead_name"] == b) & (df_sum["state"] == 1)]["mixture_weight"].iloc[0]) for b in bnames]
    f_run = [float(df_sum[(df_sum["bead_name"] == b) & (df_sum["state"] == 1)]["empirical_frac"].iloc[0]) for b in bnames]
    ax.bar(x - w / 2, w_run, width=w, color=gc.STATE_COLORS[1], alpha=0.85, label=r"GMM weight $\pi_{\mathrm{Run}}$")
    ax.bar(x + w / 2, f_run, width=w, color="#4477aa", alpha=0.75, label=r"Pooled empirical $f_{\mathrm{Run}}$")
    rng = np.random.default_rng(42)
    for xi, b in zip(x, bnames):
        if not df_particle_sum.empty:
            psub = df_particle_sum[df_particle_sum["bead_name"] == b]
            jitter = rng.uniform(-0.1, 0.1, size=len(psub))
            ax.scatter(xi + w / 2 + jitter, psub["f_run"], color="#111111", s=22, alpha=0.75, zorder=5, label="Per-cargo $f_{\\mathrm{Run}}$" if xi == 0 else None)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Run Fraction")
    ax.set_title("(B) Run State Occupancy", fontweight="bold")
    ax.legend(fontsize=8.2, framealpha=0.9)
    ax.grid(True, axis="y", ls="--", alpha=0.35)

    # (C) Separation index S_v and delta BIC
    ax = axes[0, 2]
    sv_vals = [float(df_sum[(df_sum["bead_name"] == b) & (df_sum["state"] == 1)]["separation_index_Sv"].iloc[0]) for b in bnames]
    dbic_vals = [float(df_sum[(df_sum["bead_name"] == b) & (df_sum["state"] == 1)]["delta_bic_1to2"].iloc[0]) for b in bnames]
    bars = ax.bar(x, sv_vals, width=0.45, color="#7570b3", alpha=0.85, label=r"Separation $S_v$")
    for bar, db in zip(bars, dbic_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.06, rf"$\Delta\mathrm{{BIC}}={db:.0f}$", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylim(0, max(sv_vals) * 1.25)
    ax.set_ylabel(r"Separation Index $S_v$")
    ax.set_title(r"(C) State Separation & $\Delta\mathrm{BIC}_{1\to 2}$", fontweight="bold")
    ax.grid(True, axis="y", ls="--", alpha=0.35)

    # (D) Mean dwell time
    ax = axes[1, 0]
    dw_tum = [float(df_sum[(df_sum["bead_name"] == b) & (df_sum["state"] == 0)]["mean_dwell_emp_s"].iloc[0]) for b in bnames]
    dw_run = [float(df_sum[(df_sum["bead_name"] == b) & (df_sum["state"] == 1)]["mean_dwell_emp_s"].iloc[0]) for b in bnames]
    ax.bar(x - w / 2, dw_tum, width=w, color=gc.STATE_COLORS[0], label="Tumble")
    ax.bar(x + w / 2, dw_run, width=w, color=gc.STATE_COLORS[1], label="Run")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("Mean Empirical Dwell Time [s]")
    ax.set_title("(D) State Dwell Times", fontweight="bold")
    ax.legend(fontsize=8.5, framealpha=0.9)
    ax.grid(True, axis="y", ls="--", alpha=0.35)

    # (E) MSD exponent alpha
    ax = axes[1, 1]
    if not df_msd_fits.empty:
        a_tum = [float(df_msd_fits[(df_msd_fits["bead_name"] == b) & (df_msd_fits["state"] == 0)]["alpha"].iloc[0]) for b in bnames]
        a_run = [float(df_msd_fits[(df_msd_fits["bead_name"] == b) & (df_msd_fits["state"] == 1)]["alpha"].iloc[0]) for b in bnames]
        a_all = [float(df_msd_fits[(df_msd_fits["bead_name"] == b) & (df_msd_fits["state"] == -1)]["alpha"].iloc[0]) for b in bnames]
        w3 = 0.25
        ax.bar(x - w3, a_tum, width=w3, color=gc.STATE_COLORS[0], label="Tumble")
        ax.bar(x, a_run, width=w3, color=gc.STATE_COLORS[1], label="Run")
        ax.bar(x + w3, a_all, width=w3, color="#444444", label="All")
        ax.axhline(1.0, color="#777777", ls=":", lw=1.2)
        ax.axhline(2.0, color="#777777", ls="--", lw=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylim(0, 2.3)
    ax.set_ylabel(r"MSD Exponent $\alpha$")
    ax.set_title("(E) MSD Anomalous Exponent", fontweight="bold")
    ax.legend(fontsize=8.5, framealpha=0.9)
    ax.grid(True, axis="y", ls="--", alpha=0.35)

    # (F) Turning persistence <cos dtheta>
    ax = axes[1, 2]
    if not df_angle_sum.empty:
        c_tum = [float(df_angle_sum[(df_angle_sum["bead_name"] == b) & (df_angle_sum["state"] == 0)]["mean_cos_dtheta"].iloc[0]) for b in bnames]
        c_run = [float(df_angle_sum[(df_angle_sum["bead_name"] == b) & (df_angle_sum["state"] == 1)]["mean_cos_dtheta"].iloc[0]) for b in bnames]
        c_all = [float(df_angle_sum[(df_angle_sum["bead_name"] == b) & (df_angle_sum["state"] == -1)]["mean_cos_dtheta"].iloc[0]) for b in bnames]
        w3 = 0.25
        ax.bar(x - w3, c_tum, width=w3, color=gc.STATE_COLORS[0], label="Tumble")
        ax.bar(x, c_run, width=w3, color=gc.STATE_COLORS[1], label="Run")
        ax.bar(x + w3, c_all, width=w3, color="#444444", label="All")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylim(0, 0.8)
    ax.set_ylabel(r"Turning Persistence $\langle \cos\Delta\theta \rangle$")
    ax.set_title("(F) Directional Persistence by State", fontweight="bold")
    ax.legend(fontsize=8.5, framealpha=0.9)
    ax.grid(True, axis="y", ls="--", alpha=0.35)

    fig.suptitle(r"Small-Cargo 1D GMM Summary ($0.63\,\mu\mathrm{m}$ vs $1.18\,\mu\mathrm{m}$)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_fig_to_dirs(fig, out_dirs, f"gmm_summary_comparison_k{n_components}")


def main():
    parser = argparse.ArgumentParser(
        description="Small-cargo 1D GMM analysis of ln(v(t) / u_inf(t)) (Run vs Tumble)"
    )
    parser.add_argument("--root_dir", type=str, default=None, help="Root directory containing bead experiment folders")
    parser.add_argument("--uinf_cache", type=str, default=str(CURRENT_DIR / "scratch" / "u_inf_per_frame_cache.pkl"),
                        help="Path to u_inf_per_frame_cache.pkl")
    parser.add_argument("--beads", nargs="+", default=["beads06um", "beads1um"],
                        help="Target small bead sizes (default: beads06um beads1um)")
    parser.add_argument("--n_components", type=int, default=2, help="Number of GMM components K (default: 2)")
    parser.add_argument("--covariance_type", type=str, default="full", choices=["full", "diag", "spherical", "tied"])
    parser.add_argument("--tau", type=int, default=1, help="Step lag in frames (default: 1)")
    parser.add_argument("--scale", type=float, default=0.11, help="Spatial scale in um/pixel (default: 0.11)")
    parser.add_argument("--frame_interval", type=float, default=4.0, help="Frame interval in seconds (default: 4.0)")
    parser.add_argument("--epsilon", type=float, default=0.0, help="Optional offset for ln(v/u_inf + epsilon) (default: 0.0)")
    parser.add_argument("--min_dwell_frames", type=int, default=1,
                        help="Optional glitch filter min frames (default: 1 = pure GMM MAP)")
    args = parser.parse_args()

    root_dir = Path(args.root_dir) if args.root_dir else find_default_root()
    out_dirs = [CURRENT_DIR / "figure" / "gmm_1d"]
    if root_dir != CURRENT_DIR and root_dir.exists():
        out_dirs.append(root_dir / "figure" / "gmm_1d")

    uinf_path = Path(args.uinf_cache)
    if not uinf_path.exists():
        raise FileNotFoundError(f"u_inf(t) cache not found: {uinf_path}")
    df_uinf = pd.read_pickle(uinf_path)
    strict_particles = load_strict_active_particles()

    target_beads = [b for b in SMALL_BEADS_INFO if b["name"] in args.beads]
    if not target_beads:
        raise ValueError(f"No valid target beads in {args.beads}")

    print("=================================================================")
    print("    Small-Cargo 1D Log Normalized Speed GMM Analysis             ")
    print("=================================================================")
    print(f"Root dir:        {root_dir}")
    print(f"u_inf(t) cache:  {uinf_path}")
    print(f"Target beads:    {[b['name'] for b in target_beads]}")
    print(f"Strict active:   {len(strict_particles)} particles loaded")
    print(f"GMM K:           {args.n_components}")
    print(f"Observation:     ln(v(t) / u_inf(t)" + (f" + {args.epsilon})" if args.epsilon > 0 else ")"))
    print(f"Min dwell filt:  {args.min_dwell_frames} frame(s)")
    print("=================================================================\n")

    fitted_results: Dict[str, dict] = {}
    all_summaries = []
    all_trans_records = []
    all_conf_records = []
    all_msd_curves = []
    all_msd_fits = []
    all_msad_curves = []
    all_msad_fits = []
    all_autocorr_records = []
    all_abp_records = []
    all_tumble_diff_records = []
    all_particle_records = []

    for binfo in target_beads:
        bname = binfo["name"]
        dia = binfo["diameter_um"]
        edirs = find_experiment_dirs(root_dir, bname)
        print(f"--- Processing {bname} (d = {dia:.2f} um, {len(edirs)} exp dirs) ---")

        X, lengths, df_obs = collect_bead_gmm_data(
            edirs,
            bead_name=bname,
            df_uinf=df_uinf,
            strict_particles=strict_particles,
            tau=args.tau,
            scale=args.scale,
            frame_interval=args.frame_interval,
            epsilon=args.epsilon,
        )
        if len(X) < 20:
            print(f"[WARNING] Insufficient data for {bname}. Skipping.")
            continue

        n_particles = int(df_obs["particle"].nunique())
        print(f"  Extracted {len(X):,} observations across {n_particles} active cargos ({len(lengths)} contiguous segments).")

        gmm_k1 = gc.CargoGaussianGMM(n_components=1, covariance_type=args.covariance_type, epsilon=args.epsilon, random_state=42).fit(X)
        gmm_model = gc.CargoGaussianGMM(n_components=args.n_components, covariance_type=args.covariance_type, epsilon=args.epsilon, random_state=42).fit(X)

        bic_k1, _, _ = gmm_k1.compute_bic_aic_icl(X)
        bic_k, aic_k, icl_k = gmm_model.compute_bic_aic_icl(X)
        delta_bic_1to2 = float(bic_k1 - bic_k) if args.n_components == 2 else np.nan

        raw_pred = gmm_model.predict(X)
        proba = gmm_model.predict_proba(X)
        if args.min_dwell_frames >= 2:
            pred_states = gc.filter_state_glitches(raw_pred, lengths, min_duration_frames=args.min_dwell_frames)
        else:
            pred_states = raw_pred

        df_obs["pred_state"] = pred_states
        df_obs["state_label"] = [gc.get_state_label(int(s), args.n_components) for s in pred_states]
        if args.n_components >= 2:
            df_obs["proba_tumble"] = proba[:, 0]
            df_obs["proba_run"] = proba[:, args.n_components - 1]
        else:
            df_obs["proba_run"] = proba[:, 0]

        dwell_times = gc.calc_state_dwell_times(pred_states, lengths, frame_interval=args.frame_interval)
        conf_stats = gc.calc_posterior_statistics(proba, n_components=args.n_components)
        all_conf_records.append({"bead_name": bname, "diameter_um": dia, **conf_stats})

        emp_trans = gc.calc_empirical_transitions(pred_states, lengths, n_components=args.n_components, frame_interval=args.frame_interval)
        for i in range(args.n_components):
            for j in range(args.n_components):
                all_trans_records.append({
                    "bead_name": bname,
                    "diameter_um": dia,
                    "from_state": i,
                    "from_label": gc.get_state_label(i, args.n_components),
                    "to_state": j,
                    "to_label": gc.get_state_label(j, args.n_components),
                    "count": int(emp_trans["counts"][i, j]),
                    "empirical_trans_prob": float(emp_trans["trans_prob"][i, j]),
                })

        df_msd, df_msd_fits = gc.calc_state_dependent_msd(
            df_obs, max_tau=25, frame_interval=args.frame_interval, n_components=args.n_components, fit_min_tau=1, fit_max_tau=10
        )
        df_msd = gc.relabel_state_dataframe(df_msd, args.n_components)
        df_msd_fits = gc.relabel_state_dataframe(df_msd_fits, args.n_components)
        df_msd["bead_name"] = bname
        df_msd["diameter_um"] = dia
        df_msd_fits["bead_name"] = bname
        df_msd_fits["diameter_um"] = dia
        all_msd_curves.append(df_msd)
        all_msd_fits.append(df_msd_fits)

        df_msad = gc.calc_state_dependent_msad(df_obs, max_tau=25, frame_interval=args.frame_interval, min_segment_len=3, n_components=args.n_components)
        df_msad_fits = gc.fit_msad_linear(df_msad, fit_min_tau_s=args.frame_interval, fit_max_tau_s=30.0, min_points=3, n_components=args.n_components)
        df_msad = gc.relabel_state_dataframe(df_msad, args.n_components)
        df_msad_fits = gc.relabel_state_dataframe(df_msad_fits, args.n_components)
        if not df_msad.empty:
            df_msad["bead_name"] = bname
            df_msad["diameter_um"] = dia
            all_msad_curves.append(df_msad)
        if not df_msad_fits.empty:
            df_msad_fits["bead_name"] = bname
            df_msad_fits["diameter_um"] = dia
            all_msad_fits.append(df_msad_fits)

        turning_angles = gc.calc_state_dependent_turning_angles(df_obs, n_components=args.n_components)

        autocorr_data = gc.calc_state_dependent_autocorrelations(df_obs, n_components=args.n_components, frame_interval=args.frame_interval, max_lag_frames=25)
        autocorr_fits = {}
        for s in range(args.n_components):
            autocorr_fits[s] = {}
            s_lbl = gc.get_state_label(s, args.n_components)
            for ctype in ["vacf", "oacf", "sacf"]:
                if s in autocorr_data and ctype in autocorr_data[s]:
                    f_res = gc.fit_autocorrelation_exponential(autocorr_data[s][ctype], max_lag_s=60.0)
                    autocorr_fits[s][ctype] = f_res
                    all_autocorr_records.append({
                        "bead_name": bname,
                        "diameter_um": dia,
                        "state": s,
                        "state_label": s_lbl,
                        "mode": ctype,
                        "tau_corr_s": f_res.get("tau_corr_s", np.nan),
                        "tau_err_s": f_res.get("tau_err_s", np.nan),
                        "tau_int_zero_s": f_res.get("tau_int_zero_s", np.nan),
                        "r_squared": f_res.get("r_squared", np.nan),
                        "count": f_res.get("count", 0),
                    })

        # ABP fit for Run & Brownian fit for Tumble
        abp_fit_res = {}
        tumble_diff_res = {}
        if args.n_components == 2:
            sub_run_msd = df_msd[df_msd["state"] == 1].sort_values("lag_time_s")
            run_speeds = df_obs.loc[df_obs["pred_state"] == 1, "v"].to_numpy()
            v0_run_dim = float(np.exp(np.mean(np.log(np.maximum(run_speeds, 1e-12))))) if len(run_speeds) > 0 else 0.0
            if not sub_run_msd.empty and v0_run_dim > 0:
                abp_fit_res = gc.fit_active_brownian_msd(
                    sub_run_msd["lag_time_s"].values,
                    sub_run_msd["msd_um2"].values,
                    v0=v0_run_dim,
                    sem_vals=sub_run_msd["msd_sem_um2"].values,
                )
                all_abp_records.append({
                    "bead_name": bname,
                    "diameter_um": dia,
                    "v0_geom_um_s": v0_run_dim,
                    "Dt_um2_s": abp_fit_res.get("Dt_um2_s", np.nan),
                    "Dt_err_um2_s": abp_fit_res.get("Dt_err_um2_s", np.nan),
                    "tau_r_s": abp_fit_res.get("tau_r_s", np.nan),
                    "tau_r_err_s": abp_fit_res.get("tau_r_err_s", np.nan),
                    "D_eff_um2_s": abp_fit_res.get("D_eff_um2_s", np.nan),
                    "lambda_p_um": abp_fit_res.get("lambda_p_um", np.nan),
                    "r_squared": abp_fit_res.get("r_squared", np.nan),
                })

            sub_tum_msd = df_msd[df_msd["state"] == 0].sort_values("lag_time_s")
            if not sub_tum_msd.empty:
                tumble_diff_res = gc.fit_tumble_brownian_msd(
                    sub_tum_msd["lag_time_s"].values,
                    sub_tum_msd["msd_um2"].values,
                    sem_vals=sub_tum_msd["msd_sem_um2"].values,
                    fit_max_tau_s=40.0,
                )
                d_m = dia * 1e-6
                d_th_um2_s = (1.380649e-23 * 298.15 / (3.0 * np.pi * 0.89e-3 * d_m)) * 1e12
                all_tumble_diff_records.append({
                    "bead_name": bname,
                    "diameter_um": dia,
                    "Dt_tumble_um2_s": tumble_diff_res.get("Dt_um2_s", np.nan),
                    "Dt_tumble_err_um2_s": tumble_diff_res.get("Dt_err_um2_s", np.nan),
                    "D_thermal_um2_s": d_th_um2_s,
                    "ratio_Dt_to_thermal": tumble_diff_res.get("Dt_um2_s", np.nan) / (d_th_um2_s + 1e-12),
                    "r_squared": tumble_diff_res.get("r_squared", np.nan),
                })

        df_state_sum = gmm_model.get_state_summary()
        df_state_sum["bead_name"] = bname
        df_state_sum["diameter_um"] = dia
        df_state_sum["n_observations"] = len(X)
        df_state_sum["n_particles"] = n_particles
        df_state_sum["n_segments"] = len(lengths)
        df_state_sum["separation_index_Sv"] = gmm_model.get_separation_index()
        boundary = gmm_model.get_decision_boundary()
        df_state_sum["decision_threshold_log"] = boundary if boundary is not None else np.nan
        df_state_sum["decision_threshold_v_norm"] = float(np.exp(boundary) - args.epsilon) if boundary is not None else np.nan
        df_state_sum["bic"] = bic_k
        df_state_sum["aic"] = aic_k
        df_state_sum["icl"] = icl_k
        df_state_sum["delta_bic_1to2"] = delta_bic_1to2

        state_counts = [int(np.sum(pred_states == s)) for s in range(args.n_components)]
        df_state_sum["n_state_obs"] = state_counts
        df_state_sum["empirical_frac"] = [c / max(len(pred_states), 1) for c in state_counts]
        df_state_sum["k_T_to_R_per_s"] = emp_trans["rates"].get("k_T_to_R_per_s", np.nan)
        df_state_sum["k_R_to_T_per_s"] = emp_trans["rates"].get("k_R_to_T_per_s", np.nan)

        # Dimensional speed stats per state
        dim_geom_speeds = []
        for s in range(args.n_components):
            vs = df_obs.loc[df_obs["pred_state"] == s, "v"].to_numpy()
            dim_geom_speeds.append(float(np.exp(np.mean(np.log(np.maximum(vs, 1e-12))))) if len(vs) > 0 else np.nan)
        df_state_sum["mean_speed_geom_um_s"] = dim_geom_speeds

        dwell_pdf_fits, dwell_ccdf_fits = {}, {}
        emp_dwells, tau_pdfs, tau_ccdfs, tau_ccdf_errs, r2_ccdfs = [], [], [], [], []
        for s in range(args.n_components):
            t_list = dwell_times.get(s, [])
            fp = gc.hc.fit_exponential_pdf(t_list, frame_interval=args.frame_interval)
            fc = gc.fit_exponential_ccdf(t_list)
            dwell_pdf_fits[s] = fp
            dwell_ccdf_fits[s] = fc
            emp_dwells.append(fc["mean_empirical_s"])
            tau_pdfs.append(fp["tau_fit_s"])
            tau_ccdfs.append(fc["tau_fit_s"])
            tau_ccdf_errs.append(fc["tau_err_s"])
            r2_ccdfs.append(fc["r2_log"])

        df_state_sum["mean_dwell_emp_s"] = emp_dwells
        df_state_sum["tau_fit_pdf_s"] = tau_pdfs
        df_state_sum["tau_fit_ccdf_s"] = tau_ccdfs
        df_state_sum["tau_ccdf_err_s"] = tau_ccdf_errs
        df_state_sum["ccdf_exp_r2_log"] = r2_ccdfs
        all_summaries.append(df_state_sum)

        df_p_sum = gc.calc_per_particle_summary(df_obs, bname, dia)
        all_particle_records.append(df_p_sum)

        fitted_results[bname] = {
            "X": X,
            "lengths": lengths,
            "n_particles": n_particles,
            "df_obs": df_obs,
            "model": gmm_model,
            "pred_states": pred_states,
            "proba": proba,
            "conf_stats": conf_stats,
            "delta_bic_1to2": delta_bic_1to2,
            "dwell_times": dwell_times,
            "dwell_pdf_fits": dwell_pdf_fits,
            "dwell_ccdf_fits": dwell_ccdf_fits,
            "turning_angles": turning_angles,
            "autocorrelations": autocorr_data,
            "autocorr_fits": autocorr_fits,
            "abp_fit": abp_fit_res,
            "tumble_diff_fit": tumble_diff_res,
            "summary": df_state_sum,
            "df_msd": df_msd,
            "df_msd_fits": df_msd_fits,
            "df_msad": df_msad,
            "df_msad_fits": df_msad_fits,
        }

        for _, sr in df_state_sum.iterrows():
            print(
                f"  State {int(sr['state'])} ({sr['label']:6s}): "
                f"v/u_inf_geom={sr['v_norm_geom']:.3f} (mu={sr['mean_log_v_norm']:.3f}±{sr['std_log_v_norm']:.3f}), "
                f"weight={sr['mixture_weight']*100:.1f}%, emp={sr['empirical_frac']*100:.1f}%, "
                f"<dwell>={sr['mean_dwell_emp_s']:.1f}s"
            )
        print(f"  S_v={gmm_model.get_separation_index():.2f}, dBIC(1->2)={delta_bic_1to2:.1f}, "
              f"conf={conf_stats.get('mean_confidence', 0):.3f} (>=0.8: {conf_stats.get('high_conf_ratio_80', 0)*100:.1f}%)\n")

    if not fitted_results:
        print("[ERROR] No GMM models fitted.")
        return

    df_all_summary = pd.concat(all_summaries, ignore_index=True)
    df_trans = pd.DataFrame(all_trans_records)
    df_conf_all = pd.DataFrame(all_conf_records)
    df_msd_curves_all = pd.concat(all_msd_curves, ignore_index=True)
    df_msd_fits_all = pd.concat(all_msd_fits, ignore_index=True)
    df_msad_curves_all = pd.concat(all_msad_curves, ignore_index=True) if all_msad_curves else pd.DataFrame()
    df_msad_fits_all = pd.concat(all_msad_fits, ignore_index=True) if all_msad_fits else pd.DataFrame()
    df_autocorr_all = pd.DataFrame(all_autocorr_records)
    df_abp_all = pd.DataFrame(all_abp_records)
    df_tumble_diff_all = pd.DataFrame(all_tumble_diff_records)
    df_particle_all = pd.concat(all_particle_records, ignore_index=True) if all_particle_records else pd.DataFrame()

    print("=== Generating Figures and Summary CSVs ===")
    plot_emission_density(fitted_results, target_beads, out_dirs, n_components=args.n_components, epsilon=args.epsilon)
    evaluate_model_selection(fitted_results, target_beads, out_dirs, k_range=(1, 2, 3, 4))
    plot_trajectory_segmentation(fitted_results, target_beads, out_dirs, n_components=args.n_components)
    plot_timeseries_sync(fitted_results, target_beads, out_dirs, n_components=args.n_components, frame_interval=args.frame_interval)
    plot_posterior_distributions(fitted_results, target_beads, out_dirs, n_components=args.n_components)
    plot_dwell_time_distributions(fitted_results, target_beads, out_dirs, n_components=args.n_components)
    plot_state_msd(fitted_results, target_beads, out_dirs, n_components=args.n_components)
    df_angle_sum = plot_turning_angles(fitted_results, target_beads, out_dirs, n_components=args.n_components)
    plot_state_msad(fitted_results, target_beads, out_dirs, n_components=args.n_components)
    plot_autocorrelation_grid(fitted_results, target_beads, out_dirs, n_components=args.n_components)
    plot_summary_comparison(df_all_summary, df_msd_fits_all, df_angle_sum, df_particle_all, target_beads, out_dirs, n_components=args.n_components)

    k = args.n_components
    save_csv_to_dirs(df_all_summary, out_dirs, f"gmm_state_parameters_summary_k{k}.csv")
    save_csv_to_dirs(df_trans, out_dirs, f"gmm_empirical_transitions_k{k}.csv")
    save_csv_to_dirs(df_conf_all, out_dirs, f"gmm_posterior_confidence_summary_k{k}.csv")
    save_csv_to_dirs(df_msd_curves_all, out_dirs, f"gmm_state_msd_curves_k{k}.csv")
    save_csv_to_dirs(df_msd_fits_all, out_dirs, f"gmm_state_msd_fits_k{k}.csv")
    save_csv_to_dirs(df_msad_curves_all, out_dirs, f"gmm_msad_curves_k{k}.csv")
    save_csv_to_dirs(df_msad_fits_all, out_dirs, f"gmm_msad_fits_summary_k{k}.csv")
    save_csv_to_dirs(df_autocorr_all, out_dirs, f"gmm_autocorrelation_summary_k{k}.csv")
    save_csv_to_dirs(df_abp_all, out_dirs, f"gmm_run_abp_fits_summary_k{k}.csv")
    save_csv_to_dirs(df_tumble_diff_all, out_dirs, f"gmm_tumble_diffusion_fits_summary_k{k}.csv")
    save_csv_to_dirs(df_particle_all, out_dirs, f"gmm_per_particle_summary_k{k}.csv")

    print("\n=== GMM Analysis Completed Successfully ===")


if __name__ == "__main__":
    main()
