#!/usr/bin/env python3
"""
quantify_effective_potential_depth.py

有効配向ポテンシャル U_tilde_eff(theta) = -ln(2*pi*P(theta)) の「深さ」を
複数の相補的指標（モデルフリー実測値と GvM 2モーメント理論値）で定量化する。

指標の定義（すべて実効回転ノイズエネルギー k_B T_eff = zeta_r * D_r^eff 単位）:
1. 極性前後井戸深さ (Polar well depth):
   - 実測 (畳み込み |theta| ヒストグラム, 10 deg 刻み 18 bin):
       Delta U_emp(pi - 0) = U_eff(|theta| ~ pi) - U_eff(|theta| ~ 0)
   - GvM 理論値:
       Delta U_GvM(pi - 0) = U_GvM(pi) - U_GvM(0) = 2 * kappa_1
2. 全脱出障壁高さ / ポテンシャル全深さ (Total barrier depth):
   - 実測 (畳み込み |theta|):
       Delta U_emp(barrier) = max_{|theta|} U_eff(|theta|) - U_eff(|theta| ~ 0)
     (最小値が bin 0 以外のときは min も参考記録し、max - min も算出)
   - GvM 理論値:
       U_GvM(c) = -2*kappa_2*c^2 - kappa_1*c + const の最大点 c_* = max(-1, -kappa_1/(4*kappa_2)) (kappa_2 > 0)
       Delta U_GvM(barrier) = U_GvM(c_*) - U_GvM(+1)
     ※ 4*kappa_2 > kappa_1 のとき (kappa_1 + 4*kappa_2)^2 / (8*kappa_2), それ以外は 2*kappa_1
3. ネマチック拘束深さ (Nematic rail depth):
   - GvM 2次成分: 2 * kappa_2 (= U_nematic(pi/2) - U_nematic(0))

対象データ:
- 全粒子 (未分離) 6サイズ: scratch/per_point_v_dot_u_hist_cache.pkl
- 小粒子 GMM Run / Tumble 分離 (0.63 um, 1.18 um): scratch/gmm_run_tumble_per_point_hist_cache.pkl
  相関長 R_c / xi_state: figure/gmm_1d/gmm_xi_and_rc_over_xi_summary_k2.csv (source == 'angular_correlation_w')

既存グラフ (per_point_delta_U_vs_rc_xi.*) は上書きせず、新しいファイル名で出力する。
"""

import pickle
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import verify_cos_decomposition as vcd
from plot_gvm_potential import solve_gvm_from_moments

warnings.filterwarnings("ignore")

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = vcd.ROOT_DIR

UNSPLIT_CACHE = CURRENT_DIR / "scratch" / "per_point_v_dot_u_hist_cache.pkl"
GMM_CACHE = CURRENT_DIR / "scratch" / "gmm_run_tumble_per_point_hist_cache.pkl"
GMM_XI_CSV = CURRENT_DIR / "figure" / "gmm_1d" / "gmm_xi_and_rc_over_xi_summary_k2.csv"

OUT_DIR_MAIN = CURRENT_DIR / "figure" / "per_point_v_dot_u"
OUT_DIR_GMM = CURRENT_DIR / "figure" / "gmm_1d"
MIRROR_DIRS = []
if ROOT_DIR.exists():
    MIRROR_DIRS = [
        ROOT_DIR / "figure" / "per_point_v_dot_u",
        ROOT_DIR / "figure" / "gmm_1d",
    ]

ORDER_BEADS = ["beads06um", "beads1um", "beads3um", "beads7um", "beads5um", "beads20um"]
BEAD_META = {
    "beads06um": (0.63, r"0.63 $\mu\mathrm{m}$", "#1f77b4"),
    "beads1um": (1.18, r"1.18 $\mu\mathrm{m}$", "#ff7f0e"),
    "beads3um": (3.37, r"3.37 $\mu\mathrm{m}$", "#2ca02c"),
    "beads7um": (7.24, r"7.24 $\mu\mathrm{m}$", "#d62728"),
    "beads5um": (5.00, r"5.00 $\mu\mathrm{m}$", "#9467bd"),
    "beads20um": (20.0, r"20.0 $\mu\mathrm{m}$", "#8c564b"),
}

X_LOCK = 45.0 * np.pi / 512.0   # ~0.276
X_BULK = 45.0 * np.pi / 128.0   # ~1.104

DTH_FOLD = np.pi / 18.0
TH_FOLD_DEG = np.arange(18) * 10.0 + 5.0  # 5, 15, ..., 175 deg


def gvm_barrier_metrics(k1: float, k2: float) -> Tuple[float, float, float, float]:
    """
    GvM 理論ポテンシャル U_GvM(c) = -2*k2*c^2 - k1*c の深さ指標を返す。
    Returns:
      du_polar: 2 * k1
      du_nematic: 2 * k2
      du_barrier: max_{c in [-1,1]} U_GvM(c) - U_GvM(+1)
      c_star: 障壁頂点の c = cos(theta_*)
    """
    du_polar = 2.0 * k1
    du_nematic = 2.0 * k2
    if k2 > 0.0 and 4.0 * k2 > abs(k1):
        c_star = -k1 / (4.0 * k2)
    elif k1 >= 0.0:
        c_star = -1.0
    else:
        c_star = 1.0
    c_star = float(np.clip(c_star, -1.0, 1.0))
    u_at_star = -2.0 * k2 * (c_star ** 2) - k1 * c_star
    u_at_one = -2.0 * k2 - k1
    du_barrier = float(u_at_star - u_at_one)
    return float(du_polar), float(du_nematic), du_barrier, c_star


def empirical_depth_from_hist_th(h_th: np.ndarray) -> Dict[str, float]:
    """
    36-bin (-pi..pi) ヒストグラムを |theta| in [0, pi] の 18-bin (10 deg 刻み) に畳み込み、
    実効ポテンシャル U_eff(|theta|) = -ln(pi * P(|theta|)) の深さ指標を計算する。
    """
    h_fold = (h_th[18:36] + h_th[17::-1]).astype(float)
    tot = float(np.sum(h_fold))
    if tot <= 0:
        return {
            "du_emp_pi0": np.nan,
            "du_emp_bar": np.nan,
            "du_emp_maxmin": np.nan,
            "th_bar_deg": np.nan,
        }
    p_fold = h_fold / (tot * DTH_FOLD)
    with np.errstate(divide="ignore", invalid="ignore"):
        u_fold = np.where(p_fold > 0.0, -np.log(np.pi * p_fold), np.nan)

    u0 = float(u_fold[0])
    upi = float(u_fold[-1])
    finite_mask = np.isfinite(u_fold)
    if not np.any(finite_mask):
        return {
            "du_emp_pi0": np.nan,
            "du_emp_bar": np.nan,
            "du_emp_maxmin": np.nan,
            "th_bar_deg": np.nan,
        }
    idx_max = int(np.nanargmax(u_fold))
    umax = float(u_fold[idx_max])
    umin = float(np.nanmin(u_fold))
    return {
        "du_emp_pi0": float(upi - u0),
        "du_emp_bar": float(umax - u0),
        "du_emp_maxmin": float(umax - umin),
        "th_bar_deg": float(TH_FOLD_DEG[idx_max]),
    }


def compute_group_metrics(recs: List[dict]) -> Dict[str, float]:
    """粒子レコード群を集約して全深さ指標を算出する。"""
    h_th = np.sum([r["hist_th"] for r in recs], axis=0)
    n_pts = int(sum(r["n_pts"] for r in recs))
    m1 = float(sum(r["sum_c"] for r in recs) / n_pts)
    m2 = float(2.0 * sum(r["sum_c2"] for r in recs) / n_pts - 1.0)
    k1, k2, _ = solve_gvm_from_moments(m1, m2)
    du_polar, du_nematic, du_gvm_bar, c_star = gvm_barrier_metrics(k1, k2)
    emp = empirical_depth_from_hist_th(h_th)
    return {
        "n_pts": n_pts,
        "m1": m1,
        "m2": m2,
        "kappa_1": k1,
        "kappa_2": k2,
        "c_star": c_star,
        "th_star_deg": float(np.rad2deg(np.arccos(np.clip(c_star, -1.0, 1.0)))),
        "du_gvm_polar": du_polar,
        "du_gvm_nematic": du_nematic,
        "du_gvm_bar": du_gvm_bar,
        **emp,
    }


def bootstrap_group(recs: List[dict], n_boot: int = 1000, seed: int = 42) -> Dict[str, Tuple[float, float]]:
    """粒子クラスタ・ブートストラップで 95% 信頼区間を算出する。"""
    rng = np.random.default_rng(seed)
    n = len(recs)
    keys = [
        "du_emp_pi0", "du_emp_bar", "du_emp_maxmin",
        "du_gvm_polar", "du_gvm_nematic", "du_gvm_bar",
        "kappa_1", "kappa_2",
    ]
    samples = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sub = [recs[i] for i in idx]
        met = compute_group_metrics(sub)
        for k in keys:
            val = met[k]
            if np.isfinite(val):
                samples[k].append(val)

    cis = {}
    for k in keys:
        arr = np.asarray(samples[k], dtype=float)
        if len(arr) >= 10:
            cis[k] = (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))
        else:
            cis[k] = (np.nan, np.nan)
    return cis


def add_reference_lines(ax: plt.Axes, y_text: float = None, ha_lock: str = "left", ha_bulk: str = "left") -> None:
    ax.axvline(X_LOCK, color="0.45", ls="--", lw=1.2, zorder=1)
    ax.axvline(X_BULK, color="0.45", ls=":", lw=1.2, zorder=1)
    ax.axhline(1.0, color="#555555", ls="-.", lw=1.1, alpha=0.85, zorder=1)
    if y_text is not None:
        x_l = X_LOCK * (1.05 if ha_lock == "left" else 0.95)
        x_b = X_BULK * (1.05 if ha_bulk == "left" else 0.95)
        ax.text(x_l, y_text, r"$x_{\mathrm{lock}} \approx 0.28$",
                fontsize=8.4, color="0.35", va="center", ha=ha_lock)
        ax.text(x_b, y_text, r"$x_{\mathrm{bulk}} \approx 1.10$",
                fontsize=8.4, color="0.35", va="center", ha=ha_bulk)


def format_diam_label(d: float) -> str:
    if d in (0.63, 1.18, 3.37, 7.24):
        return f"{d:.2f}" + r" $\mu\mathrm{m}$"
    if d == 5.0:
        return r"5.0 $\mu\mathrm{m}$"
    return f"{d:.0f}" + r" $\mu\mathrm{m}$"


def save_fig_multi(fig: plt.Figure, dirs: List[Path], stem: str) -> List[Path]:
    saved = []
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        for ext in ("png", "svg"):
            p = d / f"{stem}.{ext}"
            fig.savefig(p, dpi=300 if ext == "png" else None, bbox_inches="tight")
            saved.append(p)
    return saved


def main():
    plt.rcParams.update({
        "font.size": 10.5,
        "axes.labelsize": 11.5,
        "axes.titlesize": 12.0,
        "legend.fontsize": 9.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    # ------------------------------------------------------------------
    # 1. Load Unsplit 6-size cache
    # ------------------------------------------------------------------
    with open(UNSPLIT_CACHE, "rb") as f:
        unsplit_recs = pickle.load(f)
    by_bead = defaultdict(list)
    for r in unsplit_recs:
        by_bead[r["bead_name"]].append(r)

    rows_unsplit = []
    for idx_b, bname in enumerate(ORDER_BEADS):
        recs = by_bead[bname]
        diam, label, color = BEAD_META[bname]
        rc_vals = np.array([r["rc_over_xi"] for r in recs], dtype=float)
        rc_mean = float(np.mean(rc_vals))
        rc_sem = float(np.std(rc_vals, ddof=1) / np.sqrt(len(rc_vals))) if len(rc_vals) > 1 else 0.0
        met = compute_group_metrics(recs)
        cis = bootstrap_group(recs, n_boot=1000, seed=100 + idx_b)
        row = {
            "bead_name": bname,
            "diameter_um": diam,
            "label": label,
            "n_particles": len(recs),
            "rc_over_xi": rc_mean,
            "rc_over_xi_sem": rc_sem,
            **met,
        }
        for k, (lo, hi) in cis.items():
            row[f"{k}_ci_lo"] = lo
            row[f"{k}_ci_hi"] = hi
        rows_unsplit.append(row)

    df_unsplit = pd.DataFrame(rows_unsplit)

    # ------------------------------------------------------------------
    # 2. Load GMM Run/Tumble cache & state-specific R_c / xi
    # ------------------------------------------------------------------
    with open(GMM_CACHE, "rb") as f:
        gmm_recs = pickle.load(f)
    by_gmm = defaultdict(list)
    for r in gmm_recs:
        by_gmm[(r["bead_name"], r["state_label"])].append(r)

    df_xi = pd.read_csv(GMM_XI_CSV)
    xi_w = df_xi[df_xi["source"] == "angular_correlation_w"].set_index(["bead_name", "mode"])

    rows_gmm = []
    gmm_keys = [
        ("beads06um", "Run", 0.63, "0.63 $\\mu\\mathrm{m}$ Run", "#1f77b4", "^"),
        ("beads06um", "Tumble", 0.63, "0.63 $\\mu\\mathrm{m}$ Tumble", "#6baed6", "v"),
        ("beads1um", "Run", 1.18, "1.18 $\\mu\\mathrm{m}$ Run", "#d95f02", "^"),
        ("beads1um", "Tumble", 1.18, "1.18 $\\mu\\mathrm{m}$ Tumble", "#fdbe85", "v"),
    ]
    for idx_g, (bname, st, diam, label, color, marker) in enumerate(gmm_keys):
        recs = by_gmm[(bname, st)]
        xi_row = xi_w.loc[(bname, st)]
        met = compute_group_metrics(recs)
        cis = bootstrap_group(recs, n_boot=1000, seed=500 + idx_g)
        row = {
            "bead_name": bname,
            "state": st,
            "diameter_um": diam,
            "label": label,
            "xi_um": float(xi_row["xi_um"]),
            "rc_over_xi": float(xi_row["rc_over_xi"]),
            "rc_xi_ci_lo": float(xi_row["rc_xi_ci_lo"]),
            "rc_xi_ci_hi": float(xi_row["rc_xi_ci_hi"]),
            "n_particles": len(recs),
            "n_frames": int(sum(r["n_frames"] for r in recs)),
            **met,
        }
        for k, (lo, hi) in cis.items():
            row[f"{k}_ci_lo"] = lo
            row[f"{k}_ci_hi"] = hi
        rows_gmm.append(row)

    # Large cargos appended for unified state-resolved comparison
    for _, r_u in df_unsplit[df_unsplit["bead_name"].isin(["beads3um", "beads7um", "beads5um", "beads20um"])].iterrows():
        row = {
            "bead_name": r_u["bead_name"],
            "state": "All",
            "diameter_um": r_u["diameter_um"],
            "label": r_u["label"],
            "xi_um": np.nan,
            "rc_over_xi": r_u["rc_over_xi"],
            "rc_xi_ci_lo": r_u["rc_over_xi"] - r_u["rc_over_xi_sem"],
            "rc_xi_ci_hi": r_u["rc_over_xi"] + r_u["rc_over_xi_sem"],
            "n_particles": int(r_u["n_particles"]),
            "n_frames": np.nan,
        }
        for col in [
            "n_pts", "m1", "m2", "kappa_1", "kappa_2", "c_star", "th_star_deg",
            "du_gvm_polar", "du_gvm_nematic", "du_gvm_bar",
            "du_emp_pi0", "du_emp_bar", "du_emp_maxmin", "th_bar_deg",
            "du_emp_pi0_ci_lo", "du_emp_pi0_ci_hi",
            "du_emp_bar_ci_lo", "du_emp_bar_ci_hi",
            "du_emp_maxmin_ci_lo", "du_emp_maxmin_ci_hi",
            "du_gvm_polar_ci_lo", "du_gvm_polar_ci_hi",
            "du_gvm_nematic_ci_lo", "du_gvm_nematic_ci_hi",
            "du_gvm_bar_ci_lo", "du_gvm_bar_ci_hi",
            "kappa_1_ci_lo", "kappa_1_ci_hi",
            "kappa_2_ci_lo", "kappa_2_ci_hi",
        ]:
            row[col] = r_u[col]
        rows_gmm.append(row)

    df_gmm_all = pd.DataFrame(rows_gmm)

    # Save CSVs
    OUT_DIR_MAIN.mkdir(parents=True, exist_ok=True)
    OUT_DIR_GMM.mkdir(parents=True, exist_ok=True)
    csv_main = OUT_DIR_MAIN / "effective_potential_depth_summary.csv"
    csv_gmm = OUT_DIR_GMM / "effective_potential_depth_gmm_state_summary.csv"
    df_unsplit.to_csv(csv_main, index=False)
    df_gmm_all.to_csv(csv_gmm, index=False)
    for mdir in MIRROR_DIRS:
        if mdir.parent.exists():
            mdir.mkdir(parents=True, exist_ok=True)
            if "per_point_v_dot_u" in str(mdir):
                df_unsplit.to_csv(mdir / csv_main.name, index=False)
            else:
                df_gmm_all.to_csv(mdir / csv_gmm.name, index=False)

    # ------------------------------------------------------------------
    # Figure 1: Unsplit 6 sizes - Empirical vs GvM barrier + Polar/Nematic decomposition
    # ------------------------------------------------------------------
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 5.1))
    fig1.subplots_adjust(top=0.84, wspace=0.25)

    x = df_unsplit["rc_over_xi"].values
    # Panel A: Total empirical barrier vs polar depth vs GvM theoretical barrier
    y_bar = df_unsplit["du_emp_bar"].values
    y_bar_err = np.vstack([y_bar - df_unsplit["du_emp_bar_ci_lo"].values,
                           df_unsplit["du_emp_bar_ci_hi"].values - y_bar])
    y_pi0 = df_unsplit["du_emp_pi0"].values
    y_pi0_err = np.vstack([y_pi0 - df_unsplit["du_emp_pi0_ci_lo"].values,
                           df_unsplit["du_emp_pi0_ci_hi"].values - y_pi0])
    y_gvm_bar = df_unsplit["du_gvm_bar"].values

    ax1.errorbar(
        x, y_bar, yerr=y_bar_err, fmt="-o", color="#1b4f72", lw=2.0, ms=7, capsize=3.5,
        label=r"Empirical barrier $\Delta\tilde{U}_{\mathrm{emp,bar}} = \max_\theta\tilde{U}-\tilde{U}(0)$", zorder=4
    )
    ax1.errorbar(
        x, y_pi0, yerr=y_pi0_err, fmt="--s", color="#2e86c1", lw=1.6, ms=6, capsize=3.0, alpha=0.9,
        label=r"Empirical polar $\Delta\tilde{U}_{\pi-0} = \tilde{U}(\pi)-\tilde{U}(0)$", zorder=3
    )
    ax1.plot(
        x, y_gvm_bar, ":D", color="#d35400", lw=1.9, ms=6,
        label=r"GvM theory $\Delta\tilde{U}_{\mathrm{GvM,bar}} = \tilde{U}_{\mathrm{GvM}}(c_*)-\tilde{U}_{\mathrm{GvM}}(1)$", zorder=5
    )

    ax1.set_xscale("log")
    ax1.set_xlim(0.045, 3.0)
    ax1.set_ylim(-0.08, 2.75)
    add_reference_lines(ax1, y_text=1.62, ha_lock="left", ha_bulk="left")
    ax1.text(2.75, 1.05, r"Noise threshold $1\,k_B T_{\mathrm{eff}}$", fontsize=8.4, color="#444444", va="bottom", ha="right")

    # Annotate cargo diameters cleanly
    offsets_1 = {
        "beads06um": (0.0, 12),
        "beads1um": (12, 10),
        "beads3um": (10, 9),
        "beads7um": (12, 7),
        "beads5um": (10, 7),
        "beads20um": (-10, 13),
    }
    for _, r in df_unsplit.iterrows():
        dx, dy = offsets_1[r["bead_name"]]
        ax1.annotate(
            format_diam_label(r["diameter_um"]),
            xy=(r["rc_over_xi"], r["du_emp_bar"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8.2,
            color="#1b4f72",
            ha="center",
        )

    ax1.set_xlabel(r"$R_c / \xi$")
    ax1.set_ylabel(r"Effective Potential Depth $\Delta\tilde{U}$ [$k_B T_{\mathrm{eff}}$]")
    ax1.set_title("A. Total Escape Barrier vs. Polar Depth")
    ax1.legend(loc="upper right", frameon=True, framealpha=0.94, fontsize=8.1)

    # Panel B: GvM decomposition into polar 2*kappa_1 vs nematic 2*kappa_2
    y_k1 = df_unsplit["du_gvm_polar"].values
    y_k1_err = np.vstack([y_k1 - df_unsplit["du_gvm_polar_ci_lo"].values,
                          df_unsplit["du_gvm_polar_ci_hi"].values - y_k1])
    y_k2 = df_unsplit["du_gvm_nematic"].values
    y_k2_err = np.vstack([y_k2 - df_unsplit["du_gvm_nematic_ci_lo"].values,
                          df_unsplit["du_gvm_nematic_ci_hi"].values - y_k2])

    ax2.plot(
        x, y_gvm_bar, "-D", color="#d35400", lw=2.0, ms=6.5,
        label=r"Total GvM barrier $\Delta\tilde{U}_{\mathrm{GvM,bar}}$", zorder=5
    )
    ax2.errorbar(
        x, y_k1, yerr=y_k1_err, fmt="-o", color="#c0392b", lw=1.8, ms=6.5, capsize=3.0,
        label=r"Polar term $2\kappa_1$ ($\cos\theta$ drive)", zorder=4
    )
    ax2.errorbar(
        x, y_k2, yerr=y_k2_err, fmt="--^", color="#7d3c98", lw=1.8, ms=6.5, capsize=3.0,
        label=r"Nematic rail $2\kappa_2$ ($\cos 2\theta$ confinement)", zorder=3
    )

    ax2.set_xscale("log")
    ax2.set_xlim(0.045, 3.0)
    ax2.set_ylim(-0.08, 2.25)
    add_reference_lines(ax2, y_text=1.35, ha_lock="left", ha_bulk="left")
    ax2.set_xlabel(r"$R_c / \xi$")
    ax2.set_ylabel(r"GvM Depth Component [$k_B T_{\mathrm{eff}}$]")
    ax2.set_title(r"B. Polar ($2\kappa_1$) vs. Nematic ($2\kappa_2$) Contributions")
    ax2.legend(loc="upper right", frameon=True, framealpha=0.94, fontsize=8.4)

    fig1.suptitle("Effective Alignment Potential Depth across Cargo Sizes (Unsplit)", fontsize=13.0, y=0.96)
    save_fig_multi(fig1, [OUT_DIR_MAIN] + ([MIRROR_DIRS[0]] if MIRROR_DIRS else []), "effective_potential_depth_vs_rc_xi")
    plt.close(fig1)

    # ------------------------------------------------------------------
    # Figure 2: GMM State-resolved (Run vs Tumble) Potential Depth vs R_c / xi_state
    # ------------------------------------------------------------------
    fig2, (axg1, axg2) = plt.subplots(1, 2, figsize=(13.0, 5.3))
    fig2.subplots_adjust(top=0.84, wspace=0.24)

    # Faint unsplit small-cargo reference markers on axg1
    for bname in ["beads06um", "beads1um"]:
        ru = df_unsplit[df_unsplit["bead_name"] == bname].iloc[0]
        axg1.scatter(
            ru["rc_over_xi"], ru["du_emp_bar"], s=65, facecolors="none", edgecolors="0.55",
            lw=1.4, zorder=2,
            label=r"Unsplit 0.63 & 1.18 $\mu\mathrm{m}$" if bname == "beads06um" else None,
        )

    # Plot large cargos first
    df_large = df_gmm_all[df_gmm_all["state"] == "All"].copy()
    xl = df_large["rc_over_xi"].values
    yl = df_large["du_emp_bar"].values
    yl_err = np.vstack([yl - df_large["du_emp_bar_ci_lo"].values,
                        df_large["du_emp_bar_ci_hi"].values - yl])
    xl_err = np.vstack([xl - df_large["rc_xi_ci_lo"].values,
                        df_large["rc_xi_ci_hi"].values - xl])
    axg1.errorbar(
        xl, yl, xerr=xl_err, yerr=yl_err, fmt="-o", color="#2c3e50", lw=1.7, ms=7, capsize=3.0,
        label=r"Large cargos ($\geq 3.37\,\mu\mathrm{m}$, all)", zorder=4
    )
    large_offsets = {"beads3um": (8, 12), "beads7um": (14, 8), "beads5um": (10, 9), "beads20um": (-12, 13)}
    for _, rl in df_large.iterrows():
        dx, dy = large_offsets[rl["bead_name"]]
        axg1.annotate(
            format_diam_label(rl["diameter_um"]),
            xy=(rl["rc_over_xi"], rl["du_emp_bar"]),
            xytext=(dx, dy), textcoords="offset points",
            fontsize=8.2, color="#2c3e50", ha="center",
        )

    # Plot GMM states with both horizontal and vertical 95% CI
    gmm_plot_style = {
        ("beads06um", "Run"): ("#1f77b4", "^", r"0.63 $\mu\mathrm{m}$ Run", (10, -4)),
        ("beads06um", "Tumble"): ("#5dade2", "v", r"0.63 $\mu\mathrm{m}$ Tumble", (14, 14)),
        ("beads1um", "Run"): ("#d35400", "^", r"1.18 $\mu\mathrm{m}$ Run", (10, 8)),
        ("beads1um", "Tumble"): ("#f39c12", "v", r"1.18 $\mu\mathrm{m}$ Tumble", (-28, -16)),
    }
    for _, rg in df_gmm_all[df_gmm_all["state"].isin(["Run", "Tumble"])].iterrows():
        c, m, lbl, (dx, dy) = gmm_plot_style[(rg["bead_name"], rg["state"])]
        xv = rg["rc_over_xi"]
        yv = rg["du_emp_bar"]
        xerr = [[xv - rg["rc_xi_ci_lo"]], [rg["rc_xi_ci_hi"] - xv]]
        yerr = [[yv - rg["du_emp_bar_ci_lo"]], [rg["du_emp_bar_ci_hi"] - yv]]
        axg1.errorbar(
            [xv], [yv], xerr=xerr, yerr=yerr, fmt=m, color=c, ms=9.5, capsize=3.5, lw=1.6,
            markeredgecolor="black", markeredgewidth=0.6, label=lbl, zorder=6
        )
        # Also plot GvM barrier as open diamond at same x
        axg1.scatter(
            [xv], [rg["du_gvm_bar"]], s=48, facecolors="none", edgecolors=c, marker="D", lw=1.5, zorder=5
        )
        axg1.annotate(
            f"{yv:.2f}",
            xy=(xv, yv), xytext=(dx, dy), textcoords="offset points",
            fontsize=8.5, fontweight="bold", color=c,
        )

    # Proxy legend entry for GvM diamond markers
    axg1.scatter([], [], s=45, facecolors="none", edgecolors="0.3", marker="D", lw=1.4,
                 label=r"GvM barrier $\Delta\tilde{U}_{\mathrm{GvM,bar}}$ (open)")

    # Dashed connectors from unsplit -> Run and Tumble for intuition
    for bname, col in [("beads06um", "#1f77b4"), ("beads1um", "#d35400")]:
        ru = df_unsplit[df_unsplit["bead_name"] == bname].iloc[0]
        sub_g = df_gmm_all[df_gmm_all["bead_name"] == bname]
        for _, rg in sub_g.iterrows():
            axg1.plot(
                [ru["rc_over_xi"], rg["rc_over_xi"]],
                [ru["du_emp_bar"], rg["du_emp_bar"]],
                ls=":", lw=1.0, color=col, alpha=0.55, zorder=2
            )

    axg1.set_xscale("log")
    axg1.set_xlim(0.022, 3.2)
    axg1.set_ylim(-0.08, 5.1)
    add_reference_lines(axg1, y_text=2.45, ha_lock="left", ha_bulk="left")
    axg1.text(2.95, 1.08, r"Noise threshold $1\,k_B T_{\mathrm{eff}}$", fontsize=8.4, color="#444444", va="bottom", ha="right")
    axg1.set_xlabel(r"State-Resolved $R_c / \xi_{\mathrm{state}}$")
    axg1.set_ylabel(r"Empirical Barrier Depth $\Delta\tilde{U}_{\mathrm{emp,bar}}$ [$k_B T_{\mathrm{eff}}$]")
    axg1.set_title("A. Effective Barrier Depth: GMM Run vs. Tumble")
    axg1.legend(loc="upper right", frameon=True, framealpha=0.94, fontsize=7.9)

    # Panel B: Stacked / grouped comparison of Polar 2*k1 vs Nematic 2*k2 vs Empirical barrier
    labels_bar = []
    pol_vals = []
    nem_vals = []
    bar_vals = []
    emp_vals = []
    ordered_rows = [
        ("beads06um", "Run", "0.63\nRun"),
        ("beads06um", "Tumble", "0.63\nTumble"),
        ("beads1um", "Run", "1.18\nRun"),
        ("beads1um", "Tumble", "1.18\nTumble"),
        ("beads3um", "All", "3.37"),
        ("beads7um", "All", "7.24"),
        ("beads5um", "All", "5.00"),
        ("beads20um", "All", "20.0"),
    ]
    for bname, st, short_lbl in ordered_rows:
        r = df_gmm_all[(df_gmm_all["bead_name"] == bname) & (df_gmm_all["state"] == st)].iloc[0]
        labels_bar.append(short_lbl)
        pol_vals.append(r["du_gvm_polar"])
        nem_vals.append(r["du_gvm_nematic"])
        bar_vals.append(r["du_gvm_bar"])
        emp_vals.append(r["du_emp_bar"])

    xpos = np.arange(len(labels_bar))
    w = 0.27
    axg2.bar(xpos - w, pol_vals, width=w, color="#c0392b", alpha=0.88, label=r"GvM Polar $2\kappa_1$")
    axg2.bar(xpos, nem_vals, width=w, color="#7d3c98", alpha=0.88, label=r"GvM Nematic $2\kappa_2$")
    axg2.bar(xpos + w, emp_vals, width=w, color="#1b4f72", alpha=0.88, label=r"Empirical barrier $\Delta\tilde{U}_{\mathrm{emp,bar}}$")
    axg2.plot(xpos + w, bar_vals, "D", color="#f39c12", ms=5.5, markeredgecolor="black", markeredgewidth=0.5,
              label=r"GvM barrier $\Delta\tilde{U}_{\mathrm{GvM,bar}}$", zorder=5)
    axg2.axhline(1.0, color="#555555", ls="-.", lw=1.1, alpha=0.85, zorder=1)
    axg2.set_xticks(xpos)
    axg2.set_xticklabels(labels_bar, fontsize=8.8)
    axg2.set_ylabel(r"Potential Depth [$k_B T_{\mathrm{eff}}$]")
    axg2.set_title("B. Polar vs. Nematic vs. Total Escape Barrier")
    axg2.legend(loc="upper right", frameon=True, framealpha=0.94, fontsize=8.2)

    fig2.suptitle("State-Resolved Effective Potential Depth (GMM Run/Tumble & Large Cargos)", fontsize=13.0, y=0.96)
    save_fig_multi(fig2, [OUT_DIR_GMM] + ([MIRROR_DIRS[1]] if len(MIRROR_DIRS) > 1 else []), "gmm_state_potential_depth_vs_rc_xi")
    plt.close(fig2)

    print("=== Unsplit 6 Cargo Sizes (Effective Potential Depth) ===")
    cols_show_u = [
        "bead_name", "diameter_um", "rc_over_xi",
        "du_emp_pi0", "du_emp_pi0_ci_lo", "du_emp_pi0_ci_hi",
        "du_emp_bar", "du_emp_bar_ci_lo", "du_emp_bar_ci_hi", "th_bar_deg",
        "du_gvm_polar", "du_gvm_nematic", "du_gvm_bar", "th_star_deg",
    ]
    print(df_unsplit[cols_show_u].to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print("\n=== GMM State-Resolved + Large Cargos ===")
    cols_show_g = [
        "bead_name", "state", "diameter_um", "rc_over_xi", "rc_xi_ci_lo", "rc_xi_ci_hi",
        "du_emp_pi0", "du_emp_bar", "du_emp_bar_ci_lo", "du_emp_bar_ci_hi", "th_bar_deg",
        "du_gvm_polar", "du_gvm_nematic", "du_gvm_bar", "th_star_deg",
    ]
    print(df_gmm_all[cols_show_g].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nSaved CSVs:\n  {csv_main}\n  {csv_gmm}")


if __name__ == "__main__":
    main()
