#!/usr/bin/env python3
"""
plot_transfer_efficiency_synthesis.py

貨物輸送の「伝達効率」を包括的に可視化・定量化するスクリプト（既存グラフは上書きしない）。

算出する伝達効率・分解指標（95% 粒子クラスターブートストラップ CI）:
  1. 速度伝達係数（回帰型・エネルギー重み）: eta_reg = sum(v_c . u_bar) / sum(|u_bar|^2)
  2. 平行投影速度伝達効率（線形重み）     : eta_par = sum(v_parallel) / sum(|u_bar|),  v_parallel = (v_c . u_bar) / |u_bar|
  3. 流動仕事率伝達効率（正規化余弦）       : eta_T   = <v_c . u_bar> / (v_rms * u_rms)
  4. 分解因子:
       - 進行方向追随度 <cos Theta_bulk> (= mean of particle cos_bulk) と <cos theta_pt>
       - スカラー速さ保持比 <|v_c|> / <|u_bar|>
  5. 有効ポテンシャル障壁 Delta U_emp,bar と KL ダイバージェンス D_KL,total との対応

対象:
  - 全粒子 (未分離) 6 サイズ: scratch/contrast_inst_cache.pkl
  - 小粒子 (0.63 um, 1.18 um) GMM Run/Tumble 状態分離: analyze_gmm_run_tumble_potential_and_kl.build_gmm_labeled_frames()
    + figure/gmm_1d/gmm_xi_and_rc_over_xi_summary_k2.csv

出力先:
  - figure/transfer_efficiency_synthesis/
  - ROOT_DIR/figure/transfer_efficiency_synthesis/
"""

import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))

import verify_cos_decomposition as vcd
from analyze_gmm_run_tumble_potential_and_kl import build_gmm_labeled_frames

ROOT_DIR = vcd.ROOT_DIR
INST_CACHE = CURRENT_DIR / "scratch" / "contrast_inst_cache.pkl"
POT_MAIN_CSV = CURRENT_DIR / "figure" / "per_point_v_dot_u" / "effective_potential_depth_summary.csv"
POT_GMM_CSV = CURRENT_DIR / "figure" / "gmm_1d" / "effective_potential_depth_gmm_state_summary.csv"
GVM_MAIN_CSV = CURRENT_DIR / "figure" / "per_point_v_dot_u" / "gvm_moments_and_potential_summary.csv"
GVM_GMM_CSV = CURRENT_DIR / "figure" / "gmm_1d" / "gmm_run_tumble_gvm_kl_summary.csv"
GMM_XI_CSV = CURRENT_DIR / "figure" / "gmm_1d" / "gmm_xi_and_rc_over_xi_summary_k2.csv"

OUT_DIRS = [CURRENT_DIR / "figure" / "transfer_efficiency_synthesis"]
if ROOT_DIR.exists() and ROOT_DIR != CURRENT_DIR:
    OUT_DIRS.append(ROOT_DIR / "figure" / "transfer_efficiency_synthesis")

ORDER_BEADS = ["beads06um", "beads1um", "beads3um", "beads7um", "beads5um", "beads20um"]
BEAD_META = {b["name"]: b for b in vcd.BEADS_INFO}

X_LOCK = 45.0 * np.pi / 512.0   # ~0.276
X_BULK = 45.0 * np.pi / 128.0   # ~1.104
N_BOOT = 1000


def format_diam_label(d: float) -> str:
    if d in (0.63, 1.18, 3.37, 7.24):
        return f"{d:.2f}" + r" $\mu\mathrm{m}$"
    if d == 5.0:
        return r"5.0 $\mu\mathrm{m}$"
    return f"{d:.0f}" + r" $\mu\mathrm{m}$"


def per_particle_suff_stats(sub: pd.DataFrame) -> np.ndarray:
    """
    粒子ごとの十分統計量配列 shape=(n_particles, 8):
      0: sum(v_c . u_bar)
      1: sum(|u_bar|^2)
      2: sum(v_parallel)
      3: sum(|u_bar|)
      4: sum(|v_c|^2)
      5: sum(|v_c|)
      6: sum(cos_bulk)
      7: n_frames
    """
    sub = sub.copy()
    sub["cargo_id"] = sub["exp_dir"].astype(str) + "_" + sub["particle"].astype(str)
    rows = []
    for _, g in sub.groupby("cargo_id"):
        u = g["flow_u"].to_numpy(float)
        v = g["flow_v"].to_numpy(float)
        vx = g["cargo_dx"].to_numpy(float)
        vy = g["cargo_dy"].to_numpy(float)
        uspd = np.hypot(u, v)
        vspd = np.hypot(vx, vy)
        dot = vx * u + vy * v
        v_par = dot / np.where(uspd > 0, uspd, np.nan)
        cb = g["cos_bulk"].to_numpy(float)
        valid = np.isfinite(dot) & np.isfinite(v_par) & np.isfinite(cb) & (uspd > 0) & (vspd > 0)
        if not np.any(valid):
            continue
        rows.append([
            float(np.sum(dot[valid])),
            float(np.sum(uspd[valid] ** 2)),
            float(np.sum(v_par[valid])),
            float(np.sum(uspd[valid])),
            float(np.sum(vspd[valid] ** 2)),
            float(np.sum(vspd[valid])),
            float(np.sum(cb[valid])),
            float(np.sum(valid)),
        ])
    return np.asarray(rows, dtype=float)


def eval_from_suff(tot: np.ndarray) -> Dict[str, np.ndarray]:
    """tot shape (..., 8) -> dict of metric arrays"""
    dot = tot[..., 0]
    u2 = tot[..., 1]
    vpar = tot[..., 2]
    u1 = tot[..., 3]
    v2 = tot[..., 4]
    v1 = tot[..., 5]
    cb = tot[..., 6]
    nf = tot[..., 7]

    eta_reg = dot / np.where(u2 > 0, u2, np.nan)
    eta_par = vpar / np.where(u1 > 0, u1, np.nan)
    eta_T = dot / np.where((v2 > 0) & (u2 > 0), np.sqrt(v2 * u2), np.nan)
    spd_ratio = v1 / np.where(u1 > 0, u1, np.nan)
    cos_bulk = cb / np.where(nf > 0, nf, np.nan)
    return {
        "eta_reg": eta_reg,
        "eta_par": eta_par,
        "eta_T": eta_T,
        "spd_ratio": spd_ratio,
        "cos_bulk": cos_bulk,
    }


def summarize_with_bootstrap(sub: pd.DataFrame, seed: int) -> Dict[str, float]:
    mat = per_particle_suff_stats(sub)
    n_p = len(mat)
    pt = eval_from_suff(mat.sum(axis=0))
    out = {
        "n_particles": int(n_p),
        "n_frames": int(mat[:, 7].sum()) if n_p > 0 else 0,
    }
    for k, val in pt.items():
        out[k] = float(val)

    rng = np.random.default_rng(seed)
    if n_p >= 2:
        w = rng.multinomial(n_p, np.full(n_p, 1.0 / n_p), size=N_BOOT)
        boot = eval_from_suff(w @ mat)
        for k, arr in boot.items():
            valid = arr[np.isfinite(arr)]
            if len(valid) >= 10:
                out[f"{k}_ci_lo"] = float(np.percentile(valid, 2.5))
                out[f"{k}_ci_hi"] = float(np.percentile(valid, 97.5))
            else:
                out[f"{k}_ci_lo"] = np.nan
                out[f"{k}_ci_hi"] = np.nan
    else:
        for k in pt:
            out[f"{k}_ci_lo"] = np.nan
            out[f"{k}_ci_hi"] = np.nan
    return out


def save_fig_multi(fig: plt.Figure, stem: str) -> None:
    for d in OUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / f"{stem}.png", dpi=300, bbox_inches="tight")
        fig.savefig(d / f"{stem}.svg", bbox_inches="tight")


def save_csv_multi(df: pd.DataFrame, fname: str) -> None:
    for d in OUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        df.to_csv(d / fname, index=False)


def main():
    vcd.setup_matplotlib()
    plt.rcParams.update({
        "font.size": 10.5,
        "axes.labelsize": 11.5,
        "axes.titlesize": 12.0,
        "legend.fontsize": 8.4,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    df_inst = pd.read_pickle(INST_CACHE)
    df_pot_m = pd.read_csv(POT_MAIN_CSV).set_index("bead_name")
    df_gvm_m = pd.read_csv(GVM_MAIN_CSV).set_index("bead_name")
    df_pot_g = pd.read_csv(POT_GMM_CSV)
    df_gvm_g = pd.read_csv(GVM_GMM_CSV)
    df_xi_g = pd.read_csv(GMM_XI_CSV)
    xi_w = df_xi_g[df_xi_g["source"] == "angular_correlation_w"].set_index(["bead_name", "mode"])

    # 1. Unsplit 6 sizes
    rows_u = []
    for idx_b, bname in enumerate(ORDER_BEADS):
        sub = df_inst[df_inst["bead_name"] == bname]
        stats = summarize_with_bootstrap(sub, seed=100 + idx_b)
        rp = df_pot_m.loc[bname]
        rg = df_gvm_m.loc[bname]
        rows_u.append({
            "bead_name": bname,
            "diameter_um": BEAD_META[bname]["diameter_um"],
            "rc_over_xi": float(rp["rc_over_xi"]),
            "rc_over_xi_sem": float(rp["rc_over_xi_sem"]),
            "m1_cos_theta_pt": float(rg["m1_cos_theta"]),
            "du_emp_bar": float(rp["du_emp_bar"]),
            "du_emp_bar_ci_lo": float(rp["du_emp_bar_ci_lo"]),
            "du_emp_bar_ci_hi": float(rp["du_emp_bar_ci_hi"]),
            "d_kl_total": float(rg["D_KL_total"]),
            **stats,
        })
    df_u = pd.DataFrame(rows_u)

    # 2. GMM Run/Tumble + Large cargos
    df_gmm_frames = build_gmm_labeled_frames()
    rows_g = []
    for idx_g, (bname, st, diam, lbl) in enumerate([
        ("beads06um", "Run", 0.63, r"0.63 $\mu\mathrm{m}$ Run"),
        ("beads06um", "Tumble", 0.63, r"0.63 $\mu\mathrm{m}$ Tumble"),
        ("beads1um", "Run", 1.18, r"1.18 $\mu\mathrm{m}$ Run"),
        ("beads1um", "Tumble", 1.18, r"1.18 $\mu\mathrm{m}$ Tumble"),
    ]):
        sub = df_gmm_frames[(df_gmm_frames["bead_name"] == bname) & (df_gmm_frames["state_label"] == st)]
        stats = summarize_with_bootstrap(sub, seed=300 + idx_g)
        xr = xi_w.loc[(bname, st)]
        pr = df_pot_g[(df_pot_g["bead_name"] == bname) & (df_pot_g["state"] == st)].iloc[0]
        gr = df_gvm_g[(df_gvm_g["bead_name"] == bname) & (df_gvm_g["state_label"] == st)].iloc[0]
        rows_g.append({
            "bead_name": bname,
            "state": st,
            "diameter_um": diam,
            "label": lbl,
            "rc_over_xi": float(xr["rc_over_xi"]),
            "rc_xi_ci_lo": float(xr["rc_xi_ci_lo"]),
            "rc_xi_ci_hi": float(xr["rc_xi_ci_hi"]),
            "m1_cos_theta_pt": float(gr["m1_cos_theta"]),
            "du_emp_bar": float(pr["du_emp_bar"]),
            "du_emp_bar_ci_lo": float(pr["du_emp_bar_ci_lo"]),
            "du_emp_bar_ci_hi": float(pr["du_emp_bar_ci_hi"]),
            "d_kl_total": float(gr["D_KL_total"]),
            **stats,
        })

    for _, ru in df_u[df_u["bead_name"].isin(["beads3um", "beads7um", "beads5um", "beads20um"])].iterrows():
        row = {
            "bead_name": ru["bead_name"],
            "state": "All",
            "diameter_um": ru["diameter_um"],
            "label": format_diam_label(ru["diameter_um"]),
            "rc_over_xi": ru["rc_over_xi"],
            "rc_xi_ci_lo": ru["rc_over_xi"] - ru["rc_over_xi_sem"],
            "rc_xi_ci_hi": ru["rc_over_xi"] + ru["rc_over_xi_sem"],
        }
        for col in [
            "m1_cos_theta_pt", "du_emp_bar", "du_emp_bar_ci_lo", "du_emp_bar_ci_hi", "d_kl_total",
            "n_particles", "n_frames", "eta_reg", "eta_reg_ci_lo", "eta_reg_ci_hi",
            "eta_par", "eta_par_ci_lo", "eta_par_ci_hi",
            "eta_T", "eta_T_ci_lo", "eta_T_ci_hi",
            "spd_ratio", "spd_ratio_ci_lo", "spd_ratio_ci_hi",
            "cos_bulk", "cos_bulk_ci_lo", "cos_bulk_ci_hi",
        ]:
            row[col] = ru[col]
        rows_g.append(row)
    df_g = pd.DataFrame(rows_g)

    save_csv_multi(df_u, "transfer_efficiency_unsplit_summary.csv")
    save_csv_multi(df_g, "transfer_efficiency_gmm_state_summary.csv")

    # ------------------------------------------------------------------
    # Figure 1: Unsplit 6 sizes - Transfer Efficiencies & Direction vs Speed Decomposition
    # ------------------------------------------------------------------
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 5.1))
    fig1.subplots_adjust(top=0.84, wspace=0.25)

    x = df_u["rc_over_xi"].values
    for key, lbl, col, fmt in [
        ("eta_reg", r"$\eta_{\mathrm{reg}} = \sum(\mathbf{v}_c\cdot\bar{\mathbf{u}}_i)/\sum|\bar{\mathbf{u}}_i|^2$", "#1b4f72", "-o"),
        ("eta_par", r"$\eta_\parallel = \sum v_\parallel / \sum |\bar{\mathbf{u}}_i|$", "#2e86c1", "--s"),
        ("eta_T", r"$\eta_T = \langle \mathbf{v}_c\cdot\bar{\mathbf{u}}_i\rangle / (v_{\mathrm{rms}} u_{\mathrm{rms}})$", "#d35400", ":D"),
    ]:
        y = df_u[key].values
        yerr = np.vstack([np.maximum(0.0, y - df_u[f"{key}_ci_lo"].values),
                          np.maximum(0.0, df_u[f"{key}_ci_hi"].values - y)])
        ax1.errorbar(x, y, yerr=yerr, fmt=fmt, color=col, lw=1.9, ms=6.5, capsize=3.5, label=lbl, zorder=4)

    offsets_u = {
        "beads06um": (28, 6),
        "beads1um": (-18, -22),
        "beads3um": (26, 6),
        "beads7um": (-26, -16),
        "beads5um": (-24, -16),
        "beads20um": (-18, 22),
    }
    for _, r in df_u.iterrows():
        dx, dy = offsets_u[r["bead_name"]]
        ax1.annotate(
            format_diam_label(r["diameter_um"]),
            xy=(r["rc_over_xi"], r["eta_reg"]),
            xytext=(dx, dy), textcoords="offset points",
            fontsize=8.3, color="#1b4f72", ha="center",
        )

    ax1.axvline(X_LOCK, color="0.45", ls="--", lw=1.1, zorder=1)
    ax1.axvline(X_BULK, color="0.45", ls=":", lw=1.1, zorder=1)
    ax1.axhline(0.0, color="0.25", lw=0.7)
    ax1.text(X_LOCK * 1.05, 0.72, r"$x_{\mathrm{lock}} \approx 0.28$", fontsize=8.4, color="0.35")
    ax1.text(X_BULK * 1.05, 0.72, r"$x_{\mathrm{bulk}} \approx 1.10$", fontsize=8.4, color="0.35")
    ax1.set_xscale("log")
    ax1.set_xlim(0.045, 3.0)
    ax1.set_ylim(-0.05, 0.96)
    ax1.set_xlabel(r"$R_c / \xi$")
    ax1.set_ylabel(r"Velocity / Power Transfer Efficiency $\eta$")
    ax1.set_title("A. Cargo Transfer Efficiency across Definitions")
    ax1.legend(loc="upper right", frameon=True, framealpha=0.94, fontsize=8.1)
    ax1.grid(alpha=0.3, which="both")

    # Panel B: Decomposition - Scalar Speed Ratio vs Directional Alignment
    for key, lbl, col, fmt in [
        ("spd_ratio", r"Speed ratio $\langle |\mathbf{v}_c| \rangle / \langle |\bar{\mathbf{u}}_i| \rangle$ ($0.85 \to 0.53$)", "#117733", "-^"),
        ("cos_bulk", r"Bulk direction $\langle \cos\Theta_{\mathrm{bulk}} \rangle$ ($0.35 \to 0.04$)", "#882255", "-o"),
    ]:
        y = df_u[key].values
        yerr = np.vstack([np.maximum(0.0, y - df_u[f"{key}_ci_lo"].values),
                          np.maximum(0.0, df_u[f"{key}_ci_hi"].values - y)])
        ax2.errorbar(x, y, yerr=yerr, fmt=fmt, color=col, lw=1.9, ms=6.8, capsize=3.5, label=lbl, zorder=4)

    ax2.plot(x, df_u["m1_cos_theta_pt"].values, "--s", color="#cc6677", lw=1.6, ms=6.0,
             label=r"Per-point align $\langle \cos\theta_{\mathrm{pt}} \rangle$ ($0.35 \to 0.03$)", zorder=3)
    ax2.axvline(X_LOCK, color="0.45", ls="--", lw=1.1, zorder=1)
    ax2.axvline(X_BULK, color="0.45", ls=":", lw=1.1, zorder=1)
    ax2.axhline(0.0, color="0.25", lw=0.7)
    ax2.text(X_LOCK * 1.05, 0.48, r"$x_{\mathrm{lock}}$", fontsize=8.5, color="0.35")
    ax2.text(X_BULK * 1.05, 0.48, r"$x_{\mathrm{bulk}}$", fontsize=8.5, color="0.35")
    ax2.set_xscale("log")
    ax2.set_xlim(0.045, 3.0)
    ax2.set_ylim(-0.05, 1.08)
    ax2.set_xlabel(r"$R_c / \xi$")
    ax2.set_ylabel("Normalized Ratio / Directional Cosine")
    ax2.set_title("B. Why Efficiency Drops: Speed Retention vs. Direction Loss")
    ax2.legend(loc="upper right", frameon=True, framealpha=0.94, fontsize=8.1)
    ax2.grid(alpha=0.3, which="both")

    fig1.suptitle(r"Cargo-Flow Velocity Transfer Efficiency $\eta$ and Factor Decomposition (95% Cluster CI)", fontsize=13.2, y=0.96)
    save_fig_multi(fig1, "transfer_efficiency_decomposition_vs_rc_xi")
    plt.close(fig1)

    # ------------------------------------------------------------------
    # Figure 2: GMM State-Resolved eta & Coupling with Potential Depth Delta U
    # ------------------------------------------------------------------
    fig2, (axg1, axg2) = plt.subplots(1, 2, figsize=(13.0, 5.2))
    fig2.subplots_adjust(top=0.84, wspace=0.25)

    # Unsplit 0.63 & 1.18 faint open circles
    for bname in ("beads06um", "beads1um"):
        ru = df_u[df_u["bead_name"] == bname].iloc[0]
        axg1.scatter(ru["rc_over_xi"], ru["eta_reg"], s=65, facecolors="none", edgecolors="0.55", lw=1.4, zorder=2,
                     label=r"Unsplit 0.63 & 1.18 $\mu\mathrm{m}$ ($\eta_{\mathrm{reg}}$)" if bname == "beads06um" else None)

    df_large = df_g[df_g["state"] == "All"]
    xl = df_large["rc_over_xi"].values
    yl = df_large["eta_reg"].values
    xl_err = np.vstack([np.maximum(0.0, xl - df_large["rc_xi_ci_lo"].values),
                        np.maximum(0.0, df_large["rc_xi_ci_hi"].values - xl)])
    yl_err = np.vstack([np.maximum(0.0, yl - df_large["eta_reg_ci_lo"].values),
                        np.maximum(0.0, df_large["eta_reg_ci_hi"].values - yl)])
    axg1.errorbar(xl, yl, xerr=xl_err, yerr=yl_err, fmt="-o", color="#2c3e50", lw=1.7, ms=7.0, capsize=3.0,
                  label=r"Large cargos ($\geq 3.37\,\mu\mathrm{m}$)", zorder=4)
    large_off = {"beads3um": (26, 6), "beads7um": (26, 4), "beads5um": (20, 8), "beads20um": (-16, 16)}
    for _, rl in df_large.iterrows():
        dx, dy = large_off[rl["bead_name"]]
        axg1.annotate(format_diam_label(rl["diameter_um"]), xy=(rl["rc_over_xi"], rl["eta_reg"]),
                      xytext=(dx, dy), textcoords="offset points", fontsize=8.2, color="#2c3e50", ha="center")

    gmm_styles = {
        ("beads06um", "Run"): ("#1f77b4", "^", r"0.63 $\mu\mathrm{m}$ Run", (12, -16)),
        ("beads06um", "Tumble"): ("#5dade2", "v", r"0.63 $\mu\mathrm{m}$ Tumble", (16, 8)),
        ("beads1um", "Run"): ("#d35400", "^", r"1.18 $\mu\mathrm{m}$ Run", (14, -16)),
        ("beads1um", "Tumble"): ("#f39c12", "v", r"1.18 $\mu\mathrm{m}$ Tumble", (-20, 14)),
    }
    for _, rg in df_g[df_g["state"].isin(["Run", "Tumble"])].iterrows():
        c, m, lbl, (dx, dy) = gmm_styles[(rg["bead_name"], rg["state"])]
        xv, yv = rg["rc_over_xi"], rg["eta_reg"]
        xerr = [[max(0.0, xv - rg["rc_xi_ci_lo"])], [max(0.0, rg["rc_xi_ci_hi"] - xv)]]
        yerr = [[max(0.0, yv - rg["eta_reg_ci_lo"])], [max(0.0, rg["eta_reg_ci_hi"] - yv)]]
        axg1.errorbar([xv], [yv], xerr=xerr, yerr=yerr, fmt=m, color=c, ms=9.5, capsize=3.5, lw=1.6,
                      markeredgecolor="black", markeredgewidth=0.6, label=lbl, zorder=6)
        axg1.scatter([xv], [rg["eta_par"]], s=48, facecolors="none", edgecolors=c, marker="s", lw=1.5, zorder=5)
        axg1.annotate(f"{yv:.2f}", xy=(xv, yv), xytext=(dx, dy), textcoords="offset points",
                      fontsize=8.4, fontweight="bold", color=c)

    axg1.scatter([], [], s=45, facecolors="none", edgecolors="0.3", marker="s", lw=1.4,
                 label=r"$\eta_\parallel$ (open square)")

    for bname, col in [("beads06um", "#1f77b4"), ("beads1um", "#d35400")]:
        ru = df_u[df_u["bead_name"] == bname].iloc[0]
        for _, rg in df_g[(df_g["bead_name"] == bname) & (df_g["state"].isin(["Run", "Tumble"]))].iterrows():
            axg1.plot([ru["rc_over_xi"], rg["rc_over_xi"]], [ru["eta_reg"], rg["eta_reg"]],
                      ls=":", lw=1.0, color=col, alpha=0.55, zorder=2)

    axg1.axvline(X_LOCK, color="0.45", ls="--", lw=1.1, zorder=1)
    axg1.axvline(X_BULK, color="0.45", ls=":", lw=1.1, zorder=1)
    axg1.axhline(0.0, color="0.25", lw=0.7)
    axg1.text(X_LOCK * 0.94, 0.72, r"$x_{\mathrm{lock}}$", fontsize=8.4, color="0.35", ha="right")
    axg1.text(X_BULK * 1.05, 0.45, r"$x_{\mathrm{bulk}}$", fontsize=8.4, color="0.35", ha="left")
    axg1.set_xscale("log")
    axg1.set_xlim(0.022, 3.2)
    axg1.set_ylim(-0.08, 1.05)
    axg1.set_xlabel(r"State-Resolved $R_c / \xi_{\mathrm{state}}$")
    axg1.set_ylabel(r"Velocity Transfer Efficiency $\eta_{\mathrm{reg}}$ ($\eta_\parallel$ open)")
    axg1.set_title("A. GMM Run vs. Tumble Transfer Efficiency")
    axg1.legend(loc="upper right", frameon=True, framealpha=0.94, fontsize=7.8)
    axg1.grid(alpha=0.3, which="both")

    # Panel B: eta_reg vs Effective Potential Barrier Delta U_emp,bar (plus unsplit & GMM)
    for _, ru in df_u.iterrows():
        c = BEAD_META[ru["bead_name"]]["color"]
        m = BEAD_META[ru["bead_name"]]["marker"]
        axg2.errorbar(
            [ru["du_emp_bar"]], [ru["eta_reg"]],
            xerr=[[max(0.0, ru["du_emp_bar"] - ru["du_emp_bar_ci_lo"])], [max(0.0, ru["du_emp_bar_ci_hi"] - ru["du_emp_bar"])]],
            yerr=[[max(0.0, ru["eta_reg"] - ru["eta_reg_ci_lo"])], [max(0.0, ru["eta_reg_ci_hi"] - ru["eta_reg"])]],
            fmt=m, color=c, ms=8.5, capsize=3.0, lw=1.5, markeredgecolor="black", markeredgewidth=0.6,
            label=format_diam_label(ru["diameter_um"]) + " (all)", zorder=4,
        )

    for _, rg in df_g[df_g["state"].isin(["Run", "Tumble"])].iterrows():
        c, m, lbl, _ = gmm_styles[(rg["bead_name"], rg["state"])]
        axg2.errorbar(
            [rg["du_emp_bar"]], [rg["eta_reg"]],
            xerr=[[max(0.0, rg["du_emp_bar"] - rg["du_emp_bar_ci_lo"])], [max(0.0, rg["du_emp_bar_ci_hi"] - rg["du_emp_bar"])]],
            yerr=[[max(0.0, rg["eta_reg"] - rg["eta_reg_ci_lo"])], [max(0.0, rg["eta_reg_ci_hi"] - rg["eta_reg"])]],
            fmt=m, color=c, ms=9.2, capsize=3.0, lw=1.5, markeredgecolor="black", markeredgewidth=0.6,
            label=lbl, zorder=5,
        )

    all_du = np.concatenate([df_u["du_emp_bar"].values, df_g[df_g["state"].isin(["Run", "Tumble"])]["du_emp_bar"].values])
    all_eta = np.concatenate([df_u["eta_reg"].values, df_g[df_g["state"].isin(["Run", "Tumble"])]["eta_reg"].values])
    rho_du, p_du = spearmanr(all_du, all_eta)

    axg2.axvline(1.0, color="0.4", ls="-.", lw=1.2, zorder=1)
    axg2.text(1.05, 0.85, r"Noise threshold $1\,k_B T_{\mathrm{eff}}$", fontsize=8.4, color="0.35")
    axg2.axhline(0.0, color="0.25", lw=0.7)
    axg2.text(0.15, 0.78, f"Spearman $\\rho = {rho_du:.3f}$ ($p = {p_du:.4f}$)",
              fontsize=9.0, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.9))
    axg2.set_xlim(-0.05, 4.95)
    axg2.set_ylim(-0.08, 0.98)
    axg2.set_xlabel(r"Effective Potential Barrier $\Delta\tilde{U}_{\mathrm{emp,bar}}$ [$k_B T_{\mathrm{eff}}$]")
    axg2.set_ylabel(r"Velocity Transfer Efficiency $\eta_{\mathrm{reg}}$")
    axg2.set_title("B. Transfer Efficiency Controlled by Potential Depth")
    axg2.legend(loc="lower right", frameon=True, framealpha=0.94, fontsize=7.5, ncol=2)
    axg2.grid(alpha=0.3)

    fig2.suptitle("State-Resolved Transfer Efficiency & Link to Effective Alignment Potential", fontsize=13.2, y=0.96)
    save_fig_multi(fig2, "transfer_efficiency_vs_potential_and_gmm_states")
    plt.close(fig2)

    print("\n=== Unsplit 6 Cargo Sizes: Transfer Efficiency Summary ===")
    cols_u = [
        "bead_name", "diameter_um", "rc_over_xi", "n_particles", "n_frames",
        "eta_reg", "eta_reg_ci_lo", "eta_reg_ci_hi",
        "eta_par", "eta_par_ci_lo", "eta_par_ci_hi",
        "eta_T", "spd_ratio", "cos_bulk", "m1_cos_theta_pt", "du_emp_bar", "d_kl_total",
    ]
    print(df_u[cols_u].to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print("\n=== GMM State-Resolved + Large Cargos: Transfer Efficiency Summary ===")
    cols_g = [
        "bead_name", "state", "diameter_um", "rc_over_xi",
        "eta_reg", "eta_reg_ci_lo", "eta_reg_ci_hi",
        "eta_par", "eta_T", "spd_ratio", "cos_bulk", "du_emp_bar", "d_kl_total",
    ]
    print(df_g[cols_g].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nSpearman(Delta U_emp_bar, eta_reg) across all 10 conditions: rho={rho_du:.3f}, p={p_du:.3e}")


if __name__ == "__main__":
    main()
