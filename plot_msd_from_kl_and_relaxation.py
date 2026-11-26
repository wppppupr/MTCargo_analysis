#!/usr/bin/env python3
"""
plot_msd_from_kl_and_relaxation.py

1 ステップ配向 KL ダイバージェンス D_KL およびラグランジュ緩和時間
(tau_v^slow, tau_u^Lag) から巨視的 MSD(Delta t) を離散 Ornstein-Uhlenbeck /
Green-Kubo 理論式で再構成し、実測 MSD と比較可視化するスクリプト。

モデル:
  1. 0-param (KL + tau_v^slow):
       MSD(M dt) = sigma_1^2 [ M + 2 * sqrt(D_KL) * sum_{k=1}^{M-1} (M - k) exp(-k dt / tau_v^slow) ]
  2. 1-const (KL + tau_u^Lag, 貨物 VACF 未使用):
       A = min(1.0, c * D_KL), tau_p = tau_u^Lag (足元流れのラグランジュ緩和時間)
  3. OU baseline: A = max(0, C_v(4s)), tau_p = tau_v^slow
  4. Uncorrelated Brownian baseline (D_KL = 0): MSD_0(M dt) = M * sigma_1^2

出力先:
  - figure/msd/msd_reconstructed_from_kl_and_tau.{png,svg}
  - figure/msd/msd_kl_relaxation_synthesis.{png,svg}
  - figure/msd/msd_kl_relaxation_reconstruction_summary.csv
  - (上記を /mnt/NAS-Ebanaru/sasaki/MTsingleBeads/figure/msd/ にも保存)
"""

import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from libs.experiment_cutoff import is_excluded_experiment, is_excluded_particle

ROOT_DIR = Path("/mnt/NAS-Ebanaru/sasaki/MTsingleBeads")
INST_CACHE = CURRENT_DIR / "scratch" / "contrast_inst_cache.pkl"
DPI_CSV = CURRENT_DIR / "figure" / "per_point_v_dot_u" / "data_processing_inequality_summary.csv"

OUT_DIRS = [CURRENT_DIR / "figure" / "msd"]
if ROOT_DIR.exists() and ROOT_DIR != CURRENT_DIR:
    OUT_DIRS.append(ROOT_DIR / "figure" / "msd")

ORDER_BEADS = ["beads06um", "beads1um", "beads3um", "beads7um", "beads5um", "beads20um"]
BEAD_META = {
    "beads06um": {"diameter_um": 0.63, "marker": "^", "color": "#882255"},
    "beads1um":  {"diameter_um": 1.18, "marker": "o", "color": "#CC6677"},
    "beads3um":  {"diameter_um": 3.37, "marker": "d", "color": "#DDCC77"},
    "beads5um":  {"diameter_um": 5.00, "marker": "p", "color": "#999933"},
    "beads7um":  {"diameter_um": 7.24, "marker": "h", "color": "#117733"},
    "beads20um": {"diameter_um": 20.0, "marker": "s", "color": "#44AA99"},
}


def setup_matplotlib():
    style_path = CURRENT_DIR / "libs" / "my_style.mplstyle"
    if style_path.exists():
        plt.style.use(str(style_path))
    plt.rcParams.update({
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.family": "sans-serif",
        "mathtext.fontset": "dejavusans",
    })

SCALE_UM_PER_PX = 0.11
DT = 4.0
M_MAX = 75       # up to 300 s
INT_MAX_K = 60   # up to 240 s for integral relaxation times
N_BOOT = 500
MIN_PAIRS = 20


def fmt_diam(d: float) -> str:
    val = float(d)
    if abs(val - 5.0) < 0.05:
        return "5.0"
    if abs(val - 20.0) < 0.5:
        return "20"
    return f"{val:.2f}"


def compute_per_cargo_pair_arrays(groups: List[pd.DataFrame], max_lag: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return per-cargo pair sums S_v, N_v, S_u, N_u of shape (n_cargos, max_lag + 1)."""
    n_g = len(groups)
    S_v = np.zeros((n_g, max_lag + 1), dtype=float)
    N_v = np.zeros((n_g, max_lag + 1), dtype=float)
    S_u = np.zeros((n_g, max_lag + 1), dtype=float)
    N_u = np.zeros((n_g, max_lag + 1), dtype=float)

    for idx_g, g in enumerate(groups):
        g = g.sort_values("frame")
        f = g["frame"].values.astype(int)
        f0 = int(f.min())
        L = int(f.max() - f0 + 1)
        vx = np.full(L, np.nan)
        vy = np.full(L, np.nan)
        ux = np.full(L, np.nan)
        uy = np.full(L, np.nan)
        vx[f - f0] = g["cargo_dx"].values.astype(float) * SCALE_UM_PER_PX
        vy[f - f0] = g["cargo_dy"].values.astype(float) * SCALE_UM_PER_PX
        ux[f - f0] = g["flow_u"].values.astype(float) * SCALE_UM_PER_PX
        uy[f - f0] = g["flow_v"].values.astype(float) * SCALE_UM_PER_PX

        ok_v = np.isfinite(vx) & np.isfinite(vy)
        ok_u = np.isfinite(ux) & np.isfinite(uy)
        vx0 = np.where(ok_v, vx, 0.0)
        vy0 = np.where(ok_v, vy, 0.0)
        ux0 = np.where(ok_u, ux, 0.0)
        uy0 = np.where(ok_u, uy, 0.0)

        for k in range(min(max_lag, L - 1) + 1):
            mv = ok_v[: L - k] & ok_v[k:]
            cv = int(mv.sum())
            if cv > 0:
                S_v[idx_g, k] = float(np.sum((vx0[: L - k] * vx0[k:] + vy0[: L - k] * vy0[k:])[mv]))
                N_v[idx_g, k] = cv
            mu = ok_u[: L - k] & ok_u[k:]
            cu = int(mu.sum())
            if cu > 0:
                S_u[idx_g, k] = float(np.sum((ux0[: L - k] * ux0[k:] + uy0[: L - k] * uy0[k:])[mu]))
                N_u[idx_g, k] = cu

    return S_v, N_v, S_u, N_u


def eval_gk_msd(S_v: np.ndarray, N_v: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute G_v, C_v, and discrete Green-Kubo MSD(M*DT) for M=1..M_MAX."""
    s_sum = S_v.sum(axis=0)
    n_sum = N_v.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        Gv = s_sum / n_sum
    Gv[n_sum < MIN_PAIRS] = np.nan
    Cv = Gv / Gv[0] if (np.isfinite(Gv[0]) and Gv[0] > 0) else np.full_like(Gv, np.nan)

    msd = np.full(M_MAX, np.nan, dtype=float)
    gv_clean = np.where(np.isfinite(Gv), Gv, 0.0)
    for M in range(1, M_MAX + 1):
        ks = np.arange(1, M)
        msd[M - 1] = M * gv_clean[0] + 2.0 * float(np.sum((M - ks) * gv_clean[ks])) if M > 1 else gv_clean[0]
    return Gv, Cv, msd


def integral_tau(C_arr: np.ndarray, start_k: int, max_k: int) -> float:
    acc = 0.0
    hi = min(max_k, len(C_arr) - 1)
    for k in range(start_k, hi):
        c1 = C_arr[k]
        c2 = C_arr[k + 1]
        if not np.isfinite(c1) or c1 <= 0.0:
            break
        acc += 0.5 * (max(0.0, c1) + max(0.0, c2 if np.isfinite(c2) else 0.0)) * DT
    return float(acc)


def discrete_ou_msd(sigma1_sq: float, amp: float, tau_p: float, m_arr: np.ndarray) -> np.ndarray:
    out = np.zeros(len(m_arr), dtype=float)
    for i, M in enumerate(m_arr):
        if M <= 1 or not np.isfinite(tau_p) or tau_p <= 0.0 or amp <= 0.0:
            out[i] = float(M) * sigma1_sq
        else:
            ks = np.arange(1, M)
            tail = 2.0 * amp * float(np.sum((M - ks) * np.exp(-ks * DT / tau_p)))
            out[i] = sigma1_sq * (float(M) + tail)
    return out


def r2_log10(y_obs: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    mask = np.isfinite(y_obs) & np.isfinite(y_pred) & (y_obs > 0) & (y_pred > 0)
    lo = np.log10(y_obs[mask])
    lp = np.log10(y_pred[mask])
    denom = float(np.sum((lo - lo.mean()) ** 2))
    r2 = 1.0 - float(np.sum((lo - lp) ** 2)) / denom if denom > 0 else np.nan
    rmse = float(np.sqrt(np.mean((lo - lp) ** 2)))
    return float(r2), float(rmse)


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
    setup_matplotlib()
    plt.rcParams.update({
        "figure.constrained_layout.use": False,
        "font.size": 10.0,
        "axes.labelsize": 11.5,
        "axes.titlesize": 11.8,
        "legend.fontsize": 8.4,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    df_inst = pd.read_pickle(INST_CACHE)
    df_inst = df_inst[~df_inst["exp_dir"].astype(str).apply(is_excluded_experiment)].copy()
    df_inst = df_inst[~df_inst.apply(lambda r: is_excluded_particle(str(r["exp_dir"]), r["particle"]), axis=1)].copy()
    df_inst["pid_key"] = df_inst["exp_dir"].astype(str) + "_" + df_inst["particle"].astype(str)

    df_dpi = pd.read_csv(DPI_CSV).set_index("bead_name")
    m_arr = np.arange(1, M_MAX + 1)
    lags_s = m_arr * DT
    rng = np.random.default_rng(42)

    data: Dict[str, dict] = {}
    for b in ORDER_BEADS:
        sub = df_inst[df_inst["bead_name"] == b]
        groups = [g for _, g in sub.groupby("pid_key")]
        n_g = len(groups)
        S_v, N_v, S_u, N_u = compute_per_cargo_pair_arrays(groups, M_MAX)
        Gv, Cv, msd_obs = eval_gk_msd(S_v, N_v)

        u_sum = S_u.sum(axis=0)
        un_sum = N_u.sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            Gu = u_sum / un_sum
        Cu = Gu / Gu[0]

        # Particle-cluster bootstrap CI for observed MSD(t)
        boot_msd = np.full((N_BOOT, M_MAX), np.nan)
        for bi in range(N_BOOT):
            sel = rng.integers(0, n_g, size=n_g)
            _, _, bm = eval_gk_msd(S_v[sel], N_v[sel])
            boot_msd[bi] = bm
        msd_lo = np.nanpercentile(boot_msd, 2.5, axis=0)
        msd_hi = np.nanpercentile(boot_msd, 97.5, axis=0)

        tau_tail_int = integral_tau(Cv, start_k=1, max_k=INT_MAX_K)
        tau_v_slow = (tau_tail_int / float(Cv[1])) if (np.isfinite(Cv[1]) and Cv[1] > 0.05) else 0.0
        tau_u_lag = integral_tau(Cu, start_k=0, max_k=INT_MAX_K)

        # tau_GK from 240 s decomposition (M=60)
        msd0_240 = 60.0 * Gv[0]
        msd_tail_240 = msd_obs[59] - msd0_240
        tau_gk = (DT / 2.0) * (msd_tail_240 / msd0_240) if msd0_240 > 0 else np.nan
        tau_decoup = (
            1.0 / (1.0 / tau_v_slow - 1.0 / tau_u_lag)
            if (tau_v_slow > 0 and tau_u_lag > tau_v_slow)
            else np.nan
        )

        dkl = float(df_dpi.loc[b, "D_KL_obs"])
        dmt = float(df_dpi.loc[b, "D_MT_total"])
        eta_info = float(df_dpi.loc[b, "info_efficiency_D_KL_over_D_MT"])
        rc_xi = float(df_dpi.loc[b, "rc_over_xi"])
        diam = float(df_dpi.loc[b, "diameter_um"])

        data[b] = {
            "bead_name": b,
            "diameter_um": diam,
            "rc_over_xi": rc_xi,
            "n_cargos": n_g,
            "n_frames": int(len(sub)),
            "sigma1_sq": float(Gv[0]),
            "d_kl": dkl,
            "d_mt": dmt,
            "eta_info": eta_info,
            "cv_4s": float(Cv[1]),
            "tau_v_slow_s": float(tau_v_slow),
            "tau_u_lag_s": float(tau_u_lag),
            "tau_gk_s": float(tau_gk),
            "tau_decoup_s": float(tau_decoup),
            "msd_obs": msd_obs,
            "msd_ci_lo": msd_lo,
            "msd_ci_hi": msd_hi,
        }

    # Fit single universal scale c for KL + flow Lagrangian time model across all 6 sizes
    c_grid = np.linspace(1.0, 5.0, 401)
    best_c, best_sse = 3.0, 1e18
    for c_val in c_grid:
        sse = 0.0
        for b in ORDER_BEADS:
            r = data[b]
            amp = min(1.0, float(c_val) * r["d_kl"])
            pred = discrete_ou_msd(r["sigma1_sq"], amp, r["tau_u_lag_s"], m_arr)
            sse += float(np.sum((np.log10(r["msd_obs"]) - np.log10(pred)) ** 2))
        if sse < best_sse:
            best_sse, best_c = sse, float(c_val)

    # Evaluate all models & accumulate summary rows
    summary_rows = []
    all_obs, all_m_kl, all_m_kl_flow, all_m_ou = [], [], [], []
    for b in ORDER_BEADS:
        r = data[b]
        sig1 = r["sigma1_sq"]
        msd_kl = discrete_ou_msd(sig1, np.sqrt(max(0.0, r["d_kl"])), r["tau_v_slow_s"], m_arr)
        msd_kl_flow = discrete_ou_msd(sig1, min(1.0, best_c * r["d_kl"]), r["tau_u_lag_s"], m_arr)
        msd_ou = discrete_ou_msd(sig1, max(0.0, r["cv_4s"]), r["tau_v_slow_s"], m_arr)
        msd_brown = m_arr.astype(float) * sig1

        r["msd_kl"] = msd_kl
        r["msd_kl_flow"] = msd_kl_flow
        r["msd_ou"] = msd_ou
        r["msd_brown"] = msd_brown

        r2_kl, rmse_kl = r2_log10(r["msd_obs"], msd_kl)
        r2_kf, rmse_kf = r2_log10(r["msd_obs"], msd_kl_flow)
        r2_ou, rmse_ou = r2_log10(r["msd_obs"], msd_ou)
        r2_br, _ = r2_log10(r["msd_obs"], msd_brown)

        all_obs.append(r["msd_obs"])
        all_m_kl.append(msd_kl)
        all_m_kl_flow.append(msd_kl_flow)
        all_m_ou.append(msd_ou)

        summary_rows.append({
            "bead_name": b,
            "diameter_um": r["diameter_um"],
            "rc_over_xi": r["rc_over_xi"],
            "n_cargos": r["n_cargos"],
            "n_frames": r["n_frames"],
            "sigma1_sq_um2": sig1,
            "D_KL_obs": r["d_kl"],
            "sqrt_D_KL": float(np.sqrt(max(0.0, r["d_kl"]))),
            "eta_info_D_KL_over_D_MT": r["eta_info"],
            "Cv_4s": r["cv_4s"],
            "tau_v_slow_s": r["tau_v_slow_s"],
            "tau_u_lag_s": r["tau_u_lag_s"],
            "tau_gk_s": r["tau_gk_s"],
            "tau_decoup_s": r["tau_decoup_s"],
            "c_univ_kl_flow": best_c,
            "msd_obs_240s": float(r["msd_obs"][59]),
            "msd_kl_240s": float(msd_kl[59]),
            "msd_kl_flow_240s": float(msd_kl_flow[59]),
            "msd_ou_240s": float(msd_ou[59]),
            "msd_brown_240s": float(msd_brown[59]),
            "msd_obs_300s": float(r["msd_obs"][74]),
            "msd_kl_300s": float(msd_kl[74]),
            "msd_kl_flow_300s": float(msd_kl_flow[74]),
            "R2_log_kl_tau_v": r2_kl,
            "RMSE_dex_kl_tau_v": rmse_kl,
            "R2_log_kl_tau_u": r2_kf,
            "RMSE_dex_kl_tau_u": rmse_kf,
            "R2_log_ou_ref": r2_ou,
            "RMSE_dex_ou_ref": rmse_ou,
            "R2_log_brownian": r2_br,
        })

    glob_r2_kl, glob_rmse_kl = r2_log10(np.concatenate(all_obs), np.concatenate(all_m_kl))
    glob_r2_kf, glob_rmse_kf = r2_log10(np.concatenate(all_obs), np.concatenate(all_m_kl_flow))
    glob_r2_ou, glob_rmse_ou = r2_log10(np.concatenate(all_obs), np.concatenate(all_m_ou))

    df_sum = pd.DataFrame(summary_rows)
    save_csv_multi(df_sum, "msd_kl_relaxation_reconstruction_summary.csv")

    print(f"Universal constant c (KL + tau_u^Lag): {best_c:.3f}")
    print(f"GLOBAL R^2_log10 across all 6 sizes x 75 lags (N=450):")
    print(f"  KL + tau_v^slow (0-param)      : R^2 = {glob_r2_kl:.4f}, RMSE = {glob_rmse_kl:.4f} dex ({10**glob_rmse_kl:.2f}x)")
    print(f"  KL + tau_u^Lag (1 const c={best_c:.2f}): R^2 = {glob_r2_kf:.4f}, RMSE = {glob_rmse_kf:.4f} dex ({10**glob_rmse_kf:.2f}x)")
    print(f"  OU ref (Cv(4s) + tau_v^slow)   : R^2 = {glob_r2_ou:.4f}, RMSE = {glob_rmse_ou:.4f} dex ({10**glob_rmse_ou:.2f}x)")
    print("\n" + df_sum[[
        "bead_name", "n_cargos", "rc_over_xi", "D_KL_obs", "tau_v_slow_s", "tau_u_lag_s",
        "msd_obs_240s", "msd_kl_240s", "msd_kl_flow_240s", "R2_log_kl_tau_v", "R2_log_kl_tau_u"
    ]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # ======================================================================
    # Figure 1: 6-panel Comparison (Observed vs KL+tau_v vs KL+tau_u vs Brownian)
    # ======================================================================
    fig1, axes1 = plt.subplots(2, 3, figsize=(15.4, 9.8), sharex=True, sharey=True, layout=None)

    for idx_b, b in enumerate(ORDER_BEADS):
        ax = axes1.flat[idx_b]
        r = data[b]
        meta = BEAD_META[b]
        col = meta["color"]

        ax.fill_between(lags_s, r["msd_ci_lo"], r["msd_ci_hi"], color=col, alpha=0.18, lw=0)
        ax.loglog(lags_s, r["msd_obs"], "-", color=col, lw=2.5, label=r"Observed MSD (95% CI)", zorder=5)
        ax.loglog(lags_s, r["msd_kl"], "--", color="#111111", lw=2.1,
                  label=r"Theory: $D_{\mathrm{KL}} + \tau_{\mathbf{v}}^{\mathrm{slow}}$ (0-param)", zorder=6)
        ax.loglog(lags_s, r["msd_kl_flow"], "-.", color="#1f77b4", lw=1.9,
                  label=rf"Theory: $D_{{\mathrm{{KL}}}} + \tau_{{\bar{{\mathbf{{u}}}}}}^{{\mathrm{{Lag}}}}$ ($c={best_c:.2f}$)", zorder=4)
        ax.loglog(lags_s, r["msd_brown"], ":", color="0.45", lw=1.6,
                  label=r"Brownian ($D_{\mathrm{KL}} = 0$): $M\sigma_1^2$", zorder=2)

        r2_k = df_sum.loc[idx_b, "R2_log_kl_tau_v"]
        r2_f = df_sum.loc[idx_b, "R2_log_kl_tau_u"]
        tau_lbl = f"{r['tau_v_slow_s']:.1f} s" if r["tau_v_slow_s"] > 0 else "< 4 s"
        info_txt = (
            f"$x = {r['rc_over_xi']:.3f}$,  $D_{{\\mathrm{{KL}}}} = {r['d_kl']:.3f}$\n"
            f"$\\tau_{{\\mathbf{{v}}}}^{{\\mathrm{{slow}}}} = {tau_lbl}$,  "
            f"$\\tau_{{\\bar{{\\mathbf{{u}}}}}}^{{\\mathrm{{Lag}}}} = {r['tau_u_lag_s']:.0f}$ s\n"
            f"$R^2_{{\\log}}(\\tau_{{\\mathbf{{v}}}}) = {r2_k:.3f}$,  $R^2_{{\\log}}(\\tau_{{\\bar{{\\mathbf{{u}}}}}}) = {r2_f:.3f}$"
        )
        ax.text(
            0.04, 0.96, info_txt, transform=ax.transAxes, fontsize=8.1, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.32", fc="white", ec="0.72", alpha=0.92),
        )
        panel_lbl = chr(ord("A") + idx_b)
        ax.set_title(f"{panel_lbl}. {fmt_diam(r['diameter_um'])} $\\mu$m ($N = {r['n_cargos']}$)", fontsize=11.8, fontweight="bold")
        ax.set_xlim(4.0, 320.0)
        ax.set_ylim(0.08, 1200.0)
        ax.grid(True, which="both", linestyle=":", alpha=0.45)
        if idx_b >= 3:
            ax.set_xlabel(r"Lag time $\Delta t$ [s]")
        if idx_b % 3 == 0:
            ax.set_ylabel(r"$\mathrm{MSD}(\Delta t)$ [$\mu\mathrm{m}^2$]")
        if idx_b == 0:
            ax.legend(loc="lower right", fontsize=8.0, framealpha=0.94)

    fig1.suptitle(
        r"Reconstructing $\mathrm{MSD}(\Delta t)$ from 1-Step $D_{\mathrm{KL}}$ and Lagrangian Relaxation Times ($\Delta t = 4$–$300$ s)",
        fontsize=13.8, fontweight="bold", y=0.985
    )
    fig1.tight_layout(rect=[0, 0, 1, 0.945])
    save_fig_multi(fig1, "msd_reconstructed_from_kl_and_tau")
    plt.close(fig1)

    # ======================================================================
    # Figure 2: 3-Panel Synthesis (Overlay + Parity Plot + Timescale Crossover)
    # ======================================================================
    fig2, (axA, axB, axC) = plt.subplots(1, 3, figsize=(16.2, 5.2))
    fig2.subplots_adjust(top=0.83, wspace=0.28)

    # Panel A: Overlay Observed (solid) vs KL+tau_v (black dashed) across 6 sizes
    for b in ORDER_BEADS:
        r = data[b]
        meta = BEAD_META[b]
        axA.loglog(lags_s, r["msd_obs"], "-", color=meta["color"], lw=2.3,
                   label=f"{fmt_diam(r['diameter_um'])} $\\mu$m ($x={r['rc_over_xi']:.2f}$)", zorder=4)
        axA.loglog(lags_s, r["msd_kl"], "--", color="#111111", lw=1.4, alpha=0.88, zorder=5)

    h_a, l_a = axA.get_legend_handles_labels()
    h_a.append(Line2D([0], [0], color="#111111", ls="--", lw=1.6))
    l_a.append(r"KL + $\tau_{\mathbf{v}}^{\mathrm{slow}}$ (0-param)")
    axA.legend(h_a, l_a, loc="lower right", fontsize=7.6, framealpha=0.93)
    axA.set_xlim(4.0, 320.0)
    axA.set_ylim(0.10, 1000.0)
    axA.set_xlabel(r"Lag time $\Delta t$ [s]")
    axA.set_ylabel(r"$\mathrm{MSD}(\Delta t)$ [$\mu\mathrm{m}^2$]")
    axA.set_title(r"A. Observed vs. $D_{\mathrm{KL}}$-Predicted MSD", fontweight="bold")
    axA.grid(True, which="both", ls=":", alpha=0.45)

    # Panel B: Parity Plot across all 6 sizes & 75 lags (450 points)
    g_x = np.logspace(-1, 3.1, 100)
    axB.loglog(g_x, g_x, "-", color="0.2", lw=1.3, label=r"$y = x$ (exact)")
    axB.fill_between(g_x, 0.5 * g_x, 2.0 * g_x, color="0.85", alpha=0.45, label=r"$\times 2$ band")

    p240_offsets = {
        "beads06um": (-10, 10, "right"),
        "beads1um": (-10, 8, "right"),
        "beads3um": (8, -12, "left"),
        "beads7um": (-10, 8, "right"),
        "beads5um": (8, -12, "left"),
        "beads20um": (-10, 10, "right"),
    }
    for b in ORDER_BEADS:
        r = data[b]
        meta = BEAD_META[b]
        axB.loglog(r["msd_obs"], r["msd_kl"], "-", color=meta["color"], lw=2.0, alpha=0.85, zorder=4)
        # Highlight t = 240 s endpoint
        ox, oy, ha_s = p240_offsets[b]
        axB.loglog(r["msd_obs"][59], r["msd_kl"][59], marker=meta["marker"], color=meta["color"],
                   ms=7.5, markeredgecolor="k", markeredgewidth=0.7, zorder=6)
        axB.annotate(
            f"{fmt_diam(r['diameter_um'])} $\\mu$m",
            xy=(r["msd_obs"][59], r["msd_kl"][59]),
            xytext=(ox, oy), textcoords="offset points", ha=ha_s,
            fontsize=8.0, color=meta["color"], fontweight="bold"
        )

    axB.text(
        0.05, 0.95,
        f"0-param ($D_{{\\mathrm{{KL}}}} + \\tau_{{\\mathbf{{v}}}}^{{\\mathrm{{slow}}}}$):\n"
        f"Global $R^2_{{\\log}} = {glob_r2_kl:.3f}$\n"
        f"RMSE $= {glob_rmse_kl:.3f}$ dex ($\\times {10**glob_rmse_kl:.2f}$)\n"
        f"Markers: $\\Delta t = 240$ s",
        transform=axB.transAxes, fontsize=8.3, va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.72", alpha=0.92),
    )
    axB.set_xlim(0.10, 1000.0)
    axB.set_ylim(0.10, 1000.0)
    axB.set_xlabel(r"Observed $\mathrm{MSD}(\Delta t)$ [$\mu\mathrm{m}^2$]")
    axB.set_ylabel(r"Predicted $\mathrm{MSD}_{\mathrm{KL}}(\Delta t)$ [$\mu\mathrm{m}^2$]")
    axB.set_title(r"B. Parity Across All Sizes (4–300 s)", fontweight="bold")
    axB.legend(loc="lower right", fontsize=8.2, framealpha=0.93)
    axB.grid(True, which="both", ls=":", alpha=0.45)

    # Panel C: Relaxation timescales vs x = R_c / xi
    x_arr = df_sum["rc_over_xi"].values
    axC.loglog(x_arr, df_sum["tau_u_lag_s"].values, "--s", color="#1f77b4", lw=2.0, ms=6.8,
               markeredgecolor="k", markeredgewidth=0.6,
               label=r"Flow memory $\tau_{\bar{\mathbf{u}}}^{\mathrm{Lag}}$", zorder=4)

    mask_tv = df_sum["tau_v_slow_s"].values > 0
    axC.loglog(x_arr[mask_tv], df_sum["tau_v_slow_s"].values[mask_tv], "-o", color="#c0392b", lw=2.2, ms=7.2,
               markeredgecolor="k", markeredgewidth=0.6,
               label=r"Cargo slow time $\tau_{\mathbf{v}}^{\mathrm{slow}}$", zorder=5)
    # Show 20 um upper bound (< 4 s) cleanly with downward marker
    if not np.all(mask_tv):
        axC.loglog(x_arr[~mask_tv], [2.5] * int(np.sum(~mask_tv)), "v", color="#c0392b", ms=7.5,
                   markeredgecolor="k", markeredgewidth=0.6, zorder=5)
        axC.annotate("< 4 s", xy=(x_arr[-1], 2.5), xytext=(-8, 6), textcoords="offset points",
                     ha="right", fontsize=8.2, color="#c0392b", fontweight="bold")

    axC.loglog(x_arr, df_sum["tau_gk_s"].values, "-.^", color="#117733", lw=1.9, ms=6.8,
               markeredgecolor="k", markeredgewidth=0.6,
               label=r"Effective GK time $\tau_{\mathrm{GK}}$", zorder=4)

    axC.axvline(0.20, color="0.45", ls=":", lw=1.2, zorder=1)
    axC.axvline(1.0, color="0.25", ls="-.", lw=1.1, zorder=1)
    axC.text(0.105, 420.0, "Flow-lifetime\nlimited", fontsize=8.3, color="#1f77b4", fontweight="bold", ha="center", va="top")
    axC.text(0.55, 420.0, "Decoupling\nlimited", fontsize=8.3, color="#c0392b", fontweight="bold", ha="center", va="top")
    axC.text(1.05, 1.0, r"$x = 1$", fontsize=8.3, color="0.3", ha="left")

    axC.set_xlim(0.045, 3.2)
    axC.set_ylim(0.5, 500.0)
    axC.set_xlabel(r"Dimensionless Cargo Radius $x = R_c / \xi$")
    axC.set_ylabel(r"Relaxation Timescale $\tau$ [s]")
    axC.set_title(r"C. Flow-Limited vs. Decoupling Timescales", fontweight="bold")
    axC.legend(loc="lower left", fontsize=8.0, framealpha=0.93)
    axC.grid(True, which="both", ls=":", alpha=0.45)

    fig2.suptitle(
        r"Unified Link: 1-Step KL Divergence & Lagrangian Memory Predict Macroscopic MSD($\Delta t$)",
        fontsize=13.6, fontweight="bold", y=0.96
    )
    save_fig_multi(fig2, "msd_kl_relaxation_synthesis")
    plt.close(fig2)
    print("\nSaved both figures and summary CSV to figure/msd/ successfully!")


if __name__ == "__main__":
    main()
