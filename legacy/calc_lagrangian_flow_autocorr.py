#!/usr/bin/env python3
"""
calc_lagrangian_flow_autocorr.py

貨物直下（ビーズ半径内をくり抜いたマスク済み円環窓）の平均微小管流ベクトル
    u_bar_i(t) = (flow_u, flow_v)
に対するラグランジュ時間自己相関関数を計算・可視化する。

使用データ:
  - scratch/contrast_inst_cache.pkl
    （beads_flow_interaction.csv におけるビーズ直下くり抜き済み平均流 mean_flow_u, mean_flow_v と同一、
      アクティブ粒子フィルタ済み・重複なし）
  - 比較用のオイラー的空間固定流 VACF: figure/flow_vacf/flow_vacf_summary.csv

算出する自己相関指標（Δt = 4 s/frame, lag = 0..120 frames = 0..480 s）:
  1. vec_raw      : C_u(tau) = < u_bar_i(t) . u_bar_i(t+tau) > / < |u_bar_i|^2 >
                    （オイラー流 VACF calc_flow_vacf.py と同じ無差し引きベクトル自己相関）
  2. vec_exp_sub  : 各実験動画の平均流れ <u_bar>_exp を引いたゆらぎ δu_bar の自己相関
  3. vec_traj_sub : 各軌道ごとの時間平均 <u_bar>_traj を引いた軌道内ゆらぎ自己相関
  4. dir_raw      : 単位ベクトル u_hat_i(t) = u_bar_i / |u_bar_i| の方向自己相関
  5. cargo_raw    : 比較用の貨物速度 v_c(t) = (cargo_dx, cargo_dy) のラグランジュ VACF

誤差評価:
  - 粒子 (cargo_id) 単位のクラスターブートストラップ (n_boot = 1000) による 95% 信頼区間

出力先:
  - figure/lagrangian_flow_autocorr/
  - ROOT_DIR/figure/lagrangian_flow_autocorr/
"""

import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

warnings.filterwarnings("ignore")

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))
import verify_cos_decomposition as vcd

ROOT_DIR = vcd.ROOT_DIR
INST_CACHE = CURRENT_DIR / "scratch" / "contrast_inst_cache.pkl"
EULER_SUMMARY_CSV = CURRENT_DIR / "figure" / "flow_vacf" / "flow_vacf_summary.csv"

OUT_DIRS = [CURRENT_DIR / "figure" / "lagrangian_flow_autocorr"]
if ROOT_DIR.exists() and ROOT_DIR != CURRENT_DIR:
    OUT_DIRS.append(ROOT_DIR / "figure" / "lagrangian_flow_autocorr")

DT = 4.0          # s / frame
MAX_LAG = 120     # frames (480 s)
PLOT_LAG_S = 240  # default curve plot range (s)
MIN_PAIRS = 30
N_BOOT = 1000

X_LOCK = 45.0 * np.pi / 512.0   # ~0.276
X_BULK = 45.0 * np.pi / 128.0   # ~1.104

ORDER_BEADS = ["beads06um", "beads1um", "beads3um", "beads7um", "beads5um", "beads20um"]
BEAD_META = {b["name"]: b for b in vcd.BEADS_INFO}


def compute_per_traj_moments(
    groups: List[pd.DataFrame],
    ux_col: str,
    uy_col: str,
    sub_mode: str = "raw",
    normalize_unit: bool = False,
    exp_means: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    各軌道 k、ラグ lag=0..MAX_LAG について、内積和 S[k, lag] とペア数 N[k, lag] を事前計算する。
    """
    n_g = len(groups)
    S = np.zeros((n_g, MAX_LAG + 1), dtype=float)
    N = np.zeros((n_g, MAX_LAG + 1), dtype=float)

    for k, g in enumerate(groups):
        f = g["frame"].values.astype(int)
        ux = g[ux_col].values.astype(float)
        uy = g[uy_col].values.astype(float)

        if normalize_unit:
            nm = np.hypot(ux, uy)
            valid = nm > 0
            ux = np.where(valid, ux / np.where(valid, nm, 1.0), np.nan)
            uy = np.where(valid, uy / np.where(valid, nm, 1.0), np.nan)

        if sub_mode == "exp" and exp_means is not None:
            ed = str(g["exp_dir"].iloc[0])
            mx, my = exp_means[ed]
            ux = ux - mx
            uy = uy - my
        elif sub_mode == "traj":
            ux = ux - np.nanmean(ux)
            uy = uy - np.nanmean(uy)

        f0 = int(f.min())
        L = int(f.max() - f0 + 1)
        gx = np.full(L, np.nan, dtype=float)
        gy = np.full(L, np.nan, dtype=float)
        gx[f - f0] = ux
        gy[f - f0] = uy

        ok = np.isfinite(gx) & np.isfinite(gy)
        gx0 = np.where(ok, gx, 0.0)
        gy0 = np.where(ok, gy, 0.0)

        max_l = min(MAX_LAG, L - 1)
        for lag in range(max_l + 1):
            m = ok[: L - lag] & ok[lag:]
            cnt = int(np.sum(m))
            if cnt > 0:
                N[k, lag] = cnt
                S[k, lag] = float(np.sum((gx0[: L - lag] * gx0[lag:] + gy0[: L - lag] * gy0[lag:])[m]))

    return S, N


def acf_from_moments(S: np.ndarray, N: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    s = S.sum(axis=0)
    n = N.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = s / n
    raw[n < MIN_PAIRS] = np.nan
    c0 = raw[0]
    if not np.isfinite(c0) or c0 <= 0:
        return np.full_like(raw, np.nan), n
    return raw / c0, n


def extract_timescales(C: np.ndarray, do_fit: bool = True) -> Tuple[float, float, float, float]:
    """
    C(tau) から以下を算出:
      - tau_1e_s : 初めて 1/e を下回るラグ時間（線形補間）
      - tau_int_s: 最初の C <= 0 まで積分した緩和時間
      - tau_s_fit: tau >= DT に C = a * exp(-tau/tau_s) + b をフィットした遅い時定数
      - b_fit    : フィットのプラトー値 b
    """
    target = np.exp(-1.0)
    tau_1e = np.nan
    below = np.where(np.isfinite(C) & (C < target))[0]
    if len(below) > 0 and below[0] > 0:
        j = int(below[0])
        c0, c1 = C[j - 1], C[j]
        if np.isfinite(c0) and np.isfinite(c1) and (c0 - c1) > 0:
            tau_1e = (j - 1 + (c0 - target) / (c0 - c1)) * DT

    neg = np.where(np.isfinite(C) & (C <= 0.0))[0]
    if len(neg) > 0:
        jend = int(neg[0])
        tau_int = float((0.5 * C[0] + np.nansum(C[1:jend])) * DT)
    else:
        tau_int = np.nan

    tau_s_fit, b_fit = np.nan, np.nan
    if do_fit:
        lags_s = np.arange(len(C)) * DT
        m = (lags_s >= DT) & (lags_s <= PLOT_LAG_S) & np.isfinite(C)
        if int(np.sum(m)) >= 6:
            try:
                popt, _ = curve_fit(
                    lambda t, a, ts, b: a * np.exp(-t / ts) + b,
                    lags_s[m],
                    C[m],
                    p0=(0.5, 40.0, 0.1),
                    bounds=([0.0, DT / 2.0, -0.3], [1.5, 5000.0, 1.0]),
                    maxfev=8000,
                )
                tau_s_fit = float(popt[1])
                b_fit = float(popt[2])
            except Exception:
                pass

    return tau_1e, tau_int, tau_s_fit, b_fit


def pct_ci(arr: np.ndarray, q: float) -> float:
    valid = arr[np.isfinite(arr)]
    return float(np.percentile(valid, q)) if len(valid) >= 10 else np.nan


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
        "legend.fontsize": 8.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    df = pd.read_pickle(INST_CACHE)
    df["cargo_id"] = df["exp_dir"].astype(str) + "_" + df["particle"].astype(str)
    df = df.sort_values(["cargo_id", "frame"]).reset_index(drop=True)

    euler_summary = {}
    if EULER_SUMMARY_CSV.exists():
        df_eu = pd.read_csv(EULER_SUMMARY_CSV)
        for bname, grp in df_eu.groupby("bead_name"):
            if bname not in BEAD_META:
                continue
            t = grp["lag_time_s"].values
            c = grp["Cv_mean"].values
            bel = np.where(c < np.exp(-1.0))[0]
            t1e = np.nan
            if len(bel) > 0 and bel[0] > 0:
                j = int(bel[0])
                t1e = float(t[j - 1] + (c[j - 1] - np.exp(-1.0)) / (c[j - 1] - c[j]) * (t[j] - t[j - 1]))
            c4 = float(c[t == 4.0][0]) if (t == 4.0).any() else np.nan
            c60 = float(c[t == 60.0][0]) if (t == 60.0).any() else np.nan
            c240 = float(c[t == 240.0][0]) if (t == 240.0).any() else np.nan
            euler_summary[bname] = {
                "lag_s": t,
                "Cv": c,
                "Cv_sem": grp["Cv_sem"].values,
                "C_4s": c4,
                "C_60s": c60,
                "C_240s": c240,
                "tau_1e_s": t1e,
            }

    modes = [
        ("vec_raw", "flow_u", "flow_v", "raw", False),
        ("vec_exp_sub", "flow_u", "flow_v", "exp", False),
        ("vec_traj_sub", "flow_u", "flow_v", "traj", False),
        ("dir_raw", "flow_u", "flow_v", "raw", True),
        ("cargo_raw", "cargo_dx", "cargo_dy", "raw", False),
    ]

    summary_rows = []
    curve_rows = []
    rng = np.random.default_rng(42)

    for b_idx, bname in enumerate(ORDER_BEADS):
        sub = df[df["bead_name"] == bname].copy()
        groups = [g for _, g in sub.groupby("cargo_id")]
        n_cargo = len(groups)
        rc_per_cargo = np.array([float(g["rc_over_xi"].iloc[0]) for g in groups])
        rc_over_xi = float(np.mean(rc_per_cargo))
        rc_over_xi_sem = float(np.std(rc_per_cargo, ddof=1) / np.sqrt(n_cargo)) if n_cargo > 1 else 0.0
        mean_u_spd = float(sub["flow_spd"].mean())
        mean_c_spd_um_s = float(sub["cargo_spd"].mean() * vcd.SCALE_UM_PER_PX / DT)
        diam = BEAD_META[bname]["diameter_um"]

        # Precompute experiment means for flow_u, flow_v
        exp_means = {}
        for ed, egrp in sub.groupby("exp_dir"):
            exp_means[str(ed)] = (float(egrp["flow_u"].mean()), float(egrp["flow_v"].mean()))

        for m_name, ux_c, uy_c, s_mode, norm_u in modes:
            S, N = compute_per_traj_moments(groups, ux_c, uy_c, sub_mode=s_mode, normalize_unit=norm_u, exp_means=exp_means)
            C, n_pairs = acf_from_moments(S, N)
            t1e, tint, tau_s, b_fit = extract_timescales(C)

            boot_C = np.full((N_BOOT, MAX_LAG + 1), np.nan)
            boot_t1e = np.full(N_BOOT, np.nan)
            boot_tint = np.full(N_BOOT, np.nan)
            boot_taus = np.full(N_BOOT, np.nan)
            for bi in range(N_BOOT):
                sel = rng.integers(0, n_cargo, size=n_cargo)
                Cb, _ = acf_from_moments(S[sel], N[sel])
                boot_C[bi] = Cb
                bt1, bti, bts, _ = extract_timescales(Cb, do_fit=False)
                boot_t1e[bi] = bt1
                boot_tint[bi] = bti
                boot_taus[bi] = bts

            summary_rows.append({
                "bead_name": bname,
                "diameter_um": diam,
                "rc_over_xi": rc_over_xi,
                "rc_over_xi_sem": rc_over_xi_sem,
                "mode": m_name,
                "n_cargos": n_cargo,
                "n_frames": len(sub),
                "mean_flow_spd_px_frm": mean_u_spd,
                "mean_cargo_spd_um_s": mean_c_spd_um_s,
                "C_4s": float(C[1]),
                "C_4s_ci_lo": pct_ci(boot_C[:, 1], 2.5),
                "C_4s_ci_hi": pct_ci(boot_C[:, 1], 97.5),
                "fast_drop_4s": float(1.0 - C[1]),
                "fast_drop_4s_ci_lo": float(1.0 - pct_ci(boot_C[:, 1], 97.5)),
                "fast_drop_4s_ci_hi": float(1.0 - pct_ci(boot_C[:, 1], 2.5)),
                "C_20s": float(C[5]),
                "C_60s": float(C[15]),
                "C_120s": float(C[30]),
                "C_240s": float(C[60]),
                "C_240s_ci_lo": pct_ci(boot_C[:, 60], 2.5),
                "C_240s_ci_hi": pct_ci(boot_C[:, 60], 97.5),
                "tau_1e_s": t1e,
                "tau_1e_ci_lo": pct_ci(boot_t1e, 2.5),
                "tau_1e_ci_hi": pct_ci(boot_t1e, 97.5),
                "tau_int_s": tint,
                "tau_int_ci_lo": pct_ci(boot_tint, 2.5),
                "tau_int_ci_hi": pct_ci(boot_tint, 97.5),
                "tau_slow_fit_s": tau_s,
                "tau_slow_fit_ci_lo": pct_ci(boot_taus, 2.5),
                "tau_slow_fit_ci_hi": pct_ci(boot_taus, 97.5),
                "plateau_fit": b_fit,
                "euler_C_4s": euler_summary[bname]["C_4s"] if bname in euler_summary else np.nan,
                "euler_C_60s": euler_summary[bname]["C_60s"] if bname in euler_summary else np.nan,
                "euler_C_240s": euler_summary[bname]["C_240s"] if bname in euler_summary else np.nan,
                "euler_tau_1e_s": euler_summary[bname]["tau_1e_s"] if bname in euler_summary else np.nan,
            })

            for lag in range(MAX_LAG + 1):
                curve_rows.append({
                    "bead_name": bname,
                    "diameter_um": diam,
                    "rc_over_xi": rc_over_xi,
                    "mode": m_name,
                    "lag_frames": lag,
                    "lag_s": lag * DT,
                    "C": float(C[lag]),
                    "C_ci_lo": pct_ci(boot_C[:, lag], 2.5),
                    "C_ci_hi": pct_ci(boot_C[:, lag], 97.5),
                    "n_pairs": int(n_pairs[lag]),
                })

    df_sum = pd.DataFrame(summary_rows)
    df_cur = pd.DataFrame(curve_rows)
    save_csv_multi(df_sum, "lagrangian_u_bar_autocorr_summary.csv")
    save_csv_multi(df_cur, "lagrangian_u_bar_autocorr_curves.csv")

    # ------------------------------------------------------------------
    # Figure 1: Autocorrelation Curves (4 panels)
    # ------------------------------------------------------------------
    fig1, axes = plt.subplots(2, 2, figsize=(13.2, 9.6))
    fig1.subplots_adjust(top=0.90, hspace=0.32, wspace=0.22)

    panel_specs = [
        (axes[0, 0], "vec_raw", r"A. Vector VACF $C_{\bar{\mathbf{u}}}^{\mathrm{raw}}(\tau)$ (masked footprint $\bar{\mathbf{u}}_i$)", 240),
        (axes[0, 1], "vec_exp_sub", r"B. Fluctuation VACF $C_{\bar{\mathbf{u}}}^{\mathrm{exp\text{-}sub}}(\tau)$ (movie mean subtracted)", 240),
        (axes[1, 0], "dir_raw", r"C. Unit-Direction ACF $C_{\hat{\bar{\mathbf{u}}}}(\tau) = \langle \hat{\bar{\mathbf{u}}}_i(t)\cdot\hat{\bar{\mathbf{u}}}_i(t+\tau)\rangle$", 240),
        (axes[1, 1], "vec_raw", r"D. Short-Lag Semilog Decay ($0$–$120$ s) & Eulerian Ref", 120),
    ]

    for ax, m_name, title, xmax in panel_specs:
        sub_m = df_cur[(df_cur["mode"] == m_name) & (df_cur["lag_s"] <= xmax)]
        for bname in ORDER_BEADS:
            meta = BEAD_META[bname]
            g = sub_m[sub_m["bead_name"] == bname]
            n_c = int(df_sum[(df_sum["bead_name"] == bname) & (df_sum["mode"] == m_name)]["n_cargos"].iloc[0])
            ax.plot(
                g["lag_s"], g["C"],
                color=meta["color"], marker=meta["marker"], ms=3.5, lw=1.8, markevery=3 if xmax == 240 else 2,
                label=f"{meta['label']} (N={n_c})", zorder=4
            )
            if m_name in ("vec_raw", "vec_exp_sub", "dir_raw") and xmax == 240:
                ax.fill_between(g["lag_s"], g["C_ci_lo"], g["C_ci_hi"], color=meta["color"], alpha=0.14, lw=0)

        if xmax == 120 and "beads06um" in euler_summary:
            # Show Eulerian bulk average range lightly for visual comparison
            eu = euler_summary["beads3um"]
            m_eu = eu["lag_s"] <= 120
            ax.plot(eu["lag_s"][m_eu], eu["Cv"][m_eu], color="0.3", ls="--", lw=1.6,
                    label=r"Eulerian fixed-point (3.37 $\mu\mathrm{m}$ ref)", zorder=3)
            ax.set_yscale("log")
            ax.set_ylim(0.08, 1.15)
        else:
            ax.set_ylim(-0.08, 1.04)

        ax.axhline(np.exp(-1.0), color="0.45", ls=":", lw=1.2, zorder=1)
        ax.text(xmax * 0.98, np.exp(-1.0) * 1.03, r"$1/e \approx 0.368$", fontsize=8.3, color="0.35", ha="right", va="bottom")
        ax.axhline(0.0, color="0.25", lw=0.6)
        ax.set_xlim(0, xmax)
        ax.set_xlabel(r"Lag time $\tau$ [s]")
        ax.set_ylabel(r"Autocorrelation $C(\tau)$")
        ax.set_title(title)
        ax.grid(alpha=0.3)

    axes[0, 0].legend(loc="upper right", frameon=True, framealpha=0.92, fontsize=8.2, ncol=2)
    axes[1, 1].legend(loc="lower left", frameon=True, framealpha=0.92, fontsize=8.0)
    fig1.suptitle(r"Lagrangian Autocorrelation of Masked Cargo-Footprint Mean Flow $\bar{\mathbf{u}}_i(t)$ ($\Delta t = 4$ s)", fontsize=13.5, y=0.97)
    save_fig_multi(fig1, "lagrangian_u_bar_autocorr_curves")
    plt.close(fig1)

    # ------------------------------------------------------------------
    # Figure 2: Comparison vs R_c / xi & Eulerian / Cargo VACF
    # ------------------------------------------------------------------
    fig2, (ax21, ax22, ax23) = plt.subplots(1, 3, figsize=(15.5, 4.8))
    fig2.subplots_adjust(top=0.82, wspace=0.28)

    df_raw = df_sum[df_sum["mode"] == "vec_raw"].copy()
    df_exp = df_sum[df_sum["mode"] == "vec_exp_sub"].copy()
    df_dir = df_sum[df_sum["mode"] == "dir_raw"].copy()
    df_car = df_sum[df_sum["mode"] == "cargo_raw"].copy()

    # Panel A: 1-lag correlation C(4 s) vs R_c / xi (Lagrangian raw, exp-sub, dir, Eulerian, Cargo)
    for sub_df, lbl, col, fmt, ls in [
        (df_raw, r"Lagrangian $\bar{\mathbf{u}}_i$ (raw)", "#1b4f72", "-o", "-"),
        (df_exp, r"Lagrangian $\delta\bar{\mathbf{u}}_i$ (exp-sub)", "#2e86c1", "--s", "--"),
        (df_dir, r"Lagrangian $\hat{\bar{\mathbf{u}}}_i$ (direction)", "#7d3c98", ":^", ":"),
        (df_car, r"Cargo velocity $\mathbf{v}_c$ (raw)", "#c0392b", "-.v", "-."),
    ]:
        x = sub_df["rc_over_xi"].values
        y = sub_df["C_4s"].values
        yerr = np.vstack([np.maximum(0.0, y - sub_df["C_4s_ci_lo"].values),
                          np.maximum(0.0, sub_df["C_4s_ci_hi"].values - y)])
        ax21.errorbar(x, y, yerr=yerr, fmt=fmt, color=col, lw=1.7, ms=6.2, capsize=3.0, label=lbl, zorder=4)

    # Overlay Eulerian C(4s)
    ax21.plot(df_raw["rc_over_xi"], df_raw["euler_C_4s"], "D--", color="0.35", lw=1.5, ms=5.5,
              label=r"Eulerian flow $\mathbf{u}(\mathbf{r},t)$", zorder=3)
    ax21.axvline(X_LOCK, color="0.45", ls="--", lw=1.1)
    ax21.axvline(X_BULK, color="0.45", ls=":", lw=1.1)
    ax21.axhline(0.0, color="0.3", lw=0.7)
    ax21.set_xscale("log")
    ax21.set_xlim(0.045, 3.2)
    ax21.set_ylim(-0.44, 1.06)
    ax21.set_xlabel(r"$R_c / \xi$")
    ax21.set_ylabel(r"1-Step Autocorrelation $C(\Delta t = 4\,\mathrm{s})$")
    ax21.set_title(r"A. Short-Time Persistence $C(4\,\mathrm{s})$")
    ax21.legend(loc="lower left", frameon=True, framealpha=0.94, fontsize=7.6)
    ax21.grid(alpha=0.3, which="both")

    # Panel B: Long-lag persistence C(60 s) and C(240 s) vs R_c / xi
    for sub_df, col_y, lbl, col, fmt in [
        (df_raw, "C_60s", r"Lag. $\bar{\mathbf{u}}_i$: $C(60\,\mathrm{s})$", "#1b4f72", "-o"),
        (df_raw, "C_240s", r"Lag. $\bar{\mathbf{u}}_i$: $C(240\,\mathrm{s})$", "#2980b9", "--s"),
        (df_exp, "C_60s", r"Lag. $\delta\bar{\mathbf{u}}_i$ (exp-sub): $C(60\,\mathrm{s})$", "#16a085", "-.^"),
        (df_raw, "euler_C_60s", r"Eulerian $\mathbf{u}$: $C(60\,\mathrm{s})$", "0.35", ":D"),
    ]:
        ax22.plot(sub_df["rc_over_xi"], sub_df[col_y], fmt, color=col, lw=1.7, ms=6.2, label=lbl, zorder=4)

    ax22.axvline(X_LOCK, color="0.45", ls="--", lw=1.1)
    ax22.axvline(X_BULK, color="0.45", ls=":", lw=1.1)
    ax22.text(X_LOCK * 0.94, 0.04, r"$x_{\mathrm{lock}}$", fontsize=8.5, color="0.35", va="bottom", ha="right")
    ax22.text(X_BULK * 1.06, 0.04, r"$x_{\mathrm{bulk}}$", fontsize=8.5, color="0.35", va="bottom", ha="left")
    ax22.set_xscale("log")
    ax22.set_xlim(0.045, 3.2)
    ax22.set_ylim(-0.02, 1.04)
    ax22.set_xlabel(r"$R_c / \xi$")
    ax22.set_ylabel(r"Autocorrelation $C(\tau)$")
    ax22.set_title(r"B. Mid/Long-Time Persistence ($60$ s, $240$ s)")
    ax22.legend(loc="upper left", frameon=True, framealpha=0.94, fontsize=7.8)
    ax22.grid(alpha=0.3, which="both")

    # Panel C: Decay timescale tau_{1/e} (Lagrangian vs Eulerian vs Cargo)
    for sub_df, col, fmt, lbl in [
        (df_raw, "#1b4f72", "-o", r"Lagrangian $\bar{\mathbf{u}}_i$ (raw)"),
        (df_exp, "#2e86c1", "--s", r"Lagrangian $\delta\bar{\mathbf{u}}_i$ (exp-sub)"),
        (df_car, "#c0392b", "-.v", r"Cargo $\mathbf{v}_c$ (raw)"),
    ]:
        m = np.isfinite(sub_df["tau_1e_s"].values)
        x = sub_df["rc_over_xi"].values[m]
        y = sub_df["tau_1e_s"].values[m]
        y_lo = sub_df["tau_1e_ci_lo"].values[m]
        y_hi = sub_df["tau_1e_ci_hi"].values[m]
        yerr = np.vstack([np.where(np.isfinite(y_lo), np.maximum(0.0, y - y_lo), 0.0),
                          np.where(np.isfinite(y_hi), np.maximum(0.0, y_hi - y), 0.0)])
        ax23.errorbar(x, y, yerr=yerr, fmt=fmt, color=col, lw=1.7, ms=6.2, capsize=3.0, label=lbl, zorder=4)

    m_eu = np.isfinite(df_raw["euler_tau_1e_s"].values)
    ax23.plot(df_raw["rc_over_xi"].values[m_eu], df_raw["euler_tau_1e_s"].values[m_eu],
              "D:", color="0.35", lw=1.5, ms=5.5, label=r"Eulerian flow $\mathbf{u}$", zorder=3)
    ax23.axvline(X_LOCK, color="0.45", ls="--", lw=1.1)
    ax23.axvline(X_BULK, color="0.45", ls=":", lw=1.1)
    ax23.axhline(DT, color="0.5", ls=":", lw=1.0)
    ax23.text(0.048, DT * 0.85, r"$\Delta t = 4$ s", fontsize=8.2, color="0.4", va="top")
    ax23.set_xscale("log")
    ax23.set_yscale("log")
    ax23.set_xlim(0.045, 3.2)
    ax23.set_ylim(0.8, 2200.0)
    ax23.set_xlabel(r"$R_c / \xi$")
    ax23.set_ylabel(r"Relaxation Time $\tau_{1/e}$ [s]")
    ax23.set_title(r"C. Decorrelation Time $\tau_{1/e}$ vs. $R_c/\xi$")
    ax23.legend(loc="lower right", frameon=True, framealpha=0.92, fontsize=7.8)
    ax23.grid(alpha=0.3, which="both")

    fig2.suptitle(r"Lagrangian $\bar{\mathbf{u}}_i(t)$ Decorrelation vs. $R_c/\xi$, Eulerian Flow, and Cargo Velocity", fontsize=13.2, y=0.95)
    save_fig_multi(fig2, "lagrangian_vs_eulerian_u_bar_summary")
    plt.close(fig2)

    # Print key tables
    for m_show in ["vec_raw", "vec_exp_sub", "dir_raw", "cargo_raw"]:
        print(f"\n=== Summary: {m_show} ===")
        sub_s = df_sum[df_sum["mode"] == m_show]
        cols = [
            "bead_name", "diameter_um", "rc_over_xi", "n_cargos", "n_frames",
            "mean_cargo_spd_um_s", "mean_flow_spd_px_frm",
            "C_4s", "C_4s_ci_lo", "C_4s_ci_hi", "C_60s", "C_240s",
            "tau_1e_s", "tau_1e_ci_lo", "tau_1e_ci_hi", "tau_slow_fit_s", "euler_tau_1e_s"
        ]
        print(sub_s[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
