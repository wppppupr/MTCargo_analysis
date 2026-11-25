#!/usr/bin/env python3
"""
plot_run_tumble_autocorr_and_green_kubo.py

以下の 2 つの独立した新規図を作成する（既存ファイルは上書きしない）:
  1. gmm_run_tumble_lagrangian_autocorr.{png,svg}
     - 小粒子 (0.63 um, 1.18 um) の GMM Run / Tumble 状態別ラグランジュ自己相関
       (足元平均流 C_u(tau) vs 貨物速度 C_v(tau), 95% 粒子クラスターブートストラップ CI)
  2. green_kubo_msd_tail_decomposition.{png,svg}
     - 離散 Green-Kubo (Taylor 積分) による MSD(240 s) の無相関拡散項 vs 相関テール項の厳密分解、
       貨物速度自己相関 C_v(tau)、および R_c/xi に対する C_u(60 s) と C_v(60 s) / テール寄与率の逆転構造

出力先:
  - figure/lagrangian_flow_autocorr/
  - ROOT_DIR/figure/lagrangian_flow_autocorr/
"""

import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))

import verify_cos_decomposition as vcd
from analyze_gmm_run_tumble_potential_and_kl import build_gmm_labeled_frames

ROOT_DIR = vcd.ROOT_DIR
INST_CACHE = CURRENT_DIR / "scratch" / "contrast_inst_cache.pkl"
POT_MAIN_CSV = CURRENT_DIR / "figure" / "per_point_v_dot_u" / "effective_potential_depth_summary.csv"

OUT_DIRS = [CURRENT_DIR / "figure" / "lagrangian_flow_autocorr"]
if ROOT_DIR.exists() and ROOT_DIR != CURRENT_DIR:
    OUT_DIRS.append(ROOT_DIR / "figure" / "lagrangian_flow_autocorr")

ORDER_BEADS = ["beads06um", "beads1um", "beads3um", "beads7um", "beads5um", "beads20um"]
BEAD_META = {b["name"]: b for b in vcd.BEADS_INFO}

SCALE = vcd.SCALE_UM_PER_PX  # 0.11 um/px
DT = 4.0                     # s / frame
X_LOCK = 45.0 * np.pi / 512.0
X_BULK = 45.0 * np.pi / 128.0
N_BOOT = 1000
MIN_PAIRS = 30


def format_diam_short(d: float) -> str:
    if d in (0.63, 1.18, 3.37, 7.24):
        return f"{d:.2f}"
    if d == 5.0:
        return "5.0"
    return f"{d:.0f}"


def compute_state_pair_arrays(
    groups: List[pd.DataFrame],
    ux_col: str,
    uy_col: str,
    max_lag: int,
    state_filter: str = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    各粒子 k, lag=0..max_lag に対して:
      S_num[k, lag]: sum(vec(t) . vec(t+lag))
      S_den[k, lag]: sum(0.5*(|vec(t)|^2 + |vec(t+lag)|^2)) (状態フィルタ時の局所分散正規化)
      N_cnt[k, lag]: ペア数
    state_filter が指定された場合、時刻 t と t+lag の両方がその状態にあるペアのみ集計する。
    """
    n_g = len(groups)
    S_num = np.zeros((n_g, max_lag + 1), dtype=float)
    S_den = np.zeros((n_g, max_lag + 1), dtype=float)
    N_cnt = np.zeros((n_g, max_lag + 1), dtype=float)

    for k, g in enumerate(groups):
        if state_filter is not None:
            g = g[g["state_label"] == state_filter]
            if g.empty:
                continue
        f = g["frame"].values.astype(int)
        vx = g[ux_col].values.astype(float) * SCALE
        vy = g[uy_col].values.astype(float) * SCALE
        f0 = int(f.min())
        L = int(f.max() - f0 + 1)
        gx = np.full(L, np.nan)
        gy = np.full(L, np.nan)
        gx[f - f0] = vx
        gy[f - f0] = vy
        ok = np.isfinite(gx) & np.isfinite(gy)
        gx0 = np.where(ok, gx, 0.0)
        gy0 = np.where(ok, gy, 0.0)
        v2 = gx0 ** 2 + gy0 ** 2

        for lag in range(min(max_lag, L - 1) + 1):
            m = ok[: L - lag] & ok[lag:]
            cnt = int(np.sum(m))
            if cnt > 0:
                N_cnt[k, lag] = cnt
                S_num[k, lag] = float(np.sum((gx0[: L - lag] * gx0[lag:] + gy0[: L - lag] * gy0[lag:])[m]))
                S_den[k, lag] = float(0.5 * np.sum((v2[: L - lag] + v2[lag:])[m]))

    return S_num, S_den, N_cnt


def eval_acf_pairnorm(S_num: np.ndarray, S_den: np.ndarray, N_cnt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    num = S_num.sum(axis=0)
    den = S_den.sum(axis=0)
    n = N_cnt.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        C = num / den
    C[n < MIN_PAIRS] = np.nan
    return C, n


def eval_acf_c0norm(S_num: np.ndarray, N_cnt: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    num = S_num.sum(axis=0)
    n = N_cnt.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        G = num / n
    G[n < MIN_PAIRS] = np.nan
    C = G / G[0] if np.isfinite(G[0]) and G[0] > 0 else np.full_like(G, np.nan)
    return G, C, n


def bootstrap_curve_pairnorm(S_num, S_den, N_cnt, rng) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    C, n = eval_acf_pairnorm(S_num, S_den, N_cnt)
    n_g = S_num.shape[0]
    boot = np.full((N_BOOT, S_num.shape[1]), np.nan)
    for bi in range(N_BOOT):
        sel = rng.integers(0, n_g, size=n_g)
        boot[bi], _ = eval_acf_pairnorm(S_num[sel], S_den[sel], N_cnt[sel])
    lo = np.nanpercentile(boot, 2.5, axis=0)
    hi = np.nanpercentile(boot, 97.5, axis=0)
    lo[np.isnan(C)] = np.nan
    hi[np.isnan(C)] = np.nan
    return C, lo, hi, n


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

    rng = np.random.default_rng(42)

    # ==================================================================
    # PART 1: GMM Run / Tumble Lagrangian Autocorrelation (0.63 & 1.18 um)
    # ==================================================================
    df_gmm = build_gmm_labeled_frames()
    df_gmm["cargo_id"] = df_gmm["exp_dir"].astype(str) + "_" + df_gmm["particle"].astype(str)
    df_gmm = df_gmm.sort_values(["cargo_id", "frame"]).reset_index(drop=True)

    MAX_LAG_RT = 30  # 120 s
    lags_rt_s = np.arange(MAX_LAG_RT + 1) * DT

    rt_rows = []
    fig1, axes1 = plt.subplots(1, 2, figsize=(13.2, 5.2), sharey=True)
    fig1.subplots_adjust(top=0.83, wspace=0.14)

    for idx_b, (bname, diam_str) in enumerate([("beads06um", r"0.63 $\mu\mathrm{m}$"), ("beads1um", r"1.18 $\mu\mathrm{m}$")]):
        ax = axes1[idx_b]
        sub_b = df_gmm[df_gmm["bead_name"] == bname]
        groups_b = [g for _, g in sub_b.groupby("cargo_id")]
        tot_v2 = float(((sub_b["cargo_spd"] * SCALE) ** 2).sum())

        # Compute variance shares
        shares = {}
        for st in ("Run", "Tumble"):
            s = sub_b[sub_b["state_label"] == st]
            shares[st] = (
                len(s) / len(sub_b),
                float(((s["cargo_spd"] * SCALE) ** 2).sum()) / tot_v2,
            )

        plot_cfg = [
            ("Run", "flow_u", "flow_v", r"Run: Flow $\bar{\mathbf{u}}_i$", "#1f77b4", "-", "^", 2.1, True),
            ("Run", "cargo_dx", "cargo_dy", r"Run: Cargo $\mathbf{v}_c$", "#d62728", "--", "o", 2.1, True),
            ("Tumble", "flow_u", "flow_v", r"Tumble: Flow $\bar{\mathbf{u}}_i$", "#6baed6", "-", "v", 1.7, True),
            ("Tumble", "cargo_dx", "cargo_dy", r"Tumble: Cargo $\mathbf{v}_c$", "#f1948a", "--", "s", 1.7, True),
            (None, "flow_u", "flow_v", r"All (unsplit): Flow $\bar{\mathbf{u}}_i$", "0.45", ":", None, 1.4, False),
        ]

        for st, ux, uy, lbl, col, ls, mk, lw, show_ci in plot_cfg:
            Sn, Sd, Nc = compute_state_pair_arrays(groups_b, ux, uy, MAX_LAG_RT, state_filter=st)
            C, lo, hi, n_p = bootstrap_curve_pairnorm(Sn, Sd, Nc, rng)
            ax.plot(
                lags_rt_s, C, color=col, ls=ls, marker=mk, ms=4.5 if mk else 0,
                markevery=2, lw=lw, label=lbl, zorder=5 if st == "Run" else 3
            )
            if show_ci:
                ax.fill_between(lags_rt_s, lo, hi, color=col, alpha=0.14 if st == "Run" else 0.10, lw=0)

            sig_name = "u_bar" if ux == "flow_u" else "v_c"
            st_name = st if st is not None else "All"
            for lag_i in range(MAX_LAG_RT + 1):
                rt_rows.append({
                    "bead_name": bname,
                    "state": st_name,
                    "signal": sig_name,
                    "lag_s": float(lags_rt_s[lag_i]),
                    "C": float(C[lag_i]),
                    "C_ci_lo": float(lo[lag_i]),
                    "C_ci_hi": float(hi[lag_i]),
                    "n_pairs": int(n_p[lag_i]),
                })

        ax.axhline(np.exp(-1.0), color="0.45", ls=":", lw=1.1, zorder=1)
        ax.text(118, np.exp(-1.0) + 0.02, r"$1/e$", fontsize=8.4, color="0.35", ha="right")
        ax.axhline(0.0, color="0.25", lw=0.7)

        info_box = (
            f"Run: {100*shares['Run'][0]:.0f}% frames, "
            f"{100*shares['Run'][1]:.1f}% of $\\Sigma|\\Delta\\mathbf{{r}}|^2$\n"
            f"Tumble: {100*shares['Tumble'][0]:.0f}% frames, "
            f"{100*shares['Tumble'][1]:.1f}% of $\\Sigma|\\Delta\\mathbf{{r}}|^2$"
        )
        ax.text(
            0.03, 0.04, info_box, transform=ax.transAxes, fontsize=8.3,
            va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.72", alpha=0.92),
        )

        ax.set_xlim(0, MAX_LAG_RT * DT)
        ax.set_ylim(-0.12, 1.05)
        ax.set_xlabel(r"Lag time $\tau$ [s]")
        if idx_b == 0:
            ax.set_ylabel(r"Lagrangian Autocorrelation $C(\tau)$")
        ax.set_title(f"{'A' if idx_b == 0 else 'B'}. {diam_str} (N = {len(groups_b)} cargos)")
        ax.grid(alpha=0.3)

    axes1[1].legend(loc="upper right", frameon=True, framealpha=0.94, fontsize=8.2)
    fig1.suptitle(
        r"Run vs. Tumble Lagrangian Autocorrelation: Masked Flow $\bar{\mathbf{u}}_i(t)$ & Cargo Velocity $\mathbf{v}_c(t)$",
        fontsize=13.0, y=0.95
    )
    save_fig_multi(fig1, "gmm_run_tumble_lagrangian_autocorr")
    plt.close(fig1)

    df_rt_cur = pd.DataFrame(rt_rows)
    save_csv_multi(df_rt_cur, "gmm_run_tumble_lagrangian_autocorr_curves.csv")

    # ==================================================================
    # PART 2: Green-Kubo MSD(240 s) Decomposition & Inversion vs R_c/xi
    # ==================================================================
    df_inst = pd.read_pickle(INST_CACHE)
    df_inst["cargo_id"] = df_inst["exp_dir"].astype(str) + "_" + df_inst["particle"].astype(str)
    df_inst = df_inst.sort_values(["cargo_id", "frame"]).reset_index(drop=True)
    df_pot = pd.read_csv(POT_MAIN_CSV).set_index("bead_name")

    M_STEPS = 60  # 240 s
    lags_gk_s = np.arange(M_STEPS + 1) * DT

    gk_summary = []
    cv_curves = {}

    for bname in ORDER_BEADS:
        sub = df_inst[df_inst["bead_name"] == bname]
        groups = [g for _, g in sub.groupby("cargo_id")]
        n_g = len(groups)
        diam = BEAD_META[bname]["diameter_um"]
        rc_xi = float(df_pot.loc[bname, "rc_over_xi"])

        Sn_v, _, Nc_v = compute_state_pair_arrays(groups, "cargo_dx", "cargo_dy", M_STEPS, state_filter=None)
        Sn_u, _, Nc_u = compute_state_pair_arrays(groups, "flow_u", "flow_v", M_STEPS, state_filter=None)

        Gv, Cv, _ = eval_acf_c0norm(Sn_v, Nc_v)
        Gu, Cu, _ = eval_acf_c0norm(Sn_u, Nc_u)

        # Discrete Green-Kubo for M = 60 (240 s)
        def gk_metrics(Gv_arr):
            msd0 = M_STEPS * Gv_arr[0]
            msd_t = 2.0 * np.sum([(M_STEPS - k) * Gv_arr[k] for k in range(1, M_STEPS)])
            msd_tot = msd0 + msd_t
            tf = msd_t / msd_tot if msd_tot > 0 else np.nan
            alp = np.log(msd_tot / Gv_arr[0]) / np.log(M_STEPS) if (msd_tot > 0 and Gv_arr[0] > 0) else np.nan
            return msd0, msd_t, msd_tot, tf, alp

        msd0, msd_t, msd_tot, tf, alp = gk_metrics(Gv)

        # Cluster bootstrap (1000)
        b_Cv = np.full((N_BOOT, M_STEPS + 1), np.nan)
        b_Cu60 = np.full(N_BOOT, np.nan)
        b_Cv60 = np.full(N_BOOT, np.nan)
        b_tf = np.full(N_BOOT, np.nan)
        b_alp = np.full(N_BOOT, np.nan)
        b_msd_tot = np.full(N_BOOT, np.nan)

        for bi in range(N_BOOT):
            sel = rng.integers(0, n_g, size=n_g)
            Gv_b, Cv_b, _ = eval_acf_c0norm(Sn_v[sel], Nc_v[sel])
            _, Cu_b, _ = eval_acf_c0norm(Sn_u[sel], Nc_u[sel])
            b_Cv[bi] = Cv_b
            b_Cu60[bi] = Cu_b[15]
            b_Cv60[bi] = Cv_b[15]
            _, _, mt_b, tf_b, alp_b = gk_metrics(Gv_b)
            b_tf[bi] = tf_b
            b_alp[bi] = alp_b
            b_msd_tot[bi] = mt_b

        cv_lo = np.nanpercentile(b_Cv, 2.5, axis=0)
        cv_hi = np.nanpercentile(b_Cv, 97.5, axis=0)
        cv_curves[bname] = (Cv, cv_lo, cv_hi)

        gk_summary.append({
            "bead_name": bname,
            "diameter_um": diam,
            "rc_over_xi": rc_xi,
            "n_cargos": n_g,
            "Cu_60s": float(Cu[15]),
            "Cu_60s_ci_lo": float(np.nanpercentile(b_Cu60, 2.5)),
            "Cu_60s_ci_hi": float(np.nanpercentile(b_Cu60, 97.5)),
            "Cv_4s": float(Cv[1]),
            "Cv_60s": float(Cv[15]),
            "Cv_60s_ci_lo": float(np.nanpercentile(b_Cv60, 2.5)),
            "Cv_60s_ci_hi": float(np.nanpercentile(b_Cv60, 97.5)),
            "Cv_240s": float(Cv[60]),
            "msd_zero_um2": float(msd0),
            "msd_tail_um2": float(msd_t),
            "msd_total_um2": float(msd_tot),
            "msd_total_ci_lo": float(np.nanpercentile(b_msd_tot, 2.5)),
            "msd_total_ci_hi": float(np.nanpercentile(b_msd_tot, 97.5)),
            "tail_fraction": float(tf),
            "tail_fraction_ci_lo": float(np.nanpercentile(b_tf, 2.5)),
            "tail_fraction_ci_hi": float(np.nanpercentile(b_tf, 97.5)),
            "alpha_4_240": float(alp),
            "alpha_ci_lo": float(np.nanpercentile(b_alp, 2.5)),
            "alpha_ci_hi": float(np.nanpercentile(b_alp, 97.5)),
        })

    df_gk = pd.DataFrame(gk_summary)
    save_csv_multi(df_gk, "green_kubo_msd_tail_summary.csv")

    # ------------------------------------------------------------------
    # Figure 2: 3-panel Green-Kubo Decomposition & Inversion vs R_c/xi
    # ------------------------------------------------------------------
    fig2, (axA, axB, axC) = plt.subplots(1, 3, figsize=(16.0, 5.1))
    fig2.subplots_adjust(top=0.82, wspace=0.28)

    # Panel A: Cargo velocity VACF C_v(tau) across 6 sizes
    for bname in ORDER_BEADS:
        meta = BEAD_META[bname]
        Cv, lo, hi = cv_curves[bname]
        n_c = int(df_gk[df_gk["bead_name"] == bname]["n_cargos"].iloc[0])
        axA.plot(
            lags_gk_s, Cv, color=meta["color"], marker=meta["marker"], ms=3.5, markevery=4,
            lw=1.9 if bname in ("beads06um", "beads1um") else 1.5,
            label=f"{meta['label']} (N={n_c})", zorder=5 if bname in ("beads06um", "beads1um") else 3
        )
        axA.fill_between(lags_gk_s, lo, hi, color=meta["color"], alpha=0.12, lw=0)

    axA.axhline(0.0, color="0.25", lw=0.8)
    axA.annotate(
        "Persistent positive tail\n(drives superdiffusion)",
        xy=(64, 0.30), xytext=(95, 0.56),
        arrowprops=dict(arrowstyle="->", color="#882255", lw=1.2),
        fontsize=8.3, color="#882255", fontweight="bold",
    )
    axA.set_xlim(0, 240)
    axA.set_ylim(-0.42, 1.04)
    axA.set_xlabel(r"Lag time $\tau$ [s]")
    axA.set_ylabel(r"Cargo Velocity VACF $C_{\mathbf{v}_c}(\tau)$")
    axA.set_title(r"A. Cargo Velocity Autocorrelation $C_{\mathbf{v}_c}(\tau)$")
    axA.legend(loc="upper right", frameon=True, framealpha=0.94, fontsize=7.8, ncol=1)
    axA.grid(alpha=0.3)

    # Panel B: Stacked Green-Kubo MSD(240 s) decomposition
    xpos = np.arange(len(df_gk))
    xlabels = [f"{format_diam_short(d)}\n({rx:.2f})" for d, rx in zip(df_gk["diameter_um"], df_gk["rc_over_xi"])]
    axB.bar(xpos, df_gk["msd_zero_um2"], width=0.58, color="#95a5a6", edgecolor="black", linewidth=0.5,
            label=r"Uncorrelated $M\langle|\Delta\mathbf{r}|^2\rangle$")
    axB.bar(xpos, df_gk["msd_tail_um2"], bottom=df_gk["msd_zero_um2"], width=0.58, color="#2471a3", edgecolor="black", linewidth=0.5,
            label=r"Correlated tail $2\sum(M-k)\langle\Delta\mathbf{r}_0\cdot\Delta\mathbf{r}_k\rangle$")

    for i, r in df_gk.iterrows():
        top_y = r["msd_total_um2"]
        axB.text(
            i, top_y + 10.0,
            f"{100*r['tail_fraction']:.0f}%\n" + r"$\alpha=$" + f"{r['alpha_4_240']:.2f}",
            ha="center", va="bottom", fontsize=8.0, color="#1b4f72", fontweight="bold"
        )

    axB.set_xticks(xpos)
    axB.set_xticklabels(xlabels, fontsize=8.4)
    axB.set_ylim(0, 510)
    axB.set_xlabel(r"Cargo Diameter [$\mu\mathrm{m}$]  ($R_c/\xi$)")
    axB.set_ylabel(r"$\mathrm{MSD}(240\,\mathrm{s})$ [$\mu\mathrm{m}^2$]")
    axB.set_title(r"B. Green-Kubo Decomposition of $\mathrm{MSD}(240\,\mathrm{s})$")
    axB.legend(loc="upper right", frameon=True, framealpha=0.94, fontsize=8.0)
    axB.grid(axis="y", alpha=0.3)

    # Panel C: Inversion vs R_c/xi: Flow memory C_u(60s) vs Cargo memory C_v(60s) vs Tail Fraction
    x_rc = df_gk["rc_over_xi"].values
    axC.errorbar(
        x_rc, df_gk["Cu_60s"],
        yerr=[np.maximum(0.0, df_gk["Cu_60s"] - df_gk["Cu_60s_ci_lo"]),
              np.maximum(0.0, df_gk["Cu_60s_ci_hi"] - df_gk["Cu_60s"])],
        fmt="--s", color="#1f77b4", lw=1.8, ms=6.5, capsize=3.0,
        label=r"Flow memory $C_{\bar{\mathbf{u}}}(60\,\mathrm{s})$", zorder=4
    )
    axC.errorbar(
        x_rc, df_gk["Cv_60s"],
        yerr=[np.maximum(0.0, df_gk["Cv_60s"] - df_gk["Cv_60s_ci_lo"]),
              np.maximum(0.0, df_gk["Cv_60s_ci_hi"] - df_gk["Cv_60s"])],
        fmt="-o", color="#c0392b", lw=2.0, ms=6.8, capsize=3.0,
        label=r"Cargo memory $C_{\mathbf{v}_c}(60\,\mathrm{s})$", zorder=5
    )
    axC.errorbar(
        x_rc, df_gk["tail_fraction"],
        yerr=[np.maximum(0.0, df_gk["tail_fraction"] - df_gk["tail_fraction_ci_lo"]),
              np.maximum(0.0, df_gk["tail_fraction_ci_hi"] - df_gk["tail_fraction"])],
        fmt="-.^", color="#117733", lw=1.8, ms=6.8, capsize=3.0,
        label=r"$\mathrm{MSD}(240\,\mathrm{s})$ tail fraction", zorder=4
    )

    axC.axvline(X_LOCK, color="0.45", ls="--", lw=1.1, zorder=1)
    axC.axvline(X_BULK, color="0.45", ls=":", lw=1.1, zorder=1)
    axC.axhline(0.0, color="0.25", lw=0.7)
    axC.text(X_LOCK * 0.94, 0.04, r"$x_{\mathrm{lock}}$", fontsize=8.4, color="0.35", ha="right")
    axC.text(X_BULK * 1.05, 0.04, r"$x_{\mathrm{bulk}}$", fontsize=8.4, color="0.35", ha="left")
    axC.set_xscale("log")
    axC.set_xlim(0.045, 3.0)
    axC.set_ylim(-0.10, 1.08)
    axC.set_xlabel(r"$R_c / \xi$")
    axC.set_ylabel("Autocorrelation / Tail Fraction")
    axC.set_title(r"C. Inversion: Flow Memory vs. Transmitted Cargo Tail")
    axC.legend(loc="center left", frameon=True, framealpha=0.94, fontsize=7.9)
    axC.grid(alpha=0.3, which="both")

    fig2.suptitle(r"Why Small Cargos Superdiffuse while Large Cargos Diffuse: Green-Kubo Tail Decomposition ($\Delta t = 4$ s)", fontsize=13.2, y=0.95)
    save_fig_multi(fig2, "green_kubo_msd_tail_decomposition")
    plt.close(fig2)

    print("\n=== Green-Kubo MSD(240s) Summary ===")
    print(df_gk[[
        "bead_name", "diameter_um", "rc_over_xi", "Cu_60s", "Cv_4s", "Cv_60s", "Cv_240s",
        "msd_zero_um2", "msd_tail_um2", "msd_total_um2", "tail_fraction", "alpha_4_240"
    ]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
