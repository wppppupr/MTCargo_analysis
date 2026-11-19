#!/usr/bin/env python3
"""
calc_and_plot_kl_decomposition.py

角度空間 [-pi, pi] での離散確率分布 p_k (B = 36 ビン, Δtheta = 2*pi / B) を用い、
全相対エントロピー (KLダイバージェンス):
    D_KL = sum_k p_k ln(B * p_k)
を、1次モーメント m_1 = <cos theta_pt> から決まるフォン・ミーゼス分布 (I_1(kappa_1)/I_0(kappa_1) = m_1):
    D_KL^(1) = kappa_1 * m_1 - ln I_0(kappa_1)
と、高次モード（局所レールロック・非vM成分）:
    Delta D_high = D_KL - D_KL^(1)
に分解する。

さらに有限サンプルバイアスを除去するため、各サイズで実際の背景微小管円盤パッチと
シャッフルした貨物角を組み合わせた擬似データから D_KL^shuf を算出して差し引き、
粒子単位クラスターブートストラップ (B_boot = 1000) により 95% CI を評価する。

出力先:
  - figure/per_point_v_dot_u/
  - /mnt/NAS-Ebanaru/sasaki/MTsingleBeads/figure/per_point_v_dot_u/
"""

import pickle
import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.special import i0, i1

import verify_cos_decomposition as vcd

warnings.filterwarnings("ignore")

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = vcd.ROOT_DIR
OUT_DIRS = [
    CURRENT_DIR / "figure" / "per_point_v_dot_u",
    ROOT_DIR / "figure" / "per_point_v_dot_u",
]
HIST_CACHE = CURRENT_DIR / "scratch" / "per_point_v_dot_u_hist_cache.pkl"
INST_CACHE = CURRENT_DIR / "scratch" / "contrast_inst_cache.pkl"
SHUF_CACHE = CURRENT_DIR / "scratch" / "kl_shuffle_bias_cache.pkl"

B_BINS = 36
BINS_TH = np.linspace(-np.pi, np.pi, B_BINS + 1)
SCALE_S4 = 0.44
RAD_UM = {
    "beads06um": 0.315,
    "beads1um": 0.59,
    "beads3um": 1.685,
    "beads5um": 2.50,
    "beads7um": 3.62,
    "beads20um": 10.0,
}


def solve_kappa1_from_m1(m1: float) -> float:
    if not np.isfinite(m1) or m1 <= 0.0:
        return 0.0
    if m1 >= 0.9999:
        return 50.0
    return float(brentq(lambda k: (i1(k) / i0(k)) - m1, 0.0, 50.0))


def compute_kl_terms(hist_counts: np.ndarray, sum_c: float, n_pts: float):
    """カウント配列と sum(cos theta) から D_KL, D_KL^(1), Delta D_high, kappa_1 を算出"""
    if n_pts <= 0:
        return None
    p = hist_counts.astype(float) / float(n_pts)
    mask = p > 0
    d_kl = float(np.sum(p[mask] * np.log(B_BINS * p[mask])))
    m1 = float(sum_c / float(n_pts))
    k1 = solve_kappa1_from_m1(m1)
    d_kl1 = float(k1 * m1 - np.log(i0(k1)))
    d_high = float(d_kl - d_kl1)
    return {
        "m1": m1,
        "kappa_1": k1,
        "D_KL": d_kl,
        "D_KL_1": d_kl1,
        "Delta_D_high": d_high,
    }


def load_or_compute_shuffle_bias(n_repeats: int = 50, force: bool = False) -> dict:
    if SHUF_CACHE.exists() and not force:
        print(f"Loading cached shuffle bias from {SHUF_CACHE}...")
        with open(SHUF_CACHE, "rb") as f:
            return pickle.load(f)

    print("Computing finite-sample shuffle bias D_KL^shuf from disk flow patches...")
    df_inst = pd.read_pickle(INST_CACHE)
    df_inst["th_v"] = np.arctan2(df_inst["cargo_dy"], df_inst["cargo_dx"])
    rng = np.random.default_rng(42)

    shuf_map = {}
    for b_item in vcd.BEADS_INFO:
        b = b_item["name"]
        sub_c = df_inst[df_inst["bead_name"] == b]
        th_v_pool = sub_c["th_v"].values
        n_frms = len(sub_c)

        r_px = max(1.0, RAD_UM[b] / SCALE_S4)
        r_ceil = int(np.ceil(r_px))
        yg, xg = np.ogrid[-r_ceil : r_ceil + 1, -r_ceil : r_ceil + 1]
        mask = (xg**2 + yg**2) <= r_px**2

        disk_angles = []
        for ed in vcd.find_experiment_dirs(ROOT_DIR, b):
            fp = ed / "mt_flow_cache_s4_f5.h5"
            if not fp.exists():
                continue
            with h5py.File(fp, "r") as hf:
                mx = hf["mx"][::20].astype(np.float32)
                my = hf["my"][::20].astype(np.float32)
            _, H, W = mx.shape
            step = max(12, 3 * r_ceil)
            for fr in range(mx.shape[0]):
                for cy in range(r_ceil, H - r_ceil, step):
                    for cx in range(r_ceil, W - r_ceil, step):
                        ux_d = mx[fr, cy - r_ceil : cy + r_ceil + 1, cx - r_ceil : cx + r_ceil + 1][mask]
                        uy_d = my[fr, cy - r_ceil : cy + r_ceil + 1, cx - r_ceil : cx + r_ceil + 1][mask]
                        v = (ux_d != 0) | (uy_d != 0)
                        if np.any(v):
                            disk_angles.append(np.arctan2(uy_d[v], ux_d[v]))

        shuf_dkls, shuf_d1s, shuf_dhs = [], [], []
        for _ in range(n_repeats):
            idx_d = rng.integers(0, len(disk_angles), size=n_frms)
            tv_s = rng.choice(th_v_pool, size=n_frms, replace=True)
            h_tot = np.zeros(B_BINS, dtype=np.int64)
            sum_c_s = 0.0
            n_tot_s = 0
            for j, di in enumerate(idx_d):
                dth = (tv_s[j] - disk_angles[di] + np.pi) % (2.0 * np.pi) - np.pi
                hc, _ = np.histogram(dth, bins=BINS_TH)
                h_tot += hc
                sum_c_s += float(np.sum(np.cos(dth)))
                n_tot_s += len(dth)
            res_s = compute_kl_terms(h_tot, sum_c_s, n_tot_s)
            shuf_dkls.append(res_s["D_KL"])
            shuf_d1s.append(res_s["D_KL_1"])
            shuf_dhs.append(res_s["Delta_D_high"])

        shuf_map[b] = {
            "D_KL_shuf": float(np.mean(shuf_dkls)),
            "D_KL_shuf_std": float(np.std(shuf_dkls)),
            "D_KL_1_shuf": float(np.mean(shuf_d1s)),
            "Delta_D_high_shuf": float(np.mean(shuf_dhs)),
            "theory_bias_frm": float((B_BINS - 1) / (2.0 * n_frms)),
        }
        print(f"  {b:10s}: D_KL^shuf = {shuf_map[b]['D_KL_shuf']:.5f} +/- {shuf_map[b]['D_KL_shuf_std']:.5f}")

    with open(SHUF_CACHE, "wb") as f:
        pickle.dump(shuf_map, f)
    return shuf_map


def main():
    vcd.setup_matplotlib()
    for d in OUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)

    with open(HIST_CACHE, "rb") as f:
        df_rec = pd.DataFrame(pickle.load(f))
    shuf_map = load_or_compute_shuffle_bias(n_repeats=50, force=False)

    B_BOOT = 1000
    rng = np.random.default_rng(42)
    rows = []

    for b_item in vcd.BEADS_INFO:
        bname = b_item["name"]
        sub = df_rec[df_rec["bead_name"] == bname]
        if sub.empty:
            continue

        n_p = len(sub)
        rc_xi = float(sub["rc_over_xi"].mean())
        rc_xi_sem = float(sub["rc_over_xi"].sem()) if n_p > 1 else 0.0

        ht_arr = np.vstack(sub["hist_th"].values).astype(float)
        sumc_arr = sub["sum_c"].values.astype(float)
        npts_arr = sub["n_pts"].values.astype(float)

        obs = compute_kl_terms(ht_arr.sum(axis=0), sumc_arr.sum(), npts_arr.sum())
        sh = shuf_map[bname]

        # バイアス除去済み KL 指標
        d_kl_corr = max(0.0, obs["D_KL"] - sh["D_KL_shuf"])
        d_kl1_corr = max(0.0, obs["D_KL_1"] - sh["D_KL_1_shuf"])
        d_high_corr = max(0.0, obs["Delta_D_high"] - sh["Delta_D_high_shuf"])
        # 総和整合: d_kl_tot_corr = d_kl1_corr + d_high_corr
        d_kl_tot_corr = d_kl1_corr + d_high_corr
        high_frac_raw = float(obs["Delta_D_high"] / obs["D_KL"]) if obs["D_KL"] > 0 else 0.0
        high_frac_corr = float(d_high_corr / d_kl_tot_corr) if d_kl_tot_corr > 0 else 0.0

        # クラスターブートストラップ (粒子単位)
        boot_d1, boot_dh, boot_dtot, boot_frac = [], [], [], []
        for _ in range(B_BOOT):
            idx = rng.integers(0, n_p, size=n_p)
            rb = compute_kl_terms(ht_arr[idx].sum(axis=0), sumc_arr[idx].sum(), npts_arr[idx].sum())
            if rb is None:
                continue
            b_d1 = max(0.0, rb["D_KL_1"] - sh["D_KL_1_shuf"])
            b_dh = max(0.0, rb["Delta_D_high"] - sh["Delta_D_high_shuf"])
            b_dt = b_d1 + b_dh
            boot_d1.append(b_d1)
            boot_dh.append(b_dh)
            boot_dtot.append(b_dt)
            boot_frac.append(b_dh / b_dt if b_dt > 0 else 0.0)

        rows.append({
            "bead_name": bname,
            "diameter_um": b_item["diameter_um"],
            "label": b_item["label"],
            "color": b_item["color"],
            "marker": b_item["marker"],
            "rc_over_xi": rc_xi,
            "rc_over_xi_sem": rc_xi_sem,
            "n_particles": n_p,
            "n_flow_points": int(npts_arr.sum()),
            "m1_cos": obs["m1"],
            "kappa_1": obs["kappa_1"],
            "D_KL_raw": obs["D_KL"],
            "D_KL_1_raw": obs["D_KL_1"],
            "Delta_D_high_raw": obs["Delta_D_high"],
            "high_frac_raw": high_frac_raw,
            "D_KL_shuf": sh["D_KL_shuf"],
            "D_KL_corr": d_kl_tot_corr,
            "D_KL_1_corr": d_kl1_corr,
            "D_KL_1_ci_low": float(np.percentile(boot_d1, 2.5)),
            "D_KL_1_ci_high": float(np.percentile(boot_d1, 97.5)),
            "Delta_D_high_corr": d_high_corr,
            "Delta_D_high_ci_low": float(np.percentile(boot_dh, 2.5)),
            "Delta_D_high_ci_high": float(np.percentile(boot_dh, 97.5)),
            "high_frac_corr": high_frac_corr,
            "high_frac_ci_low": float(np.percentile(boot_frac, 2.5)),
            "high_frac_ci_high": float(np.percentile(boot_frac, 97.5)),
        })

    df_out = pd.DataFrame(rows).sort_values("rc_over_xi").reset_index(drop=True)
    for d in OUT_DIRS:
        df_out.drop(columns=["color", "marker"]).to_csv(d / "kl_divergence_decomposition_summary.csv", index=False)

    print("\n=== KL Divergence Decomposition Summary (B = 36, Shuffle-Debiased) ===")
    cols = ["bead_name", "diameter_um", "rc_over_xi", "m1_cos", "kappa_1",
            "D_KL_corr", "D_KL_1_corr", "Delta_D_high_corr", "high_frac_corr", "D_KL_shuf"]
    print(df_out[cols].to_string(index=False))

    # --- Figure: 1枚の統合図 (または2パネル: 情報量成分 + 高次寄与率) ---
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14.0, 5.6))

    # Left panel: D_KL^(1) (1st mode) & Delta D_high (higher-order modes) & total D_KL
    axL.plot(df_out["rc_over_xi"], df_out["D_KL_corr"], "o:", color="#444444", linewidth=1.8, markersize=6.5,
             label=r"Total $D_{\mathrm{KL}}$ (bias-corrected)", zorder=2)
    axL.plot(df_out["rc_over_xi"], df_out["D_KL_1_corr"], "s-", color="#1f77b4", linewidth=2.3, markersize=8.5,
             markeredgecolor="k", markeredgewidth=0.7,
             label=r"1st mode $D_{\mathrm{KL}}^{(1)}$ (Bulk propulsion)", zorder=4)
    axL.plot(df_out["rc_over_xi"], df_out["Delta_D_high_corr"], "^-", color="#d62728", linewidth=2.3, markersize=8.8,
             markeredgecolor="k", markeredgewidth=0.7,
             label=r"Higher modes $\Delta D_{\mathrm{high}}$ (Local rail lock)", zorder=5)

    for _, r in df_out.iterrows():
        yerr1 = [[r["D_KL_1_corr"] - r["D_KL_1_ci_low"]], [r["D_KL_1_ci_high"] - r["D_KL_1_corr"]]]
        yerrh = [[r["Delta_D_high_corr"] - r["Delta_D_high_ci_low"]], [r["Delta_D_high_ci_high"] - r["Delta_D_high_corr"]]]
        axL.errorbar(r["rc_over_xi"], r["D_KL_1_corr"], yerr=yerr1, xerr=r["rc_over_xi_sem"],
                     fmt="none", ecolor="#1f77b4", capsize=4, elinewidth=1.5, zorder=3)
        axL.errorbar(r["rc_over_xi"], r["Delta_D_high_corr"], yerr=yerrh, xerr=r["rc_over_xi_sem"],
                     fmt="none", ecolor="#d62728", capsize=4, elinewidth=1.5, zorder=3)
        axL.annotate(f"{r['diameter_um']:.2g} $\\mu\\mathrm{{m}}$",
                     xy=(r["rc_over_xi"], r["D_KL_corr"]),
                     xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8.8, fontweight="bold")

    X_LOCK = 45.0 * np.pi / 512.0  # ~0.28 (Higher-order collapse)
    X_BULK = 45.0 * np.pi / 128.0  # ~1.10 (Bulk alignment decay)

    axL.axvline(X_LOCK, color="#d95f02", linestyle="--", linewidth=1.6, alpha=0.85,
                label=r"$x_{\mathrm{lock}} = \frac{45\pi}{512} \approx 0.28$ (Higher-order collapse)")
    axL.axvline(X_BULK, color="#222222", linestyle=":", linewidth=1.7, alpha=0.85,
                label=r"$x_{\mathrm{bulk}} = \frac{45\pi}{128} \approx 1.10$ (Bulk alignment decay)")
    axL.axhline(0.0, color="#888888", linestyle="-", linewidth=0.9)
    axL.set_xscale("log")
    axL.set_xlim(0.04, 3.0)
    axL.set_ylim(-0.01, 0.22)
    axL.set_xlabel(r"Dimensionless Cargo Radius $R_c / \xi$", fontsize=13)
    axL.set_ylabel(r"KL Divergence [nats]", fontsize=13)
    axL.set_title(r"(A) Information Decomposition: $D_{\mathrm{KL}} = D_{\mathrm{KL}}^{(1)} + \Delta D_{\mathrm{high}}$",
                  fontsize=13, fontweight="bold")
    axL.grid(True, which="both", linestyle=":", alpha=0.5)
    axL.legend(loc="upper right", fontsize=8.3, framealpha=0.92)

    # Right panel: Higher-order contribution fraction Delta D_high / D_KL (and normalized D_1, D_high)
    # Note: for 20 um where D_KL -> 0 (~0.002 nats), show both raw fraction and noise-floor shade
    axR.plot(df_out["rc_over_xi"], df_out["high_frac_corr"] * 100.0, "D-", color="#882255",
             linewidth=2.3, markersize=8.5, markeredgecolor="k", markeredgewidth=0.7,
             label=r"Higher-order fraction $\Delta D_{\mathrm{high}} / D_{\mathrm{KL}}$ [\%]", zorder=4)
    for _, r in df_out.iterrows():
        yerr_f = [[(r["high_frac_corr"] - r["high_frac_ci_low"]) * 100.0],
                  [(r["high_frac_ci_high"] - r["high_frac_corr"]) * 100.0]]
        axR.errorbar(r["rc_over_xi"], r["high_frac_corr"] * 100.0, yerr=yerr_f, xerr=r["rc_over_xi_sem"],
                     fmt="none", ecolor="#882255", capsize=4, elinewidth=1.5, zorder=3)
        axR.annotate(f"{r['diameter_um']:.2g} $\\mu\\mathrm{{m}}$",
                     xy=(r["rc_over_xi"], r["high_frac_corr"] * 100.0),
                     xytext=(0, 10), textcoords="offset points", ha="center", fontsize=8.8, color="#882255", fontweight="bold")

    # 20 um 近傍のゼロ信号ノイズ限界注記 (D_KL < 0.005 nats)
    axR.axvspan(1.2, 3.0, color="#dddddd", alpha=0.45, label=r"Near-isotropic limit ($D_{\mathrm{KL}} < 0.003$ nats)")
    axR.axvline(X_LOCK, color="#d95f02", linestyle="--", linewidth=1.6, alpha=0.85,
                label=r"$x_{\mathrm{lock}} = \frac{45\pi}{512} \approx 0.28$ (Higher-order collapse)")
    axR.axvline(X_BULK, color="#222222", linestyle=":", linewidth=1.7, alpha=0.85,
                label=r"$x_{\mathrm{bulk}} = \frac{45\pi}{128} \approx 1.10$ (Bulk alignment decay)")
    axR.set_xscale("log")
    axR.set_xlim(0.04, 3.0)
    axR.set_ylim(-5.0, 105.0)
    axR.set_xlabel(r"Dimensionless Cargo Radius $R_c / \xi$", fontsize=13)
    axR.set_ylabel(r"Higher-Order Contribution $\Delta D_{\mathrm{high}} / D_{\mathrm{KL}}$ [\%]", fontsize=12.5)
    axR.set_title(r"(B) Higher-Order Share $\Delta D_{\mathrm{high}} / D_{\mathrm{KL}}$ vs $R_c/\xi$",
                  fontsize=13, fontweight="bold")
    axR.grid(True, which="both", linestyle=":", alpha=0.5)
    axR.legend(loc="upper left", fontsize=8.3, framealpha=0.92)

    fig.suptitle(r"KL Divergence Decomposition of $P(\theta_{\mathrm{pt}})$ ($B=36$, Shuffle-Debiased, 95% Cluster Bootstrap CI)",
                 fontsize=14.5, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    vcd.save_figure(fig, "kl_divergence_decomposition_vs_rc_xi", OUT_DIRS)
    plt.close(fig)

    # --- 1枚の単一パネル統合図 (ご要望の「1枚のグラフに D_KL^(1), Delta D_high, 高次寄与率」を2軸で描画) ---
    fig, ax1 = plt.subplots(figsize=(8.6, 5.8))
    ax2 = ax1.twinx()

    l1, = ax1.plot(df_out["rc_over_xi"], df_out["D_KL_1_corr"], "s-", color="#1f77b4",
                   linewidth=2.3, markersize=8.5, markeredgecolor="k", markeredgewidth=0.7,
                   label=r"$D_{\mathrm{KL}}^{(1)}$ (1st mode: bulk propulsion)")
    l2, = ax1.plot(df_out["rc_over_xi"], df_out["Delta_D_high_corr"], "^-", color="#d62728",
                   linewidth=2.3, markersize=8.8, markeredgecolor="k", markeredgewidth=0.7,
                   label=r"$\Delta D_{\mathrm{high}}$ (higher modes: rail lock)")
    l3, = ax2.plot(df_out["rc_over_xi"], df_out["high_frac_corr"] * 100.0, "D--", color="#882255",
                   linewidth=2.0, markersize=7.5, markeredgecolor="k", markeredgewidth=0.6, alpha=0.85,
                   label=r"$\Delta D_{\mathrm{high}} / D_{\mathrm{KL}}$ [\%] (right axis)")

    for _, r in df_out.iterrows():
        yerr1 = [[r["D_KL_1_corr"] - r["D_KL_1_ci_low"]], [r["D_KL_1_ci_high"] - r["D_KL_1_corr"]]]
        yerrh = [[r["Delta_D_high_corr"] - r["Delta_D_high_ci_low"]], [r["Delta_D_high_ci_high"] - r["Delta_D_high_corr"]]]
        ax1.errorbar(r["rc_over_xi"], r["D_KL_1_corr"], yerr=yerr1, fmt="none", ecolor="#1f77b4", capsize=3.5, elinewidth=1.4)
        ax1.errorbar(r["rc_over_xi"], r["Delta_D_high_corr"], yerr=yerrh, fmt="none", ecolor="#d62728", capsize=3.5, elinewidth=1.4)

    l_lock = ax1.axvline(X_LOCK, color="#d95f02", linestyle="--", linewidth=1.6, alpha=0.85,
                         label=r"$x_{\mathrm{lock}} = \frac{45\pi}{512} \approx 0.28$ (Higher-order collapse)")
    l_bulk = ax1.axvline(X_BULK, color="#222222", linestyle=":", linewidth=1.7, alpha=0.85,
                         label=r"$x_{\mathrm{bulk}} = \frac{45\pi}{128} \approx 1.10$ (Bulk alignment decay)")
    ax1.set_xscale("log")
    ax1.set_xlim(0.04, 3.0)
    ax1.set_ylim(-0.005, 0.18)
    ax2.set_ylim(-5.0, 105.0)
    ax1.set_xlabel(r"Dimensionless Cargo Radius $R_c / \xi$", fontsize=13)
    ax1.set_ylabel(r"KL Divergence $D_{\mathrm{KL}}^{(1)},\ \Delta D_{\mathrm{high}}$ [nats]", fontsize=12.5)
    ax2.set_ylabel(r"Higher-Order Share $\Delta D_{\mathrm{high}} / D_{\mathrm{KL}}$ [\%]", fontsize=12.5, color="#882255")
    ax1.set_title(r"Two-Stage Transition via KL Divergence Decomposition ($B=36$, Shuffle-Debiased)",
                  fontsize=13, fontweight="bold")
    ax1.grid(True, which="both", linestyle=":", alpha=0.5)

    lines = [l1, l2, l3, l_lock, l_bulk]
    labels = [ln.get_label() for ln in lines]
    ax1.legend(lines, labels, loc="upper center", fontsize=8.4, framealpha=0.92)
    fig.tight_layout()
    vcd.save_figure(fig, "kl_divergence_decomposition_1panel_dual_axis", OUT_DIRS)
    plt.close(fig)

    print("Saved KL divergence decomposition figures and CSV successfully!")


if __name__ == "__main__":
    main()
