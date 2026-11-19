#!/usr/bin/env python3
"""
verify_data_processing_inequality.py

情報理論のデータ処理不等式 (Data Processing Inequality: DPI):
    D_KL(R_c / xi) <= D_MT(R_c / xi)
を実験データで直接検証するスクリプト。

1. 領域内微小管の配向情報量 D_MT(R_c / xi):
   - B = 36 ビンでの円盤内微小管角偏差 delta_theta = theta_k - theta_bulk の分布 P_MT から
     D_MT = sum_k p_MT,k ln(B * p_MT,k) を算出（実験ごとの mean +/- SEM も算出）。
   - local_polar_bg.zarr から得られた背景極性秩序度 Phi_bg(R_c)、および
     plot_bg_angular_correlation.py の指数相関モデル C_bg(r) = exp(-r/xi) に基づく理論曲線
     Phi_theory(R_c/xi) = calc_disk_polar_order(R_c, xi) から 1次モード上限 D_MT^(1) も算出。
2. 畳み込みチャネル上限 D_KL^conv(R_c / xi) = D_KL(P_0 * P_MT || U):
   - 最小粒子応答 P_0 (0.63 um) を通した巡回畳み込みによるタイトな上限も算出。
3. 実測貨物情報量 D_KL(R_c / xi) (kl_divergence_decomposition_summary.csv) と比較し、
   DPI 階層 D_KL <= D_KL^conv <= D_MT および情報伝達効率 eta_info = D_KL / D_MT を評価する。

出力先:
  - figure/per_point_v_dot_u/
  - /mnt/NAS-Ebanaru/sasaki/MTsingleBeads/figure/per_point_v_dot_u/
"""

import time
import pickle
import warnings
from pathlib import Path
from multiprocessing import Pool

import h5py
import xarray as xr
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
KL_CSV = CURRENT_DIR / "figure" / "per_point_v_dot_u" / "kl_divergence_decomposition_summary.csv"
PT_CACHE = CURRENT_DIR / "scratch" / "per_point_v_dot_u_hist_cache.pkl"
MT_B36_CACHE = CURRENT_DIR / "scratch" / "mt_disk_angle_hist_B36_cache.pkl"

B_BINS = 36
BINS_TH = np.linspace(-np.pi, np.pi, B_BINS + 1)
TH_CENTERS = 0.5 * (BINS_TH[:-1] + BINS_TH[1:])
DTH = float(BINS_TH[1] - BINS_TH[0])

SCALE_UM_PER_PX = 0.11
SCALE_S4_UM_PER_PX = SCALE_UM_PER_PX * 4.0


def wrap_angle(ang: np.ndarray) -> np.ndarray:
    return (ang + np.pi) % (2.0 * np.pi) - np.pi


def solve_kappa_from_m1(m1: float) -> float:
    if not np.isfinite(m1) or m1 <= 0.0:
        return 0.0
    if m1 >= 0.9999:
        return 50.0
    return float(brentq(lambda k: (i1(k) / i0(k)) - m1, 0.0, 50.0))


def vm_kl_from_m1(m1: float) -> float:
    k = solve_kappa_from_m1(m1)
    return float(k * m1 - np.log(i0(k)))


def kl_from_hist(hist_counts: np.ndarray) -> float:
    tot = float(hist_counts.sum())
    if tot <= 0:
        return np.nan
    p = hist_counts.astype(float) / tot
    m = p > 0
    return float(np.sum(p[m] * np.log(B_BINS * p[m])))


def _extract_mt_b36_for_bead(b_item: dict):
    bname = b_item["name"]
    r_um = b_item["radius_um"]
    dirs = vcd.find_experiment_dirs(ROOT_DIR, bname)
    use_full_res = r_um < 0.8

    h_tot = np.zeros(B_BINS, dtype=np.int64)
    exp_dmt = []
    exp_phi_zarr = []

    for ed in dirs:
        # 1) local_polar_bg.zarr から実験ごとの Phi_bg(R_c) を取得
        zp = ed / "local_polar_bg.zarr"
        if zp.exists():
            try:
                ds = xr.open_zarr(zp, consolidated=False)
                ws = ds.coords["window size"].values
                idx = int(np.argmin(np.abs(ws - (r_um / SCALE_UM_PER_PX))))
                phi_val = float(ds["polar_order"].isel({"window size": idx}).mean().compute())
                if np.isfinite(phi_val):
                    exp_phi_zarr.append(phi_val)
            except Exception:
                pass

        # 2) B = 36 ビンでの背景微小管円盤ヒストグラム
        h_exp = np.zeros(B_BINS, dtype=np.int64)
        if use_full_res:
            fp = ed / "GFP_flows.h5"
            if not fp.exists():
                continue
            r_px = r_um / SCALE_UM_PER_PX
            r_ceil = int(np.ceil(r_px))
            yg, xg = np.ogrid[-r_ceil : r_ceil + 1, -r_ceil : r_ceil + 1]
            disk_mask = (xg**2 + yg**2) <= r_px**2
            step_xy = max(12, 4 * r_ceil)

            with h5py.File(fp, "r") as hf:
                flow = hf[list(hf.keys())[0]]
                n_tot, _, H, W = flow.shape
                for fr in np.linspace(0, n_tot - 1, 5, dtype=int):
                    arr = flow[int(fr)].astype(np.float32)
                    ux_f, uy_f = arr[0], arr[1]
                    for cy in range(r_ceil, H - r_ceil, step_xy):
                        for cx in range(r_ceil, W - r_ceil, step_xy):
                            ux_d = ux_f[cy - r_ceil : cy + r_ceil + 1, cx - r_ceil : cx + r_ceil + 1][disk_mask]
                            uy_d = uy_f[cy - r_ceil : cy + r_ceil + 1, cx - r_ceil : cx + r_ceil + 1][disk_mask]
                            v = (ux_d != 0) | (uy_d != 0)
                            if np.sum(v) < 5:
                                continue
                            ux_d, uy_d = ux_d[v], uy_d[v]
                            th_bulk = np.arctan2(np.mean(uy_d), np.mean(ux_d))
                            dth_k = wrap_angle(np.arctan2(uy_d, ux_d) - th_bulk)
                            hc, _ = np.histogram(dth_k, bins=BINS_TH)
                            h_exp += hc
        else:
            fp = ed / "mt_flow_cache_s4_f5.h5"
            if not fp.exists():
                continue
            r_px = r_um / SCALE_S4_UM_PER_PX
            r_ceil = int(np.ceil(r_px))
            yg, xg = np.ogrid[-r_ceil : r_ceil + 1, -r_ceil : r_ceil + 1]
            disk_mask = (xg**2 + yg**2) <= r_px**2
            step_xy = max(4, 2 * r_ceil)

            with h5py.File(fp, "r") as hf:
                mx = hf["mx"][::5].astype(np.float32)
                my = hf["my"][::5].astype(np.float32)
            _, H, W = mx.shape
            for fr in range(mx.shape[0]):
                ux_f, uy_f = mx[fr], my[fr]
                for cy in range(r_ceil, H - r_ceil, step_xy):
                    for cx in range(r_ceil, W - r_ceil, step_xy):
                        ux_d = ux_f[cy - r_ceil : cy + r_ceil + 1, cx - r_ceil : cx + r_ceil + 1][disk_mask]
                        uy_d = uy_f[cy - r_ceil : cy + r_ceil + 1, cx - r_ceil : cx + r_ceil + 1][disk_mask]
                        v = (ux_d != 0) | (uy_d != 0)
                        if np.sum(v) < 5:
                            continue
                        ux_d, uy_d = ux_d[v], uy_d[v]
                        th_bulk = np.arctan2(np.mean(uy_d), np.mean(ux_d))
                        dth_k = wrap_angle(np.arctan2(uy_d, ux_d) - th_bulk)
                        hc, _ = np.histogram(dth_k, bins=BINS_TH)
                        h_exp += hc

        h_tot += h_exp
        if h_exp.sum() > 0:
            exp_dmt.append(kl_from_hist(h_exp))

    print(f"  [MT B=36] {bname:10s}: D_MT = {kl_from_hist(h_tot):.4f}, n_exp = {len(exp_dmt)}")
    return bname, {
        "hist_mt": h_tot,
        "exp_dmt": exp_dmt,
        "exp_phi_zarr": exp_phi_zarr,
    }


def load_or_extract_mt_b36(force: bool = False) -> dict:
    if MT_B36_CACHE.exists() and not force:
        print(f"Loading cached MT B=36 histograms from {MT_B36_CACHE}...")
        with open(MT_B36_CACHE, "rb") as f:
            return pickle.load(f)

    t0 = time.time()
    print("Extracting MT orientation distributions (B=36) and local_polar_bg.zarr across 6 conditions...")
    with Pool(processes=6) as pool:
        res_list = pool.map(_extract_mt_b36_for_bead, vcd.BEADS_INFO)
    mt_dict = dict(res_list)
    with open(MT_B36_CACHE, "wb") as f:
        pickle.dump(mt_dict, f)
    print(f"Saved {MT_B36_CACHE} in {time.time() - t0:.1f}s.")
    return mt_dict


def circular_convolve_b36(p0: np.ndarray, p_mt: np.ndarray) -> np.ndarray:
    """
    B = 36 ビン（偶数）の確率密度配列 p0, p_mt の円形畳み込み:
    np.interp でゼロ中心グリッドに厳密に位置合わせして畳み込む。
    """
    # P(theta) on 36 bins: theta_j = -pi + (j + 0.5)*DTH -> shift by -pi (i.e. +pi/2 bins) via fft phase shift
    # 中心 0 の位置は (N-1)/2 = 17.5 にあるため、exp(+i * k * (-pi + 0.5*dth)) の位相補正または N=72 補間を使用
    n_fine = 73
    th_fine = np.linspace(-np.pi, np.pi, n_fine, endpoint=False) + np.pi / n_fine
    c_fine = (n_fine - 1) // 2
    p0_f = np.interp(th_fine, TH_CENTERS, p0, period=2.0 * np.pi)
    pm_f = np.interp(th_fine, TH_CENTERS, p_mt, period=2.0 * np.pi)
    dth_f = 2.0 * np.pi / n_fine
    p0_f /= np.sum(p0_f) * dth_f
    pm_f /= np.sum(pm_f) * dth_f
    p_conv_f = np.fft.irfft(np.fft.rfft(p0_f) * np.fft.rfft(np.roll(pm_f, -c_fine)), n=n_fine) * dth_f
    p_conv = np.interp(TH_CENTERS, th_fine, np.maximum(p_conv_f, 0.0), period=2.0 * np.pi)
    p_conv /= np.sum(p_conv) * DTH
    return p_conv


def main():
    vcd.setup_matplotlib()
    for d in OUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)

    df_kl = pd.read_csv(KL_CSV)
    mt_dict = load_or_extract_mt_b36(force=False)

    # 0.63 um 粒子の P_0(theta_pt) (B=36)
    with open(PT_CACHE, "rb") as f:
        df_pt = pd.DataFrame(pickle.load(f))
    h0_pt = np.vstack(df_pt[df_pt["bead_name"] == "beads06um"]["hist_th"].values).sum(axis=0).astype(float)
    p0_pt = h0_pt / (h0_pt.sum() * DTH)

    rows = []
    for _, r in df_kl.iterrows():
        b = r["bead_name"]
        m_info = mt_dict[b]
        h_mt = m_info["hist_mt"].astype(float)
        p_mt = h_mt / (h_mt.sum() * DTH)

        d_mt_tot = kl_from_hist(h_mt)
        exp_dmt = np.asarray(m_info["exp_dmt"], dtype=float)
        d_mt_sem = float(np.std(exp_dmt, ddof=1) / np.sqrt(len(exp_dmt))) if len(exp_dmt) > 1 else 0.0

        phi_mt_disk = float(np.sum(p_mt * np.cos(TH_CENTERS)) * DTH)
        d_mt_1 = vm_kl_from_m1(phi_mt_disk)

        exp_phi_zarr = np.asarray(m_info["exp_phi_zarr"], dtype=float)
        phi_zarr_mean = float(np.mean(exp_phi_zarr)) if len(exp_phi_zarr) > 0 else np.nan
        d_mt_zarr_1 = vm_kl_from_m1(phi_zarr_mean) if np.isfinite(phi_zarr_mean) else np.nan

        # 畳み込みチャネル上限 D_KL^conv = D_KL(P_0 * P_MT || U)
        p_conv = circular_convolve_b36(p0_pt, p_mt)
        d_conv = float(np.sum(p_conv[p_conv > 0] * DTH * np.log(2.0 * np.pi * p_conv[p_conv > 0])))

        # 理論曲線点 (C_bg(r) = exp(-r/xi) -> Phi_theory(R_c/xi))
        phi_th = vcd.calc_disk_polar_order(r["rc_over_xi"], 1.0)
        d_mt_theory_1 = vm_kl_from_m1(phi_th)

        d_kl_obs = float(r["D_KL_corr"])
        eta_info = float(d_kl_obs / d_mt_tot) if d_mt_tot > 0 else np.nan
        eta_conv = float(d_kl_obs / d_conv) if d_conv > 0 else np.nan

        rows.append({
            "bead_name": b,
            "diameter_um": float(r["diameter_um"]),
            "rc_over_xi": float(r["rc_over_xi"]),
            "rc_over_xi_sem": float(r["rc_over_xi_sem"]),
            "Phi_MT_disk": phi_mt_disk,
            "Phi_bg_zarr": phi_zarr_mean,
            "Phi_theory_exp": phi_th,
            "D_MT_total": d_mt_tot,
            "D_MT_total_sem": d_mt_sem,
            "D_MT_1_disk": d_mt_1,
            "D_MT_1_zarr": d_mt_zarr_1,
            "D_MT_1_theory": d_mt_theory_1,
            "D_KL_conv_bound": d_conv,
            "D_KL_obs": d_kl_obs,
            "D_KL_1_obs": float(r["D_KL_1_corr"]),
            "Delta_D_high_obs": float(r["Delta_D_high_corr"]),
            "DPI_satisfied": bool(d_kl_obs <= d_mt_tot),
            "info_efficiency_D_KL_over_D_MT": eta_info,
            "conv_efficiency_D_KL_over_D_conv": eta_conv,
        })

    df_dpi = pd.DataFrame(rows).sort_values("rc_over_xi").reset_index(drop=True)
    for d in OUT_DIRS:
        df_dpi.to_csv(d / "data_processing_inequality_summary.csv", index=False)

    print("\n=== Data Processing Inequality Verification (D_KL <= D_MT) ===")
    cols = [
        "bead_name", "diameter_um", "rc_over_xi", "Phi_MT_disk", "Phi_bg_zarr",
        "D_MT_total", "D_MT_1_zarr", "D_KL_conv_bound", "D_KL_obs", "DPI_satisfied", "info_efficiency_D_KL_over_D_MT"
    ]
    print(df_dpi[cols].to_string(index=False))

    # 理論連続曲線 (C_bg(r) = exp(-r/xi) -> Phi_theory -> D_MT^(1))
    rc_grid = np.logspace(np.log10(0.04), np.log10(3.2), 120)
    d_mt_theory_curve = np.array([vm_kl_from_m1(vcd.calc_disk_polar_order(x, 1.0)) for x in rc_grid])

    # 全情報量 D_MT の補間・外挿曲線 (DPI の厳密な上限境界用)
    log_rc_obs = np.log10(df_dpi["rc_over_xi"].values)
    log_dmt_obs = np.log10(df_dpi["D_MT_total"].values)
    log_rc_grid = np.log10(rc_grid)
    # 端点は線形外挿、内部は線形補間
    d_mt_upper_curve = 10.0 ** np.interp(log_rc_grid, log_rc_obs, log_dmt_obs)
    # 左右端の外挿
    left_mask = log_rc_grid < log_rc_obs[0]
    if np.any(left_mask):
        s_l = (log_dmt_obs[1] - log_dmt_obs[0]) / (log_rc_obs[1] - log_rc_obs[0])
        d_mt_upper_curve[left_mask] = 10.0 ** (log_dmt_obs[0] + s_l * (log_rc_grid[left_mask] - log_rc_obs[0]))
    right_mask = log_rc_grid > log_rc_obs[-1]
    if np.any(right_mask):
        s_r = (log_dmt_obs[-1] - log_dmt_obs[-2]) / (log_rc_obs[-1] - log_rc_obs[-2])
        d_mt_upper_curve[right_mask] = 10.0 ** (log_dmt_obs[-1] + s_r * (log_rc_grid[right_mask] - log_rc_obs[-1]))

    # --- Figure: 2パネル Data Processing Inequality ---
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14.0, 5.8))

    # (A) Log-Log DPI Hierarchy: D_KL(obs) <= D_KL^conv <= D_MT(total)
    axL.fill_between(rc_grid, d_mt_upper_curve, 10.0, color="#ffcccc", alpha=0.22,
                     label=r"Forbidden by DPI ($D_{\mathrm{KL}} > D_{\mathrm{MT}}$)")
    axL.errorbar(df_dpi["rc_over_xi"], df_dpi["D_MT_total"], yerr=df_dpi["D_MT_total_sem"],
                 xerr=df_dpi["rc_over_xi_sem"], fmt="o-", color="#117733", linewidth=2.3, markersize=8.5,
                 markeredgecolor="k", markeredgewidth=0.7, capsize=4,
                 label=r"$D_{\mathrm{MT}}$ (Total MT footprint KL, $B=36$)", zorder=4)
    axL.plot(df_dpi["rc_over_xi"], df_dpi["D_MT_1_zarr"], "^--", color="#44AA99", linewidth=2.0, markersize=7.8,
             markeredgecolor="k", markeredgewidth=0.6,
             label=r"$D_{\mathrm{MT}}^{(1)}$ (from background MT polar order $\Phi_{\mathrm{bg}}$)", zorder=3)
    axL.plot(rc_grid, d_mt_theory_curve, color="#117733", linestyle=":", linewidth=1.7, alpha=0.85,
             label=r"Theory $D_{\mathrm{MT}}^{(1)}$ from $C_{\mathrm{bg}}(r)=e^{-r/\xi}$", zorder=2)

    # 0.63 um を基準核 P_0 とし、R_c > 0.63 um (1.2 um 以降) に対するチャネル上限予測線を描画
    df_conv_pred = df_dpi.iloc[1:]
    axL.plot(df_conv_pred["rc_over_xi"], df_conv_pred["D_KL_conv_bound"], "D-.", color="#555555", linewidth=2.0, markersize=7.0,
             label=r"Channel bound $D_{\mathrm{KL}}(P_0 * P_{\mathrm{MT}})$ ($R_c > 0.63\,\mu\mathrm{m}$)", zorder=4)

    axL.plot(df_dpi["rc_over_xi"], df_dpi["D_KL_obs"], "s-", color="#882255", linewidth=2.5, markersize=8.8,
             markeredgecolor="k", markeredgewidth=0.8,
             label=r"Observed cargo $D_{\mathrm{KL}}(R_c/\xi)$", zorder=6)

    for _, r in df_dpi.iterrows():
        axL.annotate(f"{r['diameter_um']:.2g} $\\mu\\mathrm{{m}}$",
                     xy=(r["rc_over_xi"], r["D_KL_obs"]),
                     xytext=(0, -14), textcoords="offset points", ha="center", fontsize=8.8, color="#882255", fontweight="bold")

    axL.axvline(1.0, color="black", linestyle="-.", linewidth=1.2, alpha=0.65, label=r"$R_c = \xi$")
    axL.set_xscale("log")
    axL.set_yscale("log")
    axL.set_xlim(0.04, 3.2)
    axL.set_ylim(1e-3, 6.0)
    axL.set_xlabel(r"Dimensionless Cargo Radius $R_c / \xi$", fontsize=13)
    axL.set_ylabel(r"Orientation Information / KL Divergence [nats]", fontsize=12.5)
    axL.set_title(r"(A) Data Processing Inequality: $D_{\mathrm{KL}} \leq D_{\mathrm{KL}}^{\mathrm{conv}} \leq D_{\mathrm{MT}}$",
                  fontsize=13, fontweight="bold")
    axL.grid(True, which="both", linestyle=":", alpha=0.5)
    axL.legend(loc="lower left", fontsize=8.2, framealpha=0.92)

    # (B) Information Transmission Efficiency: D_KL / D_MT & D_KL / D_conv
    # 0.63 um を基準点 (100%) として扱う
    conv_eff_pct = df_dpi["conv_efficiency_D_KL_over_D_conv"].values * 100.0
    conv_eff_pct_ref = conv_eff_pct.copy()
    conv_eff_pct_ref[0] = 100.0

    axR.plot(df_dpi["rc_over_xi"], df_dpi["info_efficiency_D_KL_over_D_MT"] * 100.0, "s-", color="#882255",
             linewidth=2.3, markersize=8.5, markeredgecolor="k", markeredgewidth=0.7,
             label=r"Raw transfer efficiency $D_{\mathrm{KL}} / D_{\mathrm{MT}}$ [\%]")
    axR.axvline(1.0, color="black", linestyle="-.", linewidth=1.2, alpha=0.65, label=r"$R_c = \xi$")

    axR_twin = axR.twinx()
    axR_twin.plot(df_dpi["rc_over_xi"], conv_eff_pct_ref, "o--", color="#1f77b4",
                  linewidth=2.1, markersize=7.8, markeredgecolor="k", markeredgewidth=0.6,
                  label=r"Channel efficiency $D_{\mathrm{KL}} / D_{\mathrm{KL}}^{\mathrm{conv}}$ [\%] (ref. $0.63\,\mu\mathrm{m}=100\%$)")

    for _, r in df_dpi.iterrows():
        axR.annotate(f"{r['diameter_um']:.2g} $\\mu\\mathrm{{m}}$",
                     xy=(r["rc_over_xi"], r["info_efficiency_D_KL_over_D_MT"] * 100.0),
                     xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8.8, color="#882255", fontweight="bold")

    axR.set_xscale("log")
    axR.set_xlim(0.04, 3.2)
    axR.set_ylim(0.0, 7.5)
    axR_twin.set_ylim(0.0, 112.0)
    axR.set_xlabel(r"Dimensionless Cargo Radius $R_c / \xi$", fontsize=13)
    axR.set_ylabel(r"Raw Info Efficiency $D_{\mathrm{KL}} / D_{\mathrm{MT}}$ [\%]", fontsize=12.5, color="#882255")
    axR_twin.set_ylabel(r"Channel Efficiency $D_{\mathrm{KL}} / D_{\mathrm{KL}}^{\mathrm{conv}}$ [\%]", fontsize=12.5, color="#1f77b4")
    axR.set_title(r"(B) Efficiency of Reading MT Orientation Information", fontsize=13, fontweight="bold")
    axR.grid(True, which="both", linestyle=":", alpha=0.5)

    lines_r = axR.get_lines() + axR_twin.get_lines()
    labels_r = [l.get_label() for l in lines_r]
    axR.legend(lines_r, labels_r, loc="upper right", fontsize=8.8, framealpha=0.92)

    fig.suptitle(r"Experimental Test of the Data Processing Inequality: $D_{\mathrm{KL}}(R_c/\xi) \leq D_{\mathrm{MT}}(R_c/\xi)$",
                 fontsize=14.8, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    vcd.save_figure(fig, "data_processing_inequality_vs_rc_xi", OUT_DIRS)
    plt.close(fig)

    print("Saved Data Processing Inequality figure and summary CSV successfully!")


if __name__ == "__main__":
    main()
