#!/usr/bin/env python3
"""
run_cleaned_1um_full_analysis.py

beads1um から異常データ 2 群:
  1. beads1um001_crop の particle == 7 (オプティカルフロー異常スパイク)
  2. beads1um_crop 全4粒子 (流れと完全無相関・相関長外れ値動画)
を除外して、全6サイズおよび beads1um Before/After（＋GMM Run/Tumble）について
  - MSD (4..300 s) とべき指数 alpha
  - 背景空間配向相関 C(r) と相関長 xi
  - 各pxの配向確率密度 P(c)、有効ポテンシャル U_tilde_eff(c)、GvM 2モーメント理論線、脱出障壁深さ
  - ピタゴラス3項 KL ダイバージェンス分解 (D_KL^(1), D_KL^(2), Delta D_>=3)
  - 速度伝達効率 (eta_reg, eta_par, <cos Theta_bulk>, 速さ保持比)
を一貫再解析する。既存キャッシュ・既存図は一切変更しない。

出力先:
  - figure/cleaned_1um_analysis/
  - ROOT_DIR/figure/cleaned_1um_analysis/
"""

import pickle
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import i0

warnings.filterwarnings("ignore")

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))

import verify_cos_decomposition as vcd
from plot_gvm_potential import solve_gvm_from_moments, BINS_COS, compute_log_z_gvm
from calc_and_plot_kl_decomposition import solve_kappa1_from_m1
from quantify_effective_potential_depth import empirical_depth_from_hist_th, gvm_barrier_metrics
from analyze_gmm_run_tumble_potential_and_kl import build_gmm_labeled_frames

ROOT_DIR = vcd.ROOT_DIR
INST_CACHE = CURRENT_DIR / "scratch" / "contrast_inst_cache.pkl"
PP_CACHE = CURRENT_DIR / "scratch" / "per_point_v_dot_u_hist_cache.pkl"
GMM_HIST_CACHE = CURRENT_DIR / "scratch" / "gmm_run_tumble_per_point_hist_cache.pkl"
BG_POINTS_CSV = CURRENT_DIR / "figure" / "bg_angular_correlation" / "bg_angular_correlation_points.csv"
BG_XI_EXP_CSV = CURRENT_DIR / "figure" / "bg_angular_correlation" / "bg_angular_correlation_length_per_experiment.csv"

OUT_DIRS = [CURRENT_DIR / "figure" / "cleaned_1um_analysis"]
if ROOT_DIR.exists() and ROOT_DIR != CURRENT_DIR:
    OUT_DIRS.append(ROOT_DIR / "figure" / "cleaned_1um_analysis")

ORDER_BEADS = ["beads06um", "beads1um", "beads3um", "beads7um", "beads5um", "beads20um"]
BEAD_META = {b["name"]: b for b in vcd.BEADS_INFO}

SCALE = vcd.SCALE_UM_PER_PX  # 0.11 um/px
DT = 4.0
MAX_MSD_LAG = 75   # 300 s
MIN_PAIRS = 30
N_BOOT = 1000
BINS_TH_N = 36

X_LOCK = 45.0 * np.pi / 512.0   # ~0.276
X_BULK = 45.0 * np.pi / 128.0   # ~1.104

C_CENTERS = 0.5 * (BINS_COS[:-1] + BINS_COS[1:])
DC = BINS_COS[1] - BINS_COS[0]
DTH_EXACT = np.arccos(BINS_COS[:-1]) - np.arccos(BINS_COS[1:])
P_ISO_C = DTH_EXACT / (np.pi * DC)


def is_excluded_1um(exp_dir: str, pid: int = None, particle_id: str = None) -> bool:
    if str(exp_dir) == "beads1um_crop":
        return True
    if str(exp_dir) == "beads1um001_crop":
        if pid is not None and int(pid) == 7:
            return True
        if particle_id is not None and str(particle_id) == "beads1um001_crop_7":
            return True
    return False


def format_diam(d: float) -> str:
    if d in (0.63, 1.18, 3.37, 7.24):
        return f"{d:.2f}" + r" $\mu\mathrm{m}$"
    if d == 5.0:
        return r"5.0 $\mu\mathrm{m}$"
    return f"{d:.0f}" + r" $\mu\mathrm{m}$"


def save_fig_multi(fig: plt.Figure, stem: str) -> None:
    for d in OUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / f"{stem}.png", dpi=300, bbox_inches="tight")
        fig.savefig(d / f"{stem}.svg", bbox_inches="tight")


def save_csv_multi(df: pd.DataFrame, fname: str) -> None:
    for d in OUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        df.to_csv(d / fname, index=False)


# ------------------------------------------------------------------
# 1. MSD helper
# ------------------------------------------------------------------
def compute_msd_with_ci(sub: pd.DataFrame, seed: int) -> Tuple[pd.DataFrame, Dict[str, float]]:
    sub = sub.copy()
    sub["cargo_id"] = sub["exp_dir"].astype(str) + "_" + sub["particle"].astype(str)
    groups = [g.sort_values("frame") for _, g in sub.groupby("cargo_id")]
    n_p = len(groups)

    S_d2 = np.zeros((n_p, MAX_MSD_LAG + 1), dtype=float)
    N_d2 = np.zeros((n_p, MAX_MSD_LAG + 1), dtype=float)
    for k, g in enumerate(groups):
        f = g["frame"].values.astype(int)
        x = g["x_px"].values.astype(float) * SCALE
        y = g["y_px"].values.astype(float) * SCALE
        f0 = int(f.min())
        L = int(f.max() - f0 + 1)
        gx = np.full(L, np.nan)
        gy = np.full(L, np.nan)
        gx[f - f0] = x
        gy[f - f0] = y
        ok = np.isfinite(gx) & np.isfinite(gy)
        for lag in range(1, min(MAX_MSD_LAG, L - 1) + 1):
            m = ok[: L - lag] & ok[lag:]
            cnt = int(np.sum(m))
            if cnt > 0:
                dx = gx[lag:][m] - gx[: L - lag][m]
                dy = gy[lag:][m] - gy[: L - lag][m]
                N_d2[k, lag] = cnt
                S_d2[k, lag] = float(np.sum(dx ** 2 + dy ** 2))

    def eval_msd(S_arr, N_arr):
        s = S_arr.sum(axis=0)
        n = N_arr.sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            m = s / n
        m[n < MIN_PAIRS] = np.nan
        m[0] = np.nan
        return m, n

    msd, n_pairs = eval_msd(S_d2, N_d2)
    lags_s = np.arange(MAX_MSD_LAG + 1) * DT

    def fit_alpha(msd_arr):
        mask = (lags_s >= 4.0) & (lags_s <= 240.0) & np.isfinite(msd_arr) & (msd_arr > 0)
        if np.sum(mask) < 5:
            return np.nan
        slope, _ = np.polyfit(np.log(lags_s[mask]), np.log(msd_arr[mask]), 1)
        return float(slope)

    alpha = fit_alpha(msd)

    rng = np.random.default_rng(seed)
    boot_m = np.full((N_BOOT, MAX_MSD_LAG + 1), np.nan)
    boot_alp = np.full(N_BOOT, np.nan)
    boot_m60 = np.full(N_BOOT, np.nan)
    boot_m240 = np.full(N_BOOT, np.nan)
    for bi in range(N_BOOT):
        sel = rng.integers(0, n_p, size=n_p)
        mb, _ = eval_msd(S_d2[sel], N_d2[sel])
        boot_m[bi] = mb
        boot_alp[bi] = fit_alpha(mb)
        boot_m60[bi] = mb[15]
        boot_m240[bi] = mb[60]

    df_curve = pd.DataFrame({
        "lag_frames": np.arange(1, MAX_MSD_LAG + 1),
        "lag_s": lags_s[1:],
        "msd_um2": msd[1:],
        "msd_ci_lo": np.nanpercentile(boot_m[:, 1:], 2.5, axis=0),
        "msd_ci_hi": np.nanpercentile(boot_m[:, 1:], 97.5, axis=0),
        "n_pairs": n_pairs[1:].astype(int),
    })

    stats = {
        "msd_60s": float(msd[15]),
        "msd_60s_ci_lo": float(np.nanpercentile(boot_m60, 2.5)),
        "msd_60s_ci_hi": float(np.nanpercentile(boot_m60, 97.5)),
        "msd_240s": float(msd[60]),
        "msd_240s_ci_lo": float(np.nanpercentile(boot_m240, 2.5)),
        "msd_240s_ci_hi": float(np.nanpercentile(boot_m240, 97.5)),
        "alpha_4_240": alpha,
        "alpha_ci_lo": float(np.nanpercentile(boot_alp, 2.5)),
        "alpha_ci_hi": float(np.nanpercentile(boot_alp, 97.5)),
    }
    return df_curve, stats


# ------------------------------------------------------------------
# 2. Per-point P(c), GvM Potential, Depth & KL Divergence
# ------------------------------------------------------------------
def eval_per_point_group(recs: List[dict], do_bootstrap: bool = True, seed: int = 42) -> Dict[str, object]:
    def compute_from_arrays(h_c_sum, h_th_sum, s_c, s_c2, n_pt):
        m1 = float(s_c / n_pt)
        m2 = float(2.0 * s_c2 / n_pt - 1.0)
        k1, k2, log_z = solve_gvm_from_moments(m1, m2)
        p_c = h_c_sum.astype(float) / (float(h_c_sum.sum()) * DC)
        with np.errstate(divide="ignore", invalid="ignore"):
            u_c = np.where(p_c > 0, -np.log(p_c / P_ISO_C), np.nan)
        log_z_norm = log_z - np.log(2.0 * np.pi)
        offset = k2 + log_z_norm
        u_gvm_c = -2.0 * k2 * (C_CENTERS ** 2) - k1 * C_CENTERS + offset
        fin = np.isfinite(u_c)
        rmse = float(np.sqrt(np.mean((u_c[fin] - u_gvm_c[fin]) ** 2))) if np.any(fin) else np.nan

        du_polar, du_nematic, du_gvm_bar, c_star = gvm_barrier_metrics(k1, k2)
        emp = empirical_depth_from_hist_th(h_th_sum)

        # 3-term Pythagorean KL decomposition on 36-bin theta
        p_th = h_th_sum.astype(float) / float(h_th_sum.sum())
        mk = p_th > 0
        d_kl_tot = float(np.sum(p_th[mk] * np.log(BINS_TH_N * p_th[mk])))
        k1_vm = solve_kappa1_from_m1(m1)
        d_kl_1 = float(k1_vm * m1 - np.log(i0(k1_vm)))
        d_kl_gvm = float(k1 * m1 + k2 * m2 - log_z_norm)
        d_kl_2 = float(d_kl_gvm - d_kl_1)
        d_res = float(d_kl_tot - d_kl_gvm)

        return {
            "m1": m1, "m2": m2, "kappa_1": k1, "kappa_2": k2, "log_z": log_z,
            "rmse_gvm": rmse, "du_gvm_polar": du_polar, "du_gvm_nematic": du_nematic,
            "du_gvm_bar": du_gvm_bar, "c_star": c_star,
            "d_kl_total": d_kl_tot, "d_kl_1": d_kl_1, "d_kl_2": d_kl_2, "delta_d_ge3": d_res,
            "P_c": p_c, "U_c": u_c, "U_gvm_c": u_gvm_c, **emp,
        }

    hc_mat = np.vstack([r["hist_c"] for r in recs]).astype(float)
    ht_mat = np.vstack([r["hist_th"] for r in recs]).astype(float)
    sc_vec = np.array([r["sum_c"] for r in recs], dtype=float)
    sc2_vec = np.array([r["sum_c2"] for r in recs], dtype=float)
    np_vec = np.array([r["n_pts"] for r in recs], dtype=float)
    nf_tot = int(sum(r["n_frames"] for r in recs))

    pt = compute_from_arrays(hc_mat.sum(0), ht_mat.sum(0), sc_vec.sum(), sc2_vec.sum(), np_vec.sum())
    pt["n_particles"] = len(recs)
    pt["n_frames"] = nf_tot
    pt["n_pts"] = int(np_vec.sum())

    if not do_bootstrap or len(recs) < 2:
        return pt

    n_r = len(recs)
    rng = np.random.default_rng(seed)
    keys_ci = [
        "m1", "m2", "kappa_1", "kappa_2", "rmse_gvm",
        "du_emp_pi0", "du_emp_bar", "du_gvm_polar", "du_gvm_nematic", "du_gvm_bar",
        "d_kl_total", "d_kl_1", "d_kl_2", "delta_d_ge3",
    ]
    b_vals = {k: np.full(N_BOOT, np.nan) for k in keys_ci}
    b_Uc = np.full((N_BOOT, len(C_CENTERS)), np.nan)
    b_Pc = np.full((N_BOOT, len(C_CENTERS)), np.nan)

    for bi in range(N_BOOT):
        sel = rng.integers(0, n_r, size=n_r)
        res_b = compute_from_arrays(
            hc_mat[sel].sum(0), ht_mat[sel].sum(0),
            sc_vec[sel].sum(), sc2_vec[sel].sum(), np_vec[sel].sum()
        )
        for k in keys_ci:
            b_vals[k][bi] = res_b[k]
        b_Uc[bi] = res_b["U_c"]
        b_Pc[bi] = res_b["P_c"]

    for k in keys_ci:
        arr = b_vals[k][np.isfinite(b_vals[k])]
        pt[f"{k}_ci_lo"] = float(np.percentile(arr, 2.5)) if len(arr) >= 10 else np.nan
        pt[f"{k}_ci_hi"] = float(np.percentile(arr, 97.5)) if len(arr) >= 10 else np.nan
    pt["U_c_ci_lo"] = np.nanpercentile(b_Uc, 2.5, axis=0)
    pt["U_c_ci_hi"] = np.nanpercentile(b_Uc, 97.5, axis=0)
    pt["P_c_ci_lo"] = np.nanpercentile(b_Pc, 2.5, axis=0)
    pt["P_c_ci_hi"] = np.nanpercentile(b_Pc, 97.5, axis=0)
    return pt


# ------------------------------------------------------------------
# 3. Transfer Efficiency helper
# ------------------------------------------------------------------
def eval_transfer(sub: pd.DataFrame, seed: int) -> Dict[str, float]:
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
        ok = np.isfinite(dot) & np.isfinite(v_par) & np.isfinite(cb) & (uspd > 0) & (vspd > 0)
        if np.any(ok):
            rows.append([
                float(np.sum(dot[ok])), float(np.sum(uspd[ok] ** 2)),
                float(np.sum(v_par[ok])), float(np.sum(uspd[ok])),
                float(np.sum(vspd[ok] ** 2)), float(np.sum(vspd[ok])),
                float(np.sum(cb[ok])), float(np.sum(ok)),
            ])
    mat = np.asarray(rows, dtype=float)
    n_p = len(mat)

    def calc(t):
        return {
            "eta_reg": t[..., 0] / t[..., 1],
            "eta_par": t[..., 2] / t[..., 3],
            "eta_T": t[..., 0] / np.sqrt(t[..., 4] * t[..., 1]),
            "spd_ratio": t[..., 5] / t[..., 3],
            "cos_bulk": t[..., 6] / t[..., 7],
        }

    pt = {k: float(val) for k, val in calc(mat.sum(0)).items()}
    rng = np.random.default_rng(seed)
    if n_p >= 2:
        w = rng.multinomial(n_p, np.full(n_p, 1.0 / n_p), size=N_BOOT)
        b = calc(w @ mat)
        for k, arr in b.items():
            pt[f"{k}_ci_lo"] = float(np.nanpercentile(arr, 2.5))
            pt[f"{k}_ci_hi"] = float(np.nanpercentile(arr, 97.5))
    else:
        for k in list(pt.keys()):
            pt[f"{k}_ci_lo"] = np.nan
            pt[f"{k}_ci_hi"] = np.nan
    return pt


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

    # Load all read-only sources
    df_inst_orig = pd.read_pickle(INST_CACHE)
    with open(PP_CACHE, "rb") as f:
        pp_orig = pickle.load(f)
    with open(GMM_HIST_CACHE, "rb") as f:
        gmm_hist_orig = pickle.load(f)
    df_bg_pts = pd.read_csv(BG_POINTS_CSV)
    df_bg_xi = pd.read_csv(BG_XI_EXP_CSV)

    # Filter cleaned datasets (exclude pid=7 of beads1um001_crop AND all beads1um_crop)
    mask_inst_excl = df_inst_orig.apply(
        lambda r: r["bead_name"] == "beads1um" and is_excluded_1um(r["exp_dir"], pid=r["particle"]), axis=1
    )
    df_inst_clean = df_inst_orig[~mask_inst_excl].copy()

    pp_clean = [
        r for r in pp_orig
        if not (r["bead_name"] == "beads1um" and is_excluded_1um(r["exp_dir"], particle_id=r["particle_id"]))
    ]
    gmm_hist_clean = [
        r for r in gmm_hist_orig
        if not (r["bead_name"] == "beads1um" and is_excluded_1um(r["exp_dir"], particle_id=r["particle_id"]))
    ]

    # Verify exact plan expectation: 12 particles, 3,605 frames for cleaned beads1um
    sub1_i = df_inst_clean[df_inst_clean["bead_name"] == "beads1um"]
    n_p1 = sub1_i[["exp_dir", "particle"]].drop_duplicates().shape[0]
    n_f1 = len(sub1_i)
    n_e1 = sub1_i["exp_dir"].nunique()
    print(f"Verified cleaned beads1um: n_particles={n_p1}, n_frames={n_f1}, n_experiments={n_e1}")
    assert n_p1 == 12 and n_f1 == 3605 and n_e1 == 3, "Unexpected cleaned beads1um counts!"

    # Group per-point records
    pp_by_bead_clean = defaultdict(list)
    for r in pp_clean:
        pp_by_bead_clean[r["bead_name"]].append(r)
    pp_1um_orig = [r for r in pp_orig if r["bead_name"] == "beads1um"]

    # ------------------------------------------------------------------
    # Compute full metrics for all 6 cleaned sizes + Original 1.18 um ref
    # ------------------------------------------------------------------
    summary_rows = []
    msd_curves = {}
    msd_csv_rows = []
    sp_curves = {}
    sp_csv_rows = []
    pp_results = {}

    specs_to_run = [(b, True, f"{BEAD_META[b]['diameter_um']:.2f}") for b in ORDER_BEADS]
    specs_to_run.append(("beads1um", False, "1.18 (orig)"))

    for idx_s, (bname, is_clean, tag) in enumerate(specs_to_run):
        key_tag = bname if is_clean else "beads1um_orig"
        diam = BEAD_META[bname]["diameter_um"]
        rad_um = BEAD_META[bname]["radius_um"]

        sub_i = (df_inst_clean if is_clean else df_inst_orig)
        sub_i = sub_i[sub_i["bead_name"] == bname]
        recs_pp = pp_by_bead_clean[bname] if is_clean else pp_1um_orig

        # Background xi and C(r)
        if is_clean and bname == "beads1um":
            sub_xi = df_bg_xi[(df_bg_xi["bead_name"] == bname) & (df_bg_xi["exp_dir"] != "beads1um_crop")]
            sub_pts = df_bg_pts[(df_bg_pts["bead_name"] == bname) & (df_bg_pts["exp_dir"] != "beads1um_crop")]
        else:
            sub_xi = df_bg_xi[df_bg_xi["bead_name"] == bname]
            sub_pts = df_bg_pts[df_bg_pts["bead_name"] == bname]

        xi_mean = float(sub_xi["xi_um"].mean())
        xi_sem = float(sub_xi["xi_um"].std(ddof=1) / np.sqrt(len(sub_xi))) if len(sub_xi) > 1 else 0.0
        n_exps_bg = int(len(sub_xi))

        rc_vals_pp = np.array([r["rc_over_xi"] for r in recs_pp], dtype=float)
        rc_xi_mean = float(np.mean(rc_vals_pp))
        rc_xi_sem = float(np.std(rc_vals_pp, ddof=1) / np.sqrt(len(rc_vals_pp))) if len(rc_vals_pp) > 1 else 0.0

        # Spatial correlation C(r) per experiment mean +/- SEM
        sp_grp = sub_pts.groupby("distance_um")["mean_c"].agg(["mean", "sem", "count"]).reset_index()
        sp_curves[key_tag] = sp_grp
        for _, sr in sp_grp.iterrows():
            sp_csv_rows.append({
                "series": key_tag, "bead_name": bname, "cleaned": is_clean,
                "distance_um": float(sr["distance_um"]),
                "mean_c": float(sr["mean"]), "sem_c": float(sr["sem"]),
                "n_experiments": int(sr["count"]),
            })

        # MSD
        df_msd_c, msd_st = compute_msd_with_ci(sub_i, seed=100 + idx_s)
        msd_curves[key_tag] = df_msd_c
        for _, mr in df_msd_c.iterrows():
            msd_csv_rows.append({"series": key_tag, "bead_name": bname, "cleaned": is_clean, **mr.to_dict()})

        # Per-point P(c), GvM, Potential depth, KL
        pp_res = eval_per_point_group(recs_pp, do_bootstrap=True, seed=200 + idx_s)
        pp_results[key_tag] = pp_res

        # Verify Pythagorean KL sum identity
        kl_sum = pp_res["d_kl_1"] + pp_res["d_kl_2"] + pp_res["delta_d_ge3"]
        assert abs(kl_sum - pp_res["d_kl_total"]) < 1e-10, f"KL identity mismatch: {kl_sum} vs {pp_res['d_kl_total']}"

        # Transfer efficiency
        tr_st = eval_transfer(sub_i, seed=300 + idx_s)

        summary_rows.append({
            "series": key_tag, "bead_name": bname, "cleaned": is_clean,
            "diameter_um": diam, "radius_um": rad_um,
            "n_experiments_bg": n_exps_bg,
            "xi_bg_mean_um": xi_mean, "xi_bg_sem_um": xi_sem,
            "rc_over_xi_by_xi_bg_mean": rad_um / xi_mean,
            "rc_over_xi_particle_mean": rc_xi_mean, "rc_over_xi_particle_sem": rc_xi_sem,
            "n_particles": pp_res["n_particles"], "n_frames": pp_res["n_frames"], "n_pts": pp_res["n_pts"],
            **msd_st,
            **{k: v for k, v in pp_res.items() if np.isscalar(v)},
            **tr_st,
        })

    df_sum = pd.DataFrame(summary_rows)
    save_csv_multi(df_sum, "cleaned_full_summary_by_size.csv")
    save_csv_multi(pd.DataFrame(msd_csv_rows), "cleaned_msd_curves.csv")
    save_csv_multi(pd.DataFrame(sp_csv_rows), "cleaned_spatial_corr_curves.csv")

    # ------------------------------------------------------------------
    # GMM 1.18 um Run/Tumble Before vs After
    # ------------------------------------------------------------------
    df_gmm_frames_orig = build_gmm_labeled_frames()
    df_gmm_1um_orig = df_gmm_frames_orig[df_gmm_frames_orig["bead_name"] == "beads1um"].copy()
    mask_gmm_excl = df_gmm_1um_orig.apply(lambda r: is_excluded_1um(r["exp_dir"], pid=r["particle"]), axis=1)
    df_gmm_1um_clean = df_gmm_1um_orig[~mask_gmm_excl].copy()

    gmm_1um_rows = []
    gmm_1um_pot_curves = {}
    for cleaned_flag, label_tag, recs_src, df_src in [
        (False, "orig", [r for r in gmm_hist_orig if r["bead_name"] == "beads1um"], df_gmm_1um_orig),
        (True, "clean", [r for r in gmm_hist_clean if r["bead_name"] == "beads1um"], df_gmm_1um_clean),
    ]:
        for st in ("Run", "Tumble"):
            recs_st = [r for r in recs_src if r["state_label"] == st]
            sub_f = df_src[df_src["state_label"] == st]
            res_st = eval_per_point_group(recs_st, do_bootstrap=True, seed=500 if cleaned_flag else 400)
            tr_st = eval_transfer(sub_f, seed=550 if cleaned_flag else 450)
            gmm_1um_pot_curves[(label_tag, st)] = res_st
            gmm_1um_rows.append({
                "bead_name": "beads1um", "cleaned": cleaned_flag, "state": st,
                "n_particles": res_st["n_particles"], "n_frames_hist": res_st["n_frames"],
                "n_frames_matched": len(sub_f),
                **{k: v for k, v in res_st.items() if np.isscalar(v)},
                **tr_st,
            })
    df_gmm_1um = pd.DataFrame(gmm_1um_rows)
    save_csv_multi(df_gmm_1um, "cleaned_gmm_1um_before_after.csv")

    # ==================================================================
    # FIGURE 1: MSD & Background Spatial Correlation C(r)
    # ==================================================================
    fig1, (axm, axs) = plt.subplots(1, 2, figsize=(13.2, 5.3))
    fig1.subplots_adjust(top=0.84, wspace=0.23)

    for bname in ORDER_BEADS:
        meta = BEAD_META[bname]
        r_row = df_sum[df_sum["series"] == bname].iloc[0]
        dm = msd_curves[bname]
        lbl = f"{meta['label']} ($\\alpha={r_row['alpha_4_240']:.2f}$)"
        if bname == "beads1um":
            lbl = f"1.18 $\\mu\\mathrm{{m}}$ clean ($\\alpha={r_row['alpha_4_240']:.2f}$)"
        axm.plot(dm["lag_s"], dm["msd_um2"], color=meta["color"], marker=meta["marker"], ms=3.8, markevery=6,
                 lw=2.0 if bname == "beads1um" else 1.7, label=lbl, zorder=4)
        axm.fill_between(dm["lag_s"], dm["msd_ci_lo"], dm["msd_ci_hi"], color=meta["color"], alpha=0.14, lw=0)

    # Original 1.18 um dashed overlay
    r_orig = df_sum[df_sum["series"] == "beads1um_orig"].iloc[0]
    dm_o = msd_curves["beads1um_orig"]
    axm.plot(dm_o["lag_s"], dm_o["msd_um2"], color="#CC6677", ls="--", lw=1.6, alpha=0.8,
             label=f"1.18 $\\mu\\mathrm{{m}}$ orig ($\\alpha={r_orig['alpha_4_240']:.2f}$)", zorder=3)

    # Guide slopes tau^1 and tau^2
    t_ref = np.array([12.0, 200.0])
    axm.plot(t_ref, 0.25 * (t_ref / t_ref[0]) ** 1, color="0.45", ls=":", lw=1.2)
    axm.text(110, 1.3, r"$\propto \tau^1$", fontsize=9.0, color="0.35")
    axm.plot(t_ref, 3.0 * (t_ref / t_ref[0]) ** 2, color="0.45", ls="-.", lw=1.1)
    axm.text(85, 220, r"$\propto \tau^2$", fontsize=9.0, color="0.35")

    axm.set_xscale("log")
    axm.set_yscale("log")
    axm.set_xlim(3.5, 320)
    axm.set_ylim(0.15, 1200)
    axm.set_xlabel(r"Lag time $\tau$ [s]")
    axm.set_ylabel(r"$\mathrm{MSD}(\tau)$ [$\mu\mathrm{m}^2$]")
    axm.set_title(r"A. Mean Squared Displacement ($\Delta t = 4$ s, 95% Cluster CI)")
    axm.legend(loc="upper left", frameon=True, framealpha=0.92, fontsize=7.8)
    axm.grid(alpha=0.3, which="both")

    # Panel B: Background spatial orientational correlation C(r)
    for bname in ORDER_BEADS:
        meta = BEAD_META[bname]
        r_row = df_sum[df_sum["series"] == bname].iloc[0]
        sp = sp_curves[bname]
        m = sp["distance_um"] <= 25.0
        lbl = f"{meta['label']} ($\\xi={r_row['xi_bg_mean_um']:.2f}\\,\\mu\\mathrm{{m}}$)"
        if bname == "beads1um":
            lbl = f"1.18 $\\mu\\mathrm{{m}}$ clean ($\\xi={r_row['xi_bg_mean_um']:.2f}\\,\\mu\\mathrm{{m}}$, N=3)"
        axs.plot(sp.loc[m, "distance_um"], sp.loc[m, "mean"], color=meta["color"], marker=meta["marker"],
                 ms=3.5, markevery=4, lw=2.0 if bname == "beads1um" else 1.6, label=lbl, zorder=4)
        axs.fill_between(
            sp.loc[m, "distance_um"],
            sp.loc[m, "mean"] - sp.loc[m, "sem"],
            sp.loc[m, "mean"] + sp.loc[m, "sem"],
            color=meta["color"], alpha=0.12, lw=0
        )

    sp_o = sp_curves["beads1um_orig"]
    mo = sp_o["distance_um"] <= 25.0
    axs.plot(sp_o.loc[mo, "distance_um"], sp_o.loc[mo, "mean"], color="#CC6677", ls="--", lw=1.6, alpha=0.85,
             label=f"1.18 $\\mu\\mathrm{{m}}$ orig ($\\xi={r_orig['xi_bg_mean_um']:.2f}\\,\\mu\\mathrm{{m}}$, N=4)", zorder=3)

    axs.axhline(np.exp(-1.0), color="0.45", ls=":", lw=1.1)
    axs.text(24.5, np.exp(-1.0) + 0.02, r"$1/e$", fontsize=8.5, color="0.35", ha="right")
    axs.axhline(0.0, color="0.25", lw=0.6)
    axs.set_xlim(0, 25.0)
    axs.set_ylim(-0.08, 1.04)
    axs.set_xlabel(r"Distance $r$ [$\mu\mathrm{m}$]")
    axs.set_ylabel(r"Background Orientational Correlation $C(r)$")
    axs.set_title(r"B. Spatial Orientational Correlation $C(r)$ & $\xi$ (SEM)")
    axs.legend(loc="upper right", frameon=True, framealpha=0.92, fontsize=7.8)
    axs.grid(alpha=0.3)

    fig1.suptitle(r"Cleaned $1.18\,\mu\mathrm{m}$ Analysis (excl. pid=7 & beads1um_crop): MSD & Spatial Correlation", fontsize=13.0, y=0.96)
    save_fig_multi(fig1, "cleaned_msd_and_spatial_corr")
    plt.close(fig1)

    # ==================================================================
    # FIGURE 2: Per-point P(c), GvM Potential U(c), 1 um Zoom, Moments m1/m2
    # ==================================================================
    fig2, ((axp1, axp2), (axp3, axp4)) = plt.subplots(2, 2, figsize=(13.4, 9.8))
    fig2.subplots_adjust(top=0.91, hspace=0.32, wspace=0.23)

    p_iso_curve = 1.0 / (np.pi * np.sqrt(np.clip(1.0 - C_CENTERS ** 2, 1e-4, None)))

    # Panel A: Per-pixel P(c)
    for bname in ORDER_BEADS:
        meta = BEAD_META[bname]
        pr = pp_results[bname]
        lbl = "1.18 $\\mu\\mathrm{m}$ clean" if bname == "beads1um" else meta["label"]
        axp1.plot(C_CENTERS, pr["P_c"], color=meta["color"], marker=meta["marker"], ms=3.5, markevery=3,
                  lw=2.0 if bname == "beads1um" else 1.5, label=lbl, zorder=4)
    axp1.plot(C_CENTERS, pp_results["beads1um_orig"]["P_c"], color="#CC6677", ls="--", lw=1.5, alpha=0.85,
              label="1.18 $\\mu\\mathrm{m}$ orig", zorder=3)
    axp1.plot(C_CENTERS, p_iso_curve, color="0.4", ls=":", lw=1.3, label=r"$P_{\mathrm{iso}}(c)$", zorder=2)
    axp1.set_yscale("log")
    axp1.set_xlim(-1.02, 1.02)
    axp1.set_ylim(0.12, 8.5)
    axp1.set_xlabel(r"$c = \cos\theta_{\mathrm{pt}} = \hat{\mathbf{v}}_i \cdot \hat{\mathbf{u}}(\mathbf{r})$")
    axp1.set_ylabel(r"Probability Density $P(c)$")
    axp1.set_title(r"A. Per-Pixel Alignment Distribution $P(c)$")
    axp1.legend(loc="upper center", frameon=True, framealpha=0.92, fontsize=7.7, ncol=2)
    axp1.grid(alpha=0.3, which="both")

    # Panel B: Effective Potential U_eff(c) + Zero-fit GvM Theory across 6 cleaned sizes
    for bname in ORDER_BEADS:
        meta = BEAD_META[bname]
        pr = pp_results[bname]
        lbl = "1.18 $\\mu\\mathrm{m}$ clean" if bname == "beads1um" else meta["label"]
        axp2.plot(C_CENTERS, pr["U_c"], marker=meta["marker"], ms=4.2, color=meta["color"], ls="none", alpha=0.88,
                  label=lbl, zorder=4)
        axp2.plot(C_CENTERS, pr["U_gvm_c"], color=meta["color"], lw=1.6, alpha=0.9, zorder=3)
    axp2.axhline(0.0, color="0.4", ls=":", lw=1.0)
    axp2.set_xlim(-1.02, 1.02)
    axp2.set_ylim(-1.6, 1.1)
    axp2.set_xlabel(r"$c = \cos\theta_{\mathrm{pt}}$")
    axp2.set_ylabel(r"Effective Potential $\tilde{U}_{\mathrm{eff}}(c)$ [$k_B T_{\mathrm{eff}}$]")
    axp2.set_title(r"B. Effective Potential $\tilde{U}_{\mathrm{eff}}(c)$ & GvM 2-Moment Theory (solid)")
    axp2.legend(loc="lower left", frameon=True, framealpha=0.92, fontsize=7.8, ncol=2)
    axp2.grid(alpha=0.3)

    # Panel C: Before vs After 1.18 um Zoom (All, GMM Run, GMM Tumble)
    zoom_styles = [
        ("all", "clean", "1.18 All (clean)", "#CC6677", "-", "o"),
        ("all", "orig", "1.18 All (orig)", "#CC6677", "--", None),
        ("Run", "clean", "1.18 Run (clean)", "#d35400", "-", "^"),
        ("Run", "orig", "1.18 Run (orig)", "#d35400", "--", None),
        ("Tumble", "clean", "1.18 Tumble (clean)", "#2e86c1", "-", "v"),
        ("Tumble", "orig", "1.18 Tumble (orig)", "#2e86c1", "--", None),
    ]
    for st_k, tag_k, lbl, col, ls, mk in zoom_styles:
        pr = pp_results["beads1um" if tag_k == "clean" else "beads1um_orig"] if st_k == "all" else gmm_1um_pot_curves[(tag_k, st_k)]
        axp3.plot(C_CENTERS, pr["U_c"], color=col, ls=ls, marker=mk, ms=4.2 if mk else 0, markevery=2,
                  lw=2.0 if tag_k == "clean" else 1.4, alpha=0.95 if tag_k == "clean" else 0.75, label=lbl)
    axp3.axhline(0.0, color="0.4", ls=":", lw=1.0)
    axp3.set_xlim(-1.02, 1.02)
    axp3.set_ylim(-2.1, 1.3)
    axp3.set_xlabel(r"$c = \cos\theta_{\mathrm{pt}}$")
    axp3.set_ylabel(r"Effective Potential $\tilde{U}_{\mathrm{eff}}(c)$ [$k_B T_{\mathrm{eff}}$]")
    axp3.set_title(r"C. $1.18\,\mu\mathrm{m}$ Before vs. After Exclusion (All / GMM Run / Tumble)")
    axp3.legend(loc="lower left", frameon=True, framealpha=0.92, fontsize=7.8, ncol=2)
    axp3.grid(alpha=0.3)

    # Panel D: m1 & m2 vs Rc/xi across all 6 cleaned sizes (+ orig 1.18 ghost)
    df_c6 = df_sum[df_sum["cleaned"]].copy()
    xc = df_c6["rc_over_xi_particle_mean"].values
    axp4.errorbar(
        xc, df_c6["m1"],
        yerr=[np.maximum(0.0, df_c6["m1"] - df_c6["m1_ci_lo"]), np.maximum(0.0, df_c6["m1_ci_hi"] - df_c6["m1"])],
        fmt="-o", color="#1b4f72", lw=1.9, ms=6.8, capsize=3.5, label=r"Clean $m_1 = \langle\cos\theta_{\mathrm{pt}}\rangle$", zorder=4
    )
    axp4.errorbar(
        xc, df_c6["m2"],
        yerr=[np.maximum(0.0, df_c6["m2"] - df_c6["m2_ci_lo"]), np.maximum(0.0, df_c6["m2_ci_hi"] - df_c6["m2"])],
        fmt="--^", color="#7d3c98", lw=1.8, ms=6.5, capsize=3.5, label=r"Clean $m_2 = \langle\cos 2\theta_{\mathrm{pt}}\rangle$", zorder=4
    )
    axp4.scatter([r_orig["rc_over_xi_particle_mean"]], [r_orig["m1"]], s=75, facecolors="none", edgecolors="#1b4f72",
                 marker="o", lw=1.5, label=r"Orig $1.18\,\mu\mathrm{m}$ ($m_1, m_2$)", zorder=5)
    axp4.scatter([r_orig["rc_over_xi_particle_mean"]], [r_orig["m2"]], s=75, facecolors="none", edgecolors="#7d3c98",
                 marker="^", lw=1.5, zorder=5)
    axp4.axvline(X_LOCK, color="0.45", ls="--", lw=1.1)
    axp4.axvline(X_BULK, color="0.45", ls=":", lw=1.1)
    axp4.text(X_LOCK * 1.05, 0.26, r"$x_{\mathrm{lock}}$", fontsize=8.4, color="0.35")
    axp4.text(X_BULK * 1.05, 0.26, r"$x_{\mathrm{bulk}}$", fontsize=8.4, color="0.35")
    axp4.axhline(0.0, color="0.3", lw=0.7)
    axp4.set_xscale("log")
    axp4.set_xlim(0.045, 3.0)
    axp4.set_ylim(-0.08, 0.50)
    axp4.set_xlabel(r"$R_c / \xi$")
    axp4.set_ylabel("Moment Value")
    axp4.set_title(r"D. 1st & 2nd Moments vs. $R_c/\xi$ (95% Cluster CI)")
    axp4.legend(loc="upper right", frameon=True, framealpha=0.92, fontsize=8.0)
    axp4.grid(alpha=0.3, which="both")

    fig2.suptitle(r"Cleaned Per-Point $P(c)$, GvM Potential $\tilde{U}_{\mathrm{eff}}(c)$, and Moments", fontsize=13.4, y=0.97)
    save_fig_multi(fig2, "cleaned_per_point_Pc_and_gvm_potential")
    plt.close(fig2)

    # ==================================================================
    # FIGURE 3: KL Divergence Decomposition, Potential Depth, Transfer Efficiency
    # ==================================================================
    fig3, (axk1, axk2, axk3) = plt.subplots(1, 3, figsize=(16.2, 5.1))
    fig3.subplots_adjust(top=0.82, wspace=0.27)

    # Panel A: 3-term KL decomposition
    for col_y, lbl, col, fmt in [
        ("d_kl_total", r"$D_{\mathrm{KL}}(P \parallel P_{\mathrm{iso}})$ Total", "#1b4f72", "-o"),
        ("d_kl_1", r"$D_{\mathrm{KL}}^{(1)}$ (1st-order polar)", "#c0392b", "--s"),
        ("d_kl_2", r"$D_{\mathrm{KL}}^{(2)}$ (2nd-order nematic)", "#7d3c98", "-.^"),
        ("delta_d_ge3", r"$\Delta D_{\geq 3}$ (higher-order residual)", "#16a085", ":D"),
    ]:
        y = df_c6[col_y].values
        yerr = np.vstack([np.maximum(0.0, y - df_c6[f"{col_y}_ci_lo"].values),
                          np.maximum(0.0, df_c6[f"{col_y}_ci_hi"].values - y)])
        axk1.errorbar(xc, y, yerr=yerr, fmt=fmt, color=col, lw=1.8, ms=6.0, capsize=3.0, label=lbl, zorder=4)

    axk1.scatter([r_orig["rc_over_xi_particle_mean"]], [r_orig["d_kl_total"]], s=75, facecolors="none", edgecolors="#1b4f72",
                 marker="o", lw=1.5, label=r"Orig $1.18\,\mu\mathrm{m}$ Total $D_{\mathrm{KL}}$", zorder=5)
    axk1.axvline(X_LOCK, color="0.45", ls="--", lw=1.1)
    axk1.axvline(X_BULK, color="0.45", ls=":", lw=1.1)
    axk1.text(X_LOCK * 1.05, 0.16, r"$x_{\mathrm{lock}}$", fontsize=8.4, color="0.35")
    axk1.text(X_BULK * 1.05, 0.16, r"$x_{\mathrm{bulk}}$", fontsize=8.4, color="0.35")
    axk1.set_xscale("log")
    axk1.set_xlim(0.045, 3.0)
    axk1.set_ylim(-0.01, 0.36)
    axk1.set_xlabel(r"$R_c / \xi$")
    axk1.set_ylabel("KL Divergence [nats]")
    axk1.set_title("A. 3-Term KL Decomposition vs. $R_c/\\xi$")
    axk1.legend(loc="upper right", frameon=True, framealpha=0.92, fontsize=7.6)
    axk1.grid(alpha=0.3, which="both")

    # Panel B: Effective Potential Barrier vs Polar & Nematic
    for col_y, lbl, col, fmt in [
        ("du_emp_bar", r"Empirical barrier $\Delta\tilde{U}_{\mathrm{emp,bar}}$", "#1b4f72", "-o"),
        ("du_gvm_bar", r"GvM theory barrier $\Delta\tilde{U}_{\mathrm{GvM,bar}}$", "#d35400", ":D"),
        ("du_gvm_polar", r"Polar $2\kappa_1$", "#c0392b", "--s"),
        ("du_gvm_nematic", r"Nematic $2\kappa_2$", "#7d3c98", "-.^"),
    ]:
        y = df_c6[col_y].values
        yerr = np.vstack([np.maximum(0.0, y - df_c6[f"{col_y}_ci_lo"].values),
                          np.maximum(0.0, df_c6[f"{col_y}_ci_hi"].values - y)])
        axk2.errorbar(xc, y, yerr=yerr, fmt=fmt, color=col, lw=1.8, ms=6.0, capsize=3.0, label=lbl, zorder=4)

    axk2.scatter([r_orig["rc_over_xi_particle_mean"]], [r_orig["du_emp_bar"]], s=75, facecolors="none", edgecolors="#1b4f72",
                 marker="o", lw=1.5, label=r"Orig $1.18\,\mu\mathrm{m}$ barrier", zorder=5)
    axk2.axvline(X_LOCK, color="0.45", ls="--", lw=1.1)
    axk2.axvline(X_BULK, color="0.45", ls=":", lw=1.1)
    axk2.axhline(1.0, color="#444444", ls="-.", lw=1.1)
    axk2.text(2.75, 1.04, r"Noise $1\,k_B T_{\mathrm{eff}}$", fontsize=8.2, color="#444444", ha="right", va="bottom")
    axk2.set_xscale("log")
    axk2.set_xlim(0.045, 3.0)
    axk2.set_ylim(-0.08, 2.65)
    axk2.set_xlabel(r"$R_c / \xi$")
    axk2.set_ylabel(r"Potential Depth [$k_B T_{\mathrm{eff}}$]")
    axk2.set_title(r"B. Effective Potential Depth vs. $R_c/\xi$")
    axk2.legend(loc="upper right", frameon=True, framealpha=0.92, fontsize=7.6)
    axk2.grid(alpha=0.3, which="both")

    # Panel C: Transfer Efficiency (clean vs orig 1.18 um)
    for col_y, lbl, col, fmt in [
        ("eta_reg", r"$\eta_{\mathrm{reg}}$ (clean)", "#1b4f72", "-o"),
        ("eta_par", r"$\eta_\parallel$ (clean)", "#2e86c1", "--s"),
        ("cos_bulk", r"$\langle\cos\Theta_{\mathrm{bulk}}\rangle$ (clean)", "#882255", "-.^"),
    ]:
        y = df_c6[col_y].values
        yerr = np.vstack([np.maximum(0.0, y - df_c6[f"{col_y}_ci_lo"].values),
                          np.maximum(0.0, df_c6[f"{col_y}_ci_hi"].values - y)])
        axk3.errorbar(xc, y, yerr=yerr, fmt=fmt, color=col, lw=1.8, ms=6.2, capsize=3.0, label=lbl, zorder=4)

    axk3.scatter([r_orig["rc_over_xi_particle_mean"]], [r_orig["eta_reg"]], s=80, facecolors="none", edgecolors="#1b4f72",
                 marker="o", lw=1.6, label=r"Orig $1.18\,\mu\mathrm{m}$ $\eta_{\mathrm{reg}}$ (0.204)", zorder=5)
    axk3.scatter([r_orig["rc_over_xi_particle_mean"]], [r_orig["eta_par"]], s=75, facecolors="none", edgecolors="#2e86c1",
                 marker="s", lw=1.5, label=r"Orig $1.18\,\mu\mathrm{m}$ $\eta_\parallel$ (0.339)", zorder=5)
    axk3.axvline(X_LOCK, color="0.45", ls="--", lw=1.1)
    axk3.axvline(X_BULK, color="0.45", ls=":", lw=1.1)
    axk3.axhline(0.0, color="0.3", lw=0.7)
    axk3.set_xscale("log")
    axk3.set_xlim(0.045, 3.0)
    axk3.set_ylim(-0.06, 0.85)
    axk3.set_xlabel(r"$R_c / \xi$")
    axk3.set_ylabel(r"Transfer Efficiency / Alignment")
    axk3.set_title(r"C. Transfer Efficiency Recovery in $1.18\,\mu\mathrm{m}$")
    axk3.legend(loc="upper right", frameon=True, framealpha=0.92, fontsize=7.6)
    axk3.grid(alpha=0.3, which="both")

    fig3.suptitle(r"Cleaned $1.18\,\mu\mathrm{m}$ Analysis: KL Divergence, Potential Depth, & Transfer Efficiency (95% CI)", fontsize=13.2, y=0.95)
    save_fig_multi(fig3, "cleaned_kl_and_transfer_vs_rc_xi")
    plt.close(fig3)

    # Print comparison summaries
    print("\n=== Full Cleaned Summary Across All 6 Sizes (+ Orig 1.18 um Ref) ===")
    cols_print = [
        "series", "diameter_um", "xi_bg_mean_um", "rc_over_xi_particle_mean",
        "n_particles", "n_frames", "msd_240s", "alpha_4_240",
        "m1", "m2", "du_emp_bar", "du_gvm_bar",
        "d_kl_total", "d_kl_1", "d_kl_2", "delta_d_ge3",
        "eta_reg", "eta_par", "cos_bulk", "spd_ratio"
    ]
    print(df_sum[cols_print].to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print("\n=== beads1um GMM Run / Tumble: Original vs Cleaned ===")
    cols_gmm_print = [
        "cleaned", "state", "n_particles", "n_frames_hist", "n_frames_matched",
        "m1", "m2", "du_emp_bar", "du_gvm_bar",
        "d_kl_total", "d_kl_1", "d_kl_2", "delta_d_ge3",
        "eta_reg", "eta_par", "cos_bulk"
    ]
    print(df_gmm_1um[cols_gmm_print].to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
