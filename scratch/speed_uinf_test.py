#!/usr/bin/env python3
"""
scratch/speed_uinf_test.py
==========================
S8 節モデル (v = A * e_hat + zeta, s = A / sigma) を向きではなく速さで検証するスクリプト。
u_inf(t) による規格化が雑音特性・分布幅・P(s) の形状に与える影響を A, B, C の 3 段階で評価する。

出力:
  - figure/speed_uinf_test/var_vperp_vs_uinf2.png
  - figure/speed_uinf_test/lnw_hist_vs_model.png
  - figure/speed_uinf_test/*.csv
  - figure/speed_uinf_test/REPORT.md
"""

from pathlib import Path
from typing import Callable, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "figure" / "speed_uinf_test"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 単位変換定数 (contrast_analysis.py / calc_and_plot_v_tilde_sq_diff.py と同一)
SCALE_UM_PER_PX = 0.11
DT_SEC = 4.0
V_UNIT = SCALE_UM_PER_PX / DT_SEC  # 0.0275 (um/s per px/frame)

N_BOOT = 300
N_MC = 200_000
SEED = 42

BEAD_ORDER = ["beads06um", "beads1um", "beads3um", "beads5um", "beads7um", "beads20um"]
BEAD_LABELS = {
    "beads06um": r"$0.63\,\mu\mathrm{m}$",
    "beads1um": r"$1.18\,\mu\mathrm{m}$",
    "beads3um": r"$3.37\,\mu\mathrm{m}$",
    "beads5um": r"$5.00\,\mu\mathrm{m}$",
    "beads7um": r"$7.24\,\mu\mathrm{m}$",
    "beads20um": r"$20.0\,\mu\mathrm{m}$",
}

# S8.3 節の固定予測パラメータ
S83_PARAMS = {
    "beads06um": {"ln_med": 0.32, "ln_sig": 1.89, "gam_mean": 1.51, "gam_cv": 1.45},
    "beads1um":  {"ln_med": 0.26, "ln_sig": 1.84, "gam_mean": 1.26, "gam_cv": 1.45},
    "beads3um":  {"ln_med": 0.35, "ln_sig": 1.00, "gam_mean": 0.57, "gam_cv": 1.08},
    "beads5um":  {"ln_med": 0.09, "ln_sig": 1.54, "gam_mean": 0.42, "gam_cv": 1.45},
    "beads7um":  {"ln_med": 0.21, "ln_sig": 0.95, "gam_mean": 0.34, "gam_cv": 1.10},
    "beads20um": {"ln_med": 0.02, "ln_sig": 2.52, "gam_mean": 0.40, "gam_cv": 2.43},
}


def load_and_filter_data() -> Tuple[pd.DataFrame, Dict]:
    """キャッシュを読み込み、harmonics_frames と同じフレーム集合へ絞り、単位変換と除外を行う。"""
    inst = pd.read_pickle(ROOT / "scratch" / "contrast_inst_cache.pkl")
    uinf = pd.read_pickle(ROOT / "scratch" / "u_inf_per_frame_cache.pkl")
    harm = pd.read_pickle(ROOT / "scratch" / "harmonics_frames.pkl")

    inst["pid"] = inst["exp_dir"].astype(str) + "_" + inst["particle"].astype(str)
    harm_keys = harm[["exp", "pid", "frame"]].drop_duplicates()

    df = inst.merge(
        harm_keys,
        left_on=["exp_dir", "pid", "frame"],
        right_on=["exp", "pid", "frame"],
        how="inner",
    )
    n_harm_matched = len(df)

    df = df.merge(uinf[["exp_dir", "frame", "u_inf_t"]], on=["exp_dir", "frame"], how="left")

    # 速度を um/s へ変換
    vx = df["cargo_dx"].values * V_UNIT
    vy = df["cargo_dy"].values * V_UNIT
    ux = df["flow_u"].values * V_UNIT
    uy = df["flow_v"].values * V_UNIT

    u_fp = np.hypot(ux, uy)
    v_mag = np.hypot(vx, vy)

    with np.errstate(invalid="ignore", divide="ignore"):
        ex = ux / u_fp
        ey = uy / u_fp

    df["vx"] = vx
    df["vy"] = vy
    df["u_fp"] = u_fp
    df["v_mag"] = v_mag
    df["v_par"] = vx * ex + vy * ey
    df["v_perp"] = vx * ey - vy * ex  # 符号付き外積 v x e_hat

    # 除外マスク判定
    valid_uinf = df["u_inf_t"].notna() & (df["u_inf_t"] > 0)
    valid_ufp = np.isfinite(u_fp) & (u_fp > 0)
    nonzero_v = v_mag > 0

    q05 = df.groupby("exp_dir")["u_inf_t"].transform(lambda s: s.quantile(0.05))
    not_low5 = df["u_inf_t"] > q05

    keep = valid_uinf & valid_ufp & nonzero_v & not_low5

    excl_summary = {
        "n_initial": n_harm_matched,
        "n_missing_uinf": int((~valid_uinf).sum()),
        "n_zero_ufp": int((~valid_ufp).sum()),
        "n_zero_v": int((~nonzero_v).sum()),
        "n_low5_uinf": int((~not_low5).sum()),
        "n_kept": int(keep.sum()),
        "excl_frac_total": float(1.0 - keep.mean()),
    }

    per_bead_excl = []
    for b in BEAD_ORDER:
        m = df["bead_name"] == b
        n_b = int(m.sum())
        n_k = int((m & keep).sum())
        per_bead_excl.append({
            "bead_name": b,
            "n_before": n_b,
            "n_after": n_k,
            "n_low5": int((m & ~not_low5).sum()),
            "n_zero_v": int((m & ~nonzero_v).sum()),
            "excl_frac": float(1.0 - n_k / n_b) if n_b > 0 else np.nan,
        })
    excl_summary["per_bead"] = per_bead_excl

    df = df.loc[keep].copy()
    df["w_par"] = df["v_par"] / df["u_inf_t"]
    df["w_perp"] = df["v_perp"] / df["u_inf_t"]
    df["w"] = df["v_mag"] / df["u_inf_t"]
    df["w_perp_ufp"] = df["v_perp"] / df["u_fp"]
    df["ln_w"] = np.log(df["w"])
    df["ln_v"] = np.log(df["v_mag"])

    # 貨物内中心化 (within-cargo)
    df["ln_w_wc"] = df["ln_w"] - df.groupby("pid")["ln_w"].transform("mean")
    df["ln_v_wc"] = df["ln_v"] - df.groupby("pid")["ln_v"].transform("mean")

    return df, excl_summary


def build_hier_structure(df_size: pd.DataFrame, cols: List[str]) -> Tuple[List[str], Dict[str, List[np.ndarray]]]:
    """サイズ別データから、動画 -> 貨物ごとの NumPy 配列辞書を構築する。"""
    mov_dict: Dict[str, List[np.ndarray]] = {}
    for exp, df_e in df_size.groupby("exp_dir"):
        cargos = [df_c[cols].values for _, df_c in df_e.groupby("pid")]
        if cargos:
            mov_dict[str(exp)] = cargos
    return sorted(mov_dict.keys()), mov_dict


def hier_bootstrap_ci(
    movies: List[str],
    mov_dict: Dict[str, List[np.ndarray]],
    stat_fn: Callable[[np.ndarray], np.ndarray],
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> Tuple[np.ndarray, np.ndarray]:
    """動画 -> 貨物の 2 段階階層ブートストラップにより 95% CI (2.5%, 97.5%) を返す。"""
    rng = np.random.default_rng(seed)
    n_m = len(movies)
    reps = []
    for _ in range(n_boot):
        picked = []
        for m in rng.choice(movies, size=n_m, replace=True):
            cargos = mov_dict[m]
            n_c = len(cargos)
            for idx in rng.integers(0, n_c, size=n_c):
                picked.append(cargos[idx])
        mat = np.vstack(picked)
        reps.append(stat_fn(mat))
    arr = np.asarray(reps, dtype=float)
    lo = np.nanpercentile(arr, 2.5, axis=0)
    hi = np.nanpercentile(arr, 97.5, axis=0)
    return lo, hi


def fmt_ci(val: float, lo: float, hi: float, fmt: str = ".3f") -> str:
    """値と 95% CI を整形して文字列化する。"""
    if not (np.isfinite(val) and np.isfinite(lo) and np.isfinite(hi)):
        return "nan"
    return f"{val:{fmt}} [{lo:{fmt}}, {hi:{fmt}}]"


def sample_mc_models(bead: str, rng: np.random.Generator, n_mc: int = N_MC) -> Tuple[np.ndarray, np.ndarray]:
    """S8.3 パラメータから対数正規モデル・ガンマモデルの |s * e_hat + n| (n ~ N(0, I_2)) を生成する。"""
    p = S83_PARAMS[bead]
    n_par = rng.standard_normal(n_mc)
    n_perp = rng.standard_normal(n_mc)

    s_ln = p["ln_med"] * np.exp(p["ln_sig"] * rng.standard_normal(n_mc))
    r_ln = np.hypot(s_ln + n_par, n_perp)

    k = 1.0 / (p["gam_cv"] ** 2)
    theta = p["gam_mean"] / k
    s_gam = rng.gamma(shape=k, scale=theta, size=n_mc)
    r_gam = np.hypot(s_gam + n_par, n_perp)

    return r_ln, r_gam


# ---------------------------------------------------------------------------
# 解析 A: 雑音が u_inf に比例するか
# ---------------------------------------------------------------------------
def run_part_a(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """A.1 超過尖度の比較 と A.2 Var(v_perp) vs u_inf^2 の 8 分位直線フィットを行う。"""
    kurt_rows = []
    fit_rows = []

    fig, axes = plt.subplots(2, 3, figsize=(14.5, 9.0))
    axes_flat = axes.flatten()

    for idx, b in enumerate(BEAD_ORDER):
        sub = df[df["bead_name"] == b]

        # A.1 超過尖度 (v_perp, w_perp, w_perp_ufp)
        cols_k = ["v_perp", "w_perp", "w_perp_ufp"]
        movies, mov_dict = build_hier_structure(sub, cols_k)

        def stat_kurt(mat: np.ndarray) -> np.ndarray:
            return np.array([stats.kurtosis(mat[:, j], fisher=True) for j in range(3)])

        pt_k = stat_kurt(sub[cols_k].values)
        lo_k, hi_k = hier_bootstrap_ci(movies, mov_dict, stat_kurt, seed=SEED + idx)

        kurt_rows.append({
            "bead_name": b,
            "kurt_vperp": pt_k[0], "kurt_vperp_lo": lo_k[0], "kurt_vperp_hi": hi_k[0],
            "kurt_wperp": pt_k[1], "kurt_wperp_lo": lo_k[1], "kurt_wperp_hi": hi_k[1],
            "kurt_wperp_ufp": pt_k[2], "kurt_wperp_ufp_lo": lo_k[2], "kurt_wperp_ufp_hi": hi_k[2],
        })

        # A.2 u_inf(t) 8 分位ビンで Var(v_perp) vs u_inf^2 直線フィット
        _, edges = pd.qcut(sub["u_inf_t"], q=8, retbins=True, duplicates="drop")
        edges[0] = -np.inf
        edges[-1] = np.inf
        n_b = len(edges) - 1

        def bin_stats(mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            u_vals = mat[:, 0]
            vp_vals = mat[:, 1]
            b_idx = np.digitize(u_vals, edges) - 1
            xs, ys, errs = [], [], []
            for bi in range(n_b):
                m = b_idx == bi
                if m.sum() >= 5:
                    u2 = u_vals[m] ** 2
                    vp = vp_vals[m]
                    xs.append(float(np.mean(u2)))
                    v_var = float(np.var(vp, ddof=1))
                    ys.append(v_var)
                    # ビン内分散の近似標準誤差 (描画用)
                    errs.append(float(v_var * np.sqrt(2.0 / (m.sum() - 1))))
            return np.array(xs), np.array(ys), np.array(errs)

        def stat_fit(mat: np.ndarray) -> np.ndarray:
            xs, ys, _ = bin_stats(mat)
            if len(xs) < 2:
                return np.array([np.nan, np.nan])
            slope, intercept = np.polyfit(xs, ys, 1)
            return np.array([intercept, slope])

        cols_f = ["u_inf_t", "v_perp"]
        movies_f, mov_dict_f = build_hier_structure(sub, cols_f)
        xb, yb, eb = bin_stats(sub[cols_f].values)
        pt_f = stat_fit(sub[cols_f].values)
        lo_f, hi_f = hier_bootstrap_ci(movies_f, mov_dict_f, stat_fit, seed=SEED + 100 + idx)

        sig_track = np.sqrt(max(0.0, pt_f[0]))
        k_act = np.sqrt(max(0.0, pt_f[1]))
        fit_rows.append({
            "bead_name": b,
            "intercept_sigma_track2": pt_f[0], "intercept_lo": lo_f[0], "intercept_hi": hi_f[0],
            "sigma_track": sig_track,
            "slope_k2": pt_f[1], "slope_lo": lo_f[1], "slope_hi": hi_f[1],
            "k_active": k_act,
        })

        # パネル描画
        ax = axes_flat[idx]
        ax.errorbar(xb, yb, yerr=eb, fmt="o", color="#1f77b4", capsize=3, markersize=6, label="Octile bins")
        x_line = np.linspace(0.0, xb.max() * 1.05, 100)
        ax.plot(x_line, pt_f[0] + pt_f[1] * x_line, "-", color="#d62728", lw=2.0, label="OLS fit")
        ax.set_title(BEAD_LABELS[b], fontsize=13, fontweight="bold")
        ax.set_xlabel(r"$\langle u_\infty(t)^2 \rangle\ [(\mu\mathrm{m/s})^2]$", fontsize=11)
        if idx % 3 == 0:
            ax.set_ylabel(r"$\mathrm{Var}(v_\perp)\ [(\mu\mathrm{m/s})^2]$", fontsize=11)
        ax.text(
            0.05, 0.92,
            f"$\\sigma_{{\\mathrm{{track}}}}^2 = {pt_f[0]:.4f}$\n$k^2 = {pt_f[1]:.3f}$ ($k={k_act:.2f}$)",
            transform=ax.transAxes, va="top", fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.9),
        )
        if idx == 0:
            ax.legend(loc="lower right", fontsize=9)

    fig.suptitle(r"Noise Scaling: $\mathrm{Var}(v_\perp)$ vs $\langle u_\infty(t)^2 \rangle$ (8-quantile bins)", fontsize=14.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "var_vperp_vs_uinf2.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    df_kurt = pd.DataFrame(kurt_rows)
    df_fit = pd.DataFrame(fit_rows)
    df_kurt.to_csv(OUT_DIR / "partA_kurtosis.csv", index=False)
    df_fit.to_csv(OUT_DIR / "partA_var_fit.csv", index=False)
    return df_kurt, df_fit


# ---------------------------------------------------------------------------
# 解析 B: ln w の幅の予測との比較
# ---------------------------------------------------------------------------
def run_part_b(df: pd.DataFrame) -> pd.DataFrame:
    """B.3 SD[ln w] (overall / within-cargo) と SD[ln |v|] をモンテカルロ予測値と比較する。"""
    rng = np.random.default_rng(SEED + 200)
    rows = []

    for idx, b in enumerate(BEAD_ORDER):
        sub = df[df["bead_name"] == b]
        cols = ["ln_w", "ln_w_wc", "ln_v", "ln_v_wc"]
        movies, mov_dict = build_hier_structure(sub, cols)

        def stat_sd(mat: np.ndarray) -> np.ndarray:
            return np.array([np.std(mat[:, j], ddof=1) for j in range(4)])

        pt = stat_sd(sub[cols].values)
        lo, hi = hier_bootstrap_ci(movies, mov_dict, stat_sd, seed=SEED + 300 + idx)

        r_ln, r_gam = sample_mc_models(b, rng)
        pred_ln = float(np.std(np.log(r_ln), ddof=1))
        pred_gam = float(np.std(np.log(r_gam), ddof=1))

        rows.append({
            "bead_name": b,
            "sd_ln_w_all": pt[0], "sd_ln_w_all_lo": lo[0], "sd_ln_w_all_hi": hi[0],
            "sd_ln_w_wc": pt[1], "sd_ln_w_wc_lo": lo[1], "sd_ln_w_wc_hi": hi[1],
            "sd_ln_v_all": pt[2], "sd_ln_v_all_lo": lo[2], "sd_ln_v_all_hi": hi[2],
            "sd_ln_v_wc": pt[3], "sd_ln_v_wc_lo": lo[3], "sd_ln_v_wc_hi": hi[3],
            "pred_lognorm_sd": pred_ln,
            "pred_gamma_sd": pred_gam,
        })

    df_b = pd.DataFrame(rows)
    df_b.to_csv(OUT_DIR / "partB_ln_sd.csv", index=False)
    return df_b


# ---------------------------------------------------------------------------
# 解析 C: P(s) を速さから直接決める・w の分布全体の比較
# ---------------------------------------------------------------------------
def run_part_c(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """C.4 キュムラント法による s の平均・CV・歪度、および C.5 ln w 分布の KS 距離比較。"""
    rng = np.random.default_rng(SEED + 400)
    cum_rows = []
    ks_rows = []

    fig, axes = plt.subplots(2, 3, figsize=(14.5, 9.0))
    axes_flat = axes.flatten()

    for idx, b in enumerate(BEAD_ORDER):
        sub = df[df["bead_name"] == b]
        p = S83_PARAMS[b]

        # 理論値: 対数正規
        e_sig2 = np.exp(p["ln_sig"] ** 2)
        ln_mean_th = p["ln_med"] * np.exp(0.5 * p["ln_sig"] ** 2)
        ln_cv_th = np.sqrt(e_sig2 - 1.0)
        ln_skew_th = (e_sig2 + 2.0) * ln_cv_th

        # 理論値: ガンマ
        gam_mean_th = p["gam_mean"]
        gam_cv_th = p["gam_cv"]
        gam_skew_th = 2.0 * p["gam_cv"]

        # C.4 キュムラント推定
        cols_c = ["w_par", "w_perp"]
        movies, mov_dict = build_hier_structure(sub, cols_c)

        def stat_cum(mat: np.ndarray) -> np.ndarray:
            w_par = mat[:, 0]
            w_perp = mat[:, 1]
            beta = np.std(w_perp, ddof=1)
            if beta <= 0:
                return np.full(5, np.nan)
            s_tilde = w_par / beta
            mean_s = float(np.mean(s_tilde))
            var_s = float(np.var(s_tilde, ddof=1) - 1.0)
            kap3_s = float(np.mean((s_tilde - mean_s) ** 3))
            if var_s > 0:
                cv_s = np.sqrt(var_s) / mean_s if mean_s != 0 else np.nan
                skew_s = kap3_s / (var_s ** 1.5)
            else:
                cv_s = np.nan
                skew_s = np.nan
            return np.array([beta, mean_s, var_s, cv_s, skew_s])

        pt_c = stat_cum(sub[cols_c].values)
        lo_c, hi_c = hier_bootstrap_ci(movies, mov_dict, stat_cum, seed=SEED + 500 + idx)

        # どちらに近いか (CV と Skew の距離比較、対数スケール距離または絶対誤差)
        closer_cv = "n/a"
        closer_skew = "n/a"
        if np.isfinite(pt_c[3]):
            closer_cv = "Lognormal" if abs(pt_c[3] - ln_cv_th) < abs(pt_c[3] - gam_cv_th) else "Gamma"
        if np.isfinite(pt_c[4]):
            closer_skew = "Lognormal" if abs(pt_c[4] - ln_skew_th) < abs(pt_c[4] - gam_skew_th) else "Gamma"

        cum_rows.append({
            "bead_name": b,
            "beta": pt_c[0], "beta_lo": lo_c[0], "beta_hi": hi_c[0],
            "mean_s": pt_c[1], "mean_s_lo": lo_c[1], "mean_s_hi": hi_c[1],
            "var_s": pt_c[2], "var_s_lo": lo_c[2], "var_s_hi": hi_c[2],
            "cv_s": pt_c[3], "cv_s_lo": lo_c[3], "cv_s_hi": hi_c[3],
            "skew_s": pt_c[4], "skew_s_lo": lo_c[4], "skew_s_hi": hi_c[4],
            "ln_mean_th": ln_mean_th, "ln_cv_th": ln_cv_th, "ln_skew_th": ln_skew_th,
            "gam_mean_th": gam_mean_th, "gam_cv_th": gam_cv_th, "gam_skew_th": gam_skew_th,
            "closer_cv": closer_cv, "closer_skew": closer_skew,
        })

        # C.5 ln w の分布全体比較 & KS 距離
        beta_pt = pt_c[0]
        r_ln, r_gam = sample_mc_models(b, rng)
        lnw_mod_ln = np.log(beta_pt * r_ln)
        lnw_mod_gam = np.log(beta_pt * r_gam)

        cols_ks = ["ln_w", "w_perp"]
        movies_ks, mov_dict_ks = build_hier_structure(sub, cols_ks)

        def stat_ks(mat: np.ndarray) -> np.ndarray:
            lnw_obs = mat[:, 0]
            b_rep = np.std(mat[:, 1], ddof=1)
            shift = np.log(b_rep / beta_pt) if (b_rep > 0 and beta_pt > 0) else 0.0
            d_ln = stats.ks_2samp(lnw_obs, lnw_mod_ln + shift).statistic
            d_gam = stats.ks_2samp(lnw_obs, lnw_mod_gam + shift).statistic
            return np.array([d_ln, d_gam])

        pt_ks = stat_ks(sub[cols_ks].values)
        lo_ks, hi_ks = hier_bootstrap_ci(movies_ks, mov_dict_ks, stat_ks, seed=SEED + 600 + idx)

        ks_rows.append({
            "bead_name": b,
            "ks_lognorm": pt_ks[0], "ks_lognorm_lo": lo_ks[0], "ks_lognorm_hi": hi_ks[0],
            "ks_gamma": pt_ks[1], "ks_gamma_lo": lo_ks[1], "ks_gamma_hi": hi_ks[1],
            "closer_ks": "Lognormal" if pt_ks[0] < pt_ks[1] else "Gamma",
        })

        # ヒストグラム描画
        ax = axes_flat[idx]
        lnw_obs_vals = sub["ln_w"].values
        xmin = min(lnw_obs_vals.min(), np.percentile(lnw_mod_ln, 0.5), np.percentile(lnw_mod_gam, 0.5))
        xmax = max(lnw_obs_vals.max(), np.percentile(lnw_mod_ln, 99.5), np.percentile(lnw_mod_gam, 99.5))
        bins = np.linspace(xmin, xmax, 45)
        ax.hist(lnw_obs_vals, bins=bins, density=True, color="#888888", alpha=0.45, edgecolor="#444444", label="Observed")

        # モデル密度曲線 (細かいビンでのステップ/補間)
        fine_bins = np.linspace(xmin, xmax, 80)
        centers = 0.5 * (fine_bins[:-1] + fine_bins[1:])
        h_ln, _ = np.histogram(lnw_mod_ln, bins=fine_bins, density=True)
        h_gam, _ = np.histogram(lnw_mod_gam, bins=fine_bins, density=True)
        ax.plot(centers, h_ln, "-", color="#d62728", lw=2.0, label=f"Lognorm (KS={pt_ks[0]:.3f})")
        ax.plot(centers, h_gam, "--", color="#1f77b4", lw=2.0, label=f"Gamma (KS={pt_ks[1]:.3f})")

        ax.set_title(f"{BEAD_LABELS[b]} ($\\beta={beta_pt:.3f}$)", fontsize=13, fontweight="bold")
        ax.set_xlabel(r"$\ln w = \ln(|\mathbf{v}| / u_\infty(t))$", fontsize=11)
        if idx % 3 == 0:
            ax.set_ylabel("Density", fontsize=11)
        ax.legend(fontsize=8.5, loc="upper left")

    fig.suptitle(r"Distribution of $\ln w$ vs Fixed-$P(s)$ Models ($w = \beta |s\hat{\mathbf{e}} + \mathbf{n}|$, $\beta = \mathrm{SD}(w_\perp)$)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "lnw_hist_vs_model.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    df_cum = pd.DataFrame(cum_rows)
    df_ks = pd.DataFrame(ks_rows)
    df_cum.to_csv(OUT_DIR / "partC_cumulants.csv", index=False)
    df_ks.to_csv(OUT_DIR / "partC_ks.csv", index=False)
    return df_cum, df_ks


# ---------------------------------------------------------------------------
# REPORT.md の生成
# ---------------------------------------------------------------------------
def write_report(
    excl: Dict,
    df_kurt: pd.DataFrame,
    df_fit: pd.DataFrame,
    df_b: pd.DataFrame,
    df_cum: pd.DataFrame,
    df_ks: pd.DataFrame,
) -> None:
    lines: List[str] = []
    lines.append("# 貨物の速さを $u_\\infty(t)$ で規格化したモデル検証レポート")
    lines.append("")
    lines.append("## 1. 仮定・単位・データ除外")
    lines.append("- **入力データ**: `scratch/contrast_inst_cache.pkl`（`harmonics_frames.pkl` の `(exp, pid, frame)` と完全一致する 27,969 フレーム）、`scratch/u_inf_per_frame_cache.pkl`。")
    lines.append(f"- **単位系**: `cargo_dx`, `cargo_dy`, `flow_u`, `flow_v` は px/frame であるため、`SCALE = {SCALE_UM_PER_PX} µm/px`, `DT = {DT_SEC} s/frame`（`V_UNIT = {V_UNIT:.4f} µm/s per px/frame`）を乗じて µm/s に変換し、µm/s 単位の `u_inf_t` と比をとることで無次元化（`w_par`, `w_perp`, `w`）。")
    lines.append("- **方向と成分の定義**: $\\hat{\\mathbf{e}} = (\\texttt{flow\\_u}, \\texttt{flow\\_v}) / |\\cdot|$、$v_\\parallel = \\mathbf{v}\\cdot\\hat{\\mathbf{e}}$、$v_\\perp = v_x e_y - v_y e_x$。")
    lines.append(f"- **除外条件**: 動画ごとの `u_inf_t` 下位 5% 以下、または $|\\mathbf{{v}}|=0$、または `u_inf_t` / `u_fp` 欠損・非正。")
    lines.append(f"  - 全フレーム数: {excl['n_initial']:,} → 解析対象: {excl['n_kept']:,}（除外率: {100*excl['excl_frac_total']:.2f}%；うち `u_inf_t` 下位5%: {excl['n_low5_uinf']:,}、$|\\mathbf{{v}}|=0$: {excl['n_zero_v']:,}、欠損: {excl['n_missing_uinf']:,}）。")
    lines.append("")
    lines.append("| サイズ | 除外前 | 除外後 | `u_inf` 下位5% | $|\\mathbf{v}|=0$ | 除外率 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for r in excl["per_bead"]:
        lines.append(f"| {r['bead_name']} | {r['n_before']:,} | {r['n_after']:,} | {r['n_low5']:,} | {r['n_zero_v']:,} | {100*r['excl_frac']:.2f}% |")
    lines.append("")
    lines.append("- **信頼区間**: すべてサイズごとに「動画 (`exp_dir`) → 貨物 (`pid`)」の 2 段階階層ブートストラップ（300 回）による 95% CI。")
    lines.append("")

    # Part A
    lines.append("## 2. 解析 A：雑音が $u_\\infty(t)$ に比例するか")
    lines.append("### A.1 超過尖度 (Excess Kurtosis) の比較")
    lines.append("| サイズ | $\\mathrm{Kurt}(v_\\perp)$ | $\\mathrm{Kurt}(w_\\perp)$ [$u_\\infty$規格化] | $\\mathrm{Kurt}(v_\\perp/|\\langle\\mathbf{u}\\rangle_{\\mathrm{fp}}|)$ |")
    lines.append("|---|---|---|---|")
    for _, r in df_kurt.iterrows():
        lines.append(
            f"| {r['bead_name']} | "
            f"{fmt_ci(r['kurt_vperp'], r['kurt_vperp_lo'], r['kurt_vperp_hi'], '.2f')} | "
            f"{fmt_ci(r['kurt_wperp'], r['kurt_wperp_lo'], r['kurt_wperp_hi'], '.2f')} | "
            f"{fmt_ci(r['kurt_wperp_ufp'], r['kurt_wperp_ufp_lo'], r['kurt_wperp_ufp_hi'], '.2f')} |"
        )
    lines.append("")

    lines.append("### A.2 $\\mathrm{Var}(v_\\perp) = \\sigma_{\\mathrm{track}}^2 + k^2 u_\\infty^2$（8 分位ビン直線フィット）")
    lines.append("図: `var_vperp_vs_uinf2.png`")
    lines.append("")
    lines.append("| サイズ | 切片 $\\sigma_{\\mathrm{track}}^2$ [95% CI] | $\\sigma_{\\mathrm{track}}$ (µm/s) | 傾き $k^2$ [95% CI] | $k$ |")
    lines.append("|---|---|---:|---|---:|")
    for _, r in df_fit.iterrows():
        lines.append(
            f"| {r['bead_name']} | "
            f"{fmt_ci(r['intercept_sigma_track2'], r['intercept_lo'], r['intercept_hi'], '.4f')} | "
            f"{r['sigma_track']:.4f} | "
            f"{fmt_ci(r['slope_k2'], r['slope_lo'], r['slope_hi'], '.3f')} | "
            f"{r['k_active']:.3f} |"
        )
    lines.append("")

    # Part B
    lines.append("## 3. 解析 B：$\\ln$ の幅の予測との比較（$N_{\\mathrm{MC}} = 2\\times 10^5$）")
    lines.append("| サイズ | $\\mathrm{SD}[\\ln w]$ (全体) | $\\mathrm{SD}[\\ln w]$ (貨物内) | $\\mathrm{SD}[\\ln|\\mathbf{v}|]$ (全体) | $\\mathrm{SD}[\\ln|\\mathbf{v}|]$ (貨物内) | 予測 (対数正規) | 予測 (ガンマ) |")
    lines.append("|---|---|---|---|---|---:|---:|")
    for _, r in df_b.iterrows():
        lines.append(
            f"| {r['bead_name']} | "
            f"{fmt_ci(r['sd_ln_w_all'], r['sd_ln_w_all_lo'], r['sd_ln_w_all_hi'], '.3f')} | "
            f"{fmt_ci(r['sd_ln_w_wc'], r['sd_ln_w_wc_lo'], r['sd_ln_w_wc_hi'], '.3f')} | "
            f"{fmt_ci(r['sd_ln_v_all'], r['sd_ln_v_all_lo'], r['sd_ln_v_all_hi'], '.3f')} | "
            f"{fmt_ci(r['sd_ln_v_wc'], r['sd_ln_v_wc_lo'], r['sd_ln_v_wc_hi'], '.3f')} | "
            f"{r['pred_lognorm_sd']:.3f} | {r['pred_gamma_sd']:.3f} |"
        )
    lines.append("")

    # Part C
    lines.append("## 4. 解析 C：$P(s)$ を速さから直接決める（向きのフィットとは独立）")
    lines.append("### C.4 キュムラント法（$\\beta = \\mathrm{SD}(w_\\perp)$）による $\\langle s\\rangle$, $\\mathrm{CV}(s)$, $\\mathrm{Skew}(s)$")
    lines.append("| サイズ | $\\beta$ | $\\langle s\\rangle$ [実測] | $\\langle s\\rangle$ [LN / Gam] | $\\mathrm{CV}(s)$ [実測] | $\\mathrm{CV}(s)$ [LN / Gam] | $\\mathrm{Skew}(s)$ [実測] | $\\mathrm{Skew}(s)$ [LN / Gam] | 近いモデル (CV / Skew) |")
    lines.append("|---|---:|---|---|---|---:|---|---:|---|")
    for _, r in df_cum.iterrows():
        lines.append(
            f"| {r['bead_name']} | {r['beta']:.3f} | "
            f"{fmt_ci(r['mean_s'], r['mean_s_lo'], r['mean_s_hi'], '.2f')} | {r['ln_mean_th']:.2f} / {r['gam_mean_th']:.2f} | "
            f"{fmt_ci(r['cv_s'], r['cv_s_lo'], r['cv_s_hi'], '.2f')} | {r['ln_cv_th']:.2f} / {r['gam_cv_th']:.2f} | "
            f"{fmt_ci(r['skew_s'], r['skew_s_lo'], r['skew_s_hi'], '.2f')} | {r['ln_skew_th']:.2f} / {r['gam_skew_th']:.2f} | "
            f"{r['closer_cv']} / {r['closer_skew']} |"
        )
    lines.append("")

    lines.append("### C.5 $\\ln w$ の分布全体と固定モデルとの KS 距離")
    lines.append("図: `lnw_hist_vs_model.png`")
    lines.append("")
    lines.append("| サイズ | KS (対数正規) [95% CI] | KS (ガンマ) [95% CI] | 近いモデル |")
    lines.append("|---|---|---|---|")
    for _, r in df_ks.iterrows():
        lines.append(
            f"| {r['bead_name']} | "
            f"{fmt_ci(r['ks_lognorm'], r['ks_lognorm_lo'], r['ks_lognorm_hi'], '.3f')} | "
            f"{fmt_ci(r['ks_gamma'], r['ks_gamma_lo'], r['ks_gamma_hi'], '.3f')} | "
            f"{r['closer_ks']} |"
        )
    lines.append("")

    # 判定の目安
    lines.append("## 5. 判定のまとめ")
    n_kurt_improved = int((df_kurt["kurt_wperp"].abs() < df_kurt["kurt_vperp"].abs()).sum())
    within_015_ln = (df_b["sd_ln_w_all"] - df_b["pred_lognorm_sd"]).abs() <= 0.15
    within_015_gam = (df_b["sd_ln_w_all"] - df_b["pred_gamma_sd"]).abs() <= 0.15
    n_within_any = int((within_015_ln | within_015_gam).sum())
    lines.append(f"- **超過尖度の減少**: 6 サイズ中 {n_kurt_improved} サイズで $|\\mathrm{{Kurt}}(w_\\perp)| < |\\mathrm{{Kurt}}(v_\\perp)|$。")
    lines.append(f"- **$\mathrm{{SD}}[\\ln w]$ の予測との一致 (±0.15 以内)**: 全体 $\\mathrm{{SD}}[\\ln w]$ が少なくとも一方のモデル予測の ±0.15 以内に収まったのは 6 サイズ中 {n_within_any} サイズ（対数正規: {int(within_015_ln.sum())}、ガンマ: {int(within_015_gam.sum())}）。")
    ks_gam_win = int((df_ks["closer_ks"] == "Gamma").sum())
    ks_ln_win = int((df_ks["closer_ks"] == "Lognormal").sum())
    lines.append(f"- **分布の形の判定**: KS 距離ベースでは Gamma が {ks_gam_win} サイズ、Lognormal が {ks_ln_win} サイズで優位。詳細な数値一致・乖離は上表を参照。")

    (OUT_DIR / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    print("Loading and filtering data...")
    df, excl = load_and_filter_data()
    print(f"Kept {excl['n_kept']:,} / {excl['n_initial']:,} frames (excluded {100*excl['excl_frac_total']:.2f}%).")

    print("Running Part A (kurtosis & Var(v_perp) vs u_inf^2)...")
    df_kurt, df_fit = run_part_a(df)

    print("Running Part B (SD[ln w] vs MC predictions)...")
    df_b = run_part_b(df)

    print("Running Part C (cumulants & ln w distribution KS comparison)...")
    df_cum, df_ks = run_part_c(df)

    print("Writing REPORT.md...")
    write_report(excl, df_kurt, df_fit, df_b, df_cum, df_ks)
    print(f"[Complete] Outputs saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
