#!/usr/bin/env python3
"""
calc_kl_angle_histograms.py

D_KL および D_MT を算出する際に用いた、B = 36 ビン（[-pi, pi]）上の
2つの角度確率密度分布:
  1. P(theta):     貨物進行方向と足元ピクセル流の相対角 theta = theta_v - theta_u
  2. P_MT(theta):  貨物円盤相当領域内の背景微小管流ベクトルと領域平均流の相対角
                   theta = theta_MT - theta_bulk
を集計・正規化し、粒子クラスターブートストラップによる P(theta) の 95% CI と
ともに出力 CSV（kl_angle_histograms_B36.csv）に保存する。

入力キャッシュ（クレンジング済み: beads1um_crop および beads1um001_crop pid=7 除外）:
  - scratch/per_point_v_dot_u_hist_cache.pkl
  - scratch/mt_disk_angle_hist_B36_cache.pkl
  - figure/per_point_v_dot_u/kl_divergence_decomposition_summary.csv
  - figure/per_point_v_dot_u/data_processing_inequality_summary.csv

出力先:
  - figure/per_point_v_dot_u/kl_angle_histograms_B36.csv
  - /mnt/NAS-Ebanaru/sasaki/MTsingleBeads/figure/per_point_v_dot_u/kl_angle_histograms_B36.csv
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from libs.experiment_cutoff import is_excluded_experiment, is_excluded_particle

ROOT_DIR = Path("/mnt/NAS-Ebanaru/sasaki/MTsingleBeads")
OUT_DIRS = [CURRENT_DIR / "figure" / "per_point_v_dot_u"]
if ROOT_DIR.exists() and ROOT_DIR != CURRENT_DIR:
    OUT_DIRS.append(ROOT_DIR / "figure" / "per_point_v_dot_u")

PT_CACHE = CURRENT_DIR / "scratch" / "per_point_v_dot_u_hist_cache.pkl"
MT_B36_CACHE = CURRENT_DIR / "scratch" / "mt_disk_angle_hist_B36_cache.pkl"
KL_CSV = CURRENT_DIR / "figure" / "per_point_v_dot_u" / "kl_divergence_decomposition_summary.csv"
DPI_CSV = CURRENT_DIR / "figure" / "per_point_v_dot_u" / "data_processing_inequality_summary.csv"

B_BINS = 36
BINS_TH = np.linspace(-np.pi, np.pi, B_BINS + 1)
TH_CENTERS = 0.5 * (BINS_TH[:-1] + BINS_TH[1:])
DTH = float(BINS_TH[1] - BINS_TH[0])
P_ISO = 1.0 / (2.0 * np.pi)
B_BOOT = 1000

BEADS_META = [
    ("beads06um", 0.63),
    ("beads1um", 1.18),
    ("beads3um", 3.37),
    ("beads5um", 5.00),
    ("beads7um", 7.24),
    ("beads20um", 20.0),
]


def kl_from_counts(counts: np.ndarray) -> float:
    tot = float(np.sum(counts))
    if tot <= 0.0:
        return np.nan
    p = counts.astype(float) / tot
    m = p > 0.0
    return float(np.sum(p[m] * np.log(B_BINS * p[m])))


def main():
    with open(PT_CACHE, "rb") as f:
        df_pt = pd.DataFrame(pickle.load(f))
    df_pt = df_pt[~df_pt["exp_dir"].astype(str).apply(is_excluded_experiment)].copy()
    df_pt = df_pt[
        ~df_pt.apply(lambda r: is_excluded_particle(str(r["exp_dir"]), r["particle_id"]), axis=1)
    ].copy()

    with open(MT_B36_CACHE, "rb") as f:
        mt_dict = pickle.load(f)

    df_kl = pd.read_csv(KL_CSV).set_index("bead_name")
    df_dpi = pd.read_csv(DPI_CSV).set_index("bead_name")

    rng = np.random.default_rng(42)
    rows = []

    print("=== KL Angle Histograms Summary (B = 36 bins over [-pi, pi]) ===")
    for bname, diam_um in BEADS_META:
        sub_pt = df_pt[df_pt["bead_name"] == bname]
        if sub_pt.empty:
            raise RuntimeError(f"No cargo histogram records for {bname}")

        ht_arr = np.vstack(sub_pt["hist_th"].values).astype(float)
        n_particles = len(sub_pt)
        counts_cargo = ht_arr.sum(axis=0)
        tot_cargo = float(counts_cargo.sum())
        prob_cargo = counts_cargo / tot_cargo
        pdf_cargo = prob_cargo / DTH

        # Particle-cluster bootstrap 95% CI for P(theta)
        boot_pdf = np.empty((B_BOOT, B_BINS), dtype=float)
        for bi in range(B_BOOT):
            idx = rng.integers(0, n_particles, size=n_particles)
            c_b = ht_arr[idx].sum(axis=0)
            boot_pdf[bi] = c_b / (float(c_b.sum()) * DTH)
        pdf_ci_lo = np.percentile(boot_pdf, 2.5, axis=0)
        pdf_ci_hi = np.percentile(boot_pdf, 97.5, axis=0)

        m_info = mt_dict[bname]
        counts_mt = np.asarray(m_info["hist_mt"], dtype=float)
        tot_mt = float(counts_mt.sum())
        prob_mt = counts_mt / tot_mt
        pdf_mt = prob_mt / DTH
        n_exp_mt = int(m_info.get("n_exp", len(m_info["exp_dmt"])))

        dkl_raw_calc = kl_from_counts(counts_cargo)
        dmt_calc = kl_from_counts(counts_mt)
        dkl_raw_ref = float(df_kl.loc[bname, "D_KL_raw"])
        dkl_corr_ref = float(df_kl.loc[bname, "D_KL_corr"])
        dmt_ref = float(df_dpi.loc[bname, "D_MT_total"])
        rc_over_xi = float(df_dpi.loc[bname, "rc_over_xi"])

        assert np.isclose(dkl_raw_calc, dkl_raw_ref, atol=1e-6), (
            f"{bname}: D_KL_raw mismatch {dkl_raw_calc} vs {dkl_raw_ref}"
        )
        assert np.isclose(dmt_calc, dmt_ref, atol=1e-6), (
            f"{bname}: D_MT mismatch {dmt_calc} vs {dmt_ref}"
        )

        print(
            f"  {bname:9s} ({diam_um:5.2f} um): n_p={n_particles:2d}, n_exp_mt={n_exp_mt}, "
            f"D_KL_raw={dkl_raw_calc:.4f}, D_MT={dmt_calc:.4f}, "
            f"P(0)={pdf_cargo[17:19].mean():.3f}, P_MT(0)={pdf_mt[17:19].mean():.3f} rad^-1"
        )

        for k in range(B_BINS):
            rows.append({
                "bead_name": bname,
                "diameter_um": float(diam_um),
                "rc_over_xi": rc_over_xi,
                "bin_idx": int(k),
                "theta_rad": float(TH_CENTERS[k]),
                "theta_deg": float(np.degrees(TH_CENTERS[k])),
                "theta_left_rad": float(BINS_TH[k]),
                "theta_right_rad": float(BINS_TH[k + 1]),
                "dtheta_rad": DTH,
                "n_particles": int(n_particles),
                "n_exp_mt": n_exp_mt,
                "counts_cargo": int(round(counts_cargo[k])),
                "prob_cargo": float(prob_cargo[k]),
                "pdf_cargo": float(pdf_cargo[k]),
                "pdf_cargo_ci_low": float(pdf_ci_lo[k]),
                "pdf_cargo_ci_high": float(pdf_ci_hi[k]),
                "counts_mt": int(round(counts_mt[k])),
                "prob_mt": float(prob_mt[k]),
                "pdf_mt": float(pdf_mt[k]),
                "p_iso": P_ISO,
                "D_KL_raw": dkl_raw_calc,
                "D_KL_corr": dkl_corr_ref,
                "D_MT_total": dmt_calc,
            })

    df_out = pd.DataFrame(rows)
    for d in OUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        out_path = d / "kl_angle_histograms_B36.csv"
        df_out.to_csv(out_path, index=False)
        print(f"Saved: {out_path} ({len(df_out)} rows)")


if __name__ == "__main__":
    main()
