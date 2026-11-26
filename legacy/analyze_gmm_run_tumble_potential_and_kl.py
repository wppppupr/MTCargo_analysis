#!/usr/bin/env python3
"""
analyze_gmm_run_tumble_potential_and_kl.py

GMM (gmm_cargo_analysis.py 準拠: K=2, ln(v/u_inf(t))) で分離した小粒子 (0.63 um, 1.18 um) の
Run / Tumble 各状態に対して、全体解析と同一の手法を適用する:
  1. フットプリント内全点 (v_hat_i . u_hat(r)) の有効ポテンシャル U_eff(c) と 95% CI
  2. 1次・2次モーメント (<cos theta>, <cos 2theta>) からのゼロフィッティング GvM 理論線 U_tilde_GvM(c) と RMSE
  3. 3項ピタゴラス KL ダイバージェンス分解:
       D_KL(P || P_iso) = D_KL^(1) (1次極性) + D_KL^(2) (2次ネマチック) + Delta D_>=3 (3次以上残差)
  4. ストラトノビッチ型カレントによる TUR 指標 Q(tau) (tau = 4, 8, 12, 20 s)

出力先:
  - figure/gmm_1d/ (および figure/per_point_v_dot_u/)
"""

import pickle
import time
import warnings
from pathlib import Path
from multiprocessing import Pool

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import verify_cos_decomposition as vcd
from gmm_cargo_analysis import (
    find_default_root,
    find_experiment_dirs,
    collect_bead_gmm_data,
    load_strict_active_particles,
)
from libs import gmm_cargo as gc
from calc_and_plot_per_point_v_dot_u import RAD_MAP, BEST_MAP, BINS_COS, BINS_TH
from calc_and_plot_kl_decomposition import compute_kl_terms
from plot_gvm_potential import solve_gvm_from_moments, u_eff_from_hist

warnings.filterwarnings("ignore")

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = find_default_root()
OUT_DIRS = [
    CURRENT_DIR / "figure" / "gmm_1d",
    CURRENT_DIR / "figure" / "per_point_v_dot_u",
]
if ROOT_DIR != CURRENT_DIR and ROOT_DIR.exists():
    OUT_DIRS.extend([
        ROOT_DIR / "figure" / "gmm_1d",
        ROOT_DIR / "figure" / "per_point_v_dot_u",
    ])

UINF_CACHE = CURRENT_DIR / "scratch" / "u_inf_per_frame_cache.pkl"
INST_CACHE = CURRENT_DIR / "scratch" / "contrast_inst_cache.pkl"
GMM_HIST_CACHE = CURRENT_DIR / "scratch" / "gmm_run_tumble_per_point_hist_cache.pkl"
GVM_ALL_CSV = CURRENT_DIR / "figure" / "per_point_v_dot_u" / "gvm_moments_and_potential_summary.csv"

PX_TO_UM = 0.11
DT_FRAME = 4.0


def build_gmm_labeled_frames() -> pd.DataFrame:
    """gmm_cargo_analysis.py と同一条件で 0.63 um, 1.18 um の全フレームに Run/Tumble ラベルを付与し contrast_inst_cache と結合"""
    df_uinf = pd.read_pickle(UINF_CACHE)
    strict_p = load_strict_active_particles()
    df_inst = pd.read_pickle(INST_CACHE)

    labeled_parts = []
    for bname in ["beads06um", "beads1um"]:
        edirs = find_experiment_dirs(ROOT_DIR, bname)
        X, _, df_obs = collect_bead_gmm_data(
            edirs, bead_name=bname, df_uinf=df_uinf, strict_particles=strict_p,
            tau=1, scale=0.11, frame_interval=4.0, epsilon=0.0,
        )
        gmm = gc.CargoGaussianGMM(n_components=2, covariance_type="full", epsilon=0.0, random_state=42).fit(X)
        df_obs["pred_state"] = gmm.predict(X)
        df_obs["state_label"] = df_obs["pred_state"].map({0: "Tumble", 1: "Run"})

        sub_inst = df_inst[df_inst["bead_name"] == bname].copy()
        merged = sub_inst.merge(
            df_obs[["exp_dir", "orig_particle", "frame", "pred_state", "state_label", "u_inf_t", "v_norm"]],
            left_on=["exp_dir", "particle", "frame"],
            right_on=["exp_dir", "orig_particle", "frame"],
            how="inner",
        )
        labeled_parts.append(merged)

    df_all = pd.concat(labeled_parts, ignore_index=True)
    print(f"Labeled frames matched with contrast_inst_cache: {len(df_all)}")
    for (b, s), grp in df_all.groupby(["bead_name", "state_label"]):
        print(f"  {b:10s} {s:6s}: {len(grp):4d} frames ({grp['exp_dir'].nunique()} exps)")
    return df_all


def _extract_exp_state_hists(args):
    exp_name, bname, exp_rows = args
    p_rad = RAD_MAP[bname]
    best_rad = BEST_MAP[bname]
    r_ceil = best_rad

    exp_dirs = vcd.find_experiment_dirs(ROOT_DIR, bname)
    ed_list = [d for d in exp_dirs if d.name == exp_name]
    if not ed_list:
        return []
    flow_p = ed_list[0] / "GFP_flows.h5"
    if not flow_p.exists():
        return []

    rec_map = {}
    with h5py.File(flow_p, "r", rdcc_nbytes=128 * 1024 * 1024) as f:
        flow = f[list(f.keys())[0]]
        total_frms = flow.shape[0]
        h, w = flow.shape[2], flow.shape[3]

        for frm, grp in exp_rows.groupby("frame"):
            frm_int = int(frm)
            if frm_int >= total_frms:
                continue

            for _, r in grp.iterrows():
                cdx = float(r["cargo_dx"])
                cdy = float(r["cargo_dy"])
                v_norm = np.hypot(cdx, cdy)
                if not np.isfinite(v_norm) or v_norm <= 0:
                    continue
                vx_hat = cdx / v_norm
                vy_hat = cdy / v_norm

                bx = int(round(r["x_px"]))
                by = int(round(r["y_px"]))
                y_min, y_max = max(0, by - r_ceil), min(h, by + r_ceil + 1)
                x_min, x_max = max(0, bx - r_ceil), min(w, bx + r_ceil + 1)

                patch = flow[frm_int, :, y_min:y_max, x_min:x_max]
                u_box, v_box = patch[0], patch[1]
                yg, xg = np.ogrid[y_min - by : y_max - by, x_min - bx : x_max - bx]
                dsq = xg**2 + yg**2
                m = (dsq <= best_rad**2) & (dsq > p_rad**2)
                if not np.any(m):
                    continue

                u_in = u_box[m].astype(np.float32)
                v_in = v_box[m].astype(np.float32)
                u_norm = np.hypot(u_in, v_in)
                valid = u_norm > 0
                if not np.any(valid):
                    continue

                ux_hat = u_in[valid] / u_norm[valid]
                uy_hat = v_in[valid] / u_norm[valid]
                c_pt = np.clip(vx_hat * ux_hat + vy_hat * uy_hat, -1.0, 1.0)
                cross_pt = ux_hat * vy_hat - uy_hat * vx_hat
                th_pt = np.arctan2(cross_pt, c_pt)

                hc, _ = np.histogram(c_pt, bins=BINS_COS)
                ht, _ = np.histogram(th_pt, bins=BINS_TH)

                pid = f"{exp_name}_{int(r['particle'])}"
                st = str(r["state_label"])
                key = (pid, st)
                if key not in rec_map:
                    rec_map[key] = {
                        "bead_name": bname,
                        "diameter_um": float(r["diameter_um"]),
                        "exp_dir": exp_name,
                        "particle_id": pid,
                        "state_label": st,
                        "rc_over_xi": float(r["rc_over_xi"]),
                        "hist_c": np.zeros(len(BINS_COS) - 1, dtype=np.int64),
                        "hist_th": np.zeros(len(BINS_TH) - 1, dtype=np.int64),
                        "n_pts": 0,
                        "n_frames": 0,
                        "sum_c": 0.0,
                        "sum_c2": 0.0,
                    }
                d = rec_map[key]
                d["hist_c"] += hc
                d["hist_th"] += ht
                d["n_pts"] += int(len(c_pt))
                d["n_frames"] += 1
                d["sum_c"] += float(np.sum(c_pt))
                d["sum_c2"] += float(np.sum(c_pt**2))

    return list(rec_map.values())


def extract_or_load_gmm_hists(df_labeled: pd.DataFrame, force: bool = False) -> pd.DataFrame:
    if GMM_HIST_CACHE.exists() and not force:
        print(f"Loading cached GMM state per-point histograms from {GMM_HIST_CACHE}...")
        with open(GMM_HIST_CACHE, "rb") as f:
            return pd.DataFrame(pickle.load(f))

    t0 = time.time()
    tasks = [(exp_name, bname, grp) for (bname, exp_name), grp in df_labeled.groupby(["bead_name", "exp_dir"])]
    print(f"Extracting state-resolved per-point v_hat . u_hat across {len(tasks)} experiments using Pool(8)...")
    with Pool(processes=8) as pool:
        nested = pool.map(_extract_exp_state_hists, tasks)
    records = [r for sub in nested for r in sub]
    with open(GMM_HIST_CACHE, "wb") as f:
        pickle.dump(records, f)
    print(f"Extracted {len(records)} particle-state records in {time.time() - t0:.1f}s.")
    return pd.DataFrame(records)


def compute_state_tur_stratonovich(df_labeled: pd.DataFrame, n_boot: int = 1000, seed: int = 42) -> pd.DataFrame:
    """
    状態連続セグメント（同一 state_label が連続し、かつ diff(frame)==1）上で
    ストラトノビッチ増分 Delta J_t から Q(tau) = 2 <J(tau)>^2 / Var(J(tau)) を算出。
    """
    rng = np.random.default_rng(seed)
    rows = []

    for bname, dia in [("beads06um", 0.63), ("beads1um", 1.18)]:
        sub_b = df_labeled[df_labeled["bead_name"] == bname].copy()
        sub_b["pid"] = sub_b["exp_dir"].astype(str) + "_" + sub_b["particle"].astype(str)
        f_norm = np.hypot(sub_b["flow_u"].values, sub_b["flow_v"].values)
        sub_b = sub_b[(f_norm > 0) & np.isfinite(sub_b["cargo_dx"].values) & np.isfinite(sub_b["cargo_dy"].values)].copy()
        f_norm = np.hypot(sub_b["flow_u"].values, sub_b["flow_v"].values)
        sub_b["ux"] = sub_b["flow_u"].values / f_norm
        sub_b["uy"] = sub_b["flow_v"].values / f_norm
        sub_b["dx_um"] = sub_b["cargo_dx"].values * PX_TO_UM
        sub_b["dy_um"] = sub_b["cargo_dy"].values * PX_TO_UM

        for st_label in ["All", "Run", "Tumble"]:
            sub_s = sub_b if st_label == "All" else sub_b[sub_b["state_label"] == st_label]
            # 連続セグメント抽出 (diff(frame) == 1)
            particles_segs = {}
            for pid, pg in sub_s.groupby("pid"):
                pg = pg.sort_values("frame")
                frm = pg["frame"].values
                ux, uy = pg["ux"].values, pg["uy"].values
                dx, dy = pg["dx_um"].values, pg["dy_um"].values
                splits = np.where(np.diff(frm) != 1)[0] + 1
                segs = []
                for idx_arr in np.split(np.arange(len(frm)), splits):
                    if len(idx_arr) >= 2:
                        i0, i1 = idx_arr[:-1], idx_arr[1:]
                        dJ = 0.5 * (ux[i0] + ux[i1]) * dx[i0] + 0.5 * (uy[i0] + uy[i1]) * dy[i0]
                        segs.append(dJ)
                if segs:
                    particles_segs[pid] = segs

            pids_list = list(particles_segs.keys())
            n_p = len(pids_list)

            for k_f in [1, 2, 3, 5]:  # tau = 4, 8, 12, 20 s
                tau_s = k_f * DT_FRAME
                p_arrs = []
                for pid in pids_list:
                    vals = []
                    for s in particles_segs[pid]:
                        if len(s) >= k_f:
                            cs = np.concatenate([[0.0], np.cumsum(s)])
                            vals.append(cs[k_f:] - cs[:-k_f])
                    if vals:
                        p_arrs.append(np.concatenate(vals))
                if not p_arrs:
                    continue
                j_all = np.concatenate(p_arrs)
                if len(j_all) < 10:
                    continue

                mean_j = float(np.mean(j_all))
                var_j = float(np.var(j_all, ddof=1))
                q_obs = float(2.0 * (mean_j ** 2) / var_j) if var_j > 0 else np.nan
                rate_obs = q_obs / tau_s if np.isfinite(q_obs) else np.nan

                # ブートストラップ 95% CI
                boot_q = []
                n_pa = len(p_arrs)
                for _ in range(n_boot):
                    idx = rng.integers(0, n_pa, size=n_pa)
                    jb = np.concatenate([p_arrs[i] for i in idx])
                    if len(jb) < 5:
                        continue
                    vb = float(np.var(jb, ddof=1))
                    if vb > 0:
                        boot_q.append(float(2.0 * (float(np.mean(jb)) ** 2) / vb))

                rows.append({
                    "bead_name": bname,
                    "diameter_um": dia,
                    "state_label": st_label,
                    "tau_s": tau_s,
                    "n_particles": n_p,
                    "n_samples": int(len(j_all)),
                    "mean_J_um": mean_j,
                    "var_J_um2": var_j,
                    "Q": q_obs,
                    "Q_ci_low": float(np.percentile(boot_q, 2.5)) if boot_q else np.nan,
                    "Q_ci_high": float(np.percentile(boot_q, 97.5)) if boot_q else np.nan,
                    "rate_TUR_s_inv": rate_obs,
                })

    return pd.DataFrame(rows)


def main():
    vcd.setup_matplotlib()
    for d in OUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)

    df_labeled = build_gmm_labeled_frames()
    df_rec = extract_or_load_gmm_hists(df_labeled, force=False)

    c_centers = 0.5 * (BINS_COS[:-1] + BINS_COS[1:])
    dc = float(BINS_COS[1] - BINS_COS[0])
    dth_exact = np.arccos(BINS_COS[:-1]) - np.arccos(BINS_COS[1:])
    p_iso_k = dth_exact / (np.pi * dc)
    c_fine = np.linspace(-0.99, 0.99, 400)

    n_boot = 1000
    rng = np.random.default_rng(42)

    gvm_rows = []
    panel_cache = {}

    combos = [
        ("beads06um", 0.63, "Run",    "#1b9e77", "^"),
        ("beads06um", 0.63, "Tumble", "#d95f02", "v"),
        ("beads1um",  1.18, "Run",    "#1b9e77", "o"),
        ("beads1um",  1.18, "Tumble", "#d95f02", "s"),
    ]

    for bname, dia, st_label, col, marker in combos:
        sub = df_rec[(df_rec["bead_name"] == bname) & (df_rec["state_label"] == st_label)]
        n_p = len(sub)
        rc_xi = float(sub["rc_over_xi"].mean())
        hc_arr = np.vstack(sub["hist_c"].values).astype(float)
        ht_arr = np.vstack(sub["hist_th"].values).astype(float)
        sumc_arr = sub["sum_c"].values.astype(float)
        sumc2_arr = sub["sum_c2"].values.astype(float)
        npts_arr = sub["n_pts"].values.astype(float)
        nfrm_tot = int(sub["n_frames"].sum())

        tot_n = float(npts_arr.sum())
        tot_sc = float(sumc_arr.sum())
        m1_obs = float(tot_sc / tot_n)
        m2_obs = float(2.0 * (sumc2_arr.sum() / tot_n) - 1.0)

        k1_obs, k2_obs, log_z_obs = solve_gvm_from_moments(m1_obs, m2_obs)
        offset_obs = k2_obs + log_z_obs - np.log(2.0 * np.pi)
        u_gvm_fine = -2.0 * k2_obs * (c_fine ** 2) - k1_obs * c_fine + offset_obs
        u_gvm_bins = -2.0 * k2_obs * (c_centers ** 2) - k1_obs * c_centers + offset_obs

        u_obs = u_eff_from_hist(hc_arr.sum(axis=0), p_iso_k, dc)
        valid_u = np.isfinite(u_obs)
        rmse_u = float(np.sqrt(np.mean((u_obs[valid_u] - u_gvm_bins[valid_u]) ** 2)))

        kl_obs = compute_kl_terms(ht_arr.sum(axis=0), tot_sc, tot_n)
        d_tot_obs = float(kl_obs["D_KL"])
        d1_obs = float(kl_obs["D_KL_1"])
        d_gvm_obs = float(k1_obs * m1_obs + k2_obs * m2_obs - (log_z_obs - np.log(2.0 * np.pi)))
        d2_obs = max(0.0, d_gvm_obs - d1_obs)
        d3_obs = max(0.0, d_tot_obs - d_gvm_obs)

        # クラスターブートストラップ 95% CI
        boot_u = np.full((n_boot, len(c_centers)), np.nan)
        boot_m1, boot_m2, boot_k1, boot_k2 = [], [], [], []
        boot_d1, boot_d2, boot_d3, boot_dtot, boot_rmse = [], [], [], [], []
        for bi in range(n_boot):
            idx = rng.integers(0, n_p, size=n_p)
            nb = float(npts_arr[idx].sum())
            if nb <= 0:
                continue
            sc_b = float(sumc_arr[idx].sum())
            ub = u_eff_from_hist(hc_arr[idx].sum(axis=0), p_iso_k, dc)
            boot_u[bi] = ub
            m1_b = float(sc_b / nb)
            m2_b = float(2.0 * (sumc2_arr[idx].sum() / nb) - 1.0)
            k1_b, k2_b, lz_b = solve_gvm_from_moments(m1_b, m2_b)
            boot_m1.append(m1_b)
            boot_m2.append(m2_b)
            boot_k1.append(k1_b)
            boot_k2.append(k2_b)

            off_b = k2_b + lz_b - np.log(2.0 * np.pi)
            ug_b = -2.0 * k2_b * (c_centers ** 2) - k1_b * c_centers + off_b
            vu = np.isfinite(ub)
            if np.any(vu):
                boot_rmse.append(float(np.sqrt(np.mean((ub[vu] - ug_b[vu]) ** 2))))

            kl_b = compute_kl_terms(ht_arr[idx].sum(axis=0), sc_b, nb)
            if kl_b is not None:
                dt_b = float(kl_b["D_KL"])
                d1_b = float(kl_b["D_KL_1"])
                dg_b = float(k1_b * m1_b + k2_b * m2_b - (lz_b - np.log(2.0 * np.pi)))
                boot_dtot.append(dt_b)
                boot_d1.append(d1_b)
                boot_d2.append(max(0.0, dg_b - d1_b))
                boot_d3.append(max(0.0, dt_b - dg_b))

        panel_cache[(bname, st_label)] = {
            "u_obs": u_obs,
            "u_lo": np.nanpercentile(boot_u, 2.5, axis=0),
            "u_hi": np.nanpercentile(boot_u, 97.5, axis=0),
            "u_gvm_fine": u_gvm_fine,
            "m1": m1_obs,
            "m2": m2_obs,
            "k1": k1_obs,
            "k2": k2_obs,
            "rmse_u": rmse_u,
            "n_frames": nfrm_tot,
            "n_particles": n_p,
            "col": col,
            "marker": marker,
            "dia": dia,
        }

        gvm_rows.append({
            "bead_name": bname,
            "diameter_um": dia,
            "state_label": st_label,
            "rc_over_xi": rc_xi,
            "n_particles": n_p,
            "n_frames": nfrm_tot,
            "n_flow_points": int(tot_n),
            "m1_cos_theta": m1_obs,
            "m1_ci_low": float(np.percentile(boot_m1, 2.5)),
            "m1_ci_high": float(np.percentile(boot_m1, 97.5)),
            "m2_cos_2theta": m2_obs,
            "m2_ci_low": float(np.percentile(boot_m2, 2.5)),
            "m2_ci_high": float(np.percentile(boot_m2, 97.5)),
            "kappa_1_gvm": k1_obs,
            "kappa_1_ci_low": float(np.percentile(boot_k1, 2.5)),
            "kappa_1_ci_high": float(np.percentile(boot_k1, 97.5)),
            "kappa_2_gvm": k2_obs,
            "kappa_2_ci_low": float(np.percentile(boot_k2, 2.5)),
            "kappa_2_ci_high": float(np.percentile(boot_k2, 97.5)),
            "rmse_U_obs_vs_GvM": rmse_u,
            "rmse_ci_low": float(np.percentile(boot_rmse, 2.5)),
            "rmse_ci_high": float(np.percentile(boot_rmse, 97.5)),
            "D_KL_total": d_tot_obs,
            "D_KL_total_ci_low": float(np.percentile(boot_dtot, 2.5)),
            "D_KL_total_ci_high": float(np.percentile(boot_dtot, 97.5)),
            "D_KL_1_polar": d1_obs,
            "D_KL_1_ci_low": float(np.percentile(boot_d1, 2.5)),
            "D_KL_1_ci_high": float(np.percentile(boot_d1, 97.5)),
            "D_KL_2_nematic": d2_obs,
            "D_KL_2_ci_low": float(np.percentile(boot_d2, 2.5)),
            "D_KL_2_ci_high": float(np.percentile(boot_d2, 97.5)),
            "Delta_D_ge3_residual": d3_obs,
            "Delta_D_ge3_ci_low": float(np.percentile(boot_d3, 2.5)),
            "Delta_D_ge3_ci_high": float(np.percentile(boot_d3, 97.5)),
        })

    df_gmm_gvm = pd.DataFrame(gvm_rows)
    df_tur = compute_state_tur_stratonovich(df_labeled, n_boot=1000, seed=42)

    for d in OUT_DIRS:
        df_gmm_gvm.to_csv(d / "gmm_run_tumble_gvm_kl_summary.csv", index=False)
        df_tur.to_csv(d / "gmm_run_tumble_tur_summary.csv", index=False)

    print("\n=== GMM Run vs Tumble: GvM & 3-Term KL Decomposition ===")
    print(df_gmm_gvm[["bead_name", "state_label", "n_frames", "m1_cos_theta", "m2_cos_2theta",
                      "kappa_1_gvm", "kappa_2_gvm", "rmse_U_obs_vs_GvM",
                      "D_KL_total", "D_KL_1_polar", "D_KL_2_nematic", "Delta_D_ge3_residual"]].to_string(index=False))

    print("\n=== GMM Run vs Tumble: Stratonovich TUR Q(tau=8s) ===")
    print(df_tur[df_tur["tau_s"] == 8.0][["bead_name", "state_label", "n_samples", "mean_J_um",
                                          "var_J_um2", "Q", "Q_ci_low", "Q_ci_high", "rate_TUR_s_inv"]].to_string(index=False))

    # ------------------------------------------------------------------
    # Fig 1: 4パネル 実測 U_eff(c) vs ゼロフィッティング GvM (0.63/1.18 um x Run/Tumble)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 9.6), sharex=True, sharey=True)
    for idx, (bname, dia, st_label, col, marker) in enumerate(combos):
        ax = axes.flatten()[idx]
        pc = panel_cache[(bname, st_label)]

        ax.axhline(0.0, color="#888888", linestyle="--", linewidth=0.9, zorder=1)
        ax.axvline(0.0, color="#888888", linestyle=":", linewidth=0.9, zorder=1)

        valid = np.isfinite(pc["u_obs"])
        ax.plot(c_centers[valid], pc["u_obs"][valid], color=col, marker=marker, markersize=5.2,
                linewidth=1.9, label=rf"Observed $\tilde{{U}}_{{\mathrm{{eff}}}}(c)$ ({pc['n_frames']} frm)", zorder=4)
        ax.fill_between(c_centers[valid], pc["u_lo"][valid], pc["u_hi"][valid],
                        color=col, alpha=0.20, label=r"95% CI (Bootstrap)", zorder=2)
        ax.plot(c_fine, pc["u_gvm_fine"], color="#111111", linestyle="--", linewidth=2.2,
                label=rf"GvM theory ($\kappa_1={pc['k1']:.2f}, \kappa_2={pc['k2']:.2f}$)", zorder=5)

        ax.text(
            0.03, 0.05,
            rf"$\langle\cos\theta\rangle={pc['m1']:.3f}$" + "\n" + rf"$\langle\cos 2\theta\rangle={pc['m2']:.3f}$",
            transform=ax.transAxes, fontsize=9.5, va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cccccc", alpha=0.9),
            zorder=6,
        )

        ax.set_xlim(-1.02, 1.02)
        ax.set_ylim(-2.15, 1.95)
        ax.set_title(rf"${dia:.2f}\,\mu\mathrm{{m}}$ — {st_label} ($\mathrm{{RMSE}}={pc['rmse_u']:.3f}$)",
                     fontsize=13.0, fontweight="bold")
        if idx >= 2:
            ax.set_xlabel(r"Per-Point Alignment $c = \hat{\mathbf{v}}_i \cdot \hat{\mathbf{u}}(\mathbf{r})$", fontsize=12.5)
        if idx % 2 == 0:
            ax.set_ylabel(r"Effective Potential $\tilde{U}_{\mathrm{eff}}(c)$", fontsize=12.5)
        ax.grid(True, linestyle=":", alpha=0.55)
        ax.legend(loc="upper right", fontsize=8.8, framealpha=0.92)

    fig.suptitle(r"GMM State-Resolved Potential $\tilde{U}_{\mathrm{eff}}(c)$ vs Zero-Fitting GvM Prediction (Run vs Tumble)",
                 fontsize=14.2, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    vcd.save_figure(fig, "per_point_cos_potential_gvm_gmm_run_tumble_4panels", OUT_DIRS)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Fig 2: 2x2 まとめ (All / Run / Tumble の比較 + 大粒子参照)
    # ------------------------------------------------------------------
    df_all_gvm = pd.read_csv(GVM_ALL_CSV)
    # 比較カテゴリリスト
    cats = []
    for bname, lbl_short in [("beads06um", "0.63"), ("beads1um", "1.18")]:
        r_all = df_all_gvm[df_all_gvm["bead_name"] == bname].iloc[0].to_dict()
        r_all["cat_label"] = f"{lbl_short}\n(All)"
        r_all["group_type"] = "All"
        cats.append(r_all)
        for st_lbl in ["Run", "Tumble"]:
            r_st = df_gmm_gvm[(df_gmm_gvm["bead_name"] == bname) & (df_gmm_gvm["state_label"] == st_lbl)].iloc[0].to_dict()
            r_st["cat_label"] = f"{lbl_short}\n({st_lbl})"
            r_st["group_type"] = st_lbl
            cats.append(r_st)
    for bname, lbl_short in [("beads3um", "3.37"), ("beads7um", "7.24"), ("beads5um", "5.00"), ("beads20um", "20.0")]:
        r_ref = df_all_gvm[df_all_gvm["bead_name"] == bname].iloc[0].to_dict()
        r_ref["cat_label"] = f"{lbl_short}\n(All)"
        r_ref["group_type"] = "LargeRef"
        cats.append(r_ref)

    df_cat = pd.DataFrame(cats)
    x_pos = np.arange(len(df_cat))

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.2))
    axA, axB, axC, axD = axes.flatten()

    # (A) <cos theta> & <cos 2theta>
    w_bar = 0.36
    axA.bar(x_pos - w_bar/2, df_cat["m1_cos_theta"], width=w_bar, color="#1f77b4", edgecolor="k", linewidth=0.6, label=r"$\langle\cos\theta\rangle$")
    axA.bar(x_pos + w_bar/2, df_cat["m2_cos_2theta"], width=w_bar, color="#d62728", edgecolor="k", linewidth=0.6, label=r"$\langle\cos 2\theta\rangle$")
    axA.axvline(2.5, color="#888888", ls="--", lw=1.0)
    axA.axvline(5.5, color="#333333", ls=":", lw=1.3)
    axA.axhline(0.0, color="#666666", lw=0.8)
    axA.set_xticks(x_pos)
    axA.set_xticklabels(df_cat["cat_label"], fontsize=8.8)
    axA.set_ylabel(r"Moments ($\langle\cos\theta\rangle, \langle\cos 2\theta\rangle$)", fontsize=11.5)
    axA.set_title(r"(A) 1st & 2nd Moments: GMM Run vs Tumble", fontsize=12.5, fontweight="bold")
    axA.set_ylim(-0.02, 0.82)
    axA.grid(True, axis="y", linestyle=":", alpha=0.55)
    axA.legend(loc="upper right", fontsize=9.5, framealpha=0.92)

    # (B) GvM (kappa_1, kappa_2)
    axB.bar(x_pos - w_bar/2, df_cat["kappa_1_gvm"], width=w_bar, color="#1f77b4", edgecolor="k", linewidth=0.6, label=r"Polar $\kappa_1$")
    axB.bar(x_pos + w_bar/2, df_cat["kappa_2_gvm"], width=w_bar, color="#d62728", edgecolor="k", linewidth=0.6, label=r"Nematic $\kappa_2$")
    axB.axvline(2.5, color="#888888", ls="--", lw=1.0)
    axB.axvline(5.5, color="#333333", ls=":", lw=1.3)
    axB.axhline(0.0, color="#666666", lw=0.8)
    axB.set_xticks(x_pos)
    axB.set_xticklabels(df_cat["cat_label"], fontsize=8.8)
    axB.set_ylabel(r"GvM Couplings ($\kappa_1, \kappa_2$)", fontsize=11.5)
    axB.set_title(r"(B) Moment-Matched GvM Couplings", fontsize=12.5, fontweight="bold")
    axB.set_ylim(-0.05, 1.50)
    axB.grid(True, axis="y", linestyle=":", alpha=0.55)
    axB.legend(loc="upper right", fontsize=9.5, framealpha=0.92)

    # (C) GvM Potential RMSE
    bar_cols = ["#882255" if t == "All" else ("#1b9e77" if t == "Run" else ("#d95f02" if t == "Tumble" else "#777777")) for t in df_cat["group_type"]]
    axC.bar(x_pos, df_cat["rmse_U_obs_vs_GvM"], width=0.58, color=bar_cols, edgecolor="k", linewidth=0.7)
    for i, r in df_cat.iterrows():
        yerr = [[max(0.0, r["rmse_U_obs_vs_GvM"] - r["rmse_ci_low"])], [max(0.0, r["rmse_ci_high"] - r["rmse_U_obs_vs_GvM"])]]
        axC.errorbar(i, r["rmse_U_obs_vs_GvM"], yerr=yerr, fmt="none", ecolor="#111111", capsize=3.5, elinewidth=1.4)
    axC.axvline(2.5, color="#888888", ls="--", lw=1.0)
    axC.axvline(5.5, color="#333333", ls=":", lw=1.3)
    axC.set_xticks(x_pos)
    axC.set_xticklabels(df_cat["cat_label"], fontsize=8.8)
    axC.set_ylabel(r"Potential Residual $\mathrm{RMSE}(\tilde{U}_{\mathrm{obs}} - \tilde{U}_{\mathrm{GvM}})$", fontsize=11.5)
    axC.set_title(r"(C) Zero-Fitting GvM Potential Residual (RMSE)", fontsize=12.5, fontweight="bold")
    axC.set_ylim(0.0, 0.90)
    axC.grid(True, axis="y", linestyle=":", alpha=0.55)

    # (D) 3-Term KL Decomposition Stacked Bar
    d1 = df_cat["D_KL_1_polar"].values
    d2 = df_cat["D_KL_2_nematic"].values
    d3 = df_cat["Delta_D_ge3_residual"].values
    axD.bar(x_pos, d1, width=0.58, color="#1f77b4", edgecolor="k", linewidth=0.6, label=r"$D_{\mathrm{KL}}^{(1)}$ (1st polar)")
    axD.bar(x_pos, d2, width=0.58, bottom=d1, color="#d95f02", edgecolor="k", linewidth=0.6, label=r"$D_{\mathrm{KL}}^{(2)}$ (2nd nematic)")
    axD.bar(x_pos, d3, width=0.58, bottom=d1 + d2, color="#7570b3", edgecolor="k", linewidth=0.6, label=r"$\Delta D_{\geq 3}$ ($\geq 3$rd residual)")
    for i, r in df_cat.iterrows():
        tot_i = float(r["D_KL_total"])
        if tot_i > 0.02:
            p1 = 100.0 * r["D_KL_1_polar"] / tot_i
            p2 = 100.0 * r["D_KL_2_nematic"] / tot_i
            p3 = 100.0 * r["Delta_D_ge3_residual"] / tot_i
            axD.text(i, tot_i + 0.015, f"{p1:.0f}/{p2:.0f}/{p3:.0f}%", ha="center", va="bottom", fontsize=7.6, fontweight="bold")
    axD.axvline(2.5, color="#888888", ls="--", lw=1.0)
    axD.axvline(5.5, color="#333333", ls=":", lw=1.3)
    axD.set_xticks(x_pos)
    axD.set_xticklabels(df_cat["cat_label"], fontsize=8.8)
    axD.set_ylabel(r"KL Divergence Contributions [nats]", fontsize=11.5)
    axD.set_title(r"(D) 3-Term Orthogonal KL Decomposition ($D_{\mathrm{KL}}^{(1)} / D_{\mathrm{KL}}^{(2)} / \Delta D_{\geq 3}$)", fontsize=12.5, fontweight="bold")
    axD.set_ylim(0.0, 0.92)
    axD.grid(True, axis="y", linestyle=":", alpha=0.55)
    axD.legend(loc="upper right", fontsize=9.0, framealpha=0.92)

    fig.suptitle(r"GMM Run / Tumble Separation for Small Cargos ($0.63\,\mu\mathrm{m}, 1.18\,\mu\mathrm{m}$) vs Large Cargos",
                 fontsize=14.5, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    vcd.save_figure(fig, "gmm_run_tumble_kl_and_gvm_summary", OUT_DIRS)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Fig 3: Stratonovich TUR Q(tau) for GMM Run vs Tumble (0.63 um, 1.18 um)
    # ------------------------------------------------------------------
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.2, 5.3), sharey=True)
    st_styles = {
        "All":    ("#882255", "o", "-"),
        "Run":    ("#1b9e77", "^", "-"),
        "Tumble": ("#d95f02", "s", "--"),
    }
    for ax, (bname, dia) in zip([axL, axR], [("beads06um", 0.63), ("beads1um", 1.18)]):
        for st_lbl, (col, mk, ls) in st_styles.items():
            sub_t = df_tur[(df_tur["bead_name"] == bname) & (df_tur["state_label"] == st_lbl)].sort_values("tau_s")
            if sub_t.empty:
                continue
            lo_err = np.maximum(0.0, sub_t["Q"] - np.maximum(1e-3, sub_t["Q_ci_low"]))
            hi_err = np.maximum(0.0, sub_t["Q_ci_high"] - sub_t["Q"])
            ax.errorbar(
                sub_t["tau_s"], sub_t["Q"],
                yerr=[lo_err, hi_err],
                fmt=mk + ls, color=col, linewidth=2.1, markersize=7.5,
                markeredgecolor="k", markeredgewidth=0.6, capsize=3.5,
                label=rf"{st_lbl} ($\mathcal{{Q}}(8\,\mathrm{{s}})={sub_t.loc[sub_t['tau_s']==8.0, 'Q'].values[0]:.2f}$)",
                zorder=4,
            )
        ax.set_yscale("log")
        ax.set_xlim(2.0, 22.0)
        ax.set_ylim(0.02, 120.0)
        ax.set_xlabel(r"Observation Window $\tau$ [s]", fontsize=12.5)
        ax.set_title(rf"${dia:.2f}\,\mu\mathrm{{m}}$ — Stratonovich TUR $\mathcal{{Q}}(\tau)$", fontsize=13.0, fontweight="bold")
        ax.grid(True, which="both", linestyle=":", alpha=0.55)
        ax.legend(loc="upper left", fontsize=9.2, framealpha=0.92)

    axL.set_ylabel(r"TUR Precision Index $\mathcal{Q}(\tau) = 2\langle J(\tau)\rangle^2 / \mathrm{Var}(J(\tau))$", fontsize=12.0)
    fig.suptitle(r"GMM State-Resolved Stratonovich TUR $\mathcal{Q}(\tau)$: Run vs Tumble (95% Bootstrap CI)",
                 fontsize=14.0, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    vcd.save_figure(fig, "gmm_run_tumble_tur_q_comparison", OUT_DIRS)
    plt.close(fig)

    print("\nSaved all GMM Run/Tumble potential, GvM, KL, and TUR figures successfully!")


if __name__ == "__main__":
    main()
