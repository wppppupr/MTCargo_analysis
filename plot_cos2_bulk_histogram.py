#!/usr/bin/env python3
"""
plot_cos2_bulk_histogram.py

P(cos 2*Theta_bulk) のヒストグラム・確率密度分布を作成するスクリプト。
verify_cos_bulk_potential.py と完全に同一のデータセット (19,899 frames, 71 active cargos) を使用し、
クラスターブートストラップ (B=1000, 粒子単位) による 95% 信頼区間および
2次元等方分布の参照線 P_iso(c2) = 1 / (pi * sqrt(1 - c2^2)) と併せて描画する。

出力先:
  - figure/cos_bulk_analysis/
  - /mnt/NAS-Ebanaru/sasaki/MTsingleBeads/figure/cos_bulk_analysis/
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from verify_cos_decomposition import (
    CURRENT_DIR, ROOT_DIR, BEADS_INFO,
    setup_matplotlib, save_figure, load_active_particles,
    load_experiment_correlation_lengths, load_all_data
)

OUT_DIRS = [
    CURRENT_DIR / "figure" / "cos_bulk_analysis",
    ROOT_DIR / "figure" / "cos_bulk_analysis",
]


def main():
    setup_matplotlib()
    for d in OUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)

    active_set = load_active_particles(ROOT_DIR)
    xi_dict = load_experiment_correlation_lengths(ROOT_DIR, CURRENT_DIR)

    df_instant, _ = load_all_data(ROOT_DIR, active_set)
    if df_instant.empty:
        print("Error: No data loaded.")
        return

    df_instant["cargo_id"] = df_instant["exp_dir"].astype(str) + "_" + df_instant["particle"].astype(str)
    # cos(2 * Theta_bulk) = 2 * cos^2(Theta_bulk) - 1
    df_instant["cos2_bulk"] = np.clip(2.0 * (df_instant["cos_bulk"] ** 2) - 1.0, -1.0, 1.0)

    bins_c2 = np.linspace(-1.0, 1.0, 29)  # 28 bins (verify_cos_bulk_potential.py 準拠)
    bin_centers = 0.5 * (bins_c2[:-1] + bins_c2[1:])
    bin_width = bins_c2[1] - bins_c2[0]

    B_BOOT = 1000
    rng = np.random.default_rng(42)

    boot_dict = {}
    summary_rows = []

    for b_item in BEADS_INFO:
        bname = b_item["name"]
        sub = df_instant[df_instant["bead_name"] == bname]
        if sub.empty:
            continue

        c2_all = sub["cos2_bulk"].dropna().values
        P_obs, _ = np.histogram(c2_all, bins=bins_c2, density=True)

        cargos_data = [
            grp["cos2_bulk"].dropna().values
            for _, grp in sub.groupby("cargo_id")
            if len(grp["cos2_bulk"].dropna()) > 0
        ]
        n_cargos = len(cargos_data)

        boot_P = np.zeros((B_BOOT, len(bin_centers)), dtype=float)
        boot_mean = np.zeros(B_BOOT, dtype=float)
        for bi in range(B_BOOT):
            idx_c = rng.integers(0, n_cargos, size=n_cargos)
            pts = np.concatenate([cargos_data[i] for i in idx_c])
            pb, _ = np.histogram(pts, bins=bins_c2, density=True)
            boot_P[bi] = pb
            boot_mean[bi] = float(np.mean(pts))

        ci_low_p = np.percentile(boot_P, 2.5, axis=0)
        ci_high_p = np.percentile(boot_P, 97.5, axis=0)
        mean_c2 = float(np.mean(c2_all))
        mean_ci_lo = float(np.percentile(boot_mean, 2.5))
        mean_ci_hi = float(np.percentile(boot_mean, 97.5))
        frac_par = float(np.mean(c2_all > 0.5))
        frac_perp = float(np.mean(c2_all < -0.5))

        boot_dict[bname] = {
            "P_obs": P_obs,
            "ci_low_p": ci_low_p,
            "ci_high_p": ci_high_p,
            "mean_c2": mean_c2,
            "mean_ci_lo": mean_ci_lo,
            "mean_ci_hi": mean_ci_hi,
            "n_points": len(c2_all),
            "n_cargos": n_cargos,
        }

        summary_rows.append({
            "bead_name": bname,
            "label": b_item["label"],
            "n_cargos": n_cargos,
            "n_frames": len(c2_all),
            "mean_cos2_bulk": mean_c2,
            "mean_cos2_ci_low": mean_ci_lo,
            "mean_cos2_ci_high": mean_ci_hi,
            "frac_parallel_gt_0_5": frac_par,
            "frac_perp_lt_minus_0_5": frac_perp,
        })

    df_sum = pd.DataFrame(summary_rows)
    for d in OUT_DIRS:
        df_sum.to_csv(d / "cos2_bulk_histogram_summary.csv", index=False)

    print("=== P(cos 2*Theta_bulk) Summary ===")
    print(df_sum[["bead_name", "n_cargos", "n_frames", "mean_cos2_bulk", "mean_cos2_ci_low", "mean_cos2_ci_high", "frac_parallel_gt_0_5", "frac_perp_lt_minus_0_5"]].to_string(index=False))

    # 1. 6パネル ヒストグラム + 95% CI
    fig, axes = plt.subplots(2, 3, figsize=(16.0, 10.0), sharex=True, sharey=True)
    c_iso = np.linspace(-0.96, 0.96, 200)
    p_iso = 1.0 / (np.pi * np.sqrt(1.0 - c_iso**2))

    for idx, b_item in enumerate(BEADS_INFO):
        ax = axes.flatten()[idx]
        bname, col, lbl = b_item["name"], b_item["color"], b_item["label"]
        if bname not in boot_dict:
            continue
        res = boot_dict[bname]

        ax.bar(bin_centers, res["P_obs"], width=bin_width * 0.92, color=col, alpha=0.35,
               edgecolor=col, linewidth=0.9, zorder=2,
               label=rf"Histogram ($N={res['n_points']}$, $N_{{\mathrm{{cargo}}}}={res['n_cargos']}$)")
        ax.plot(bin_centers, res["P_obs"], color=col, linewidth=2.2, marker=b_item["marker"],
                markersize=4.5, zorder=4)
        ax.fill_between(bin_centers, res["ci_low_p"], res["ci_high_p"], color=col, alpha=0.20,
                        label=r"95% CI (Cluster Bootstrap)", zorder=3)

        ax.plot(c_iso, p_iso, color="#555555", linestyle=":", linewidth=1.5,
                label=r"Isotropic $1/(\pi\sqrt{1-c_2^2})$", zorder=1)
        ax.axvline(0.0, color="#888888", linestyle=":", linewidth=1.0, zorder=1)
        ax.axvline(res["mean_c2"], color=col, linestyle="--", linewidth=1.8,
                   label=rf"Mean $\langle\cos 2\Theta_{{\mathrm{{bulk}}}}\rangle = {res['mean_c2']:.2f}$", zorder=5)

        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(0.0, 3.0)
        ax.set_title(rf"{lbl}  ($\langle \cos 2\Theta_{{\mathrm{{bulk}}}} \rangle = {res['mean_c2']:.2f}$)",
                     fontsize=13.5, fontweight="bold")
        ax.set_xlabel(r"$\cos(2\Theta_{\mathrm{bulk}})$", fontsize=13)
        if idx % 3 == 0:
            ax.set_ylabel(r"Probability Density $P(\cos 2\Theta_{\mathrm{bulk}})$", fontsize=13)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper center", fontsize=8.8, framealpha=0.92)

    fig.suptitle(r"Probability Density Histogram of $P(\cos 2\Theta_{\mathrm{bulk}})$ (Cluster Bootstrap 95% CI)",
                 fontsize=16.5, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, "cos2_bulk_probability_density_6panels", OUT_DIRS)
    plt.close(fig)

    # 2. 1パネル重ね描き
    fig, ax = plt.subplots(figsize=(10.0, 7.0))
    ax.plot(c_iso, p_iso, color="#555555", linestyle="--", linewidth=1.5,
            label=r"Isotropic 2D $1/(\pi\sqrt{1-c_2^2})$", zorder=1)
    ax.axvline(0.0, color="#888888", linestyle=":", linewidth=1.2, zorder=1)
    for idx, b_item in enumerate(BEADS_INFO):
        bname, col, lbl = b_item["name"], b_item["color"], b_item["label"]
        if bname not in boot_dict:
            continue
        res = boot_dict[bname]
        ax.plot(bin_centers, res["P_obs"], marker=b_item["marker"], markersize=6, color=col,
                linewidth=2.2, label=rf"{lbl} ($\langle\cos 2\Theta\rangle={res['mean_c2']:.2f}$)", zorder=3 + idx)
        ax.fill_between(bin_centers, res["ci_low_p"], res["ci_high_p"], color=col, alpha=0.14, zorder=2)

    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(0.0, 3.0)
    ax.set_xlabel(r"$\cos(2\Theta_{\mathrm{bulk}})$", fontsize=14, labelpad=8)
    ax.set_ylabel(r"Probability Density $P(\cos 2\Theta_{\mathrm{bulk}})$", fontsize=14, labelpad=8)
    ax.set_title(r"Probability Density $P(\cos 2\Theta_{\mathrm{bulk}})$ Across Cargo Sizes",
                 fontsize=15, fontweight="bold", pad=12)
    ax.grid(True, linestyle=":", alpha=0.7)
    ax.legend(loc="upper center", fontsize=10.5, framealpha=0.92)
    fig.tight_layout()
    save_figure(fig, "cos2_bulk_probability_density_1panel", OUT_DIRS)
    plt.close(fig)

    print("Saved P(cos 2*Theta_bulk) histograms successfully!")


if __name__ == "__main__":
    main()
