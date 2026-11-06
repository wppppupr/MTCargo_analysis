#!/usr/bin/env python3
"""
eta_vs_vpar_uinf_noise.py
=========================
【4.4】u_∞(t) 版で、2 つの「無次元化した輸送効率」の雑音を比べる。

  A: v_∥/u_∞  = Σ v_∥ / Σ u_∞(t)          (u_∞(t) は load_data(u_inf_mode='frame'))
  B: η        = Σ v_∥ / Σ |⟨u⟩_fp|        (u_fp: 足印円環内の平均流速ベクトルの大きさ)

部分集合: 全フレーム / Φ_fp > 0.95 のフレームのみ。
サイズごとに
  - 点推定 (全フレームの和の比)
  - 階層ブートストラップ (動画 → 貨物; 比はリサンプルごと) の 95% CI と相対幅 (hi−lo)/|点推定|
    A と B は同じリサンプル (同じ動画・貨物の組) で計算するので、幅の比 B/A は対応のある比較になる。
  - 動画間 CV = SD(動画ごとの比, ddof=1) / |mean(動画ごとの比)|
    CV_B/CV_A の 95% CI は動画のリサンプル (サイズ内) で出す。
  - 分母の雑音の内訳: 動画ごとの ⟨v_∥⟩, ⟨u_∞⟩, ⟨u_fp⟩ の CV と、動画間相関 corr(⟨v_∥⟩, 分母)。
20 µm は参考 (灰色)。

出力: figure/eta_vs_vpar_uinf_noise/
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_vpar_and_phi_vs_scaled_radius import load_data, BEADS_INFO

CUR = Path(__file__).resolve().parent
OUT = CUR / "figure" / "eta_vs_vpar_uinf_noise"
INFO = {b["name"]: b for b in BEADS_INFO}
PHI_TH = 0.95
COLS = ["sv", "su", "sf", "n"]  # Σv_∥, Σu_∞, Σu_fp, n_frames
C_A, C_B = "#4C72B0", "#DD8452"


def cargo_table(df):
    g = df.groupby(["bead_name", "exp_dir", "unique_pid"])
    return pd.DataFrame({"sv": g["v_parallel"].sum(), "su": g["u_inf"].sum(),
                         "sf": g["u_fp"].sum(), "n": g.size()}).reset_index()


def hboot(movies, n_boot, rng):
    """movies: list of (k_c x 4) arrays → (n_boot x 4) resampled sums"""
    out = np.empty((n_boot, 4))
    M = len(movies)
    for i in range(n_boot):
        tot = np.zeros(4)
        for j in rng.integers(0, M, M):
            a = movies[j]
            tot += a[rng.integers(0, len(a), len(a))].sum(0)
        out[i] = tot
    return out


def cv(x):
    x = np.asarray(x, float)
    return np.std(x, ddof=1) / abs(np.mean(x)) if len(x) > 1 else np.nan


def to_md(d):
    lines = ["| " + " | ".join(map(str, d.columns)) + " |", "|" + "---|" * len(d.columns)]
    for _, r in d.iterrows():
        lines.append("| " + " | ".join(f"{v:.3g}" if isinstance(v, (float, np.floating)) else str(v) for v in r) + " |")
    return "\n".join(lines)


def analyse(df, subset, n_boot, rng):
    ct = cargo_table(df)
    rows, movie_rows = [], []
    for b in [x["name"] for x in BEADS_INFO]:
        sub = ct[ct.bead_name == b]
        if sub.empty:
            continue
        movies = [g[COLS].to_numpy(float) for _, g in sub.groupby("exp_dir")]
        tot = sub[COLS].sum().to_numpy(float)
        A0, B0 = tot[0] / tot[1], tot[0] / tot[2]
        bs = hboot(movies, n_boot, rng)
        A, B = bs[:, 0] / bs[:, 1], bs[:, 0] / bs[:, 2]
        aL, aH = np.percentile(A, [2.5, 97.5])
        bL, bH = np.percentile(B, [2.5, 97.5])
        rwA, rwB = (aH - aL) / abs(A0), (bH - bL) / abs(B0)
        # bootstrap SD / mean as a second relative-noise measure
        rsA, rsB = np.std(A, ddof=1) / abs(A0), np.std(B, ddof=1) / abs(B0)

        mv = sub.groupby("exp_dir")[COLS].sum()
        mA, mB = mv.sv / mv.su, mv.sv / mv.sf
        for e, r in mv.iterrows():
            movie_rows.append(dict(subset=subset, bead=b, exp_dir=e, n_frames=int(r.n),
                                   vpar_over_uinf=r.sv / r.su, eta=r.sv / r.sf,
                                   mean_vpar=r.sv / r.n, mean_uinf=r.su / r.n, mean_ufp=r.sf / r.n,
                                   ufp_over_uinf=r.sf / r.su))
        cvA, cvB = cv(mA), cv(mB)
        # movie-level bootstrap of CV ratio
        M = len(mv)
        arrA, arrB = mA.to_numpy(), mB.to_numpy()
        cr = []
        for _ in range(n_boot):
            ii = rng.integers(0, M, M)
            if len(np.unique(ii)) < 2:
                continue
            ca, cb = cv(arrA[ii]), cv(arrB[ii])
            if ca > 0:
                cr.append(cb / ca)
        cr = np.array(cr)
        mvp, mu, mf = mv.sv / mv.n, mv.su / mv.n, mv.sf / mv.n
        rows.append(dict(
            subset=subset, bead=b, diameter_um=INFO[b]["diameter_um"],
            n_movies=M, n_cargo=len(sub), n_frames=int(tot[3]),
            vpar_uinf=A0, vpar_uinf_lo=aL, vpar_uinf_hi=aH,
            eta=B0, eta_lo=bL, eta_hi=bH,
            relCIw_vpar_uinf=rwA, relCIw_eta=rwB, relCIw_ratio_eta_over_A=rwB / rwA,
            relSD_vpar_uinf=rsA, relSD_eta=rsB,
            CV_movie_vpar_uinf=cvA, CV_movie_eta=cvB, CV_ratio_eta_over_A=cvB / cvA,
            CV_ratio_lo=np.percentile(cr, 2.5), CV_ratio_hi=np.percentile(cr, 97.5),
            P_eta_CV_smaller=float(np.mean(cr < 1)),
            CV_movie_mean_vpar=cv(mvp), CV_movie_mean_uinf=cv(mu), CV_movie_mean_ufp=cv(mf),
            corr_vpar_uinf=np.corrcoef(mvp, mu)[0, 1] if M > 2 else np.nan,
            corr_vpar_ufp=np.corrcoef(mvp, mf)[0, 1] if M > 2 else np.nan,
            ufp_over_uinf=tot[2] / tot[1],
        ))
    return pd.DataFrame(rows), pd.DataFrame(movie_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    df = load_data(u_inf_mode="frame")
    df = df[np.isfinite(df.u_inf) & np.isfinite(df.v_parallel) & np.isfinite(df.u_fp) & np.isfinite(df.phi_t)]
    subsets = {"all": df, f"phi>{PHI_TH}": df[df.phi_t > PHI_TH]}
    tabs, mtabs = [], []
    for name, d in subsets.items():
        t, m = analyse(d, name, args.n_boot, rng)
        tabs.append(t)
        mtabs.append(m)
    tab = pd.concat(tabs, ignore_index=True)
    tab["_o"] = tab.subset.map({k: i for i, k in enumerate(subsets)})
    tab = tab.sort_values(["_o", "diameter_um"]).drop(columns="_o").reset_index(drop=True)
    mtab = pd.concat(mtabs, ignore_index=True)

    # pooled summary over sizes (20 µm excluded): geometric mean of ratios, #sizes with η less noisy
    summ = []
    for excl, lab in [(["beads20um"], "excl 20"), (["beads20um", "beads06um"], "excl 20 & 0.63")]:
      for s, t in tab[~tab.bead.isin(excl)].groupby("subset", sort=False):
        summ.append(dict(subset=s, sizes=lab,
                         gmean_relCIw_ratio=float(np.exp(np.log(t.relCIw_ratio_eta_over_A).mean())),
                         n_sizes_eta_CIw_smaller=f"{int((t.relCIw_ratio_eta_over_A < 1).sum())}/{len(t)}",
                         gmean_CV_ratio=float(np.exp(np.log(t.CV_ratio_eta_over_A).mean())),
                         n_sizes_eta_CV_smaller=f"{int((t.CV_ratio_eta_over_A < 1).sum())}/{len(t)}",
                         median_relCIw_A=float(t.relCIw_vpar_uinf.median()),
                         median_relCIw_eta=float(t.relCIw_eta.median()),
                         median_CV_A=float(t.CV_movie_vpar_uinf.median()),
                         median_CV_eta=float(t.CV_movie_eta.median())))
    summ = pd.DataFrame(summ)

    tab.to_csv(OUT / "noise_per_size.csv", index=False)
    mtab.to_csv(OUT / "noise_per_movie.csv", index=False)
    summ.to_csv(OUT / "noise_summary.csv", index=False)

    main_cols = ["subset", "diameter_um", "n_movies", "n_cargo", "n_frames",
                 "vpar_uinf", "vpar_uinf_lo", "vpar_uinf_hi", "eta", "eta_lo", "eta_hi",
                 "relCIw_vpar_uinf", "relCIw_eta", "relCIw_ratio_eta_over_A",
                 "CV_movie_vpar_uinf", "CV_movie_eta", "CV_ratio_eta_over_A", "CV_ratio_lo", "CV_ratio_hi"]
    dec_cols = ["subset", "diameter_um", "CV_movie_mean_vpar", "CV_movie_mean_uinf", "CV_movie_mean_ufp",
                "corr_vpar_uinf", "corr_vpar_ufp", "ufp_over_uinf", "relSD_vpar_uinf", "relSD_eta"]
    with open(OUT / "noise_tables.md", "w") as fh:
        fh.write(f"## A = Σv_∥/Σu_∞(t) vs B = η = Σv_∥/Σ|⟨u⟩_fp|  (hierarchical bootstrap movie→cargo, n_boot={args.n_boot})\n\n")
        fh.write("relCIw = (CI_hi − CI_lo)/|point|; CV = SD/|mean| over per-movie ratios (ddof=1); "
                 "CV_ratio CI from movie resampling within size.\n\n")
        fh.write(to_md(tab[main_cols]) + "\n\n## Summary (geometric mean of η/A ratios over sizes)\n\n" + to_md(summ) + "\n\n")
        fh.write("## Noise decomposition (per-movie means)\n\n" + to_md(tab[dec_cols]) + "\n\n")
        fh.write("## Per movie\n\n" + to_md(mtab) + "\n")

    pd.set_option("display.width", 250)
    print(tab[main_cols].round(3).to_string())
    print(summ.round(3).to_string())
    print(tab[dec_cols].round(3).to_string())
    print(mtab.round(3).to_string())

    # ---------------- Figure ----------------
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
                         "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8})
    fig, axs = plt.subplots(2, 3, figsize=(16, 9))
    for row, (s, t_all) in enumerate(tab.groupby("subset", sort=False)):
        t = t_all[t_all.bead != "beads20um"]  # 20 µm (mean ≈ 0) only in tables / panel 3
        x = np.arange(len(t))
        labs = [f"{d:g}" for d in t.diameter_um]
        w = 0.38
        # (1) relative CI width
        ax = axs[row, 0]
        ax.bar(x - w / 2, t.relCIw_vpar_uinf, w, color=C_A, label=r"$v_\parallel/u_\infty$")
        ax.bar(x + w / 2, t.relCIw_eta, w, color=C_B, label=r"$\eta=\Sigma v_\parallel/\Sigma|\langle u\rangle_{fp}|$")
        for i, r in enumerate(t.itertuples()):
            ax.text(i, max(r.relCIw_vpar_uinf, r.relCIw_eta) * 1.03, f"×{r.relCIw_ratio_eta_over_A:.2f}",
                    ha="center", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(labs)
        ax.set_ylabel("(hi−lo)/|estimate|")
        ax.set_title(f"[{s}] hierarchical 95% CI rel. width (×: η/A)")
        ax.legend(fontsize=8)
        # (2) movie CV
        ax = axs[row, 1]
        ax.bar(x - w / 2, t.CV_movie_vpar_uinf, w, color=C_A, label=r"$v_\parallel/u_\infty$")
        ax.bar(x + w / 2, t.CV_movie_eta, w, color=C_B, label=r"$\eta$")
        for i, r in enumerate(t.itertuples()):
            ax.text(i, max(r.CV_movie_vpar_uinf, r.CV_movie_eta) * 1.03,
                    f"×{r.CV_ratio_eta_over_A:.2f}\n[{r.CV_ratio_lo:.2f},{r.CV_ratio_hi:.2f}]", ha="center", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(labs)
        ax.set_ylabel("between-movie CV")
        ax.set_title(f"[{s}] between-movie CV (×: η/A [95% CI])")
        ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
        # (3) per-movie values normalised by size estimate
        ax = axs[row, 2]
        mm = mtab[mtab.subset == s]
        x3 = np.arange(len(t_all))
        for i, r in enumerate(t_all.itertuples()):
            sub = mm[mm.bead == r.bead]
            jit = np.linspace(-0.08, 0.08, len(sub)) if len(sub) > 1 else np.zeros(1)
            sz = 15 + 60 * sub.n_frames / mm.n_frames.max()
            ax.scatter(i - 0.18 + jit, sub.vpar_over_uinf / r.vpar_uinf, s=sz, color=C_A, edgecolors="k", lw=0.4,
                       label=r"$v_\parallel/u_\infty$" if i == 0 else None)
            ax.scatter(i + 0.18 + jit, sub.eta / r.eta, s=sz, color=C_B, edgecolors="k", lw=0.4,
                       label=r"$\eta$" if i == 0 else None)
        ax.axhline(1, color="k", lw=0.6)
        ax.set_xticks(x3)
        ax.set_xticklabels([f"{d:g}" for d in t_all.diameter_um])
        ax.set_ylabel("per-movie value / size estimate")
        ax.set_title(f"[{s}] per-movie / size value (marker ~ frames)")
        ax.legend(fontsize=8)
        for a in axs[row]:
            a.set_xlabel(r"diameter ($\mu$m)")
        k = list(t_all.bead).index("beads20um") if "beads20um" in t_all.bead.values else None
        if k is not None:
            axs[row, 2].axvspan(k - 0.5, k + 0.5, color="0.9", zorder=0)
    fig.suptitle(r"Noise of $v_\parallel/u_\infty(t)$ vs $\eta=\Sigma v_\parallel/\Sigma|\langle u\rangle_{fp}|$"
                 "  (20 µm: tables only / grey band in right panels)", fontsize=12)
    fig.tight_layout()
    for ext in ["png", "svg"]:
        fig.savefig(OUT / f"eta_vs_vpar_uinf_noise.{ext}", dpi=200, bbox_inches="tight")
    print("saved", OUT / "eta_vs_vpar_uinf_noise.png")


if __name__ == "__main__":
    main()
