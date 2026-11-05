#!/usr/bin/env python3
"""result3.md 4.3 節 / D3 の【要確認】【要解析】をまとめて検証する。

1. 階層ブートストラップ（動画→貨物）による η の CI と、粒子のみブートストラップとの比較
2. フレームごとの η_i = v_par / |<u>_fp| の分布（D3: 一山か＝「ほとんど外れない」か）
3. Φ_fp≈1 かつ |<u>_fp|/u_inf が大きいフレームが特定の動画・時刻に固まっていないか
4. η の Φ_fp 依存（サイズごとに Φ_fp>0.95 と <0.8 を比較）と、外れ値除去への頑健性

入力: scratch/footprint_heterogeneity_cache.pkl, scratch/cargo_flow_profile_extended_cache.pkl
出力: figure/eta_robustness_result3/*.png, *.csv, summary.txt
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figure" / "eta_robustness_result3"
OUT.mkdir(parents=True, exist_ok=True)
ORDER = ["beads06um", "beads1um", "beads3um", "beads5um", "beads7um", "beads20um"]
DIAM = {"beads06um": 0.63, "beads1um": 1.18, "beads3um": 3.37, "beads5um": 5.00, "beads7um": 7.24, "beads20um": 20.0}
RNG = np.random.default_rng(0)
NB = 2000
lines = []


def log(s=""):
    print(s)
    lines.append(s)


def load():
    d = pd.read_pickle(ROOT / "scratch" / "footprint_heterogeneity_cache.pkl")
    e = pd.read_pickle(ROOT / "scratch" / "cargo_flow_profile_extended_cache.pkl")
    m = {Path(p).name: float(u) for p, u in zip(e["exp_dir"], e["u_inf"])}
    d["u_inf"] = d["exp_dir"].map(m)
    if d["u_inf"].isna().any():
        log(f"WARNING: u_inf missing for {d.loc[d.u_inf.isna(), 'exp_dir'].unique()}")
    d["ufp_n"] = d["u_fp"] / d["u_inf"]
    d["vpar_n"] = d["v_parallel"] / d["u_inf"]
    return d


def eta_ratio(sub):
    return sub["v_parallel"].sum() / sub["u_fp"].sum()


def boot_particle(sub):
    g = sub.groupby("unique_pid")[["v_parallel", "u_fp"]].sum().values
    out = []
    for _ in range(NB):
        s = g[RNG.integers(0, len(g), len(g))].sum(0)
        out.append(s[0] / s[1])
    return np.percentile(out, [2.5, 97.5])


def boot_hier(sub):
    mov = {k: v.groupby("unique_pid")[["v_parallel", "u_fp"]].sum().values for k, v in sub.groupby("exp_dir")}
    keys = list(mov)
    out = []
    for _ in range(NB):
        tot = np.zeros(2)
        for k in RNG.choice(keys, len(keys)):
            g = mov[k]
            tot += g[RNG.integers(0, len(g), len(g))].sum(0)
        out.append(tot[0] / tot[1])
    return np.percentile(out, [2.5, 97.5])


def main():
    d = load()
    # ---------- 1. size-level eta with two bootstrap schemes ----------
    log("=== 1. η (= Σv_par/Σ|<u>_fp|) per size: particle-only vs hierarchical (movie→particle) 95% CI ===")
    rows = []
    for b in ORDER:
        s = d[d.bead_name == b]
        eta = eta_ratio(s)
        cp, ch = boot_particle(s), boot_hier(s)
        # per-movie eta (movie-to-movie spread)
        pm = s.groupby("exp_dir").apply(eta_ratio)
        # robustness: drop frames with ufp_n > 1.5 (suspicious fast flow)
        s2 = s[s.ufp_n <= 1.5]
        rows.append(dict(bead=b, d_um=DIAM[b], n_frames=len(s), n_movies=s.exp_dir.nunique(), n_cargo=s.unique_pid.nunique(),
                         eta=eta, ci_part_lo=cp[0], ci_part_hi=cp[1], ci_hier_lo=ch[0], ci_hier_hi=ch[1],
                         eta_movie_min=pm.min(), eta_movie_max=pm.max(), eta_drop_fast=eta_ratio(s2),
                         frac_fast=(s.ufp_n > 1.5).mean()))
        log(f"{b:10s} N={s.unique_pid.nunique():2d} cargoes/{s.exp_dir.nunique()} movies  η={eta:.3f}  "
            f"part CI[{cp[0]:.3f},{cp[1]:.3f}]  hier CI[{ch[0]:.3f},{ch[1]:.3f}]  "
            f"per-movie η {pm.min():.2f}–{pm.max():.2f}  η(drop u_fp/u∞>1.5)={eta_ratio(s2):.3f} (dropped {100*(s.ufp_n>1.5).mean():.1f}%)")
    pd.DataFrame(rows).to_csv(OUT / "eta_size_bootstrap.csv", index=False)

    # ---------- 2. per-frame eta_i distribution ----------
    log("\n=== 2. フレームごとの η_i = v_par/|<u>_fp|（|<u>_fp| > 中央値のフレームのみ; 分母が小さいと発散するため）===")
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharex=True)
    for ax, b in zip(axes.flat, ORDER):
        s = d[d.bead_name == b]
        s = s[s.u_fp > s.u_fp.median()]
        ei = (s.v_parallel / s.u_fp).clip(-1.5, 2.5)
        ax.hist(ei, bins=np.linspace(-1.5, 2.5, 81), color="0.4")
        ax.axvline(eta_ratio(s), color="C3", lw=1, label="ratio of sums")
        ax.axvline(0, color="k", lw=0.5, ls=":")
        f_stuck = (ei.abs() < 0.1).mean()
        f_run = (ei > 0.5).mean()
        f_back = (ei < -0.1).mean()
        ax.set_title(f"{DIAM[b]} µm  |η_i|<0.1:{f_stuck:.0%}  η_i>0.5:{f_run:.0%}", fontsize=9)
        log(f"{b:10s} median η_i={ei.median():.3f}  frac |η_i|<0.1 = {f_stuck:.2f}, η_i>0.5 = {f_run:.2f}, η_i<-0.1 = {f_back:.2f}")
    for ax in axes[-1]:
        ax.set_xlabel(r"$\eta_i=v_\parallel/|\langle u\rangle_{fp}|$")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "eta_per_frame_hist.png", dpi=200)
    plt.close(fig)

    # ---------- 3. high-Phi fast-flow frames ----------
    log("\n=== 3. Φ_fp>0.99 のフレームで |<u>_fp|/u∞ > 1.5 の内訳（動画・フレーム範囲）===")
    hp = d[(d.phi_t > 0.99)]
    fast = hp[hp.ufp_n > 1.5]
    log(f"Φ_fp>0.99 frames: {len(hp)}, of which u_fp/u∞>1.5: {len(fast)} ({len(fast)/max(len(hp),1):.1%})")
    tab = fast.groupby("exp_dir").agg(n=("frame", "size"), n_cargo=("unique_pid", "nunique"),
                                       f_min=("frame", "min"), f_max=("frame", "max"), f_nuniq=("frame", "nunique"),
                                       ufp_n_med=("ufp_n", "median"))
    tab["share"] = tab.n / len(fast)
    tab["movie_frames"] = d.groupby("exp_dir").frame.nunique().reindex(tab.index)
    tab = tab.sort_values("n", ascending=False)
    log(tab.to_string())
    tab.to_csv(OUT / "high_phi_fast_frames_by_movie.csv")
    # time trace of median u_fp/u_inf per movie for the top movies
    top = tab.index[:6]
    fig, axes = plt.subplots(len(top), 1, figsize=(8, 1.6 * len(top)), sharex=False)
    for ax, m in zip(np.atleast_1d(axes), top):
        s = d[d.exp_dir == m].groupby("frame").ufp_n.median()
        ax.plot(s.index, s.values, lw=0.6, color="0.3")
        ax.axhline(1.5, color="C3", lw=0.6, ls="--")
        ax.set_ylabel(m, fontsize=7)
    np.atleast_1d(axes)[-1].set_xlabel("frame")
    fig.suptitle(r"median $|\langle u\rangle_{fp}|/u_\infty$ over cargoes vs frame", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "ufp_norm_time_trace_top_movies.png", dpi=200)
    plt.close(fig)

    # ---------- 4. eta vs Phi_fp, high vs low ----------
    log("\n=== 4. η の Φ_fp 依存：Φ_fp>0.95 vs Φ_fp<0.8（階層ブートストラップ 95% CI）===")
    rows = []
    for b in ORDER:
        s = d[d.bead_name == b]
        hi, lo = s[s.phi_t > 0.95], s[s.phi_t < 0.8]
        r = dict(bead=b)
        for tag, ss in (("hi", hi), ("lo", lo)):
            if ss.unique_pid.nunique() >= 2 and ss.u_fp.sum() > 0:
                c = boot_hier(ss)
                r.update({f"eta_{tag}": eta_ratio(ss), f"{tag}_lo": c[0], f"{tag}_hi": c[1], f"n_{tag}": len(ss)})
            else:
                r.update({f"eta_{tag}": np.nan, f"{tag}_lo": np.nan, f"{tag}_hi": np.nan, f"n_{tag}": len(ss)})
        rows.append(r)
        log(f"{b:10s} Φ>0.95: η={r['eta_hi']:.3f} [{r['hi_lo']:.3f},{r['hi_hi']:.3f}] (n={r['n_hi']})   "
            f"Φ<0.8: η={r['eta_lo']:.3f} [{r['lo_lo']:.3f},{r['lo_hi']:.3f}] (n={r['n_lo']})")
    pd.DataFrame(rows).to_csv(OUT / "eta_hi_vs_lo_phi.csv", index=False)

    # summary figure: eta per size with both CIs
    df = pd.DataFrame([r for r in pd.read_csv(OUT / "eta_size_bootstrap.csv").to_dict("records")])
    fig, ax = plt.subplots(figsize=(4.5, 3.4))
    x = df.d_um.values
    ax.errorbar(x * 0.96, df.eta, yerr=[df.eta - df.ci_part_lo, df.ci_part_hi - df.eta], fmt="o", ms=4, color="0.6",
                capsize=2, label="particle bootstrap")
    ax.errorbar(x * 1.04, df.eta, yerr=[df.eta - df.ci_hier_lo, df.ci_hier_hi - df.eta], fmt="s", ms=4, color="C3",
                capsize=2, label="hierarchical (movie→cargo)")
    ax.set_xscale("log")
    ax.set_xlabel(r"$2R_c$ (µm)")
    ax.set_ylabel(r"$\eta=\langle v_\parallel\rangle/\langle|\langle u\rangle_{fp}|\rangle$")
    ax.axhline(0, color="k", lw=0.5)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "eta_vs_size_hier_bootstrap.png", dpi=200)
    plt.close(fig)

    (OUT / "summary.txt").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
