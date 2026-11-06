#!/usr/bin/env python3
"""
eta0_hierarchical.py
====================
η_0 = (Σ v_∥ / Σ u_∞(t)) ÷ ⟨Φ_fp⟩   (Φ_fp は frame 平均 = 足印の phi_t の全フレーム平均)

1. サイズごとに階層ブートストラップ (動画 → 貨物)。比はリサンプルごとに計算する。
2. 2 群 (small: 0.63/1.18/3.37 µm, large: 5.00/7.24 µm) の差 Δη_0 = η_0(large) − η_0(small) の CI。
   群の η_0 はサイズごとに層別して動画→貨物をリサンプルし、
   (a) 群内の全フレームをプールした比、(b) サイズ別 η_0 の単純平均 の 2 通り。
3. 動画ごとの η_0 を log R_c と log(R_c/ξ) に単回帰し、R², AIC, LOO-RMSE を比較。
   さらに η_0 = a + b log R_c + c log ξ の 2 変数回帰で c≈0 (R_c 型) か c≈−b (R_c/ξ 型) かを見る。
   ξ は 2 通り: cache (主図の横軸 rc_over_xi) と CSV (bg_angular_correlation_length_per_experiment.csv)。
   20 µm は除外 (参考に含めた版も出す)。

データ: plot_vpar_and_phi_vs_scaled_radius.load_data(u_inf_mode='frame')
出力: figure/eta0_hierarchical/
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_vpar_and_phi_vs_scaled_radius import load_data, BEADS_INFO

CUR = Path(__file__).resolve().parent
OUT = CUR / "figure" / "eta0_hierarchical"
XI_CSV = CUR / "figure" / "bg_angular_correlation" / "bg_angular_correlation_length_per_experiment.csv"
SMALL = ["beads06um", "beads1um", "beads3um"]
LARGE = ["beads5um", "beads7um"]
INFO = {b["name"]: b for b in BEADS_INFO}


def cargo_table(df):
    """貨物ごとの和 (Σv_∥, Σu_∞, ΣΦ, n)"""
    g = df.groupby(["bead_name", "exp_dir", "unique_pid"])
    t = pd.DataFrame({"sv": g["v_parallel"].sum(), "su": g["u_inf"].sum(),
                      "sp": g["phi_t"].sum(), "n": g.size()}).reset_index()
    return t


def eta0_from_sums(s):
    sv, su, sp, n = s
    return (sv / su) / (sp / n)


def movie_arrays(ct, beads):
    """{bead: [ (k_c x 4) array per movie ]}"""
    out = {}
    for b in beads:
        sub = ct[ct.bead_name == b]
        out[b] = [grp[["sv", "su", "sp", "n"]].to_numpy(float) for _, grp in sub.groupby("exp_dir")]
    return out


def resample_sums(movies, rng):
    idx = rng.integers(0, len(movies), len(movies))
    tot = np.zeros(4)
    for i in idx:
        a = movies[i]
        tot += a[rng.integers(0, len(a), len(a))].sum(0)
    return tot


def ols(x, y):
    X = np.column_stack([np.ones(len(x))] + ([x] if x.ndim == 1 else [x[:, j] for j in range(x.shape[1])]))
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    n, k = X.shape
    rss = float(res @ res)
    r2 = 1 - rss / float(((y - y.mean()) ** 2).sum())
    aic = n * np.log(rss / n) + 2 * (k + 1)
    # LOO via hat matrix
    H = X @ np.linalg.pinv(X.T @ X) @ X.T
    loo = res / (1 - np.diag(H))
    sigma2 = rss / max(n - k, 1)
    cov = sigma2 * np.linalg.pinv(X.T @ X)
    return dict(beta=beta, se=np.sqrt(np.diag(cov)), r2=r2, aic=aic, loo_rmse=float(np.sqrt(np.mean(loo ** 2))), rss=rss)


def to_md(d):
    d = d.copy()
    lines = ["| " + " | ".join(map(str, d.columns)) + " |", "|" + "---|" * len(d.columns)]
    for _, r in d.iterrows():
        lines.append("| " + " | ".join(f"{v:.4g}" if isinstance(v, (float, np.floating)) else str(v) for v in r) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    df = load_data(u_inf_mode="frame")
    df = df[np.isfinite(df.u_inf) & np.isfinite(df.v_parallel) & np.isfinite(df.phi_t)]
    ct = cargo_table(df)
    beads = [b["name"] for b in BEADS_INFO]
    mv = movie_arrays(ct, beads)

    # ---- 1. サイズごと ----
    rows, boots = [], {}
    for b in beads:
        sub = ct[ct.bead_name == b]
        tot = sub[["sv", "su", "sp", "n"]].sum().to_numpy(float)
        bs = np.array([resample_sums(mv[b], rng) for _ in range(args.n_boot)])
        e = (bs[:, 0] / bs[:, 1]) / (bs[:, 2] / bs[:, 3])
        r = bs[:, 0] / bs[:, 1]
        p = bs[:, 2] / bs[:, 3]
        boots[b] = bs
        lo, hi = np.percentile(e, [2.5, 97.5])
        rows.append(dict(bead=b, diameter_um=INFO[b]["diameter_um"], n_movies=sub.exp_dir.nunique(),
                         n_cargo=len(sub), n_frames=int(tot[3]),
                         vpar_over_uinf=tot[0] / tot[1], vpar_over_uinf_lo=np.percentile(r, 2.5),
                         vpar_over_uinf_hi=np.percentile(r, 97.5),
                         phi_fp=tot[2] / tot[3], phi_lo=np.percentile(p, 2.5), phi_hi=np.percentile(p, 97.5),
                         eta0=eta0_from_sums(tot), eta0_lo=lo, eta0_hi=hi, eta0_sem=np.std(e, ddof=1),
                         eta0_boot_median=np.median(e)))
    size_tab = pd.DataFrame(rows)
    size_tab.to_csv(OUT / "eta0_per_size.csv", index=False)

    # ---- 2. 2 群の差 (サイズ層別リサンプル。boots[b] はサイズ間独立) ----
    def grp_pooled(names, i):
        return eta0_from_sums(sum(boots[b][i] for b in names))

    def grp_mean(names, i):
        return np.mean([eta0_from_sums(boots[b][i]) for b in names])

    def grp_point(names, how):
        if how == "pooled":
            return eta0_from_sums(sum(ct[ct.bead_name == b][["sv", "su", "sp", "n"]].sum().to_numpy(float) for b in names))
        return np.mean([size_tab.set_index("bead").loc[b, "eta0"] for b in names])

    diff_rows = []
    for how, fn in [("pooled_frames", grp_pooled), ("mean_of_sizes", grp_mean)]:
        s = np.array([fn(SMALL, i) for i in range(args.n_boot)])
        l = np.array([fn(LARGE, i) for i in range(args.n_boot)])
        d = l - s
        rt = l / s
        key = "pooled" if how == "pooled_frames" else "mean"
        ps, pl = grp_point(SMALL, key), grp_point(LARGE, key)
        diff_rows.append(dict(method=how,
                              eta0_small=ps, small_lo=np.percentile(s, 2.5), small_hi=np.percentile(s, 97.5),
                              eta0_large=pl, large_lo=np.percentile(l, 2.5), large_hi=np.percentile(l, 97.5),
                              diff_large_minus_small=pl - ps, diff_lo=np.percentile(d, 2.5), diff_hi=np.percentile(d, 97.5),
                              ratio_large_over_small=pl / ps, ratio_lo=np.percentile(rt, 2.5), ratio_hi=np.percentile(rt, 97.5),
                              p_two_sided=2 * min(np.mean(d <= 0), np.mean(d >= 0))))
    diff_tab = pd.DataFrame(diff_rows)
    diff_tab.to_csv(OUT / "eta0_group_difference.csv", index=False)

    # ---- 3. 動画ごとの回帰 ----
    xi_csv = pd.read_csv(XI_CSV).set_index("exp_dir")["xi_um"]
    mvt = ct.groupby(["bead_name", "exp_dir"])[["sv", "su", "sp", "n"]].sum().reset_index()
    mvt["eta0"] = (mvt.sv / mvt.su) / (mvt.sp / mvt.n)
    mvt["vpar_over_uinf"] = mvt.sv / mvt.su
    mvt["phi_fp"] = mvt.sp / mvt.n
    mvt["Rc"] = mvt.bead_name.map(lambda b: INFO[b]["radius_um"])
    rcxi_cache = df.groupby("exp_dir")["rc_over_xi"].first()
    mvt["xi_cache"] = mvt.Rc / mvt.exp_dir.map(rcxi_cache)
    mvt["xi_csv"] = mvt.exp_dir.map(xi_csv)
    mvt.to_csv(OUT / "eta0_per_movie.csv", index=False)

    reg_rows = []
    for incl20 in [False, True]:
        m = mvt if incl20 else mvt[mvt.bead_name != "beads20um"]
        y = m.eta0.to_numpy()
        lr = np.log(m.Rc.to_numpy())
        for xs in ["xi_cache", "xi_csv"]:
            mm = np.isfinite(m[xs].to_numpy())
            lx = np.log(m[xs].to_numpy())[mm]
            yy, lrr = y[mm], lr[mm]
            f1 = ols(lrr, yy)
            f2 = ols(lrr - lx, yy)
            f3 = ols(np.column_stack([lrr, lx]), yy)
            # 動画ブートストラップで ΔR², ΔLOO の安定性
            dr2, win = [], []
            n = len(yy)
            for _ in range(2000):
                ii = rng.integers(0, n, n)
                if len(np.unique(lrr[ii])) < 2:
                    continue
                a, bb = ols(lrr[ii], yy[ii]), ols((lrr - lx)[ii], yy[ii])
                dr2.append(bb["r2"] - a["r2"])
                win.append(bb["r2"] > a["r2"])
            dr2 = np.array(dr2)
            base = dict(include_20um=incl20, xi_source=xs, n_movies=n)
            for name, f in [("log Rc", f1), ("log(Rc/xi)", f2)]:
                reg_rows.append({**base, "model": name, "slope": f["beta"][1], "slope_se": f["se"][1],
                                 "R2": f["r2"], "AIC": f["aic"], "LOO_RMSE": f["loo_rmse"],
                                 "dR2_ratio_minus_Rc": np.nan, "dR2_lo": np.nan, "dR2_hi": np.nan,
                                 "P_boot_ratio_better": np.nan, "c_logxi": np.nan, "c_se": np.nan})
            reg_rows.append({**base, "model": "log Rc + log xi", "slope": f3["beta"][1], "slope_se": f3["se"][1],
                             "R2": f3["r2"], "AIC": f3["aic"], "LOO_RMSE": f3["loo_rmse"],
                             "dR2_ratio_minus_Rc": f2["r2"] - f1["r2"],
                             "dR2_lo": np.percentile(dr2, 2.5), "dR2_hi": np.percentile(dr2, 97.5),
                             "P_boot_ratio_better": float(np.mean(win)),
                             "c_logxi": f3["beta"][2], "c_se": f3["se"][2]})
    reg_tab = pd.DataFrame(reg_rows)
    reg_tab.to_csv(OUT / "eta0_regression_logRc_vs_logRcxi.csv", index=False)

    # ---- Markdown ----
    with open(OUT / "eta0_tables.md", "w") as fh:
        fh.write("## η_0 per size (hierarchical bootstrap movie→cargo, n_boot=%d)\n\n" % args.n_boot)
        fh.write(to_md(size_tab) + "\n\n")
        fh.write("## Group difference (large 5.00/7.24 − small 0.63/1.18/3.37)\n\n")
        fh.write(to_md(diff_tab) + "\n\n")
        fh.write("## Per-movie regression\n\n")
        fh.write(to_md(reg_tab) + "\n\n")
        fh.write("## Per-movie η_0\n\n")
        fh.write(to_md(mvt.drop(columns=["sv", "su", "sp"])) + "\n")

    print(size_tab.round(3).to_string())
    print(diff_tab.round(3).T.to_string())
    print(reg_tab.round(3).to_string())
    print(mvt[["exp_dir", "n", "vpar_over_uinf", "phi_fp", "eta0", "xi_cache", "xi_csv"]].round(3).to_string())

    # ---- Figure ----
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
    ax = axs[0]
    for _, r in size_tab.iterrows():
        info = INFO[r.bead]
        c = "0.6" if r.bead == "beads20um" else info["color"]
        ax.errorbar(r.diameter_um, r.eta0, yerr=[[r.eta0 - r.eta0_lo], [r.eta0_hi - r.eta0]],
                    fmt=info["marker"], color=c, ms=9, capsize=4, mec="k", zorder=3)
        sub = mvt[mvt.bead_name == r.bead]
        ax.scatter(sub.Rc * 2, sub.eta0, marker=info["marker"], color=c, s=22, alpha=0.5, zorder=2)
    ax.set_xscale("log")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel(r"diameter ($\mu$m)")
    ax.set_ylabel(r"$\eta_0=(\Sigma v_\parallel/\Sigma u_\infty)/\langle\Phi_{fp}\rangle$")
    ax.set_title("(a) per size, hierarchical 95% CI")

    ax = axs[1]
    labs, ys, los, his = [], [], [], []
    for _, r in diff_tab.iterrows():
        labs.append(r.method)
        ys.append(r.diff_large_minus_small)
        los.append(r.diff_lo)
        his.append(r.diff_hi)
    yy = np.arange(len(labs))
    ax.errorbar(ys, yy, xerr=[np.array(ys) - los, np.array(his) - ys], fmt="o", color="k", capsize=5)
    ax.axvline(0, color="r", ls="--")
    ax.set_yticks(yy)
    ax.set_yticklabels(labs)
    ax.set_ylim(-0.7, len(labs) - 0.3)
    ax.set_xlabel(r"$\Delta\eta_0$ = (5.00, 7.24) − (0.63, 1.18, 3.37)")
    ax.set_title("(b) group difference, 95% CI")

    ax = axs[2]
    m = mvt[mvt.bead_name != "beads20um"]
    for b, sub in m.groupby("bead_name"):
        info = INFO[b]
        ax.scatter(sub.Rc, sub.eta0, marker=info["marker"], color=info["color"], s=45, edgecolors="k", label=f"{info['diameter_um']} µm: vs $R_c$")
        ax.scatter(sub.Rc / sub.xi_cache, sub.eta0, marker=info["marker"], facecolors="none", edgecolors=info["color"], s=45)
    for xs_name, xv, ls in [("log Rc", np.log(m.Rc), "-"), ("log(Rc/xi)", np.log(m.Rc / m.xi_cache), "--")]:
        f = ols(xv.to_numpy(), m.eta0.to_numpy())
        xx = np.linspace(xv.min(), xv.max(), 50)
        ax.plot(np.exp(xx), f["beta"][0] + f["beta"][1] * xx, "k", ls=ls,
                label=f"{xs_name}: R²={f['r2']:.2f}, LOO={f['loo_rmse']:.3f}")
    ax.set_xscale("log")
    ax.set_xlabel(r"$R_c$ ($\mu$m, filled) or $R_c/\xi$ (open)")
    ax.set_ylabel(r"$\eta_0$ per movie")
    ax.set_title("(c) per-movie regression (20 µm excluded)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    for ext in ["png", "svg"]:
        fig.savefig(OUT / f"eta0_hierarchical.{ext}", dpi=200, bbox_inches="tight")
    print("saved", OUT / "eta0_hierarchical.png")


if __name__ == "__main__":
    main()
