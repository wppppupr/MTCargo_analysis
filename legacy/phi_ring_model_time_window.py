#!/usr/bin/env python3
"""
phi_ring_model_time_window.py
=============================

【4.4】円環モデル Φ_fp² = (1/π)∫_0^π exp(−2(R/ξ)sinφ)dφ が「同じ動画の中」でも成り立つかを、
200 フレームの時間窓ごとに調べる。

- ξ(t): figure/xi_time_window/xi_t_windows_w200.csv（scratch/xi_time_window.py, 窓ごとに局所ベースライン補正）
- 実測: 同じ窓 [win_start, win_end) に入る全貨物・全フレームの Φ_t（plot_vpar_and_phi_vs_scaled_radius.load_data, u_inf_mode='frame'）
  モデルは <Φ²> を予測するので主比較は Φ_rms = sqrt(<Φ_t²>)、参考に <Φ_t> も出す。
- 予測: Φ_pred(t) = ring_phi(R/ξ(t))、対照として動画全体の ξ（per_experiment.csv）による一定値 Φ_pred,movie。
- 動画内の検定: 窓ごとの値を動画平均からの偏差 Δ にして、Δobs = k·Δpred の傾き k（動画クラスタのブートストラップ CI）。
  k≈1 なら式が動画内でも成り立つ、k≈0 なら ξ(t) の揺れに Φ は追随しない。
- 20 µm は参考扱い（統計から除外、図では灰色）。

出力: figure/phi_ring_model_time_window/
  windows.csv, per_size.csv, per_size.md, phi_window_scatter.png/.svg

実行: .pixi/envs/default/bin/python phi_ring_model_time_window.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

import phi_ring_model as M
import plot_vpar_and_phi_vs_scaled_radius as V

ROOT = Path(__file__).resolve().parent
WIN_CSV = ROOT / "figure" / "xi_time_window" / "xi_t_windows_w200.csv"
OUT = ROOT / "figure" / "phi_ring_model_time_window"
MIN_OBS = 20          # 窓内の (貨物×フレーム) 数の下限
NBOOT = 2000
RNG = np.random.default_rng(0)
COLORS = {0.63: "#2a78d6", 1.18: "#eb6834", 3.37: "#1baf7a", 5.0: "#eda100", 7.24: "#e87ba4", 20.0: "#9a9a9a"}


def window_table(df, win, xi_movie):
    rows = []
    for w in win.itertuples():
        g = df[(df.exp_dir == w.exp_dir) & (df.frame >= w.win_start) & (df.frame < w.win_end)]
        if len(g) < MIN_OBS or not np.isfinite(w.xi_um):
            continue
        R = M.ring_mid_radius_um(w.bead_name)
        phi2 = g.phi_t.values ** 2
        # 貨物クラスタのブートストラップで Φ_rms の SE
        cg = g.assign(p2=phi2).groupby("unique_pid").p2.agg(["sum", "size"])
        s, n = cg["sum"].values, cg["size"].values
        if len(cg) > 1:
            idx = RNG.integers(0, len(cg), (300, len(cg)))
            se = float(np.std(np.sqrt(s[idx].sum(1) / n[idx].sum(1))))
        else:
            se = np.nan
        xm = float(xi_movie[w.exp_dir])
        rows.append(dict(
            bead_name=w.bead_name, diameter_um=w.diameter_um, exp_dir=w.exp_dir,
            reference=w.bead_name in M.REFERENCE_BEADS, win_start=w.win_start,
            xi_t_um=w.xi_um, xi_t_err_um=w.xi_err_um, xi_movie_um=xm, R_ring_um=R,
            n_obs=len(g), n_cargo=int(g.unique_pid.nunique()),
            phi_mean_obs=float(g.phi_t.mean()), phi_rms_obs=float(np.sqrt(phi2.mean())), phi_rms_se=se,
            phi_pred_t=float(M.ring_phi(R / w.xi_um)), phi_pred_movie=float(M.ring_phi(R / xm)),
            # ξ の当てはめ誤差を予測に伝搬
            phi_pred_t_lo=float(M.ring_phi(R / max(w.xi_um - w.xi_err_um, 1e-3))),
            phi_pred_t_hi=float(M.ring_phi(R / (w.xi_um + w.xi_err_um))),
        ))
    t = pd.DataFrame(rows)
    t["resid_t"] = t.phi_rms_obs - t.phi_pred_t
    t["resid_movie"] = t.phi_rms_obs - t.phi_pred_movie
    t["resid_mean_t"] = t.phi_mean_obs - t.phi_pred_t
    gm = t.groupby("exp_dir")
    for c in ["phi_rms_obs", "phi_pred_t", "xi_t_um"]:
        t["d_" + c] = t[c] - gm[c].transform("mean")
    t["n_win_movie"] = gm.win_start.transform("size")
    return t


def slope0(dx, dy):
    d = (dx ** 2).sum()
    return float((dx * dy).sum() / d) if d > 0 else np.nan


def movie_boot(t, fn):
    movies = t.exp_dir.unique()
    groups = {m: t[t.exp_dir == m] for m in movies}
    out = []
    for _ in range(NBOOT):
        pick = RNG.choice(movies, len(movies))
        out.append(fn(pd.concat([groups[m] for m in pick])))
    out = np.array(out, float)
    return np.nanpercentile(out, [2.5, 97.5])


def wmean(t, c):
    return float(np.average(t[c], weights=t.n_obs))


def per_size(t):
    rows = []
    for d, g in t.groupby("diameter_um"):
        wi = g[g.n_win_movie >= 2]
        r = dict(diameter_um=d, reference=bool(g.reference.iloc[0]), n_movies=g.exp_dir.nunique(), n_win=len(g),
                 R_over_xi_t_range=f"{(g.R_ring_um / g.xi_t_um).min():.2f}–{(g.R_ring_um / g.xi_t_um).max():.2f}",
                 phi_rms_obs=wmean(g, "phi_rms_obs"), phi_pred_t=wmean(g, "phi_pred_t"),
                 resid_t=wmean(g, "resid_t"), resid_movie=wmean(g, "resid_movie"),
                 resid_mean_t=wmean(g, "resid_mean_t"),
                 rms_resid_t=float(np.sqrt(np.average(g.resid_t ** 2, weights=g.n_obs))),
                 rms_resid_movie=float(np.sqrt(np.average(g.resid_movie ** 2, weights=g.n_obs))),
                 sd_d_pred=float(wi.d_phi_pred_t.std()) if len(wi) > 1 else np.nan,
                 sd_d_obs=float(wi.d_phi_rms_obs.std()) if len(wi) > 1 else np.nan,
                 med_obs_se=float(g.phi_rms_se.median()))
        lo, hi = movie_boot(g, lambda x: wmean(x, "resid_t"))
        r.update(resid_t_lo=lo, resid_t_hi=hi)
        if len(wi) >= 3:
            r["k_within"] = slope0(wi.d_phi_pred_t, wi.d_phi_rms_obs)
            r["k_lo"], r["k_hi"] = movie_boot(wi, lambda x: slope0(x.d_phi_pred_t, x.d_phi_rms_obs))
            r["r_within"] = float(np.corrcoef(wi.d_phi_pred_t, wi.d_phi_rms_obs)[0, 1])
        rows.append(r)
    return pd.DataFrame(rows)


def plot(t, ps, pooled):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(16, 5.2))
    # (a) 絶対値の散布図
    a = ax[0]
    for d, g in t.groupby("diameter_um"):
        c = COLORS.get(d, "k")
        a.errorbar(g.phi_pred_t, g.phi_rms_obs, yerr=g.phi_rms_se,
                   xerr=[g.phi_pred_t - g.phi_pred_t_lo, g.phi_pred_t_hi - g.phi_pred_t],
                   fmt="o", ms=6, color=c, mec="white", mew=0.8, elinewidth=0.8, alpha=0.9,
                   label=f"{d:g} µm" + (" (ref.)" if d == 20 else ""))
    a.plot([0.2, 1], [0.2, 1], color="#555", lw=1, ls="--", zorder=0)
    a.set(xlabel=r"predicted $\Phi_{fp}$ from $\xi(t)$  (window)", ylabel=r"observed $\Phi_{rms}=\sqrt{\langle\Phi_t^2\rangle}$  (window)",
          xlim=(0.2, 1.0), ylim=(0.2, 1.02), title="(a) 200-frame windows: absolute")
    a.legend(frameon=False, fontsize=9, loc="upper left")
    # (b) 動画内の偏差
    b = ax[1]
    tw = t[(t.n_win_movie >= 2)]
    for d, g in tw.groupby("diameter_um"):
        b.scatter(g.d_phi_pred_t, g.d_phi_rms_obs, s=40, color=COLORS.get(d, "k"), edgecolor="white", lw=0.8,
                  label=f"{d:g} µm")
    lim = 0.2
    xx = np.array([-lim, lim])
    b.plot(xx, xx, color="#555", ls="--", lw=1, label="slope 1 (model)")
    b.plot(xx, pooled["k"] * xx, color="k", lw=1.5,
           label=f"fit (excl. 20 µm): k={pooled['k']:.2f} [{pooled['k_lo']:.2f}, {pooled['k_hi']:.2f}]")
    b.axhline(0, color="#ccc", lw=0.6, zorder=0); b.axvline(0, color="#ccc", lw=0.6, zorder=0)
    b.set(xlim=(-lim, lim), ylim=(-0.3, 0.3), xlabel=r"$\Delta\Phi_{pred}$ (window − movie mean)",
          ylabel=r"$\Delta\Phi_{rms,obs}$ (window − movie mean)", title="(b) within-movie deviations")
    b.legend(frameon=False, fontsize=8, loc="upper left")
    # (c) サイズごとの残差平均
    c = ax[2]
    x = np.arange(len(ps))
    for i, r in ps.reset_index(drop=True).iterrows():
        col = COLORS.get(r.diameter_um, "k")
        c.errorbar(i - 0.12, r.resid_t, yerr=[[r.resid_t - r.resid_t_lo], [r.resid_t_hi - r.resid_t]],
                   fmt="o", ms=8, color=col, mec="white", capsize=3)
        c.plot(i + 0.12, r.resid_movie, "D", ms=6, mfc="white", mec=col, mew=1.5)
    c.plot([], [], "o", color="#555", label=r"$\xi(t)$ per window (95% CI, movie bootstrap)")
    c.plot([], [], "D", mfc="white", mec="#555", label=r"constant movie $\xi$")
    c.axhline(0, color="#555", lw=1)
    c.set_xticks(x, [f"{d:g}" for d in ps.diameter_um])
    c.set(xlabel="bead diameter (µm)", ylabel=r"mean residual $\Phi_{rms}-\Phi_{pred}$",
          title="(c) residual per size")
    c.legend(frameon=False, fontsize=9, loc="upper left")
    for s in ax:
        s.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"phi_window_scatter.{ext}", dpi=200)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = V.load_data(u_inf_mode="frame")
    win = pd.read_csv(WIN_CSV)
    t = window_table(df, win, M.load_xi())
    t.to_csv(OUT / "windows.csv", index=False)
    ps = per_size(t)
    ps.to_csv(OUT / "per_size.csv", index=False)

    m = t[(~t.reference) & (t.n_win_movie >= 2)]
    k = slope0(m.d_phi_pred_t, m.d_phi_rms_obs)
    k_lo, k_hi = movie_boot(m, lambda x: slope0(x.d_phi_pred_t, x.d_phi_rms_obs))
    pooled = dict(k=k, k_lo=float(k_lo), k_hi=float(k_hi), r=float(np.corrcoef(m.d_phi_pred_t, m.d_phi_rms_obs)[0, 1]),
                  n_win=len(m), n_movies=m.exp_dir.nunique())
    m2 = m[m.diameter_um != 5.0]
    pooled["k_excl5"] = slope0(m2.d_phi_pred_t, m2.d_phi_rms_obs)
    pooled["k_excl5_lo"], pooled["k_excl5_hi"] = map(float, movie_boot(m2, lambda x: slope0(x.d_phi_pred_t, x.d_phi_rms_obs)))
    pooled["r_excl5"] = float(np.corrcoef(m2.d_phi_pred_t, m2.d_phi_rms_obs)[0, 1])
    a = t[~t.reference]
    pooled["r_abs"] = float(np.corrcoef(a.phi_pred_t, a.phi_rms_obs)[0, 1])
    pooled["rms_t"] = float(np.sqrt(np.average(a.resid_t ** 2, weights=a.n_obs)))
    pooled["rms_movie"] = float(np.sqrt(np.average(a.resid_movie ** 2, weights=a.n_obs)))
    # 動画内: 偏差に ξ(t) の予測を足した方が良いか（Δobs の分散がどれだけ減るか）
    pooled["var_dobs"] = float(m.d_phi_rms_obs.var())
    pooled["var_dobs_minus_dpred"] = float((m.d_phi_rms_obs - m.d_phi_pred_t).var())
    plot(t, ps, pooled)

    pd.set_option("display.width", 250)
    print(t[["exp_dir", "win_start", "n_obs", "n_cargo", "xi_t_um", "xi_movie_um", "phi_rms_obs", "phi_rms_se",
             "phi_pred_t", "phi_pred_movie", "resid_t", "resid_movie"]].round(3).to_string(index=False))
    print()
    print(ps.round(3).to_string(index=False))
    print()
    print({k_: round(v, 4) if isinstance(v, float) else v for k_, v in pooled.items()})
    with open(OUT / "per_size.md", "w") as f:
        f.write("# 時間窓 (200 frame) ごとの円環モデル検証\n\n")
        f.write("```\n" + ps.round(3).to_string(index=False) + "\n```")
        f.write("\n\n## 全体 (20 µm 除外)\n\n")
        for k_, v in pooled.items():
            f.write(f"- {k_}: {v:.4g}\n" if isinstance(v, float) else f"- {k_}: {v}\n")


if __name__ == "__main__":
    main()
