#!/usr/bin/env python3
"""
scratch/msd_anomaly_and_fit.py

MSD.py の計算結果に対する (A) 異常値検証 と (B) 新しいフィッティング関数の比較。
MSD.py の関数 (concat_pooled_particles_MSD 等) をそのまま再利用し、同一条件のデータを使う。

(A) 異常値チェック
  A1. トラックのフレーム欠損 (imsd はフレーム番号ではなく行番号から lag time を作るため、
      欠損があると lag time がずれる)
  A2. 1フレーム変位のジャンプ (リンク誤り候補): |Δr| > max(10×粒子内中央値, 5 μm)
  A3. MSD 値そのものの NaN / inf / <=0
  A4. 個別 iMSD の外れ値: 各 lag で log10(MSD) の robust z (MAD) > 5
  A5. 単一粒子のアンサンブル平均への寄与率 (MSD(300 s) で最大寄与粒子の割合)
  A6. alpha (4-300 s) の物理的にありえない値 (>2.2) と、除外粒子 (alpha<=0.5) 数
  A7. lag ごとの粒子数 (生存バイアス: 長 lag は長時間追跡粒子のみ)

(B) アンサンブル平均 MSD への候補モデル比較 (log 空間・重み付き最小二乗, AIC/BIC)
  注: libs/displacement._calculate_msd は (dx^2+dy^2)/2 = 1成分あたり MSD を返す。
  以下のモデルはこの「1成分 MSD」に対して書いている (2D の <Δr^2> は 2 倍)。
  M1 power-law + noise      : 2σ² + K t^α
  M2 diffusion + noise      : 2σ² + 2D t
  M3 PRW / ABP (Fürth)      : 2σ² + 2D t + v²τ² (t/τ − 1 + e^{−t/τ})
  M4 caged + hopping        : L² (1 − e^{−t/τ}) + 2D t + 2σ²
  M5 crossover power-law    : 2σ² + K t^{α1} (1 + t/t_c)^{α2−α1}
"""

import sys
import glob
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import MSD as msdmod  # noqa: E402

OUT = ROOT / "figure" / "msd" / "anomaly_fit"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = ROOT / "scratch" / "msd_pool_cache.pkl"

BEADS = [
    ("beads06um", 0.63, [4, 4, 4, 4]),
    ("beads1um", 1.18, [4, 4, 4, 4]),
    ("beads3um", 3.37, [4, 4, 4, 4]),
    ("beads5um", 5.00, [4, 4, 4, 4]),
    ("beads7um", 7.24, [4, 4, 4]),
    ("beads20um", 20.0, [4, 4, 4]),
]
SCALE = 0.11


# ---------------------------------------------------------------- data
def load_pools():
    if CACHE.exists():
        with open(CACHE, "rb") as f:
            return pickle.load(f)
    root = msdmod.find_default_root()
    pools = {}
    for name, d, iv in BEADS:
        filt, raw, nt, nf = msdmod.concat_pooled_particles_MSD(root / name, iv, scale=SCALE, alpha_threshold=0.5)
        pools[name] = dict(d=d, filt=filt, raw=raw, n_tot=nt, n_filt=nf)
        print(f"  loaded {name}: {nt} -> {nf}")
    with open(CACHE, "wb") as f:
        pickle.dump(pools, f)
    return pools


def track_checks(name):
    """A1, A2: raw track files に対するフレーム欠損・ジャンプ検出"""
    root = msdmod.find_default_root()
    files = sorted(glob.glob(str(root / name / "*" / "*" / "beads_tracks.csv")))
    n_part = n_gap = n_jump_part = 0
    gap_frac = []
    jumps = []
    for vid, fp in enumerate(files):
        df = pd.read_csv(fp, usecols=["x", "y", "frame", "particle"])
        for pid, g in df.groupby("particle"):
            g = g.sort_values("frame")
            fr = g["frame"].to_numpy()
            n_part += 1
            span = fr[-1] - fr[0] + 1
            if span != len(fr):
                n_gap += 1
                gap_frac.append(1 - len(fr) / span)
            if len(fr) < 3:
                continue
            dr = SCALE * np.hypot(np.diff(g["x"].to_numpy()), np.diff(g["y"].to_numpy()))
            med = np.median(dr)
            thr = max(10 * med, 5.0)
            bad = dr > thr
            if bad.any():
                n_jump_part += 1
                jumps.append(dict(bead=name, video_idx=vid, particle=pid, n_jumps=int(bad.sum()),
                                  max_step_um=float(dr.max()), median_step_um=float(med),
                                  frame=int(fr[1:][bad][0])))
    dup = 0
    return dict(n_particles=n_part, n_with_frame_gaps=n_gap,
                median_missing_frac=float(np.median(gap_frac)) if gap_frac else 0.0,
                n_with_jumps=n_jump_part), jumps


def msd_checks(name, P):
    filt, raw = P["filt"], P["raw"]
    res = {}
    # A3
    m = raw["MSD"].to_numpy(float)
    res["n_msd_nonfinite"] = int((~np.isfinite(m)).sum())
    res["n_msd_le0"] = int((m <= 0).sum())
    # A6
    alph = raw.groupby("unique_particle_id")["alpha"].first()
    res["n_alpha_nan"] = int(alph.isna().sum())
    res["n_alpha_le_0.5_removed"] = int((alph <= 0.5).sum())
    res["n_alpha_gt_2.2"] = int((alph > 2.2).sum())
    res["alpha_median"] = float(alph.median())
    # A4: per-lag robust z on log10 MSD (filtered pool, lags with >=10 particles)
    f = filt[filt["MSD"] > 0].copy()
    f["lm"] = np.log10(f["MSD"])
    g = f.groupby("lag time")["lm"]
    med = g.transform("median")
    mad = g.transform(lambda s: np.median(np.abs(s - np.median(s)))) * 1.4826
    cnt = g.transform("count")
    f["z"] = (f["lm"] - med) / mad.replace(0, np.nan)
    sel = (cnt >= 10) & (f["z"].abs() > 5)
    out_pids = f.loc[sel].groupby("unique_particle_id")["z"].agg(["count", "max", "min"])
    res["n_particles_with_robust_z_gt5"] = int(len(out_pids))
    # A5: dominance at 300 s
    near = f[(f["lag time"] >= 288) & (f["lag time"] <= 312)]
    best = near.loc[(near["lag time"] - 300).abs().groupby(near["unique_particle_id"]).idxmin()]
    if len(best):
        v = best["MSD"].to_numpy()
        res["msd300_max_share_of_sum"] = float(v.max() / v.sum())
        res["msd300_mean_over_median"] = float(v.mean() / np.median(v))
        top = best.sort_values("MSD", ascending=False).head(3)
        res["msd300_top3"] = ";".join(f"{p}:{x:.1f}" for p, x in zip(top["unique_particle_id"], top["MSD"]))
    # A7
    c = filt.groupby("lag time")["MSD"].count()
    res["n_filt"] = P["n_filt"]
    res["max_lag_with_N>=10"] = float(c[c >= 10].index.max()) if (c >= 10).any() else np.nan
    res["max_lag_with_N>=50%"] = float(c[c >= 0.5 * P["n_filt"]].index.max()) if P["n_filt"] else np.nan
    res["max_lag_plotted(N>=2)"] = float(c[c >= 2].index.max()) if (c >= 2).any() else np.nan
    return res, out_pids.reset_index().assign(bead=name)


def gap_bias_check(name, pools):
    """A8: フレーム欠損を考慮した (frame 番号ベース) MSD と MSD.py (行番号ベース) の比較 @ 300 s"""
    root = msdmod.find_default_root()
    files = sorted(glob.glob(str(root / name / "*" / "*" / "beads_tracks.csv")))
    keep = set(pools[name]["filt"]["unique_particle_id"].unique())
    dt = 4.0
    ratios, rows_fa = [], []
    for vid, fp in enumerate(files):
        df = pd.read_csv(fp, usecols=["x", "y", "frame", "particle"])
        for pid, g in df.groupby("particle"):
            upid = f"vid{vid}_{pid}"
            if upid not in keep:
                continue
            g = g.sort_values("frame")
            fr = g["frame"].to_numpy() - g["frame"].min()
            pos = np.full((fr[-1] + 1, 2), np.nan)
            pos[fr] = SCALE * g[["x", "y"]].to_numpy()
            nmax = len(pos)
            lags = np.arange(1, nmax)
            msd = np.array([np.nanmean(np.sum((pos[k:] - pos[:-k]) ** 2, axis=1) / 2) if np.isfinite(pos[k:, 0] - pos[:-k, 0]).any() else np.nan for k in lags])
            rows_fa.append(pd.DataFrame({"unique_particle_id": upid, "lag time": dt * lags, "MSD": msd}))
    fa = pd.concat(rows_fa).dropna()
    cur = pools[name]["filt"]
    def at(dfm, t0):
        sub = dfm[(dfm["lag time"] >= t0 - 12) & (dfm["lag time"] <= t0 + 12)]
        b = sub.loc[(sub["lag time"] - t0).abs().groupby(sub["unique_particle_id"]).idxmin()]
        return b.set_index("unique_particle_id")["MSD"]
    out = {}
    for t0 in (40.0, 300.0):
        a, b = at(cur, t0), at(fa, t0)
        com = a.index.intersection(b.index)
        r = (a[com] / b[com]).to_numpy()
        out[f"rowbased/frameaware_MSD{int(t0)}_median"] = float(np.median(r)) if len(r) else np.nan
        out[f"rowbased/frameaware_MSD{int(t0)}_max"] = float(np.max(r)) if len(r) else np.nan
        out[f"ensMSD{int(t0)}_rowbased"] = float(a.mean()) if len(a) else np.nan
        out[f"ensMSD{int(t0)}_frameaware"] = float(b.mean()) if len(b) else np.nan
    return out, fa


# ---------------------------------------------------------------- models (1-component MSD)
def m_power(t, s2, K, a):
    return 2 * s2 + K * t ** a


def m_diff(t, s2, D):
    return 2 * s2 + 2 * D * t


def m_prw(t, s2, D, v, tau):
    return 2 * s2 + 2 * D * t + v ** 2 * tau ** 2 * (t / tau - 1 + np.exp(-t / tau))


def m_caged(t, s2, L2, tau, D):
    return 2 * s2 + L2 * (1 - np.exp(-t / tau)) + 2 * D * t


def m_cross(t, s2, K, a1, a2, tc):
    return 2 * s2 + K * t ** a1 * (1 + t / tc) ** (a2 - a1)


MODELS = {
    "M1_power+noise": (m_power, ["sigma2_um2", "K", "alpha"],
                       lambda t, y: [y[0] * 0.1, y[0] / t[0], 1.0],
                       ([0, 0, 0], [np.inf, np.inf, 3])),
    "M2_diffusion+noise": (m_diff, ["sigma2_um2", "D_um2s"],
                           lambda t, y: [y[0] * 0.1, y[-1] / (2 * t[-1])],
                           ([0, 0], [np.inf, np.inf])),
    "M3_PRW_ABP": (m_prw, ["sigma2_um2", "D_um2s", "v_ums", "tau_s"],
                   lambda t, y: [y[0] * 0.1, 1e-4, np.sqrt(y[-1]) / t[-1] * 2, 50.0],
                   ([0, 0, 0, 1], [np.inf, np.inf, np.inf, 1e5])),
    "M4_caged+hop": (m_caged, ["sigma2_um2", "L2_um2", "tau_s", "D_um2s"],
                     lambda t, y: [y[0] * 0.1, y[len(y) // 3], 50.0, y[-1] / (2 * t[-1])],
                     ([0, 0, 1, 0], [np.inf, np.inf, 1e5, np.inf])),
    "M5_crossover_power": (m_cross, ["sigma2_um2", "K", "alpha1", "alpha2", "tc_s"],
                           lambda t, y: [y[0] * 0.1, y[0] / t[0], 1.5, 1.0, 100.0],
                           ([0, 0, 0, 0, 4], [np.inf, np.inf, 3, 3, 5000])),
}


def fit_all(t, y, ylo, yhi):
    """log10 空間で重み付き最小二乗。sigma = bootstrap 95%CI の log 幅 / (2*1.96)"""
    ly = np.log10(y)
    sig = (np.log10(np.clip(yhi, 1e-12, None)) - np.log10(np.clip(ylo, 1e-12, None))) / (2 * 1.96)
    sig = np.where(np.isfinite(sig) & (sig > 0), sig, np.nanmedian(sig))
    sig = np.maximum(sig, 0.01)
    out = {}
    n = len(t)
    for key, (f, pn, p0f, bnd) in MODELS.items():
        def lf(tt, *p, f=f):
            return np.log10(np.maximum(f(tt, *p), 1e-12))
        best = None
        p0 = np.array(p0f(t, y), float)
        p0 = np.clip(p0, np.array(bnd[0]) + 1e-9, np.where(np.isinf(bnd[1]), 1e12, np.array(bnd[1]) - 1e-9))
        # 数個の初期値から最良を選ぶ (tau 系の局所解対策)
        trials = [p0]
        if "tau_s" in pn or "tc_s" in pn:
            idx = pn.index("tau_s") if "tau_s" in pn else pn.index("tc_s")
            for tv in (10.0, 30.0, 100.0, 300.0):
                q = p0.copy(); q[idx] = tv; trials.append(q)
        for q in trials:
            try:
                popt, pcov = curve_fit(lf, t, ly, p0=q, sigma=sig, absolute_sigma=True, bounds=bnd, maxfev=40000)
            except Exception:
                continue
            chi2 = float(np.sum(((ly - lf(t, *popt)) / sig) ** 2))
            if best is None or chi2 < best[2]:
                best = (popt, pcov, chi2)
        if best is None:
            out[key] = None
            continue
        popt, pcov, chi2 = best
        k = len(popt)
        perr = np.sqrt(np.clip(np.diag(pcov), 0, None))
        out[key] = dict(params=dict(zip(pn, popt)), errs=dict(zip(pn, perr)), chi2=chi2,
                        chi2_red=chi2 / max(n - k, 1), AIC=chi2 + 2 * k, BIC=chi2 + k * np.log(n), k=k, n=n,
                        rmse_log=float(np.sqrt(np.mean((ly - lf(t, *popt)) ** 2))), func=f, popt=popt)
    return out


# ---------------------------------------------------------------- main
def main():
    pools = load_pools()

    # ---- (A)
    rows, jumps_all, outl_all = [], [], []
    for name, d, _ in BEADS:
        tc, jumps = track_checks(name)
        mc, outl = msd_checks(name, pools[name])
        gb, fa = gap_bias_check(name, pools)
        pools[name]["filt_frameaware"] = fa
        rows.append(dict(bead=name, diameter_um=d, **tc, **mc, **gb))
        jumps_all += jumps
        outl_all.append(outl)
    dfA = pd.DataFrame(rows)
    dfA.to_csv(OUT / "msd_anomaly_summary.csv", index=False)
    pd.DataFrame(jumps_all).to_csv(OUT / "msd_anomaly_jump_particles.csv", index=False)
    pd.concat(outl_all).to_csv(OUT / "msd_anomaly_robust_z_particles.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 50):
        print("\n=== (A) anomaly summary ===")
        print(dfA.T.to_string())

    # ---- (B)
    fit_rows = []
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True)
    fig2, axes2 = plt.subplots(2, 3, figsize=(15, 6.5), sharex=True)
    colors = {"M1_power+noise": "#1f77b4", "M2_diffusion+noise": "#7f7f7f", "M3_PRW_ABP": "#d62728",
              "M4_caged+hop": "#2ca02c", "M5_crossover_power": "#9467bd"}
    for ax, axr, (name, d, _) in zip(axes.flat, axes2.flat, BEADS):
        P = pools[name]
        # フレーム欠損を補正した (frame 番号ベース) iMSD をフィットに使用
        st = msdmod.calc_pooled_MSD_stats(P["filt_frameaware"], n_bootstraps=500)
        # 生存バイアスを避け、粒子数が N_filt の 50% 以上 (最低3) の lag のみ使用
        nmin = min(P["n_filt"], max(3, int(np.ceil(0.5 * P["n_filt"]))))
        st = st[(st["count"] >= nmin) & (st["mean"] > 0)]
        t = st.index.to_numpy(float)
        y = st["mean"].to_numpy(float)
        lo, hi = st["boot_ci_low"].to_numpy(float), st["boot_ci_high"].to_numpy(float)
        res = fit_all(t, y, lo, hi)
        ax.fill_between(t, lo, hi, color="k", alpha=0.12, lw=0)
        ax.plot(t, y, "k.", ms=3, label=f"data (N≥{nmin})")
        tt = np.geomspace(t[0], t[-1], 300)
        valid = {k: v for k, v in res.items() if v is not None}
        bestk = min(valid, key=lambda k: valid[k]["BIC"])
        for k, r in valid.items():
            ax.plot(tt, r["func"](tt, *r["popt"]), color=colors[k], lw=2.2 if k == bestk else 1.2,
                    ls="-" if k == bestk else "--", label=f"{k} (ΔBIC={r['BIC'] - valid[bestk]['BIC']:.0f})")
            axr.plot(t, np.log10(y) - np.log10(r["func"](t, *r["popt"])), color=colors[k],
                     lw=2.0 if k == bestk else 1.0)
            row = dict(bead=name, diameter_um=d, model=k, n_points=r["n"], t_max_s=t[-1], chi2_red=r["chi2_red"],
                       AIC=r["AIC"], BIC=r["BIC"], dBIC=r["BIC"] - valid[bestk]["BIC"], rmse_log10=r["rmse_log"])
            for p in r["params"]:
                row[p] = r["params"][p]
                row[p + "_err"] = r["errs"][p]
            fit_rows.append(row)
        ax.set(xscale="log", yscale="log", title=f"{name} ({d} μm), best={bestk}",
               xlabel="lag time [s]", ylabel="1-comp. MSD [μm²]")
        ax.legend(fontsize=7, loc="upper left")
        axr.axhline(0, color="k", lw=0.8)
        axr.set(xscale="log", title=name, xlabel="lag time [s]", ylabel="log10 residual")
        axr.set_ylim(-0.3, 0.3)
    fig.tight_layout(); fig2.tight_layout()
    for fg, nm in ((fig, "msd_model_comparison"), (fig2, "msd_model_residuals")):
        fg.savefig(OUT / f"{nm}.png", dpi=150); fg.savefig(OUT / f"{nm}.svg")
    dfB = pd.DataFrame(fit_rows)
    dfB.to_csv(OUT / "msd_model_comparison_summary.csv", index=False)
    print("\n=== (B) model comparison (dBIC=0 is best) ===")
    cols = ["bead", "model", "chi2_red", "dBIC", "rmse_log10"]
    print(dfB[cols].to_string(index=False, float_format=lambda x: f"{x:.3g}"))
    print("\n--- best-model parameters ---")
    for b, g in dfB.groupby("bead", sort=False):
        r = g.loc[g["dBIC"].idxmin()]
        pr = {k: round(v, 4) for k, v in r.items() if isinstance(v, float) and not k.endswith("_err")
              and k not in ("diameter_um", "t_max_s", "chi2_red", "AIC", "BIC", "dBIC", "rmse_log10") and np.isfinite(v)}
        print(f"  {b}: {r['model']}  {pr}")
    m3 = dfB[dfB["model"] == "M3_PRW_ABP"][["bead", "sigma2_um2", "D_um2s", "v_ums", "tau_s", "tau_s_err", "dBIC"]]
    print("\n--- M3 (PRW/ABP) params ---\n" + m3.to_string(index=False, float_format=lambda x: f"{x:.3g}"))


if __name__ == "__main__":
    main()
