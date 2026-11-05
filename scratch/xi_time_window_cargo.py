#!/usr/bin/env python3
"""
scratch/xi_time_window_cargo.py

同じ貨物の中で cos ψ と η が窓ごとの ξ(t) に依存するかを調べる (貨物内 = 固定効果)。
入力: scratch/xi_time_window_cache_w{WIN}.pkl (xi_time_window.py), scratch/footprint_heterogeneity_cache.pkl
(貨物, 窓) ごとに η_w = Σv_∥/Σ|<u>_fp|, cosψ_w = <v_∥/|v_c|> (v_c>0) を作り (窓内 MINF フレーム以上)、
窓が 2 つ以上ある貨物で ξ と指標をそれぞれ貨物平均から引いて傾き β = Σ δξ δy / Σ δξ² をサイズごとに求める。
95% CI は階層ブートストラップ (動画→貨物)。ξ は動画平均で割った ξ/ξ_movie でも同じ傾きを出す (log 比)。
出力: figure/xi_time_window/{cargo_window_table_w*.csv, within_cargo_slope_w*.csv/.txt}
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
WIN = int(sys.argv[sys.argv.index("--win") + 1]) if "--win" in sys.argv else 200
MINF = 20
NB = 2000
RNG = np.random.default_rng(0)
OUT = ROOT / "figure" / "xi_time_window"
ORDER = ["beads06um", "beads1um", "beads3um", "beads5um", "beads7um", "beads20um"]

win = pd.read_pickle(ROOT / "scratch" / f"xi_time_window_cache_w{WIN}.pkl")["win"]
fp = pd.read_pickle(ROOT / "scratch" / "footprint_heterogeneity_cache.pkl")
fp["win_start"] = (fp.frame // WIN) * WIN
fp["cpsi"] = np.where(fp.v_cargo > 0, fp.v_parallel / fp.v_cargo, np.nan)
cw = fp.groupby(["bead_name", "exp_dir", "unique_pid", "win_start"]).agg(
    n=("frame", "size"), sv=("v_parallel", "sum"), su=("u_fp", "sum"), cos_psi=("cpsi", "mean")).reset_index()
cw = cw[cw.n >= MINF]
cw["eta"] = cw.sv / cw.su
cw = cw.merge(win[["exp_dir", "win_start", "xi_um", "xi_movie_um"]], on=["exp_dir", "win_start"], how="inner")
cw = cw.dropna(subset=["xi_um"])
cw["lnxi"] = np.log(cw.xi_um / cw.xi_movie_um)
cw = cw[cw.groupby("unique_pid").win_start.transform("nunique") >= 2].copy()
for c in ["xi_um", "lnxi", "eta", "cos_psi"]:
    cw["d_" + c] = cw[c] - cw.groupby("unique_pid")[c].transform("mean")
cw.to_csv(OUT / f"cargo_window_table_w{WIN}.csv", index=False)


def slope(d, x, y):
    ok = d[["d_" + x, "d_" + y]].dropna()
    sxx = (ok["d_" + x] ** 2).sum()
    return (ok["d_" + x] * ok["d_" + y]).sum() / sxx if sxx > 0 else np.nan


def boot(d, x, y):
    mov = {k: [g for _, g in v.groupby("unique_pid")] for k, v in d.groupby("exp_dir")}
    keys = list(mov); res = []
    for _ in range(NB):
        parts = []
        for k in RNG.choice(keys, len(keys)):
            cs = mov[k]; parts += [cs[i] for i in RNG.integers(0, len(cs), len(cs))]
        res.append(slope(pd.concat(parts), x, y))
    return np.nanpercentile(res, [2.5, 97.5])


rows = []
for b in ORDER:
    d = cw[cw.bead_name == b]
    if d.empty: continue
    r = dict(bead_name=b, n_movies=d.exp_dir.nunique(), n_cargo=d.unique_pid.nunique(), n_cargo_win=len(d),
             sd_dxi_within=d.d_xi_um.std())
    for x in ["xi_um", "lnxi"]:
        for y in ["cos_psi", "eta"]:
            lo, hi = boot(d, x, y) if d.exp_dir.nunique() >= 2 else (np.nan, np.nan)
            r[f"b_{y}_{x}"] = slope(d, x, y); r[f"b_{y}_{x}_lo"] = lo; r[f"b_{y}_{x}_hi"] = hi
    rows.append(r); print(b, "done", flush=True)
tab = pd.DataFrame(rows)
tab.to_csv(OUT / f"within_cargo_slope_w{WIN}.csv", index=False)
fmt = lambda v: f"{v:+.3f}"
lines = [f"window {WIN} frames, >= {MINF} frames per (cargo, window), cargo with >= 2 windows; slope = within-cargo (fixed effect) OLS, 95% CI = hierarchical bootstrap movie->cargo (n={NB})", ""]
for _, r in tab.iterrows():
    lines.append(f"{r.bead_name:10s} movies={r.n_movies} cargo={r.n_cargo} cargo-win={r.n_cargo_win} SD(δξ)={r.sd_dxi_within:.2f} µm")
    for x, u in [("xi_um", "per µm"), ("lnxi", "per ln ξ")]:
        for y in ["cos_psi", "eta"]:
            k = f"b_{y}_{x}"
            lines.append(f"    d{y}/d{x} ({u}) = {fmt(r[k])} [{fmt(r[k+'_lo'])}, {fmt(r[k+'_hi'])}]")
txt = "\n".join(lines); print(txt)
(OUT / f"within_cargo_slope_w{WIN}.txt").write_text(txt + "\n")
