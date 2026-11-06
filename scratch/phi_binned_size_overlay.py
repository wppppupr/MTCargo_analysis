#!/usr/bin/env python3
"""
scratch/phi_binned_size_overlay.py

Φ_fp (phi_t) のビンごとに cos ψ と η をサイズ別に重ねて描き、Φ_fp をそろえてもサイズ差が残るかを
ビンごとの階層ブートストラップ（動画→貨物）で判定する。
定義（footprint_inner_radius_sweep と同じ）: η = Σv_∥ / Σ|<u>_fp|,  cos ψ = <v_∥/|v_c|> (v_c>0)
判定: 各ビンで 統計量 vs log10(d) の OLS 傾き と (大: d≥3.37) − (小: d≤1.18) の差 の 95% CI。
      CI が 0 を跨がなければ「サイズ差が残る」。各サイズは独立に、動画を復元抽出→その中で貨物を復元抽出。
入力: scratch/footprint_heterogeneity_cache.pkl
出力: figure/phi_binned_size_overlay/{overlay.png, per_size.csv, size_test.csv, size_test.txt}
"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os
ROOT = Path(__file__).resolve().parent.parent
OUT_DIRS = [ROOT / "figure" / "phi_binned_size_overlay"]
NAS_BASE = Path("/mnt/NAS-Ebanaru/Sasaki/MTsingleBeads/figure/phi_binned_size_overlay")
if Path("/mnt/NAS-Ebanaru/Sasaki/MTsingleBeads/figure").exists():
    OUT_DIRS.append(NAS_BASE)

if os.environ.get("EXCL20"):
    OUT_DIRS = [d / "excl20um" for d in OUT_DIRS]

for out_dir in OUT_DIRS:
    out_dir.mkdir(parents=True, exist_ok=True)

EDGES = np.array([0, 0.3, 0.5, 0.7, 0.85, 0.95, 1.0 + 1e-9])
NB, MIN_N, MIN_MOV = 2000, 30, 2
rng = np.random.default_rng(0)

d = pd.read_pickle(ROOT / "scratch" / "footprint_heterogeneity_cache.pkl")
d = d[np.isfinite(d[["phi_t", "v_parallel", "u_fp", "v_cargo"]]).all(axis=1)].copy()
if os.environ.get("EXCL20"): d = d[d.diameter_um < 20]
d["bin"] = np.digitize(d["phi_t"], EDGES) - 1
d["movie"] = d["unique_pid"].str.rsplit("_", n=1).str[0] if "exp_dir" not in d else d["exp_dir"].astype(str)
d["cpos"] = (d["v_cargo"] > 0).astype(float)
d["cos"] = np.where(d["v_cargo"] > 0, d["v_parallel"] / d["v_cargo"].where(d["v_cargo"] > 0), 0.0)
nbin = len(EDGES) - 1
sizes = sorted(d["diameter_um"].unique())

# 貨物×ビン の十分統計量: [n, Σv∥, Σ|<u>|, Σcos, n_cos]
def suff(g):
    S = np.zeros((nbin, 5))
    for b, gb in g.groupby("bin"):
        S[b] = [len(gb), gb.v_parallel.sum(), gb.u_fp.sum(), gb["cos"].sum(), gb.cpos.sum()]
    return S

def stats(S):  # S: (..., nbin, 5) -> eta, cos, n
    with np.errstate(invalid="ignore", divide="ignore"):
        return S[..., 1] / S[..., 2], S[..., 3] / S[..., 4], S[..., 0]

per, boot = [], {}
for sz in sizes:
    ds = d[d.diameter_um == sz]
    movies = sorted(ds.movie.unique())
    cargo_S = {m: np.stack([suff(g) for _, g in ds[ds.movie == m].groupby("unique_pid")]) for m in movies}
    full = sum(c.sum(0) for c in cargo_S.values())
    nmov = np.array([[np.sum(c[:, b, 0].sum() > 0) for b in range(nbin)] for c in cargo_S.values()]).sum(0)
    B = np.zeros((NB, nbin, 5))
    for i in range(NB):
        for m in rng.choice(movies, len(movies)):
            c = cargo_S[m]
            B[i] += c[rng.integers(0, len(c), len(c))].sum(0)
    eta0, cos0, n0 = stats(full); etab, cosb, nb = stats(B)
    ok = (n0 >= MIN_N) & (nmov >= MIN_MOV)
    etab[:, ~ok] = np.nan; cosb[:, ~ok] = np.nan
    etab[nb < 5] = np.nan; cosb[nb < 5] = np.nan
    boot[sz] = (etab, cosb)
    for b in range(nbin):
        per.append(dict(d_um=sz, bin=b, phi_lo=EDGES[b], phi_hi=min(EDGES[b+1], 1), n=int(n0[b]), n_movies=int(nmov[b]),
                        n_cargo=int(sum((c[:, b, 0] > 0).sum() for c in cargo_S.values())), ok=bool(ok[b]),
                        eta=eta0[b] if ok[b] else np.nan, eta_lo=np.nanpercentile(etab[:, b], 2.5) if ok[b] else np.nan,
                        eta_hi=np.nanpercentile(etab[:, b], 97.5) if ok[b] else np.nan,
                        cos=cos0[b] if ok[b] else np.nan, cos_lo=np.nanpercentile(cosb[:, b], 2.5) if ok[b] else np.nan,
                        cos_hi=np.nanpercentile(cosb[:, b], 97.5) if ok[b] else np.nan))
per = pd.DataFrame(per)
for out_dir in OUT_DIRS:
    per.to_csv(out_dir / "per_size.csv", index=False)

# ビンごとのサイズ差判定
tests = []
for b in range(nbin):
    pb = per[(per.bin == b) & per.ok]
    szs = pb.d_um.values
    row = dict(bin=b, phi=f"[{EDGES[b]:.2f},{min(EDGES[b+1],1):.2f})", sizes=",".join(f"{s:g}" for s in szs))
    for q, k in [("eta", 0), ("cos", 1)]:
        point = pb[q].values
        if len(szs) >= 3:
            x = np.log10(szs); X = x - x.mean()
            Y = np.stack([boot[s][k][:, b] for s in szs], 1)
            slope_b = (Y - np.nanmean(Y, 1, keepdims=True)) @ X / (X @ X)
            slope0 = (point - point.mean()) @ X / (X @ X)
            lo, hi = np.nanpercentile(slope_b, [2.5, 97.5])
            row.update({f"{q}_slope": slope0, f"{q}_slope_lo": lo, f"{q}_slope_hi": hi,
                        f"{q}_slope_sig": bool(lo > 0 or hi < 0)})
        small = [s for s in szs if s <= 1.18]; large = [s for s in szs if s >= 3.37]
        if small and large:
            diff_b = np.nanmean(np.stack([boot[s][k][:, b] for s in large], 1), 1) - \
                     np.nanmean(np.stack([boot[s][k][:, b] for s in small], 1), 1)
            diff0 = pb[pb.d_um.isin(large)][q].mean() - pb[pb.d_um.isin(small)][q].mean()
            lo, hi = np.nanpercentile(diff_b, [2.5, 97.5])
            row.update({f"{q}_LmS": diff0, f"{q}_LmS_lo": lo, f"{q}_LmS_hi": hi, f"{q}_LmS_sig": bool(lo > 0 or hi < 0)})
    tests.append(row)
tests = pd.DataFrame(tests)
test_str = (per[["d_um", "bin", "n", "n_movies", "n_cargo", "eta", "eta_lo", "eta_hi", "cos", "cos_lo", "cos_hi"]]
            .to_string(index=False, float_format="%.3f") + "\n\n" +
            tests.to_string(index=False, float_format="%.3f") + "\n")
for out_dir in OUT_DIRS:
    tests.to_csv(out_dir / "size_test.csv", index=False)
    with open(out_dir / "size_test.txt", "w") as f:
        f.write(test_str)
print(test_str)

# 図
cmap = plt.get_cmap("viridis"); col = {s: cmap(i / (len(sizes) - 1)) for i, s in enumerate(sizes)}
centers = 0.5 * (EDGES[:-1] + np.minimum(EDGES[1:], 1))
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, q, lab in [(axes[0], "cos", r"$\cos\psi$"), (axes[1], "eta", r"$\eta$")]:
    for j, s in enumerate(sizes):
        p = per[per.d_um == s].sort_values("bin")
        xx = centers + (j - (len(sizes) - 1) / 2) * 0.008
        ax.errorbar(xx, p[q], yerr=[p[q] - p[f"{q}_lo"], p[f"{q}_hi"] - p[q]], color=col[s], marker="o", ms=4,
                    capsize=2, lw=1.2, label=f"{s:g} µm")
    for b, r in tests.iterrows():
        if r.get(f"{q}_slope_sig", False) is True:
            ax.text(centers[b], 1.02, "*", transform=ax.get_xaxis_transform(), ha="center", fontsize=13)
    for e in EDGES[1:-1]: ax.axvline(e, color="0.85", lw=0.6, zorder=0)
    ax.set_xlabel(r"$\Phi_{fp}$ (bin centre)"); ax.set_ylabel(lab); ax.set_xlim(0, 1)
axes[0].legend(title="diameter", fontsize=8, frameon=False)
fig.suptitle("Φ_fp-binned cos ψ and η by cargo size (95% hierarchical bootstrap: movie→cargo; * slope vs log d ≠ 0)", fontsize=9)
fig.tight_layout()
for out_dir in OUT_DIRS:
    fig.savefig(out_dir / "overlay.png", dpi=200)
    fig.savefig(out_dir / "overlay.svg")
    print("saved", out_dir / "overlay.png", "and .svg")
