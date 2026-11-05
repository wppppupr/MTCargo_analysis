#!/usr/bin/env python3
"""
scratch/eta_hist_vs_lag.py

η_i のヒストグラムが速度を求める時間間隔 τ = 4, 8, 16 s (k = 1, 2, 4 フレーム) で二山のまま残るかを確認する。

定義 (各 k):
  v_c(t; k)   = [x(t+k) - x(t)] / (k Δt)                (cargo_dx の k フレーム和; 欠損フレームを跨ぐ窓は除外)
  U_fp(t; k)  = (1/k) Σ_{s=t}^{t+k-1} <u>_fp(s)        (貨物と一緒に動く足印平均流ベクトルの時間平均 = 同じ窓の流れの変位)
  η(t; k)     = v_c · Û_fp / |U_fp|
  k=1 は既存 footprint_heterogeneity_cache の η と同一定義 (整合チェックを出力)。
窓は重なりあり (全 t) で集計 (ヒストグラム用。CI は貨物単位なので重なりの影響は受けない)。
二山の判定: η∈[-1,2.5] のヒストグラム (bin 0.05) を幅3でならし、ピーク (停止側 |η|<0.3, 乗車側 η>0.5) と
  その間の谷の高さ比 valley/min(peak) を求める。比 < 1 なら谷あり、小さいほど二山が明瞭。
  95% CI: 動画→貨物の階層ブートストラップ (n=500)。
足印ベクトルは extract_footprint_heterogeneity.py と同じ環状マスクで再抽出し scratch/footprint_vec_cache.pkl に保存。
出力: figure/eta_hist_vs_lag/
"""
import sys, time
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd, h5py
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT)); sys.path.append(str(ROOT / "scratch"))
import extract_footprint_heterogeneity as efh
vcd = efh.vcd
OUT = ROOT / "figure" / "eta_hist_vs_lag"; OUT.mkdir(parents=True, exist_ok=True)
VEC = ROOT / "scratch" / "footprint_vec_cache.pkl"
DT = 4.0; KS = [1, 2, 4]; NB = 500
ORDER = ["beads06um", "beads1um", "beads3um", "beads5um", "beads7um", "beads20um"]
BINS = np.arange(-1.0, 2.5 + 1e-9, 0.05); CEN = 0.5 * (BINS[1:] + BINS[:-1])
RNG = np.random.default_rng(0)


def extract(task):
    exp_name, bname, rows = task
    p_rad, best = efh.RAD_MAP[bname], efh.BEST_MAP[bname]
    ed = [d for d in vcd.find_experiment_dirs(efh.ROOT_DIR, bname) if d.name == exp_name][0]
    out = []
    with h5py.File(ed / "GFP_flows.h5", "r", rdcc_nbytes=128 * 1024 * 1024) as f:
        flow = f[list(f.keys())[0]]; T, _, h, w = flow.shape
        for frm, grp in rows.groupby("frame"):
            if int(frm) >= T: continue
            for _, r in grp.iterrows():
                bx, by = int(round(r.x_px)), int(round(r.y_px))
                y0, y1, x0, x1 = max(0, by - best), min(h, by + best + 1), max(0, bx - best), min(w, bx + best + 1)
                patch = flow[int(frm), :, y0:y1, x0:x1]
                yg, xg = np.ogrid[y0 - by:y1 - by, x0 - bx:x1 - bx]; d2 = xg ** 2 + yg ** 2
                m = (d2 <= best ** 2) & (d2 > p_rad ** 2)
                if not m.any(): continue
                u, v = patch[0][m].astype(np.float32), patch[1][m].astype(np.float32)
                if not (np.hypot(u, v) > 0).any(): continue
                out.append((bname, exp_name, f"{exp_name}_{r.particle}", int(frm), float(u.mean()), float(v.mean()),
                            float(r.cargo_dx), float(r.cargo_dy)))
    print(f"[{exp_name}] {len(out)} rows", flush=True)
    return out


if VEC.exists():
    vec = pd.read_pickle(VEC)
else:
    t0 = time.time(); inst = pd.read_pickle(ROOT / "scratch" / "contrast_inst_cache.pkl")
    tasks = [(e, g.bead_name.iloc[0], g.copy()) for e, g in inst.groupby("exp_dir")]
    with Pool(6) as pool: res = pool.map(extract, tasks)
    vec = pd.DataFrame([x for r in res for x in r], columns=["bead_name", "exp_dir", "unique_pid", "frame", "ub", "vb", "dx", "dy"])
    vec.to_pickle(VEC); print(f"extracted {len(vec)} rows in {time.time() - t0:.0f} s")
vec = vec.sort_values(["unique_pid", "frame"]).reset_index(drop=True)


def eta_lag(k):
    rows = []
    for pid, g in vec.groupby("unique_pid", sort=False):
        fr = g.frame.values; A = g[["ub", "vb", "dx", "dy"]].values
        if len(fr) < k: continue
        c = np.vstack([np.zeros((1, 4)), np.cumsum(A, 0)]); S = c[k:] - c[:-k]          # window sums
        ok = (fr[k - 1:] - fr[:len(fr) - k + 1]) == k - 1                                 # no gaps inside window
        U = S[:, :2] / k; V = S[:, 2:] / k; nU = np.hypot(U[:, 0], U[:, 1])
        ok &= nU > 0
        eta = (V[ok] * U[ok]).sum(1) / nU[ok] ** 2
        rows.append(pd.DataFrame(dict(bead_name=g.bead_name.iloc[0], exp_dir=g.exp_dir.iloc[0], unique_pid=pid,
                                      frame=fr[:len(fr) - k + 1][ok], eta=eta)))
    return pd.concat(rows, ignore_index=True)


def hist(e):
    h, _ = np.histogram(e, BINS); h = h / max(h.sum(), 1) / 0.05
    return np.convolve(h, np.ones(3) / 3, mode="same")


def bimod(e):
    h = hist(e)
    i_s = np.where(np.abs(CEN) < 0.3)[0]; i_r = np.where(CEN > 0.5)[0]
    a = i_s[np.argmax(h[i_s])]; b = i_r[np.argmax(h[i_r])]
    if b <= a + 1: return dict(pk_stop=CEN[a], pk_run=CEN[b], h_stop=h[a], h_run=h[b], valley=np.nan, ratio=np.nan)
    j = a + np.argmin(h[a:b + 1])
    return dict(pk_stop=CEN[a], pk_run=CEN[b], h_stop=h[a], h_run=h[b], eta_valley=CEN[j],
                ratio=h[j] / min(h[a], h[b]), f_stop=np.mean(np.abs(e) < 0.1), median=np.median(e))


def hboot(d):
    mov = {m: {p: gg.eta.values for p, gg in g.groupby("unique_pid")} for m, g in d.groupby("exp_dir")}
    keys = list(mov); r = []
    for _ in range(NB):
        parts = []
        for m in RNG.choice(keys, len(keys)):
            ps = list(mov[m].values()); parts += [ps[i] for i in RNG.integers(0, len(ps), len(ps))]
        r.append(bimod(np.concatenate(parts))["ratio"])
    return np.nanpercentile(r, [2.5, 97.5])


log = []
P = lambda *a: (print(*a, flush=True), log.append(" ".join(map(str, a))))
E = {k: eta_lag(k) for k in KS}
fp = pd.read_pickle(ROOT / "scratch" / "footprint_heterogeneity_cache.pkl"); fp = fp[fp.u_fp > 0]
m = E[1].merge(fp.assign(eta0=fp.v_parallel / fp.u_fp)[["unique_pid", "frame", "eta0"]], on=["unique_pid", "frame"])
P(f"consistency k=1 vs footprint cache: n={len(m)}, max|Δη|={np.nanmax(np.abs(m.eta - m.eta0)):.2e}")
P("valley/min(peak) ratio of smoothed hist (bin 0.05, 3-bin boxcar); <1 = dip between stop and run peaks. CI = hier. boot movie→cargo")
rows = []
for b in ORDER + ["all"]:
    for k in KS:
        d = E[k] if b == "all" else E[k][E[k].bead_name == b]
        r = bimod(d.eta.values); lo, hi = hboot(d)
        rows.append(dict(bead_name=b, tau_s=k * DT, n=len(d), n_cargo=d.unique_pid.nunique(), **r, ratio_lo=lo, ratio_hi=hi))
        P(f"{b:10s} τ={k * DT:4.0f}s n={len(d):5d}  peaks η={r['pk_stop']:+.2f} / {r['pk_run']:.2f}  valley η={r.get('eta_valley', np.nan):.2f} "
          f"ratio={r['ratio']:.2f} [{lo:.2f},{hi:.2f}]  frac|η|<0.1={r['f_stop']:.3f}  median={r['median']:.2f}")
tab = pd.DataFrame(rows); tab.to_csv(OUT / "bimodality_vs_lag.csv", index=False)
(OUT / "summary.txt").write_text("\n".join(log) + "\n")

col = dict(zip(KS, ["#1f77b4", "#ff7f0e", "#2ca02c"]))
fig, axs = plt.subplots(2, 4, figsize=(15, 6.5), sharex=True); axs = axs.ravel()
for ax, b in zip(axs, ORDER + ["all"]):
    for k in KS:
        d = E[k] if b == "all" else E[k][E[k].bead_name == b]
        r = tab[(tab.bead_name == b) & (tab.tau_s == k * DT)].iloc[0]
        ax.plot(CEN, hist(d.eta.values), color=col[k], lw=1.4, label=f"τ={k * DT:.0f} s (dip ratio {r.ratio:.2f})")
    ax.axvline(0, color="0.6", lw=0.6); ax.axvline(1, color="0.6", lw=0.6, ls="--")
    ax.set_title(b.replace("beads", "") if b != "all" else "all sizes pooled", fontsize=10); ax.legend(fontsize=7)
    ax.set_xlim(-1, 2.5)
axs[-1].axis("off")
for ax in axs[4:7]: ax.set_xlabel(r"$\eta_i = v_\parallel / |\langle u\rangle_{fp}|$ (both averaged over τ)")
for ax in axs[[0, 4]]: ax.set_ylabel("PDF")
fig.tight_layout(); fig.savefig(OUT / "eta_hist_vs_lag.png", dpi=200)

fig, ax = plt.subplots(figsize=(5.5, 3.8))
for i, k in enumerate(KS):
    t = tab[tab.tau_s == k * DT].reset_index(drop=True); x = np.arange(len(t)) + (i - 1) * 0.15
    ax.errorbar(x, t.ratio, yerr=[t.ratio - t.ratio_lo, t.ratio_hi - t.ratio], fmt="o", color=col[k], ms=4, capsize=2, label=f"τ={k * DT:.0f} s")
ax.axhline(1, color="0.5", ls="--", lw=0.8)
ax.set_xticks(range(len(t))); ax.set_xticklabels([b.replace("beads", "") for b in t.bead_name])
ax.set_ylabel("valley / smaller peak"); ax.set_title("two-peak contrast vs velocity lag (lower = clearer dip)"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "dip_ratio_vs_lag.png", dpi=200)
print("saved", OUT)
