#!/usr/bin/env python3
"""
scratch/phi_ring_robustness_delta_width.py

【4.4】円環モデルの予測 Φ_fp が足印の取り方に頑健か。
足印の円環を  R_in = RAD_MAP + δ,  R_out = R_in + w  (δ = 0, 0.5, 1 µm; w = 0.2, 0.4, 0.6, 0.8, 1.0 µm)
に変え、実測 Φ_rms = sqrt(<Φ_t²>)（動画ごと）と予測を比べる。元の解析の円環 (w = BEST−RAD) も "orig" として含める
(δ=0, orig は footprint_heterogeneity_cache の phi_t と一致するはず: sanity)。

予測は 2 種類:
  thin : phi_ring_model.ring_phi(R_mid/ξ)  (細い円環近似, R_mid = 円環中央半径)  ← 元のモデル
  ann  : 実際の画素マスクの全画素対で Φ² = (1/N²) Σ_ij exp(−|r_i−r_j|/ξ)（幅の効果を含む; 同じ C(r)=e^{−r/ξ} 仮定）
ξ は bg_angular_correlation_length_per_experiment.csv の xi_um（動画ごと）。

出力: scratch/phi_ring_robustness_cache.pkl, figure/phi_ring_robustness/{per_movie.csv, per_size.csv, summary.csv, tables.md, *.png}
"""
import sys, time
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd, h5py
from scipy.signal import fftconvolve

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT)); sys.path.append(str(ROOT / "scratch"))
import verify_cos_decomposition as vcd
from extract_footprint_heterogeneity import RAD_MAP, BEST_MAP, SCALE_UM_PER_PX
from phi_ring_model import ring_phi, REFERENCE_BEADS

CACHE = ROOT / "scratch" / "phi_ring_robustness_cache.pkl"
OUT = ROOT / "figure" / "phi_ring_robustness"
DELTAS = [0.0, 0.5, 1.0]
WIDTHS = [0.2, 0.4, 0.6, 0.8, 1.0]
XI = pd.read_csv(ROOT / "figure" / "bg_angular_correlation" /
                 "bg_angular_correlation_length_per_experiment.csv").set_index("exp_dir").xi_um
BEADS = ["beads06um", "beads1um", "beads3um", "beads5um", "beads7um", "beads20um"]
DIAM = {"beads06um": 0.63, "beads1um": 1.18, "beads3um": 3.37, "beads5um": 5.0, "beads7um": 7.24, "beads20um": 20.0}


def configs(bname):
    """[(tag, delta_um, width_um, r_in_px, r_out_px)]"""
    p = RAD_MAP[bname]; out = [("orig", 0.0, (BEST_MAP[bname] - p) * SCALE_UM_PER_PX, p, BEST_MAP[bname])]
    for d in DELTAS:
        for w in WIDTHS:
            ri = p + d / SCALE_UM_PER_PX
            out.append((f"d{d:g}_w{w:g}", d, w, ri, ri + w / SCALE_UM_PER_PX))
    return out


def process(task):
    exp_name, bname, rows = task
    t0 = time.time()
    ed = [d for d in vcd.find_experiment_dirs(vcd.ROOT_DIR, bname) if d.name == exp_name]
    if not ed or not (ed[0] / "GFP_flows.h5").exists():
        print(f"[{exp_name}] flow not found"); return []
    cf = configs(bname); r_ceil = int(np.ceil(max(c[4] for c in cf)))
    out = []
    with h5py.File(ed[0] / "GFP_flows.h5", "r", rdcc_nbytes=128 * 1024 * 1024) as f:
        flow = f[list(f.keys())[0]]
        T, h, w = flow.shape[0], flow.shape[2], flow.shape[3]
        for frm, grp in rows.groupby("frame"):
            frm = int(frm)
            if frm >= T: continue
            for _, r in grp.iterrows():
                bx, by = int(round(r["x_px"])), int(round(r["y_px"]))
                y0, y1 = max(0, by - r_ceil), min(h, by + r_ceil + 1)
                x0, x1 = max(0, bx - r_ceil), min(w, bx + r_ceil + 1)
                patch = flow[frm, :, y0:y1, x0:x1].astype(np.float32)
                yg, xg = np.ogrid[y0 - by:y1 - by, x0 - bx:x1 - bx]
                dsq = xg ** 2 + yg ** 2
                nrm = np.hypot(patch[0], patch[1]); ok = nrm > 0
                ux = np.where(ok, patch[0] / np.where(ok, nrm, 1), 0); uy = np.where(ok, patch[1] / np.where(ok, nrm, 1), 0)
                for tag, d, wd, ri, ro in cf:
                    m = (dsq <= ro ** 2) & (dsq > ri ** 2) & ok
                    n = int(m.sum())
                    if n == 0: continue
                    out.append(dict(bead_name=bname, exp_dir=exp_name, unique_pid=f"{exp_name}_{r['particle']}",
                                    frame=frm, cfg=tag, delta_um=d, width_um=wd, n_px=n,
                                    phi=float(np.hypot(ux[m].sum() / n, uy[m].sum() / n))))
    print(f"[{exp_name}] {len(out)} rows, {time.time() - t0:.1f}s", flush=True)
    return out


def annulus_phi(ri, ro, xi_um):
    """全画素対 Φ² = <exp(−d/ξ)> を、マスクの自己相関（画素対の変位の数）から計算。"""
    R = int(np.ceil(ro))
    y, x = np.mgrid[-R:R + 1, -R:R + 1]; dsq = x ** 2 + y ** 2
    m = ((dsq <= ro ** 2) & (dsq > ri ** 2)).astype(float)
    A = np.rint(fftconvolve(m, m[::-1, ::-1]))
    yy, xx = np.mgrid[-2 * R:2 * R + 1, -2 * R:2 * R + 1]
    dist = np.hypot(xx, yy) * SCALE_UM_PER_PX
    return float(np.sqrt((A * np.exp(-dist / xi_um)).sum() / m.sum() ** 2))


def main():
    if CACHE.exists() and "--recompute" not in sys.argv:
        d = pd.read_pickle(CACHE)
    else:
        inst = pd.read_pickle(ROOT / "scratch" / "contrast_inst_cache.pkl")
        tasks = [(e, g.bead_name.iloc[0], g.copy()) for e, g in inst.groupby("exp_dir")]
        with Pool(6) as p:
            d = pd.DataFrame([x for l in p.map(process, tasks) for x in l])
        d.to_pickle(CACHE)
    ncfg = 1 + len(DELTAS) * len(WIDTHS)
    k = d.groupby(["unique_pid", "frame"]).cfg.transform("nunique") == ncfg
    print(f"common cargo-frames: {k.sum() // ncfg} / {d.groupby(['unique_pid', 'frame']).ngroups}")
    d = d[k]
    # sanity vs original footprint cache
    ref = pd.read_pickle(ROOT / "scratch" / "footprint_heterogeneity_cache.pkl")
    z = d[d.cfg == "orig"].merge(ref, on=["unique_pid", "frame"])
    print(f"sanity orig vs footprint cache: n={len(z)}, median|ΔΦ|={np.median(np.abs(z.phi - z.phi_t)):.2e}, "
          f"max={np.abs(z.phi - z.phi_t).max():.2e}")

    # per movie
    g = d.assign(phi2=d.phi ** 2).groupby(["bead_name", "exp_dir", "cfg"])
    mv = g.agg(delta_um=("delta_um", "first"), width_um=("width_um", "first"), phi2=("phi2", "mean"),
               phi_mean=("phi", "mean"), n=("phi", "size"), n_px=("n_px", "mean")).reset_index()
    mv["phi_rms_obs"] = np.sqrt(mv.phi2)
    mv["xi_um"] = mv.exp_dir.map(XI)
    cfmap = {b: {c[0]: c for c in configs(b)} for b in BEADS}
    ann_cache = {}
    thin, ann, Rmid = [], [], []
    for r in mv.itertuples():
        _, _, _, ri, ro = cfmap[r.bead_name][r.cfg]
        R = 0.5 * (ri + ro) * SCALE_UM_PER_PX; Rmid.append(R)
        thin.append(float(ring_phi(R / r.xi_um)))
        key = (ri, ro, r.xi_um)
        if key not in ann_cache: ann_cache[key] = annulus_phi(ri, ro, r.xi_um)
        ann.append(ann_cache[key])
    mv["R_mid_um"], mv["phi_pred_thin"], mv["phi_pred_ann"] = Rmid, thin, ann
    mv["res_thin"] = mv.phi_rms_obs - mv.phi_pred_thin
    mv["res_ann"] = mv.phi_rms_obs - mv.phi_pred_ann
    mv["diameter_um"] = mv.bead_name.map(DIAM)
    mv["reference"] = mv.bead_name.isin(REFERENCE_BEADS)
    OUT.mkdir(parents=True, exist_ok=True)
    mv.to_csv(OUT / "per_movie.csv", index=False)

    # per size (frame weighted, as phi_ring_model.size_table) + movie bootstrap CI of residual
    rng = np.random.default_rng(0)
    rows = []
    for (b, c), s in mv.groupby(["bead_name", "cfg"]):
        w = s.n.values
        row = dict(bead_name=b, diameter_um=DIAM[b], cfg=c, delta_um=s.delta_um.iloc[0], width_um=s.width_um.iloc[0],
                   n_movies=len(s), n_px=np.average(s.n_px, weights=w))
        for col in ["phi_rms_obs", "phi_pred_thin", "phi_pred_ann", "res_thin", "res_ann"]:
            row[col] = np.average(s[col], weights=w)
        bs = []
        for _ in range(2000):
            i = rng.integers(0, len(s), len(s)); bs.append(np.average(s.res_ann.values[i], weights=w[i]))
        row["res_ann_lo"], row["res_ann_hi"] = np.percentile(bs, [2.5, 97.5])
        rows.append(row)
    ps = pd.DataFrame(rows).sort_values(["diameter_um", "delta_um", "width_um"]).reset_index(drop=True)
    ps.to_csv(OUT / "per_size.csv", index=False)

    # agreement summary per cfg (20 µm excluded)
    from scipy.stats import pearsonr
    sm = []
    for c, s in mv[~mv.reference].groupby("cfg"):
        sm.append(dict(cfg=c, delta_um=s.delta_um.iloc[0] if c != "orig" else 0.0,
                       width_um=np.nan if c == "orig" else s.width_um.iloc[0],
                       mean_res_thin=s.res_thin.mean(), rms_res_thin=np.sqrt((s.res_thin ** 2).mean()),
                       mean_res_ann=s.res_ann.mean(), rms_res_ann=np.sqrt((s.res_ann ** 2).mean()),
                       r_ann=pearsonr(s.phi_pred_ann, s.phi_rms_obs)[0],
                       slope_ann=np.polyfit(s.phi_pred_ann, s.phi_rms_obs, 1)[0],
                       phi_obs_range=s.phi_rms_obs.max() - s.phi_rms_obs.min()))
    sm = pd.DataFrame(sm).sort_values(["delta_um", "width_um"], na_position="first").reset_index(drop=True)
    sm.to_csv(OUT / "summary.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
    print(sm.round(3).to_string(index=False))
    print(ps.round(3).to_string(index=False))

    # markdown tables: residual (obs − ann) per size × (δ, w)
    with open(OUT / "tables.md", "w") as f:
        f.write("# 円環モデルの頑健性: 実測 Φ_rms − 予測 (画素対モデル), サイズ × (δ, w)\n\n")
        f.write("値は 実測 / 予測 (残差)。動画は frame 数重み。20 µm は参考。\n\n")
        for dlt in DELTAS:
            f.write(f"## δ = {dlt:g} µm\n\n| 粒径 | " + " | ".join(f"w={w:g}" for w in WIDTHS) + " |\n|---|" + "---|" * len(WIDTHS) + "\n")
            for b in BEADS:
                cells = []
                for wd in WIDTHS:
                    r = ps[(ps.bead_name == b) & (ps.cfg == f"d{dlt:g}_w{wd:g}")].iloc[0]
                    cells.append(f"{r.phi_rms_obs:.3f} / {r.phi_pred_ann:.3f} ({r.res_ann:+.3f})")
                f.write(f"| {DIAM[b]:g} µm | " + " | ".join(cells) + " |\n")
            f.write("\n")
        f.write("## 一致の指標（20 µm 除く, 動画 n=19）\n\n")
        f.write(sm.round(3).to_markdown(index=False) if hasattr(sm, "to_markdown") else sm.round(3).to_string(index=False))
        f.write("\n")
    plot(mv, ps, sm)


def plot(mv, ps, sm):
    import matplotlib.pyplot as plt
    sp = ROOT / "libs" / "my_style.mplstyle"
    if sp.exists(): plt.style.use(str(sp))
    col = {"beads06um": "#882255", "beads1um": "#CC6677", "beads3um": "#DDCC77",
           "beads5um": "#117733", "beads7um": "#44AA99", "beads20um": "#999999"}
    fig, ax = plt.subplots(2, 3, figsize=(14, 8.5))
    # top: per-size residual vs width, one panel per δ
    for j, dlt in enumerate(DELTAS):
        a = ax[0, j]
        for b in BEADS:
            s = ps[(ps.bead_name == b) & (ps.delta_um == dlt) & (ps.cfg != "orig")].sort_values("width_um")
            ref = b in REFERENCE_BEADS
            a.errorbar(s.width_um, s.res_ann, yerr=[s.res_ann - s.res_ann_lo, s.res_ann_hi - s.res_ann],
                       color=col[b], marker="o", ms=4, capsize=2, lw=1.2, ls="--" if ref else "-",
                       alpha=0.5 if ref else 1, label=f"{DIAM[b]:g} µm" + (" (ref.)" if ref else ""))
            a.plot(s.width_um, s.res_thin, color=col[b], ls=":", lw=1, alpha=0.5 if ref else 0.8)
        a.axhline(0, color="k", lw=0.7)
        a.set(xlabel="ring width w (µm)", ylabel=r"$\Phi_{rms}^{obs} - \Phi^{pred}$",
              title=f"({'abc'[j]}) δ = {dlt:g} µm", ylim=(-0.25, 0.5))
    ax[0, 0].legend(fontsize=7.5, title="solid: pixel-pair model\ndotted: thin-ring model", title_fontsize=7.5)
    # bottom left: parity for all cfgs (non-ref)
    a = ax[1, 0]
    m = mv[(mv.cfg != "orig")]
    for b in BEADS:
        s = m[m.bead_name == b]
        a.scatter(s.phi_pred_ann, s.phi_rms_obs, s=10, color=col[b], alpha=0.35 if b in REFERENCE_BEADS else 0.7, lw=0)
    a.plot([0, 1], [0, 1], "k--", lw=1)
    a.set(xlim=(0.2, 1.0), ylim=(0.2, 1.0), aspect="equal", xlabel=r"predicted $\Phi$ (pixel-pair model)",
          ylabel=r"observed $\sqrt{\langle\Phi_t^2\rangle}$", title="(d) per movie, all 15 (δ, w)")
    # bottom middle: heatmap mean residual (ann), bottom right: heatmap RMS residual
    s2 = sm[sm.cfg != "orig"]
    for a, key, ttl, cm, vr in [(ax[1, 1], "mean_res_ann", "(e) mean residual (20 µm excl.)", "RdBu_r", (-0.1, 0.1)),
                                (ax[1, 2], "rms_res_ann", "(f) RMS residual (20 µm excl.)", "viridis", (0, 0.12))]:
        M = s2.pivot(index="delta_um", columns="width_um", values=key).loc[DELTAS, WIDTHS].values
        im = a.imshow(M, cmap=cm, vmin=vr[0], vmax=vr[1], origin="lower", aspect="auto")
        for i in range(M.shape[0]):
            for jj in range(M.shape[1]):
                a.text(jj, i, f"{M[i, jj]:+.3f}" if key.startswith("mean") else f"{M[i, jj]:.3f}",
                       ha="center", va="center", fontsize=8, color="k" if key.startswith("mean") else "w")
        a.set_xticks(range(len(WIDTHS)), [f"{w:g}" for w in WIDTHS]); a.set_yticks(range(len(DELTAS)), [f"{d:g}" for d in DELTAS])
        a.set(xlabel="ring width w (µm)", ylabel="δ (µm)", title=ttl); fig.colorbar(im, ax=a, shrink=0.8)
    fig.suptitle("Robustness of the ring model Φ_fp(R/ξ) to footprint choice: inner radius R_c+δ, width w", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "phi_ring_robustness.png", dpi=200)
    print("saved", OUT / "phi_ring_robustness.png")


if __name__ == "__main__":
    main()
