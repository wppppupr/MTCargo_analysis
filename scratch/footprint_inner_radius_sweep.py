#!/usr/bin/env python3
"""
scratch/footprint_inner_radius_sweep.py

足印円環の内側半径を R_c+δ (δ = 0, 0.5, 1 µm, ξ/2) に変え、Φ_fp, |<u>_fp|, η, cos ψ を再計算する。
目的: 貨物自身の像が GFP フロー (RAFT) に写り込み、足印の流れを貨物方向へ引っ張る効果
      (figure/cargo_image_injection) が η, cos ψ を見かけ上押し上げていないかの確認。
円環の幅は元の解析 (BEST_MAP - RAD_MAP) に固定し、円環全体を外側へずらす。
ξ はその動画の値 (contrast_inst_cache の xi, µm)。
定義: η = Σv_∥ / Σ|<u>_fp|, cos ψ = <v_∥ / |v_c|> (v_c>0 のフレーム), v_∥ = v_c·û_fp
出力: scratch/footprint_inner_radius_sweep_cache.pkl, figure/footprint_inner_radius_sweep/{table.csv, table.txt}
"""
import sys, time
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd, h5py

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT)); sys.path.append(str(ROOT / "scratch"))
import verify_cos_decomposition as vcd
from extract_footprint_heterogeneity import RAD_MAP, BEST_MAP, SCALE_UM_PER_PX, V_UNIT

OUT_CACHE = ROOT / "scratch" / "footprint_inner_radius_sweep_cache.pkl"
OUT_DIR = ROOT / "figure" / "footprint_inner_radius_sweep"
DELTAS = ["0", "0.5", "1", "xi/2"]
NB = 1000
RNG = np.random.default_rng(0)


def delta_px(tag, xi_um):
    d_um = {"0": 0.0, "0.5": 0.5, "1": 1.0, "xi/2": xi_um / 2}[tag]
    return d_um / SCALE_UM_PER_PX


def process(task):
    exp_name, bname, rows = task
    t0 = time.time()
    ed = [d for d in vcd.find_experiment_dirs(vcd.ROOT_DIR, bname) if d.name == exp_name]
    if not ed or not (ed[0] / "GFP_flows.h5").exists():
        print(f"[{exp_name}] flow not found"); return []
    p_rad, width = RAD_MAP[bname], BEST_MAP[bname] - RAD_MAP[bname]
    xi_um = float(rows["xi"].iloc[0])
    rings = {t: (p_rad + delta_px(t, xi_um), p_rad + delta_px(t, xi_um) + width) for t in DELTAS}
    r_ceil = int(np.ceil(max(o for _, o in rings.values())))
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
                vc = float(np.hypot(r["cargo_dx"], r["cargo_dy"]))
                for t, (ri, ro) in rings.items():
                    m = (dsq <= ro ** 2) & (dsq > ri ** 2)
                    u, v = patch[0][m], patch[1][m]
                    nrm = np.hypot(u, v); ok = nrm > 0
                    if not ok.any(): continue
                    phi = float(np.hypot(np.mean(u[ok] / nrm[ok]), np.mean(v[ok] / nrm[ok])))
                    ub, vb = float(u.mean()), float(v.mean()); ufp = float(np.hypot(ub, vb))
                    vpar = (r["cargo_dx"] * ub + r["cargo_dy"] * vb) / ufp if ufp > 0 else 0.0
                    out.append(dict(bead_name=bname, exp_dir=exp_name, xi_um=xi_um,
                                    rc_over_xi=r["rc_over_xi"], unique_pid=f"{exp_name}_{r['particle']}",
                                    frame=frm, delta=t, delta_um=(ri - p_rad) * SCALE_UM_PER_PX,
                                    n_px=int(m.sum()), phi_fp=phi, u_fp=ufp * V_UNIT,
                                    v_parallel=vpar * V_UNIT, v_cargo=vc * V_UNIT,
                                    cos_psi=(vpar / vc) if (vc > 0 and ufp > 0) else np.nan))
    print(f"[{exp_name}] {len(out)} rows, {time.time() - t0:.1f}s", flush=True)
    return out


def boot_hier_eta(sub):
    mov = {k: v.groupby("unique_pid")[["v_parallel", "u_fp"]].sum().values for k, v in sub.groupby("exp_dir")}
    keys = list(mov); res = []
    for _ in range(NB):
        tot = np.zeros(2)
        for k in RNG.choice(keys, len(keys)):
            g = mov[k]; tot += g[RNG.integers(0, len(g), len(g))].sum(0)
        res.append(tot[0] / tot[1])
    return np.percentile(res, [2.5, 97.5])


def summarize(d):
    rows = []
    for (b, t), s in d.groupby(["bead_name", "delta"], sort=False):
        lo, hi = boot_hier_eta(s)
        rows.append(dict(bead_name=b, rc_over_xi=s.groupby("exp_dir").rc_over_xi.first().mean(), delta=t,
                         delta_um=s.delta_um.mean(), n_px=s.n_px.mean(), N=len(s),
                         Phi_fp=s.phi_fp.mean(), u_fp=s.u_fp.mean(),
                         eta=s.v_parallel.sum() / s.u_fp.sum(), eta_lo=lo, eta_hi=hi,
                         cos_psi=s.cos_psi.mean()))
    tab = pd.DataFrame(rows)
    tab["_o"] = tab.bead_name.map({b: i for i, b in enumerate(RAD_MAP)})
    tab["_d"] = tab.delta.map({t: i for i, t in enumerate(DELTAS)})
    return tab.sort_values(["_o", "_d"]).drop(columns=["_o", "_d"]).reset_index(drop=True)


def main():
    if OUT_CACHE.exists() and "--recompute" not in sys.argv:
        d = pd.read_pickle(OUT_CACHE)
    else:
        inst = pd.read_pickle(ROOT / "scratch" / "contrast_inst_cache.pkl")
        tasks = [(e, g.bead_name.iloc[0], g.copy()) for e, g in inst.groupby("exp_dir")]
        with Pool(6) as p:
            d = pd.DataFrame([x for l in p.map(process, tasks) for x in l])
        d.to_pickle(OUT_CACHE)
    # sanity: δ=0 should reproduce footprint_heterogeneity_cache
    ref = pd.read_pickle(ROOT / "scratch" / "footprint_heterogeneity_cache.pkl")
    z = d[d.delta == "0"].merge(ref, on=["unique_pid", "frame"], suffixes=("", "_ref"))
    print(f"sanity δ=0 vs cache: n={len(z)}, max|ΔΦ|={np.abs(z.phi_fp - z.phi_t).max():.2e}, "
          f"max|Δu_fp|={np.abs(z.u_fp - z.u_fp_ref).max():.2e}")
    # keep only frames where every δ is defined (same sample across δ)
    k = d.groupby(["unique_pid", "frame"]).delta.transform("nunique") == len(DELTAS)
    print(f"common frames: {k.sum() // len(DELTAS)} / {d.groupby(['unique_pid','frame']).ngroups}")
    tab = summarize(d[k])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tab.to_csv(OUT_DIR / "table.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 100):
        txt = tab.round(3).to_string(index=False)
    (OUT_DIR / "table.txt").write_text(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
