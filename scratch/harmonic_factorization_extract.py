# per (particle, frame): substrate n-fold order S_n and per-point harmonics A_n = mean_j exp(i n (th_v - th_j))
import sys, pickle, numpy as np, pandas as pd, h5py
from multiprocessing import Pool
sys.path.insert(0, "/home/sasaki/MTCargo_analysis")
import verify_cos_decomposition as vcd
from calc_and_plot_per_point_v_dot_u import RAD_MAP, BEST_MAP, INST_CACHE
NMAX = 6
OUT = "/home/sasaki/MTCargo_analysis/scratch/harmonics_frames.pkl"

def proc(args):
    exp_name, bname, rows = args
    p_rad, best = RAD_MAP[bname], BEST_MAP[bname]
    eds = [d for d in vcd.find_experiment_dirs(vcd.ROOT_DIR, bname) if d.name == exp_name]
    if not eds: return []
    fp = eds[0] / "GFP_flows.h5"
    if not fp.exists(): return []
    out = []
    ns = np.arange(1, NMAX + 1)[:, None]
    with h5py.File(fp, "r") as f:
        flow = f[list(f.keys())[0]]
        T, h, w = flow.shape[0], flow.shape[2], flow.shape[3]
        for frm, grp in rows.groupby("frame"):
            fi = int(frm)
            if fi >= T: continue
            for _, r in grp.iterrows():
                dx, dy = float(r["cargo_dx"]), float(r["cargo_dy"])
                if not np.isfinite(dx) or not np.isfinite(dy) or np.hypot(dx, dy) <= 0: continue
                thv = np.arctan2(dy, dx)
                bx, by = int(round(r["x_px"])), int(round(r["y_px"]))
                y0, y1, x0, x1 = max(0, by-best), min(h, by+best+1), max(0, bx-best), min(w, bx+best+1)
                p = flow[fi, :, y0:y1, x0:x1]
                yg, xg = np.ogrid[y0-by:y1-by, x0-bx:x1-bx]
                d2 = xg**2 + yg**2
                m = (d2 <= best**2) & (d2 > p_rad**2)
                u, v = p[0][m].astype(float), p[1][m].astype(float)
                ok = np.hypot(u, v) > 0
                if ok.sum() < 3: continue
                thj = np.arctan2(v[ok], u[ok])
                M = np.exp(1j * ns * thj[None, :]).mean(axis=1)      # substrate harmonics
                A = np.exp(1j * ns * (thv - thj)[None, :]).mean(axis=1)  # per-point harmonics
                rec = dict(bead=bname, exp=exp_name, pid=f"{exp_name}_{int(r['particle'])}",
                           frame=fi, rcxi=float(r["rc_over_xi"]), npts=int(ok.sum()))
                for k in range(NMAX):
                    rec[f"S{k+1}"] = abs(M[k]); rec[f"Ac{k+1}"] = A[k].real; rec[f"As{k+1}"] = A[k].imag
                    rec[f"cpsi{k+1}"] = np.cos((k+1)*thv - np.angle(M[k]))
                out.append(rec)
    return out

if __name__ == "__main__":
    df = pd.read_pickle(INST_CACHE)
    tasks = [(e, b, g) for (b, e), g in df.groupby(["bead_name", "exp_dir"])]
    with Pool(10) as pool:
        res = pool.map(proc, tasks)
    recs = pd.DataFrame([x for s in res for x in s])
    recs.to_pickle(OUT); print(len(recs), "records")
