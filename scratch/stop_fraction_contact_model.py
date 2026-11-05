"""接触領域の微小管本数 N ∝ R_c とした停止割合の簡単な模型と実測の比較.
入力: figure/eta_stop_run_dwell/cargo_stop_fraction_{simple,hyst}.csv, size_summary_*.csv
模型:
 M0  p = const
 M1  (Poisson, 逆向き MT を1本でも含めば止まる) p = p0 + (1-p0)(1-exp(-R_c/R*))
 M1' (同, ノイズ床なし p0=0)                     p = 1-exp(-R_c/R*)
 M2  (速度論: k_RS = c R_c, k_SR 一定)            k_RS ∝ R_c を直接検定
"""
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
rng = np.random.default_rng(0)
D = {"beads06um":0.63,"beads1um":1.18,"beads3um":3.37,"beads5um":5.00,"beads7um":7.24,"beads20um":20.0}
IN = "figure/eta_stop_run_dwell"; OUT = "figure/stop_fraction_contact_model"
sizes = list(D); R = np.array([D[s]/2 for s in sizes])

def pooled(df):  # 時間で重みづけた p_stop
    return df.T_stop.sum()/df.n.sum()
def boot(df, B=1000):
    out = np.empty((B, len(sizes)))
    for j, s in enumerate(sizes):
        d = df[df.bead_name==s]; movs = d.exp_dir.unique(); g = {m: d[d.exp_dir==m] for m in movs}
        for b in range(B):
            parts = [g[m].sample(len(g[m]), replace=True, random_state=rng.integers(1<<31)) for m in rng.choice(movs, len(movs))]
            out[b, j] = pooled(pd.concat(parts))
    return out
M = {
 "M0 const":       (lambda th, r: np.full_like(r, th[0]), [0.3], ([0],[1])),
 "M1 p0+Poisson":  (lambda th, r: th[0]+(1-th[0])*(1-np.exp(-r/th[1])), [0.2, 10.], ([0,1e-3],[1,1e4])),
 "M1' Poisson":    (lambda th, r: 1-np.exp(-r/th[0]), [5.], ([1e-3],[1e4])),
}
def fit(p, sig):
    res = {}
    for k,(f,x0,bd) in M.items():
        r = least_squares(lambda th: (f(th,R)-p)/sig, x0, bounds=bd)
        chi2 = float((r.fun**2).sum()); kpar=len(x0)
        res[k] = dict(th=r.x, chi2=chi2, aic=chi2+2*kpar, f=f)
    return res

lines = []; fig, ax = plt.subplots(1, 3, figsize=(13, 4))
for i, sd in enumerate(["simple", "hyst"]):
    df = pd.read_csv(f"{IN}/cargo_stop_fraction_{sd}.csv")
    p = np.array([pooled(df[df.bead_name==s]) for s in sizes])
    bs = boot(df); lo, hi = np.percentile(bs, [2.5, 97.5], axis=0); sig = np.maximum(bs.std(0), 0.01)
    res = fit(p, sig)
    # 各ブートストラップ複製で M1 を当てて R*, p0 の CI
    th_b = np.array([least_squares(lambda th: (M["M1 p0+Poisson"][0](th,R)-b)/sig, [0.2,10.], bounds=([0,1e-3],[1,1e4])).x for b in bs[:300]])
    lines.append(f"== {sd} ==  p_stop(R_c): " + ", ".join(f"{s}:{v:.2f}[{l:.2f},{h:.2f}]" for s,v,l,h in zip(sizes,p,lo,hi)))
    for k,v in res.items():
        lines.append(f"  {k:15s} th={np.round(v['th'],3)}  chi2={v['chi2']:.1f} (dof={6-len(v['th'])})  AIC={v['aic']:.1f}")
    q = np.percentile(th_b, [2.5,50,97.5], axis=0)
    lines.append(f"  M1 boot: p0={q[1,0]:.2f}[{q[0,0]:.2f},{q[2,0]:.2f}]  R*={q[1,1]:.1f}[{q[0,1]:.1f},{q[2,1]:.1f}] um  (R*=1e4 は飽和なし=線形極限)")
    # 小粒子 3 点 (R_c<2um) だけで単調増加か
    lines.append(f"  small-cargo trend: slope dp/dR (R_c<=1.7um, WLS) = {np.polyfit(R[:3], p[:3], 1, w=1/sig[:3])[0]:+.3f} /um")
    a = ax[i]; a.errorbar(2*R, p, yerr=[p-lo, hi-p], fmt="o", color="k", capsize=3, label="data (hier. boot 95%)")
    rr = np.geomspace(0.25, 12, 200)
    for (k,v), c in zip(res.items(), ["0.6","C0","C3"]):
        a.plot(2*rr, v["f"](v["th"], rr), color=c, label=f"{k} (AIC {v['aic']:.0f})")
    a.set_xscale("log"); a.set_xlabel("diameter 2R_c (µm)"); a.set_ylabel("p_stop"); a.set_ylim(0,1)
    a.set_title(f"stop fraction, state def: {sd}"); a.legend(fontsize=7, frameon=False)
    # M2: k_RS vs R_c
    ss = pd.read_csv(f"{IN}/size_summary_{sd}.csv").set_index("bead_name").loc[sizes]
    k = ss.k_RS.values; e = [k-ss.k_RS_lo.values, ss.k_RS_hi.values-k]
    sl, ic = np.polyfit(np.log(R), np.log(k), 1)
    lines.append(f"  M2: k_RS = {np.round(k,3)} /s ; log-log slope = {sl:.2f} (N∝R_c なら 1)  ; k_SR = {np.round(ss.k_SR.values,3)}")
    ax[2].errorbar(2*R*(1+0.04*i), k, yerr=e, fmt="os"[i], capsize=3, label=f"k_R→S ({sd}), slope {sl:.2f}")
    ax[2].plot(2*rr, k[0]*rr/R[0], "--", color=f"C{i}", lw=1)
ax[2].set_xscale("log"); ax[2].set_yscale("log"); ax[2].set_xlabel("diameter 2R_c (µm)"); ax[2].set_ylabel("k_R→S (1/s)")
ax[2].set_title("M2: k_R→S ∝ R_c (dashed) vs data"); ax[2].legend(fontsize=7, frameon=False)
fig.tight_layout(); fig.savefig(f"{OUT}/stop_fraction_contact_model.png", dpi=200)
open(f"{OUT}/summary.txt","w").write("\n".join(lines)+"\n"); print("\n".join(lines))
