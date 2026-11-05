#!/usr/bin/env python3
"""
scratch/switch_triggered_phi_fopp.py
状態切り替え (eta_stop_run_dwell.py と同じ定義: stop |η|<0.1; hyst は run 復帰 η>0.5) の瞬間 k=0
(新しい状態の最初のフレーム) の前後 ±W フレームで Φ_fp (phi_t) と f_opp を平均する。
 - 貨物間差を除くため貨物ごとの平均を引いた偏差 ΔΦ, Δf_opp を使う。
 - 同一連続区間 (Δframe=1) 内のフレームのみ使用。
 - 対照: R→S には「R が続く」時刻 (k=-1,0 とも R)、S→R には「S が続く」時刻を同じ貨物から全数使う。
 - 指標: pre 窓 k∈[-3,-1] の平均 (イベント − 対照)。95% CI = 階層ブートストラップ (動画→貨物)。
出力: figure/switch_triggered_phi_fopp/
"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figure" / "switch_triggered_phi_fopp"; OUT.mkdir(parents=True, exist_ok=True)
DT, TH, TH_RUN, W, NB = 4.0, 0.1, 0.5, 10, 500
PRE = [-3, -2, -1]
RNG = np.random.default_rng(0)
ORDER = ["beads06um", "beads1um", "beads3um", "beads5um", "beads7um", "beads20um"]
COL = dict(zip(ORDER, plt.cm.viridis(np.linspace(0, 0.9, len(ORDER)))))
LAGS = np.arange(-W, W + 1)
VARS = ["phi_t", "f_opp"]

fp = pd.read_pickle(ROOT / "scratch" / "footprint_heterogeneity_cache.pkl")
fp = fp[fp.u_fp > 0].sort_values(["unique_pid", "frame"]).reset_index(drop=True)
fp["eta"] = fp.v_parallel / fp.u_fp
fp["seg"] = (fp.groupby("unique_pid").frame.diff() != 1).cumsum()
for v in VARS:
    fp["d_" + v] = fp[v] - fp.groupby("unique_pid")[v].transform("mean")


def states(eta, hyst):
    s = (np.abs(eta) < TH).astype(int)
    if not hyst: return s
    out = np.empty_like(s); cur = s[0]
    for i, e in enumerate(eta):
        if abs(e) < TH: cur = 1
        elif e > TH_RUN: cur = 0
        out[i] = cur
    return out


def collect(hyst):
    """per cargo: dict kind -> (sum[lag,var], count[lag]) ; kinds RS, RR(null), SR, SS(null)."""
    rec = []
    for (b, ex, pid), g in fp.groupby(["bead_name", "exp_dir", "unique_pid"], sort=False):
        acc = {k: [np.zeros((len(LAGS), 2)), np.zeros(len(LAGS)), 0] for k in ["RS", "RR", "SR", "SS"]}
        for _, gs in g.groupby("seg", sort=False):
            s = states(gs.eta.values, hyst); X = gs[["d_phi_t", "d_f_opp"]].values; n = len(s)
            for t in range(1, n):
                k = {(0, 1): "RS", (0, 0): "RR", (1, 0): "SR", (1, 1): "SS"}[(s[t - 1], s[t])]
                lo, hi = max(0, t - W), min(n, t + W + 1)
                idx = np.arange(lo, hi) - t + W
                a = acc[k]; a[0][idx] += X[lo:hi]; a[1][idx] += 1; a[2] += 1
        rec.append(dict(bead_name=b, exp_dir=ex, unique_pid=pid, acc=acc))
    return rec


def curve(recs, k):
    S = sum(r["acc"][k][0] for r in recs); C = sum(r["acc"][k][1] for r in recs)
    return S / np.maximum(C, 1)[:, None]


def pre_diff(recs, ev, nl):
    pi = np.isin(LAGS, PRE)
    return curve(recs, ev)[pi].mean(0) - curve(recs, nl)[pi].mean(0)


def hboot(recs, fn):
    mov = {}
    for r in recs: mov.setdefault(r["exp_dir"], []).append(r)
    keys = list(mov); out = []
    for _ in range(NB):
        sm = []
        for m in RNG.choice(keys, len(keys)):
            v = mov[m]; sm += [v[i] for i in RNG.integers(0, len(v), len(v))]
        out.append(fn(sm))
    return np.percentile(np.array(out), [2.5, 97.5], axis=0)


log = []
P = lambda *a: (print(*a, flush=True), log.append(" ".join(map(str, a))))
P(f"switch-triggered average of ΔΦ_fp, Δf_opp (cargo-mean subtracted), ±{W} frames (Δt={DT:.0f}s). pre window k={PRE}. "
  f"null: RR (for R→S) / SS (for S→R). CI: hier. bootstrap movie→cargo n={NB}")
res = {}
for hyst in (False, True):
    tag = "hyst" if hyst else "simple"; rec = collect(hyst); res[tag] = rec
    P(f"\n== {tag} ==")
    rows = []
    for b in ORDER + ["all_excl20um"]:
        r = [x for x in rec if (x["bead_name"] == b if b in ORDER else x["bead_name"] != "beads20um")]
        if not r: continue
        for ev, nl, name in [("RS", "RR", "R→S(stop)"), ("SR", "SS", "S→R(unstick)")]:
            d = pre_diff(r, ev, nl); ci = hboot(r, lambda s: pre_diff(s, ev, nl))
            ne = sum(x["acc"][ev][2] for x in r)
            rows.append(dict(state_def=tag, bead_name=b, event=ev, n_events=ne, dPhi_pre=d[0], dPhi_lo=ci[0, 0], dPhi_hi=ci[1, 0],
                             dfopp_pre=d[1], dfopp_lo=ci[0, 1], dfopp_hi=ci[1, 1]))
            P(f"{b:13s} {name:13s} n={ne:5d}  pre(ev−null): ΔΦ={d[0]:+.3f} [{ci[0,0]:+.3f},{ci[1,0]:+.3f}]  "
              f"Δf_opp={d[1]:+.3f} [{ci[0,1]:+.3f},{ci[1,1]:+.3f}]")
    pd.DataFrame(rows).to_csv(OUT / f"pre_window_{tag}.csv", index=False)
(OUT / "summary.txt").write_text("\n".join(log) + "\n")

for tag, rec in res.items():
    fig, ax = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for b in ORDER:
        r = [x for x in rec if x["bead_name"] == b]
        for j, (ev, nl, name) in enumerate([("RS", "RR", "R→S (stop onset)"), ("SR", "SS", "S→R (unstick)")]):
            c, cn = curve(r, ev), curve(r, nl)
            for i, lab in enumerate([r"$\Delta\Phi_{fp}$", r"$\Delta f_{opp}$"]):
                ax[i, j].plot(LAGS * DT, c[:, i] - cn[:, i], "-o", ms=2.5, color=COL[b], label=b.replace("beads", ""),
                              lw=2.2 if b == "beads20um" else 1.2, alpha=0.5 if b == "beads20um" else 1)
                ax[i, j].set_ylabel(lab + " (event − null)"); ax[0, j].set_title(name)
    for a in ax.flat:
        a.axvline(0, color="k", lw=0.8); a.axhline(0, color="0.5", lw=0.6, ls="--")
        a.axvspan(PRE[0] * DT, PRE[-1] * DT, color="0.9", zorder=0)
    for a in ax[1]: a.set_xlabel("time from switch [s]")
    ax[0, 0].legend(fontsize=7, ncol=2); fig.suptitle(f"switch-triggered average ({tag}); shaded = pre window")
    fig.tight_layout(); fig.savefig(OUT / f"switch_triggered_{tag}.png", dpi=200); plt.close(fig)
print("saved", OUT)
