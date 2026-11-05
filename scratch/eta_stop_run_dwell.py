#!/usr/bin/env python3
"""
scratch/eta_stop_run_dwell.py

フレームごとの η_i(t) = v_∥ / |<u>_fp| (footprint_heterogeneity_cache.pkl) から
 1. 貨物ごとの停止割合 f_stop = frac(|η_i|<0.1)。張り付き貨物の判定:
    サイズ内の全フレームをまとめた f_stop に対し、貨物ごとの f_stop のばらつきが
    「全貨物が同じ確率で止まる」帰無仮説 (フレームを貨物間で入れ替え; 時間相関を保つため
    連続区間ブロックごとに円状シフトはせず、貨物ラベルを区間ブロック(BLK フレーム)単位で入れ替え) より大きいかを見る。
    f_stop >= 0.8 の貨物を「張り付き候補」として列挙する。
 2. 状態列: 停止 S (|η|<0.1) / 乗っている R (η >= 0.1 と η <= -0.1 も含む非停止)。
    主解析 = 単純閾値、頑健性 = ヒステリシス (S に入るのは |η|<0.1, R に入るのは η>0.5, 中間は直前の状態を保持)。
    フレーム欠損 (Δframe>1) で区間を切る。滞在時間: 両端が観測された (打ち切りなし) 滞在のみ分布に使う。
    切り替え速さ: k_{S→R} = (S→R 遷移数)/(S にいた総時間), k_{R→S} も同様 (打ち切りを含む指数 MLE)。
    95% CI は階層ブートストラップ (動画→貨物)。
出力: figure/eta_stop_run_dwell/
"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figure" / "eta_stop_run_dwell"; OUT.mkdir(parents=True, exist_ok=True)
DT = 4.0  # s / frame
TH, TH_RUN = 0.1, 0.5
STUCK = 0.8
NB = 1000; BLK = 25
RNG = np.random.default_rng(0)
ORDER = ["beads06um", "beads1um", "beads3um", "beads5um", "beads7um", "beads20um"]
COL = dict(zip(ORDER, plt.cm.viridis(np.linspace(0, 0.9, len(ORDER)))))

fp = pd.read_pickle(ROOT / "scratch" / "footprint_heterogeneity_cache.pkl")
fp = fp[fp.u_fp > 0].sort_values(["unique_pid", "frame"]).reset_index(drop=True)
fp["eta"] = fp.v_parallel / fp.u_fp
fp["seg"] = (fp.groupby("unique_pid").frame.diff() != 1).cumsum()


def states(eta, hyst):
    s = (np.abs(eta) < TH).astype(int)  # 1 = stop
    if not hyst: return s
    out = np.empty_like(s); cur = s[0]
    for i, e in enumerate(eta):
        if abs(e) < TH: cur = 1
        elif e > TH_RUN: cur = 0
        out[i] = cur
    return out


def runs(s):
    """(state, length, censored) for each run in one contiguous segment."""
    edges = np.flatnonzero(np.diff(s)) + 1
    st = np.r_[0, edges]; en = np.r_[edges, len(s)]
    return [(s[a], b - a, a == 0 or b == len(s)) for a, b in zip(st, en)]


def cargo_stats(hyst):
    rows, dw = [], []
    for (b, ex, pid), g in fp.groupby(["bead_name", "exp_dir", "unique_pid"], sort=False):
        ns = {0: 0, 1: 0}; tr = {0: 0, 1: 0}
        for _, gs in g.groupby("seg", sort=False):
            s = states(gs.eta.values, hyst)
            ns[1] += s.sum(); ns[0] += len(s) - s.sum()
            d = np.diff(s); tr[1] += (d == -1).sum(); tr[0] += (d == 1).sum()  # tr[1]: S->R, tr[0]: R->S
            for st, L, cen in runs(s):
                dw.append(dict(bead_name=b, unique_pid=pid, state="stop" if st else "run", L=L, censored=cen))
        rows.append(dict(bead_name=b, exp_dir=ex, unique_pid=pid, n=len(g), f_stop=(np.abs(g.eta) < TH).mean(),
                         T_stop=ns[1], T_run=ns[0], n_SR=tr[1], n_RS=tr[0],
                         med_vc=g.v_cargo.median(), med_ufp=g.u_fp.median(), eta_pool=g.v_parallel.sum() / g.u_fp.sum()))
    return pd.DataFrame(rows), pd.DataFrame(dw)


def rates(c):
    return pd.Series(dict(k_SR=c.n_SR.sum() / (c.T_stop.sum() * DT), k_RS=c.n_RS.sum() / (c.T_run.sum() * DT),
                          p_stop=c.T_stop.sum() / (c.T_stop.sum() + c.T_run.sum())))


def hboot(c, fn):
    mov = {k: v for k, v in c.groupby("exp_dir")}; keys = list(mov); res = []
    for _ in range(NB):
        parts = []
        for k in RNG.choice(keys, len(keys)):
            v = mov[k]; parts.append(v.iloc[RNG.integers(0, len(v), len(v))])
        res.append(fn(pd.concat(parts)))
    return pd.DataFrame(res).quantile([0.025, 0.975])


def null_sd_fstop(b):
    """SD of per-cargo f_stop if stop frames were exchangeable across cargo (block-shuffled, BLK frames)."""
    g = fp[fp.bead_name == b]
    blocks = [(gg.unique_pid.iloc[0], (np.abs(gg.eta.values) < TH)) for _, gg in g.groupby(["unique_pid", g.groupby("unique_pid").cumcount() // BLK])]
    pids = np.array([p for p, _ in blocks]); vals = [v for _, v in blocks]
    sds = []
    for _ in range(300):
        perm = RNG.permutation(len(vals))
        num = pd.Series([vals[i].sum() for i in perm]).groupby(pids).sum()
        den = pd.Series([len(vals[i]) for i in perm]).groupby(pids).sum()
        sds.append((num / den).std())
    return np.percentile(sds, [50, 97.5])


log = []
P = lambda *a: (print(*a, flush=True), log.append(" ".join(map(str, a))))
P(f"η_i(t)=v_∥/|<u>_fp| per frame (Δt={DT:.0f} s). stop: |η|<{TH}; hysteresis run: η>{TH_RUN}. stuck candidate: f_stop>={STUCK}. CI = hier. bootstrap movie→cargo (n={NB})")
results = {}
for hyst in (False, True):
    cs, dw = cargo_stats(hyst)
    tag = "hyst" if hyst else "simple"
    cs.to_csv(OUT / f"cargo_stop_fraction_{tag}.csv", index=False); dw.to_csv(OUT / f"dwell_runs_{tag}.csv", index=False)
    rows = []
    for b in ORDER:
        c = cs[cs.bead_name == b]
        if c.empty: continue
        r = rates(c); ci = hboot(c, rates) if c.exp_dir.nunique() >= 2 else None
        d = dw[(dw.bead_name == b) & ~dw.censored]
        row = dict(bead_name=b, n_cargo=len(c), n_movies=c.exp_dir.nunique(), **r.to_dict(),
                   tau_stop_mean_s=d[d.state == "stop"].L.mean() * DT, tau_run_mean_s=d[d.state == "run"].L.mean() * DT,
                   n_stop_dw=(d.state == "stop").sum(), n_run_dw=(d.state == "run").sum())
        for k in ["k_SR", "k_RS", "p_stop"]:
            row[k + "_lo"], row[k + "_hi"] = (ci[k].values if ci is not None else (np.nan, np.nan))
        if not hyst:
            row["sd_fstop_obs"] = c.f_stop.std(); row["sd_fstop_null50"], row["sd_fstop_null975"] = null_sd_fstop(b)
            row["n_stuck"] = int((c.f_stop >= STUCK).sum()); row["fstop_max"] = c.f_stop.max()
        rows.append(row)
    tab = pd.DataFrame(rows); tab.to_csv(OUT / f"size_summary_{tag}.csv", index=False)
    results[tag] = (cs, dw, tab)
    P(f"\n== state def: {tag} ==")
    for _, r in tab.iterrows():
        s = (f"{r.bead_name:10s} N={r.n_cargo:2d}/{r.n_movies}mov  p_stop={r.p_stop:.2f}  "
             f"k_S→R={r.k_SR:.3f} [{r.k_SR_lo:.3f},{r.k_SR_hi:.3f}] /s  k_R→S={r.k_RS:.3f} [{r.k_RS_lo:.3f},{r.k_RS_hi:.3f}] /s  "
             f"<τ_stop>={r.tau_stop_mean_s:.1f}s(n={r.n_stop_dw}) <τ_run>={r.tau_run_mean_s:.1f}s(n={r.n_run_dw})")
        if tag == "simple":
            s += f"  SD(f_stop) obs={r.sd_fstop_obs:.3f} null={r.sd_fstop_null50:.3f}(97.5%={r.sd_fstop_null975:.3f})  max f_stop={r.fstop_max:.2f} stuck={r.n_stuck}"
        P(s)

cs, dw, tab = results["simple"]
st = cs[cs.f_stop >= STUCK]
P("\nstuck candidates (f_stop>=%.1f):" % STUCK + ("" if len(st) else " none"))
for _, r in st.iterrows():
    P(f"  {r.bead_name} {r.unique_pid} n={r.n} f_stop={r.f_stop:.2f} median|v_c|={r.med_vc:.3f} median|<u>_fp|={r.med_ufp:.3f} η_pool={r.eta_pool:.2f}")
top = cs.sort_values("f_stop", ascending=False).head(5)
P("top-5 f_stop: " + ", ".join(f"{r.unique_pid}({r.bead_name}) {r.f_stop:.2f}" for _, r in top.iterrows()))
(OUT / "summary.txt").write_text("\n".join(log) + "\n")

# ---------- figures ----------
fig, ax = plt.subplots(1, 3, figsize=(13, 4))
for i, b in enumerate(ORDER):
    c = cs[cs.bead_name == b]
    if c.empty: continue
    x = i + RNG.uniform(-0.15, 0.15, len(c))
    ax[0].scatter(x, c.f_stop, s=12 + c.n / 20, color=COL[b], alpha=0.8)
    r = tab[tab.bead_name == b].iloc[0]
    ax[0].hlines(r.p_stop, i - 0.3, i + 0.3, color="k", lw=1)
ax[0].axhline(STUCK, ls="--", color="0.5", lw=0.8)
ax[0].set_xticks(range(len(ORDER))); ax[0].set_xticklabels([b.replace("beads", "") for b in ORDER])
ax[0].set_ylabel(r"$f_{stop}$ = frac$(|\eta_i|<0.1)$ per cargo"); ax[0].set_title("stop fraction (bar = pooled; marker ∝ frames)")
for b in ORDER:
    d = dw[(dw.bead_name == b) & ~dw.censored]
    for j, stt in enumerate(["stop", "run"]):
        L = np.sort(d[d.state == stt].L.values) * DT
        if len(L) < 5: continue
        ax[1 + j].step(L, 1 - np.arange(len(L)) / len(L), where="post", color=COL[b], label=b.replace("beads", ""))
for j, stt in enumerate(["stop", "run"]):
    ax[1 + j].set_yscale("log"); ax[1 + j].set_xscale("log"); ax[1 + j].set_xlabel(f"dwell time τ_{stt} [s]")
    ax[1 + j].set_ylabel("CCDF"); ax[1 + j].set_title(f"{stt} dwell (uncensored, simple threshold)")
ax[2].legend(fontsize=7)
fig.tight_layout(); fig.savefig(OUT / "fstop_and_dwell_ccdf.png", dpi=200)

fig, ax = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
for j, (tag, (_, _, t)) in enumerate(results.items()):
    x = np.arange(len(t))
    for k, m, lab in [("k_SR", "o", r"$k_{S\to R}$ (unstick)"), ("k_RS", "s", r"$k_{R\to S}$ (stop)")]:
        ax[j].errorbar(x + (0.1 if m == "s" else -0.1), t[k], yerr=[t[k] - t[k + "_lo"], t[k + "_hi"] - t[k]], fmt=m, ms=4, capsize=2, label=lab)
    ax[j].set_xticks(x); ax[j].set_xticklabels([b.replace("beads", "") for b in t.bead_name]); ax[j].set_yscale("log")
    ax[j].set_title(f"switching rate ({tag})"); ax[j].set_xlabel("bead")
ax[0].set_ylabel("rate [1/s]"); ax[0].legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "switching_rates.png", dpi=200)
print("saved", OUT)
