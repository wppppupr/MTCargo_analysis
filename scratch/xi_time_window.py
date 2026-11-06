#!/usr/bin/env python3
"""
scratch/xi_time_window.py

時間窓 (既定 200 フレーム, 重なりなし) ごとに背景 GFP 流れの配向相関長 ξ(t) を求める。
定義は plot_bg_angular_correlation.py と同一: 仮想粒子サンプル (distance, frame, vp) を
<|u_bar|^2> (動画全体の値, per_experiment.csv) でベースライン補正し、窓内フレームで
frame_block_stats → fit_xi_from_curve (fit_range 0–20 µm, cutoff 1/e, min_corr 0.01, frame SEM 重み)。
窓のフレーム数が WIN/2 未満の端の窓は捨てる。
検証: 全フレームで同じ計算をして per_experiment.csv の xi_um と一致するかを確認する。
出力: scratch/xi_time_window_cache.pkl, figure/xi_time_window/{xi_t_windows.csv, xi_t_per_movie.csv, xi_t_summary.txt}
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
import verify_cos_decomposition as vcd
import plot_bg_angular_correlation as pbg

WIN = int(sys.argv[sys.argv.index("--win") + 1]) if "--win" in sys.argv else 200
SCALE = 0.11
OUT_DIR = ROOT / "figure" / "xi_time_window"
CACHE = ROOT / "scratch" / f"xi_time_window_cache_w{WIN}.pkl"
PER_EXP = ROOT / "figure" / "bg_angular_correlation" / "bg_angular_correlation_length_per_experiment.csv"


def fit(samples, r_um):
    m, s, _ = pbg.frame_block_stats(samples)
    res = pbg.fit_xi_from_curve(r_um, m, s, 0.0, 20.0, 0.01, cutoff_ratio=1 / np.e)
    return float(res["xi_um"]), float(res["xi_err_um"]), float(res["r2_log"])


def main():
    ref = pd.read_csv(PER_EXP)
    ref = ref[ref.bead_name != "control"]
    rows, check = [], []
    for _, r in ref.iterrows():
        zname = r.source.split(":")[1]
        ed = [d for d in vcd.find_experiment_dirs(vcd.ROOT_DIR, r.bead_name) if d.name == r.exp_dir]
        if not ed:
            print(f"[{r.exp_dir}] dir not found"); continue
        ds = pbg.open_virtual_point_dataset(ed[0] / zname)
        if ds is None:
            print(f"[{r.exp_dir}] {zname} not found"); continue
        frames = np.asarray(ds.coords["frame"].values, dtype=np.int64)
        samp, _, _, dist_px, npts, _ = pbg.virtual_point_samples(ds)
        ds.close()
        r_um = dist_px * SCALE

        # 動画全体の検証（per_experiment.csv との一致確認）
        u2_movie = float(r.u_bar_sq_mean)
        samp_movie_corr = (samp - u2_movie) / max(1 - u2_movie, 1e-4)
        xi_all, _, _ = fit(samp_movie_corr, r_um)
        check.append(dict(exp_dir=r.exp_dir, xi_ref=r.xi_um, xi_recomp=xi_all))

        # 遠距離（15-25 um）のプラトー判定用インデックス
        sub_plat = (r_um >= 15.0) & (r_um <= 25.0)

        for w0 in range(0, int(frames.max()) + 1, WIN):
            sel = (frames >= w0) & (frames < w0 + WIN)
            if sel.sum() < WIN / 2: continue
            sw = samp[:, sel, :]
            
            # 時間窓ローカルのベースライン補正:
            # 活性の非定常時間変化による長距離の残留相関プラトーを窓ごとに推定
            m_raw = np.nanmean(sw, axis=(1, 2))
            u2_local = float(np.mean(m_raw[sub_plat]))
            sw_corr = (sw - u2_local) / max(1 - u2_local, 1e-4)

            xi, err, r2 = fit(sw_corr, r_um)
            rows.append(dict(bead_name=r.bead_name, diameter_um=r.diameter_um, exp_dir=r.exp_dir,
                             win_start=w0, win_end=w0 + WIN, n_frames=int(sel.sum()),
                             xi_um=xi, xi_err_um=err, r2_log=r2, u2_local=u2_local,
                             xi_movie_um=r.xi_um, u2_movie=u2_movie))
        print(f"[{r.exp_dir}] xi_ref={r.xi_um:.2f} recomp={xi_all:.2f} n_win={sum(1 for q in rows if q['exp_dir']==r.exp_dir)}", flush=True)
    win = pd.DataFrame(rows); chk = pd.DataFrame(check)
    pd.to_pickle(dict(win=win, check=chk), CACHE)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    win.to_csv(OUT_DIR / f"xi_t_windows_w{WIN}.csv", index=False)

    g = win.groupby(["bead_name", "exp_dir"], sort=False)
    per = g.agg(diameter_um=("diameter_um", "first"), xi_movie=("xi_movie_um", "first"), n_win=("xi_um", "size"),
                xi_mean=("xi_um", "mean"), xi_sd=("xi_um", "std"), xi_min=("xi_um", "min"), xi_max=("xi_um", "max"),
                err_med=("xi_err_um", "median")).reset_index()
    per["cv"] = per.xi_sd / per.xi_mean
    per["max_over_min"] = per.xi_max / per.xi_min
    per.to_csv(OUT_DIR / f"xi_t_per_movie_w{WIN}.csv", index=False)

    # 全体の分散分解: 動画間 vs 動画内
    w = win.dropna(subset=["xi_um"])
    tot = w.xi_um.var(ddof=0)
    within = (w.xi_um - w.groupby("exp_dir").xi_um.transform("mean")).pow(2).mean()
    lines = [f"window = {WIN} frames (non-overlapping; partial windows < {WIN//2} frames dropped)",
             f"check (all frames recomputed vs per_experiment.csv): max |diff| = {(chk.xi_recomp-chk.xi_ref).abs().max():.3g} µm",
             "", per.to_string(index=False, float_format=lambda v: f"{v:.2f}"), "",
             f"pooled var(xi) = {tot:.3f} µm^2, within-movie fraction = {within/tot:.2f}",
             f"median within-movie CV = {per.cv.median():.3f}; median fit err / xi = {(w.xi_err_um/w.xi_um).median():.3f}"]
    txt = "\n".join(lines); print(txt)
    (OUT_DIR / f"xi_t_summary_w{WIN}.txt").write_text(txt + "\n")


if __name__ == "__main__":
    main()
