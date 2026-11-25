#!/usr/bin/env python3
"""
apply_global_data_filter.py

異常データ除外フィルタ（libs/experiment_cutoff.py: EXCLUDED_EXPERIMENTS, EXCLUDED_PARTICLES）を
全共有キャッシュ（scratch/*.pkl）と粒子・背景統計 CSV に一括適用する冪等スクリプト。

変更前には初回のみ *.unfiltered.bak バックアップを作成し、可逆性を保つ。
"""

import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))

from libs.experiment_cutoff import (
    EXCLUDED_EXPERIMENTS,
    EXCLUDED_PARTICLES,
    filter_excluded_data,
    is_excluded_experiment,
    is_excluded_particle,
    load_strict_active_particles,
)
import verify_cos_decomposition as vcd

ROOT_DIR = vcd.ROOT_DIR


def ensure_backup(path: Path) -> Path:
    bak = path.with_name(path.name + ".unfiltered.bak")
    if not bak.exists() and path.exists():
        shutil.copy2(path, bak)
    return bak


def sanitize_csv(path: Path, exp_col: str = "exp_dir", pid_col: str = "particle", cid_col: str = None) -> None:
    if not path.exists():
        print(f"  [SKIP] not found: {path}")
        return
    ensure_backup(path)
    df = pd.read_csv(path)
    n_before = len(df)
    df_clean = filter_excluded_data(df, exp_col=exp_col, particle_col=pid_col, cargo_id_col=cid_col)
    n_after = len(df_clean)
    df_clean.to_csv(path, index=False)
    print(f"  [CSV] {path.relative_to(CURRENT_DIR) if path.is_relative_to(CURRENT_DIR) else path}: {n_before} -> {n_after} (removed {n_before - n_after})")


def sanitize_df_pickle(path: Path, exp_col: str = "exp_dir", pid_col: str = "particle", cid_col: str = None) -> None:
    if not path.exists():
        print(f"  [SKIP] not found: {path}")
        return
    ensure_backup(path)
    df = pd.read_pickle(path)
    n_before = len(df)
    df_clean = filter_excluded_data(df, exp_col=exp_col, particle_col=pid_col, cargo_id_col=cid_col)
    n_after = len(df_clean)
    df_clean.to_pickle(path)
    print(f"  [PKL-DF] {path.relative_to(CURRENT_DIR)}: {n_before} -> {n_after} (removed {n_before - n_after})")


def sanitize_list_pickle(path: Path) -> None:
    if not path.exists():
        print(f"  [SKIP] not found: {path}")
        return
    ensure_backup(path)
    with open(path, "rb") as f:
        recs = pickle.load(f)
    n_before = len(recs)
    recs_clean = [
        r for r in recs
        if not is_excluded_particle(
            exp_name_or_path=r.get("exp_dir"),
            particle_id=r.get("particle"),
            cargo_id=r.get("particle_id") or r.get("cargo_id"),
        )
    ]
    n_after = len(recs_clean)
    with open(path, "wb") as f:
        pickle.dump(recs_clean, f)
    print(f"  [PKL-LIST] {path.relative_to(CURRENT_DIR)}: {n_before} -> {n_after} (removed {n_before - n_after})")


def sanitize_msd_pool_cache(path: Path) -> None:
    """
    scratch/msd_pool_cache.pkl の beads1um エントリをフィルタする。
    (vid0=beads1um001_crop の pid=7 -> vid0_7, および vid3=beads1um_crop -> vid3_*)
    """
    if not path.exists():
        print(f"  [SKIP] not found: {path}")
        return
    ensure_backup(path)
    with open(path, "rb") as f:
        obj = pickle.load(f)
    b1 = obj.get("beads1um")
    if isinstance(b1, dict):
        for key in ("filt", "raw"):
            df = b1.get(key)
            if isinstance(df, pd.DataFrame) and "unique_particle_id" in df.columns:
                n_b = len(df)
                uid = df["unique_particle_id"].astype(str)
                mask = ~((uid == "vid0_7") | uid.str.startswith("vid3_"))
                b1[key] = df[mask].copy()
                print(f"  [MSD-POOL] beads1um[{key}]: {n_b} -> {len(b1[key])}")
        if isinstance(b1.get("filt"), pd.DataFrame):
            b1["n_filt"] = int(b1["filt"]["unique_particle_id"].nunique())
        if isinstance(b1.get("raw"), pd.DataFrame):
            b1["n_tot"] = int(b1["raw"]["unique_particle_id"].nunique())
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def main():
    print(f"Applying canonical filter:\n  EXCLUDED_EXPERIMENTS = {sorted(EXCLUDED_EXPERIMENTS)}\n  EXCLUDED_PARTICLES   = {sorted(EXCLUDED_PARTICLES)}\n")

    # 1. Canonical particle & experiment CSVs
    sanitize_csv(CURRENT_DIR / "strict_active_particles.csv", exp_col="exp_name", pid_col="particle")
    sanitize_csv(CURRENT_DIR / "figure" / "chi_vs_radius" / "cos_vs_radius_particle_summary.csv", exp_col="exp_dir", pid_col="particle")
    sanitize_csv(CURRENT_DIR / "figure" / "cos_decomposition" / "cos_i_cargo_individual_stats.csv", exp_col=None, pid_col=None, cid_col="cargo_id")
    sanitize_csv(CURRENT_DIR / "figure" / "bg_angular_correlation" / "bg_angular_correlation_length_per_experiment.csv", exp_col="exp_dir", pid_col=None)
    sanitize_csv(CURRENT_DIR / "figure" / "bg_angular_correlation" / "bg_angular_correlation_points.csv", exp_col="exp_dir", pid_col=None)

    if ROOT_DIR.exists() and ROOT_DIR != CURRENT_DIR:
        for rel in [
            "figure/chi_vs_radius/cos_vs_radius_particle_summary.csv",
            "figure/cos_decomposition/cos_i_cargo_individual_stats.csv",
            "figure/bg_angular_correlation/bg_angular_correlation_length_per_experiment.csv",
            "figure/bg_angular_correlation/bg_angular_correlation_points.csv",
        ]:
            p_mir = ROOT_DIR / rel
            if p_mir.exists():
                sanitize_csv(
                    p_mir,
                    exp_col="exp_dir" if "cos_i_cargo" not in rel else None,
                    pid_col="particle" if "chi_vs_radius" in rel else None,
                    cid_col="cargo_id" if "cos_i_cargo" in rel else None,
                )

    # 2. Shared DataFrame caches in scratch/
    sanitize_df_pickle(CURRENT_DIR / "scratch" / "contrast_inst_cache.pkl", exp_col="exp_dir", pid_col="particle")
    sanitize_df_pickle(CURRENT_DIR / "scratch" / "contrast_imsd_cache.pkl", exp_col="exp_dir", pid_col="particle")
    sanitize_df_pickle(CURRENT_DIR / "scratch" / "df_inst_cache.pkl", exp_col="exp_dir", pid_col="particle")
    sanitize_df_pickle(CURRENT_DIR / "scratch" / "df_win_cache.pkl", exp_col="exp_dir", pid_col="particle")
    sanitize_df_pickle(CURRENT_DIR / "scratch" / "harmonics_frames.pkl", exp_col="exp", pid_col="pid")
    sanitize_df_pickle(CURRENT_DIR / "scratch" / "u_inf_per_frame_cache.pkl", exp_col="exp_dir", pid_col=None)
    sanitize_df_pickle(CURRENT_DIR / "scratch" / "cargo_flow_profile_extended_cache.pkl", exp_col="exp_dir", pid_col="particle")
    sanitize_df_pickle(CURRENT_DIR / "scratch" / "cargo_flow_profile_cache.pkl", exp_col="exp_dir", pid_col="particle")

    # 3. Shared list-of-dicts caches in scratch/
    sanitize_list_pickle(CURRENT_DIR / "scratch" / "per_point_v_dot_u_hist_cache.pkl")
    sanitize_list_pickle(CURRENT_DIR / "scratch" / "gmm_run_tumble_per_point_hist_cache.pkl")
    sanitize_msd_pool_cache(CURRENT_DIR / "scratch" / "msd_pool_cache.pkl")

    # Backup and clear kl_shuffle_bias_cache.pkl so calc_and_plot_kl_decomposition.py recomputes cleanly
    kl_shuf = CURRENT_DIR / "scratch" / "kl_shuffle_bias_cache.pkl"
    if kl_shuf.exists():
        ensure_backup(kl_shuf)
        kl_shuf.unlink()
        print("  [RESET] scratch/kl_shuffle_bias_cache.pkl removed after backup (will recompute from filtered data)")

    # 4. Assert zero remaining excluded items across core datasets
    strict = load_strict_active_particles()
    assert len(strict) == 65, f"Expected 65 strict active particles, got {len(strict)}"
    for ex_exp in EXCLUDED_EXPERIMENTS:
        assert not any(e == ex_exp for e, _ in strict), f"Found excluded exp {ex_exp} in strict_active_particles!"
    for ex_pair in EXCLUDED_PARTICLES:
        assert ex_pair not in strict, f"Found excluded particle {ex_pair} in strict_active_particles!"

    df_ci = pd.read_pickle(CURRENT_DIR / "scratch" / "contrast_inst_cache.pkl")
    assert len(df_ci) == 27135, f"Expected 27135 rows in contrast_inst_cache.pkl, got {len(df_ci)}"
    df_hf = pd.read_pickle(CURRENT_DIR / "scratch" / "harmonics_frames.pkl")
    assert len(df_hf) == 27135, f"Expected 27135 rows in harmonics_frames.pkl, got {len(df_hf)}"
    sub1 = df_ci[df_ci["bead_name"] == "beads1um"]
    assert sub1[["exp_dir", "particle"]].drop_duplicates().shape[0] == 12
    assert len(sub1) == 3605

    with open(CURRENT_DIR / "scratch" / "per_point_v_dot_u_hist_cache.pkl", "rb") as f:
        pp = pickle.load(f)
    assert len(pp) == 66, f"Expected 66 per-point records, got {len(pp)}"

    with open(CURRENT_DIR / "scratch" / "gmm_run_tumble_per_point_hist_cache.pkl", "rb") as f:
        gm = pickle.load(f)
    assert len(gm) == 50, f"Expected 50 GMM hist records, got {len(gm)}"

    print("\nALL ASSERTIONS PASSED: strict=65, contrast_inst=27135 (1um=12p/3605f), per_point=66, gmm_hist=50.")


if __name__ == "__main__":
    main()
