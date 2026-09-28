"""plot_cargo_spin_velocity の純関数（円板領域・速度・統計・テーブル・作図）の単体テスト。"""
import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

current_dir = Path(__file__).resolve().parent.parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

import plot_cargo_spin_velocity as csv_mod  # noqa: E402
import plot_mt_orientation_distribution as mt_ori  # noqa: E402


def make_synthetic_points(n_per_bead: int = 40, seed: int = 0) -> pd.DataFrame:
    """points_table と同じ列構成の合成データ（M と v に相関を持たせる）。"""
    rng = np.random.default_rng(seed)
    rows = []
    for bead_name, dia, rc, offset in (('beads1um', 1.18, 0.59, 0.0),
                                       ('beads3um', 3.37, 1.685, 0.2)):
        for pid in range(2):
            for k in range(n_per_bead):
                m = float(np.clip(rng.normal(0.0, 0.5), -1, 1))
                v = float(abs(0.1 + offset + 0.3 * m + rng.normal(0, 0.02)))
                polar = float(np.clip(0.5 + 0.3 * rng.random(), 0, 1))
                rows.append({
                    'bead_name': bead_name,
                    'diameter_um': dia,
                    'radius_um': rc,
                    'exp_dir': f'exp{pid}',
                    'particle': pid,
                    'frame': k,
                    'time_s': 4.0 * k,
                    'x_um': 10.0, 'y_um': 10.0,
                    'm_ising': m,
                    'polar': polar,
                    'v_um_s': v,
                    'v_track_um_s': v,
                    'v_flow_um_s': v,
                    'v_flow_absmean_um_s': v,
                    'region_n_valid': 30,
                    'region_n_disk': 32,
                    'region_valid_fraction': 30 / 32,
                    'region_radius_um': max(2.0 * rc, 1.0),
                    'region_inner_um': 0.0,
                    'theta_rad': 0.5,
                    'dir_sign': -1,
                    'velocity_source': 'tracked',
                })
    return pd.DataFrame(rows)


class TestRegionUtilities(unittest.TestCase):

    def test_region_radius_um_uses_factor_and_floor(self):
        """R_region = max(factor * R_c, min_region_um) になる。"""
        bead = {'radius_um': 0.315}
        self.assertAlmostEqual(csv_mod.region_radius_um(bead, 2.0, 1.0), 1.0)
        self.assertAlmostEqual(csv_mod.region_radius_um(bead, 10.0, 1.0), 3.15)
        self.assertAlmostEqual(csv_mod.region_radius_um(bead, 2.0, 0.0), 0.63)

    def test_disk_region_sums_matches_brute_force(self):
        """円板内の和が全画面ループの素朴実装と一致する（端の円板も含む）。"""
        rng = np.random.default_rng(1)
        mx = rng.normal(size=(20, 24)).astype(np.float32)
        my = rng.normal(size=(20, 24)).astype(np.float32)
        valid = np.ones_like(mx, dtype=bool)
        valid[0, :] = False
        phi = np.arctan2(my, mx)
        ux = np.where(valid, np.cos(phi), 0.0)
        uy = np.where(valid, np.sin(phi), 0.0)
        sp = np.where(valid, np.sign(mx), 0.0)
        sm = np.where(valid, np.sign(my), 0.0)

        yy, xx = np.mgrid[0:20, 0:24]
        for cx, cy, r in ((10.0, 10.0, 3.0), (0.5, 0.5, 2.5), (23.0, 19.5, 2.0)):
            res = csv_mod.disk_region_sums(mx, my, sp, sm, ux, uy, valid,
                                           cx=cx, cy=cy, r_out=r)
            mask = ((yy - cy) ** 2 + (xx - cx) ** 2) <= r ** 2
            m = mask & valid
            self.assertEqual(res['n_disk'], int(mask.sum()))
            self.assertEqual(res['n_valid'], int(m.sum()))
            self.assertAlmostEqual(res['sum_sigma_plus'], float(sp[m].sum()), places=6)
            self.assertAlmostEqual(res['sum_sigma_minus'], float(sm[m].sum()), places=6)
            self.assertAlmostEqual(res['sum_ux'], float(ux[m].sum()), places=6)
            self.assertAlmostEqual(res['sum_uy'], float(uy[m].sum()), places=6)
            self.assertAlmostEqual(res['sum_mag'], float(np.hypot(mx[m], my[m]).sum()), places=6)

    def test_disk_region_sums_inner_radius_and_empty(self):
        """内半径でビーズ直下をくり抜ける / 領域外や各有効画素なしは None。"""
        mx = np.ones((10, 10), dtype=np.float32)
        my = np.zeros((10, 10), dtype=np.float32)
        zeros = np.zeros((10, 10))
        valid = np.ones((10, 10), dtype=bool)
        phi = np.arctan2(my, mx)
        ux, uy = np.cos(phi), np.sin(phi)
        sp = np.sign(mx + my)
        sm = np.sign(mx - my)

        ann = csv_mod.disk_region_sums(mx, my, sp, sm, ux, uy, valid,
                                       cx=5.0, cy=5.0, r_out=3.0, r_in=1.5)
        yy, xx = np.mgrid[0:10, 0:10]
        d2 = (yy - 5.0) ** 2 + (xx - 5.0) ** 2
        self.assertEqual(ann['n_disk'], int(((d2 <= 9.0) & (d2 > 2.25)).sum()))
        self.assertLess(ann['n_disk'], int((d2 <= 9.0).sum()))

        # 画像外（完全にクリップ）
        self.assertIsNone(csv_mod.disk_region_sums(mx, my, sp, sm, ux, uy, valid,
                                                   cx=50.0, cy=50.0, r_out=2.0))
        # 有効画素が 0
        self.assertIsNone(csv_mod.disk_region_sums(mx, my, zeros, zeros,
                                                   np.zeros_like(ux), np.zeros_like(uy),
                                                   ~valid, cx=5.0, cy=5.0, r_out=3.0))


class TestVelocityUtilities(unittest.TestCase):

    def test_positions_and_velocity_lookup(self):
        """位置辞書と v = |dr| / (tau dt) の値（連続フレーム対のみ）を確認する。"""
        tracks = pd.DataFrame({
            'particle': [0, 0, 0, 0, 1, 1],
            'frame': [0, 1, 2, 4, 0, 1],
            'x': [0.0, 3.0, 6.0, 12.0, 0.0, 0.0],
            'y': [0.0, 4.0, 8.0, 16.0, 0.0, 5.0],
        })
        pos = csv_mod.cargo_positions_with_ids(tracks)
        self.assertEqual(sorted(pos.keys()), [0, 1, 2, 4])
        parts, xs, ys = pos[1]
        np.testing.assert_allclose(parts, [0, 1])
        np.testing.assert_allclose(xs, [3.0, 0.0])
        np.testing.assert_allclose(ys, [4.0, 5.0])

        v = csv_mod.tracked_velocity_lookup(tracks, tau=1, scale=0.11, frame_interval=4.0)
        # |dr| = 5 px -> 5 * 0.11 / 4 = 0.1375 um/s
        self.assertAlmostEqual(v[(0, 0)], 5.0 * 0.11 / 4.0, places=12)
        self.assertAlmostEqual(v[(1, 0)], 5.0 * 0.11 / 4.0, places=12)
        # frame 2 -> 3 が存在しないので (0, 2) は無い
        self.assertNotIn((0, 2), v)
        self.assertNotIn((0, 3), v)
        # frame 4 が最後なので (0, 4) も無い
        self.assertNotIn((0, 4), v)

    def test_velocity_tau_scaling(self):
        """tau = 2 では 2 フレーム分の変位を 2 フレーム時間で割る。"""
        tracks = pd.DataFrame({
            'particle': [0, 0, 0],
            'frame': [0, 1, 2],
            'x': [0.0, 1.0, 3.0],
            'y': [0.0, 0.0, 0.0],
        })
        v1 = csv_mod.tracked_velocity_lookup(tracks, tau=1, scale=1.0, frame_interval=2.0)
        v2 = csv_mod.tracked_velocity_lookup(tracks, tau=2, scale=1.0, frame_interval=2.0)
        self.assertAlmostEqual(v1[(0, 0)], 1.0 / 2.0)
        self.assertAlmostEqual(v1[(0, 1)], 2.0 / 2.0)
        self.assertAlmostEqual(v2[(0, 0)], 3.0 / 4.0)
        self.assertNotIn((0, 1), v2)


class TestBinning(unittest.TestCase):

    def test_bin_profile_records_quantile_bins(self):
        """等点数ビンの中央値・四分位が正しく、min_count 未満のビンは落ちる。"""
        x = np.linspace(0.0, 1.0, 100)
        y = np.linspace(0.0, 10.0, 100)
        recs = csv_mod.bin_profile_records(x, y, n_bins=5, min_count=5)
        self.assertEqual(len(recs), 5)
        for rec in recs:
            self.assertEqual(rec['n_points'], 20)
            self.assertAlmostEqual(rec['y_median'], rec['y_mean'], places=6)
            self.assertLessEqual(rec['y_q25'], rec['y_median'])
            self.assertLessEqual(rec['y_median'], rec['y_q75'])
        # ビン中心は昇順
        centers = [r['x_center'] for r in recs]
        self.assertEqual(centers, sorted(centers))
        # min_count が大きすぎると全ビンが落ちる
        self.assertEqual(csv_mod.bin_profile_records(x, y, n_bins=5, min_count=21), [])

    def test_bin_profile_records_drops_nan(self):
        """非有限のペアは除外される。"""
        x = np.array([0.0, 0.1, np.nan, 0.2, 0.3, 0.4])
        y = np.array([1.0, 2.0, 3.0, np.nan, 5.0, 6.0])
        recs = csv_mod.bin_profile_records(x, y, n_bins=2, min_count=1)
        self.assertEqual(sum(r['n_points'] for r in recs), 4)

    def test_binned_table_reduces_bins_for_small_samples(self):
        """点数が少ないときはビン数を自動的に減らす（n_bins_effective）。"""
        df_long = csv_mod.long_points_table(make_synthetic_points(n_per_bead=20))
        df_binned = csv_mod.binned_table(df_long, n_bins=12, min_count=10)
        self.assertFalse(df_binned.empty)
        self.assertTrue((df_binned['n_bins_effective'] <= 12).all())
        # 合成データは 1 条件あたり 40 点 -> 12 ビンでは 3 点/ビンになるため縮小される
        self.assertLessEqual(int(df_binned['n_bins_effective'].max()), 4)
        self.assertTrue({'m_ising', 'polar'}.issubset(set(df_binned['x_variable'])))


def make_synthetic_results() -> list:
    """process_experiment_cargo の戻り値と同じ構造の合成メタ情報を作る。"""
    records = []
    for pid in range(2):
        for k in range(6):
            m = 0.2 * (k - 2.5)
            records.append({
                'bead_name': 'beads1um',
                'exp_dir': 'exp',
                'particle': pid,
                'frame': k,
                'time_s': 4.0 * k,
                'x_um': 1.0, 'y_um': 2.0,
                'm_ising_plus': m,
                'm_ising_minus': -m,
                'polar': abs(m),
                'v_track_um_s': 0.3 + 0.1 * m,
                'v_flow_um_s': 0.25 + 0.1 * m,
                'v_flow_absmean_um_s': 0.25 + 0.12 * m,
                'region_n_valid': 10,
                'region_n_disk': 12,
                'region_valid_fraction': 10.0 / 12.0,
                'theta_rad': 0.0,
            })
    return [{
        'exp_dir': '/tmp/exp', 'bead_name': 'beads1um', 'records': records,
        'n_frames_used': 6, 'n_frames_total': 12, 'dataset_shape': (12, 2, 8, 8),
        'grid_shape': (2, 2), 'pixel_stride': 4, 'frame_stride': 5,
        'director': 'global', 'theta_source': 'global:zarr',
        'variants': {'plus': {'cos2_sum': 1.0, 'n_valid_sum': 10.0},
                     'minus': {'cos2_sum': 2.0, 'n_valid_sum': 10.0}},
        'region_radius_um': 1.18, 'region_radius_px': 10.7, 'region_inner_um': 0.0,
        'n_rejected_region': 1, 'n_rejected_no_velocity': 2,
        'n_frames_without_positions': 0, 'flow_cache_source': 'cache',
        'flow_cache_path': '/tmp/cache.h5', 'tau': 1, 'velocity': 'tracked',
    }]


class TestTables(unittest.TestCase):

    def test_points_table_selects_director_sign(self):
        """採用したディレクター符号に応じて m_ising（= ± m_ising_plus）が選ばれる。"""
        results = make_synthetic_results()
        df_p = csv_mod.points_table(results, sign=+1)
        df_m = csv_mod.points_table(results, sign=-1)
        # 合成データは polar = |m| なので、|m_ising| が polar と一致する
        np.testing.assert_allclose(np.abs(df_p['m_ising'].to_numpy()),
                                   df_p['polar'].to_numpy())
        np.testing.assert_allclose(df_m['m_ising'].to_numpy(),
                                   -df_p['m_ising'].to_numpy())
        self.assertTrue((df_m['m_ising'].abs() == df_p['m_ising'].abs()).all())
        self.assertTrue((df_p['dir_sign'] == 1).all())
        self.assertTrue((df_m['dir_sign'] == -1).all())
        self.assertTrue((df_p['region_radius_um'] == 1.18).all())

    def test_points_table_velocity_source(self):
        """--velocity flow では v_um_s がフロー速度になる。"""
        results = make_synthetic_results()
        df_t = csv_mod.points_table(results, velocity='tracked')
        df_f = csv_mod.points_table(results, velocity='flow')
        np.testing.assert_allclose(df_t['v_um_s'].to_numpy(),
                                   df_t['v_track_um_s'].to_numpy())
        np.testing.assert_allclose(df_f['v_um_s'].to_numpy(),
                                   df_f['v_flow_um_s'].to_numpy())
        self.assertNotAlmostEqual(float(df_f['v_um_s'].iloc[0]),
                                  float(df_f['v_track_um_s'].iloc[0]))

    def test_extraction_table(self):
        """実験ごとの集計テーブルが棄却数などを保持する。"""
        df = csv_mod.extraction_table(make_synthetic_results())
        self.assertEqual(len(df), 1)
        self.assertEqual(int(df['n_points'].iloc[0]), 12)
        self.assertEqual(int(df['n_rejected_region'].iloc[0]), 1)
        self.assertEqual(df['theta_source'].iloc[0], 'global:zarr')

    def test_long_points_table_duplicates_rows(self):
        """縦持ちテーブルは行数が 2 倍（M と P）になる。"""
        df_points = make_synthetic_points(n_per_bead=5)
        df_long = csv_mod.long_points_table(df_points)
        self.assertEqual(len(df_long), 2 * len(df_points))
        self.assertEqual(set(df_long['x_var']), set(csv_mod.X_VARIABLES))
        m_rows = df_long[df_long['x_var'] == 'm_ising']
        np.testing.assert_allclose(m_rows['x_value'].to_numpy(),
                                   m_rows['m_ising'].to_numpy())


class TestCorrelations(unittest.TestCase):

    def test_within_group_correlation(self):
        """群内で平均を引くと、群間のオフセットを除いた相関が 1 になる。"""
        x = np.array([0.0, 1.0, 2.0, 10.0, 11.0, 12.0])
        y = np.array([0.0, 2.0, 4.0, 1000.0, 1002.0, 1004.0])
        groups = np.array(['a', 'a', 'a', 'b', 'b', 'b'], dtype=object)
        r, n = csv_mod.within_group_correlation(x, y, groups)
        self.assertAlmostEqual(r, 1.0, places=10)
        self.assertEqual(n, 6)

    def test_within_group_correlation_zero_variance(self):
        """群内の変動が無ければ相関は NaN。"""
        x = np.array([1.0, 1.0, 2.0, 2.0])
        y = np.array([1.0, 1.0, 2.0, 2.0])
        groups = np.array(['a', 'a', 'b', 'b'], dtype=object)
        r, _ = csv_mod.within_group_correlation(x, y, groups)
        self.assertTrue(np.isnan(r))

    def test_corr_stats_linear_relation(self):
        """y = 2x + 1 のとき r = 1、傾き 2、切片 1。"""
        x = np.linspace(-1.0, 1.0, 21)
        v = 2.0 * x + 1.0
        groups = np.array(['g'] * x.size, dtype=object)
        st = csv_mod._corr_stats(x, v, groups)
        self.assertAlmostEqual(st['pearson_r'], 1.0, places=10)
        self.assertAlmostEqual(st['slope_ols'], 2.0, places=10)
        self.assertAlmostEqual(st['intercept_ols'], 1.0, places=10)
        self.assertEqual(st['within_n'], 21)

    def test_corr_stats_insufficient_samples(self):
        """サンプル不足 / 分散 0 では NaN（libs.ising_magnetization と同じ規約）。"""
        st = csv_mod._corr_stats(np.array([1.0, 1.0]), np.array([1.0, 2.0]),
                                 np.array(['g', 'g'], dtype=object))
        self.assertTrue(np.isnan(st['pearson_r']))
        self.assertTrue(np.isnan(st['slope_ols']))

    def test_summary_table(self):
        """条件 x 変数ごとの要約テーブルが相関と within_r を含む。"""
        df_long = csv_mod.long_points_table(make_synthetic_points(n_per_bead=30))
        df = csv_mod.summary_table(df_long, velocity='tracked', n_bins=5, min_count=5)
        self.assertEqual(len(df), 4)  # 2 条件 x 2 変数
        self.assertTrue({'pearson_r', 'spearman_rho', 'slope_ols', 'within_r'}
                        .issubset(set(df.columns)))
        self.assertTrue((df['n_points'] > 0).all())
        self.assertTrue(df['pearson_r'].between(-1.0, 1.0).all())
        self.assertTrue(df['within_r'].between(-1.0, 1.0).all())


class TestPlotHelpers(unittest.TestCase):

    def test_velocity_limits(self):
        """線形軸は 0 起点、対数軸は正の分位点から下限を作る。"""
        v = np.array([0.0, 0.1, 0.2, 0.3, 1.0])
        lo, hi = csv_mod._velocity_limits(v, 'linear')
        self.assertEqual(lo, 0.0)
        self.assertGreater(hi, 0.9)
        lo_l, hi_l = csv_mod._velocity_limits(v, 'log')
        self.assertGreater(lo_l, 0.0)
        self.assertGreater(hi_l, lo_l)

    def test_condition_arrays_subsamples(self):
        """max_points を指定すると等間隔に間引かれる（決定的）。"""
        df_long = csv_mod.long_points_table(make_synthetic_points(n_per_bead=50))
        x_full, v_full, p_full, g_full = csv_mod._condition_arrays(
            df_long, 'beads1um', 'm_ising', max_points=0)
        x_sub, v_sub, p_sub, g_sub = csv_mod._condition_arrays(
            df_long, 'beads1um', 'm_ising', max_points=10)
        self.assertEqual(x_sub.size, 10)
        self.assertEqual(v_sub.size, x_sub.size)
        self.assertEqual(p_sub.size, x_sub.size)
        self.assertEqual(g_sub.size, x_sub.size)
        self.assertGreaterEqual(x_full.size, x_sub.size)
        # 間引き後も元の要素が保たれる
        self.assertTrue(np.all(np.isin(x_sub, x_full)))


class TestPlots(unittest.TestCase):

    def test_all_figures_are_written(self):
        """条件別 / パネル / 重ね描き（M と P の 2 系統）が PNG と SVG を出力する。"""
        df_points = make_synthetic_points(n_per_bead=40)
        df_long = csv_mod.long_points_table(df_points)
        df_binned = csv_mod.binned_table(df_long, n_bins=6, min_count=5)
        beads = [mt_ori.BEAD_LOOKUP['beads1um'], mt_ori.BEAD_LOOKUP['beads3um']]

        with tempfile.TemporaryDirectory() as td:
            out = [Path(td)]
            for x_var in csv_mod.X_VARIABLES:
                for bead in beads:
                    csv_mod.plot_per_condition(df_long, df_binned, bead, x_var, out,
                                               max_points=100, sign_note='unit-test')
                csv_mod.plot_panels(df_long, df_binned, beads, x_var, out,
                                    ncols=2, max_points=100, sign_note='unit-test')
                csv_mod.plot_overlay(df_long, df_binned, beads, x_var, out,
                                     max_points=100, sign_note='unit-test')

            for x_var in csv_mod.X_VARIABLES:
                tag = csv_mod.X_FILE_TAG[x_var]
                for bead in beads:
                    for ext in ('png', 'svg'):
                        self.assertTrue((Path(td) / f"{tag}_{bead['name']}.{ext}").exists())
                for name in (f"{tag}_all_beads", f"{tag}_overlay"):
                    for ext in ('png', 'svg'):
                        self.assertTrue((Path(td) / f"{name}.{ext}").exists())
        plt.close('all')


class TestJointDensityHeatmap(unittest.TestCase):
    """2D ヒストグラム（同時分布 P(v, M)）のビン・密度・テーブル・作図のテスト。"""

    def test_heatmap_edges_modes(self):
        """横軸は物理範囲の等幅、縦軸 log は等比、quantile は等点数ビンになる。"""
        rng = np.random.default_rng(3)
        x = rng.uniform(-1.0, 1.0, 500)
        v = np.exp(rng.uniform(np.log(1e-3), np.log(1.0), 500))

        xe, ye = csv_mod.heatmap_edges(x, v, 'm_ising', x_bins=10, y_bins=12,
                                       y_edges_mode='log')
        self.assertEqual(xe.size, 11)
        self.assertAlmostEqual(float(xe[0]), -1.0)
        self.assertAlmostEqual(float(xe[-1]), 1.0)
        np.testing.assert_allclose(np.diff(xe), np.diff(xe)[0])
        self.assertTrue(np.all(np.diff(ye) > 0.0))
        self.assertTrue(np.all(ye > 0.0))
        ratio = ye[1:] / ye[:-1]
        np.testing.assert_allclose(ratio, ratio[0], rtol=1e-9)

        # P の物理範囲は [0, 1]
        xe_p, _ = csv_mod.heatmap_edges(x, v, 'polar', x_bins=8)
        self.assertAlmostEqual(float(xe_p[0]), 0.0)
        self.assertAlmostEqual(float(xe_p[-1]), 1.0)

        # quantile は各ビンの点数がほぼ等しい
        _, ye_q = csv_mod.heatmap_edges(x, v, 'polar', y_bins=10, y_edges_mode='quantile')
        cnt, _ = np.histogram(v, bins=ye_q)
        self.assertEqual(int(cnt.sum()), v.size)
        self.assertLess(int(cnt.max() - cnt.min()), 10)

        # 退化した入力でも 2 本以上の境界を返す
        _, ye_deg = csv_mod.heatmap_edges(np.zeros(5), np.zeros(5), 'polar',
                                          x_bins=4, y_bins=4)
        self.assertGreaterEqual(ye_deg.size, 3)

    def test_joint_density_normalization_and_peak(self):
        """密度 = count / (N_in * dx * dv)（面積を掛けた和が 1）でピーク位置が正しい。"""
        x = np.array([0.2, 0.21, 0.22, 0.81])
        v = np.array([0.1, 0.11, 0.12, 0.9])
        xe = np.array([0.0, 0.5, 1.0])
        ye = np.array([0.0, 0.5, 1.0])
        counts, density, n_in = csv_mod.joint_density(x, v, xe, ye)
        self.assertEqual(counts.shape, (2, 2))          # (ny, nx)
        self.assertEqual(n_in, 4)
        self.assertEqual(int(counts[0, 0]), 3)          # 低 v・低 x に 3 点
        self.assertEqual(int(counts[1, 1]), 1)
        self.assertEqual(int(counts.sum()), n_in)
        area = np.diff(ye)[:, None] * np.diff(xe)[None, :]
        self.assertAlmostEqual(float((density * area).sum()), 1.0, places=12)
        # ビンの面積は等しいので密度比 = 個数比
        self.assertAlmostEqual(float(density[0, 0] / density[1, 1]), 3.0, places=9)

    def test_joint_density_drops_out_of_range(self):
        """ビン範囲外の点は個数に含まれない。"""
        x = np.array([0.0, 0.1, 0.2, 0.3])
        v = np.array([0.05, 0.06, 5.0, 6.0])
        counts, _, n_in = csv_mod.joint_density(x, v, np.array([-0.5, 0.5]),
                                                np.array([0.0, 0.1]))
        self.assertEqual(n_in, 2)
        self.assertEqual(int(counts.sum()), 2)


    def test_pooled_arrays_concatenates_conditions(self):
        """プールすると各条件の点が連結される（空指定なら空配列）。"""
        df_long = csv_mod.long_points_table(make_synthetic_points(n_per_bead=10))
        beads = [mt_ori.BEAD_LOOKUP['beads1um'], mt_ori.BEAD_LOOKUP['beads3um']]
        x1, _, _, _ = csv_mod._condition_arrays(df_long, 'beads1um', 'polar')
        x2, _, _, _ = csv_mod._condition_arrays(df_long, 'beads3um', 'polar')
        xp, vp, pp, gp = csv_mod.pooled_arrays(df_long, beads, 'polar')
        self.assertEqual(xp.size, x1.size + x2.size)
        self.assertEqual(vp.size, xp.size)
        self.assertEqual(pp.size, xp.size)
        self.assertEqual(gp.size, xp.size)
        empty = csv_mod.pooled_arrays(df_long, [], 'polar')
        self.assertEqual(empty[0].size, 0)
        self.assertEqual(empty[1].size, 0)

    def test_heatmap_density_data_and_table(self):
        """プールした 2D ヒストグラムと long 形式テーブルが整合する。"""
        df_long = csv_mod.long_points_table(make_synthetic_points(n_per_bead=120))
        beads = [mt_ori.BEAD_LOOKUP['beads1um'], mt_ori.BEAD_LOOKUP['beads3um']]
        data = csv_mod.heatmap_density_data(df_long, beads, 'm_ising', x_bins=8, y_bins=10)
        self.assertIsNotNone(data)
        self.assertEqual(data['n_points'], 480)          # 2 条件 x 2 粒子 x 120
        self.assertLessEqual(data['n_points_in_range'], data['n_points'])
        self.assertGreater(data['n_points_in_range'], 0)
        self.assertEqual(data['counts'].shape,
                         (data['y_edges'].size - 1, data['x_edges'].size - 1))
        self.assertEqual(data['counts'].sum(), data['n_points_in_range'])

        df = csv_mod.heatmap_density_table(data, 'all')
        self.assertFalse(df.empty)
        self.assertTrue((df['count'] > 0).all())          # count = 0 のビンは含めない
        self.assertEqual(int(df['count'].sum()), data['n_points_in_range'])
        self.assertAlmostEqual(float((df['prob_density'] * df['bin_area']).sum()),
                               1.0, places=12)
        self.assertTrue((df['bead_name'] == 'all').all())
        self.assertTrue((df['x_variable'] == 'm_ising').all())
        self.assertTrue((df['x_center'] >= df['x_low']).all())
        self.assertTrue((df['x_center'] <= df['x_high']).all())
        # y_center は対数等間隔ビンの幾何平均
        np.testing.assert_allclose(
            df['y_center'].to_numpy(),
            np.sqrt(df['y_low'].to_numpy() * df['y_high'].to_numpy()))
        self.assertEqual(data['bead_names'], ['beads1um', 'beads3um'])

        # データが無い条件・空データでは None / 空テーブル
        self.assertIsNone(csv_mod.heatmap_density_data(df_long, [{'name': 'nosuch'}],
                                                       'polar'))
        self.assertTrue(csv_mod.heatmap_density_table(None).empty)

    def test_heatmap_figures_are_written(self):
        """プール / 周辺分布なし / 対数色 / 線形軸のヒートマップが PNG と SVG を出力する。"""
        df_long = csv_mod.long_points_table(make_synthetic_points(n_per_bead=60))
        beads = [mt_ori.BEAD_LOOKUP['beads1um'], mt_ori.BEAD_LOOKUP['beads3um']]
        with tempfile.TemporaryDirectory() as td:
            out = [Path(td)]
            for x_var in csv_mod.X_VARIABLES:
                tag = csv_mod.X_FILE_TAG[x_var]
                data = csv_mod.heatmap_density_data(df_long, beads, x_var,
                                                    x_bins=6, y_bins=8)
                csv_mod.plot_heatmap(data, x_var, out, title='unit-test',
                                     sign_note='unit-test')
                for ext in ('png', 'svg'):
                    self.assertTrue((Path(td) / f"{tag}_heatmap.{ext}").exists())

                csv_mod.plot_heatmap(data, x_var, out, basename=f"{tag}_nomarg",
                                     marginals=False)
                self.assertTrue((Path(td) / f"{tag}_nomarg.png").exists())

                csv_mod.plot_heatmap(data, x_var, out, basename=f"{tag}_logc",
                                     log_color=True)
                self.assertTrue((Path(td) / f"{tag}_logc.png").exists())

                data_lin = csv_mod.heatmap_density_data(df_long, beads, x_var, x_bins=6,
                                                        y_bins=8, y_edges_mode='linear',
                                                        x_edges_mode='quantile')
                csv_mod.plot_heatmap(data_lin, x_var, out, basename=f"{tag}_linear")
                self.assertTrue((Path(td) / f"{tag}_linear.png").exists())

            # データが無ければ何も描かずに戻る（例外を出さない）
            csv_mod.plot_heatmap(None, 'm_ising', out)
        plt.close('all')


if __name__ == '__main__':
    unittest.main()
