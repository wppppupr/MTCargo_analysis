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
                v_mt = 0.25
                v_tilde = v / v_mt
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
                    'v_tilde': v_tilde,
                    'v_track_tilde': v_tilde,
                    'v_flow_tilde': v_tilde,
                    'v_um_s': v,
                    'v_track_um_s': v,
                    'v_flow_um_s': v,
                    'v_flow_absmean_um_s': v,
                    'v_mt_um_s': v_mt,
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

        v = csv_mod.tracked_velocity_lookup(tracks, tau=1, scale=0.11, frame_interval=4.0,
                                            smooth_method='none')
        # (vx, vy, v_mag)
        # dx = 3, dy = 4 -> vx = 3 * 0.11 / 4, vy = 4 * 0.11 / 4, v_mag = 5 * 0.11 / 4 = 0.1375 um/s
        vx0, vy0, vmag0 = v[(0, 0)]
        self.assertAlmostEqual(vx0, 3.0 * 0.11 / 4.0, places=12)
        self.assertAlmostEqual(vy0, 4.0 * 0.11 / 4.0, places=12)
        self.assertAlmostEqual(vmag0, 5.0 * 0.11 / 4.0, places=12)
        vx1, vy1, vmag1 = v[(1, 0)]
        self.assertAlmostEqual(vx1, 0.0, places=12)
        self.assertAlmostEqual(vy1, 5.0 * 0.11 / 4.0, places=12)
        self.assertAlmostEqual(vmag1, 5.0 * 0.11 / 4.0, places=12)
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
        v1 = csv_mod.tracked_velocity_lookup(tracks, tau=1, scale=1.0, frame_interval=2.0,
                                             smooth_method='none')
        v2 = csv_mod.tracked_velocity_lookup(tracks, tau=2, scale=1.0, frame_interval=2.0,
                                             smooth_method='none')
        self.assertAlmostEqual(v1[(0, 0)][2], 1.0 / 2.0)
        self.assertAlmostEqual(v1[(0, 1)][2], 2.0 / 2.0)
        self.assertAlmostEqual(v2[(0, 0)][2], 3.0 / 4.0)
        self.assertNotIn((0, 1), v2)

    def test_moving_average_smoothing(self):
        """移動平均フィルタ（moving_average_1d, tracked_velocity_lookup）による軌跡平滑化をテストする。"""
        # 1D 移動平均の端点と内部の計算
        x = np.array([1.0, 5.0, 2.0, 8.0, 3.0])
        ma3 = csv_mod.moving_average_1d(x, window=3)
        self.assertAlmostEqual(ma3[1], (1.0 + 5.0 + 2.0) / 3.0)
        self.assertAlmostEqual(ma3[2], (5.0 + 2.0 + 8.0) / 3.0)
        self.assertAlmostEqual(ma3[3], (2.0 + 8.0 + 3.0) / 3.0)

        # 線形等速運動（x(t) = 2.0 * t）: 移動平均をかけても速度は厳密に保たれる
        frames = np.arange(20)
        tracks_linear = pd.DataFrame({
            'particle': 0,
            'frame': frames,
            'x': 2.0 * frames,
            'y': 1.0 * frames,
        })
        v_ma = csv_mod.tracked_velocity_lookup(tracks_linear, tau=1, scale=1.0, frame_interval=1.0,
                                               smooth_method='moving_average', smooth_window=3)
        for f in range(1, 18):
            vx, vy, vmag = v_ma[(0, f)]
            self.assertAlmostEqual(vx, 2.0, places=6)
            self.assertAlmostEqual(vy, 1.0, places=6)
            self.assertAlmostEqual(vmag, np.sqrt(5.0), places=6)

        # ノイズ付き軌跡: 移動平均により速度の標準偏差が減少する
        rng = np.random.default_rng(42)
        noise_x = rng.normal(0, 0.5, size=50)
        noise_y = rng.normal(0, 0.5, size=50)
        tracks_noisy = pd.DataFrame({
            'particle': 0,
            'frame': np.arange(50),
            'x': 2.0 * np.arange(50) + noise_x,
            'y': 1.0 * np.arange(50) + noise_y,
        })
        v_raw = csv_mod.tracked_velocity_lookup(tracks_noisy, tau=1, scale=1.0, frame_interval=1.0,
                                                smooth_method='none')
        v_smooth = csv_mod.tracked_velocity_lookup(tracks_noisy, tau=1, scale=1.0, frame_interval=1.0,
                                                   smooth_method='moving_average', smooth_window=3)
        vx_raw = np.array([v_raw[(0, f)][0] for f in range(49)])
        vx_smooth = np.array([v_smooth[(0, f)][0] for f in range(49)])
        self.assertLess(np.std(vx_smooth), np.std(vx_raw))

    def test_savgol_filter_smoothing(self):
        """Savitzky-Golay フィルタによる軌跡平滑化と速度ゆらぎ低減をテストする。"""
        # 線形等速運動（x(t) = 2.0 * t, y(t) = 1.0 * t）: SGフィルタをかけても速度は厳密に保たれる
        frames = np.arange(20)
        tracks_linear = pd.DataFrame({
            'particle': 0,
            'frame': frames,
            'x': 2.0 * frames,
            'y': 1.0 * frames,
        })
        v_sg = csv_mod.tracked_velocity_lookup(tracks_linear, tau=1, scale=1.0, frame_interval=1.0,
                                               savgol_window=5, savgol_poly=2)
        for f in range(19):
            vx, vy, vmag = v_sg[(0, f)]
            self.assertAlmostEqual(vx, 2.0, places=6)
            self.assertAlmostEqual(vy, 1.0, places=6)
            self.assertAlmostEqual(vmag, np.sqrt(5.0), places=6)

        # ノイズ付き軌跡: SGフィルタ適用により速度の標準偏差が減少する
        rng = np.random.default_rng(42)
        noise_x = rng.normal(0, 0.5, size=50)
        noise_y = rng.normal(0, 0.5, size=50)
        tracks_noisy = pd.DataFrame({
            'particle': 0,
            'frame': np.arange(50),
            'x': 2.0 * np.arange(50) + noise_x,
            'y': 1.0 * np.arange(50) + noise_y,
        })
        v_raw = csv_mod.tracked_velocity_lookup(tracks_noisy, tau=1, scale=1.0, frame_interval=1.0,
                                                savgol_window=1)
        v_smooth = csv_mod.tracked_velocity_lookup(tracks_noisy, tau=1, scale=1.0, frame_interval=1.0,
                                                   savgol_window=7, savgol_poly=2)
        vx_raw = np.array([v_raw[(0, f)][0] for f in range(49)])
        vx_smooth = np.array([v_smooth[(0, f)][0] for f in range(49)])
        self.assertLess(np.std(vx_smooth), np.std(vx_raw))

    def test_load_mean_mt_velocity(self):
        """velocities_mean.csv からの平均微小管速度の読み込みをテストする。"""
        with tempfile.TemporaryDirectory() as td:
            exp_path = Path(td)
            # ファイルが存在しない場合
            self.assertTrue(np.isnan(csv_mod.load_mean_mt_velocity(exp_path)))

            # mean_velocity 列がある場合
            df = pd.DataFrame({'frame': [0, 1, 2], 'mean_velocity': [0.2, 0.3, 0.4]})
            df.to_csv(exp_path / 'velocities_mean.csv', index=False)
            self.assertAlmostEqual(csv_mod.load_mean_mt_velocity(exp_path), 0.3)


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
            v_t = 0.3 + 0.1 * m
            v_f = 0.25 + 0.1 * m
            v_mt = 0.25
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
                'v_tilde': v_t / v_mt,
                'v_track_tilde': v_t / v_mt,
                'v_flow_tilde': v_f / v_mt,
                'v_selected_um_s': v_t,
                'v_track_um_s': v_t,
                'v_flow_um_s': v_f,
                'v_flow_absmean_um_s': 0.25 + 0.12 * m,
                'v_mt_um_s': v_mt,
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
        'v_mt_mean_um_s': 0.25,
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

    def test_hexbin_figures_are_written(self):
        """Hexbin プロット（全条件プール / 周辺分布なし / 線形色 / 絶対値）が PNG と SVG を出力する。"""
        df_long = csv_mod.long_points_table(make_synthetic_points(n_per_bead=60))
        beads = [mt_ori.BEAD_LOOKUP['beads1um'], mt_ori.BEAD_LOOKUP['beads3um']]
        with tempfile.TemporaryDirectory() as td:
            out = [Path(td)]
            for x_var in csv_mod.X_VARIABLES:
                tag = csv_mod.X_FILE_TAG[x_var]
                csv_mod.plot_hexbin(df_long, beads, x_var, out, title='unit-test',
                                    sign_note='unit-test', gridsize=15)
                for ext in ('png', 'svg'):
                    self.assertTrue((Path(td) / f"{tag}_hexbin.{ext}").exists())

                csv_mod.plot_hexbin(df_long, beads, x_var, out, basename=f"{tag}_nomarg",
                                    marginals=False, gridsize=15)
                self.assertTrue((Path(td) / f"{tag}_nomarg.png").exists())

                csv_mod.plot_hexbin(df_long, beads, x_var, out, basename=f"{tag}_linc",
                                    log_color=False, gridsize=15)
                self.assertTrue((Path(td) / f"{tag}_linc.png").exists())

                csv_mod.plot_hexbin(df_long, beads, x_var, out, basename=f"{tag}_abs",
                                    abs_values=True, gridsize=15)
                self.assertTrue((Path(td) / f"{tag}_abs.png").exists())

            # 空データではスキップ（例外を出さない）
            csv_mod.plot_hexbin(pd.DataFrame(columns=['bead_name', 'x_var', 'x_value', 'v_um_s']),
                                beads, 'm_ising', out)
        plt.close('all')

    def test_heatmap_abs_folds_signs(self):
        """--heatmap_abs: |M| と |v| に折り畳んだ同時密度（符号反転で不変、横軸 [0, 1]）。"""
        df_long = csv_mod.long_points_table(make_synthetic_points(n_per_bead=50))
        beads = [mt_ori.BEAD_LOOKUP['beads1um'], mt_ori.BEAD_LOOKUP['beads3um']]
        kw = dict(x_bins=8, y_bins=10)

        # (M, v) -> (-M, -v) としても |M| / |v| は変わらない（不変性の確認用）
        flipped = df_long.copy()
        mask = flipped['x_var'] == 'm_ising'
        flipped.loc[mask, 'x_value'] = -flipped.loc[mask, 'x_value']
        flipped.loc[mask, 'v_um_s'] = -flipped.loc[mask, 'v_um_s']

        data = csv_mod.heatmap_density_data(df_long, beads, 'm_ising',
                                            abs_values=True, **kw)
        self.assertTrue(data['abs_values'])
        self.assertTrue(np.all(data['x'] >= 0.0))       # |M| >= 0
        self.assertTrue(np.all(data['v'] >= 0.0))       # |v| >= 0
        self.assertAlmostEqual(float(data['x_edges'][0]), 0.0)
        self.assertAlmostEqual(float(data['x_edges'][-1]), 1.0)
        self.assertTrue(np.all(data['y_edges'] > 0.0))  # 対数ビン

        # 折り畳んだ値を手で集計した結果と一致する
        x, v, _, _ = csv_mod.pooled_arrays(df_long, beads, 'm_ising')
        counts, density, n_in = csv_mod.joint_density(np.abs(x), np.abs(v),
                                                      data['x_edges'], data['y_edges'])
        np.testing.assert_array_equal(data['counts'], counts)
        np.testing.assert_allclose(data['density'], density)
        self.assertEqual(int(data['n_points']), x.size)
        self.assertEqual(int(data['n_points_in_range']), n_in)

        # 符号を反転しても同じヒストグラムになる
        data_flip = csv_mod.heatmap_density_data(flipped, beads, 'm_ising',
                                                 abs_values=True, **kw)
        np.testing.assert_array_equal(data['counts'], data_flip['counts'])

        # テーブルは abs_values フラグ付きで、ビンが非負側だけになる
        df = csv_mod.heatmap_density_table(data, 'all')
        self.assertFalse(df.empty)
        self.assertTrue(df['abs_values'].all())
        self.assertTrue((df['x_low'] >= 0.0).all())
        self.assertTrue((df['y_low'] > 0.0).all())
        self.assertEqual(int(df['count'].sum()), int(data['n_points_in_range']))

        # 既定（abs_values なし）は従来どおり符号付きのまま
        signed = csv_mod.heatmap_density_data(df_long, beads, 'm_ising', **kw)
        self.assertFalse(signed['abs_values'])
        self.assertAlmostEqual(float(signed['x_edges'][0]), -1.0)
        self.assertFalse(bool(csv_mod.heatmap_density_table(signed)['abs_values'].any()))

        # P は元から非負なので横軸 [0, 1] のまま、縦軸だけ |v| になる
        pol = csv_mod.heatmap_density_data(df_long, beads, 'polar',
                                           abs_values=True, **kw)
        self.assertTrue(pol['abs_values'])
        self.assertTrue(np.all(pol['x'] >= 0.0))
        self.assertTrue(np.all(pol['v'] >= 0.0))

        # 統計ボックスの見出しとラベル（横軸は |M|、縦軸は |v|）
        txt = csv_mod._stats_text(np.abs(x), np.abs(v),
                                  np.zeros(x.size, dtype=np.int64),
                                  'm_ising', abs_values=True)
        self.assertIn(r'|\tilde{v}_{\parallel,i,t}|', txt)
        self.assertIn(r'|M_{i,t}|', csv_mod.X_LABELS_ABS['m_ising'])
        self.assertEqual(csv_mod.X_LABELS_ABS['polar'], csv_mod.X_LABELS['polar'])
        self.assertIn(r'|\tilde{v}_{\parallel,i,t}|', csv_mod.VELOCITY_LABEL_ABS)

        # x_limits を渡すと境界を上書きできる
        xe, _ = csv_mod.heatmap_edges(np.abs(x), np.abs(v), 'm_ising', x_bins=5,
                                      x_limits=(0.0, 1.0))
        self.assertAlmostEqual(float(xe[0]), 0.0)
        self.assertAlmostEqual(float(xe[-1]), 1.0)

    def test_heatmap_abs_figures_are_written(self):
        """絶対値版の図（<tag>_heatmap_abs）が PNG と SVG を出力する。"""
        df_long = csv_mod.long_points_table(make_synthetic_points(n_per_bead=60))
        beads = [mt_ori.BEAD_LOOKUP['beads1um'], mt_ori.BEAD_LOOKUP['beads3um']]
        with tempfile.TemporaryDirectory() as td:
            out = [Path(td)]
            for x_var in csv_mod.X_VARIABLES:
                tag = csv_mod.X_FILE_TAG[x_var]
                data = csv_mod.heatmap_density_data(df_long, beads, x_var,
                                                    x_bins=6, y_bins=8,
                                                    abs_values=True)
                self.assertTrue(data['abs_values'])
                csv_mod.plot_heatmap(data, x_var, out,
                                     basename=f"{tag}_heatmap_abs",
                                     title='unit-test', sign_note='unit-test')
                for ext in ('png', 'svg'):
                    self.assertTrue((Path(td) / f"{tag}_heatmap_abs.{ext}").exists())
        plt.close('all')


class TestPercentileCurves(unittest.TestCase):
    """パーセンタイル速度 vs |M| の曲線（ビン・分位点・テーブル・作図）のテスト。"""

    def test_percentile_edges_modes(self):
        """横軸は [0, 1] の等幅、quantile では等点数、退化入力でも境界を返す。"""
        xe = csv_mod.percentile_edges(np.array([0.1, 0.5, 0.9]), x_bins=5)
        self.assertEqual(xe.size, 6)
        self.assertAlmostEqual(float(xe[0]), 0.0)
        self.assertAlmostEqual(float(xe[-1]), 1.0)
        np.testing.assert_allclose(np.diff(xe), np.diff(xe)[0])

        x = np.linspace(0.0, 1.0, 200) ** 2          # 低 |M| 側に偏った分布
        xe_q = csv_mod.percentile_edges(x, x_bins=8, x_edges_mode='quantile')
        cnt, _ = np.histogram(x, bins=xe_q)
        self.assertEqual(int(cnt.sum()), x.size)
        self.assertLess(int(cnt.max() - cnt.min()), 5)

        xe_deg = csv_mod.percentile_edges(np.zeros(3), x_bins=4)
        self.assertGreaterEqual(xe_deg.size, 3)

    def test_percentile_profile_values(self):
        """ビンごとの p80/p90/p95 が np.percentile と一致し、min_count で間引かれる。"""
        x = np.array([0.05, 0.10, 0.15, 0.20, 0.85])     # 最後の 1 点だけ別のビン
        v = np.array([1.0, 2.0, 3.0, 4.0, 9.0])
        xe = np.array([0.0, 0.5, 1.0])
        recs = csv_mod.percentile_profile(x, v, xe, percentiles=(80.0, 90.0, 95.0),
                                          min_count=1)
        self.assertEqual(len(recs), 2)
        lo = recs[0]
        self.assertEqual(lo['count'], 4)
        self.assertAlmostEqual(lo['x_center'], 0.25)
        for p in (80.0, 90.0, 95.0):
            self.assertAlmostEqual(lo['percentiles'][p], float(np.percentile(v[:4], p)))
        self.assertEqual(recs[1]['count'], 1)
        self.assertAlmostEqual(recs[1]['percentiles'][80.0], 9.0)

        # |M| = 1（上限）は最後のビンに含める
        recs2 = csv_mod.percentile_profile(np.array([1.0, 1.0, 1.0]),
                                           np.array([1.0, 2.0, 3.0]), xe, min_count=1)
        self.assertEqual(len(recs2), 1)
        self.assertEqual(recs2[0]['count'], 3)

        # min_count 未満のビンは出力しない
        self.assertEqual(len(csv_mod.percentile_profile(x, v, xe, min_count=2)), 1)

    def test_percentile_data_and_table(self):
        """|M| に折り畳まれ、全体分位点とビン別分位点が整合する。"""
        df_long = csv_mod.long_points_table(make_synthetic_points(n_per_bead=50))
        beads = [mt_ori.BEAD_LOOKUP['beads1um'], mt_ori.BEAD_LOOKUP['beads3um']]
        data = csv_mod.percentile_data(df_long, beads, percentiles=(80.0, 90.0, 95.0),
                                       x_bins=8, min_count=5)
        self.assertIsNotNone(data)
        self.assertEqual(data['x_var'], 'm_ising')
        self.assertTrue(np.all(data['x'] >= 0.0))       # |M|
        self.assertTrue(np.all(data['v'] >= 0.0))       # |v|
        self.assertAlmostEqual(float(data['x_edges'][0]), 0.0)
        self.assertAlmostEqual(float(data['x_edges'][-1]), 1.0)
        self.assertEqual(data['n_points'], 200)         # 2 条件 x 2 粒子 x 50
        self.assertEqual(data['n_points_in_range'], data['n_points'])
        self.assertGreater(len(data['records']), 0)
        for p, val in data['global_percentiles'].items():
            self.assertAlmostEqual(val, float(np.percentile(np.abs(data['v']), p)))

        df = csv_mod.percentile_table(data, 'all')
        self.assertFalse(df.empty)
        self.assertEqual(len(df), len(data['records']) * 3)       # ビン x 3 分位点
        self.assertTrue((df['bead_name'] == 'all').all())
        self.assertTrue(df['abs_values'].all())
        self.assertTrue(df['x_variable'].eq('m_ising').all())
        self.assertTrue(df['percentile'].isin([80.0, 90.0, 95.0]).all())
        self.assertTrue((df['count'] >= 5).all())
        self.assertTrue((df['x_low'] >= 0.0).all())
        self.assertTrue((df['v_percentile_tilde'] >= 0.0).all())

        # 同じビン・同じ分位点の値が records と一致する
        rec0 = data['records'][0]
        row = df[(df['percentile'] == 80.0)
                 & np.isclose(df['x_center'], rec0['x_center'])].iloc[0]
        self.assertAlmostEqual(float(row['v_percentile_tilde']), rec0['percentiles'][80.0])
        self.assertEqual(int(row['count']), rec0['count'])
        # 分位点は 80 <= 90 <= 95 の順に単調
        wide = df.pivot_table(index='x_center', columns='percentile',
                              values='v_percentile_tilde')
        self.assertTrue((wide[95.0] >= wide[90.0] - 1e-12).all())
        self.assertTrue((wide[90.0] >= wide[80.0] - 1e-12).all())

        # データが無い条件では None / 空テーブル
        self.assertIsNone(csv_mod.percentile_data(df_long, [{'name': 'nosuch'}]))
        self.assertTrue(csv_mod.percentile_table(None).empty)

    def test_percentile_stats_text(self):
        """統計ボックスの文言に N・Spearman・全体分位点・ビン情報と条件ラベルが入る。"""
        df_long = csv_mod.long_points_table(make_synthetic_points(n_per_bead=50))
        beads = [mt_ori.BEAD_LOOKUP['beads1um'], mt_ori.BEAD_LOOKUP['beads3um']]
        data = csv_mod.percentile_data(df_long, beads, x_bins=6, min_count=5)
        txt = csv_mod._percentile_stats_text(data, 'all conditions')
        self.assertIn('all conditions', txt)
        self.assertIn('Spearman', txt)
        self.assertIn('global', txt)
        self.assertIn('bins =', txt)
        self.assertIn('$N$ = 200', txt)

        # 点数が少なくても / 空の辞書でも例外を出さない
        small = csv_mod.percentile_data(df_long, beads[:1], x_bins=2, min_count=0)
        self.assertIsInstance(csv_mod._percentile_stats_text(small), str)
        self.assertIsInstance(csv_mod._percentile_stats_text({}), str)

    def test_percentile_figures_are_written(self):
        """プール / 条件別 / 線形軸 / 基準線なしの図が PNG と SVG を出力する。"""
        df_long = csv_mod.long_points_table(make_synthetic_points(n_per_bead=60))
        beads = [mt_ori.BEAD_LOOKUP['beads1um'], mt_ori.BEAD_LOOKUP['beads3um']]
        kw = dict(percentiles=(80.0, 90.0, 95.0), x_bins=6, min_count=5)
        data = csv_mod.percentile_data(df_long, beads, **kw)
        with tempfile.TemporaryDirectory() as td:
            out = [Path(td)]
            csv_mod.plot_percentiles([('all conditions', data)], out,
                                     title='unit-test', sign_note='unit-test')
            for ext in ('png', 'svg'):
                self.assertTrue((Path(td) / f"{csv_mod.PERCENTILE_TAG}.{ext}").exists())

            csv_mod.plot_percentiles([('all conditions', data)], out,
                                     basename=f"{csv_mod.PERCENTILE_TAG}_lin",
                                     yscale='linear', show_global=False)
            self.assertTrue((Path(td) / f"{csv_mod.PERCENTILE_TAG}_lin.png").exists())

            series = [(csv_mod._bead_label(b), csv_mod.percentile_data(df_long, [b], **kw))
                      for b in beads]
            csv_mod.plot_percentiles(series, out,
                                     basename=f"{csv_mod.PERCENTILE_TAG}_per_condition",
                                     global_ref=data)
            self.assertTrue(
                (Path(td) / f"{csv_mod.PERCENTILE_TAG}_per_condition.png").exists())

            # データが無ければ何も描かずに戻る（例外を出さない）
            csv_mod.plot_percentiles([('none', None)], out)

            # 分位点やビン境界を変えても例外を出さない
            data_q = csv_mod.percentile_data(df_long, beads, percentiles=(50.0, 99.0),
                                             x_bins=4, x_edges_mode='quantile',
                                             min_count=5)
            csv_mod.plot_percentiles([('all conditions', data_q)], out,
                                     basename=f"{csv_mod.PERCENTILE_TAG}_q")
            self.assertTrue((Path(td) / f"{csv_mod.PERCENTILE_TAG}_q.png").exists())
        plt.close('all')


if __name__ == '__main__':
    unittest.main()
