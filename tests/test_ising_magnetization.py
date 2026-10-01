"""
tests/test_ising_magnetization.py

plot_ising_magnetization.py / libs/ising_magnetization.py の単体テスト。

- イジングスピン sigma = sign(u . n) の定義（符号・閾値・マスク・n -> -n 不変性）
- ブロック磁化 M_Ising(R)（一様場 / 反平行ドメイン / 無秩序場 / 無効ブロック）
- スケーリング解析（局所指数・べき則フィット）
- 窓サイズ指定の解決（auto / 明示指定 / 丸め）
- 合成 GFP_flows.h5（+ MTs_im_theta.zarr）を用いた process_experiment_ising / main() の
  エンドツーエンド動作（CSV / 図の生成）
"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import h5py
import pandas as pd

current_dir = Path(__file__).resolve().parent.parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from libs import ising_magnetization as ising
import plot_ising_magnetization as pim


def make_synthetic_flow(theta_a, theta_b, n_frames=6, rows=48, cols=48):
    """上半分を theta_a、下半分を theta_b の一様流にした合成フロー (T, 2, y, x)。"""
    mx = np.zeros((rows, cols), dtype=np.float32)
    my = np.zeros((rows, cols), dtype=np.float32)
    mx[: rows // 2, :] = np.cos(theta_a)
    my[: rows // 2, :] = np.sin(theta_a)
    mx[rows // 2:, :] = np.cos(theta_b)
    my[rows // 2:, :] = np.sin(theta_b)
    flows = np.zeros((n_frames, 2, rows, cols), dtype=np.float32)
    flows[:, 0] = mx
    flows[:, 1] = my
    return flows


def make_random_domain_flow(theta0, n_frames=3, rows=32, cols=32, block=4, seed=0):
    """
    4 px 程度のランダムな ± ドメイン（flow: theta0 または theta0 + pi）からなる合成フロー。
    sigma が空間的にランダムな場になり <|M(R)|> が R とともに減衰する。
    """
    rng = np.random.default_rng(seed)
    by = int(np.ceil(rows / block))
    bx = int(np.ceil(cols / block))
    sign_map = rng.choice([1.0, -1.0], size=(by, bx))
    sign_full = np.repeat(np.repeat(sign_map, block, axis=0), block, axis=1)[:rows, :cols]
    angles = np.where(sign_full > 0, theta0, theta0 + np.pi)
    mx = np.cos(angles).astype(np.float32)
    my = np.sin(angles).astype(np.float32)
    flows = np.zeros((n_frames, 2, rows, cols), dtype=np.float32)
    flows[:, 0] = mx
    flows[:, 1] = my
    return flows


def write_synthetic_experiment(exp_dir: Path, flows: np.ndarray, theta_nem: float = None):
    """GFP_flows.h5（と任意で MTs_im_theta.zarr）を書き出す。実データ同様 float16 保存。"""
    exp_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(exp_dir / pim.FLOW_NAME), 'w') as f:
        f.create_dataset('flows', data=np.asarray(flows, dtype=np.float16))

    if theta_nem is not None:
        import zarr
        z = zarr.open_array(str(exp_dir / "MTs_im_theta.zarr"), mode='w',
                            shape=(flows.shape[0], 8, 8), dtype='float64')
        z[:] = float(theta_nem)
    return exp_dir


def uniform_valid(shape):
    return np.ones(shape, dtype=bool)


class TestIntegralImageAndWindows(unittest.TestCase):

    def test_integral_image_matches_cumsum(self):
        rng = np.random.default_rng(0)
        a = rng.normal(size=(7, 5))
        c = ising.integral_image(a)
        self.assertEqual(c.shape, (8, 6))
        for i in range(1, 8):
            for j in range(1, 6):
                self.assertAlmostEqual(c[i, j], float(a[:i, :j].sum()), places=10)

    def test_default_window_sizes(self):
        ws = ising.default_window_sizes(100, 1, 12)
        self.assertEqual(ws[0], 1)
        self.assertEqual(ws[-1], 100)
        self.assertTrue(np.all(np.diff(ws) > 0))
        self.assertTrue(np.all(ws <= 100))
        self.assertEqual(ising.default_window_sizes(4, 4, 10).tolist(), [4])
        self.assertEqual(ising.default_window_sizes(3, 8, 10).size, 0)


class TestSpinField(unittest.TestCase):

    def test_parallel_and_antiparallel(self):
        rows, cols = 8, 8
        theta0 = 0.7
        mx = np.full((rows, cols), np.cos(theta0))
        my = np.full((rows, cols), np.sin(theta0))
        sigma, valid = ising.ising_spin_field(mx, my, np.cos(theta0), np.sin(theta0))
        self.assertTrue(np.all(sigma == 1.0))
        self.assertTrue(np.all(valid))

        sigma2, _ = ising.ising_spin_field(-mx, -my, np.cos(theta0), np.sin(theta0))
        self.assertTrue(np.all(sigma2 == -1.0))

    def test_perpendicular_gives_zero_spin(self):
        mx = np.ones((4, 4))
        my = np.zeros((4, 4))
        sigma, valid = ising.ising_spin_field(mx, my, 0.0, 1.0)  # n = y 方向
        self.assertTrue(np.all(sigma == 0.0))
        self.assertTrue(np.all(valid))  # |u| > 0 なので画素自体は有効

    def test_n_sign_flip_keeps_abs_magnetization(self):
        rng = np.random.default_rng(1)
        mx = rng.normal(size=(32, 32))
        my = rng.normal(size=(32, 32))
        theta = 0.3
        s_plus, _ = ising.ising_spin_field(mx, my, np.cos(theta), np.sin(theta))
        s_flip, _ = ising.ising_spin_field(mx, my, -np.cos(theta), -np.sin(theta))
        np.testing.assert_allclose(s_plus, -s_flip)
        v = uniform_valid(mx.shape)
        self.assertAlmostEqual(
            ising.block_magnetization_stats(s_plus, v, 8)['abs_mean'],
            ising.block_magnetization_stats(s_flip, v, 8)['abs_mean'], places=12)

    def test_min_flow_mag_and_exclude_mask(self):
        mx = np.ones((4, 4))
        my = np.zeros((4, 4))
        mx[0, 0] = 0.0                 # |u| = 0 -> 無効
        sigma, valid = ising.ising_spin_field(mx, my, 1.0, 0.0, min_flow_mag=1e-6)
        self.assertFalse(valid[0, 0])
        self.assertEqual(sigma[0, 0], 0.0)
        self.assertEqual(int(valid.sum()), 15)

        excl = np.zeros((4, 4), dtype=bool)
        excl[1, 1] = True
        _, valid2 = ising.ising_spin_field(mx, my, 1.0, 0.0, exclude_mask=excl)
        self.assertFalse(valid2[1, 1])
        self.assertEqual(int(valid2.sum()), 14)

    def test_spin_fractions(self):
        sigma = np.array([[1.0, 1.0], [-1.0, 0.0]])
        valid = np.ones((2, 2), dtype=bool)
        st = ising.spin_fractions(sigma, valid)
        self.assertAlmostEqual(st['frac_plus'], 2.0 / 3.0)
        self.assertAlmostEqual(st['frac_minus'], 1.0 / 3.0)
        self.assertAlmostEqual(st['polar_bias'], 1.0 / 3.0)
        self.assertEqual(st['n_valid'], 4.0)


class TestBlockMagnetization(unittest.TestCase):

    def test_uniform_field_gives_one(self):
        sigma = np.ones((32, 32))
        valid = uniform_valid(sigma.shape)
        for w in (1, 2, 4, 8, 16, 32):
            st = ising.block_magnetization_stats(sigma, valid, w)
            self.assertAlmostEqual(st['abs_mean'], 1.0, places=12)
            self.assertAlmostEqual(st['signed_mean'], 1.0, places=12)
            self.assertGreater(st['n_blocks'], 0)

    def test_antiparallel_half_field(self):
        sigma = np.ones((64, 64))
        sigma[32:, :] = -1.0
        valid = uniform_valid(sigma.shape)
        for w in (1, 8, 16, 32):
            self.assertAlmostEqual(ising.block_magnetization_stats(sigma, valid, w)['abs_mean'],
                                   1.0, places=12)
        st = ising.block_magnetization_stats(sigma, valid, 64)
        self.assertEqual(st['n_blocks'], 1)
        self.assertAlmostEqual(st['abs_mean'], 0.0, places=12)
        self.assertAlmostEqual(st['signed_mean'], 0.0, places=12)

    def test_uncorrelated_field_decays_like_r_minus_one(self):
        rng = np.random.default_rng(7)
        sigma = rng.choice([-1.0, 1.0], size=(256, 256))
        valid = uniform_valid(sigma.shape)
        means = [ising.block_magnetization_stats(sigma, valid, w)['abs_mean']
                 for w in (2, 4, 8, 16)]
        self.assertTrue(np.all(np.diff(means) < 0))
        # 空間無相関なら <|M(R)|> ~ R^{-1}: 窓を 2 倍にすると約 1/2（有限サイズで緩やか）
        for i in range(len(means) - 1):
            ratio = means[i + 1] / means[i]
            self.assertLess(ratio, 0.75)
            self.assertGreater(ratio, 0.2)

    def test_window_larger_than_image_is_empty(self):
        sigma = np.ones((8, 8))
        valid = uniform_valid(sigma.shape)
        st = ising.block_magnetization_stats(sigma, valid, 16)
        self.assertEqual(st['n_blocks'], 0)
        self.assertTrue(np.isnan(st['abs_mean']))
        self.assertEqual(ising.block_magnetizations(sigma, valid, 16).shape, (0, 0))

    def test_invalid_blocks_are_excluded(self):
        sigma = np.ones((8, 8))
        valid = np.zeros((8, 8), dtype=bool)
        valid[:4, :] = True
        m = ising.block_magnetizations(sigma, valid, 4, min_valid_fraction=0.5)
        self.assertEqual(m.shape, (2, 2))
        self.assertTrue(np.all(np.isfinite(m[0])))
        self.assertTrue(np.all(np.isnan(m[1])))
        st = ising.block_magnetization_stats(sigma, valid, 4, min_valid_fraction=0.5)
        self.assertEqual(st['n_blocks'], 2)

    def test_overlap_increases_block_count(self):
        rng = np.random.default_rng(3)
        sigma = rng.choice([-1.0, 1.0], size=(32, 32))
        valid = uniform_valid(sigma.shape)
        n_no = ising.block_magnetization_stats(sigma, valid, 8, step=8)['n_blocks']
        n_ov = ising.block_magnetization_stats(sigma, valid, 8, step=4)['n_blocks']
        self.assertGreater(n_ov, n_no)

    def test_magnetization_curve_matches_single_window(self):
        rng = np.random.default_rng(5)
        sigma = rng.choice([-1.0, 1.0], size=(64, 64))
        valid = uniform_valid(sigma.shape)
        windows = [4, 8, 16]
        curve = ising.magnetization_curve(sigma, valid, windows)
        for i, w in enumerate(windows):
            st = ising.block_magnetization_stats(sigma, valid, w)
            self.assertAlmostEqual(curve['abs_mean'][i], st['abs_mean'], places=12)
            self.assertEqual(int(curve['n_blocks'][i]), st['n_blocks'])

    def test_curve_with_overlap(self):
        rng = np.random.default_rng(11)
        sigma = rng.choice([-1.0, 1.0], size=(32, 32))
        valid = uniform_valid(sigma.shape)
        c0 = ising.magnetization_curve(sigma, valid, [4, 8], overlap=0.0)
        c5 = ising.magnetization_curve(sigma, valid, [4, 8], overlap=0.5)
        self.assertTrue(np.all(c5['n_blocks'] > c0['n_blocks']))


class TestScalingHelpers(unittest.TestCase):

    def test_local_log_slope_constant_for_power_law(self):
        x = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
        y = 3.0 * x ** (-0.25)
        xm, p = ising.local_log_slope(x, y)
        self.assertEqual(xm.size, 4)
        np.testing.assert_allclose(p, 0.25, rtol=1e-10)

    def test_fit_power_law_exponent(self):
        x = np.geomspace(1.0, 100.0, 20)
        y = 2.5 * x ** (-0.125)
        p, r2, amp, n = ising.fit_power_law_exponent(x, y)
        self.assertAlmostEqual(p, 0.125, places=9)
        self.assertAlmostEqual(r2, 1.0, places=9)
        self.assertAlmostEqual(amp, 2.5, places=9)
        self.assertEqual(n, 20)

    def test_fit_power_law_exponent_restricted_range(self):
        x = np.array([1.0, 2.0, 4.0, 8.0])
        y = x ** (-1.0)
        p, _, _, n = ising.fit_power_law_exponent(x, y, x_min=2.0)
        self.assertAlmostEqual(p, 1.0, places=9)
        self.assertEqual(n, 3)

    def test_fit_returns_nan_when_insufficient(self):
        p, r2, amp, n = ising.fit_power_law_exponent([1.0], [1.0])
        self.assertTrue(np.isnan(p))
        self.assertEqual(n, 1)


class TestResolveWindows(unittest.TestCase):

    def test_auto_windows_are_multiples_of_stride(self):
        grid, px, adj = pim.resolve_windows('auto', 8, 16, 12)
        self.assertEqual(adj, [])
        self.assertTrue(np.all(px % 8 == 0))
        self.assertEqual(int(px[0]), 8)
        self.assertEqual(int(px[-1]), 8 * 16)
        self.assertTrue(np.all(grid >= 1))

    def test_explicit_spec(self):
        grid, px, adj = pim.resolve_windows('8:32:8', 8, 32, 12)
        np.testing.assert_array_equal(px, [8, 16, 24, 32])
        np.testing.assert_array_equal(grid, [1, 2, 3, 4])
        self.assertEqual(adj, [])

    def test_snapping_and_dropping(self):
        grid, px, adj = pim.resolve_windows('10,1000', 8, 16, 12)
        np.testing.assert_array_equal(px, [8])
        self.assertEqual(len(adj), 2)
        self.assertEqual(adj[0], (10, 8))
        self.assertEqual(adj[1], (1000, None))

    def test_parse_int_spec(self):
        self.assertEqual(pim.parse_int_spec('2:8:2, 12'), [2, 4, 6, 8, 12])
        self.assertEqual(pim.parse_int_spec('4'), [4])


class TestProcessExperiment(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.bead = pim.BEAD_LOOKUP['beads06um']
        self.windows = np.array([1, 2, 4, 8, 16, 32])
        self.theta0 = 0.4

    def _exp_dir(self, name='syn001'):
        return self.root / 'beads06um' / '20260101' / name

    def test_global_director_from_flow_fallback(self):
        flows = make_synthetic_flow(self.theta0, self.theta0 + np.pi, n_frames=4,
                                    rows=32, cols=32)
        exp_dir = write_synthetic_experiment(self._exp_dir(), flows)

        res = pim.process_experiment_ising(exp_dir, self.bead, self.windows,
                                           pixel_stride=1, frame_stride=1,
                                           progress=False)
        self.assertIsNotNone(res)
        self.assertEqual(res['theta_source'], 'global:flow')
        self.assertEqual(res['n_frames_used'], 4)
        self.assertEqual(res['grid_shape'], (32, 32))

        sign, info = pim.choose_director_sign([res])
        self.assertIn(sign, (1, -1))
        v = pim.select_variant(res, sign)
        # 窓 1 画素では |M| = 1、全視野を覆う窓 32 では平行 / 反平行が打ち消し合って 0
        self.assertAlmostEqual(v['abs_mean'][0], 1.0, places=6)
        self.assertAlmostEqual(v['abs_mean'][-1], 0.0, places=6)
        self.assertLess(v['pooled_abs_mean'][-1], 0.02)
        # 平行 / 反平行が半々 -> 極性バイアスは 0
        self.assertAlmostEqual(v['polar_bias'], 0.0, places=3)
        # ネマチック整合度は 1（フローがディレクター軸に沿っている）
        self.assertGreater(v['nematic_order_cos2'], 0.99)

    def test_global_director_from_zarr(self):
        flows = make_synthetic_flow(self.theta0, self.theta0 + np.pi, n_frames=4,
                                    rows=32, cols=32)
        exp_dir = write_synthetic_experiment(self._exp_dir('syn002'), flows,
                                             theta_nem=self.theta0)
        res = pim.process_experiment_ising(exp_dir, self.bead, self.windows,
                                           pixel_stride=1, frame_stride=1, progress=False)
        self.assertEqual(res['theta_source'], 'global:zarr')
        self.assertAlmostEqual(res['theta_mean_rad'], self.theta0, places=5)
        sign, _ = pim.choose_director_sign([res])
        self.assertEqual(sign, 1)  # zarr の規約はフローと整合

    def test_auto_mirror_correction_recovers_the_same_spins(self):
        flows = make_synthetic_flow(self.theta0, self.theta0 + np.pi, n_frames=3,
                                    rows=32, cols=32)
        ref = pim.process_experiment_ising(
            write_synthetic_experiment(self._exp_dir('syn101'), flows, theta_nem=self.theta0),
            self.bead, self.windows, pixel_stride=1, frame_stride=1, progress=False)
        mir = pim.process_experiment_ising(
            write_synthetic_experiment(self._exp_dir('syn102'), flows, theta_nem=-self.theta0),
            self.bead, self.windows, pixel_stride=1, frame_stride=1, progress=False)

        sign_ref, _ = pim.choose_director_sign([ref])
        sign_mir, _ = pim.choose_director_sign([mir])
        self.assertEqual(sign_ref, 1)
        self.assertEqual(sign_mir, -1)
        v_ref = pim.select_variant(ref, sign_ref)
        v_mir = pim.select_variant(mir, sign_mir)
        np.testing.assert_allclose(v_ref['abs_mean'], v_mir['abs_mean'], atol=1e-6)
        np.testing.assert_allclose(v_ref['signed_mean'], v_mir['signed_mean'], atol=1e-6)

    def test_local_director_mode(self):
        flows = make_synthetic_flow(self.theta0, self.theta0 + np.pi, n_frames=3,
                                    rows=32, cols=32)
        exp_dir = write_synthetic_experiment(self._exp_dir('syn003'), flows,
                                             theta_nem=self.theta0)
        res = pim.process_experiment_ising(exp_dir, self.bead, self.windows,
                                           pixel_stride=1, frame_stride=1,
                                           director='local', progress=False)
        self.assertEqual(res['theta_source'], 'local:zarr')
        sign, _ = pim.choose_director_sign([res])
        v = pim.select_variant(res, sign)
        self.assertAlmostEqual(v['abs_mean'][0], 1.0, places=6)
        self.assertAlmostEqual(v['abs_mean'][-1], 0.0, places=6)

    def test_pixel_stride_controls_grid_and_window_units(self):
        flows = make_synthetic_flow(self.theta0, self.theta0 + np.pi, n_frames=2,
                                    rows=32, cols=32)
        exp_dir = write_synthetic_experiment(self._exp_dir('syn004'), flows)
        res = pim.process_experiment_ising(exp_dir, self.bead, np.array([1, 2, 4, 8]),
                                           pixel_stride=4, frame_stride=1, progress=False)
        self.assertEqual(res['grid_shape'], (8, 8))
        np.testing.assert_array_equal(res['windows_px'], [4, 8, 16, 32])

    def test_missing_flow_returns_none(self):
        empty = self.root / 'beads06um' / '20260101' / 'empty'
        empty.mkdir(parents=True, exist_ok=True)
        res = pim.process_experiment_ising(empty, self.bead, self.windows,
                                           pixel_stride=1, frame_stride=1,
                                           flow_cache='off', progress=False)
        self.assertIsNone(res)

    def test_min_flow_mag_marks_pixels_invalid(self):
        flows = make_synthetic_flow(self.theta0, self.theta0 + np.pi, n_frames=2,
                                    rows=16, cols=16)
        flows[:, :, 0, :] = 0.0   # 1 行目をゼロ流速にする
        exp_dir = write_synthetic_experiment(self._exp_dir('syn005'), flows)
        res = pim.process_experiment_ising(
            exp_dir, self.bead, np.array([1, 2]), pixel_stride=1, frame_stride=1,
            min_flow_mag=1e-4, min_valid_fraction=0.99, progress=False)
        v = pim.select_variant(res, 1)
        # 有効率 99% 未満のブロックは無効化されるため、ゼロ流速行を含む窓 2 のブロックは減る
        self.assertGreater(int(v['n_blocks'][0]), int(v['n_blocks'][1]))


class TestTables(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.beads = [pim.BEAD_LOOKUP['beads06um'], pim.BEAD_LOOKUP['beads1um']]
        self.windows = np.array([1, 2, 4, 8, 16, 32])
        self.results = []
        for bead, theta0 in zip(self.beads, (0.4, -0.3)):
            # 1 画素ごとにランダムな +/- スピン（<|M(R)|> が R^-1 的に減衰する場）
            flows = make_random_domain_flow(theta0, n_frames=3, rows=32, cols=32, block=1)
            base = self.root / bead['name'] / '20260101'
            for k in range(2):
                res = pim.process_experiment_ising(
                    write_synthetic_experiment(base / f'exp{k:03d}', flows, theta_nem=theta0),
                    bead, self.windows, pixel_stride=1, frame_stride=1, progress=False)
                self.results.append(res)

    def test_per_experiment_table(self):
        for sign in (1, -1):
            df = pim.per_experiment_table(self.results, sign)
            self.assertEqual(len(df), 2 * 2 * len(self.windows))
            self.assertEqual(sorted(df['bead_name'].unique()), ['beads06um', 'beads1um'])
            self.assertTrue(np.isfinite(df['abs_mean']).all())
            self.assertTrue((df['n_blocks'] > 0).all())

    def test_condition_curve_table(self):
        sign, _ = pim.choose_director_sign(self.results)
        df = pim.condition_curve_table(self.results, self.beads, sign)
        self.assertEqual(len(df), 2 * len(self.windows))
        self.assertTrue((df['n_experiments'] == 2).all())
        # 1 画素ランダムスピン場では <|M(R)|> が窓サイズとともに単調に減衰する
        for bead in self.beads:
            d = df[df['bead_name'] == bead['name']].sort_values('window_px')
            self.assertAlmostEqual(float(d['abs_mean'].iloc[0]), 1.0, places=6)
            self.assertLess(float(d['abs_mean'].iloc[-1]), 0.2)

    def test_condition_summary_table(self):
        sign, _ = pim.choose_director_sign(self.results)
        df = pim.condition_summary_table(self.results, self.beads, sign)
        self.assertEqual(len(df), 2)
        for col in ('power_law_exponent', 'power_law_r2', 'polar_bias_mean',
                    'frac_plus_mean', 'nematic_order_cos2_mean', 'theta_source'):
            self.assertIn(col, df.columns)
        self.assertTrue((df['n_experiments'] == 2).all())
        self.assertTrue((df['n_frames_used'] == 6).all())
        self.assertTrue(np.isfinite(df['power_law_exponent']).all())
        # 1 画素ランダムスピン場では R^-1（2D 無相関）に近い減衰を示す
        self.assertTrue((df['power_law_exponent'] > 0.5).all())
        self.assertTrue((df['power_law_exponent'] < 1.5).all())

    def test_choose_director_sign_forced(self):
        sign, info = pim.choose_director_sign(self.results, theta_sign='-1')
        self.assertEqual(sign, -1)
        self.assertIn('forced', info['decision'])
        sign2, _ = pim.choose_director_sign(self.results, theta_sign='+1')
        self.assertEqual(sign2, 1)


def make_spread_flow(theta0, spread_rad, n_frames=3, rows=96, cols=96, seed=7):
    """
    各画素の向きが theta0 ± spread_rad に一様分布する合成フロー。

    全画素が同じ符号（スピンはすべて +1）なので |M(R)| = 1 になる一方、向きには角度
    広がりがあるため

        P(R) = | <e^{i delta}> | = | sin(spread) / spread |

    という解析値を持つ（ポーラーオーダーの検証用）。
    """
    rng = np.random.default_rng(seed)
    ang = theta0 + rng.uniform(-spread_rad, spread_rad, size=(rows, cols))
    flows = np.zeros((n_frames, 2, rows, cols), dtype=np.float32)
    flows[:, 0] = np.cos(ang).astype(np.float32)
    flows[:, 1] = np.sin(ang).astype(np.float32)
    return flows


class TestUnitFlowAndPolarBlocks(unittest.TestCase):

    def test_unit_flow_components_are_normalized(self):
        mx = np.array([[3.0, 0.0], [-4.0, 0.5]])
        my = np.array([[4.0, 0.0], [3.0, 0.0]])
        ux, uy = ising.unit_flow_components(mx, my)
        np.testing.assert_allclose(np.hypot(ux, uy)[np.hypot(mx, my) > 0], 1.0, atol=1e-12)
        self.assertEqual(ux[0, 1], 0.0)   # |u| = 0 は無効として 0 ベクトル
        self.assertEqual(uy[0, 1], 0.0)
        np.testing.assert_allclose(ux[0, 0], 0.6, atol=1e-12)
        np.testing.assert_allclose(uy[0, 0], 0.8, atol=1e-12)

    def test_unit_flow_components_valid_mask(self):
        mx = np.ones((2, 2))
        my = np.zeros((2, 2))
        valid = np.array([[True, False], [True, True]])
        ux, uy = ising.unit_flow_components(mx, my, valid)
        self.assertEqual(ux[0, 1], 0.0)
        self.assertEqual(ux[0, 0], 1.0)

    def test_uniform_field_gives_polar_order_one(self):
        theta0 = 0.7
        mx = np.full((32, 32), np.cos(theta0))
        my = np.full((32, 32), np.sin(theta0))
        ux, uy = ising.unit_flow_components(mx, my)
        valid = uniform_valid(mx.shape)
        for w in (1, 2, 8, 32):
            st = ising.block_polar_stats(ux, uy, valid, w)
            self.assertAlmostEqual(st['polar_mean'], 1.0, places=12)
            self.assertGreater(st['n_blocks'], 0)

    def test_random_field_gives_small_polar_order(self):
        rng = np.random.default_rng(0)
        ang = rng.uniform(0.0, 2.0 * np.pi, size=(64, 64))
        ux, uy = ising.unit_flow_components(np.cos(ang), np.sin(ang))
        valid = uniform_valid(ang.shape)
        p1 = ising.block_polar_stats(ux, uy, valid, 1)['polar_mean']
        p32 = ising.block_polar_stats(ux, uy, valid, 32)['polar_mean']
        self.assertAlmostEqual(p1, 1.0, places=12)          # 1 画素では必ず 1
        self.assertLess(p32, 0.35)                          # ランダム -> ほぼ 0

    def test_block_polar_orders_matches_bruteforce(self):
        rng = np.random.default_rng(2)
        rows, cols = 13, 17
        valid = rng.random((rows, cols)) > 0.2
        ang = rng.uniform(0.0, 2.0 * np.pi, size=(rows, cols))
        ux, uy = ising.unit_flow_components(np.cos(ang), np.sin(ang), valid)
        for w in (1, 2, 3, 5):
            p = ising.block_polar_orders(ux, uy, valid, w)
            ref = []
            for y in range(0, rows - w + 1, w):
                for x in range(0, cols - w + 1, w):
                    sl = (slice(y, y + w), slice(x, x + w))
                    v = valid[sl]
                    n = int(v.sum())
                    if n <= 0 or n < 0.5 * w * w:
                        continue
                    ref.append(np.hypot(ux[sl][v].sum(), uy[sl][v].sum()) / n)
            np.testing.assert_allclose(p[np.isfinite(p)].ravel(), ref, atol=1e-12)

    def test_block_polar_orders_uses_same_blocks_as_magnetization(self):
        """有効画素の欠落があっても、P と |M| のブロック集合は完全に一致する。"""
        rng = np.random.default_rng(3)
        rows, cols = 21, 19
        valid = rng.random((rows, cols)) > 0.15
        sigma = np.where(valid, rng.choice([-1.0, 1.0], size=(rows, cols)), 0.0)
        ang = rng.uniform(0.0, 2.0 * np.pi, size=(rows, cols))
        ux, uy = ising.unit_flow_components(np.cos(ang), np.sin(ang), valid)
        for w in (1, 2, 4, 7):
            for minvf in (0.0, 0.5, 0.9):
                m = ising.block_magnetizations(sigma, valid, w, min_valid_fraction=minvf)
                p = ising.block_polar_orders(ux, uy, valid, w, min_valid_fraction=minvf)
                self.assertEqual(m.shape, p.shape)
                np.testing.assert_array_equal(np.isfinite(m), np.isfinite(p))


    def test_two_state_field_polar_equals_abs_magnetization(self):
        """±反平行 2 状態場では P_b = |M_b| が厳密に成り立つ（Delta = 0 の根拠）。"""
        rng = np.random.default_rng(4)
        rows, cols = 32, 32
        theta0 = 0.37
        sign_map = rng.choice([-1.0, 1.0], size=(rows, cols))
        mx = np.cos(theta0) * sign_map
        my = np.sin(theta0) * sign_map
        ux, uy = ising.unit_flow_components(mx, my)
        sigma, valid = ising.ising_spin_field(mx, my, np.cos(theta0), np.sin(theta0))
        for w in (1, 2, 4, 8):
            pr = ising.block_order_pairs(sigma, ux, uy, valid, w, min_valid_fraction=1.0)
            np.testing.assert_allclose(pr['polar'], pr['abs_m'], atol=1e-12)

    def test_spread_field_matches_sinc_prediction(self):
        """同符号で角度広がり s の場では P(R) = |sin s / s| という解析値になる。"""
        spread = 1.2
        flows = make_spread_flow(0.4, spread, n_frames=1, rows=96, cols=96)
        mx, my = flows[0, 0], flows[0, 1]
        valid = uniform_valid(mx.shape)
        ux, uy = ising.unit_flow_components(mx, my)
        p = ising.block_polar_stats(ux, uy, valid, 96, min_valid_fraction=1.0)
        analytic = abs(np.sin(spread) / spread)
        self.assertAlmostEqual(p['polar_mean'], analytic, delta=0.02)
        # 同符号なので |M| = 1 -> Delta = |P - 1| / P = (1 - P) / P
        sigma, _ = ising.ising_spin_field(mx, my, np.cos(0.4), np.sin(0.4))
        m = ising.block_magnetization_stats(sigma, valid, 96, min_valid_fraction=1.0)
        self.assertAlmostEqual(m['abs_mean'], 1.0, places=6)
        self.assertAlmostEqual(float(ising.relative_gap(p['polar_mean'], m['abs_mean'])),
                               (1.0 - analytic) / analytic, delta=0.03)

    def test_polar_order_curve_windows_and_counts(self):
        theta0 = -0.2
        mx = np.full((32, 32), np.cos(theta0))
        my = np.full((32, 32), np.sin(theta0))
        ux, uy = ising.unit_flow_components(mx, my)
        valid = uniform_valid(mx.shape)
        curve = ising.polar_order_curve(ux, uy, valid, [1, 2, 4, 8, 16])
        self.assertEqual(curve['window'].tolist(), [1, 2, 4, 8, 16])
        np.testing.assert_allclose(curve['polar_mean'], 1.0, atol=1e-12)
        # 非重複タイルなのでブロック数は (32/w)^2
        np.testing.assert_array_equal(curve['n_blocks'], [32 * 32, 16 * 16, 8 * 8, 4 * 4, 2 * 2])
        # 窓が画像より大きい場合はブロック 0
        self.assertEqual(ising.block_polar_stats(ux, uy, valid, 64)['n_blocks'], 0)

    def test_polar_curve_matches_magnetization_curve_for_two_state_field(self):
        theta0 = 0.6
        flows = make_random_domain_flow(theta0, n_frames=1, rows=32, cols=32, block=4)
        mx, my = flows[0, 0], flows[0, 1]
        valid = uniform_valid(mx.shape)
        ux, uy = ising.unit_flow_components(mx, my)
        sigma, _ = ising.ising_spin_field(mx, my, np.cos(theta0), np.sin(theta0))
        pc = ising.polar_order_curve(ux, uy, valid, [1, 2, 4, 8, 16], min_valid_fraction=1.0)
        mc = ising.magnetization_curve(sigma, valid, [1, 2, 4, 8, 16], min_valid_fraction=1.0)
        np.testing.assert_allclose(pc['polar_mean'], mc['abs_mean'], atol=1e-12)
        np.testing.assert_array_equal(pc['n_blocks'], mc['n_blocks'])


class TestBlockOrderPairs(unittest.TestCase):

    def test_pairs_report_expected_values(self):
        rows, cols = 16, 16
        sigma = np.ones((rows, cols))
        sigma[:, 8:] = -1.0                     # 半分が反平行
        ang = np.full((rows, cols), 0.3)
        ang[:, 8:] = 0.3 + np.pi
        ux, uy = ising.unit_flow_components(np.cos(ang), np.sin(ang))
        valid = uniform_valid((rows, cols))

        pair = ising.block_order_pairs(sigma, ux, uy, valid, 16, min_valid_fraction=1.0)
        self.assertEqual(pair['n_blocks'], 1)
        self.assertAlmostEqual(float(pair['signed_m'][0, 0]), 0.0, places=12)
        self.assertAlmostEqual(float(pair['abs_m'][0, 0]), 0.0, places=12)
        self.assertAlmostEqual(float(pair['polar'][0, 0]), 0.0, places=12)

        pair2 = ising.block_order_pairs(sigma, ux, uy, valid, 8, min_valid_fraction=1.0)
        self.assertEqual(pair2['n_blocks'], 4)
        np.testing.assert_allclose(pair2['abs_m'][0, 0], 1.0, atol=1e-12)
        np.testing.assert_allclose(pair2['polar'][0, 0], 1.0, atol=1e-12)
        np.testing.assert_allclose(pair2['signed_m'][0, 0], 1.0, atol=1e-12)

    def test_pairs_invalid_blocks_are_nan(self):
        rows, cols = 8, 8
        sigma = np.ones((rows, cols))
        valid = np.zeros((rows, cols), dtype=bool)
        valid[:4, :] = True                      # 下半分は完全に無効
        ang = np.zeros((rows, cols))
        ux, uy = ising.unit_flow_components(np.cos(ang), np.sin(ang), valid)
        pair = ising.block_order_pairs(sigma, ux, uy, valid, 4, min_valid_fraction=0.5)
        self.assertEqual(pair['n_blocks'], 2)    # 上段 2 ブロックのみ有効
        self.assertTrue(np.isfinite(pair['polar'][0, :]).all())
        self.assertTrue(np.isnan(pair['polar'][1, :]).all())

    def test_window_too_large_returns_empty(self):
        ux, uy = ising.unit_flow_components(np.ones((4, 4)), np.zeros((4, 4)))
        valid = uniform_valid((4, 4))
        pair = ising.block_order_pairs(np.ones((4, 4)), ux, uy, valid, 8)
        self.assertEqual(pair['n_blocks'], 0)
        self.assertEqual(pair['polar'].size, 0)
        self.assertEqual(ising.block_polar_orders(ux, uy, valid, 8).size, 0)

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            ising.block_order_pairs(np.ones((4, 4)), np.ones((4, 4)), np.ones((4, 3)),
                                    np.ones((4, 4), dtype=bool), 2)
        with self.assertRaises(ValueError):
            ising.block_polar_orders(np.ones((4, 4)), np.ones((4, 4)),
                                     np.ones((4, 5), dtype=bool), 2)
        with self.assertRaises(ValueError):
            ising.unit_flow_components(np.ones((4, 4)), np.ones((3, 4)))


class TestComparisonStats(unittest.TestCase):

    def test_relative_gap_values(self):
        p = np.array([1.0, 0.5, 0.25, 0.8, 0.0, np.nan])
        m = np.array([1.0, 0.25, 0.25, 0.4, 0.2, 0.1])
        np.testing.assert_allclose(ising.relative_gap(p, m)[:4],
                                   [0.0, 0.5, 0.0, 0.5], atol=1e-12)
        self.assertTrue(np.isnan(ising.relative_gap(p, m)[4]))   # P = 0 は未定義
        self.assertTrue(np.isnan(ising.relative_gap(p, m)[5]))   # 非有限も NaN
        # スカラーでも使える（実験レベルの Delta 計算）
        self.assertAlmostEqual(float(ising.relative_gap(0.8, 0.4)), 0.5, places=12)

    def test_even_stride_indices(self):
        self.assertEqual(ising.even_stride_indices(10, 10).tolist(), list(range(10)))
        self.assertEqual(ising.even_stride_indices(10, 0).tolist(), list(range(10)))
        self.assertEqual(ising.even_stride_indices(10, 100).tolist(), list(range(10)))
        idx = ising.even_stride_indices(10, 3)
        self.assertEqual(idx.size, 3)
        self.assertEqual(idx[0], 0)
        self.assertEqual(idx[-1], 9)
        self.assertTrue(np.all(np.diff(idx) > 0))
        self.assertEqual(ising.even_stride_indices(0, 5).size, 0)

    def test_pearson_and_spearman_correlation(self):
        x = np.linspace(0.0, 1.0, 20)
        self.assertAlmostEqual(ising.pearson_correlation(x, 2.0 * x), 1.0, places=12)
        self.assertAlmostEqual(ising.spearman_rank_correlation(x, 2.0 * x), 1.0, places=12)
        self.assertAlmostEqual(ising.pearson_correlation(x, -x), -1.0, places=12)
        # 非線形でも単調なら Spearman は 1（Pearson は 1 未満）
        y = x ** 3
        self.assertAlmostEqual(ising.spearman_rank_correlation(x, y), 1.0, places=12)
        self.assertLess(ising.pearson_correlation(x, y), 0.99)
        # 分散 0 / サンプル不足は NaN
        self.assertTrue(np.isnan(ising.pearson_correlation(np.ones(5), np.arange(5.0))))
        self.assertTrue(np.isnan(ising.spearman_rank_correlation(np.array([1.0]),
                                                                 np.array([2.0]))))

    def test_average_ranks_handles_ties(self):
        np.testing.assert_allclose(ising._average_ranks(np.array([1.0, 1.0, 3.0, 5.0, 5.0])),
                                   [1.5, 1.5, 3.0, 4.5, 4.5])

    def test_slope_through_origin(self):
        x = np.linspace(0.05, 1.0, 50)
        a, r2 = ising.slope_through_origin(x, 0.75 * x)
        self.assertAlmostEqual(a, 0.75, places=12)
        self.assertAlmostEqual(r2, 1.0, places=12)
        # 切片があると原点通過フィットは傾きを過大評価する
        a2, r22 = ising.slope_through_origin(x, 0.75 * x + 0.1)
        self.assertGreater(a2, 0.75)
        self.assertLess(r22, 1.0)

    def test_paired_correlation_stats_keys_and_values(self):
        rng = np.random.default_rng(5)
        x = rng.random(400)
        y = 0.8 * x + rng.normal(scale=0.01, size=400)
        st = ising.paired_correlation_stats(x, y)
        for key in ('n', 'pearson_r', 'spearman_rho', 'slope_origin', 'r2_origin',
                    'slope_ols', 'intercept_ols', 'r2_ols', 'mean_x', 'mean_y'):
            self.assertIn(key, st)
        self.assertEqual(st['n'], 400)
        self.assertGreater(st['pearson_r'], 0.99)
        self.assertGreater(st['spearman_rho'], 0.99)
        self.assertAlmostEqual(st['slope_origin'], 0.8, delta=0.03)
        self.assertAlmostEqual(st['mean_x'], float(np.mean(x)), places=12)
        # 全 NaN -> n = 0、統計量は NaN
        st2 = ising.paired_correlation_stats(np.array([np.nan, np.nan]), np.array([1.0, 2.0]))
        self.assertEqual(st2['n'], 0)
        self.assertTrue(np.isnan(st2['pearson_r']))

    def test_binned_median(self):
        x = np.linspace(0.0, 1.0, 100)
        centers, medians, counts = ising.binned_median(x, 3.0 * x, n_bins=4)
        self.assertEqual(centers.size, 4)
        self.assertEqual(int(counts.sum()), 100)
        np.testing.assert_allclose(medians, 3.0 * centers, atol=0.05)
        # ビン内サンプルが min_count 未満なら NaN
        _, med2, cnt2 = ising.binned_median(x, 3.0 * x, n_bins=200, min_count=5)
        self.assertTrue(np.isnan(med2[cnt2 < 5]).all())


class TestPolarComparisonTables(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.beads = [pim.BEAD_LOOKUP['beads06um'], pim.BEAD_LOOKUP['beads1um']]
        self.windows = np.array([1, 2, 4, 8, 16])
        self.results = []
        for bead, theta0 in zip(self.beads, (0.4, -0.3)):
            # ±反平行 2 状態場 -> ブロックごとに P = |M|、Delta = 0
            flows = make_random_domain_flow(theta0, n_frames=3, rows=32, cols=32, block=2)
            base = self.root / bead['name'] / '20260101'
            for k in range(2):
                self.results.append(pim.process_experiment_ising(
                    write_synthetic_experiment(base / f'exp{k:03d}', flows, theta_nem=theta0),
                    bead, self.windows, pixel_stride=1, frame_stride=1, progress=False,
                    block_sample_max=8))

    def test_block_sample_max_limits_samples(self):
        df = pim.polar_block_table(self.results, self.beads, 1)
        self.assertGreater(len(df), 0)
        # 窓サイズ 1 では 32 x 32 = 1024 ブロックあるが、8 個に間引かれている
        d1 = df[df['window_px'] == 1]
        self.assertEqual(int(d1['n_blocks_grid'].iloc[0]), 32 * 32)
        self.assertEqual(int(d1['n_blocks_sampled'].iloc[0]), 8)
        self.assertEqual(len(d1.groupby('exp_dir')), 4)
        self.assertEqual(sorted(df.groupby('bead_name').size().index.tolist()),
                         ['beads06um', 'beads1um'])

    def test_polar_block_table_pairs_are_consistent(self):
        df = pim.polar_block_table(self.results, self.beads, 1)
        for col in ('bead_name', 'exp_dir', 'window_px', 'window_um', 'block_row',
                    'block_col', 'n_frames', 'polar', 'abs_m', 'signed_m',
                    'n_blocks_grid', 'n_blocks_sampled'):
            self.assertIn(col, df.columns)
        self.assertTrue((df['n_frames'] == 3).all())
        self.assertTrue(np.isfinite(df['polar']).all())
        # ±2 状態場では P = |M|（散布図が y = x に乗る）
        np.testing.assert_allclose(df['polar'], df['abs_m'], atol=1e-6)
        self.assertTrue((df['block_row'] >= 0).all())
        self.assertTrue((df['block_col'] >= 0).all())

    def test_polar_correlation_table_is_perfect(self):
        df = pim.polar_block_table(self.results, self.beads, 1)
        df_corr = pim.polar_correlation_table(df, self.beads)
        self.assertEqual(len(df_corr), 2 * len(self.windows))
        self.assertTrue((df_corr['n_points'] > 0).all())
        # 窓 1-2 画素ではブロックが 1 ドメインに収まり P = |M| = 1（分散 0）-> 相関 NaN
        nan_rows = df_corr[df_corr['pearson_r'].isna()]
        self.assertEqual(len(nan_rows), 2 * 2)
        self.assertTrue((nan_rows['window_px'] <= 2).all())
        fin = df_corr[df_corr['pearson_r'].notna()]
        self.assertGreater(len(fin), 0)
        np.testing.assert_allclose(fin['pearson_r'], 1.0, atol=1e-9)
        np.testing.assert_allclose(fin['slope_origin'], 1.0, atol=1e-6)
        np.testing.assert_allclose(fin['r2_origin'], 1.0, atol=1e-9)
        # Spearman は論理的に同値な値（float 誤差で ~1e-8 だけ割れる）の順位付けが
        # 安定しないため僅かに 1 を下回る。対して Pearson は厳密に 1 になる。
        self.assertTrue((fin['spearman_rho'] > 0.9).all())

    def test_polar_curve_table_delta_is_zero_for_two_state_field(self):
        df = pim.polar_curve_table(self.results, self.beads, 1)
        self.assertEqual(len(df), 2 * len(self.windows))
        for col in ('polar_mean', 'polar_sem', 'pooled_polar', 'abs_mean', 'abs_sem',
                    'delta_relative_mean', 'delta_relative_pooled', 'ratio_abs_to_polar_pooled'):
            self.assertIn(col, df.columns)
        np.testing.assert_allclose(df['polar_mean'], df['abs_mean'], atol=1e-6)
        np.testing.assert_allclose(df['delta_relative_pooled'], 0.0, atol=1e-3)
        np.testing.assert_allclose(df['ratio_abs_to_polar_pooled'], 1.0, atol=1e-3)
        # 窓サイズとともに P = |M| が減衰する（ランダム ± ドメイン）
        for bead in self.beads:
            d = df[df['bead_name'] == bead['name']].sort_values('window_um')
            self.assertAlmostEqual(float(d['polar_mean'].iloc[0]), 1.0, places=6)
            self.assertLess(float(d['polar_mean'].iloc[-1]), 0.5)

    def test_condition_summary_has_polar_comparison_columns(self):
        df_blocks = pim.polar_block_table(self.results, self.beads, 1)
        sign, _ = pim.choose_director_sign(self.results)
        df = pim.condition_summary_table(self.results, self.beads, sign, df_blocks=df_blocks)
        for col in ('polar_mean_at_min_R', 'polar_mean_at_max_R', 'delta_relative_at_min_R',
                    'delta_relative_at_max_R', 'delta_relative_pooled',
                    'ratio_abs_to_polar_pooled', 'paired_n_points', 'paired_pearson_r',
                    'paired_spearman_rho', 'paired_slope_origin', 'paired_r2_origin'):
            self.assertIn(col, df.columns)
        np.testing.assert_allclose(df['delta_relative_pooled'], 0.0, atol=1e-3)
        np.testing.assert_allclose(df['paired_pearson_r'], 1.0, atol=1e-3)
        np.testing.assert_allclose(df['paired_slope_origin'], 1.0, atol=1e-3)
        self.assertTrue((df['paired_n_points'] > 0).all())
        # df_blocks を渡さない場合は比較列が NaN / 0 になる（後方互換）
        df_old = pim.condition_summary_table(self.results, self.beads, sign)
        self.assertTrue(np.isnan(df_old['paired_pearson_r']).all())
        self.assertTrue((df_old['paired_n_points'] == 0).all())

    def test_spread_field_delta_equals_one_minus_sinc(self):
        """同符号・角度広がり s の場では Delta = (1 - |sin s / s|) / (|sin s / s|)。"""
        spread = 1.0
        flows = make_spread_flow(0.3, spread, n_frames=2, rows=64, cols=64)
        bead = self.beads[0]
        res = pim.process_experiment_ising(
            write_synthetic_experiment(self.root / 'spread' / 'exp000', flows),
            bead, np.array([64]), pixel_stride=1, frame_stride=1, progress=False)
        df = pim.polar_curve_table([res], [bead], 1)
        analytic = abs(np.sin(spread) / spread)
        self.assertAlmostEqual(float(df['polar_mean'].iloc[0]), analytic, delta=0.02)
        self.assertAlmostEqual(float(df['abs_mean'].iloc[0]), 1.0, delta=0.01)
        self.assertAlmostEqual(float(df['delta_relative_pooled'].iloc[0]),
                               (1.0 - analytic) / analytic, delta=0.03)
        # |M| = 1 > P なので比 |M| / P は 1 を超える（= 符号秩序が向きの秩序を過大評価）
        self.assertGreater(float(df['ratio_abs_to_polar_pooled'].iloc[0]), 1.0)


class TestMainEndToEnd(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'root'
        self.out_dir = Path(self.tmp.name) / 'out'

    def _build_synthetic_root(self):
        for bead_name, theta0 in [('beads06um', 0.4), ('beads1um', -0.3)]:
            # 1 画素ごとにランダムな +/- スピン（<|M(R)|> が減衰する場）
            flows = make_random_domain_flow(theta0, n_frames=4, rows=32, cols=32, block=1)
            for k in range(2):
                write_synthetic_experiment(
                    self.root / bead_name / '20260101' / f'exp{k:03d}', flows,
                    theta_nem=theta0)
        return self.root

    def test_main_generates_csv_and_figures(self):
        self._build_synthetic_root()
        argv = ['plot_ising_magnetization.py',
                '--root_dir', str(self.root),
                '--beads', 'beads06um', 'beads1um',
                '--output_dir', str(self.out_dir),
                '--pixel_stride', '2', '--frame_stride', '1',
                '--window_sizes', '2:16:2',
                '--no_progress', '--no_save_root']
        with mock.patch.object(sys, 'argv', argv):
            pim.main()
        plt.close('all')

        for csv_name in ['ising_magnetization_per_experiment',
                         'ising_magnetization_curve',
                         'ising_magnetization_summary']:
            path = self.out_dir / f"{csv_name}.csv"
            self.assertTrue(path.exists(), msg=f"{csv_name}.csv missing")
            self.assertGreater(len(pd.read_csv(path)), 0)

        for fig_name in ['ising_magnetization_vs_window',
                         'ising_magnetization_vs_window_linear',
                         'ising_magnetization_per_experiment',
                         'ising_polar_bias',
                         'ising_spin_map_examples']:
            for ext in ('png', 'svg'):
                self.assertTrue((self.out_dir / f"{fig_name}.{ext}").exists(),
                                msg=f"{fig_name}.{ext} missing")

    def test_main_csv_values_are_consistent(self):
        self._build_synthetic_root()
        argv = ['plot_ising_magnetization.py',
                '--root_dir', str(self.root),
                '--beads', 'beads06um', 'beads1um',
                '--output_dir', str(self.out_dir),
                '--pixel_stride', '2', '--frame_stride', '1',
                '--window_sizes', '2:16:2',
                '--no_progress', '--no_save_root']
        with mock.patch.object(sys, 'argv', argv):
            pim.main()
        plt.close('all')

        df_curve = pd.read_csv(self.out_dir / 'ising_magnetization_curve.csv')
        self.assertEqual(sorted(df_curve['bead_name'].unique()), ['beads06um', 'beads1um'])
        self.assertTrue((df_curve['n_experiments'] == 2).all())
        for bead in ['beads06um', 'beads1um']:
            d = df_curve[df_curve['bead_name'] == bead].sort_values('window_um')
            self.assertAlmostEqual(float(d['abs_mean'].iloc[0]), 1.0, places=5)
            self.assertLess(float(d['abs_mean'].iloc[-1]), 0.4)

        df_sum = pd.read_csv(self.out_dir / 'ising_magnetization_summary.csv')
        self.assertEqual(sorted(df_sum['bead_name']), ['beads06um', 'beads1um'])
        self.assertTrue(np.isfinite(df_sum['power_law_exponent']).all())
        self.assertTrue((df_sum['power_law_exponent'] > 0.3).all())
        self.assertTrue((df_sum['theta_sign'] == 1).all())
        # 平行 / 反平行が同数 -> 極性バイアスはほぼ 0、+1 スピン比は 0.5
        np.testing.assert_allclose(df_sum['polar_bias_mean'].abs(), 0.0, atol=0.05)
        np.testing.assert_allclose(df_sum['frac_plus_mean'], 0.5, atol=0.05)

        df_exp = pd.read_csv(self.out_dir / 'ising_magnetization_per_experiment.csv')
        self.assertEqual(len(df_exp), 2 * 2 * 8)  # 条件 2 x 実験 2 x 窓 8
        self.assertEqual(sorted(df_exp['theta_source'].unique()), ['global:zarr'])


    def test_main_generates_polar_comparison_outputs(self):
        self._build_synthetic_root()
        argv = ['plot_ising_magnetization.py',
                '--root_dir', str(self.root),
                '--beads', 'beads06um', 'beads1um',
                '--output_dir', str(self.out_dir),
                '--pixel_stride', '2', '--frame_stride', '1',
                '--window_sizes', '2:16:2',
                '--block_sample_max', '16',
                '--no_progress', '--no_save_root']
        with mock.patch.object(sys, 'argv', argv):
            pim.main()
        plt.close('all')

        for csv_name in ['ising_polar_order_curve',
                         'ising_polar_magnetization_blocks',
                         'ising_polar_correlation']:
            path = self.out_dir / f"{csv_name}.csv"
            self.assertTrue(path.exists(), msg=f"{csv_name}.csv missing")
            self.assertGreater(len(pd.read_csv(path)), 0)

        for fig_name in ['ising_polar_order_vs_magnetization',
                         'ising_polar_vs_magnetization_scatter']:
            for ext in ('png', 'svg'):
                self.assertTrue((self.out_dir / f"{fig_name}.{ext}").exists(),
                                msg=f"{fig_name}.{ext} missing")

        df_polar = pd.read_csv(self.out_dir / 'ising_polar_order_curve.csv')
        for col in ('bead_name', 'window_um', 'polar_mean', 'polar_sem', 'pooled_polar',
                    'abs_mean', 'abs_sem', 'pooled_abs_mean', 'delta_relative_mean',
                    'delta_relative_sem', 'delta_relative_pooled',
                    'ratio_abs_to_polar_pooled', 'n_blocks_total'):
            self.assertIn(col, df_polar.columns)
        # ±反平行 2 状態場なので P(R) = |M(R)| -> Delta = 0、比 = 1
        np.testing.assert_allclose(df_polar['polar_mean'], df_polar['abs_mean'], atol=1e-6)
        np.testing.assert_allclose(df_polar['delta_relative_pooled'], 0.0, atol=1e-3)
        np.testing.assert_allclose(df_polar['ratio_abs_to_polar_pooled'], 1.0, atol=1e-3)

        df_blocks = pd.read_csv(self.out_dir / 'ising_polar_magnetization_blocks.csv')
        self.assertTrue((df_blocks['n_blocks_sampled'] <= 16).all())
        np.testing.assert_allclose(df_blocks['polar'], df_blocks['abs_m'], atol=1e-6)

        df_corr = pd.read_csv(self.out_dir / 'ising_polar_correlation.csv')
        self.assertTrue((df_corr['n_points'] > 0).all())
        # 窓 1 画素（pixel_stride 単位）は分散 0 で相関が未定義 -> NaN
        self.assertTrue(np.isnan(df_corr.loc[df_corr['window_px'] == 2, 'pearson_r']).all())
        fin = df_corr[df_corr['window_px'] > 2]
        self.assertGreater(len(fin), 0)
        np.testing.assert_allclose(fin['pearson_r'], 1.0, atol=1e-9)
        np.testing.assert_allclose(fin['slope_origin'], 1.0, atol=1e-6)
        self.assertTrue((fin['spearman_rho'] > 0.9).all())

        df_sum = pd.read_csv(self.out_dir / 'ising_magnetization_summary.csv')
        for col in ('delta_relative_pooled', 'paired_n_points', 'paired_pearson_r',
                    'paired_spearman_rho', 'paired_slope_origin'):
            self.assertIn(col, df_sum.columns)
        self.assertTrue((df_sum['paired_n_points'] > 0).all())
        np.testing.assert_allclose(df_sum['paired_pearson_r'], 1.0, atol=1e-3)
        np.testing.assert_allclose(df_sum['delta_relative_pooled'], 0.0, atol=1e-3)


class TestCircularDiskConvolutionAndTheory(unittest.TestCase):

    def test_disk_kernel_creation(self):
        k1 = ising.create_disk_kernel(0.5)
        self.assertEqual(k1.shape, (1, 1))
        self.assertEqual(k1[0, 0], 1.0)

        k2 = ising.create_disk_kernel(2.0)
        self.assertEqual(k2.shape, (5, 5))
        # 半径 2 の円板内画素数: 1 + 4 + 4 + 4 = 13 画素
        self.assertEqual(int(np.sum(k2)), 13)

    def test_disk_magnetizations_uniform(self):
        sigma = np.ones((64, 64))
        valid = np.ones((64, 64), dtype=bool)
        m = ising.disk_magnetizations(sigma, valid, radius=4.0, step=4)
        m_fin = m[np.isfinite(m)]
        self.assertGreater(m_fin.size, 0)
        np.testing.assert_allclose(m_fin, 1.0)

        st = ising.disk_magnetization_stats(sigma, valid, radius=4.0, step=4)
        self.assertAlmostEqual(st['abs_mean'], 1.0)
        self.assertAlmostEqual(st['squared_mean'], 1.0)

    def test_disk_polar_orders_uniform(self):
        ux = np.ones((64, 64))
        uy = np.zeros((64, 64))
        valid = np.ones((64, 64), dtype=bool)
        p = ising.disk_polar_orders(ux, uy, valid, radius=4.0, step=4)
        p_fin = p[np.isfinite(p)]
        self.assertGreater(p_fin.size, 0)
        np.testing.assert_allclose(p_fin, 1.0)

    def test_disk_order_pairs_antiparallel(self):
        # 上半分 +1, 下半分 -1
        sigma = np.ones((64, 64))
        sigma[32:, :] = -1.0
        ux = np.zeros((64, 64))
        ux[:32, :] = 1.0
        ux[32:, :] = -1.0
        uy = np.zeros((64, 64))
        valid = np.ones((64, 64), dtype=bool)
        pairs = ising.disk_order_pairs(sigma, ux, uy, valid, radius=4.0, step=4)
        self.assertGreater(pairs['n_blocks'], 0)
        np.testing.assert_allclose(pairs['abs_m'], pairs['polar'], atol=1e-12)

    def test_theoretical_master_curve_disk(self):
        import plot_ising_scaling_master_curve as psmc
        # x -> 0 で 1
        self.assertAlmostEqual(float(psmc.theoretical_master_curve(np.array([1e-4]))[0]), 1.0, delta=0.01)
        # x = 1.0 で約 0.4413
        val_1 = float(psmc.theoretical_master_curve(np.array([1.0]))[0])
        self.assertAlmostEqual(val_1, 0.44134, places=3)
        # x = 2.0 で約 0.2282
        val_2 = float(psmc.theoretical_master_curve(np.array([2.0]))[0])
        self.assertAlmostEqual(val_2, 0.22823, places=3)
        # 単調減少性
        xs = np.logspace(-2, 2, 20)
        ys = psmc.theoretical_master_curve(xs)
        self.assertTrue(np.all(np.diff(ys) < 0))
        # 大スケール漸近解 ~ 2 / x^2
        val_large = float(psmc.theoretical_master_curve(np.array([50.0]))[0])
        self.assertAlmostEqual(val_large * (50.0 ** 2), 2.0, delta=0.1)


if __name__ == '__main__':
    unittest.main()
