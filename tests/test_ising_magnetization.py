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


if __name__ == '__main__':
    unittest.main()
