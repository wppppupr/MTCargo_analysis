"""
tests/test_mt_orientation_distribution.py

plot_mt_orientation_distribution.py の単体テスト。

- 角度ラップ / ネマチック折り返しの基本演算
- 配向がそろったフローフィールドでの Delta theta ヒストグラム（0 と +-pi の 2 ピーク）
- 流速ノルム重み付け・貨物粒子近傍マスク・密度規格化・周期平滑化
- 合成 GFP_flows.h5（+ MTs_im_theta.zarr）を用いた process_experiment / main() の
  エンドツーエンド動作（CSV / ヒストグラム / レーダーチャートの生成）
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

import plot_mt_orientation_distribution as mt_ori


def make_synthetic_flow(theta_a, theta_b, n_frames=6, rows=48, cols=48):
    """上半分を theta_a、下半分を theta_b の一様流にした合成フロー (T, 2, y, x) を作る。"""
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


def write_synthetic_experiment(exp_dir: Path, flows: np.ndarray, theta_nem: float = None):
    """GFP_flows.h5（と任意で MTs_im_theta.zarr）を書き出す。実データ同様 float16 で保存する。"""
    exp_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(exp_dir / mt_ori.FLOW_NAME), 'w') as f:
        f.create_dataset('flows', data=np.asarray(flows, dtype=np.float16))

    if theta_nem is not None:
        import zarr
        z = zarr.open_array(str(exp_dir / "MTs_im_theta.zarr"), mode='w',
                            shape=(flows.shape[0], 8, 8), dtype='float64')
        z[:] = float(theta_nem)
    return exp_dir


class TestAngleUtilities(unittest.TestCase):

    def test_wrap_to_pi(self):
        a = np.array([0.0, np.pi, -np.pi, 3.0 * np.pi, -3.0 * np.pi, np.pi / 2])
        w = mt_ori.wrap_to_pi(a)
        self.assertTrue(np.all(w <= np.pi))
        self.assertTrue(np.all(w >= -np.pi))
        np.testing.assert_allclose(mt_ori.wrap_to_pi(np.array([0.3])), [0.3])
        np.testing.assert_allclose(mt_ori.wrap_to_pi(np.array([0.3 + 2 * np.pi])), [0.3], atol=1e-9)
        # +pi と -pi は同一方向であり、ラップ後は -pi 側に代表される（[-pi, pi)）
        np.testing.assert_allclose(mt_ori.wrap_to_pi(np.array([np.pi])), [-np.pi], atol=1e-12)
        np.testing.assert_allclose(mt_ori.wrap_to_pi(np.array([-np.pi])), [-np.pi], atol=1e-12)

    def test_fold_to_half_pi(self):
        a = np.array([0.0, np.pi, -np.pi, np.pi / 2, np.pi / 2 + 0.1, np.deg2rad(95.0)])
        f = mt_ori.fold_to_half_pi(a)
        self.assertTrue(np.all(f <= 0.5 * np.pi))
        self.assertTrue(np.all(f >= -0.5 * np.pi))
        np.testing.assert_allclose(mt_ori.fold_to_half_pi(np.array([0.2, np.pi - 0.2])), [0.2, -0.2],
                                   atol=1e-9)
        # pi/2 と -pi/2 は同一（境界）で、折り返し後は -pi/2 に代表される
        np.testing.assert_allclose(mt_ori.fold_to_half_pi(np.array([0.5 * np.pi])), [-0.5 * np.pi],
                                   atol=1e-12)

    def test_make_angle_bins(self):
        edges = mt_ori.make_angle_bins(8)
        self.assertEqual(edges.size, 9)
        np.testing.assert_allclose(edges[0], -np.pi)
        np.testing.assert_allclose(edges[-1], np.pi)
        edges_f = mt_ori.make_angle_bins(4, fold=True)
        np.testing.assert_allclose(edges_f[0], -0.5 * np.pi)
        np.testing.assert_allclose(edges_f[-1], 0.5 * np.pi)

    def test_normalize_density_integrates_to_one(self):
        edges = mt_ori.make_angle_bins(36)
        hist = np.arange(1.0, 37.0)
        dens = mt_ori.normalize_density(hist, edges)
        self.assertAlmostEqual(float(np.sum(dens * np.diff(edges))), 1.0, places=9)

    def test_smooth_circular_conserves_mean(self):
        y = np.array([0.0, 1.0, 5.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        sm = mt_ori.smooth_circular(y, sigma_bins=1.0)
        self.assertEqual(sm.size, y.size)
        self.assertAlmostEqual(float(sm.mean()), float(y.mean()), places=9)
        # 平滑化により孤立ピークは広がり、最大値は下がる
        self.assertLess(sm.max(), y.max())
        # 周期的な一様分布は変化しない
        flat = np.full(12, 0.42)
        np.testing.assert_allclose(mt_ori.smooth_circular(flat, 1.5), flat, atol=1e-12)


class TestDeltaThetaHistogram(unittest.TestCase):

    def setUp(self):
        self.bin_edges = mt_ori.make_angle_bins(36)
        self.bin_edges_fold = mt_ori.make_angle_bins(18, fold=True)
        self.theta0 = 0.4

    def test_aligned_field_peaks_at_zero(self):
        """ネマチック主軸と完全に平行な流れ -> Delta theta = 0 に単一ピーク。"""
        rows, cols = 32, 32
        mx = np.full((rows, cols), np.cos(self.theta0), dtype=np.float32)
        my = np.full((rows, cols), np.sin(self.theta0), dtype=np.float32)

        hist, hist_fold, stats = mt_ori.delta_theta_histogram(
            mx, my, self.theta0, self.bin_edges, self.bin_edges_fold)

        self.assertEqual(stats['n_pixels'], rows * cols)
        # 0 を含むビンに全サンプルが入る
        self.assertEqual(int(hist.sum()), rows * cols)
        self.assertGreater(hist[18], 0)  # bin 18 = [0, +5deg)
        self.assertEqual(int(np.count_nonzero(hist)), 1)
        self.assertEqual(int(np.count_nonzero(hist_fold)), 1)

        cs = mt_ori.summarized_circular_stats(stats)
        self.assertAlmostEqual(cs['mean_cos'], 1.0, places=5)
        self.assertAlmostEqual(cs['R'], 1.0, places=5)
        self.assertAlmostEqual(cs['S2'], 1.0, places=5)
        self.assertAlmostEqual(cs['mean_abs_deg'], 0.0, places=4)
        self.assertAlmostEqual(cs['frac_aligned'], 1.0, places=6)

    def test_two_lobes_at_zero_and_pi(self):
        """平行領域と反平行領域が半々 -> Delta theta = 0 と +-pi に 2 ピーク。"""
        flows = make_synthetic_flow(self.theta0, self.theta0 + np.pi)
        mx, my = flows[0, 0], flows[0, 1]
        hist, hist_fold, stats = mt_ori.delta_theta_histogram(
            mx, my, self.theta0, self.bin_edges, self.bin_edges_fold)

        n = hist.sum()
        self.assertEqual(int(n), mx.size)
        # +-pi/2 のビン（直交方向）は空
        self.assertEqual(int(hist[9]), 0)
        self.assertEqual(int(hist[27]), 0)
        # 非ゼロは「0 のビン」と「+-pi 側の端ビン」の 2 つだけ
        nonzero = set(int(i) for i in np.nonzero(hist)[0])
        self.assertEqual(len(nonzero), 2)
        self.assertIn(18, nonzero)
        self.assertTrue(nonzero - {18} <= {0, 35})
        # 0 側のビンと +-pi 側の端ビンに半々ずつ入る（+pi と -pi は同一方向）
        self.assertAlmostEqual(hist[18] / n, 0.5, places=6)
        self.assertAlmostEqual((hist[0] + hist[-1]) / n, 0.5, places=6)

        cs = mt_ori.summarized_circular_stats(stats)
        self.assertAlmostEqual(cs['mean_cos'], 0.0, places=5)   # 極性秩序はゼロ
        self.assertAlmostEqual(cs['S2'], 1.0, places=5)         # ネマチック秩序は完全
        self.assertAlmostEqual(cs['frac_aligned'], 0.5, places=6)
        self.assertAlmostEqual(cs['frac_anti'], 0.5, places=6)

        # 折り返し分布（mod pi）でも 0 に単一ピーク
        self.assertEqual(int(hist_fold.sum()), int(n))
        self.assertGreater(hist_fold[9], 0)

    def test_isotropic_field_is_flat(self):
        """等方（ランダム配向）場 -> 全ビンほぼ 1/(2 pi) の平坦分布。"""
        rng = np.random.default_rng(0)
        phi = rng.uniform(-np.pi, np.pi, size=(256, 256)).astype(np.float32)
        mx, my = np.cos(phi), np.sin(phi)
        hist, _, stats = mt_ori.delta_theta_histogram(
            mx, my, 0.0, self.bin_edges, self.bin_edges_fold)
        dens = mt_ori.normalize_density(hist, self.bin_edges)
        np.testing.assert_allclose(dens, np.full_like(dens, 1.0 / (2 * np.pi)), rtol=0.25)
        cs = mt_ori.summarized_circular_stats(stats)
        self.assertLess(cs['R'], 0.05)
        self.assertLess(abs(cs['S2']), 0.05)

    def test_min_flow_mag_and_mask(self):
        """閾値以下 / マスクされた画素はカウントされない。"""
        rows, cols = 10, 10
        mx = np.full((rows, cols), np.cos(self.theta0), dtype=np.float32)
        my = np.full((rows, cols), np.sin(self.theta0), dtype=np.float32)
        mx[0, :] = my[0, :] = 0.0  # 静止画素

        mask = np.ones((rows, cols), dtype=bool)
        mask[1, :] = False  # 1 行をマスク

        _, _, stats = mt_ori.delta_theta_histogram(
            mx, my, self.theta0, self.bin_edges, self.bin_edges_fold,
            min_flow_mag=1e-4, valid_mask=mask)
        self.assertEqual(stats['n_pixels'], (rows - 2) * cols)

    def test_magnitude_weighting(self):
        """流速ノルム重み付けでは速い画素の寄与が支配する。"""
        mx = np.array([[np.cos(self.theta0), np.cos(self.theta0 + np.pi / 2)]], dtype=np.float32)
        my = np.array([[np.sin(self.theta0), np.sin(self.theta0 + np.pi / 2)]], dtype=np.float32)
        mx[0, 1] *= 5.0
        my[0, 1] *= 5.0

        _, _, stats_none = mt_ori.delta_theta_histogram(
            mx, my, self.theta0, self.bin_edges, self.bin_edges_fold, weighting='none')
        _, _, stats_mag = mt_ori.delta_theta_histogram(
            mx, my, self.theta0, self.bin_edges, self.bin_edges_fold, weighting='magnitude')

        cs_none = mt_ori.summarized_circular_stats(stats_none)
        cs_mag = mt_ori.summarized_circular_stats(stats_mag)
        # 等重み: 平行 (cos=1) と直交 (cos=0) が 1:1 -> <cos> = 0.5
        self.assertAlmostEqual(cs_none['mean_cos'], 0.5, places=6)
        # ノルム重み: (1*1 + 5*0) / (1 + 5) = 1/6
        self.assertAlmostEqual(cs_mag['mean_cos'], 1.0 / 6.0, places=5)
        self.assertLess(cs_mag['mean_cos'], cs_none['mean_cos'])


class TestCargoMask(unittest.TestCase):

    def test_mask_center_and_radius(self):
        shape = (20, 20)
        mask = mt_ori.make_cargo_mask(np.array([10.0]), np.array([10.0]), shape, stride=2, radius_px=4.0)
        self.assertTrue(mask[5, 5])           # 中心 (y=10/2, x=10/2)
        self.assertTrue(mask[4, 5])           # 2 ストライド上 = 4 px 上（半径内）
        self.assertTrue(mask[3, 5])           # 4 ストライド上 = 8 px 上（ちょうど半径）
        self.assertFalse(mask[2, 5])          # 6 px*2 = 12 px 上（半径外）
        # stride=2, r=2 ストライド -> 距離 <= 2 の格子点 (5x5 中の 13 点)
        self.assertEqual(int(mask.sum()), 13)

    def test_mask_box_is_limited_to_neighbourhood(self):
        """マスクは粒子近傍のバウンディングボックス内のみ True。"""
        mask = mt_ori.make_cargo_mask(np.array([4.0]), np.array([4.0]), (32, 32), stride=1, radius_px=3.0)
        self.assertTrue(mask[4, 4])
        self.assertFalse(mask[10, 10])
        self.assertEqual(int(mask.sum()), 29)  # 半径 3 の円内格子点（1 + 4 + 8 + 12 + 4）

    def test_mask_empty_positions(self):
        mask = mt_ori.make_cargo_mask(None, None, (8, 8), stride=1, radius_px=5.0)
        self.assertFalse(mask.any())
        mask2 = mt_ori.make_cargo_mask(np.array([]), np.array([]), (8, 8), stride=1, radius_px=5.0)
        self.assertFalse(mask2.any())

    def test_cargo_positions_by_frame(self):
        df = pd.DataFrame({'frame': [0, 0, 1], 'x': [1.0, 2.0, 3.0], 'y': [4.0, 5.0, 6.0]})
        pos = mt_ori.cargo_positions_by_frame(df)
        self.assertEqual(sorted(pos.keys()), [0, 1])
        np.testing.assert_allclose(pos[0][0], [1.0, 2.0])
        np.testing.assert_allclose(pos[1][1], [6.0])
        self.assertEqual(mt_ori.cargo_positions_by_frame(None), {})



class TestProcessExperiment(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.bin_edges = mt_ori.make_angle_bins(36)
        self.bin_edges_fold = mt_ori.make_angle_bins(18, fold=True)
        self.bead = mt_ori.BEAD_LOOKUP['beads06um']

    def _exp_dir(self, name='syn001'):
        return self.root / 'beads06um' / '20260101' / name

    def test_theta_from_flow_fallback(self):
        """MTs_im_theta.zarr が無い場合はフローからネマチック主軸を推定する。"""
        theta0 = 0.4
        flows = make_synthetic_flow(theta0, theta0 + np.pi, n_frames=4, rows=32, cols=32)
        exp_dir = write_synthetic_experiment(self._exp_dir(), flows)

        res = mt_ori.process_experiment(exp_dir, self.bead, self.bin_edges, self.bin_edges_fold,
                                        pixel_stride=1, frame_stride=1, progress=False)
        self.assertIsNotNone(res)
        self.assertEqual(res['theta_source'], 'flow')
        self.assertEqual(res['n_frames_used'], 4)
        self.assertEqual(res['n_pixels'], 4 * 32 * 32)
        self.assertEqual(res['dataset_shape'], (4, 2, 32, 32))

        dens = mt_ori.normalize_density(res['hist'], self.bin_edges)
        self.assertAlmostEqual(float(np.sum(dens * np.diff(self.bin_edges))), 1.0, places=8)
        # 0（bin 17-19）と +-pi（両端ビン）にピーク、+-90deg（bin 9, 27）はほぼ空
        peak_zero = float(dens[17:20].max())
        peak_pi = float(max(dens[0:2].max(), dens[-2:].max()))
        self.assertGreater(peak_zero, 0.0)
        self.assertGreater(peak_pi, 0.0)
        self.assertLess(float(dens[9]), 0.05 * peak_zero)
        self.assertLess(float(dens[27]), 0.05 * peak_zero)
        cs = mt_ori.summarized_circular_stats(res['stats'])
        self.assertAlmostEqual(cs['S2'], 1.0, places=4)

    def test_theta_from_zarr(self):
        """MTs_im_theta.zarr がある場合はそちらを主軸として使用する。"""
        theta0 = 0.4
        flows = make_synthetic_flow(theta0, theta0, n_frames=3, rows=16, cols=16)
        exp_dir = write_synthetic_experiment(self._exp_dir('syn002'), flows, theta_nem=theta0)

        res = mt_ori.process_experiment(exp_dir, self.bead, self.bin_edges, self.bin_edges_fold,
                                        pixel_stride=1, frame_stride=1, progress=False)
        self.assertEqual(res['theta_source'], 'MTs_im_theta.zarr')
        cs = mt_ori.summarized_circular_stats(res['stats'])
        self.assertAlmostEqual(cs['mean_cos'], 1.0, places=3)
        self.assertAlmostEqual(cs['mean_abs_deg'], 0.0, places=2)

    def test_frame_and_pixel_stride(self):
        """frame_stride / pixel_stride / max_frames_per_exp がサンプル数に反映される。"""
        theta0 = 0.2
        flows = make_synthetic_flow(theta0, theta0, n_frames=8, rows=32, cols=32)
        exp_dir = write_synthetic_experiment(self._exp_dir('syn003'), flows)

        res = mt_ori.process_experiment(exp_dir, self.bead, self.bin_edges, self.bin_edges_fold,
                                        pixel_stride=2, frame_stride=2, max_frames_per_exp=3,
                                        progress=False)
        self.assertEqual(res['n_frames_used'], 3)
        self.assertEqual(res['n_pixels'], 3 * 16 * 16)

    def test_cargo_masking_reduces_samples(self):
        """beads_tracks.csv があれば貨物近傍画素が除外される。"""
        theta0 = 0.2
        flows = make_synthetic_flow(theta0, theta0, n_frames=2, rows=32, cols=32)
        exp_dir = write_synthetic_experiment(self._exp_dir('syn004'), flows)
        pd.DataFrame({'frame': [0, 1], 'x': [16.0, 16.0], 'y': [16.0, 16.0]}).to_csv(
            exp_dir / mt_ori.TRACKS_NAME, index=False)

        res_nomask = mt_ori.process_experiment(exp_dir, self.bead, self.bin_edges, self.bin_edges_fold,
                                               pixel_stride=1, frame_stride=1, progress=False)
        res_mask = mt_ori.process_experiment(exp_dir, self.bead, self.bin_edges, self.bin_edges_fold,
                                             pixel_stride=1, frame_stride=1, progress=False,
                                             mask_radius_factor=2.0, min_mask_radius_px=6.0)
        self.assertGreater(res_mask['mask_radius_px'], 0.0)
        self.assertLess(res_mask['n_pixels'], res_nomask['n_pixels'])

    def test_directors_match_canonical_implementation(self):
        """load_nematic_directors が libs の load_nematic_thetas と同じ theta_nem を与える。"""
        import zarr
        from libs.calc_bg_angular_correlation import load_nematic_thetas

        exp_dir = self._exp_dir('syn005')
        flows = make_synthetic_flow(0.3, 0.3, n_frames=4, rows=16, cols=16)
        exp_dir.mkdir(parents=True, exist_ok=True)
        with h5py.File(str(exp_dir / mt_ori.FLOW_NAME), 'w') as f:
            f.create_dataset('flows', data=flows)

        rng = np.random.default_rng(7)
        local_theta = rng.uniform(-0.5 * np.pi, 0.5 * np.pi, size=(5, 8, 8))
        local_theta[0, 0, 0] = np.nan  # NaN 混入時のフォールバックも検証
        z = zarr.open_array(str(exp_dir / "MTs_im_theta.zarr"), mode='w',
                            shape=local_theta.shape, dtype='float64')
        z[:] = local_theta

        mine = mt_ori.load_nematic_directors(exp_dir, [0, 1, 2, 3], 4)
        ref = load_nematic_thetas(exp_dir, 4, flow_data=None)
        self.assertIsNotNone(mine)
        np.testing.assert_allclose(mine, ref, atol=1e-6)

    def test_delta_theta_variants_mirror(self):
        """delta_theta_variants が +theta と -theta の両方を計算し、theta=None では無効化される。"""
        theta0 = 0.4
        flows = make_synthetic_flow(theta0, theta0)
        mx, my = flows[0, 0], flows[0, 1]
        v = mt_ori.delta_theta_variants(mx, my, theta0, self.bin_edges, self.bin_edges_fold)
        self.assertIsNotNone(v['plus'])
        self.assertIsNotNone(v['minus'])
        # +theta 基準: Delta theta = 0 にピーク（S2 = 1）
        self.assertAlmostEqual(
            mt_ori.summarized_circular_stats(v['plus'][2])['S2'], 1.0, places=5)
        # -theta 基準: Delta theta = 2 theta0 (cos 4 theta0)
        self.assertAlmostEqual(
            mt_ori.summarized_circular_stats(v['minus'][2])['S2'], np.cos(4 * theta0), places=5)
        # 主軸をフローから推定する場合（theta_nem=None）はミラー候補を作らない
        v2 = mt_ori.delta_theta_variants(mx, my, None, self.bin_edges, self.bin_edges_fold)
        self.assertIsNotNone(v2['plus'])
        self.assertIsNone(v2['minus'])
        self.assertAlmostEqual(
            mt_ori.summarized_circular_stats(v2['plus'][2])['S2'], 1.0, places=4)

    def test_choose_and_select_sign(self):
        """ミラー規約のデータでは -1 が選ばれ、Delta theta が 0 にピークする。"""
        b = mt_ori.make_angle_bins(36)
        bf = mt_ori.make_angle_bins(18, fold=True)
        theta_true = 0.4

        flows = make_synthetic_flow(theta_true, theta_true)
        mx, my = flows[0, 0], flows[0, 1]

        def make_result(theta_ref):
            variants = mt_ori.delta_theta_variants(mx, my, theta_ref, b, bf)
            hp, hfp, sp = variants['plus']
            res = {
                'bead_name': 'beads06um', 'exp_dir': '/tmp/x', 'hist': hp, 'hist_fold': hfp,
                'stats': sp, 'stats_minus': variants['minus'][2] if variants['minus'] else None,
                'hist_minus': variants['minus'][0] if variants['minus'] else None,
                'hist_fold_minus': variants['minus'][1] if variants['minus'] else None,
                'n_frames_used': 1, 'n_frames_total': 1, 'n_pixels': int(sp['n_pixels']),
                'weight_sum': sp['wsum'], 'mask_radius_px': 0.0,
                'theta_source': 'MTs_im_theta.zarr', 'theta_mean_rad': theta_ref, 'theta_sign': 1,
            }
            res['S2_plus'] = sp['c2'] / sp['wsum']
            res['S2_minus'] = (variants['minus'][2]['c2'] / variants['minus'][2]['wsum']
                               if variants['minus'] else np.nan)
            return res

        # (1) zarr が +theta_true（フローと整合）-> +1
        res_ok = make_result(theta_true)
        sign, info = mt_ori.choose_theta_sign([res_ok], 'auto', margin=0.05)
        self.assertEqual(sign, 1)
        self.assertGreater(info['S2_pooled_plus'], 0.99)

        # (2) zarr が -theta_true（ミラー規約）-> -1
        res_mirror = make_result(-theta_true)
        sign2, info2 = mt_ori.choose_theta_sign([res_mirror], 'auto', margin=0.05)
        self.assertEqual(sign2, -1)
        self.assertGreater(info2['S2_pooled_minus'], info2['S2_pooled_plus'])

        selected = mt_ori.select_sign_variant(res_mirror, -1)
        self.assertEqual(selected['theta_sign'], -1)
        cs = mt_ori.summarized_circular_stats(selected['stats'])
        self.assertAlmostEqual(cs['S2'], 1.0, places=5)
        self.assertAlmostEqual(cs['mean_abs_deg'], 0.0, places=3)
        # 強制指定も尊重される
        self.assertEqual(mt_ori.choose_theta_sign([res_mirror], '+1')[0], 1)
        self.assertEqual(mt_ori.choose_theta_sign([res_ok], '-1')[0], -1)

    def test_theta_series_is_recorded_per_frame(self):
        """process_experiment が使用フレームごとの theta_nem(t) を保持する。"""
        theta0 = 0.3
        flows = make_synthetic_flow(theta0, theta0, n_frames=6, rows=16, cols=16)
        exp_dir = write_synthetic_experiment(self._exp_dir('syn006'), flows, theta_nem=theta0)

        res = mt_ori.process_experiment(exp_dir, self.bead, self.bin_edges, self.bin_edges_fold,
                                        pixel_stride=1, frame_stride=2, progress=False)
        self.assertEqual(res['n_frames_used'], 3)                      # frames 0, 2, 4
        np.testing.assert_array_equal(res['theta_frames'], [0, 2, 4])
        self.assertEqual(res['theta_series_rad'].size, 3)
        self.assertTrue(np.isfinite(res['theta_series_rad']).all())
        np.testing.assert_allclose(res['theta_series_rad'], np.full(3, theta0), atol=1e-4)

        # 選択符号が -1 のときは実際に使った基準軸（-theta）が時系列に反映される
        mirrored = mt_ori.select_sign_variant(res, -1)
        if mirrored.get('stats_minus') is not None:
            self.assertEqual(mirrored['theta_sign'], -1)
            np.testing.assert_allclose(mirrored['theta_series_rad'], -np.full(3, theta0), atol=1e-4)
        else:
            self.assertEqual(mirrored['theta_sign'], 1)

    def test_missing_flow_returns_none(self):
        empty_dir = self.root / 'beads06um' / '20260101' / 'empty'
        empty_dir.mkdir(parents=True)
        self.assertIsNone(mt_ori.process_experiment(empty_dir, self.bead, self.bin_edges,
                                                    self.bin_edges_fold, progress=False))



class TestHistogramUniform(unittest.TestCase):

    def test_matches_numpy_histogram(self):
        """histogram_uniform が np.histogram（一様ビン）と完全一致する。"""
        rng = np.random.default_rng(0)
        edges = mt_ori.make_angle_bins(37)
        for n in (0, 1, 10, 1000):
            v = rng.uniform(-np.pi, np.pi, size=n)
            np.testing.assert_array_equal(mt_ori.histogram_uniform(v, edges),
                                          np.histogram(v, bins=edges)[0])
        # 境界値: -pi（最左端）と +pi（最右端）は numpy と同じビンに入る
        v = np.array([-np.pi, np.pi, 0.0])
        np.testing.assert_array_equal(mt_ori.histogram_uniform(v, edges),
                                      np.histogram(v, bins=edges)[0])
        # 範囲外の値は端ビンにクリップされる（NaN は入らない前提）
        out = mt_ori.histogram_uniform(np.array([-10.0, 10.0]), edges)
        self.assertEqual(int(out.sum()), 2)

    def test_matches_numpy_histogram_weighted(self):
        rng = np.random.default_rng(1)
        edges = mt_ori.make_angle_bins(18, fold=True)
        v = rng.uniform(-0.5 * np.pi, 0.5 * np.pi, size=500)
        w = rng.uniform(0.0, 1.0, size=500)
        np.testing.assert_allclose(
            mt_ori.histogram_uniform(v, edges, weights=w),
            np.histogram(v, bins=edges, weights=w)[0], rtol=1e-12, atol=0.0)

    def test_histogram_agrees_with_old_implementation(self):
        """delta_theta_histogram の出力が np.histogram 実装と一致する（回帰確認）。"""
        theta0 = 0.35
        flows = make_synthetic_flow(theta0, theta0 + np.pi, n_frames=1, rows=32, cols=32)
        mx, my = flows[0, 0], flows[0, 1]
        be = mt_ori.make_angle_bins(72)
        bef = mt_ori.make_angle_bins(36, fold=True)
        hist, hist_fold, stats = mt_ori.delta_theta_histogram(mx, my, theta0, be, bef)

        phi = np.arctan2(my.ravel(), mx.ravel())
        dth = mt_ori.wrap_to_pi(phi - theta0)
        np.testing.assert_array_equal(hist, np.histogram(dth, bins=be)[0])
        np.testing.assert_array_equal(hist_fold, np.histogram(mt_ori.fold_to_half_pi(dth), bins=bef)[0])
        self.assertAlmostEqual(stats['c2'], float(np.sum(np.cos(2 * dth))), places=6)
        self.assertAlmostEqual(stats['c1'], float(np.sum(np.cos(dth))), places=6)
        self.assertAlmostEqual(stats['s1'], float(np.sum(np.sin(dth))), places=6)
        self.assertAlmostEqual(stats['abs_sum'], float(np.sum(np.abs(dth))), places=6)


class TestFlowCache(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.bin_edges = mt_ori.make_angle_bins(24)
        self.bin_edges_fold = mt_ori.make_angle_bins(12, fold=True)
        self.bead = mt_ori.BEAD_LOOKUP['beads06um']
        self.theta0 = 0.4

    def _exp_dir(self, name='cache001'):
        d = self.root / 'beads06um' / '20260101' / name
        flows = make_synthetic_flow(self.theta0, self.theta0 + np.pi, n_frames=6, rows=24, cols=24)
        write_synthetic_experiment(d, flows, theta_nem=self.theta0)
        return d

    def _run(self, exp_dir, flow_cache='auto'):
        return mt_ori.process_experiment(
            exp_dir, self.bead, self.bin_edges, self.bin_edges_fold,
            pixel_stride=4, frame_stride=2, progress=False, flow_cache=flow_cache)

    def test_auto_builds_then_reuses_cache(self):
        exp_dir = self._exp_dir()
        res1 = self._run(exp_dir, 'auto')
        cache = exp_dir / 'mt_flow_cache_s4_f2.h5'
        self.assertTrue(cache.exists())
        self.assertEqual(res1['flow_cache_source'], 'flow')

        res2 = self._run(exp_dir, 'auto')
        self.assertEqual(res2['flow_cache_source'], 'cache')
        np.testing.assert_array_equal(res1['hist'], res2['hist'])
        np.testing.assert_array_equal(res1['hist_fold'], res2['hist_fold'])
        self.assertEqual(res1['n_pixels'], res2['n_pixels'])
        for k in res1['stats']:
            self.assertAlmostEqual(res1['stats'][k], res2['stats'][k], places=6)
        np.testing.assert_array_equal(res1['theta_frames'], res2['theta_frames'])

    def test_cache_used_when_flow_file_is_absent(self):
        exp_dir = self._exp_dir()
        res1 = self._run(exp_dir, 'auto')
        (exp_dir / mt_ori.FLOW_NAME).rename(exp_dir / 'GFP_flows.h5.bak')
        res2 = self._run(exp_dir, 'auto')
        self.assertEqual(res2['flow_cache_source'], 'cache')
        np.testing.assert_array_equal(res1['hist'], res2['hist'])
        self.assertEqual(res1['n_pixels'], res2['n_pixels'])

    def test_off_creates_no_cache(self):
        exp_dir = self._exp_dir('cache002')
        res = self._run(exp_dir, 'off')
        self.assertEqual(res['flow_cache_source'], 'flow')
        self.assertFalse((exp_dir / 'mt_flow_cache_s4_f2.h5').exists())

    def test_stale_cache_is_rebuilt(self):
        exp_dir = self._exp_dir('cache003')
        self._run(exp_dir, 'auto')
        # フローを書き換える（サイズが変われば auto は作り直す）
        flows = make_synthetic_flow(self.theta0, self.theta0, n_frames=9, rows=24, cols=24)
        write_synthetic_experiment(exp_dir, flows, theta_nem=self.theta0)
        res = self._run(exp_dir, 'auto')
        self.assertEqual(res['flow_cache_source'], 'flow')
        # refresh は常にフローから読み直す
        res_refresh = self._run(exp_dir, 'refresh')
        self.assertEqual(res_refresh['flow_cache_source'], 'flow')

    def test_resolve_flow_cache_name(self):
        root = Path('/data/root')
        exp = root / 'beads06um' / '20260101' / 'exp001'
        self.assertIsNone(mt_ori.resolve_flow_cache_name(None, root, exp, 4, 5))
        self.assertEqual(mt_ori.resolve_flow_cache_name('', root, exp, 4, 5), None)
        # ローカルディスク指定時は root 相対のパスを組み立てる
        got = mt_ori.resolve_flow_cache_name('/tmp/cache', root, exp, 8, 5)
        self.assertEqual(got, '/tmp/cache/beads06um/20260101/exp001/mt_flow_cache_s8_f5.h5')
        got2 = mt_ori.resolve_flow_cache_name('/tmp/cache', root, Path('/other/x'), 4, 2)
        self.assertTrue(got2.endswith('x/mt_flow_cache_s4_f2.h5'))

    def test_missing_flow_and_cache_returns_none(self):
        exp_dir = self.root / 'beads06um' / '20260101' / 'empty'
        exp_dir.mkdir(parents=True)
        self.assertIsNone(self._run(exp_dir, 'auto'))


class TestThetaTimeseries(unittest.TestCase):

    def test_stats_constant_series(self):
        th = np.full(20, 0.3)
        st = mt_ori.theta_timeseries_stats(th)
        self.assertAlmostEqual(st['theta_nem_mean_deg'], np.rad2deg(0.3), places=5)
        self.assertAlmostEqual(st['theta_nem_std_deg_continuous'], 0.0, places=6)
        self.assertAlmostEqual(st['theta_nem_range_deg'], 0.0, places=6)
        self.assertAlmostEqual(st['theta_nem_drift_deg'], 0.0, places=6)
        self.assertAlmostEqual(st['theta_nem_axis_order'], 1.0, places=6)
        self.assertEqual(st['n_frames'], 20)

    def test_stats_linear_rotation(self):
        """時間とともに主軸が回転 -> レンジ / ドリフトが回転量を再現する。"""
        th = np.deg2rad(np.linspace(0.0, 25.0, 50))
        st = mt_ori.theta_timeseries_stats(th)
        self.assertAlmostEqual(st['theta_nem_range_deg'], 25.0, places=6)
        self.assertAlmostEqual(st['theta_nem_drift_deg'], 25.0, places=6)
        self.assertTrue(0.0 < st['theta_nem_axis_order'] < 1.0)

    def test_stats_pi_jump_is_same_axis(self):
        """+pi のジャンプは同一軸なので連続化後は変化なしとして扱われる。"""
        th = np.concatenate([np.full(10, 0.2), np.full(10, 0.2 + np.pi)])
        st = mt_ori.theta_timeseries_stats(th)
        self.assertAlmostEqual(st['theta_nem_range_deg'], 0.0, places=6)
        self.assertAlmostEqual(st['theta_nem_drift_deg'], 0.0, places=6)
        self.assertAlmostEqual(st['theta_nem_axis_order'], 1.0, places=6)

    def test_stats_empty(self):
        st = mt_ori.theta_timeseries_stats(np.array([np.nan, np.nan]))
        self.assertEqual(st['n_frames'], 0)
        self.assertTrue(np.isnan(st['theta_nem_std_deg_continuous']))

    def test_timeseries_table(self):
        frames = np.array([0, 5, 10, 15])
        series = np.deg2rad(np.array([0.0, 10.0, 20.0, 30.0]))
        res = {
            'bead_name': 'beads06um', 'exp_dir': '/tmp/a',
            'theta_series_rad': series, 'theta_frames': frames,
        }
        df = mt_ori.theta_timeseries_table([res], frame_interval=4.0)
        self.assertEqual(len(df), 4)
        for col in ['bead_name', 'exp_dir', 'sample_index', 'frame', 'time_s',
                    'theta_nem_rad', 'theta_nem_deg', 'theta_nem_deg_continuous',
                    'theta_nem_deg_relative']:
            self.assertIn(col, df.columns)
        np.testing.assert_allclose(df['frame'].values, frames)
        np.testing.assert_allclose(df['time_s'].values, frames * 4.0)
        np.testing.assert_allclose(df['theta_nem_deg_continuous'].values, [0.0, 10.0, 20.0, 30.0])
        np.testing.assert_allclose(df['theta_nem_deg_relative'].values, [0.0, 10.0, 20.0, 30.0])

    def test_timeseries_table_skips_missing_and_nan(self):
        self.assertTrue(mt_ori.theta_timeseries_table([{'bead_name': 'x', 'exp_dir': 'y'}]).empty)
        res = {'bead_name': 'x', 'exp_dir': 'y', 'theta_series_rad': np.array([np.nan, np.nan]),
               'theta_frames': np.array([0, 1])}
        self.assertTrue(mt_ori.theta_timeseries_table([res]).empty)


class TestTablesAndPlots(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out_dir = Path(self.tmp.name) / 'figs'
        self.bin_edges = mt_ori.make_angle_bins(24)
        self.bin_edges_fold = mt_ori.make_angle_bins(12, fold=True)
        self.beads = [mt_ori.BEAD_LOOKUP['beads06um'], mt_ori.BEAD_LOOKUP['beads1um']]

        # 2 条件 x 2 実験のダミー結果（0 と +-pi に 2 ピーク）
        self.results = []
        for bead in self.beads:
            for k in range(2):
                hist = np.zeros(self.bin_edges.size - 1)
                hist[12] = 50.0 + k          # 0 側（bin 12 = [0, 15deg)）
                hist[0] = 50.0 - k           # -pi 側
                hist_fold = np.zeros(self.bin_edges_fold.size - 1)
                hist_fold[6] = 100.0
                self.results.append({
                    'bead_name': bead['name'],
                    'exp_dir': f"/tmp/{bead['name']}/{k}",
                    'hist': hist,
                    'hist_fold': hist_fold,
                    'stats': {'wsum': 100.0, 'n_pixels': 100.0, 'c1': 4.0, 's1': 0.0,
                              'c2': 88.0, 'abs_sum': 160.0, 'n_align': 50.0, 'n_anti': 50.0},
                    'n_frames_used': 3,
                    'n_frames_total': 5,
                    'dataset_shape': (5, 2, 16, 16),
                    'channel_first': True,
                    'theta_source': 'MTs_im_theta.zarr',
                    'mask_radius_px': 0.0,
                    'weight_sum': 100.0,
                    'n_pixels': 100,
                    'theta_mean_rad': np.deg2rad(10.0 * k),
                    'theta_frames': np.arange(4),
                    'theta_series_rad': np.deg2rad(np.array([0.0, 5.0, 10.0, 15.0]) + 10.0 * k),
                })

    def test_tables(self):
        df_exp = mt_ori.per_experiment_table(self.results)
        self.assertEqual(len(df_exp), 4)
        self.assertTrue(np.isfinite(df_exp['nematic_order_cos2']).all())

        df_curve, _ = mt_ori.condition_histogram_table(
            self.results, self.beads, self.bin_edges, self.bin_edges_fold)
        self.assertEqual(len(df_curve), 2 * (self.bin_edges.size - 1))
        sub = df_curve[df_curve['bead_name'] == 'beads06um']
        # 密度は積分 1
        self.assertAlmostEqual(float(np.sum(sub['mean_density'] * np.diff(self.bin_edges))), 1.0, places=6)
        # 0 と -pi のビンが上位 2 つ（+pi は -pi と同一方向として左端ビンに落ちる）
        centers = sub['angle_center_rad'].to_numpy()
        top = sub['mean_density'].to_numpy().argsort()[::-1][:2]
        expected = {int(np.argmin(np.abs(centers))), int(np.argmin(np.abs(centers + np.pi)))}
        self.assertEqual(set(int(i) for i in top), expected)
        self.assertEqual(expected, {0, 12})

        cs = mt_ori.summarized_circular_stats(self.results[0]['stats'])
        self.assertAlmostEqual(cs['S2'], 0.88, places=6)
        self.assertAlmostEqual(cs['R'], 0.04, places=6)

        df_sum = mt_ori.condition_summary_table(self.results, self.beads, align_tol_deg=30.0)
        self.assertEqual(len(df_sum), 2)
        self.assertAlmostEqual(df_sum['nematic_order_cos2_pooled'].iloc[0], 0.88, places=6)
        self.assertAlmostEqual(df_sum['R_pooled'].iloc[0], 0.04, places=6)
        self.assertAlmostEqual(df_sum['mean_abs_delta_theta_deg_pooled'].iloc[0],
                               np.rad2deg(1.6), places=6)
        self.assertAlmostEqual(df_sum['frac_parallel_within_tol_mean'].iloc[0], 0.5, places=6)

    def test_all_figures_are_created(self):
        df_curve, df_curve_fold = mt_ori.condition_histogram_table(
            self.results, self.beads, self.bin_edges, self.bin_edges_fold)
        df_sum = mt_ori.condition_summary_table(self.results, self.beads, align_tol_deg=30.0)
        out_dirs = [self.out_dir]

        mt_ori.plot_histogram_panels(df_curve, df_sum, self.beads, out_dirs)
        mt_ori.plot_histogram_overlay(df_curve, self.beads, out_dirs)
        mt_ori.plot_radar_panels(df_curve, df_sum, self.beads, out_dirs)
        mt_ori.plot_radar_overlay(df_curve, self.beads, out_dirs)
        mt_ori.plot_nematic_folded(df_curve_fold, df_sum, self.beads, out_dirs)
        mt_ori.plot_theta_timeseries(mt_ori.theta_timeseries_table(self.results, 4.0),
                                     df_sum, self.beads, out_dirs)
        plt.close('all')

        for name in ['mt_orientation_histogram_panels', 'mt_orientation_histogram_overlay',
                     'mt_orientation_radar_panels', 'mt_orientation_radar_overlay',
                     'mt_orientation_nematic_folded', 'mt_orientation_theta_time_series']:
            for ext in ('png', 'svg'):
                self.assertTrue((self.out_dir / f"{name}.{ext}").exists(), msg=f"{name}.{ext} missing")



class TestMainEndToEnd(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'root'
        self.out_dir = Path(self.tmp.name) / 'out'

    def _build_synthetic_root(self, theta_sign_zarr=1.0):
        for bead_name, theta0 in [('beads06um', 0.4), ('beads1um', -0.3)]:
            flows = make_synthetic_flow(theta0, theta0 + np.pi, n_frames=4, rows=32, cols=32)
            for k in range(2):
                write_synthetic_experiment(
                    self.root / bead_name / '20260101' / f'exp{k:03d}', flows,
                    theta_nem=theta_sign_zarr * theta0)
        return self.root

    def test_main_generates_csv_and_figures(self):
        self._build_synthetic_root()
        argv = ['plot_mt_orientation_distribution.py',
                '--root_dir', str(self.root),
                '--beads', 'beads06um', 'beads1um',
                '--output_dir', str(self.out_dir),
                '--pixel_stride', '1', '--frame_stride', '1',
                '--bins', '18', '--bins_fold', '9',
                '--no_progress']
        with mock.patch.object(sys, 'argv', argv):
            mt_ori.main()
        plt.close('all')

        for csv_name in ['mt_orientation_per_experiment', 'mt_orientation_histogram',
                         'mt_orientation_nematic_folded_histogram', 'mt_orientation_summary',
                         'mt_orientation_theta_nem_timeseries']:
            path = self.out_dir / f"{csv_name}.csv"
            self.assertTrue(path.exists(), msg=f"{csv_name}.csv missing")
            self.assertGreater(len(pd.read_csv(path)), 0)

        # theta_nem(t) 時系列（各実験 x 使用フレーム）と変化統計列
        df_ts = pd.read_csv(self.out_dir / 'mt_orientation_theta_nem_timeseries.csv')
        self.assertEqual(len(df_ts), 2 * 2 * 4)   # 条件 2 x 実験 2 x フレーム 4
        self.assertEqual(sorted(df_ts['bead_name'].unique()), ['beads06um', 'beads1um'])
        np.testing.assert_allclose(
            df_ts['theta_nem_deg_relative'].abs().max(), 0.0, atol=1e-6)  # 時間変化なし（定数 zarr）
        df_exp = pd.read_csv(self.out_dir / 'mt_orientation_per_experiment.csv')
        for col in ['theta_nem_mean_deg', 'theta_nem_std_deg_continuous',
                    'theta_nem_range_deg', 'theta_nem_drift_deg', 'theta_nem_axis_order']:
            self.assertIn(col, df_exp.columns)
        self.assertTrue((df_exp['theta_nem_std_deg_continuous'] < 1e-6).all())
        self.assertTrue((df_exp['theta_nem_axis_order'] > 0.999).all())

        df_sum = pd.read_csv(self.out_dir / 'mt_orientation_summary.csv')
        self.assertIn('theta_nem_std_deg_continuous_mean', df_sum.columns)
        self.assertEqual(sorted(df_sum['bead_name']), ['beads06um', 'beads1um'])
        # 平行 / 反平行が半々なのでネマティック秩序は 1 に近く、極性秩序はほぼ 0
        self.assertTrue((df_sum['nematic_order_cos2_pooled'] > 0.99).all())
        self.assertTrue((df_sum['R_mean'] < 0.1).all())
        self.assertTrue((df_sum['mean_abs_delta_theta_deg_mean'] > 89.0).all())
        self.assertTrue((df_sum['mean_abs_delta_theta_deg_pooled'] > 89.0).all())
        self.assertTrue((df_sum['frac_parallel_within_tol_mean'] == 0.5).all())
        self.assertEqual(int(df_sum['n_experiments'].min()), 2)
        # zarr がフローと整合しているので符号補正は不要
        self.assertEqual(set(df_sum['theta_sign']), {1})

        df_hist = pd.read_csv(self.out_dir / 'mt_orientation_histogram.csv')
        for bead_name in ['beads06um', 'beads1um']:
            sub = df_hist[df_hist['bead_name'] == bead_name]
            widths = np.diff(np.linspace(-np.pi, np.pi, 19))
            self.assertAlmostEqual(float(np.sum(sub['mean_density'] * widths)), 1.0, places=6)
            # 0 のビン (index 9) と +-pi 側の端ビン (0 か 17) が最大 2 つ
            top = set(int(i) for i in sub['mean_density'].to_numpy().argsort()[::-1][:2])
            self.assertIn(9, top)
            self.assertTrue(top - {9} <= {0, 17})

        # 図は root_dir 側にも保存される
        for name in ['mt_orientation_histogram_panels', 'mt_orientation_radar_overlay']:
            self.assertTrue((self.root / 'figure' / 'mt_orientation' / f"{name}.png").exists(),
                            msg=f"{name}.png missing in root figure dir")

    def test_main_auto_corrects_mirrored_theta_convention(self):
        """theta zarr がミラー規約でも auto が符号を補正し、0 / pi にピークが現れる。"""
        self._build_synthetic_root(theta_sign_zarr=-1.0)
        out_dir = self.out_dir / 'mirror'
        argv = ['plot_mt_orientation_distribution.py',
                '--root_dir', str(self.root),
                '--beads', 'beads06um', 'beads1um',
                '--output_dir', str(out_dir),
                '--pixel_stride', '1', '--frame_stride', '1',
                '--bins', '18', '--bins_fold', '9',
                '--no_save_root', '--no_progress']
        with mock.patch.object(sys, 'argv', argv):
            mt_ori.main()
        plt.close('all')

        df_sum = pd.read_csv(out_dir / 'mt_orientation_summary.csv')
        self.assertEqual(set(df_sum['theta_sign']), {-1})
        self.assertTrue((df_sum['nematic_order_cos2_pooled'] > 0.99).all())

        df_exp = pd.read_csv(out_dir / 'mt_orientation_per_experiment.csv')
        # -theta 基準の方が <cos 2 dtheta> が大きい（= 0/pi ピーク）ことが記録されている
        self.assertTrue((df_exp['nematic_order_cos2_signminus']
                         > df_exp['nematic_order_cos2_signplus']).all())

        df_hist = pd.read_csv(out_dir / 'mt_orientation_histogram.csv')
        sub = df_hist[df_hist['bead_name'] == 'beads06um']
        top = set(int(i) for i in sub['mean_density'].to_numpy().argsort()[::-1][:2])
        self.assertIn(9, top)                 # Delta theta = 0
        self.assertTrue(top - {9} <= {0, 17})  # +-pi（左端 or 右端の同一方向ビン）


if __name__ == '__main__':
    unittest.main()

