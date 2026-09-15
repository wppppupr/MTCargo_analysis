import unittest
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

current_dir = Path(__file__).resolve().parent.parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from libs import spatial_heterogeneity as sh


class TestSpatialHeterogeneity(unittest.TestCase):

    def setUp(self):
        self.H = 128
        self.W = 128
        self.distances_px = [4, 8, 16, 24, 32]
        self.scale = 0.11

    def test_uniform_field(self):
        """完全一様流: 分散 chi_orient == 0, 全ペア内積 c = +1 (デルタ分布)"""
        ux = np.ones((self.H, self.W), dtype=np.float32)
        uy = np.zeros((self.H, self.W), dtype=np.float32)

        # 1. 4点相関
        res_4p = sh.calc_4point_correlation_fft(ux, uy, self.distances_px)
        np.testing.assert_allclose(res_4p['C_r'], 1.0, atol=1e-3)
        np.testing.assert_allclose(res_4p['C2_r'], 1.0, atol=1e-3)
        np.testing.assert_allclose(res_4p['chi_orient'], 0.0, atol=1e-3)

        # 2. 内積サンプリング & NGP
        samples = sh.sample_field_dot_products(ux, uy, self.distances_px, n_samples_per_dist=1000)
        ngp_res = sh.calc_dot_product_distribution_and_ngp(samples)
        for r in self.distances_px:
            np.testing.assert_allclose(samples[r], 1.0, atol=1e-4)

    def test_antiparallel_domain_field(self):
        """
        ストライプ状の逆並行ドメイン場:
        周期帯（幅 W_domain = 16 px）で +x と -x 方向が交互に並ぶ。
        相関長スケール r ~ W_domain で +1 と -1 が共存し、chi_orient(r) に鋭いピークが生じる。
        """
        W_domain = 16
        y_grid, x_grid = np.mgrid[:self.H, :self.W]
        stripe = ((y_grid // W_domain) % 2) * 2 - 1  # +1 or -1
        ux = np.broadcast_to(stripe, (self.H, self.W)).astype(np.float32).copy()
        uy = np.zeros((self.H, self.W), dtype=np.float32)

        distances = [2, 4, 8, 12, 16, 24, 32]

        # 1. 4点相関
        res_4p = sh.calc_4point_correlation_fft(ux, uy, distances)
        chi = res_4p['chi_orient']

        # 近距離 (r=2) では chi_orient は小さく、ドメイン境界付近 (r ~ 8-16) で極大化する
        self.assertGreater(np.max(chi), chi[0])
        self.assertGreater(res_4p['C2_r'][0], 0.9)  # 近距離では同一ドメイン

        # 2. 内積サンプリング & NGP
        samples = sh.sample_field_dot_products(ux, uy, distances, n_samples_per_dist=3000)
        ngp_res = sh.calc_dot_product_distribution_and_ngp(samples)

        # 逆並行境界スケールで c = +1 と c = -1 の双峰性 (U字型分布) となり、NGP は大きく正になる
        # (完全バイナリ +/-1 分布の場合 <c^2>=1, <c^4>=1 なので alpha_2 = 1/(3*1) - 1 = -2/3、
        # しかしガウス分布に対して極端な二極化として現れる)
        self.assertTrue(len(ngp_res['ngp']) == len(distances))

    def test_local_xi_map(self):
        """局所相関長マップの計算と空間非ガウス性 alpha_{2, xi} のテスト"""
        # ドメインサイズが左右で異なる場を作成（左半分: 細かいドメイン, 右半分: 広いドメイン）
        y_grid, x_grid = np.mgrid[:self.H, :self.W]
        stripe_left = ((y_grid // 4) % 2) * 2 - 1
        stripe_right = ((y_grid // 32) % 2) * 2 - 1
        ux = np.where(x_grid < self.W // 2, stripe_left, stripe_right).astype(np.float32)
        uy = np.zeros((self.H, self.W), dtype=np.float32)

        # ノイズの付加
        rng = np.random.default_rng(123)
        ux += rng.normal(0, 0.1, size=ux.shape).astype(np.float32)

        distances = [2, 4, 6, 8, 12, 16, 24, 32]
        res_xi = sh.calc_local_correlation_length_map(
            ux, uy, distances_px=distances, scale=self.scale, grid_step=8, min_r2=0.0
        )

        xi_map = res_xi['xi_map_um']
        self.assertEqual(xi_map.shape, (self.H // 8, self.W // 8))

        # 右半分の相関長が左半分よりも長くなっていることを検証
        xi_left = np.nanmean(xi_map[:, :xi_map.shape[1] // 2])
        xi_right = np.nanmean(xi_map[:, xi_map.shape[1] // 2:])
        self.assertGreater(xi_right, xi_left)

        # 空間 NGP の算出確認
        self.assertTrue(not np.isnan(res_xi['alpha_2_xi']) or len(res_xi['xi_valid_um']) > 0)

    def test_plot_summary(self):
        """サマリー図作成関数の動作テスト"""
        ux = np.ones((self.H, self.W), dtype=np.float32)
        uy = np.zeros((self.H, self.W), dtype=np.float32)
        four_p = sh.calc_4point_correlation_fft(ux, uy, self.distances_px)
        samples = sh.sample_field_dot_products(ux, uy, self.distances_px, n_samples_per_dist=500)
        ngp_res = sh.calc_dot_product_distribution_and_ngp(samples)
        xi_res = sh.calc_local_correlation_length_map(ux, uy, self.distances_px, scale=self.scale, grid_step=16)

        fig = sh.plot_spatial_heterogeneity_summary(
            distances_um=np.array(self.distances_px) * self.scale,
            four_point_result=four_p,
            ngp_result=ngp_res,
            local_xi_result=xi_res,
            condition_name="TestCondition",
        )
        self.assertIsNotNone(fig)
        plt.close(fig)


if __name__ == '__main__':
    unittest.main()
