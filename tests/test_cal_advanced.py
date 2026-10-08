"""
tests/test_cal_advanced.py

libs/cal_vel.py の cal_advanced のテスト。
"""

import unittest
import numpy as np
import pandas as pd
from libs.cal_vel import cal_advanced


class TestCalAdvanced(unittest.TestCase):
    def setUp(self):
        n = 30
        t = np.arange(n) * 0.1
        self.df = pd.DataFrame({
            'frame': np.arange(n),
            'particle': 0,
            'x': 2.0 * t + np.random.normal(0, 0.01, n),
            'y': 1.0 * t + np.random.normal(0, 0.01, n),
        })

    def test_cal_advanced_kalman(self):
        res = cal_advanced(self.df, method='kalman_rts', frame_interval=0.1, scale=1.0)
        self.assertIn('vx', res.columns)
        self.assertIn('v', res.columns)
        self.assertIn('omega', res.columns)
        self.assertEqual(len(res), len(self.df))

    def test_cal_default_raw(self):
        from libs.cal_vel import cal
        res = cal(self.df.copy(), scale=1.0, frame_interval=0.1)
        self.assertIn('vx', res.columns)
        self.assertIn('vy', res.columns)
        self.assertIn('v', res.columns)
        self.assertIn('theta', res.columns)
        self.assertIn('dtheta', res.columns)
        self.assertIn('omega', res.columns)
        self.assertEqual(len(res), len(self.df))

    def test_cal_spline(self):
        from libs.cal_vel import cal
        res = cal(self.df.copy(), scale=1.0, frame_interval=0.1, smooth_method='spline')
        self.assertIn('vx', res.columns)
        self.assertIn('vy', res.columns)
        self.assertIn('v', res.columns)
        self.assertIn('theta', res.columns)
        self.assertIn('dtheta', res.columns)
        self.assertIn('omega', res.columns)
        self.assertIn('x_smooth', res.columns)
        self.assertEqual(len(res), len(self.df))
        mean_v = res['v'].mean()
        self.assertAlmostEqual(mean_v, np.sqrt(5.0), delta=0.2)

    def test_cal_moving_average(self):
        from libs.cal_vel import cal
        res = cal(self.df.copy(), scale=1.0, frame_interval=0.1, pos_window=3)
        self.assertIn('v', res.columns)
        self.assertIn('vx', res.columns)
        self.assertIn('x_s', res.columns)


if __name__ == '__main__':
    unittest.main()
