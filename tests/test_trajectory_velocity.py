"""
tests/test_trajectory_velocity.py

速度推定手法 (Kalman/RTS, Smoothing Spline, 差分法) の単体テスト。
"""

import unittest
import numpy as np
import pandas as pd

from libs.trajectory_velocity import (
    KalmanRTSSmoother,
    SplineVelocityEstimator,
    FiniteDifferenceVelocityEstimator,
    estimate_particle_velocities,
)


def create_linear_motion(n=50, dt=0.1, vx=2.0, vy=1.5, noise=0.01):
    t = np.arange(n) * dt
    x = vx * t + np.random.normal(0, noise, n)
    y = vy * t + np.random.normal(0, noise, n)
    return t, x, y, vx, vy


class TestTrajectoryVelocity(unittest.TestCase):
    def test_kalman_rts_smoother(self):
        t, x, y, vx_true, vy_true = create_linear_motion(n=40, dt=0.1, vx=3.0, vy=-1.0, noise=0.02)
        smoother = KalmanRTSSmoother(process_noise_std=0.5, obs_noise_std=0.02)
        res = smoother.estimate(t, x, y)

        self.assertEqual(len(res.vx), len(t))
        self.assertEqual(len(res.vy), len(t))
        self.assertIsNotNone(res.vx_std)
        mid = slice(10, 30)
        np.testing.assert_allclose(res.vx[mid], vx_true, rtol=0.15, atol=0.3)
        np.testing.assert_allclose(res.vy[mid], vy_true, rtol=0.15, atol=0.3)

    def test_spline_estimator(self):
        t, x, y, vx_true, vy_true = create_linear_motion(n=40, dt=0.1, vx=-2.0, vy=2.5, noise=0.01)
        estimator = SplineVelocityEstimator(noise_std=0.01, k=3)
        res = estimator.estimate(t, x, y)

        self.assertEqual(len(res.vx), len(t))
        self.assertEqual(len(res.vy), len(t))
        mid = slice(10, 30)
        np.testing.assert_allclose(res.vx[mid], vx_true, rtol=0.15, atol=0.2)
        np.testing.assert_allclose(res.vy[mid], vy_true, rtol=0.15, atol=0.2)

    def test_moving_average_diff(self):
        t, x, y, vx_true, vy_true = create_linear_motion(n=40, dt=0.1, vx=1.0, vy=1.0, noise=0.01)
        estimator = FiniteDifferenceVelocityEstimator(window=3)
        res = estimator.estimate(t, x, y)

        self.assertEqual(len(res.vx), len(t))
        self.assertEqual(len(res.vy), len(t))
        self.assertEqual(res.params["window"], 3)

    def test_estimate_particle_velocities_df(self):
        t1, x1, y1, _, _ = create_linear_motion(n=30, dt=0.1, vx=1.0, vy=1.0)
        t2, x2, y2, _, _ = create_linear_motion(n=30, dt=0.1, vx=-1.0, vy=2.0)

        df1 = pd.DataFrame({'frame': np.arange(30), 'x': x1, 'y': y1, 'particle': 0})
        df2 = pd.DataFrame({'frame': np.arange(30), 'x': x2, 'y': y2, 'particle': 1})
        df_combined = pd.concat([df1, df2], ignore_index=True)

        # Kalman RTS で適用
        res_df = estimate_particle_velocities(df_combined, method='kalman_rts', frame_interval=0.1)
        self.assertIn('vx', res_df.columns)
        self.assertIn('vy', res_df.columns)
        self.assertIn('v', res_df.columns)
        self.assertIn('x_smooth', res_df.columns)
        self.assertFalse(res_df['v'].isna().all())

        # Spline で適用
        res_df_spline = estimate_particle_velocities(df_combined, method='spline', frame_interval=0.1)
        self.assertFalse(res_df_spline['v'].isna().all())


if __name__ == '__main__':
    unittest.main()
