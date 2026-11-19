"""Unit tests for libs/gmm_cargo.py."""

import unittest
import numpy as np
import pandas as pd

from libs import gmm_cargo as gc


class TestGmmCargo(unittest.TestCase):
    def test_extract_gmm_features_normalizes_by_time_varying_uinf_and_splits_gaps(self):
        # Two contiguous segments for particle 1 in expA: frames [0,1,2] and [5,6]
        df_tracks = pd.DataFrame({
            "exp_dir": ["expA"] * 5,
            "particle": [1] * 5,
            "frame": [0, 1, 2, 5, 6],
            "x": [0.0, 10.0, 20.0, 30.0, 31.0],
            "y": [0.0, 0.0, 0.0, 0.0, 0.0],
        })
        # u_inf(t) doubles at frame 1 so normalized speed halves between frame 0 and 1
        df_uinf = pd.DataFrame({
            "exp_dir": ["expA"] * 6,
            "frame": [0, 1, 2, 5, 6, 7],
            "u_inf_t": [0.275, 0.550, 0.275, 0.275, 0.275, 0.275],
        })

        X, lengths, df_obs = gc.extract_gmm_features(
            df_tracks, df_uinf, tau=1, scale=0.11, frame_interval=4.0, epsilon=0.0
        )

        self.assertEqual(lengths, [2, 1])
        self.assertEqual(X.shape, (3, 1))
        np.testing.assert_allclose(df_obs["v_norm"].to_numpy(), [1.0, 0.5, 0.1], rtol=1e-6)
        np.testing.assert_allclose(X[:, 0], np.log([1.0, 0.5, 0.1]), rtol=1e-6)

    def test_cargo_gaussian_gmm_sorts_tumble_and_run_and_boundary(self):
        rng = np.random.default_rng(123)
        x_tumble = rng.normal(loc=-2.5, scale=0.4, size=(400, 1))
        x_run = rng.normal(loc=-0.1, scale=0.3, size=(200, 1))
        X = np.vstack([x_run, x_tumble])  # intentionally Run first

        gmm = gc.CargoGaussianGMM(n_components=2, random_state=42).fit(X)
        summary = gmm.get_state_summary()

        self.assertEqual(list(summary["state"]), [0, 1])
        self.assertEqual(list(summary["label"]), ["Tumble", "Run"])
        self.assertLess(gmm.means_[0, 0], gmm.means_[1, 0])
        self.assertGreater(gmm.get_separation_index(), 4.0)

        proba = gmm.predict_proba(X)
        pred = gmm.predict(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)
        np.testing.assert_array_equal(pred, np.argmax(proba, axis=1))

        boundary = gmm.get_decision_boundary()
        self.assertIsNotNone(boundary)
        p_at_boundary = gmm.predict_proba(np.array([[boundary]]))
        np.testing.assert_allclose(p_at_boundary[0, 1], 0.5, atol=1e-4)

        bic, aic, icl = gmm.compute_bic_aic_icl(X)
        self.assertTrue(np.isfinite(bic) and np.isfinite(aic) and icl >= bic)

    def test_autocorrelation_respects_exp_dir_and_frame_gaps(self):
        df_obs = pd.DataFrame({
            "exp_dir": ["e1"] * 4 + ["e2"] * 4,
            "particle": [0] * 4 + [0] * 4,
            "frame": [0, 1, 2, 3, 0, 1, 5, 6],
            "dx_um": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0],
            "dy_um": [0.0] * 8,
            "v_norm": [1.0] * 8,
            "pred_state": [1] * 8,
        })

        ac = gc.calc_state_dependent_autocorrelations(df_obs, n_components=2, frame_interval=4.0, max_lag_frames=2)
        oacf = ac[1]["oacf"]
        lag1_corr = float(oacf.loc[oacf["lag_frames"] == 1, "corr"].iloc[0])
        self.assertAlmostEqual(lag1_corr, 1.0, places=9)


if __name__ == "__main__":
    unittest.main()
