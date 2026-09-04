"""
scratch/test_hmm.py
HMM 2変数モデルの単体テスト・検証用スクリプト
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# 親ディレクトリ追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from libs.hmm_cargo import extract_hmm_features, CargoGaussianHMM, calc_state_dwell_times, fit_exponential_distribution


def generate_synthetic_tracks(n_particles=10, n_steps=200, dt=4.0):
    """
    State 0: Tumble (low speed ~0.05 um/s, random direction)
    State 1: Run (high speed ~0.8 um/s, persistent direction)
    """
    np.random.seed(42)
    records = []

    # 遷移確率
    # A = [[0.8, 0.2], [0.15, 0.85]]
    trans_matrix = np.array([
        [0.80, 0.20],
        [0.15, 0.85]
    ])

    for p in range(n_particles):
        state = 0 if np.random.rand() < 0.5 else 1
        x, y = 0.0, 0.0
        angle = np.random.uniform(-np.pi, np.pi)

        for step in range(n_steps):
            records.append({
                'particle': p,
                'frame': step,
                'x': x / 0.11,  # pixel
                'y': y / 0.11,
                'true_state': state,
            })

            # 次の状態へ遷移
            state = np.random.choice([0, 1], p=trans_matrix[state])

            if state == 0:
                # Tumble: 低速・方向転換大
                v = np.random.lognormal(mean=np.log(0.05), sigma=0.4)
                dth = np.random.uniform(-np.pi, np.pi)
            else:
                # Run: 高速・直進性高 (dth は 0 付近)
                v = np.random.lognormal(mean=np.log(0.8), sigma=0.3)
                dth = np.random.normal(loc=0.0, scale=0.2)

            angle += dth
            angle = np.arctan2(np.sin(angle), np.cos(angle))
            x += v * dt * np.cos(angle)
            y += v * dt * np.sin(angle)

    return pd.DataFrame(records)


def main():
    print("=== Generating synthetic tracks ===")
    df_synthetic = generate_synthetic_tracks(n_particles=15, n_steps=250, dt=4.0)
    print(f"Total points: {len(df_synthetic)}")

    print("\n=== Extracting HMM features [ln(v+eps), cos(dth)] ===")
    X, lengths, df_obs = extract_hmm_features(df_synthetic, tau=1, scale=0.11, frame_interval=4.0, epsilon=1e-3)
    print(f"Observation matrix shape: {X.shape}, Sequences count: {len(lengths)}")
    print(df_obs.head())

    print("\n=== Fitting CargoGaussianHMM (K=2) ===")
    hmm_model = CargoGaussianHMM(n_components=2, covariance_type='full', random_state=42)
    hmm_model.fit(X, lengths)

    print("\n=== Transition Matrix A ===")
    print(hmm_model.model.transmat_)

    print("\n=== Means ===")
    print("State 0 (Tumble) [ln(v+eps), cos(dth)]:", hmm_model.model.means_[0])
    print("State 1 (Run)    [ln(v+eps), cos(dth)]:", hmm_model.model.means_[1])

    print("\n=== State Summary ===")
    summary = hmm_model.get_state_summary(frame_interval=4.0)
    print(summary)

    print("\n=== BIC / AIC ===")
    bic, aic = hmm_model.compute_bic_aic(X, lengths)
    print(f"BIC: {bic:.2f}, AIC: {aic:.2f}")

    print("\n=== Decoding States ===")
    pred_states = hmm_model.predict(X, lengths)
    df_obs['pred_state'] = pred_states
    print("Predicted state counts:\n", pd.Series(pred_states).value_counts())

    print("\n=== Dwell Times ===")
    dwells = calc_state_dwell_times(pred_states, lengths, frame_interval=4.0)
    for s, times in dwells.items():
        tau_mle, tau_err, r2 = fit_exponential_distribution(times)
        print(f"State {s} ({summary.loc[s, 'label']}): N_events={len(times)}, mean_dwell={tau_mle:.2f} s (+/- {tau_err:.2f}), R2={r2:.3f}")

    print("\n[SUCCESS] HMM test completed successfully!")


if __name__ == '__main__':
    main()
