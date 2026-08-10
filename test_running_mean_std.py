#!/usr/bin/env python3
"""
test_running_mean_std.py — Unit Test & Verification of RunningMeanStd against Gymnasium Wrapper

Verifies:
1. Running mean and running variance update accuracy against gymnasium's RunningMeanStd.
2. Normalized observation output match tolerance (atol=1e-5).
"""

import unittest
import numpy as np
import gymnasium as gym

from running_mean_std import RunningMeanStd as CustomRunningMeanStd

# Import Gymnasium's official RunningMeanStd
try:
    from gymnasium.wrappers.utils import RunningMeanStd as GymRunningMeanStd
except ImportError:
    from gymnasium.wrappers.vector.stateful_observation import RunningMeanStd as GymRunningMeanStd


class TestRunningMeanStd(unittest.TestCase):

    def setUp(self):
        self.obs_dim = 80
        self.custom_rms = CustomRunningMeanStd(shape=(self.obs_dim,))
        self.gym_rms = GymRunningMeanStd(shape=(self.obs_dim,))

    def test_running_mean_std_verification(self):
        """Feeds identical observation batches to custom and gymnasium RunningMeanStd and verifies outputs match."""
        np.random.seed(42)

        # Generate 10 batches of observations (batch_size=16, obs_dim=80)
        for i in range(10):
            batch_obs = np.random.randn(16, self.obs_dim).astype(np.float64) * (i + 1) + (i * 2.5)

            # Update custom RMS
            self.custom_rms.update(batch_obs)

            # Update Gymnasium RMS
            self.gym_rms.update(batch_obs)

            # Compare running mean and variance
            mean_diff = np.max(np.abs(self.custom_rms.mean - self.gym_rms.mean))
            var_diff = np.max(np.abs(self.custom_rms.var - self.gym_rms.var))

            print(f" Batch {i+1:2d} Update | Mean Diff: {mean_diff:.8e} | Var Diff: {var_diff:.8e}")
            self.assertTrue(np.allclose(self.custom_rms.mean, self.gym_rms.mean, atol=1e-5))
            self.assertTrue(np.allclose(self.custom_rms.var, self.gym_rms.var, atol=1e-5))

        # Test observation normalization output
        sample_obs = np.random.randn(4, self.obs_dim).astype(np.float32)
        custom_norm = self.custom_rms.normalize(sample_obs)
        gym_norm = (sample_obs - self.gym_rms.mean) / np.sqrt(self.gym_rms.var + 1e-8)
        gym_norm = np.clip(gym_norm, -10.0, 10.0).astype(np.float32)

        norm_diff = np.max(np.abs(custom_norm - gym_norm))
        print(f"\n[+] Normalized Observation Difference vs Gymnasium: {norm_diff:.8e}")
        self.assertTrue(np.allclose(custom_norm, gym_norm, atol=1e-5))


if __name__ == "__main__":
    unittest.main(verbosity=2)
