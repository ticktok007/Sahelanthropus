#!/usr/bin/env python3
"""
test_superopt_env_graph_obs.py — Verification of SuperoptEnv returning HeteroData graph objects.
"""

import unittest
from torch_geometric.data import HeteroData
from superopt_env import SuperoptEnv


class TestSuperoptEnvGraphObs(unittest.TestCase):

    def test_reset_returns_hetero_data(self):
        """Test reset() returns HeteroData graph observation when use_graph_obs=True."""
        env = SuperoptEnv(corpus_path="corpus.json", use_graph_obs=True)
        obs, info = env.reset(seed=42)

        self.assertIsInstance(obs, HeteroData)
        self.assertIn("inst", obs.node_types)
        self.assertIn("reg", obs.node_types)
        self.assertEqual(obs["inst"].x.shape, (16, 177))
        self.assertEqual(obs["reg"].x.shape, (32, 1))

    def test_step_returns_hetero_data(self):
        """Test step() returns HeteroData graph observation when use_graph_obs=True."""
        env = SuperoptEnv(corpus_path="corpus.json", use_graph_obs=True)
        obs_init, info = env.reset(seed=42)

        action = 0  # Apply rule 0 at slot 0
        obs_next, reward, terminated, truncated, step_info = env.step(action)

        self.assertIsInstance(obs_next, HeteroData)
        self.assertIn("inst", obs_next.node_types)
        self.assertEqual(obs_next["inst"].x.shape, (16, 177))

    def test_fallback_flat_obs(self):
        """Test reset() returns flat numpy array when use_graph_obs=False."""
        env = SuperoptEnv(corpus_path="corpus.json", use_graph_obs=False)
        obs, info = env.reset(seed=42)

        self.assertNotIsInstance(obs, HeteroData)
        self.assertEqual(obs.shape, (80,))


if __name__ == "__main__":
    unittest.main(verbosity=2)
