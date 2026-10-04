#!/usr/bin/env python3
"""
test_superopt_step.py — Unit Test for SuperoptEnv.step(action) & info["action_mask"]
"""

import unittest
import numpy as np
from superopt_env import SuperoptEnv


class TestSuperoptStep(unittest.TestCase):

    def setUp(self):
        self.env = SuperoptEnv(corpus_path="corpus.json", max_len=16, num_rules=5, use_reward_shaping=False)

    def test_reset_info_action_mask(self):
        """Tests that reset() populates info['action_mask'] correctly."""
        obs, info = self.env.reset(seed=42)
        self.assertIn("action_mask", info)
        self.assertIsInstance(info["action_mask"], np.ndarray)
        self.assertEqual(info["action_mask"].shape, (80,))  # num_rules(5) * max_len(16)

    def test_step_valid_rewrite_rule0(self):
        """Tests applying Rule 0 (mul -> slli) and verifying info['action_mask']."""
        obs, info = self.env.reset(seed=42)
        
        # Override program with (li t0, 42), (li t1, 8), (mul t2, t0, t1)
        self.env.current_program = [
            [1, 0, 0, 5, 42],  # li t0, 42
            [1, 0, 0, 6, 8],   # li t1, 8
            [5, 5, 6, 7, 0],   # mul t2, t0, t1
            [16, 0, 0, 0, 0]   # ecall
        ] + [[0, 0, 0, 0, 0]] * 12

        # Check action mask before step
        action_mask_before = self.env.get_action_mask()
        self.assertTrue(action_mask_before[2], "Rule 0 at slot 2 should be True in per-slot action_mask!")

        # Action: Rule 0 on target_idx 2 -> action = 0 * 16 + 2 = 2
        action = 2
        obs_next, reward, terminated, truncated, step_info = self.env.step(action)

        print(f"\n[+] Valid Rule 0 Step Test: Applied = {step_info['applied']}, Reward = {reward:+.4f}")
        print(f"    action_mask in step_info: {step_info['action_mask']}")
        
        self.assertTrue(step_info['applied'])
        self.assertIn("action_mask", step_info)
        self.assertEqual(step_info["action_mask"].shape, (80,))  # num_rules(5) * max_len(16)

    def test_step_unmatched_pattern(self):
        """Tests applying an unmatched action pattern returns reward = 0.0."""
        obs_init, info = self.env.reset(seed=42)
        action = 15  # Target index 15 with no pattern match
        obs_next, reward, terminated, truncated, step_info = self.env.step(action)

        self.assertFalse(step_info['applied'])
        self.assertEqual(reward, 0.0)
        self.assertIn("action_mask", step_info)


if __name__ == "__main__":
    unittest.main(verbosity=2)
