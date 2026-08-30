#!/usr/bin/env python3
"""
curriculum_scheduler.py — Curriculum Scheduler for RISC-V Assembly Superoptimization.
Formula:
    max_len = min(8 + floor(17 * step / 500000), 25)
"""

import math
from typing import Tuple


def compute_curriculum_max_len(step: int) -> int:
    """
    Computes curriculum maximum sequence length max_len for global environment step t.
    Formula:
        max_len = min(8 + floor(17 * step / 500000), 25)
    
    Step Milestones:
        step = 0        -> max_len = 8
        step = 100,000  -> max_len = 11
        step = 250,000  -> max_len = 16
        step = 500,000  -> max_len = 25
        step > 500,000  -> max_len = 25
    """
    val = 8 + math.floor(17.0 * float(step) / 500000.0)
    return min(int(val), 25)


class CurriculumScheduler:
    """
    Tracks training progress and manages max_len updates for environment instances.
    """

    def __init__(self, start_step: int = 0):
        self.current_step = start_step
        self.current_max_len = compute_curriculum_max_len(start_step)

    def update(self, global_step: int) -> Tuple[int, bool]:
        """
        Updates global step and returns (new_max_len, is_updated_bool).
        """
        self.current_step = global_step
        new_max_len = compute_curriculum_max_len(global_step)
        changed = (new_max_len != self.current_max_len)
        if changed:
            self.current_max_len = new_max_len
        return new_max_len, changed


if __name__ == "__main__":
    print("=" * 70)
    print(" TESTING CURRICULUM SCHEDULER: max_len = min(8 + floor(17*step/500000), 25)")
    print("=" * 70)

    test_steps = [0, 50000, 100000, 200000, 250000, 350000, 500000, 750000, 1000000]
    for s in test_steps:
        m_len = compute_curriculum_max_len(s)
        print(f" Step {s:>9,d} | max_len = {m_len:>2d}")

    assert compute_curriculum_max_len(0) == 8, "Expected max_len=8 at step 0"
    assert compute_curriculum_max_len(250000) == 16, "Expected max_len=16 at step 250,000"
    assert compute_curriculum_max_len(500000) == 25, "Expected max_len=25 at step 500,000"
    assert compute_curriculum_max_len(1000000) == 25, "Expected max_len=25 at step 1,000,000"

    print("=" * 70)
    print(" ALL CURRICULUM SCHEDULER CHECKS PASSED SUCCESSFULLY!")
    print("=" * 70)
