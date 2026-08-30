#!/usr/bin/env python3
"""
utils_seed.py — Unified Reproducibility & Deterministic Seeding Manager.
Identically seeds:
    1. Python built-in random
    2. OS Hash Seed (PYTHONHASHSEED)
    3. NumPy random generator (np.random)
    4. PyTorch CPU manual seed (torch.manual_seed)
    5. PyTorch GPU / CUDA manual seeds (torch.cuda.manual_seed, torch.cuda.manual_seed_all)
    6. PyTorch CUDNN deterministic backend flags (cudnn.deterministic, cudnn.benchmark)
    7. Gymnasium Environment & Action/Observation Spaces (if env is provided)
"""

import os
import random
from typing import Optional, Any
import numpy as np
import torch
import gymnasium as gym


def set_seed(seed: int = 42, env: Optional[Any] = None) -> int:
    """
    Sets deterministic random seed identically across random, numpy, PyTorch (CPU/CUDA),
    CUDNN backends, and Gymnasium environment action/observation spaces.

    Args:
        seed: Integer seed value (default: 42)
        env: Optional Gymnasium environment instance to seed

    Returns:
        The integer seed applied.
    """
    seed = int(seed)

    # 1. Python built-in random & OS hash seed
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # 2. NumPy random generator
    np.random.seed(seed)

    # 3. PyTorch CPU & CUDA manual seeds
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # 4. PyTorch CUDNN deterministic backend flags
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # 5. Gymnasium Environment & Action Space Seeding
    if env is not None:
        if hasattr(env, "reset"):
            try:
                env.reset(seed=seed)
            except Exception:
                pass
        if hasattr(env, "action_space") and hasattr(env.action_space, "seed"):
            env.action_space.seed(seed)
        if hasattr(env, "observation_space") and hasattr(env.observation_space, "seed"):
            env.observation_space.seed(seed)

    print(f" [+] Unified Reproducibility Seed set to: {seed} (random, numpy, torch, cuda, cudnn, env)")
    return seed


if __name__ == "__main__":
    print("=" * 70)
    print(" TESTING UNIFIED SET_SEED UTILITY")
    print("=" * 70)

    from superopt_env import SuperoptEnv
    test_env = SuperoptEnv()

    applied_seed = set_seed(1234, env=test_env)

    # Verify reproducibility across components
    val_py = random.randint(0, 100000)
    val_np = np.random.randint(0, 100000)
    val_torch = torch.randint(0, 100000, (1,)).item()
    if torch.cuda.is_available():
        val_cuda = torch.randint(0, 100000, (1,), device="cuda").item()
    else:
        val_cuda = None

    print(f" Python Random Sample : {val_py}")
    print(f" NumPy Random Sample  : {val_np}")
    print(f" PyTorch CPU Sample   : {val_torch}")
    print(f" PyTorch CUDA Sample  : {val_cuda}")

    # Re-seed and verify identical values
    set_seed(1234, env=test_env)

    assert random.randint(0, 100000) == val_py, "Python random seed non-reproducible!"
    assert np.random.randint(0, 100000) == val_np, "NumPy random seed non-reproducible!"
    assert torch.randint(0, 100000, (1,)).item() == val_torch, "PyTorch CPU random seed non-reproducible!"
    if torch.cuda.is_available():
        assert torch.randint(0, 100000, (1,), device="cuda").item() == val_cuda, "PyTorch CUDA random seed non-reproducible!"

    print("=" * 70)
    print(" ALL UNIFIED SET_SEED REPRODUCIBILITY CHECKS PASSED SUCCESSFULLY!")
    print("=" * 70)
