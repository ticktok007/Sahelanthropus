#!/usr/bin/env python3
"""
test_corpus_sampler_curriculum.py — Verify that SuperoptEnv corpus sampler
only samples functions with len(instrs) <= current max_len.
"""

from superopt_env import SuperoptEnv


def test_corpus_sampler_filtering():
    print("=" * 70)
    print(" TESTING CORPUS SAMPLER CURRICULUM FILTERING (len(instrs) <= max_len)")
    print("=" * 70)

    # Synthetic corpus with varying instruction lengths: 4, 8, 12, 16, 24
    test_corpus = [
        {"id": "func_4", "name": "f4", "assembly": "li t0, 1\nli t1, 2\nadd t2, t0, t1\nret\n"},
        {"id": "func_8", "name": "f8", "assembly": "li t0, 1\nli t1, 2\nadd t2, t0, t1\nsub t3, t2, t0\nmul t4, t3, t1\nadd t5, t4, t2\nsub t6, t5, t1\nret\n"},
        {"id": "func_16", "name": "f16", "assembly": "\n".join([f"addi t0, t0, {i}" for i in range(16)])},
        {"id": "func_25", "name": "f25", "assembly": "\n".join([f"addi t0, t0, {i}" for i in range(25)])},
    ]

    env = SuperoptEnv(max_len=8)
    env.corpus = test_corpus

    # Test sampling at max_len = 8 (only func_4 and func_8 are valid)
    sampled_lengths_8 = []
    for s in range(50):
        obs, info = env.reset(seed=s)
        n_insts = env._count_instructions(env.current_func_meta)
        sampled_lengths_8.append(n_insts)
        assert n_insts <= 8, f"Sampled function length {n_insts} exceeds max_len=8!"

    print(f" ✓ [max_len = 8] Tested 50 resets: All sampled lengths <= 8 (Sampled: {set(sampled_lengths_8)})")

    # Expand max_len to 16
    env.update_curriculum(250000)  # max_len = 16
    sampled_lengths_16 = []
    for s in range(50):
        obs, info = env.reset(seed=s)
        n_insts = env._count_instructions(env.current_func_meta)
        sampled_lengths_16.append(n_insts)
        assert n_insts <= 16, f"Sampled function length {n_insts} exceeds max_len=16!"

    print(f" ✓ [max_len = 16] Tested 50 resets: All sampled lengths <= 16 (Sampled: {set(sampled_lengths_16)})")

    # Expand max_len to 25
    env.update_curriculum(500000)  # max_len = 25
    sampled_lengths_25 = []
    for s in range(50):
        obs, info = env.reset(seed=s)
        n_insts = env._count_instructions(env.current_func_meta)
        sampled_lengths_25.append(n_insts)
        assert n_insts <= 25, f"Sampled function length {n_insts} exceeds max_len=25!"

    print(f" ✓ [max_len = 25] Tested 50 resets: All sampled lengths <= 25 (Sampled: {set(sampled_lengths_25)})")

    print("=" * 70)
    print(" ALL CORPUS SAMPLER CURRICULUM FILTERING TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    test_corpus_sampler_filtering()
