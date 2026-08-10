import json
import unittest
from pathlib import Path
from reward_env import RewardEnv, ToolchainError


class TestEndToEndCorpus(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        corpus_path = Path("corpus.json").resolve()
        if not corpus_path.exists():
            raise unittest.SkipTest(f"Corpus file '{corpus_path}' not found.")
        
        with open(corpus_path, "r") as f:
            cls.corpus = json.load(f)

        try:
            cls.env = RewardEnv(strict=True)
        except ToolchainError as exc:
            raise unittest.SkipTest(f"Required toolchain unavailable: {exc}") from exc

    def test_load_corpus_and_verify_cycles(self):
        """
        End-to-End Test: Load functions from corpus.json -> execute through RewardEnv QEMU wrapper
        -> verify live baseline cycles & scores match precomputed values deterministically.
        """
        print("\n" + "=" * 70)
        print(" End-to-End Test: Corpus Loading -> QEMU Execution -> Cycle Verification")
        print("=" * 70)
        print(f"{'Function ID':<25} | {'Expected Cycles':<15} | {'Live Cycles':<12} | {'Score':<10} | Status")
        print("-" * 70)

        results = {}

        for item in self.corpus:
            func_id = item["id"]
            asm_code = item["assembly"]
            expected_cycles = float(item["baseline_cycles"])
            expected_score = float(item["baseline_score"])

            # Execute through QEMU wrapper
            live_cycles = float(self.env.compile_and_run(asm_code))
            live_score = -float(live_cycles)

            results[func_id] = {
                "live_cycles": live_cycles,
                "live_score": live_score,
                "expected_cycles": expected_cycles
            }

            # Verification assertions
            self.assertEqual(live_cycles, expected_cycles,
                f"Mismatch for {func_id}: expected {expected_cycles}, got {live_cycles}")
            self.assertEqual(live_score, expected_score,
                f"Score mismatch for {func_id}: expected {expected_score}, got {live_score}")

            print(f"{func_id:<25} | {expected_cycles:<15.1f} | {live_cycles:<12.1f} | {live_score:<10.1f} | PASS")

        print("=" * 70)

        # Relative Ordering Checks
        fast_cycles = results["func_exit_minimal"]["live_cycles"]
        slow_cycles = results["func_arithmetic_chain"]["live_cycles"]
        fast_score  = results["func_exit_minimal"]["live_score"]
        slow_score  = results["func_arithmetic_chain"]["live_score"]

        self.assertLess(fast_cycles, slow_cycles, "Minimal exit should take fewer cycles than arithmetic chain")
        self.assertGreater(fast_score, slow_score, "Minimal exit should receive a higher score than arithmetic chain")

        loop10_cycles  = results["func_simple_loop_10"]["live_cycles"]
        loop100_cycles = results["func_simple_loop_100"]["live_cycles"]
        self.assertLess(loop10_cycles, loop100_cycles, "10-iter loop should take fewer cycles than 100-iter loop")

        print(f"\n[+] Relative Ordering Verified:")
        print(f"    exit_minimal cycles ({fast_cycles}) < arithmetic_chain cycles ({slow_cycles})")
        print(f"    exit_minimal score ({fast_score}) > arithmetic_chain score ({slow_score})")
        print(f"    loop_10 cycles ({loop10_cycles}) < loop_100 cycles ({loop100_cycles})")


if __name__ == "__main__":
    unittest.main(verbosity=2)
