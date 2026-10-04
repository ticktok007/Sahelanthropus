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
            expected_cycles = item.get("baseline_cycles")

            # Execute through QEMU wrapper
            live_cycles = float(self.env.compile_and_run(asm_code))
            live_score = -float(live_cycles)

            results[func_id] = {
                "live_cycles": live_cycles,
                "live_score": live_score,
                "expected_cycles": expected_cycles
            }

            self.assertGreater(live_cycles, 0, f"Cycles should be positive for {func_id}")

            if expected_cycles is not None:
                expected_cycles = float(expected_cycles)
                self.assertEqual(live_cycles, expected_cycles,
                    f"Mismatch for {func_id}: expected {expected_cycles}, got {live_cycles}")

            print(f"{func_id:<25} | {str(expected_cycles):<15} | {live_cycles:<12.1f} | {live_score:<10.1f} | PASS")

        print("=" * 70)

        # Relative Ordering Checks (if test items present)
        if "func_exit_minimal" in results and "func_arithmetic_chain" in results:
            fast_cycles = results["func_exit_minimal"]["live_cycles"]
            slow_cycles = results["func_arithmetic_chain"]["live_cycles"]
            fast_score  = results["func_exit_minimal"]["live_score"]
            slow_score  = results["func_arithmetic_chain"]["live_score"]

            self.assertLess(fast_cycles, slow_cycles, "Minimal exit should take fewer cycles than arithmetic chain")
            self.assertGreater(fast_score, slow_score, "Minimal exit should receive a higher score than arithmetic chain")

            print(f"\n[+] Relative Ordering Verified:")
            print(f"    exit_minimal cycles ({fast_cycles}) < arithmetic_chain cycles ({slow_cycles})")
            print(f"    exit_minimal score ({fast_score}) > arithmetic_chain score ({slow_score})")


if __name__ == "__main__":
    unittest.main(verbosity=2)
