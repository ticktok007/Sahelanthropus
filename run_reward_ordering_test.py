import sys
import unittest
import test_reward_ordering


def main():
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(test_reward_ordering)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
