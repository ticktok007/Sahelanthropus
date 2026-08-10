import time
import statistics
from reward_env import RewardEnv

ASM_TEST = """\
.section .text
.globl _start

_start:
    li a0, 0
    li a7, 93
    ecall
"""

def benchmark_step():
    env = RewardEnv(strict=True)
    env.reset()

    latencies = []
    # Warmup
    env.step(ASM_TEST)

    for i in range(20):
        t0 = time.perf_counter()
        obs, reward, done, info = env.step(ASM_TEST)
        t1 = time.perf_counter()
        lat_ms = (t1 - t0) * 1000.0
        latencies.append(lat_ms)

    avg_ms = statistics.mean(latencies)
    min_ms = min(latencies)
    max_ms = max(latencies)

    print(f"RewardEnv.step() Latency Over 20 Runs:")
    print(f"  Average : {avg_ms:.2f} ms")
    print(f"  Min     : {min_ms:.2f} ms")
    print(f"  Max     : {max_ms:.2f} ms")

    if avg_ms < 50.0:
        print("[+] SUCCESS: RewardEnv.step() takes < 50ms!")
    else:
        print("[-] WARNING: RewardEnv.step() takes >= 50ms, optimization needed.")

if __name__ == "__main__":
    benchmark_step()
