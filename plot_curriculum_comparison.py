#!/usr/bin/env python3
"""
plot_curriculum_comparison.py — Generate comparative learning curve plots
for Curriculum vs. No-Curriculum strategies and calculate convergence speedup.
"""

import os
import csv
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-darkgrid" if "seaborn-v0_8-darkgrid" in plt.style.available else "default")


def read_csv(path):
    steps, max_lens, rewards, expl_vars, rules = [], [], [], [], []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row["global_step"]))
            max_lens.append(int(row.get("max_len", 25)))
            rewards.append(float(row["episode_reward_mean"]))
            expl_vars.append(float(row["explained_variance"]))
            rules.append(int(row["rules_applied"]))
    return steps, max_lens, rewards, expl_vars, rules


def main():
    c_path = "curriculum_curve.csv"
    nc_path = "nocurriculum_curve.csv"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    if os.path.exists(c_path) and os.path.exists(nc_path):
        c_steps, c_lens, c_rew, c_exp, c_rules = read_csv(c_path)
        nc_steps, nc_lens, nc_rew, nc_exp, nc_rules = read_csv(nc_path)

        # Plot 1: Rules Applied (Cumulative Verified Rewrites)
        axes[0].plot(c_steps, c_rules, label="Curriculum Strategy (max_len 8 -> 25)", color="#2ca02c", linewidth=2.5)
        axes[0].plot(nc_steps, nc_rules, label="No-Curriculum Strategy (Fixed max_len 25)", color="#d62728", linewidth=2.0, linestyle="--")
        axes[0].set_title("Z3-Verified Rewrites Applied (Cumulative Convergence)", fontsize=12, fontweight="bold")
        axes[0].set_xlabel("Environment Steps", fontsize=10)
        axes[0].set_ylabel("Verified Rules Applied", fontsize=10)
        axes[0].legend(loc="upper left", frameon=True)
        axes[0].grid(True, linestyle="--", alpha=0.6)

        # Plot 2: Critic Explained Variance Fit
        axes[1].plot(c_steps, c_exp, label="Curriculum Strategy", color="#2ca02c", linewidth=2.5)
        axes[1].plot(nc_steps, nc_exp, label="No-Curriculum Strategy", color="#d62728", linewidth=2.0, linestyle="--")
        axes[1].set_title("Critic Value Function Explained Variance", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("Environment Steps", fontsize=10)
        axes[1].set_ylabel("Explained Variance", fontsize=10)
        axes[1].legend(loc="lower right", frameon=True)
        axes[1].grid(True, linestyle="--", alpha=0.6)

        plt.tight_layout()
        out_png = "curriculum_vs_nocurriculum.png"
        plt.savefig(out_png, dpi=300)
        print(f"[+] Saved comparative plot to {out_png}")

        # Compute milestone convergence speedup
        # Find step when first rule was accepted in Curriculum vs No-Curriculum
        c_first = next((s for s, r in zip(c_steps, c_rules) if r > 0), None)
        nc_first = next((s for s, r in zip(nc_steps, nc_rules) if r > 0), None)

        print("\n" + "=" * 70)
        print(" CONVERGENCE SPEEDUP ANALYSIS: CURRICULUM VS NO-CURRICULUM")
        print("=" * 70)
        print(f" First Verified Rule Accepted (Curriculum)    : {f'{c_first:,d} steps' if c_first else 'N/A'}")
        print(f" First Verified Rule Accepted (No-Curriculum) : {f'{nc_first:,d} steps' if nc_first else 'N/A'}")

        if c_first and nc_first:
            speedup = ((nc_first - c_first) / nc_first) * 100.0
            ratio = nc_first / max(1, c_first)
            print(f" Convergence Speedup                           : {speedup:+.1f}% faster ({ratio:.2f}x speedup)")
            print(f" Hypothesis Status (Target >= 30% Speedup)     : {'✅ CONFIRMED' if speedup >= 30.0 else '🟡 Below 30%'}")
        print("=" * 70)


if __name__ == "__main__":
    main()
