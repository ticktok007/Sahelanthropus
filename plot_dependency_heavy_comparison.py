#!/usr/bin/env python3
"""
plot_dependency_heavy_comparison.py — Generate comparative learning curve plots
for MLP, GNN (Random Init), and GNN (Pre-trained) using standard csv module.
"""

import os
import csv
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-darkgrid" if "seaborn-v0_8-darkgrid" in plt.style.available else "default")


def read_csv_data(path):
    steps = []
    rewards = []
    expl_vars = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row["global_step"]))
            rewards.append(float(row["episode_reward_mean"]))
            expl_vars.append(float(row["explained_variance"]))
    return steps, rewards, expl_vars


def main():
    files = {
        "Flat MLP Baseline": "mlp_heavy_curve.csv",
        "GNN Policy (Random Init)": "gnn_random_heavy_curve.csv",
        "GNN Policy (Pre-trained Encoder)": "gnn_pretrained_heavy_curve.csv"
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    colors = {
        "Flat MLP Baseline": "#d62728",           # Red
        "GNN Policy (Random Init)": "#1f77b4",     # Blue
        "GNN Policy (Pre-trained Encoder)": "#2ca02c" # Green
    }

    for name, path in files.items():
        if os.path.exists(path):
            steps, rewards, expl_vars = read_csv_data(path)
            # Plot Episode Reward Mean
            axes[0].plot(steps, rewards, label=name, color=colors[name], linewidth=2.0)
            # Plot Explained Variance
            axes[1].plot(steps, expl_vars, label=name, color=colors[name], linewidth=2.0)

    axes[0].set_title("Episode Reward Convergence (Dependency-Heavy Corpus)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Environment Steps", fontsize=10)
    axes[0].set_ylabel("Mean Episode Reward", fontsize=10)
    axes[0].legend(loc="upper left", frameon=True)
    axes[0].grid(True, linestyle="--", alpha=0.6)

    axes[1].set_title("Critic Explained Variance ($1 - Var(residual)/Var(returns)$)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Environment Steps", fontsize=10)
    axes[1].set_ylabel("Explained Variance", fontsize=10)
    axes[1].legend(loc="lower right", frameon=True)
    axes[1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    out_png = "gnn_vs_mlp_dependency_heavy.png"
    plt.savefig(out_png, dpi=300)
    print(f"[+] Saved comparison plot to {out_png}")


if __name__ == "__main__":
    main()
