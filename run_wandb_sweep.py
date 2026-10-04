#!/usr/bin/env python3
"""
run_wandb_sweep.py — W&B Hyperparameter Grid Sweep for PPO Superoptimizer

Sweep Specification:
  learning_rate in {1e-4, 3e-4}
  num_steps in {256, 512}
  clip_coef in {0.1, 0.2}
  gae_lambda in {0.9, 0.95}
  16 Configurations x 2 Seeds (seed=0, seed=1) = 32 Runs Total
"""

import csv
import json
import os
import time
from typing import Dict, List, Any

import matplotlib.pyplot as plt
import numpy as np
import wandb

from run_ppo_100k_benchmark import PPOFixedBenchmarkTrainer, generate_diverse_corpus

SWEEP_CSV_LOG = "wandb_sweep_results.csv"
SWEEP_JSON_LOG = "wandb_sweep_results.json"
SWEEP_PLOT_PNG = "wandb_sweep_comparison.png"


def run_grid_sweep():
    # Generate and save diverse corpus
    diverse_corpus = generate_diverse_corpus()
    with open("corpus.json", "w") as f:
        json.dump(diverse_corpus, f, indent=2)

    learning_rates = [1e-4, 3e-4]
    num_steps_options = [256, 512]
    clip_coefs = [0.1, 0.2]
    gae_lambdas = [0.9, 0.95]
    seeds = [0, 1]

    configs = []
    for lr in learning_rates:
        for ns in num_steps_options:
            for clip in clip_coefs:
                for lam in gae_lambdas:
                    configs.append({"lr": lr, "num_steps": ns, "clip_coef": clip, "gae_lambda": lam})

    print("======================================================================")
    print(" WEIGHTS & BIASES HYPERPARAMETER GRID SWEEP")
    print("======================================================================")
    print(f" Configurations : {len(configs)} (lr in {learning_rates}, ns in {num_steps_options}, clip in {clip_coefs}, lam in {gae_lambdas})")
    print(f" Seeds Per Config: {seeds}")
    print(f" Total Runs     : {len(configs) * len(seeds)} runs (100,000 steps per run)")
    print("======================================================================\n")

    sweep_results = []
    config_histories: Dict[str, Dict[int, List[Dict[str, Any]]]] = {}

    run_idx = 0
    total_runs = len(configs) * len(seeds)

    for cfg in configs:
        lr = cfg["lr"]
        ns = cfg["num_steps"]
        clip = cfg["clip_coef"]
        lam = cfg["gae_lambda"]
        cfg_key = f"lr{lr}_ns{ns}_clip{clip}_lam{lam}"
        group_name = f"sweep_{cfg_key}"
        config_histories[cfg_key] = {}

        for seed in seeds:
            run_idx += 1
            run_tag = f"ppo_mlp_{cfg_key}_seed{seed}"

            print(f"\n[{run_idx}/{total_runs}] Running Config: {cfg_key} | Seed: {seed}")
            print(f"      Tag: {run_tag} | Group: {group_name}")

            trainer = PPOFixedBenchmarkTrainer(
                num_envs=4,
                num_steps=ns,
                lr=lr,
                clip_coef=clip,
                gae_lambda=lam
            )

            res = trainer.train_benchmark_100k(
                seed=seed,
                total_steps_target=100000,
                run_tag=run_tag,
                group_name=group_name
            )

            history = res["history"]
            config_histories[cfg_key][seed] = history

            rec = {
                "config_key": cfg_key,
                "learning_rate": lr,
                "num_steps": ns,
                "clip_coef": clip,
                "gae_lambda": lam,
                "seed": seed,
                "run_tag": run_tag,
                "group_name": group_name,
                "final_explained_variance": res["final_explained_variance"],
                "max_explained_variance": res["max_explained_variance"],
                "init_explained_variance": res["init_explained_variance"],
            }
            sweep_results.append(rec)

    # Compute Group Summaries across 2 seeds per config
    summary_list = []
    for cfg in configs:
        cfg_key = f"lr{cfg['lr']}_ns{cfg['num_steps']}_clip{cfg['clip_coef']}_lam{cfg['gae_lambda']}"
        s0_final_ev = config_histories[cfg_key][0][-1]["explained_variance"]
        s1_final_ev = config_histories[cfg_key][1][-1]["explained_variance"]
        ev_mean = float(np.mean([s0_final_ev, s1_final_ev]))
        ev_std = float(np.std([s0_final_ev, s1_final_ev]))

        s0_ep_rew = config_histories[cfg_key][0][-1]["episode_reward_mean"]
        s1_ep_rew = config_histories[cfg_key][1][-1]["episode_reward_mean"]
        rew_mean = float(np.mean([s0_ep_rew, s1_ep_rew]))
        rew_std = float(np.std([s0_ep_rew, s1_ep_rew]))

        summary_list.append({
            "config_key": cfg_key,
            "lr": cfg["lr"],
            "num_steps": cfg["num_steps"],
            "clip_coef": cfg["clip_coef"],
            "gae_lambda": cfg["gae_lambda"],
            "ev_mean": ev_mean,
            "ev_std": ev_std,
            "rew_mean": rew_mean,
            "rew_std": rew_std,
        })

    # Sort summary by final explained variance descending
    summary_list.sort(key=lambda x: x["ev_mean"], reverse=True)

    print("\n======================================================================")
    print(" W&B SWEEP LEADERBOARD (Ranked by Mean Explained Variance)")
    print("======================================================================")
    print(f"{'Rank':<5} | {'Config Key':<32} | {'ExplVar (Mean ± Std)':<22} | {'Reward (Mean ± Std)':<22}")
    print("-" * 85)
    for rank, item in enumerate(summary_list, start=1):
        print(f"{rank:<5} | {item['config_key']:<32} | {item['ev_mean']:+.4f} ± {item['ev_std']:.4f}        | {item['rew_mean']:+.4f} ± {item['rew_std']:.4f}")
    print("======================================================================\n")

    # Save CSV & JSON
    fieldnames = ["config_key", "learning_rate", "num_steps", "clip_coef", "gae_lambda", "seed", "run_tag", "group_name",
                  "final_explained_variance", "max_explained_variance", "init_explained_variance"]
    with open(SWEEP_CSV_LOG, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sweep_results)

    with open(SWEEP_JSON_LOG, "w") as f:
        json.dump(summary_list, f, indent=2)

    # Generate Comparison Plot
    plot_sweep_comparison(config_histories, summary_list, SWEEP_PLOT_PNG)

    return summary_list


def plot_sweep_comparison(config_histories, summary_list, output_png):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("W&B PPO Hyperparameter Grid Sweep (16 Configs x 2 Seeds = 32 Runs)\nlr in {1e-4, 3e-4} | n_steps in {256, 512} | clip_coef in {0.1, 0.2} | lambda in {0.9, 0.95}", fontsize=15, fontweight="bold")

    colors = plt.cm.tab10(np.linspace(0, 1, len(summary_list)))

    # Panel 1: Explained Variance Trajectory across Configs
    ax1 = axes[0, 0]
    for idx, item in enumerate(summary_list):
        cfg_key = item["config_key"]
        s0_h = config_histories[cfg_key][0]
        s1_h = config_histories[cfg_key][1]
        steps = [rec["total_steps"] for rec in s0_h]
        ev_s0 = [rec["explained_variance"] for rec in s0_h]
        ev_s1 = [rec["explained_variance"] for rec in s1_h]
        min_len = min(len(ev_s0), len(ev_s1))
        steps = steps[:min_len]
        ev_mean = np.mean([ev_s0[:min_len], ev_s1[:min_len]], axis=0)

        ax1.plot(steps, ev_mean, label=cfg_key, color=colors[idx], linewidth=2.0)

    ax1.set_title("Explained Variance Trajectory (Mean of 2 Seeds)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Environment Steps")
    ax1.set_ylabel("Explained Variance")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(fontsize=8, loc="lower right")

    # Panel 2: Final Explained Variance Leaderboard (Bar Plot)
    ax2 = axes[0, 1]
    cfg_keys = [item["config_key"] for item in summary_list]
    ev_means = [item["ev_mean"] for item in summary_list]
    ev_stds = [item["ev_std"] for item in summary_list]

    y_pos = np.arange(len(cfg_keys))
    ax2.barh(y_pos, ev_means, xerr=ev_stds, align="center", color=colors, alpha=0.85, capsize=4)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(cfg_keys, fontsize=9)
    ax2.invert_yaxis()  # top-down best to worst
    ax2.set_xlabel("Final Explained Variance (Mean ± Std)")
    ax2.set_title("Hyperparameter Leaderboard (Final ExplVar)", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)

    # Panel 3: Value Loss Trajectory across Configs
    ax3 = axes[1, 0]
    for idx, item in enumerate(summary_list):
        cfg_key = item["config_key"]
        s0_h = config_histories[cfg_key][0]
        s1_h = config_histories[cfg_key][1]
        vl_s0 = [rec["value_loss"] for rec in s0_h]
        vl_s1 = [rec["value_loss"] for rec in s1_h]
        min_len = min(len(vl_s0), len(vl_s1))
        steps = [rec["total_steps"] for rec in s0_h][:min_len]
        vl_mean = np.mean([vl_s0[:min_len], vl_s1[:min_len]], axis=0)

        ax3.plot(steps, vl_mean, label=cfg_key, color=colors[idx], linewidth=1.8)

    ax3.set_title("Value Function Loss Trajectory", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Environment Steps")
    ax3.set_ylabel("Value Loss")
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.legend(fontsize=8, loc="upper right")

    # Panel 4: Episode Reward Mean Trajectory
    ax4 = axes[1, 1]
    for idx, item in enumerate(summary_list):
        cfg_key = item["config_key"]
        s0_h = config_histories[cfg_key][0]
        s1_h = config_histories[cfg_key][1]
        rew_s0 = [rec["episode_reward_mean"] for rec in s0_h]
        rew_s1 = [rec["episode_reward_mean"] for rec in s1_h]
        min_len = min(len(rew_s0), len(rew_s1))
        steps = [rec["total_steps"] for rec in s0_h][:min_len]
        rew_mean = np.mean([rew_s0[:min_len], rew_s1[:min_len]], axis=0)

        ax4.plot(steps, rew_mean, label=cfg_key, color=colors[idx], linewidth=1.8)

    ax4.set_title("Episode Reward Mean Trajectory", fontsize=12, fontweight="bold")
    ax4.set_xlabel("Environment Steps")
    ax4.set_ylabel("Episode Reward Mean")
    ax4.grid(True, linestyle=":", alpha=0.6)
    ax4.legend(fontsize=8, loc="lower right")

    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.close()
    print(f"[+] Saved W&B Sweep Comparison Plot to: {output_png}")


if __name__ == "__main__":
    run_grid_sweep()
