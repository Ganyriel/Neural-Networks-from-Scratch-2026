#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_runs(results_dir: str) -> list[dict]:
    runs = []
    for p in Path(results_dir).glob("*.json"):
        with open(p) as f:
            runs.append(json.load(f))
    return runs


def group_runs(runs: list[dict], group_by: str = "config_hash") -> dict[str, list[dict]]:
    """we are grouping the runs by confi_hashs = same config, different seeds)."""
    groups = defaultdict(list)
    for run in runs:
        if group_by == "config_hash":
            key = run["config_hash"]
        else:
            key = run["config"].get(group_by, "unknown")
        groups[key].append(run)
    return groups


def moving_average(data, window=50):
    data = np.asarray(data, dtype=float)
    if len(data) < window:
        return data[~np.isnan(data)]
    smoothed = np.full(len(data) - window + 1, np.nan)
    for j in range(len(smoothed)):
        chunk = data[j:j + window]
        if not np.isnan(chunk).any():   # skip windows containing NaN, because early in training the current_loss list can have NaNs
            smoothed[j] = chunk.mean()
    return smoothed


def aggregate_series(all_series: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Aggregates multiple runs with different seeds to mean ± std.
    Shortens all the runs to the length of the shortest runs.
    """
    min_len = min(len(s) for s in all_series)
    trimmed = np.array([s[:min_len] for s in all_series])
    mean = trimmed.mean(axis=0)
    std = trimmed.std(axis=0)
    return mean, std, min_len


def plot_comparison(results_dir: str, output_path: str, group_by: str = "config_hash", ma_window: int = 50):
    runs = load_runs(results_dir)
    if not runs:
        print(f"No runs found in {results_dir}.")
        return

    groups = group_runs(runs, group_by)
    print(f"Found: {len(groups)} config(s), {len(runs)} run(s) in total.")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(groups), 1)))

    # plot smoothed Training Reward (mean ± std over Seeds)
    
    ax = axes[0, 0]
    for (label, group_runs_list), color in zip(groups.items(), colors):
        smoothed_runs = [moving_average(np.array(r["rewards"]), ma_window) for r in group_runs_list]
        mean, std, n = aggregate_series(smoothed_runs)
        episodes = np.arange(n)
        ax.plot(episodes, mean, color=color, label=label, linewidth=2)
        ax.fill_between(episodes, mean - std, mean + std, color=color, alpha=0.2)
    ax.set_title("Smoothed Training Reward (mean ± std over seeds)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


    # plot Greedy Eval Reward (mean ± std over Seeds)

    ax = axes[0, 1]
    for (label, group_runs_list), color in zip(groups.items(), colors):
        eval_runs = []
        ep_step_candidates = []
        for r in group_runs_list:
            er = np.array(r.get("eval_rewards", []), dtype=float)
            if er.size == 0:
                continue
            if np.isnan(er).any():
                print(f"WARNING: {label} has NaN in eval_rewards — dropping run")
                continue
            eval_runs.append(er)
            eps = np.array(r.get("eval_episodes", []), dtype=float)
            if eps.size >= er.size:
                eps = eps[:er.size]
            else:
                print(f"WARNING: {label}: eval_episodes shorter than eval_rewards")
                eps = np.arange(er.size)  # fallback x-axis
            ep_step_candidates.append(eps)

        if not eval_runs:
            print(f"WARNING: no valid eval data for group {label}")
            continue

        mean, std, n = aggregate_series(eval_runs)
        # use x-steps from the first *valid* run, trimmed to match
        ep_steps = ep_step_candidates[0][:n]
        if ep_steps.size != n:
            ep_steps = np.arange(n)

        ax.plot(ep_steps, mean, color=color, label=label, linewidth=2)
        ax.fill_between(ep_steps, mean - std, mean + std, color=color, alpha=0.2)

    ax.set_xlim(left=0)
    ax.set_title("Greedy Evaluation Reward (ε=0)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Avg Eval Reward")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # plot Loss Curves (mean ± std über Seeds, log-scale)
    
    ax = axes[1, 0]
    for (label, group_runs_list), color in zip(groups.items(), colors):
        loss_runs = [moving_average(np.array(r["losses"]), ma_window) for r in group_runs_list]
        mean, std, n = aggregate_series(loss_runs)
        episodes = np.arange(n)
        ax.plot(episodes, mean, color=color, label=label, linewidth=2)
        ax.fill_between(episodes, mean - std, mean + std, color=color, alpha=0.2)
    ax.set_yscale("log")
    ax.set_title("Training Loss (log-scale)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")


    # plot Bar Chart: Final Reward ± std (last 100 episodes)

    ax = axes[1, 1]
    labels = []
    means = []
    errors = []
    for label, group_runs_list in groups.items():
        finals = []
        for r in group_runs_list:
            rewards = np.array(r["rewards"])
            if len(rewards) >= 100:
                finals.append(rewards[-100:].mean())
            elif len(rewards) > 0:
                finals.append(rewards.mean())
        if finals:
            labels.append(label)
            means.append(np.mean(finals))
            errors.append(np.std(finals))

    x_pos = np.arange(len(labels))
    bars = ax.bar(x_pos, means, yerr=errors, capsize=5, color=colors[:len(labels)], alpha=0.85)
    ax.axhline(y=0, color="green", linestyle="--", alpha=0.5, linewidth=1, label="Learning Threshold (0)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_title("Final Performance (last 100 episodes, mean ± std)")
    ax.set_ylabel("Reward")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout(pad=2.5)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved comparison to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--output", default="comparison.png")
    parser.add_argument("--group_by", default="config_hash",
                        help="field to group by (default: config_hash = same config, different seeds). "
                             "Example: --group_by architecture")
    parser.add_argument("--ma_window", type=int, default=50)
    args = parser.parse_args()

    plot_comparison(args.results_dir, args.output, args.group_by, args.ma_window)