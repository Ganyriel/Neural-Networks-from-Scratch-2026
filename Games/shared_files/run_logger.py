import json
import os
import hashlib
from pathlib import Path


def config_hash(config: dict) -> str:
    # Deterministically hashing the config to group the same architecture with seed
    config_no_seed = {k: v for k, v in sorted(config.items()) if k != "seed"}
    raw = json.dumps(config_no_seed, sort_keys=True, default=str)
    return hashlib.md5(raw.encode()).hexdigest()[:8]


class RunLogger:
    """
    Here we will collect metrics during training and store them in a JSON

    Usage in der train-Funktion:
        logger = RunLogger(config, output_dir="results")
        ...
        logger.log_episode(reward, loss)
        logger.log_eval(episode, eval_reward)
        ...
        logger.save()
    """

    def __init__(self, config: dict, output_dir: str = "results"):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.rewards = []
        self.losses = []
        self.eval_rewards = []       # List of (episode, mean_reward)
        self.eval_episodes = []

    def log_episode(self, reward: float, loss: float):
        self.rewards.append(float(reward))
        self.losses.append(float(loss))

    def log_eval(self, episode: int, eval_reward: float):
        self.eval_episodes.append(episode)
        self.eval_rewards.append(float(eval_reward))

    def save(self):
        chash = config_hash(self.config)
        seed = self.config.get("seed", 0)
        filename = f"{chash}_seed{seed}.json"
        filepath = self.output_dir / filename

        data = {
            "config": self.config,
            "config_hash": chash,
            "seed": seed,
            "rewards": self.rewards,
            "losses": self.losses,
            "eval_episodes": self.eval_episodes,
            "eval_rewards": self.eval_rewards,
            "num_episodes": len(self.rewards),
        }
        with open(filepath, "w") as f:
            json.dump(data, f)

        print(f"[RunLogger] Saved to {filepath}")
        return filepath