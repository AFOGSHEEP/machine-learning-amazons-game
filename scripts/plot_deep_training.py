from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _rolling(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()


def _load_log(path: Path, smooth_window: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "episode" not in df.columns or "winner" not in df.columns or "reward_0" not in df.columns:
        raise ValueError(f"Unexpected log format: {path}")
    out = df.copy()
    out["win0"] = (out["winner"] == 0).astype(float)
    out["win0_ma"] = _rolling(out["win0"], smooth_window)
    out["reward0_ma"] = _rolling(out["reward_0"], smooth_window)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot deep RL training curves")
    parser.add_argument("--dqn-csv", type=str, default="results/train_dqn_gpu.csv")
    parser.add_argument("--a2c-csv", type=str, default="results/train_a2c_gpu.csv")
    parser.add_argument("--ppo-csv", type=str, default="results/train_ppo_gpu.csv")
    parser.add_argument("--window", type=int, default=30, help="Moving average window")
    parser.add_argument("--out", type=str, default="results/figures/deep_training_compare.png")
    args = parser.parse_args()

    logs = {
        "DQN": Path(args.dqn_csv),
        "A2C": Path(args.a2c_csv),
        "PPO": Path(args.ppo_csv),
    }
    for name, p in logs.items():
        if not p.exists():
            raise FileNotFoundError(f"{name} log missing: {p}")

    data = {name: _load_log(p, args.window) for name, p in logs.items()}

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    ax0, ax1 = axes

    for name, df in data.items():
        ax0.plot(df["episode"], df["win0_ma"], label=name)
    ax0.set_title(f"Agent0 Win Rate (MA{args.window})")
    ax0.set_xlabel("Episode")
    ax0.set_ylabel("Win Rate")
    ax0.set_ylim(0.0, 1.0)
    ax0.grid(alpha=0.25)
    ax0.legend()

    for name, df in data.items():
        ax1.plot(df["episode"], df["reward0_ma"], label=name)
    ax1.set_title(f"Agent0 Reward (MA{args.window})")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")
    ax1.grid(alpha=0.25)
    ax1.legend()

    fig.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

