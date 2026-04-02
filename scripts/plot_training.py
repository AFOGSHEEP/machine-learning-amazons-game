from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    log_path = Path("results/train_log.csv")
    if not log_path.exists():
        raise FileNotFoundError(f"Missing training log: {log_path}")

    df = pd.read_csv(log_path)
    fig_dir = Path("results/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    win_series = (df["winner"] == 0).astype(float)
    win_ma = win_series.rolling(window=200, min_periods=1).mean()
    reward_ma = df["reward_0"].rolling(window=200, min_periods=1).mean()

    plt.figure(figsize=(10, 4))
    plt.plot(df["episode"], win_ma, label="WinRate@200 (agent0)")
    plt.plot(df["episode"], reward_ma, label="Reward0@200")
    plt.xlabel("Episode")
    plt.ylabel("Value")
    plt.title("Training Progress (Self-play)")
    plt.legend()
    plt.tight_layout()

    out = fig_dir / "training_progress.png"
    plt.savefig(out, dpi=160)
    plt.close()

    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
