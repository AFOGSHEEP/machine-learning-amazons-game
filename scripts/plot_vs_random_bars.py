from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def wilson_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Approximate 95% Wilson score interval for binomial proportion."""
    if n <= 0:
        return 0.0, 1.0
    p = min(1.0, max(0.0, p))
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bar chart: win rate vs random with Wilson 95% CI")
    parser.add_argument("--csv", type=str, default="results/eval_summary_long.csv")
    parser.add_argument("--out", type=str, default="results/figures/vs_random_winrate_bars.png")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if not {"experiment", "episodes", "win_rate_agent0"}.issubset(df.columns):
        raise ValueError(f"Unexpected CSV columns: {df.columns.tolist()}")

    labels = []
    ps = []
    lows = []
    highs = []
    for _, row in df.iterrows():
        n = int(row["episodes"])
        p = float(row["win_rate_agent0"])
        lo, hi = wilson_ci(p, n)
        labels.append(str(row["experiment"]).replace("_vs_random", "").replace("_", " "))
        ps.append(p)
        lows.append(lo)
        highs.append(hi)

    err_low = [p - lo for p, lo in zip(ps, lows)]
    err_high = [hi - p for p, hi in zip(ps, highs)]

    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = range(len(labels))
    colors = plt.cm.viridis([i / max(1, len(labels) - 1) for i in range(len(labels))])
    ax.bar(x, ps, yerr=[err_low, err_high], capsize=4, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Win rate (agent0)")
    ax.set_ylim(0.0, 1.0)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="random baseline")
    ax.set_title("Trained agents vs Random (Wilson 95% CI)")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
