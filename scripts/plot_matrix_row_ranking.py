from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Bar chart: mean row win-rate from eval matrix CSV")
    parser.add_argument("--csv", type=str, default="results/eval_matrix_long_models_n30.csv")
    parser.add_argument("--out", type=str, default="results/figures/eval_matrix_long_n30_row_ranking.png")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if "agent0" not in df.columns or "win_rate_agent0" not in df.columns:
        raise ValueError("CSV must have agent0 and win_rate_agent0")

    s = df.groupby("agent0")["win_rate_agent0"].mean().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = plt.cm.plasma([i / max(1, len(s) - 1) for i in range(len(s))])
    ax.barh(s.index.astype(str), s.values, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Mean win_rate_agent0 (row average)")
    ax.set_xlim(0.0, 1.05)
    ax.set_title("Head-to-head matrix: average win rate by row agent")
    ax.grid(axis="x", alpha=0.3)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
