from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot win-rate heatmap from eval matrix CSV")
    parser.add_argument("--csv", type=str, default="results/eval_matrix.csv")
    parser.add_argument("--out", type=str, default="results/figures/eval_matrix_heatmap.png")
    args = parser.parse_args()

    in_path = Path(args.csv)
    if not in_path.exists():
        raise FileNotFoundError(f"Missing CSV: {in_path}")

    df = pd.read_csv(in_path)
    if df.empty:
        raise ValueError("CSV is empty")

    pivot = df.pivot(index="agent0", columns="agent1", values="win_rate_agent0")
    pivot = pivot.sort_index().reindex(sorted(pivot.columns), axis=1)

    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(pivot.columns)), max(5, 0.8 * len(pivot.index))))
    im = ax.imshow(pivot.values, cmap="viridis", vmin=0.0, vmax=1.0)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticklabels(pivot.index)
    ax.set_title("Win-rate Heatmap (agent0 vs agent1)")

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = pivot.values[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="white", fontsize=9)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("win_rate_agent0")
    fig.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

