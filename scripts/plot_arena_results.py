from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _safe_read_summary(summary_json: str) -> pd.DataFrame:
    payload = json.loads(Path(summary_json).read_text(encoding="utf-8"))
    agents = payload.get("agents", {})
    rows: list[dict[str, float | int | str]] = []
    for name, s in agents.items():
        rows.append(
            {
                "agent": name,
                "games": int(s.get("games", 0)),
                "wins": int(s.get("wins", 0)),
                "losses": int(s.get("losses", 0)),
                "draws": int(s.get("draws", 0)),
                "win_rate": float(s.get("win_rate", 0.0)),
                "draw_rate": float(s.get("draw_rate", 0.0)),
                "avg_turns": float(s.get("avg_turns", 0.0)),
            }
        )
    return pd.DataFrame(rows)


def plot_winrate_bars(summary_df: pd.DataFrame, out_path: str) -> None:
    if summary_df.empty:
        return
    d = summary_df.sort_values("win_rate", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(d["agent"], d["win_rate"], color="#4C78A8")
    ax.set_ylim(0, 1.0)
    ax.set_title("Arena Agent Win Rate")
    ax.set_xlabel("Agent")
    ax.set_ylabel("Win Rate")
    for i, v in enumerate(d["win_rate"].tolist()):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_turns_box(games_df: pd.DataFrame, out_path: str) -> None:
    if games_df.empty:
        return
    long_rows: list[dict[str, float | str]] = []
    for _, row in games_df.iterrows():
        long_rows.append({"agent": row["white"], "turns": float(row["turns"])})
        long_rows.append({"agent": row["black"], "turns": float(row["turns"])})
    long_df = pd.DataFrame(long_rows)
    if long_df.empty:
        return
    labels = sorted(long_df["agent"].unique().tolist())
    data = [long_df.loc[long_df["agent"] == a, "turns"].tolist() for a in labels]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot(data, labels=labels, showfliers=False)
    ax.set_title("Arena Game Length Distribution by Agent")
    ax.set_xlabel("Agent")
    ax.set_ylabel("Turns")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_rolling_winrate(games_df: pd.DataFrame, out_path: str, window: int = 30) -> None:
    if games_df.empty:
        return
    agents = sorted(set(games_df["white"].tolist() + games_df["black"].tolist()))
    fig, ax = plt.subplots(figsize=(10, 5))
    for agent in agents:
        s = pd.Series(index=games_df.index, dtype=float)
        for idx, row in games_df.iterrows():
            w = row["winner"]
            if w == "draw":
                if row["white"] == agent or row["black"] == agent:
                    s.loc[idx] = 0.5
            elif w == agent:
                s.loc[idx] = 1.0
            elif row["white"] == agent or row["black"] == agent:
                s.loc[idx] = 0.0
        s = s.dropna()
        if s.empty:
            continue
        roll = s.rolling(window=window, min_periods=5).mean()
        ax.plot(s.index.tolist(), roll.tolist(), label=agent, linewidth=1.8)
    ax.set_ylim(0, 1.0)
    ax.set_title(f"Arena Rolling Win Rate (window={window})")
    ax.set_xlabel("Game Index")
    ax.set_ylabel("Rolling Win Rate")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot arena analysis figures")
    parser.add_argument("--games-csv", type=str, required=True, help="Arena per-game CSV")
    parser.add_argument("--summary-json", type=str, required=True, help="Arena summary JSON")
    parser.add_argument("--out-dir", type=str, default="results/figures")
    parser.add_argument("--prefix", type=str, default="arena")
    parser.add_argument("--rolling-window", type=int, default=30)
    args = parser.parse_args()

    games_df = pd.read_csv(args.games_csv)
    summary_df = _safe_read_summary(args.summary_json)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    p1 = str(out_dir / f"{args.prefix}_winrate_bars.png")
    p2 = str(out_dir / f"{args.prefix}_turns_boxplot.png")
    p3 = str(out_dir / f"{args.prefix}_rolling_winrate.png")

    plot_winrate_bars(summary_df, p1)
    plot_turns_box(games_df, p2)
    plot_rolling_winrate(games_df, p3, window=max(5, int(args.rolling_window)))

    print("Saved figures:")
    print(p1)
    print(p2)
    print(p3)


if __name__ == "__main__":
    main()
