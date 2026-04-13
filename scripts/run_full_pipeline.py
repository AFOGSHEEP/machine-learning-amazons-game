from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd: list[str], cwd: Path) -> None:
    print("RUN:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full RL pipeline: train -> arena -> plots")
    parser.add_argument("--episodes", type=int, default=600)
    parser.add_argument("--arena-games", type=int, default=300)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--prefix", type=str, default="arena_ui")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    py = sys.executable

    run_cmd(
        [
            py,
            "-m",
            "src.main",
            "train-dqn",
            "--episodes",
            str(args.episodes),
            "--size",
            "6",
            "--max-turns",
            "200",
            "--device",
            args.device,
            "--use-per",
            "--n-step",
            "3",
            "--reward-mobility-weight",
            "0.05",
            "--reward-center-weight",
            "0.02",
            "--prune-top-k",
            "24",
            "--prune-keep-ratio",
            "0.5",
            "--model-dir",
            "results/models/dqn_ui_run",
            "--log-csv",
            "results/train_dqn_ui_run.csv",
            "--metadata-json",
            "results/runs/train_dqn_ui_run_meta.json",
        ],
        root,
    )

    run_cmd(
        [
            py,
            "-m",
            "src.main",
            "run-arena",
            "--agent",
            "rnd:random",
            "--agent",
            "heu:heuristic",
            "--agent",
            "mm:minimax:1",
            "--agent",
            "dqn:dqn:results/models/dqn_ui_run/agent0_dqn.pt",
            "--games",
            str(args.arena_games),
            "--mode",
            "pool",
            "--size",
            "6",
            "--max-turns",
            "200",
            "--device",
            args.device,
            "--out-games-csv",
            "results/arena_games_ui.csv",
            "--out-summary-json",
            "results/arena_summary_ui.json",
        ],
        root,
    )

    run_cmd(
        [
            py,
            "scripts/plot_arena_results.py",
            "--games-csv",
            "results/arena_games_ui.csv",
            "--summary-json",
            "results/arena_summary_ui.json",
            "--out-dir",
            "results/figures",
            "--prefix",
            args.prefix,
            "--rolling-window",
            "30",
        ],
        root,
    )
    print("FULL PIPELINE DONE")


if __name__ == "__main__":
    main()
