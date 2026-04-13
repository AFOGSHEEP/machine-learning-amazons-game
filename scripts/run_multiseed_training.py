from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.train.train_a2c_selfplay import A2CTrainConfig, train_a2c_selfplay
from src.train.train_dqn_selfplay import DQNTrainConfig, train_dqn_selfplay
from src.train.train_ppo_selfplay import PPOTrainConfig, train_ppo_selfplay


def parse_seeds(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-seed self-play training")
    parser.add_argument("--algo", type=str, default="dqn", choices=["dqn", "a2c", "ppo"])
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4")
    parser.add_argument("--episodes", type=int, default=800)
    parser.add_argument("--size", type=int, default=6)
    parser.add_argument("--max-turns", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--prune-top-k", type=int, default=0)
    parser.add_argument("--prune-keep-ratio", type=float, default=1.0)
    parser.add_argument("--reward-mobility-weight", type=float, default=0.0)
    parser.add_argument("--reward-center-weight", type=float, default=0.0)
    parser.add_argument("--out-csv", type=str, default="results/multiseed_summary.csv")
    args = parser.parse_args()

    out_rows: list[dict[str, str | int | float]] = []
    for seed in parse_seeds(args.seeds):
        model_dir = f"results/models/{args.algo}_seed{seed}"
        log_csv = f"results/{args.algo}_seed{seed}_log.csv"
        meta_json = f"results/runs/{args.algo}_seed{seed}_meta.json"
        if args.algo == "dqn":
            out = train_dqn_selfplay(
                DQNTrainConfig(
                    seed=seed,
                    episodes=args.episodes,
                    size=args.size,
                    max_turns=args.max_turns,
                    device=args.device,
                    model_dir=model_dir,
                    log_csv=log_csv,
                    metadata_json=meta_json,
                    prune_top_k=args.prune_top_k,
                    prune_keep_ratio=args.prune_keep_ratio,
                    reward_mobility_weight=args.reward_mobility_weight,
                    reward_center_weight=args.reward_center_weight,
                )
            )
        elif args.algo == "a2c":
            out = train_a2c_selfplay(
                A2CTrainConfig(
                    seed=seed,
                    episodes=args.episodes,
                    size=args.size,
                    max_turns=args.max_turns,
                    device=args.device,
                    model_dir=model_dir,
                    log_csv=log_csv,
                    metadata_json=meta_json,
                    prune_top_k=args.prune_top_k,
                    prune_keep_ratio=args.prune_keep_ratio,
                    reward_mobility_weight=args.reward_mobility_weight,
                    reward_center_weight=args.reward_center_weight,
                )
            )
        else:
            out = train_ppo_selfplay(
                PPOTrainConfig(
                    seed=seed,
                    episodes=args.episodes,
                    size=args.size,
                    max_turns=args.max_turns,
                    device=args.device,
                    model_dir=model_dir,
                    log_csv=log_csv,
                    metadata_json=meta_json,
                    prune_top_k=args.prune_top_k,
                    prune_keep_ratio=args.prune_keep_ratio,
                    reward_mobility_weight=args.reward_mobility_weight,
                    reward_center_weight=args.reward_center_weight,
                )
            )
        out_rows.append(
            {
                "algo": args.algo,
                "seed": seed,
                "episodes": args.episodes,
                "model0": out["model0"],
                "log_csv": out["log"],
                "run_id": out.get("run_id", ""),
            }
        )

    p = Path(args.out_csv)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["algo", "seed", "episodes", "model0", "log_csv", "run_id"])
        writer.writeheader()
        writer.writerows(out_rows)

    # Tiny console summary.
    seeds = [int(r["seed"]) for r in out_rows]
    print(f"Multi-seed finished: algo={args.algo}, seeds={seeds}, output={args.out_csv}")


if __name__ == "__main__":
    main()
