from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import random
from typing import Callable

from src.agents.random_agent import RandomAmazonsAgent
from src.envs.amazons_env import BLOCK, EMPTY, AmazonsConfig, MiniAmazonsEnv


@dataclass
class GeneralizationConfig:
    episodes: int = 100
    seeds: tuple[int, ...] = (0, 1, 2)
    size: int = 6
    max_turns: int = 200
    extra_blocks: tuple[int, ...] = (0, 2, 4)
    out_csv: str = "results/eval_generalization.csv"


def _inject_random_blocks(env: MiniAmazonsEnv, count: int, rng: random.Random) -> None:
    if count <= 0:
        return
    s = env.cfg.size
    cells: list[tuple[int, int]] = []
    for r in range(s):
        for c in range(s):
            if env.board[r][c] == EMPTY:
                cells.append((r, c))
    rng.shuffle(cells)
    for r, c in cells[:count]:
        env.board[r][c] = BLOCK


def evaluate_with_preset(
    build_agent: Callable[[], object],
    episodes: int,
    size: int,
    max_turns: int,
    seed: int,
    extra_blocks: int,
) -> dict[str, float]:
    rng = random.Random(seed)
    wins = {0: 0, 1: 0, -1: 0}
    for _ in range(episodes):
        env = MiniAmazonsEnv(AmazonsConfig(size=size, max_turns=max_turns))
        state = env.reset()
        _inject_random_blocks(env, extra_blocks, rng)
        state = env.get_obs()
        done = False
        info = {"winner": -1}
        agent0 = build_agent()
        agent1 = RandomAmazonsAgent()
        while not done:
            p = env.current_player
            legal = env.legal_actions(p)
            action = (
                agent0.select_action(state, legal, training=False)
                if p == 0
                else agent1.select_action(state, legal, training=False)
            )
            state, _, done, info = env.step(action)
        wins[info["winner"]] += 1
    return {
        "win_rate_agent0": wins[0] / episodes,
        "win_rate_agent1": wins[1] / episodes,
        "draw_rate": wins[-1] / episodes,
    }


def run_generalization_benchmark(
    build_agent: Callable[[], object],
    cfg: GeneralizationConfig | None = None,
) -> str:
    c = cfg or GeneralizationConfig()
    out_path = Path(c.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | int]] = []
    for seed in c.seeds:
        random.seed(seed)
        for extra_blocks in c.extra_blocks:
            out = evaluate_with_preset(
                build_agent=build_agent,
                episodes=c.episodes,
                size=c.size,
                max_turns=c.max_turns,
                seed=seed,
                extra_blocks=extra_blocks,
            )
            rows.append(
                {
                    "seed": seed,
                    "extra_blocks": extra_blocks,
                    "episodes": c.episodes,
                    "size": c.size,
                    "max_turns": c.max_turns,
                    "win_rate_agent0": out["win_rate_agent0"],
                    "win_rate_agent1": out["win_rate_agent1"],
                    "draw_rate": out["draw_rate"],
                }
            )

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "seed",
                "extra_blocks",
                "episodes",
                "size",
                "max_turns",
                "win_rate_agent0",
                "win_rate_agent1",
                "draw_rate",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return str(out_path)
