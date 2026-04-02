from __future__ import annotations

from typing import Dict

from src.agents.q_learning_agent import TabularQLearningAmazonsAgent
from src.agents.random_agent import RandomAmazonsAgent
from src.envs.amazons_env import MiniAmazonsEnv, AmazonsConfig


def evaluate_vs_random(model_path: str, episodes: int = 300) -> Dict[str, float]:
    env = MiniAmazonsEnv(AmazonsConfig(size=6, max_turns=200))
    trained = TabularQLearningAmazonsAgent.load(model_path)
    random_agent = RandomAmazonsAgent()

    wins = {0: 0, 1: 0, -1: 0}

    for _ in range(episodes):
        state = env.reset()
        done = False
        info = {"winner": -1}

        while not done:
            p = env.current_player
            legal = env.legal_actions(p)
            if p == 0:
                action = trained.select_action(state, legal, training=False)
            else:
                action = random_agent.select_action(legal, training=False)

            state, _, done, info = env.step(action)

        wins[info["winner"]] += 1

    return {
        "win_rate_trained": wins[0] / episodes,
        "win_rate_random": wins[1] / episodes,
        "draw_rate": wins[-1] / episodes,
    }


def evaluate_agents(
    agent0,
    agent1,
    episodes: int = 300,
    size: int = 6,
    max_turns: int = 200,
) -> Dict[str, float]:
    """
    Generic evaluation of any two agents that expose:
    select_action(state, legal_actions, training=False) -> Action
    """
    env = MiniAmazonsEnv(AmazonsConfig(size=size, max_turns=max_turns))
    wins = {0: 0, 1: 0, -1: 0}

    for _ in range(episodes):
        state = env.reset()
        done = False
        info = {"winner": -1}

        while not done:
            p = env.current_player
            legal = env.legal_actions(p)

            if p == 0:
                action = agent0.select_action(state, legal, training=False)
            else:
                action = agent1.select_action(state, legal, training=False)

            state, _, done, info = env.step(action)

        wins[info["winner"]] += 1

    return {
        "win_rate_agent0": wins[0] / episodes,
        "win_rate_agent1": wins[1] / episodes,
        "draw_rate": wins[-1] / episodes,
    }
