from __future__ import annotations

import argparse
from pprint import pprint

from src.evaluation.evaluate import evaluate_vs_random
from src.evaluation.evaluate import evaluate_agents
from src.train.train_selfplay import TrainConfig, train_selfplay
from src.agents.random_agent import RandomAmazonsAgent
from src.agents.heuristic_agent import HeuristicAmazonsAgent
from src.agents.minimax_agent import MinimaxAmazonsAgent
from src.agents.mcts_agent import MCTSAmazonsAgent
from src.train.train_dqn_selfplay import DQNTrainConfig, train_dqn_selfplay
from src.train.train_dqn_opponent_pool import OpponentPoolTrainConfig, train_dqn_opponent_pool
from src.train.train_a2c_selfplay import A2CTrainConfig, train_a2c_selfplay
from src.agents.dqn_agent import DQNAmazonsAgent
from src.agents.a2c_agent import A2CAmazonsAgent
from src.agents.q_learning_agent import TabularQLearningAmazonsAgent
from src.agents.ppo_agent import PPOAmazonsAgent
from src.train.train_ppo_selfplay import PPOTrainConfig, train_ppo_selfplay
from src.train.train_alphazero_selfplay import AZTrainConfig, train_alphazero_selfplay
from src.train.train_bc_from_mcts import BCTrainConfig, train_bc_from_mcts


def cmd_train(args):
    cfg = TrainConfig(episodes=args.episodes, model_dir=args.model_dir, log_csv=args.log_csv)
    out = train_selfplay(cfg)
    print("Training finished:")
    pprint(out)


def cmd_eval(args):
    out = evaluate_vs_random(args.model, args.episodes)
    print("Evaluation:")
    pprint(out)


def _build_agent(name: str, minimax_depth: int):
    name = (name or "").lower()
    if name == "random":
        return RandomAmazonsAgent()
    if name == "heuristic":
        return HeuristicAmazonsAgent()
    if name == "minimax":
        return MinimaxAmazonsAgent(depth=minimax_depth)
    if name == "mcts":
        # keep default sims moderate for CLI responsiveness
        from src.agents.mcts_agent import MCTSConfig

        return MCTSAmazonsAgent(config=MCTSConfig(simulations=120, rollout_depth=40))
    raise ValueError(f"Unknown agent: {name}")


def cmd_eval_agents(args):
    agent0 = _build_agent(args.agent0, args.minimax_depth)
    agent1 = _build_agent(args.agent1, args.minimax_depth)
    out = evaluate_agents(agent0, agent1, episodes=args.episodes)
    print("Evaluation (agent vs agent):")
    pprint(out)


def _build_agent_from_model(agent_type: str, model_path: str, device: str | None = None):
    t = (agent_type or "").lower()
    if t == "dqn":
        return DQNAmazonsAgent.load(model_path, device=device)
    if t == "a2c":
        return A2CAmazonsAgent.load(model_path, device=device)
    if t == "q":
        return TabularQLearningAmazonsAgent.load(model_path)
    if t == "ppo":
        return PPOAmazonsAgent.load(model_path, device=device)
    if t == "az":
        from src.agents.alphazero_net import load_checkpoint
        from src.agents.alphazero_mcts import AlphaZeroMCTS, PUCTConfig
        from src.agents.alphazero_player import AlphaZeroPlayer

        net = load_checkpoint(model_path, device=device)
        mcts = AlphaZeroMCTS(size=net.size, net=net, cfg=PUCTConfig(simulations=120, seed=0), device=device)
        return AlphaZeroPlayer(size=net.size, net=net, mcts=mcts)
    if t == "bc":
        from src.agents.alphazero_net import load_checkpoint
        from src.agents.bc_player import BCPolicyPlayer

        net = load_checkpoint(model_path, device=device)
        return BCPolicyPlayer(size=net.size, net=net)
    raise ValueError(f"Unknown agent_type: {agent_type}")


def cmd_eval_trained_agents(args):
    agent0 = _build_agent_from_model(args.agent0_type, args.agent0_model, device=args.device)
    agent1 = _build_agent_from_model(args.agent1_type, args.agent1_model, device=args.device)
    out = evaluate_agents(agent0, agent1, episodes=args.episodes, size=args.size, max_turns=args.max_turns)
    print("Evaluation (trained agents):")
    pprint(out)


def cmd_eval_trained_vs_random(args):
    trained = _build_agent_from_model(args.agent_type, args.model, device=args.device)
    random_agent = RandomAmazonsAgent()
    out = evaluate_agents(trained, random_agent, episodes=args.episodes, size=args.size, max_turns=args.max_turns)
    print("Evaluation (trained vs random):")
    pprint(out)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mini Amazons MARL Project")
    sub = parser.add_subparsers(required=True)

    p_train = sub.add_parser("train", help="Train tabular Q-learning in self-play")
    p_train.add_argument("--episodes", type=int, default=5000)
    p_train.add_argument("--model-dir", type=str, default="results/models")
    p_train.add_argument("--log-csv", type=str, default="results/train_log.csv")
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("eval", help="Evaluate trained model against random")
    p_eval.add_argument("--model", type=str, default="results/models/agent0_q.json")
    p_eval.add_argument("--episodes", type=int, default=300)
    p_eval.set_defaults(func=cmd_eval)

    p_eval_agents = sub.add_parser("eval-agents", help="Evaluate two simple agents")
    p_eval_agents.add_argument("--agent0", type=str, default="heuristic", help="random|heuristic|minimax|mcts")
    p_eval_agents.add_argument("--agent1", type=str, default="random", help="random|heuristic|minimax|mcts")
    p_eval_agents.add_argument("--episodes", type=int, default=300)
    p_eval_agents.add_argument("--minimax-depth", type=int, default=2)
    p_eval_agents.set_defaults(func=cmd_eval_agents)

    p_train_dqn = sub.add_parser("train-dqn", help="Train DQN self-play")
    p_train_dqn.add_argument("--episodes", type=int, default=2000)
    p_train_dqn.add_argument("--size", type=int, default=6)
    p_train_dqn.add_argument("--max-turns", type=int, default=200)
    p_train_dqn.add_argument("--model-dir", type=str, default="results/models")
    p_train_dqn.add_argument("--log-csv", type=str, default="results/train_dqn_log.csv")
    p_train_dqn.add_argument("--device", type=str, default="cuda")
    p_train_dqn.add_argument("--use-per", action="store_true", help="Prioritized replay (Schaul et al.)")
    p_train_dqn.add_argument("--n-step", type=int, default=1, help="n-step return horizon (1=off)")
    p_train_dqn.add_argument("--per-alpha", type=float, default=0.6)
    p_train_dqn.add_argument("--per-beta-start", type=float, default=0.4)
    p_train_dqn.add_argument("--per-beta-end", type=float, default=1.0)
    p_train_dqn.add_argument("--per-beta-anneal-steps", type=int, default=100_000)
    p_train_dqn.set_defaults(
        func=lambda a: print(
            train_dqn_selfplay(
                DQNTrainConfig(
                    episodes=a.episodes,
                    size=a.size,
                    max_turns=a.max_turns,
                    model_dir=a.model_dir,
                    log_csv=a.log_csv,
                    device=a.device,
                    use_per=a.use_per,
                    n_step=a.n_step,
                    per_alpha=a.per_alpha,
                    per_beta_start=a.per_beta_start,
                    per_beta_end=a.per_beta_end,
                    per_beta_anneal_steps=a.per_beta_anneal_steps,
                )
            )
        )
    )

    p_pool = sub.add_parser(
        "train-dqn-pool",
        help="DQN vs opponent pool (PER+n-step + PFSP-lite frozen snapshots)",
    )
    p_pool.add_argument("--episodes", type=int, default=3000)
    p_pool.add_argument("--size", type=int, default=6)
    p_pool.add_argument("--max-turns", type=int, default=200)
    p_pool.add_argument("--model-dir", type=str, default="results/models/dqn_opponent_pool")
    p_pool.add_argument("--log-csv", type=str, default="results/train_dqn_opponent_pool.csv")
    p_pool.add_argument("--device", type=str, default="cuda")
    p_pool.add_argument("--n-step", type=int, default=3)
    p_pool.add_argument("--frozen-refresh", type=int, default=120)
    p_pool.set_defaults(
        func=lambda a: print(
            train_dqn_opponent_pool(
                OpponentPoolTrainConfig(
                    episodes=a.episodes,
                    size=a.size,
                    max_turns=a.max_turns,
                    model_dir=a.model_dir,
                    log_csv=a.log_csv,
                    device=a.device,
                    n_step=a.n_step,
                    frozen_refresh_episodes=a.frozen_refresh,
                )
            )
        )
    )

    p_train_a2c = sub.add_parser("train-a2c", help="Train A2C self-play")
    p_train_a2c.add_argument("--episodes", type=int, default=2000)
    p_train_a2c.add_argument("--size", type=int, default=6)
    p_train_a2c.add_argument("--max-turns", type=int, default=200)
    p_train_a2c.add_argument("--model-dir", type=str, default="results/models")
    p_train_a2c.add_argument("--log-csv", type=str, default="results/train_a2c_log.csv")
    p_train_a2c.add_argument("--device", type=str, default="cuda")
    p_train_a2c.set_defaults(
        func=lambda a: print(
            train_a2c_selfplay(
                A2CTrainConfig(
                    episodes=a.episodes,
                    size=a.size,
                    max_turns=a.max_turns,
                    model_dir=a.model_dir,
                    log_csv=a.log_csv,
                    device=a.device,
                )
            )
        )
    )

    p_train_ppo = sub.add_parser("train-ppo", help="Train PPO self-play")
    p_train_ppo.add_argument("--episodes", type=int, default=2000)
    p_train_ppo.add_argument("--size", type=int, default=6)
    p_train_ppo.add_argument("--max-turns", type=int, default=200)
    p_train_ppo.add_argument("--model-dir", type=str, default="results/models")
    p_train_ppo.add_argument("--log-csv", type=str, default="results/train_ppo_log.csv")
    p_train_ppo.add_argument("--device", type=str, default="cuda")
    p_train_ppo.set_defaults(
        func=lambda a: print(
            train_ppo_selfplay(
                PPOTrainConfig(
                    episodes=a.episodes,
                    size=a.size,
                    max_turns=a.max_turns,
                    model_dir=a.model_dir,
                    log_csv=a.log_csv,
                    device=a.device,
                )
            )
        )
    )

    p_train_az = sub.add_parser("train-az", help="Train AlphaZero-style (MCTS + policy/value net)")
    p_train_az.add_argument("--episodes", type=int, default=400)
    p_train_az.add_argument("--size", type=int, default=6)
    p_train_az.add_argument("--max-turns", type=int, default=120)
    p_train_az.add_argument("--model-dir", type=str, default="results/models")
    p_train_az.add_argument("--log-csv", type=str, default="results/train_az_log.csv")
    p_train_az.add_argument("--device", type=str, default="cuda")
    p_train_az.add_argument("--mcts-sims", type=int, default=120)
    p_train_az.add_argument("--save-every", type=int, default=50)
    p_train_az.set_defaults(
        func=lambda a: print(
            train_alphazero_selfplay(
                AZTrainConfig(
                    episodes=a.episodes,
                    size=a.size,
                    max_turns=a.max_turns,
                    model_dir=a.model_dir,
                    log_csv=a.log_csv,
                    device=a.device,
                    mcts_sims=a.mcts_sims,
                    save_every=a.save_every,
                )
            )
        )
    )

    p_train_bc = sub.add_parser("train-bc", help="Train behavior cloning policy from MCTS teacher")
    p_train_bc.add_argument("--episodes-generate", type=int, default=200)
    p_train_bc.add_argument("--size", type=int, default=6)
    p_train_bc.add_argument("--max-turns", type=int, default=120)
    p_train_bc.add_argument("--mcts-sims", type=int, default=120)
    p_train_bc.add_argument("--device", type=str, default="cuda")
    p_train_bc.add_argument("--model-dir", type=str, default="results/models")
    p_train_bc.add_argument("--log-csv", type=str, default="results/train_bc_log.csv")
    p_train_bc.set_defaults(
        func=lambda a: print(
            train_bc_from_mcts(
                BCTrainConfig(
                    episodes_generate=a.episodes_generate,
                    size=a.size,
                    max_turns=a.max_turns,
                    mcts_sims=a.mcts_sims,
                    device=a.device,
                    model_dir=a.model_dir,
                    log_csv=a.log_csv,
                )
            )
        )
    )

    p_eval_trained = sub.add_parser("eval-trained-agents", help="Evaluate two trained deep agents")
    p_eval_trained.add_argument("--agent0-type", type=str, default="dqn", help="dqn|a2c|ppo|q")
    p_eval_trained.add_argument("--agent0-model", type=str, default="results/models/agent0_dqn.pt")
    p_eval_trained.add_argument("--agent1-type", type=str, default="dqn", help="dqn|a2c|ppo|q")
    p_eval_trained.add_argument("--agent1-model", type=str, default="results/models/agent1_dqn.pt")
    p_eval_trained.add_argument("--episodes", type=int, default=300)
    p_eval_trained.add_argument("--size", type=int, default=6)
    p_eval_trained.add_argument("--max-turns", type=int, default=200)
    p_eval_trained.add_argument("--device", type=str, default="cuda")
    p_eval_trained.set_defaults(func=cmd_eval_trained_agents)

    p_eval_trained_vs_random = sub.add_parser("eval-trained-vs-random", help="Evaluate trained agent vs random")
    p_eval_trained_vs_random.add_argument("--agent-type", type=str, default="dqn", help="dqn|a2c|ppo|q")
    p_eval_trained_vs_random.add_argument("--model", type=str, default="results/models/agent0_dqn.pt")
    p_eval_trained_vs_random.add_argument("--episodes", type=int, default=300)
    p_eval_trained_vs_random.add_argument("--size", type=int, default=6)
    p_eval_trained_vs_random.add_argument("--max-turns", type=int, default=200)
    p_eval_trained_vs_random.add_argument("--device", type=str, default=None, help="e.g. cuda or cpu")
    p_eval_trained_vs_random.set_defaults(func=cmd_eval_trained_vs_random)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
