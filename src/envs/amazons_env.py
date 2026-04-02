from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import copy


EMPTY = 0
P0 = 1
P1 = 2
BLOCK = 3

Action = Tuple[int, int, int, int, int, int]


@dataclass
class AmazonsConfig:
    size: int = 6
    max_turns: int = 200


class MiniAmazonsEnv:
    """Mini Amazons environment.

    Rules:
    1) One amazon per player on a size x size board.
    2) Current player moves as queen, then shoots one arrow as queen from new position.
    3) Arrow square becomes blocked forever.
    4) If a player has no legal action at turn start, that player loses.
    """

    def __init__(self, config: AmazonsConfig | None = None) -> None:
        self.cfg = config or AmazonsConfig()
        self.board: List[List[int]] = []
        self.positions = {0: (0, 0), 1: (0, 0)}
        self.current_player = 0
        self.turns = 0

    def reset(self):
        s = self.cfg.size
        self.board = [[EMPTY for _ in range(s)] for _ in range(s)]

        # Symmetric initial positions.
        p0 = (s - 1, 1)
        p1 = (0, s - 2)
        self.positions[0] = p0
        self.positions[1] = p1
        self.board[p0[0]][p0[1]] = P0
        self.board[p1[0]][p1[1]] = P1

        self.current_player = 0
        self.turns = 0
        return self.get_obs()

    def get_obs(self) -> Tuple[int, ...]:
        flat = []
        for row in self.board:
            flat.extend(row)
        flat.append(self.current_player)
        return tuple(flat)

    def legal_actions(self, player: int | None = None) -> List[Action]:
        p = self.current_player if player is None else player
        r, c = self.positions[p]

        actions: List[Action] = []
        move_targets = self._ray_moves(r, c, board=self.board)

        for mr, mc in move_targets:
            temp_board = copy.deepcopy(self.board)
            temp_board[r][c] = EMPTY
            temp_board[mr][mc] = P0 if p == 0 else P1
            arrow_targets = self._ray_moves(mr, mc, board=temp_board)
            for ar, ac in arrow_targets:
                actions.append((r, c, mr, mc, ar, ac))

        return actions

    def step(self, action: Action):
        player = self.current_player
        legal = self.legal_actions(player)
        if action not in legal:
            # Illegal action loses immediately.
            winner = 1 - player
            done = True
            rewards = {0: -1.0, 1: -1.0}
            rewards[winner] = 1.0
            return self.get_obs(), rewards, done, {"winner": winner, "illegal": True}

        fr, fc, tr, tc, ar, ac = action
        piece = P0 if player == 0 else P1

        self.board[fr][fc] = EMPTY
        self.board[tr][tc] = piece
        self.positions[player] = (tr, tc)
        self.board[ar][ac] = BLOCK

        self.turns += 1
        self.current_player = 1 - self.current_player

        # If next player has no legal actions, current player wins.
        next_legal = self.legal_actions(self.current_player)
        if not next_legal:
            winner = player
            done = True
            rewards = {0: -1.0, 1: -1.0}
            rewards[winner] = 1.0
            return self.get_obs(), rewards, done, {"winner": winner, "illegal": False}

        if self.turns >= self.cfg.max_turns:
            return self.get_obs(), {0: 0.0, 1: 0.0}, True, {"winner": -1, "illegal": False}

        # Small step penalty to shorten games.
        return self.get_obs(), {0: -0.001, 1: -0.001}, False, {"winner": -1, "illegal": False}

    def render(self) -> str:
        mapper = {EMPTY: ".", P0: "A", P1: "B", BLOCK: "x"}
        lines = [" ".join(mapper[v] for v in row) for row in self.board]
        lines.append(f"turn={self.turns} current_player={self.current_player}")
        return "\n".join(lines)

    def _ray_moves(self, r: int, c: int, board: List[List[int]]) -> List[Tuple[int, int]]:
        dirs = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ]
        s = self.cfg.size
        moves: List[Tuple[int, int]] = []

        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            while 0 <= nr < s and 0 <= nc < s and board[nr][nc] == EMPTY:
                moves.append((nr, nc))
                nr += dr
                nc += dc

        return moves
