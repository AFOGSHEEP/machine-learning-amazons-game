from __future__ import annotations

from copy import deepcopy
from math import isqrt
from typing import List, Tuple

from src.envs.amazons_env import BLOCK, EMPTY, P0, P1, Action


def obs_to_board_and_positions(
    obs: Tuple[int, ...],
    size: int | None = None,
) -> Tuple[List[List[int]], dict[int, Tuple[int, int]], int]:
    """
    Observation format (same as env):
    - flattened board of length size*size
    - last element is current_player
    """
    if not obs:
        raise ValueError("Empty observation")

    current_player = int(obs[-1])
    board_flat = list(obs[:-1])

    if size is None:
        n = len(board_flat)
        s = isqrt(n)
        if s * s != n:
            raise ValueError(f"Invalid observation length for square board: {n}")
        size = s
    else:
        expected = size * size
        if len(board_flat) != expected:
            raise ValueError(f"Invalid observation length: {len(board_flat)} != {expected}")

    board = [board_flat[i * size : (i + 1) * size] for i in range(size)]

    positions: dict[int, Tuple[int, int]] = {0: (-1, -1), 1: (-1, -1)}
    for r in range(size):
        for c in range(size):
            if board[r][c] == P0:
                positions[0] = (r, c)
            elif board[r][c] == P1:
                positions[1] = (r, c)

    if positions[0] == (-1, -1) or positions[1] == (-1, -1):
        raise ValueError("Board must contain exactly one P0 and one P1 piece")

    return board, positions, current_player


def board_to_obs(board: List[List[int]], current_player: int) -> Tuple[int, ...]:
    flat: list[int] = []
    for row in board:
        flat.extend(int(x) for x in row)
    flat.append(int(current_player))
    return tuple(flat)


def ray_moves(board: List[List[int]], r: int, c: int, size: int) -> List[Tuple[int, int]]:
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

    moves: List[Tuple[int, int]] = []
    for dr, dc in dirs:
        nr, nc = r + dr, c + dc
        while 0 <= nr < size and 0 <= nc < size and board[nr][nc] == EMPTY:
            moves.append((nr, nc))
            nr += dr
            nc += dc
    return moves


def legal_actions_from_board(
    board: List[List[int]],
    positions: dict[int, Tuple[int, int]],
    player: int,
    size: int,
) -> List[Action]:
    r, c = positions[player]
    piece = P0 if player == 0 else P1
    if board[r][c] != piece:
        # State inconsistency (should not happen if board was produced from env).
        return []

    actions: List[Action] = []
    move_targets = ray_moves(board, r, c, size)
    for mr, mc in move_targets:
        temp_board = deepcopy(board)
        temp_board[r][c] = EMPTY
        temp_board[mr][mc] = piece

        arrow_targets = ray_moves(temp_board, mr, mc, size)
        for ar, ac in arrow_targets:
            actions.append((r, c, mr, mc, ar, ac))

    return actions


def apply_action(
    board: List[List[int]],
    positions: dict[int, Tuple[int, int]],
    action: Action,
    player: int,
) -> Tuple[List[List[int]], dict[int, Tuple[int, int]]]:
    fr, fc, tr, tc, ar, ac = action
    piece = P0 if player == 0 else P1

    b2 = deepcopy(board)
    pos2 = dict(positions)

    b2[fr][fc] = EMPTY
    b2[tr][tc] = piece
    b2[ar][ac] = BLOCK
    pos2[player] = (tr, tc)

    return b2, pos2


def step_state(
    board: List[List[int]],
    positions: dict[int, Tuple[int, int]],
    current_player: int,
    action: Action,
    size: int,
) -> tuple[Tuple[int, ...], bool, int]:
    """
    Stateless transition.
    Returns: (next_obs, done, winner)
    winner: 0/1 if terminal else -1
    """
    b2, pos2 = apply_action(board, positions, action, current_player)
    next_player = 1 - current_player
    next_legal = legal_actions_from_board(b2, pos2, next_player, size)
    if not next_legal:
        return board_to_obs(b2, next_player), True, current_player
    return board_to_obs(b2, next_player), False, -1

