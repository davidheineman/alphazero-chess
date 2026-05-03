import chess
import numpy as np
from tqdm import trange

from .encode import action_to_move, move_to_action
from .mcts import run_mcts, select_action
from .config import AlphaZeroConfig


def play_match(white_net, black_net, config: AlphaZeroConfig) -> float:
    board = chess.Board()
    move_count = 0

    eval_config = AlphaZeroConfig(
        num_simulations=config.eval_simulations,
        c_puct=config.c_puct,
        device=config.device,
    )

    while not board.is_game_over(claim_draw=True) and move_count < config.max_moves:
        net = white_net if board.turn == chess.WHITE else black_net
        action_probs, _ = run_mcts(board, net, eval_config, add_noise=False)
        action = select_action(action_probs, temperature=0.1)

        move = action_to_move(action, board)
        if move not in board.legal_moves:
            legal_actions = []
            for m in board.legal_moves:
                a = move_to_action(m, board)
                legal_actions.append((action_probs[a], a, m))
            legal_actions.sort(reverse=True)
            move = legal_actions[0][2]

        board.push(move)
        move_count += 1

    result = board.result(claim_draw=True)
    if result == "1-0":
        return 1.0
    elif result == "0-1":
        return -1.0
    return 0.0


def evaluate(new_net, old_net, config: AlphaZeroConfig) -> tuple[float, int, int, int]:
    wins = 0
    draws = 0
    losses = 0
    half = config.num_eval_games // 2

    for i in trange(config.num_eval_games, desc="Evaluation"):
        if i < half:
            result = play_match(new_net, old_net, config)
            if result > 0:
                wins += 1
            elif result < 0:
                losses += 1
            else:
                draws += 1
        else:
            result = play_match(old_net, new_net, config)
            if result < 0:
                wins += 1
            elif result > 0:
                losses += 1
            else:
                draws += 1

    total = wins + draws + losses
    win_rate = (wins + 0.5 * draws) / total if total > 0 else 0.0
    return win_rate, wins, draws, losses
