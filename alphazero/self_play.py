import time
import chess
import numpy as np
from tqdm import tqdm

from .encode import encode_board, move_to_action, action_to_move
from .mcts import run_mcts, select_action
from .config import AlphaZeroConfig


def play_one_game(network, config: AlphaZeroConfig, game_num: int = 0, verbose: bool = False):
    board = chess.Board()
    history = []
    move_count = 0
    max_moves = config.max_moves

    while not board.is_game_over(claim_draw=True):
        t0 = time.time()
        temp = 1.0 if move_count < config.temp_threshold else 0.1

        action_probs, root = run_mcts(board, network, config, add_noise=True)
        action = select_action(action_probs, temperature=temp)

        state_planes = encode_board(board)
        history.append((state_planes, action_probs))

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
        elapsed = time.time() - t0

        if verbose and move_count <= 5:
            print(f"    game {game_num} move {move_count}: {move.uci()} ({elapsed:.2f}s)")

        if move_count >= max_moves:
            break

    result = board.result(claim_draw=True)
    if result == "1-0":
        z = 1.0
    elif result == "0-1":
        z = -1.0
    else:
        z = 0.0

    training_data = []
    for i, (state, pi) in enumerate(history):
        value = z if (i % 2 == 0) else -z
        training_data.append((state, pi, value))

    return training_data


def run_self_play(network, config: AlphaZeroConfig):
    all_data = []
    pbar = tqdm(range(config.num_self_play_games), desc="Self-play")
    for i in pbar:
        verbose = (i == 0)
        game_data = play_one_game(network, config, game_num=i, verbose=verbose)
        all_data.extend(game_data)
        pbar.set_postfix(moves=len(game_data), total=len(all_data))
    return all_data
