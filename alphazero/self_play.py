import chess
import numpy as np
import torch
import torch.multiprocessing as mp
from omegaconf import DictConfig
from tqdm import tqdm

from .encode import encode_board, move_to_action, action_to_move, ACTION_SPACE
from .mcts import run_mcts, select_action


def _play_one_game(network, cfg: DictConfig, device: str) -> list:
    board = chess.Board()
    history = []
    move_count = 0
    max_moves = cfg.self_play.max_moves
    temp_threshold = cfg.self_play.temp_threshold

    while not board.is_game_over(claim_draw=True) and move_count < max_moves:
        action_probs = run_mcts(board.fen(), network, cfg, device, add_noise=True)

        history.append((encode_board(board), action_probs))

        temp = 1.0 if move_count < temp_threshold else 0.1
        action = select_action(action_probs, temperature=temp)
        move = action_to_move(action, board)
        if move not in board.legal_moves:
            legal = [(action_probs[move_to_action(m, board)], m) for m in board.legal_moves]
            legal.sort(reverse=True)
            move = legal[0][1]

        board.push(move)
        move_count += 1

    result = board.result(claim_draw=True)
    z = {"1-0": 1.0, "0-1": -1.0}.get(result, 0.0)

    data = []
    for i, (state, pi) in enumerate(history):
        value = z if (i % 2 == 0) else -z
        data.append((state, pi, value))
    return data


def _worker_fn(network, cfg: DictConfig, device: str, num_games: int) -> list:
    torch.set_num_threads(1)
    network.eval()
    all_data = []
    with torch.no_grad():
        for _ in range(num_games):
            all_data.extend(_play_one_game(network, cfg, device))
    return all_data


def run_self_play(network, cfg: DictConfig, device: str):
    total_games = cfg.self_play.games
    num_workers = min(cfg.self_play.get("num_workers", 4), total_games)

    if num_workers <= 1:
        network.eval()
        all_data = []
        with torch.no_grad():
            for _ in tqdm(range(total_games), desc="Self-play"):
                all_data.extend(_play_one_game(network, cfg, device))
        return all_data

    network.share_memory()

    games_per_worker = [total_games // num_workers] * num_workers
    for i in range(total_games % num_workers):
        games_per_worker[i] += 1

    ctx = mp.get_context("spawn")
    with ctx.Pool(num_workers) as pool:
        results = pool.starmap(
            _worker_fn,
            [(network, cfg, device, g) for g in games_per_worker],
        )

    all_data = []
    for r in results:
        all_data.extend(r)
    return all_data
