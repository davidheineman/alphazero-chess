import chess
import chess.engine
import numpy as np
import torch
from omegaconf import DictConfig
from tqdm import trange

from .encode import action_to_move, move_to_action
from .mcts import run_mcts, select_action

SKILL_TO_ELO = {0: 800, 1: 900, 2: 1000, 3: 1100, 5: 1300, 8: 1600, 10: 1800, 15: 2200, 20: 3200}


def play_vs_stockfish(
    network,
    cfg: DictConfig,
    device: str,
    stockfish_path: str = "stockfish",
    skill_level: int = 1,
    move_time: float = 0.01,
    network_is_white: bool = True,
) -> float:
    board = chess.Board()
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    engine.configure({"Skill Level": skill_level})
    move_count = 0

    try:
        while not board.is_game_over(claim_draw=True) and move_count < cfg.self_play.max_moves:
            is_network_turn = (board.turn == chess.WHITE) == network_is_white

            if is_network_turn:
                action_probs, _ = run_mcts(board, network, cfg, device, add_noise=False)
                action = select_action(action_probs, temperature=0.1)
                move = action_to_move(action, board)
                if move not in board.legal_moves:
                    legal = [(action_probs[move_to_action(m, board)], m) for m in board.legal_moves]
                    legal.sort(reverse=True)
                    move = legal[0][1]
            else:
                result = engine.play(board, chess.engine.Limit(time=move_time))
                move = result.move

            board.push(move)
            move_count += 1

        result = board.result(claim_draw=True)
        if result == "1-0":
            return 1.0 if network_is_white else -1.0
        elif result == "0-1":
            return -1.0 if network_is_white else 1.0
        return 0.0
    finally:
        engine.quit()


@torch.no_grad()
def eval_vs_stockfish(
    network,
    cfg: DictConfig,
    device: str,
    num_games: int = 6,
    skill_level: int = 1,
    stockfish_path: str = "stockfish",
    move_time: float = 0.01,
) -> dict:
    network.eval()
    wins, draws, losses = 0, 0, 0
    half = num_games // 2

    for i in trange(num_games, desc=f"vs Stockfish (skill={skill_level})"):
        is_white = i < half
        r = play_vs_stockfish(network, cfg, device, stockfish_path, skill_level, move_time, is_white)
        if r > 0:
            wins += 1
        elif r < 0:
            losses += 1
        else:
            draws += 1

    total = wins + draws + losses
    win_rate = (wins + 0.5 * draws) / total if total > 0 else 0.0
    elo_est = SKILL_TO_ELO.get(skill_level, skill_level * 100 + 800)

    return {
        "win_rate": win_rate,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "skill_level": skill_level,
        "elo_estimate": elo_est,
    }
