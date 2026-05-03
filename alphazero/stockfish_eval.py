import chess
import chess.engine
import torch
from omegaconf import DictConfig

from .encode import action_to_move, move_to_action
from .mcts import run_mcts, select_action

# Stockfish skill level → approximate ELO (Stockfish 16 calibration)
SKILL_ELO = [
    800, 900, 1000, 1100, 1200, 1300, 1400, 1500,
    1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300,
    2400, 2500, 2600, 2700, 2800,
]  # index = skill level 0..20


def play_vs_stockfish(
    network, cfg: DictConfig, device: str,
    stockfish_path: str, skill_level: int,
    move_time: float, network_is_white: bool,
) -> float:
    board = chess.Board()
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    engine.configure({"Skill Level": skill_level})

    try:
        move_count = 0
        while not board.is_game_over(claim_draw=True) and move_count < cfg.self_play.max_moves:
            if (board.turn == chess.WHITE) == network_is_white:
                action_probs, _ = run_mcts(board, network, cfg, device, add_noise=False)
                action = select_action(action_probs, temperature=0.1)
                move = action_to_move(action, board)
                if move not in board.legal_moves:
                    legal = [(action_probs[move_to_action(m, board)], m) for m in board.legal_moves]
                    legal.sort(reverse=True)
                    move = legal[0][1]
            else:
                move = engine.play(board, chess.engine.Limit(time=move_time)).move

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


def _play_n(network, cfg, device, sf_path, skill, move_time, n) -> tuple[int, int, int]:
    wins, draws, losses = 0, 0, 0
    half = n // 2
    for i in range(n):
        r = play_vs_stockfish(network, cfg, device, sf_path, skill, move_time, i < half)
        if r > 0:   wins += 1
        elif r < 0:  losses += 1
        else:        draws += 1
    return wins, draws, losses


@torch.no_grad()
def eval_vs_stockfish(
    network, cfg: DictConfig, device: str,
    num_games: int = 4, skill_level: int = 0,
    stockfish_path: str = "stockfish", move_time: float = 0.01,
) -> dict:
    """Play num_games at a single skill level."""
    network.eval()
    w, d, l = _play_n(network, cfg, device, stockfish_path, skill_level, move_time, num_games)
    total = w + d + l
    wr = (w + 0.5 * d) / total if total > 0 else 0.0
    return {
        "win_rate": wr, "wins": w, "draws": d, "losses": l,
        "skill_level": skill_level, "elo_estimate": SKILL_ELO[skill_level],
    }


@torch.no_grad()
def estimate_elo(
    network, cfg: DictConfig, device: str,
    games_per_level: int = 4,
    stockfish_path: str = "stockfish",
    move_time: float = 0.01,
    max_skill: int = 10,
) -> dict:
    """
    Sweep Stockfish skill levels upward until win rate drops below 50%.
    Interpolate to estimate the network's ELO.
    """
    network.eval()
    results = []

    for skill in range(0, max_skill + 1):
        w, d, l = _play_n(network, cfg, device, stockfish_path, skill, move_time, games_per_level)
        total = w + d + l
        wr = (w + 0.5 * d) / total if total > 0 else 0.0
        elo = SKILL_ELO[skill]
        results.append({"skill": skill, "elo": elo, "wr": wr, "w": w, "d": d, "l": l})
        print(f"  Skill {skill:>2} (~{elo} ELO): W={w} D={d} L={l} ({wr:.0%})")

        # Stop early if clearly losing
        if wr == 0 and skill >= 2:
            break

    # Find crossover: last level where wr >= 50%
    estimated_elo = SKILL_ELO[0]  # floor
    for i, r in enumerate(results):
        if r["wr"] >= 0.5:
            estimated_elo = r["elo"]
        elif i > 0:
            # Linear interpolation between this level and the previous
            prev = results[i - 1]
            if prev["wr"] > r["wr"]:
                frac = (0.5 - r["wr"]) / (prev["wr"] - r["wr"])
                estimated_elo = r["elo"] - frac * (r["elo"] - prev["elo"])
            break

    return {
        "estimated_elo": round(estimated_elo),
        "levels": results,
    }
