import chess
from omegaconf import OmegaConf, DictConfig
from tqdm import trange

from .encode import action_to_move, move_to_action
from .mcts import run_mcts, select_action


def play_match(white_net, black_net, cfg: DictConfig, device: str) -> float:
    board = chess.Board()
    move_count = 0

    eval_cfg = OmegaConf.merge(cfg, {"mcts": {"simulations": cfg.eval.simulations}})

    while not board.is_game_over(claim_draw=True) and move_count < cfg.self_play.max_moves:
        net = white_net if board.turn == chess.WHITE else black_net
        action_probs = run_mcts(board.fen(), net, eval_cfg, device, add_noise=False)
        action = select_action(action_probs, temperature=cfg.eval.temperature)

        move = action_to_move(action, board)
        if move not in board.legal_moves:
            legal = [(action_probs[move_to_action(m, board)], m) for m in board.legal_moves]
            legal.sort(reverse=True)
            move = legal[0][1]

        board.push(move)
        move_count += 1

    result = board.result(claim_draw=True)
    if result == "1-0":
        return 1.0
    elif result == "0-1":
        return -1.0
    return 0.0


def evaluate(new_net, old_net, cfg: DictConfig, device: str) -> tuple[float, int, int, int]:
    wins, draws, losses = 0, 0, 0
    half = cfg.eval.games // 2

    for i in trange(cfg.eval.games, desc="Evaluation"):
        if i < half:
            r = play_match(new_net, old_net, cfg, device)
            wins += r > 0; draws += r == 0; losses += r < 0
        else:
            r = play_match(old_net, new_net, cfg, device)
            wins += r < 0; draws += r == 0; losses += r > 0

    total = wins + draws + losses
    win_rate = (wins + 0.5 * draws) / total if total > 0 else 0.0
    return win_rate, wins, draws, losses
