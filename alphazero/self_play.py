import os
import chess
import numpy as np
import torch
import torch.multiprocessing as mp
from omegaconf import DictConfig
from tqdm import tqdm

from .encode import encode_board, move_to_action, action_to_move, ACTION_SPACE
from .mcts import MCTSNode, add_dirichlet_noise, select_action


class ParallelGame:
    __slots__ = ("board", "history", "move_count", "done", "root", "sims_done")

    def __init__(self):
        self.board = chess.Board()
        self.history: list[tuple[np.ndarray, np.ndarray]] = []
        self.move_count = 0
        self.done = False
        self.root: MCTSNode | None = None
        self.sims_done = 0


def _game_result(board: chess.Board) -> float:
    result = board.result(claim_draw=True)
    return {"1-0": 1.0, "0-1": -1.0}.get(result, 0.0)


def _game_result_from_node(board: chess.Board) -> float:
    result = board.result()
    if result == "1-0":
        return 1.0 if board.turn == chess.BLACK else -1.0
    elif result == "0-1":
        return 1.0 if board.turn == chess.WHITE else -1.0
    return 0.0


def _pick_move(action_probs: np.ndarray, board: chess.Board, temperature: float) -> chess.Move:
    action = select_action(action_probs, temperature=temperature)
    move = action_to_move(action, board)
    if move in board.legal_moves:
        return move
    legal = [(action_probs[move_to_action(m, board)], m) for m in board.legal_moves]
    legal.sort(reverse=True)
    return legal[0][1]


def _worker_play_games(network, cfg: DictConfig, device: str, num_games: int) -> list:
    """Run a batch of games in a single worker process."""
    torch.set_num_threads(1)
    network.eval()
    all_data = []
    num_parallel = min(cfg.self_play.parallel_games, num_games)
    games_spawned = 0

    c_puct = cfg.mcts.c_puct
    num_sims = cfg.mcts.simulations
    batch_sz = cfg.mcts.batch_size
    max_moves = cfg.self_play.max_moves
    temp_threshold = cfg.self_play.temp_threshold

    def spawn(n):
        nonlocal games_spawned
        out = []
        for _ in range(min(n, num_games - games_spawned)):
            out.append(ParallelGame())
            games_spawned += 1
        return out

    games = spawn(num_parallel)

    with torch.no_grad():
        while games:
            needs_root = [g for g in games if not g.done and g.root is None]
            if needs_root:
                boards = np.stack([encode_board(g.board) for g in needs_root])
                tensor = torch.from_numpy(boards).to(device)
                logits, _ = network(tensor)
                policies = torch.softmax(logits, dim=-1).cpu().numpy()

                for g, policy in zip(needs_root, policies):
                    g.root = MCTSNode(g.board.copy(stack=False))
                    g.root.expand(policy)
                    add_dirichlet_noise(g.root, cfg.mcts.dirichlet_alpha, cfg.mcts.dirichlet_epsilon)
                    g.sims_done = 0

            active = [g for g in games if not g.done and g.root is not None and g.sims_done < num_sims]

            while active:
                all_leaves = []
                leaf_to_game = []

                for g in active:
                    for _ in range(min(batch_sz, num_sims - g.sims_done)):
                        leaf = g.root.select_leaf(c_puct)
                        if leaf.board.is_game_over():
                            leaf.backup(_game_result_from_node(leaf.board))
                            g.sims_done += 1
                        else:
                            leaf.add_virtual_loss()
                            all_leaves.append(leaf)
                            leaf_to_game.append(g)

                if all_leaves:
                    boards = np.stack([encode_board(l.board) for l in all_leaves])
                    tensor = torch.from_numpy(boards).to(device)
                    logits, values = network(tensor)
                    policies = torch.softmax(logits, dim=-1).cpu().numpy()
                    vals = values.cpu().numpy()

                    for leaf, policy, v, g in zip(all_leaves, policies, vals, leaf_to_game):
                        leaf.revert_virtual_loss()
                        leaf.expand(policy)
                        leaf.backup(v)
                        g.sims_done += 1

                active = [g for g in active if g.sims_done < num_sims]

            newly_done = []
            for g in games:
                if g.done or g.root is None or g.sims_done < num_sims:
                    continue

                action_probs = np.zeros(ACTION_SPACE, dtype=np.float32)
                for child in g.root.children:
                    action_probs[child.action] = child.visit_count
                total = action_probs.sum()
                if total > 0:
                    action_probs /= total

                g.history.append((encode_board(g.board), action_probs))

                temp = 1.0 if g.move_count < temp_threshold else 0.1
                move = _pick_move(action_probs, g.board, temp)
                g.board.push(move)
                g.move_count += 1
                g.root = None

                if g.board.is_game_over(claim_draw=True) or g.move_count >= max_moves:
                    g.done = True
                    newly_done.append(g)

            for g in newly_done:
                z = _game_result(g.board)
                for i, (state, pi) in enumerate(g.history):
                    value = z if (i % 2 == 0) else -z
                    all_data.append((state, pi, value))

            games = [g for g in games if not g.done]
            games.extend(spawn(num_parallel - len(games)))

    return all_data


def run_self_play(network, cfg: DictConfig, device: str):
    total_games = cfg.self_play.games
    num_workers = min(cfg.self_play.get("num_workers", 4), total_games)

    if num_workers <= 1:
        data = _worker_play_games(network, cfg, device, total_games)
        print(f"  Self-play: {len(data)} positions (1 worker)")
        return data

    network.share_memory()

    games_per_worker = [total_games // num_workers] * num_workers
    for i in range(total_games % num_workers):
        games_per_worker[i] += 1

    # Per-worker parallel_games to keep each worker's batch reasonable
    worker_cfg = cfg.copy()

    ctx = mp.get_context("spawn")
    with ctx.Pool(num_workers) as pool:
        results = pool.starmap(
            _worker_play_games,
            [(network, cfg, device, g) for g in games_per_worker],
        )

    all_data = []
    for r in results:
        all_data.extend(r)

    return all_data
