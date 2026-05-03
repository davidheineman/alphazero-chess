import time
import chess
import numpy as np
import torch
from tqdm import tqdm

from .encode import encode_board, get_legal_move_mask, move_to_action, action_to_move, ACTION_SPACE
from .mcts import MCTSNode, add_dirichlet_noise, select_action
from .config import AlphaZeroConfig


class ParallelGame:
    __slots__ = ("board", "history", "move_count", "done", "root",
                 "sims_done", "pending_leaves")

    def __init__(self):
        self.board = chess.Board()
        self.history: list[tuple[np.ndarray, np.ndarray]] = []
        self.move_count = 0
        self.done = False
        self.root: MCTSNode | None = None
        self.sims_done = 0
        self.pending_leaves: list[MCTSNode] = []


@torch.no_grad()
def run_self_play(network, config: AlphaZeroConfig):
    network.eval()
    all_data = []
    total_games = config.num_self_play_games
    num_parallel = min(32, total_games)
    completed = 0

    pbar = tqdm(total=total_games, desc="Self-play")

    games: list[ParallelGame] = []
    games_spawned = 0

    def spawn_games(n):
        nonlocal games_spawned
        new = []
        for _ in range(n):
            if games_spawned >= total_games:
                break
            new.append(ParallelGame())
            games_spawned += 1
        return new

    games = spawn_games(num_parallel)

    while games:
        # Phase 1: for games that need a new MCTS root, collect boards for root expansion
        needs_root = [g for g in games if not g.done and g.root is None]
        if needs_root:
            boards = np.stack([encode_board(g.board) for g in needs_root])
            tensor = torch.from_numpy(boards).to(config.device)
            logits, _ = network(tensor)
            policies = torch.softmax(logits, dim=-1).cpu().numpy()

            for g, policy in zip(needs_root, policies):
                g.root = MCTSNode(g.board.copy(stack=False))
                g.root.expand(policy)
                add_dirichlet_noise(g.root, config.dirichlet_alpha, config.dirichlet_epsilon)
                g.sims_done = 0

        # Phase 2: run MCTS simulations across all active games, batching evals
        active = [g for g in games if not g.done and g.root is not None and g.sims_done < config.num_simulations]

        while active:
            all_leaves = []
            leaf_to_game = []

            for g in active:
                batch_sz = min(config.mcts_batch_size, config.num_simulations - g.sims_done)
                for _ in range(batch_sz):
                    leaf = g.root.select_leaf(config.c_puct)
                    if leaf.board.is_game_over():
                        result = leaf.board.result()
                        if result == "1-0":
                            v = 1.0 if leaf.board.turn == chess.BLACK else -1.0
                        elif result == "0-1":
                            v = 1.0 if leaf.board.turn == chess.WHITE else -1.0
                        else:
                            v = 0.0
                        leaf.backup(v)
                        g.sims_done += 1
                    else:
                        leaf.add_virtual_loss()
                        all_leaves.append(leaf)
                        leaf_to_game.append(g)

            if all_leaves:
                boards = np.stack([encode_board(l.board) for l in all_leaves])
                tensor = torch.from_numpy(boards).to(config.device)
                logits, values = network(tensor)
                policies = torch.softmax(logits, dim=-1).cpu().numpy()
                vals = values.cpu().numpy()

                for leaf, policy, v, g in zip(all_leaves, policies, vals, leaf_to_game):
                    leaf.revert_virtual_loss()
                    leaf.expand(policy)
                    leaf.backup(v)
                    g.sims_done += 1

            active = [g for g in active if g.sims_done < config.num_simulations]

        # Phase 3: pick moves for all games that finished MCTS
        newly_done = []
        for g in games:
            if g.done or g.root is None or g.sims_done < config.num_simulations:
                continue

            action_probs = np.zeros(ACTION_SPACE, dtype=np.float32)
            for child in g.root.children:
                action_probs[child.action] = child.visit_count
            total = action_probs.sum()
            if total > 0:
                action_probs /= total

            state_planes = encode_board(g.board)
            g.history.append((state_planes, action_probs))

            temp = 1.0 if g.move_count < config.temp_threshold else 0.1
            action = select_action(action_probs, temperature=temp)
            move = action_to_move(action, g.board)
            if move not in g.board.legal_moves:
                legal_actions = []
                for m in g.board.legal_moves:
                    a = move_to_action(m, g.board)
                    legal_actions.append((action_probs[a], a, m))
                legal_actions.sort(reverse=True)
                move = legal_actions[0][2]

            g.board.push(move)
            g.move_count += 1
            g.root = None

            if g.board.is_game_over(claim_draw=True) or g.move_count >= config.max_moves:
                g.done = True
                newly_done.append(g)

        # Phase 4: collect data from finished games, replace with new ones
        for g in newly_done:
            result = g.board.result(claim_draw=True)
            if result == "1-0":
                z = 1.0
            elif result == "0-1":
                z = -1.0
            else:
                z = 0.0

            for i, (state, pi) in enumerate(g.history):
                value = z if (i % 2 == 0) else -z
                all_data.append((state, pi, value))

            completed += 1
            pbar.update(1)
            pbar.set_postfix(moves=g.move_count, total=len(all_data))

        games = [g for g in games if not g.done]
        new = spawn_games(num_parallel - len(games))
        games.extend(new)

    pbar.close()
    return all_data
