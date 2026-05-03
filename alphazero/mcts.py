import math
import chess
import numpy as np
import torch
from omegaconf import DictConfig

from .encode import (
    encode_board,
    get_legal_move_mask,
    move_to_action,
    ACTION_SPACE,
)

VIRTUAL_LOSS = 3.0


class MCTSNode:
    __slots__ = ("board", "parent", "action", "children",
                 "visit_count", "value_sum", "prior", "is_expanded",
                 "virtual_loss_count", "_move", "_parent_board")

    def __init__(self, board: chess.Board = None, parent=None, action: int = -1,
                 prior: float = 0.0, move: chess.Move = None, parent_board: chess.Board = None):
        self.board = board
        self.parent = parent
        self.action = action
        self.prior = prior
        self.children: list["MCTSNode"] = []
        self.visit_count = 0
        self.value_sum = 0.0
        self.is_expanded = False
        self.virtual_loss_count = 0
        self._move = move
        self._parent_board = parent_board

    def ensure_board(self):
        if self.board is None and self._parent_board is not None:
            self.board = self._parent_board.copy(stack=False)
            self.board.push(self._move)
            self._parent_board = None
            self._move = None

    @property
    def q_value(self) -> float:
        total_visits = self.visit_count + self.virtual_loss_count
        if total_visits == 0:
            return 0.0
        return (self.value_sum - self.virtual_loss_count * VIRTUAL_LOSS) / total_visits

    def ucb_score(self, c_puct: float, parent_visits: int) -> float:
        total_visits = self.visit_count + self.virtual_loss_count
        exploration = c_puct * self.prior * math.sqrt(parent_visits) / (1 + total_visits)
        return self.q_value + exploration

    def best_child(self, c_puct: float) -> "MCTSNode":
        pv = self.visit_count + self.virtual_loss_count
        best = None
        best_score = -1e9
        for c in self.children:
            s = c.ucb_score(c_puct, pv)
            if s > best_score:
                best_score = s
                best = c
        return best

    def select_leaf(self, c_puct: float) -> "MCTSNode":
        node = self
        while node.is_expanded and node.children:
            node = node.best_child(c_puct)
        node.ensure_board()
        return node

    def expand(self, policy: np.ndarray):
        self.is_expanded = True
        legal_mask = get_legal_move_mask(self.board)
        masked = policy * legal_mask
        total = masked.sum()
        if total > 0:
            masked /= total
        else:
            masked = legal_mask / legal_mask.sum()

        for move in self.board.legal_moves:
            action = move_to_action(move, self.board)
            child = MCTSNode(
                parent=self, action=action, prior=masked[action],
                move=move, parent_board=self.board,
            )
            self.children.append(child)

    def add_virtual_loss(self):
        node = self
        while node is not None:
            node.virtual_loss_count += 1
            node = node.parent

    def revert_virtual_loss(self):
        node = self
        while node is not None:
            node.virtual_loss_count -= 1
            node = node.parent

    def backup(self, value: float):
        node = self
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            value = -value
            node = node.parent


def add_dirichlet_noise(node: MCTSNode, alpha: float, epsilon: float):
    if not node.children:
        return
    noise = np.random.dirichlet([alpha] * len(node.children))
    for child, n in zip(node.children, noise):
        child.prior = (1 - epsilon) * child.prior + epsilon * n


def _batch_evaluate(leaves: list[MCTSNode], network: torch.nn.Module, device: str):
    boards = np.stack([encode_board(leaf.board) for leaf in leaves])
    tensor = torch.from_numpy(boards).to(device)
    policy_logits, values = network(tensor)
    policies = torch.softmax(policy_logits, dim=-1).cpu().numpy()
    values = values.cpu().numpy()
    return policies, values


def _terminal_value(board: chess.Board) -> float:
    result = board.result()
    if result == "1-0":
        return 1.0 if board.turn == chess.BLACK else -1.0
    elif result == "0-1":
        return 1.0 if board.turn == chess.WHITE else -1.0
    return 0.0


@torch.no_grad()
def run_mcts(
    board: chess.Board,
    network: torch.nn.Module,
    cfg: DictConfig,
    device: str,
    add_noise: bool = True,
) -> tuple[np.ndarray, MCTSNode]:
    network.eval()
    root = MCTSNode(board=board.copy(stack=False))

    root_boards = np.expand_dims(encode_board(board), 0)
    root_tensor = torch.from_numpy(root_boards).to(device)
    policy_logits, _ = network(root_tensor)
    policy = torch.softmax(policy_logits, dim=-1).cpu().numpy().flatten()
    root.expand(policy)

    if add_noise:
        add_dirichlet_noise(root, cfg.mcts.dirichlet_alpha, cfg.mcts.dirichlet_epsilon)

    sims_done = 0
    num_sims = cfg.mcts.simulations
    batch_sz = cfg.mcts.batch_size
    c_puct = cfg.mcts.c_puct

    while sims_done < num_sims:
        cur_batch = min(batch_sz, num_sims - sims_done)
        leaves = []
        terminal_leaves = []

        for _ in range(cur_batch):
            leaf = root.select_leaf(c_puct)

            if leaf.board.is_game_over():
                terminal_leaves.append((leaf, _terminal_value(leaf.board)))
            else:
                leaf.add_virtual_loss()
                leaves.append(leaf)

        for leaf, v in terminal_leaves:
            leaf.backup(v)
            sims_done += 1

        if leaves:
            policies, values = _batch_evaluate(leaves, network, device)
            for i, leaf in enumerate(leaves):
                leaf.revert_virtual_loss()
                leaf.expand(policies[i])
                leaf.backup(values[i])
                sims_done += 1

    action_probs = np.zeros(ACTION_SPACE, dtype=np.float32)
    for child in root.children:
        action_probs[child.action] = child.visit_count

    total = action_probs.sum()
    if total > 0:
        action_probs /= total

    return action_probs, root


def select_action(action_probs: np.ndarray, temperature: float = 1.0) -> int:
    if temperature < 1e-3:
        return int(np.argmax(action_probs))

    probs = action_probs ** (1.0 / temperature)
    total = probs.sum()
    if total == 0:
        nonzero = action_probs > 0
        probs = nonzero.astype(np.float32)
        probs /= probs.sum()
    else:
        probs /= total

    return int(np.random.choice(len(probs), p=probs))
