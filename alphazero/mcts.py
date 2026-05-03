import numpy as np
import torch
from omegaconf import DictConfig

from .encode import ACTION_SPACE

try:
    import mcts_cpp
    HAS_CPP = True
except ImportError:
    HAS_CPP = False


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


@torch.no_grad()
def run_mcts(
    fen: str,
    network: torch.nn.Module,
    cfg: DictConfig,
    device: str,
    add_noise: bool = True,
) -> np.ndarray:
    """Run MCTS from FEN. Returns action probability distribution."""
    network.eval()

    search = mcts_cpp.MCTSSearch(
        fen, cfg.mcts.c_puct, cfg.mcts.simulations, cfg.mcts.batch_size,
        cfg.mcts.dirichlet_alpha if add_noise else 0.0,
        cfg.mcts.dirichlet_epsilon if add_noise else 0.0,
    )

    # Root expansion
    root_enc = torch.from_numpy(search.encode_root()).to(device)
    logits, _ = network(root_enc)
    policy = torch.softmax(logits, dim=-1).cpu().numpy().flatten()
    search.expand_root(policy)

    # MCTS simulations
    while not search.is_complete():
        leaves = search.select_leaves()
        if leaves.shape[0] == 0:
            continue
        tensor = torch.from_numpy(leaves).to(device)
        logits, values = network(tensor)
        policies = torch.softmax(logits, dim=-1).cpu().numpy()
        vals = values.cpu().numpy()
        search.process_results(policies, vals)

    return search.get_action_probs()
