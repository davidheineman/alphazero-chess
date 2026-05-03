import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from .config import AlphaZeroConfig


class ReplayBuffer:
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.buffer: list[tuple[np.ndarray, np.ndarray, float]] = []

    def push(self, data: list[tuple[np.ndarray, np.ndarray, float]]):
        self.buffer.extend(data)
        if len(self.buffer) > self.max_size:
            self.buffer = self.buffer[-self.max_size:]

    def __len__(self):
        return len(self.buffer)


class ChessDataset(Dataset):
    def __init__(self, data: list[tuple[np.ndarray, np.ndarray, float]]):
        self.states = np.array([d[0] for d in data], dtype=np.float32)
        self.policies = np.array([d[1] for d in data], dtype=np.float32)
        self.values = np.array([d[2] for d in data], dtype=np.float32)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.states[idx]),
            torch.from_numpy(self.policies[idx]),
            torch.tensor(self.values[idx], dtype=torch.float32),
        )


def train_network(network, replay_buffer: ReplayBuffer, config: AlphaZeroConfig):
    if len(replay_buffer) < config.batch_size:
        return 0.0

    network.train()
    optimizer = torch.optim.Adam(
        network.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    dataset = ChessDataset(replay_buffer.buffer)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)

    total_loss = 0.0
    num_batches = 0

    for epoch in range(config.num_epochs):
        for states, target_pis, target_vs in loader:
            states = states.to(config.device)
            target_pis = target_pis.to(config.device)
            target_vs = target_vs.to(config.device)

            policy_logits, pred_vs = network(states)

            value_loss = F.mse_loss(pred_vs, target_vs)

            log_probs = F.log_softmax(policy_logits, dim=-1)
            policy_loss = -(target_pis * log_probs).sum(dim=-1).mean()

            loss = value_loss + policy_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

    return total_loss / max(num_batches, 1)
