import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.data import Dataset, DataLoader


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


def train_network(network, optimizer, replay_buffer: ReplayBuffer, cfg: DictConfig, device: str):
    if len(replay_buffer) < cfg.train.batch_size:
        return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "lr": cfg.train.lr, "grad_steps": 0}

    network.train()

    dataset = ChessDataset(replay_buffer.buffer)
    loader = DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=True, drop_last=True)

    total_loss = 0.0
    total_ploss = 0.0
    total_vloss = 0.0
    n = 0

    for epoch in range(cfg.train.epochs):
        for states, target_pis, target_vs in loader:
            states = states.to(device)
            target_pis = target_pis.to(device)
            target_vs = target_vs.to(device)

            policy_logits, pred_vs = network(states)

            value_loss = F.mse_loss(pred_vs, target_vs)
            log_probs = F.log_softmax(policy_logits, dim=-1)
            policy_loss = -(target_pis * log_probs).sum(dim=-1).mean()
            loss = value_loss + policy_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_ploss += policy_loss.item()
            total_vloss += value_loss.item()
            n += 1

    n = max(n, 1)
    return {
        "loss": total_loss / n,
        "policy_loss": total_ploss / n,
        "value_loss": total_vloss / n,
        "lr": optimizer.param_groups[0]["lr"],
        "grad_steps": n,
    }
