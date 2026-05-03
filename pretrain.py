import io
import os
import random
import sys
import time
import urllib.request

import chess
import chess.pgn
import numpy as np
import torch
import torch.nn.functional as F
import wandb
import zstandard as zstd
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from alphazero.config import load_config
from alphazero.encode import encode_board, move_to_action, ACTION_SPACE
from alphazero.network import AlphaZeroNet
from alphazero.trainer import ChessDataset


def download_pgn(url: str, dest: str):
    if os.path.exists(dest):
        print(f"  Already downloaded: {dest}")
        return
    print(f"  Downloading {url}...")
    urllib.request.urlretrieve(url, dest)
    print(f"  Saved to {dest}")


def stream_games(pgn_path: str, min_elo: int):
    if pgn_path.endswith(".zst"):
        dctx = zstd.ZstdDecompressor()
        with open(pgn_path, "rb") as f:
            reader = dctx.stream_reader(f)
            text_stream = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
            yield from _parse_games(text_stream, min_elo)
    else:
        with open(pgn_path, "r") as f:
            yield from _parse_games(f, min_elo)


def _parse_games(text_stream, min_elo: int):
    while True:
        game = chess.pgn.read_game(text_stream)
        if game is None:
            break
        result = game.headers.get("Result", "*")
        if result not in ("1-0", "0-1", "1/2-1/2"):
            continue
        try:
            if int(game.headers.get("WhiteElo", "0")) < min_elo:
                continue
            if int(game.headers.get("BlackElo", "0")) < min_elo:
                continue
        except ValueError:
            continue
        yield game, result


def extract_positions(pgn_path: str, max_pos: int, min_elo: int, sample_rate: float):
    data = []
    games_seen = 0

    for game, result in stream_games(pgn_path, min_elo):
        z = {"1-0": 1.0, "0-1": -1.0}.get(result, 0.0)
        board = game.board()

        for move in game.mainline_moves():
            if random.random() < sample_rate:
                state = encode_board(board)
                policy = np.zeros(ACTION_SPACE, dtype=np.float32)
                policy[move_to_action(move, board)] = 1.0
                value = z if board.turn == chess.WHITE else -z
                data.append((state, policy, value))

                if len(data) >= max_pos:
                    return data, games_seen
            board.push(move)

        games_seen += 1
        if games_seen % 1000 == 0:
            print(f"  {games_seen} games, {len(data)} positions...")

    return data, games_seen


def pretrain(network, data, cfg, device: str):
    network.train()
    optimizer = torch.optim.Adam(network.parameters(), lr=cfg.pretrain.lr, weight_decay=cfg.train.weight_decay)
    epochs = cfg.pretrain.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    dataset = ChessDataset(data)
    loader = DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=True, drop_last=True)
    step = 0

    for epoch in range(epochs):
        total_loss, total_pl, total_vl, n = 0.0, 0.0, 0.0, 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")

        for states, target_pis, target_vs in pbar:
            states = states.to(device)
            target_pis = target_pis.to(device)
            target_vs = target_vs.to(device)

            logits, pred_vs = network(states)
            vl = F.mse_loss(pred_vs, target_vs)
            pl = -(target_pis * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
            loss = vl + pl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item(); total_pl += pl.item(); total_vl += vl.item(); n += 1; step += 1
            pbar.set_postfix(loss=f"{total_loss/n:.3f}", p=f"{total_pl/n:.3f}", v=f"{total_vl/n:.3f}")

            if step % 50 == 0:
                wandb.log({"pretrain/loss": loss.item(), "pretrain/policy_loss": pl.item(),
                           "pretrain/value_loss": vl.item(), "pretrain/step": step})

        scheduler.step()
        wandb.log({"pretrain/epoch": epoch+1, "pretrain/epoch_loss": total_loss/n,
                   "pretrain/epoch_policy_loss": total_pl/n, "pretrain/epoch_value_loss": total_vl/n})
        print(f"  Epoch {epoch+1}: loss={total_loss/n:.4f} p={total_pl/n:.4f} v={total_vl/n:.4f}")


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    config_files = [a for a in sys.argv[1:] if a.endswith((".yaml", ".yml")) and not a.startswith("-")]
    cfg = load_config(*config_files)
    device = get_device()

    print(f"Device: {device}")

    os.makedirs("data", exist_ok=True)
    filename = cfg.pretrain.url.split("/")[-1]
    pgn_path = f"data/{filename}"
    download_pgn(cfg.pretrain.url, pgn_path)

    print(f"\nExtracting positions...")
    t0 = time.time()
    data, games_seen = extract_positions(pgn_path, cfg.pretrain.max_positions, cfg.pretrain.min_elo, cfg.pretrain.sample_rate)
    print(f"  {len(data)} positions from {games_seen} games in {time.time()-t0:.1f}s")

    network = AlphaZeroNet(cfg.network.res_blocks, cfg.network.channels).to(device)
    params = sum(p.numel() for p in network.parameters())
    print(f"Network: {params:,} params\n")

    wandb.init(
        project=cfg.wandb.project, entity=cfg.wandb.entity,
        config=OmegaConf.to_container(cfg, resolve=True),
        name=f"pretrain-{cfg.network.res_blocks}b{cfg.network.channels}c",
    )

    pretrain(network, data, cfg, device)

    out = cfg.run.checkpoint_dir + "/pretrained.pt"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save(network.state_dict(), out)
    wandb.finish()
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
