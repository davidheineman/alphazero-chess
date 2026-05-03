import argparse
import io
import os
import random
import time
import urllib.request

import chess
import chess.pgn
import numpy as np
import torch
import torch.nn.functional as F
import zstandard as zstd
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb

from alphazero.config import AlphaZeroConfig
from alphazero.encode import encode_board, move_to_action, ACTION_SPACE
from alphazero.network import AlphaZeroNet
from alphazero.trainer import ChessDataset

# Lichess monthly DBs: https://database.lichess.org/
DEFAULT_URL = "https://database.lichess.org/standard/lichess_db_standard_rated_2013-01.pgn.zst"


def download_pgn(url: str, dest: str):
    if os.path.exists(dest):
        print(f"  Already downloaded: {dest}")
        return
    print(f"  Downloading {url}...")
    urllib.request.urlretrieve(url, dest, reporthook=lambda b, bs, t: None)
    print(f"  Saved to {dest}")


def stream_games(pgn_path: str, min_elo: int = 1500):
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

        white_elo = game.headers.get("WhiteElo", "?")
        black_elo = game.headers.get("BlackElo", "?")
        if white_elo == "?" or black_elo == "?":
            continue
        if int(white_elo) < min_elo or int(black_elo) < min_elo:
            continue

        yield game, result


def extract_positions(pgn_path: str, max_positions: int, min_elo: int = 1500, sample_rate: float = 0.25):
    data = []
    games_seen = 0

    for game, result in stream_games(pgn_path, min_elo):
        if result == "1-0":
            z = 1.0
        elif result == "0-1":
            z = -1.0
        else:
            z = 0.0

        board = game.board()
        for move in game.mainline_moves():
            if random.random() > sample_rate:
                board.push(move)
                continue

            state = encode_board(board)
            policy = np.zeros(ACTION_SPACE, dtype=np.float32)
            action = move_to_action(move, board)
            policy[action] = 1.0

            value = z if board.turn == chess.WHITE else -z
            data.append((state, policy, value))

            board.push(move)

            if len(data) >= max_positions:
                return data, games_seen

        games_seen += 1
        if games_seen % 1000 == 0:
            print(f"  Parsed {games_seen} games, {len(data)} positions...")

    return data, games_seen


def pretrain(network, data, config: AlphaZeroConfig, epochs: int = 5, lr: float = 0.001):
    network.train()
    optimizer = torch.optim.Adam(network.parameters(), lr=lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    dataset = ChessDataset(data)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
    global_step = 0

    for epoch in range(epochs):
        total_loss = 0.0
        total_ploss = 0.0
        total_vloss = 0.0
        n = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
        for states, target_pis, target_vs in pbar:
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
            total_ploss += policy_loss.item()
            total_vloss += value_loss.item()
            n += 1
            global_step += 1
            pbar.set_postfix(loss=f"{total_loss/n:.3f}", p=f"{total_ploss/n:.3f}", v=f"{total_vloss/n:.3f}")

            if global_step % 50 == 0:
                wandb.log({
                    "pretrain/loss": loss.item(),
                    "pretrain/policy_loss": policy_loss.item(),
                    "pretrain/value_loss": value_loss.item(),
                    "pretrain/step": global_step,
                })

        scheduler.step()
        wandb.log({
            "pretrain/epoch": epoch + 1,
            "pretrain/epoch_loss": total_loss / n,
            "pretrain/epoch_policy_loss": total_ploss / n,
            "pretrain/epoch_value_loss": total_vloss / n,
            "pretrain/lr": scheduler.get_last_lr()[0],
        })
        print(f"  Epoch {epoch+1}: loss={total_loss/n:.4f} policy={total_ploss/n:.4f} value={total_vloss/n:.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pgn", type=str, help="Path to .pgn or .pgn.zst file")
    p.add_argument("--url", type=str, default=DEFAULT_URL, help="Lichess DB URL to download")
    p.add_argument("--max-positions", type=int, default=500_000)
    p.add_argument("--min-elo", type=int, default=1500)
    p.add_argument("--sample-rate", type=float, default=0.25)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--res-blocks", type=int, default=10)
    p.add_argument("--channels", type=int, default=128)
    p.add_argument("--output", type=str, default="checkpoints/pretrained.pt")
    p.add_argument("--wandb-project", type=str, default="mcts")
    p.add_argument("--wandb-entity", type=str, default="bobcrables")
    args = p.parse_args()

    config = AlphaZeroConfig(
        num_res_blocks=args.res_blocks,
        num_channels=args.channels,
        batch_size=args.batch_size,
    )

    if torch.cuda.is_available():
        config.device = "cuda"
    elif torch.backends.mps.is_available():
        config.device = "mps"
    print(f"Using device: {config.device}")

    # Get PGN file
    if args.pgn:
        pgn_path = args.pgn
    else:
        os.makedirs("data", exist_ok=True)
        filename = args.url.split("/")[-1]
        pgn_path = f"data/{filename}"
        download_pgn(args.url, pgn_path)

    # Extract positions
    print(f"\nExtracting positions from {pgn_path}...")
    t0 = time.time()
    data, games_seen = extract_positions(pgn_path, args.max_positions, args.min_elo, args.sample_rate)
    print(f"  Extracted {len(data)} positions from {games_seen} games in {time.time()-t0:.1f}s")

    # Train
    network = AlphaZeroNet(config.num_res_blocks, config.num_channels).to(config.device)
    param_count = sum(p.numel() for p in network.parameters())
    print(f"\nNetwork: {param_count:,} parameters")
    print(f"Training for {args.epochs} epochs...\n")

    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config={
            "stage": "pretrain",
            "num_res_blocks": args.res_blocks,
            "num_channels": args.channels,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "max_positions": args.max_positions,
            "min_elo": args.min_elo,
            "parameters": param_count,
            "num_positions": len(data),
            "num_games": games_seen,
        },
        name=f"pretrain-{args.res_blocks}b{args.channels}c-{len(data)//1000}k",
    )

    pretrain(network, data, config, epochs=args.epochs, lr=args.lr)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(network.state_dict(), args.output)
    wandb.finish()
    print(f"\nSaved pretrained model to {args.output}")


if __name__ == "__main__":
    main()
