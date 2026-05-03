import argparse
import copy
import os
import time
import torch
import wandb

from alphazero.config import AlphaZeroConfig
from alphazero.network import AlphaZeroNet
from alphazero.self_play import run_self_play
from alphazero.trainer import ReplayBuffer, train_network
from alphazero.arena import evaluate


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--res-blocks", type=int)
    p.add_argument("--channels", type=int)
    p.add_argument("--simulations", type=int)
    p.add_argument("--mcts-batch", type=int)
    p.add_argument("--games", type=int)
    p.add_argument("--max-moves", type=int)
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--eval-games", type=int)
    p.add_argument("--iterations", type=int)
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p.add_argument("--pretrain", type=str, help="Path to pretrained checkpoint to start from")
    p.add_argument("--wandb-project", type=str, default="mcts")
    p.add_argument("--wandb-entity", type=str, default="bobcrables")
    return p.parse_args()


def main():
    args = parse_args()
    config = AlphaZeroConfig()

    if args.res_blocks is not None:
        config.num_res_blocks = args.res_blocks
    if args.channels is not None:
        config.num_channels = args.channels
    if args.simulations is not None:
        config.num_simulations = args.simulations
    if args.mcts_batch is not None:
        config.mcts_batch_size = args.mcts_batch
    if args.games is not None:
        config.num_self_play_games = args.games
    if args.max_moves is not None:
        config.max_moves = args.max_moves
    if args.epochs is not None:
        config.num_epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.lr is not None:
        config.learning_rate = args.lr
    if args.eval_games is not None:
        config.num_eval_games = args.eval_games
    if args.iterations is not None:
        config.num_iterations = args.iterations

    if torch.cuda.is_available():
        config.device = "cuda"
    elif torch.backends.mps.is_available():
        config.device = "mps"
    print(f"Using device: {config.device}")

    ckpt_dir = args.checkpoint_dir
    os.makedirs(ckpt_dir, exist_ok=True)

    network = AlphaZeroNet(config.num_res_blocks, config.num_channels).to(config.device)

    if args.pretrain:
        print(f"Loading pretrained checkpoint: {args.pretrain}")
        network.load_state_dict(torch.load(args.pretrain, map_location=config.device, weights_only=True))

    replay_buffer = ReplayBuffer(config.replay_buffer_size)

    param_count = sum(p.numel() for p in network.parameters())
    print(f"Network parameters: {param_count:,}")
    print(f"Config: {config}")
    print()

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config={
            "num_res_blocks": config.num_res_blocks,
            "num_channels": config.num_channels,
            "num_simulations": config.num_simulations,
            "mcts_batch_size": config.mcts_batch_size,
            "num_self_play_games": config.num_self_play_games,
            "max_moves": config.max_moves,
            "num_epochs": config.num_epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "num_iterations": config.num_iterations,
            "pretrained": args.pretrain is not None,
            "parameters": param_count,
        },
        name=f"az-{config.num_res_blocks}b{config.num_channels}c-{config.num_simulations}sims",
    )

    best_net = copy.deepcopy(network)
    best_iteration = 0

    for iteration in range(1, config.num_iterations + 1):
        print(f"{'='*60}")
        print(f"ITERATION {iteration}/{config.num_iterations}")
        print(f"{'='*60}")

        # Self-play
        t0 = time.time()
        print(f"\n[1/3] Self-play ({config.num_self_play_games} games, "
              f"{config.num_simulations} sims/move)...")
        new_data = run_self_play(network, config)
        replay_buffer.push(new_data)
        self_play_time = time.time() - t0
        avg_game_len = len(new_data) / config.num_self_play_games
        print(f"  Generated {len(new_data)} positions in {self_play_time:.1f}s")
        print(f"  Replay buffer: {len(replay_buffer)} positions")

        # Training
        t0 = time.time()
        print(f"\n[2/3] Training ({config.num_epochs} epochs, "
              f"batch_size={config.batch_size})...")
        loss_dict = train_network(network, replay_buffer, config)
        train_time = time.time() - t0
        print(f"  Average loss: {loss_dict['loss']:.4f} ({train_time:.1f}s)")

        # Evaluation (tracking only — always keep latest network)
        t0 = time.time()
        print(f"\n[3/3] Evaluating current vs previous best...")
        win_rate, wins, draws, losses = evaluate(network, best_net, config)
        eval_time = time.time() - t0
        print(f"  Results: W={wins} D={draws} L={losses} "
              f"(win rate: {win_rate:.1%})")

        accepted = win_rate >= config.win_threshold
        if accepted:
            print(f"  >>> New best network!")

        best_net = copy.deepcopy(network)
        best_iteration = iteration
        torch.save(best_net.state_dict(), f"{ckpt_dir}/best.pt")

        torch.save(network.state_dict(), f"{ckpt_dir}/iter_{iteration:04d}.pt")

        wandb.log({
            "iteration": iteration,
            "self_play/positions": len(new_data),
            "self_play/avg_game_length": avg_game_len,
            "self_play/time_s": self_play_time,
            "self_play/replay_buffer_size": len(replay_buffer),
            "train/loss": loss_dict["loss"],
            "train/policy_loss": loss_dict["policy_loss"],
            "train/value_loss": loss_dict["value_loss"],
            "train/time_s": train_time,
            "eval/win_rate": win_rate,
            "eval/wins": wins,
            "eval/draws": draws,
            "eval/losses": losses,
            "eval/accepted": int(accepted),
            "eval/best_iteration": best_iteration,
            "eval/time_s": eval_time,
            "total_time_s": self_play_time + train_time + eval_time,
        })

        print()

    wandb.finish()
    print("Training complete!")
    print(f"Best model saved to {ckpt_dir}/best.pt")


if __name__ == "__main__":
    main()
