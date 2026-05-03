import argparse
import copy
import os
import time
import torch

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

    best_net = copy.deepcopy(network)

    for iteration in range(1, config.num_iterations + 1):
        print(f"{'='*60}")
        print(f"ITERATION {iteration}/{config.num_iterations}")
        print(f"{'='*60}")

        t0 = time.time()
        print(f"\n[1/3] Self-play ({config.num_self_play_games} games, "
              f"{config.num_simulations} sims/move)...")
        new_data = run_self_play(network, config)
        replay_buffer.push(new_data)
        print(f"  Generated {len(new_data)} positions in {time.time()-t0:.1f}s")
        print(f"  Replay buffer: {len(replay_buffer)} positions")

        t0 = time.time()
        print(f"\n[2/3] Training ({config.num_epochs} epochs, "
              f"batch_size={config.batch_size})...")
        avg_loss = train_network(network, replay_buffer, config)
        print(f"  Average loss: {avg_loss:.4f} ({time.time()-t0:.1f}s)")

        t0 = time.time()
        print(f"\n[3/3] Evaluating new network vs best...")
        win_rate, wins, draws, losses = evaluate(network, best_net, config)
        print(f"  Results: W={wins} D={draws} L={losses} "
              f"(win rate: {win_rate:.1%})")

        if win_rate >= config.win_threshold:
            print(f"  >>> New best network! (win rate {win_rate:.1%} >= "
                  f"{config.win_threshold:.1%})")
            best_net = copy.deepcopy(network)
            torch.save(best_net.state_dict(), f"{ckpt_dir}/best.pt")
        else:
            print(f"  Keeping old network (win rate {win_rate:.1%} < "
                  f"{config.win_threshold:.1%})")
            network.load_state_dict(best_net.state_dict())

        torch.save(network.state_dict(), f"{ckpt_dir}/iter_{iteration:04d}.pt")
        print()

    print("Training complete!")
    print(f"Best model saved to {ckpt_dir}/best.pt")


if __name__ == "__main__":
    main()
