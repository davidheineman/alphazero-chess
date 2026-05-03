import copy
import os
import sys
import time
import torch
import wandb
from omegaconf import OmegaConf

from alphazero.config import load_config
from alphazero.network import AlphaZeroNet
from alphazero.self_play import run_self_play
from alphazero.trainer import ReplayBuffer, train_network
from alphazero.arena import evaluate
from alphazero.stockfish_eval import estimate_elo


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
    print(OmegaConf.to_yaml(cfg))

    ckpt_dir = cfg.run.checkpoint_dir
    os.makedirs(ckpt_dir, exist_ok=True)

    network = AlphaZeroNet(cfg.network.res_blocks, cfg.network.channels).to(device)

    if cfg.run.pretrain_path:
        print(f"Loading pretrained: {cfg.run.pretrain_path}")
        network.load_state_dict(torch.load(cfg.run.pretrain_path, map_location=device, weights_only=True))

    replay_buffer = ReplayBuffer(cfg.train.replay_buffer_size)

    param_count = sum(p.numel() for p in network.parameters())
    print(f"Parameters: {param_count:,}\n")

    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        config=OmegaConf.to_container(cfg, resolve=True),
        name=f"az-{cfg.network.res_blocks}b{cfg.network.channels}c-{cfg.mcts.simulations}s",
    )

    optimizer = torch.optim.Adam(
        network.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )
    best_net = copy.deepcopy(network)

    for iteration in range(1, cfg.run.iterations + 1):
        print(f"{'='*50}")
        print(f"ITERATION {iteration}/{cfg.run.iterations}")
        print(f"{'='*50}")

        # Self-play
        t0 = time.time()
        new_data = run_self_play(network, cfg, device)
        replay_buffer.push(new_data)
        sp_time = time.time() - t0
        avg_len = len(new_data) / cfg.self_play.games
        print(f"  Self-play: {len(new_data)} positions, {sp_time:.1f}s (buffer: {len(replay_buffer)})")

        # Train
        t0 = time.time()
        losses = train_network(network, optimizer, replay_buffer, cfg, device)
        tr_time = time.time() - t0
        print(f"  Train: loss={losses['loss']:.4f} (p={losses['policy_loss']:.4f} v={losses['value_loss']:.4f}) {tr_time:.1f}s")

        # Eval (tracking only)
        t0 = time.time()
        win_rate, w, d, l = evaluate(network, best_net, cfg, device)
        ev_time = time.time() - t0
        print(f"  Eval: W={w} D={d} L={l} (win_rate={win_rate:.1%}) {ev_time:.1f}s")

        best_net = copy.deepcopy(network)
        torch.save(network.state_dict(), f"{ckpt_dir}/best.pt")
        torch.save(network.state_dict(), f"{ckpt_dir}/iter_{iteration:04d}.pt")

        log_data = {
            "iteration": iteration,
            "self_play/positions": len(new_data),
            "self_play/avg_game_length": avg_len,
            "self_play/time_s": sp_time,
            "self_play/buffer_size": len(replay_buffer),
            "train/loss": losses["loss"],
            "train/policy_loss": losses["policy_loss"],
            "train/value_loss": losses["value_loss"],
            "train/lr": losses["lr"],
            "train/grad_steps": losses["grad_steps"],
            "train/time_s": tr_time,
            "eval/win_rate": win_rate,
            "eval/wins": w,
            "eval/draws": d,
            "eval/losses": l,
            "eval/time_s": ev_time,
        }

        # Stockfish ELO estimation
        sf = cfg.get("stockfish", {})
        if sf.get("enabled", False) and iteration % sf.get("every_n_iters", 5) == 0:
            t0 = time.time()
            print(f"  Stockfish ELO sweep:")
            elo_result = estimate_elo(
                network, cfg, device,
                games_per_level=sf.get("games", 4),
                stockfish_path=sf.get("path", "stockfish"),
                move_time=sf.get("move_time", 0.01),
                max_skill=sf.get("max_skill", 5),
            )
            sf_time = time.time() - t0
            print(f"  Estimated ELO: {elo_result['estimated_elo']} ({sf_time:.1f}s)")
            log_data["stockfish/estimated_elo"] = elo_result["estimated_elo"]
            log_data["stockfish/time_s"] = sf_time
            for lvl in elo_result["levels"]:
                log_data[f"stockfish/skill_{lvl['skill']}_wr"] = lvl["wr"]

        wandb.log(log_data)
        print()

    wandb.finish()
    print(f"Done. Best model: {ckpt_dir}/best.pt")


if __name__ == "__main__":
    main()
