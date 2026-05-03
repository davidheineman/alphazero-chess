import chess
import torch
import numpy as np

from alphazero.config import load_config
from alphazero.encode import encode_board, encode_board_tensor, get_legal_move_mask, move_to_action, action_to_move, ACTION_SPACE
from alphazero.network import AlphaZeroNet
from alphazero.mcts import run_mcts, select_action
from alphazero.trainer import ReplayBuffer, train_network


def test_encoding():
    print("--- Encoding ---")
    board = chess.Board()
    planes = encode_board(board)
    assert planes.shape == (19, 8, 8)

    mask = get_legal_move_mask(board)
    assert int(mask.sum()) == 20

    for move in board.legal_moves:
        action = move_to_action(move, board)
        recovered = action_to_move(action, board)
        assert move == recovered, f"{move} -> {action} -> {recovered}"
    print("  OK")


def test_network():
    print("--- Network ---")
    cfg = load_config()
    net = AlphaZeroNet(cfg.network.res_blocks, cfg.network.channels)
    x = encode_board_tensor(chess.Board())
    logits, value = net(x)
    assert logits.shape == (1, ACTION_SPACE)
    assert value.shape == (1,)
    print(f"  OK ({sum(p.numel() for p in net.parameters()):,} params)")


def test_mcts():
    print("--- MCTS ---")
    cfg = load_config("mcts.simulations=20", "mcts.batch_size=10")
    net = AlphaZeroNet(cfg.network.res_blocks, cfg.network.channels)
    board = chess.Board()

    probs, root = run_mcts(board, net, cfg, "cpu", add_noise=True)
    assert probs.shape == (ACTION_SPACE,)
    move = action_to_move(select_action(probs, 1.0), board)
    assert move in board.legal_moves
    print(f"  OK (selected {move.uci()})")


def test_self_play_and_train():
    print("--- Self-play + Train ---")
    cfg = load_config(
        "mcts.simulations=10", "mcts.batch_size=10",
        "self_play.games=1", "self_play.max_moves=20", "self_play.parallel_games=1",
        "train.epochs=1", "train.batch_size=8",
    )
    net = AlphaZeroNet(cfg.network.res_blocks, cfg.network.channels)

    from alphazero.self_play import run_self_play
    data = run_self_play(net, cfg, "cpu")
    print(f"  Generated {len(data)} positions")

    replay = ReplayBuffer(1000)
    replay.push(data)

    if len(replay) >= cfg.train.batch_size:
        result = train_network(net, replay, cfg, "cpu")
        print(f"  Loss: {result['loss']:.4f}")
    else:
        print(f"  Skipped training ({len(replay)} < {cfg.train.batch_size})")
    print("  OK")


if __name__ == "__main__":
    test_encoding()
    test_network()
    test_mcts()
    test_self_play_and_train()
    print("\nAll tests passed!")
