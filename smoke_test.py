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
    print("--- MCTS (C++) ---")
    cfg = load_config("mcts.simulations=50", "mcts.batch_size=25")
    net = AlphaZeroNet(cfg.network.res_blocks, cfg.network.channels)

    probs = run_mcts(chess.STARTING_FEN, net, cfg, "cpu", add_noise=True)
    assert probs.shape == (ACTION_SPACE,)
    assert abs(probs.sum() - 1.0) < 1e-5

    board = chess.Board()
    action = select_action(probs, 1.0)
    move = action_to_move(action, board)
    assert move in board.legal_moves
    print(f"  OK (selected {move.uci()})")


def test_self_play_and_train():
    print("--- Self-play + Train ---")
    cfg = load_config(
        "mcts.simulations=20", "mcts.batch_size=10",
        "self_play.games=2", "self_play.max_moves=30", "self_play.num_workers=1",
        "train.epochs=1", "train.batch_size=8",
    )
    net = AlphaZeroNet(cfg.network.res_blocks, cfg.network.channels)

    from alphazero.self_play import run_self_play
    data = run_self_play(net, cfg, "cpu")
    print(f"  Generated {len(data)} positions")

    if len(data) >= cfg.train.batch_size:
        opt = torch.optim.Adam(net.parameters(), lr=cfg.train.lr)
        result = train_network(net, opt, ReplayBuffer(1000), cfg, "cpu")
        # push data first
        buf = ReplayBuffer(1000)
        buf.push(data)
        result = train_network(net, opt, buf, cfg, "cpu")
        print(f"  Loss: {result['loss']:.4f}")
    print("  OK")


if __name__ == "__main__":
    test_encoding()
    test_network()
    test_mcts()
    test_self_play_and_train()
    print("\nAll tests passed!")
