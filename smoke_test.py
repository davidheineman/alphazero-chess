import chess
import torch
import numpy as np

from alphazero.config import AlphaZeroConfig
from alphazero.encode import encode_board, encode_board_tensor, get_legal_move_mask, move_to_action, action_to_move, ACTION_SPACE
from alphazero.network import AlphaZeroNet
from alphazero.mcts import run_mcts, select_action
from alphazero.trainer import ReplayBuffer, train_network


def test_encoding():
    print("--- Board encoding ---")
    board = chess.Board()
    planes = encode_board(board)
    assert planes.shape == (19, 8, 8), f"Bad shape: {planes.shape}"
    print(f"  Board planes shape: {planes.shape}")

    tensor = encode_board_tensor(board)
    assert tensor.shape == (1, 19, 8, 8)
    print(f"  Tensor shape: {tensor.shape}")

    mask = get_legal_move_mask(board)
    legal_count = int(mask.sum())
    assert legal_count == 20, f"Expected 20 legal moves at start, got {legal_count}"
    print(f"  Legal moves at start: {legal_count}")

    # Round-trip test for all legal moves
    for move in board.legal_moves:
        action = move_to_action(move, board)
        recovered = action_to_move(action, board)
        assert move == recovered, f"Round-trip failed: {move} -> {action} -> {recovered}"
    print(f"  All {legal_count} moves round-trip OK")
    print()


def test_network():
    print("--- Network ---")
    config = AlphaZeroConfig(num_res_blocks=2, num_channels=32)
    net = AlphaZeroNet(config.num_res_blocks, config.num_channels)
    param_count = sum(p.numel() for p in net.parameters())
    print(f"  Parameters (small net): {param_count:,}")

    board = chess.Board()
    x = encode_board_tensor(board)
    policy_logits, value = net(x)
    assert policy_logits.shape == (1, ACTION_SPACE)
    assert value.shape == (1,)
    print(f"  Policy logits shape: {policy_logits.shape}")
    print(f"  Value: {value.item():.4f}")
    print()


def test_mcts():
    print("--- MCTS ---")
    config = AlphaZeroConfig(num_res_blocks=2, num_channels=32, num_simulations=20)
    net = AlphaZeroNet(config.num_res_blocks, config.num_channels)
    board = chess.Board()

    action_probs, root = run_mcts(board, net, config, add_noise=True)
    assert action_probs.shape == (ACTION_SPACE,)
    total_visits = sum(c.visit_count for c in root.children)
    print(f"  Total root visit counts: {total_visits}")
    print(f"  Root children: {len(root.children)}")

    action = select_action(action_probs, temperature=1.0)
    move = action_to_move(action, board)
    print(f"  Selected move: {move.uci()}")
    assert move in board.legal_moves
    print()


def test_self_play_and_train():
    print("--- Self-play + Training (1 short game) ---")
    config = AlphaZeroConfig(
        num_res_blocks=2,
        num_channels=32,
        num_simulations=10,
        num_self_play_games=1,
        num_epochs=1,
        batch_size=8,
        temp_threshold=5,
    )
    net = AlphaZeroNet(config.num_res_blocks, config.num_channels).to(config.device)

    from alphazero.self_play import run_self_play
    data = run_self_play(net, config)
    print(f"  Game generated {len(data)} positions")

    replay = ReplayBuffer(1000)
    replay.push(data)

    if len(replay) >= config.batch_size:
        loss_dict = train_network(net, replay, config)
        print(f"  Training loss: {loss_dict['loss']:.4f}")
    else:
        print(f"  Skipped training (only {len(replay)} samples, need {config.batch_size})")
    print()


if __name__ == "__main__":
    test_encoding()
    test_network()
    test_mcts()
    test_self_play_and_train()
    print("All smoke tests passed!")
