import chess
import torch

from .network import AlphaZeroNet
from .mcts import run_mcts, select_action
from .encode import action_to_move, move_to_action
from .config import AlphaZeroConfig


def load_model(path: str, config: AlphaZeroConfig) -> AlphaZeroNet:
    net = AlphaZeroNet(config.num_res_blocks, config.num_channels).to(config.device)
    net.load_state_dict(torch.load(path, map_location=config.device, weights_only=True))
    net.eval()
    return net


def print_board(board: chess.Board):
    print()
    print(board.unicode(borders=True))
    print()


def human_vs_ai(model_path: str, human_color: str = "white", simulations: int = 200):
    config = AlphaZeroConfig(num_simulations=simulations)

    if torch.backends.mps.is_available():
        config.device = "mps"
    elif torch.cuda.is_available():
        config.device = "cuda"

    net = load_model(model_path, config)
    board = chess.Board()
    human_is_white = human_color.lower().startswith("w")

    print("=== AlphaZero Chess ===")
    print(f"You are {'White' if human_is_white else 'Black'}")
    print(f"AI is using {config.num_simulations} MCTS simulations")
    print("Enter moves in UCI format (e.g., e2e4) or 'quit' to exit\n")

    while not board.is_game_over(claim_draw=True):
        print_board(board)
        is_human_turn = (board.turn == chess.WHITE) == human_is_white

        if is_human_turn:
            while True:
                move_str = input("Your move: ").strip()
                if move_str.lower() == "quit":
                    print("Game abandoned.")
                    return
                try:
                    move = chess.Move.from_uci(move_str)
                    if move in board.legal_moves:
                        break
                    print(f"Illegal move. Legal moves: {', '.join(m.uci() for m in board.legal_moves)}")
                except ValueError:
                    print("Invalid format. Use UCI notation (e.g., e2e4)")
            board.push(move)
        else:
            print("AI is thinking...")
            action_probs, _ = run_mcts(board, net, config, add_noise=False)
            action = select_action(action_probs, temperature=0.1)
            move = action_to_move(action, board)
            if move not in board.legal_moves:
                legal_actions = []
                for m in board.legal_moves:
                    a = move_to_action(m, board)
                    legal_actions.append((action_probs[a], a, m))
                legal_actions.sort(reverse=True)
                move = legal_actions[0][2]
            print(f"AI plays: {move.uci()}")
            board.push(move)

    print_board(board)
    result = board.result()
    if result == "1-0":
        winner = "White wins!"
    elif result == "0-1":
        winner = "Black wins!"
    else:
        winner = "Draw!"
    print(f"Game over: {result} — {winner}")


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/best.pt"
    color = sys.argv[2] if len(sys.argv) > 2 else "white"
    human_vs_ai(path, color)
