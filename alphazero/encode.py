import chess
import numpy as np
import torch

NUM_PLANES = 19
ACTION_SPACE = 8 * 8 * 73  # 4672

_QUEEN_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1),
               (-1, -1), (-1, 1), (1, -1), (1, 1)]

_KNIGHT_MOVES = [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
                 (1, -2), (1, 2), (2, -1), (2, 1)]

_UNDERPROMO_PIECES = [chess.KNIGHT, chess.BISHOP, chess.ROOK]
_UNDERPROMO_DIRS = [-1, 0, 1]  # file deltas for pawn captures/push


def encode_board(board: chess.Board) -> np.ndarray:
    planes = np.zeros((NUM_PLANES, 8, 8), dtype=np.float32)

    for piece_type in range(1, 7):  # PAWN..KING
        for color in [chess.WHITE, chess.BLACK]:
            plane_idx = (piece_type - 1) * 2 + (0 if color == chess.WHITE else 1)
            for sq in board.pieces(piece_type, color):
                r, c = divmod(sq, 8)
                planes[plane_idx, r, c] = 1.0

    if board.turn == chess.WHITE:
        planes[12] = 1.0

    planes[13] = float(board.has_kingside_castling_rights(chess.WHITE))
    planes[14] = float(board.has_queenside_castling_rights(chess.WHITE))
    planes[15] = float(board.has_kingside_castling_rights(chess.BLACK))
    planes[16] = float(board.has_queenside_castling_rights(chess.BLACK))

    if board.ep_square is not None:
        r, c = divmod(board.ep_square, 8)
        planes[17, r, c] = 1.0

    planes[18] = board.halfmove_clock / 100.0

    return planes


def _move_to_action(move: chess.Move, board: chess.Board) -> int:
    from_sq = move.from_square
    to_sq = move.to_square
    from_r, from_c = divmod(from_sq, 8)
    to_r, to_c = divmod(to_sq, 8)
    dr = to_r - from_r
    dc = to_c - from_c

    if move.promotion and move.promotion != chess.QUEEN:
        direction = dc
        dir_idx = _UNDERPROMO_DIRS.index(max(-1, min(1, direction)))
        piece_idx = _UNDERPROMO_PIECES.index(move.promotion)
        move_type = 56 + 8 + dir_idx * 3 + piece_idx
        return from_sq * 73 + move_type

    if (abs(dr), abs(dc)) in [(2, 1), (1, 2)]:
        knight_delta = (dr, dc)
        knight_idx = _KNIGHT_MOVES.index(knight_delta)
        move_type = 56 + knight_idx
        return from_sq * 73 + move_type

    # Queen-type move (includes queen promotions)
    if dr != 0 or dc != 0:
        direction = (
            (1 if dr > 0 else -1 if dr < 0 else 0),
            (1 if dc > 0 else -1 if dc < 0 else 0),
        )
        dir_idx = _QUEEN_DIRS.index(direction)
        distance = max(abs(dr), abs(dc)) - 1  # 0-indexed
        move_type = dir_idx * 7 + distance
        return from_sq * 73 + move_type

    return from_sq * 73


def _action_to_move(action: int, board: chess.Board) -> chess.Move:
    from_sq = action // 73
    move_type = action % 73

    from_r, from_c = divmod(from_sq, 8)

    if move_type < 56:
        dir_idx = move_type // 7
        distance = move_type % 7 + 1
        dr, dc = _QUEEN_DIRS[dir_idx]
        to_r = from_r + dr * distance
        to_c = from_c + dc * distance
        to_sq = to_r * 8 + to_c

        piece = board.piece_at(from_sq)
        promo = None
        if piece and piece.piece_type == chess.PAWN:
            if (board.turn == chess.WHITE and to_r == 7) or \
               (board.turn == chess.BLACK and to_r == 0):
                promo = chess.QUEEN
        return chess.Move(from_sq, to_sq, promotion=promo)

    if move_type < 64:
        knight_idx = move_type - 56
        dr, dc = _KNIGHT_MOVES[knight_idx]
        to_r = from_r + dr
        to_c = from_c + dc
        to_sq = to_r * 8 + to_c
        return chess.Move(from_sq, to_sq)

    # Underpromotion
    underpromo_idx = move_type - 64
    dir_idx = underpromo_idx // 3
    piece_idx = underpromo_idx % 3
    dc = _UNDERPROMO_DIRS[dir_idx]
    dr = 1 if board.turn == chess.WHITE else -1
    to_r = from_r + dr
    to_c = from_c + dc
    to_sq = to_r * 8 + to_c
    promo = _UNDERPROMO_PIECES[piece_idx]
    return chess.Move(from_sq, to_sq, promotion=promo)


def get_legal_move_mask(board: chess.Board) -> np.ndarray:
    mask = np.zeros(ACTION_SPACE, dtype=np.float32)
    for move in board.legal_moves:
        action = _move_to_action(move, board)
        mask[action] = 1.0
    return mask


def move_to_action(move: chess.Move, board: chess.Board) -> int:
    return _move_to_action(move, board)


def action_to_move(action: int, board: chess.Board) -> chess.Move:
    return _action_to_move(action, board)


def encode_board_tensor(board: chess.Board, device: str = "cpu") -> torch.Tensor:
    return torch.from_numpy(encode_board(board)).unsqueeze(0).to(device)
