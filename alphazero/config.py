from dataclasses import dataclass


@dataclass
class AlphaZeroConfig:
    # --- Network ---
    num_res_blocks: int = 5
    num_channels: int = 64

    # --- MCTS ---
    num_simulations: int = 100
    mcts_batch_size: int = 32
    c_puct: float = 1.25
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25

    # --- Self-play ---
    num_self_play_games: int = 25
    max_moves: int = 150
    temp_threshold: int = 30

    # --- Training ---
    num_epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    replay_buffer_size: int = 50_000

    # --- Evaluation ---
    num_eval_games: int = 10
    eval_simulations: int = 50
    win_threshold: float = 0.55

    # --- System ---
    num_iterations: int = 50
    device: str = "cpu"
