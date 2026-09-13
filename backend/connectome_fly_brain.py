"""
The real fly-brain chess opponent: board state is injected as stimulus into
real FAFB sensory (afferent) neurons, propagated through the actual synaptic
wiring via a leaky integrate-and-fire (LIF) spiking simulation, and read out
at real descending/motor neurons to score legal chess moves.

This is a reservoir-computing style engine: the *reservoir* (the connectome's
fixed synaptic weights) is 100% real data. The *readout* (mapping motor
activity -> a move score) is an untrained, fixed random projection, since
there is no training signal that would make an insect brain "want" to play
chess. It's an honest novelty engine, not literal insect cognition.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import scipy.sparse as sp
import chess

from fly_brain import FlyBrain

GRAPH_DIR = Path(__file__).resolve().parent.parent / "data" / "graph"

# --- Simulation hyperparameters ---
N_STEPS = 25
DECAY = 0.8
THRESHOLD = 0.35
STIMULUS_MAGNITUDE = 1.0
PIECE_VALUES = {
    chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
    chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 4,
}
FEATURE_DIM = 64 + 64 + 6 + 3  # from-sq, to-sq, piece type, capture/promo/check
RNG_SEED = 42


class ConnectomeFlyBrain(FlyBrain):
    name = "fafb-connectome-lif"

    def __init__(self):
        print("[fly-brain] loading connectome graph...")
        W = sp.load_npz(GRAPH_DIR / "weights.npz").tocsr().astype(np.float32)
        # Row-normalize so no neuron's total input can blow up regardless of
        # how many presynaptic partners fire at once.
        row_abs_sum = np.abs(W).sum(axis=1).A.ravel()
        row_abs_sum[row_abs_sum == 0] = 1.0
        inv = sp.diags(1.0 / row_abs_sum)
        self.W = (inv @ W).tocsr()

        self.root_ids = np.load(GRAPH_DIR / "root_ids.npy")
        self.sensory_idx = np.load(GRAPH_DIR / "sensory_idx.npy")
        self.motor_idx = np.load(GRAPH_DIR / "motor_idx.npy")
        self.n = self.W.shape[0]
        self.n_motor = len(self.motor_idx)

        # Fixed, deterministic mapping: each of the 64 board squares gets its
        # own cluster of sensory neurons to stimulate (arbitrary but stable).
        rng = np.random.RandomState(RNG_SEED)
        shuffled_sensory = self.sensory_idx.copy()
        rng.shuffle(shuffled_sensory)
        self.square_to_sensory = np.array_split(shuffled_sensory, 64)

        # Fixed random readout: move-feature vector -> motor-space code.
        self.readout = rng.normal(size=(FEATURE_DIM, self.n_motor)).astype(np.float32)
        self.readout /= np.sqrt(FEATURE_DIM)

        print(f"[fly-brain] ready: {self.n} neurons, {self.W.nnz} synapses, "
              f"{len(self.sensory_idx)} sensory, {self.n_motor} motor")

    # -- board encoding --------------------------------------------------
    def _encode_board(self, board: chess.Board) -> np.ndarray:
        external = np.zeros(self.n, dtype=np.float32)
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is None:
                continue
            value = PIECE_VALUES[piece.piece_type]
            sign = 1.0 if piece.color == board.turn else -1.0
            mag = STIMULUS_MAGNITUDE * (0.4 + 0.6 * value / 9.0) * sign
            neurons = self.square_to_sensory[square]
            external[neurons] += mag
        return external

    # -- simulation --------------------------------------------------
    def _simulate(self, external: np.ndarray) -> np.ndarray:
        v = np.zeros(self.n, dtype=np.float32)
        spikes = np.zeros(self.n, dtype=np.float32)
        motor_accum = np.zeros(self.n_motor, dtype=np.float32)

        for _ in range(N_STEPS):
            input_current = self.W @ spikes + external
            v = DECAY * v + input_current
            fired = v >= THRESHOLD
            v = np.where(fired, 0.0, v)  # hard reset
            spikes = fired.astype(np.float32)
            motor_accum += spikes[self.motor_idx]

        return motor_accum

    # -- move scoring --------------------------------------------------
    def _move_features(self, board: chess.Board, move: chess.Move) -> np.ndarray:
        f = np.zeros(FEATURE_DIM, dtype=np.float32)
        f[move.from_square] = 1.0
        f[64 + move.to_square] = 1.0
        piece = board.piece_at(move.from_square)
        piece_type = piece.piece_type if piece else chess.PAWN
        f[128 + (piece_type - 1)] = 1.0
        f[134] = 1.0 if board.is_capture(move) else 0.0
        f[135] = 1.0 if move.promotion else 0.0
        board.push(move)
        f[136] = 1.0 if board.is_check() else 0.0
        board.pop()
        return f

    def choose_move(self, board: chess.Board) -> chess.Move:
        legal = list(board.legal_moves)
        if not legal:
            raise ValueError("No legal moves available")

        external = self._encode_board(board)
        motor_activity = self._simulate(external)
        norm = np.linalg.norm(motor_activity)
        if norm > 0:
            motor_activity = motor_activity / norm

        best_move, best_score = None, -np.inf
        for move in legal:
            code = self._move_features(board, move) @ self.readout
            code_norm = np.linalg.norm(code)
            if code_norm > 0:
                code = code / code_norm
            score = float(motor_activity @ code)
            if score > best_score:
                best_score, best_move = score, move
        return best_move


_singleton: ConnectomeFlyBrain | None = None


def get_connectome_brain() -> ConnectomeFlyBrain:
    global _singleton
    if _singleton is None:
        _singleton = ConnectomeFlyBrain()
    return _singleton
