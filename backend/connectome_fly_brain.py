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
CHANGE_BOOST = 4.0  # extra stimulus on squares touched by the last move played
REPETITION_PENALTY = 1000.0  # steer away from repeating a position when not forced
SHUFFLE_HISTORY = 6  # how many of the brain's own past moves to look back over
SHUFFLE_PENALTY = 40.0  # per past occurrence of this same piece-pair shuffle
CAPTURE_WEIGHT = 0.5  # material-awareness term, see choose_move for why this exists
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

        # Change-sensitive boost: real sensory systems are far more driven by
        # what just moved than by the (mostly static) rest of the board. Without
        # this, one turn's stimulus looks almost identical to the last one -
        # a handful of far-away opening moves barely nudges a 138k-neuron
        # network - and the readout collapses onto whichever move type wins
        # by default (observed in practice: the brain got stuck shuffling one
        # rook back and forth regardless of what the opponent played).
        if board.move_stack:
            last = board.move_stack[-1]
            for square in (last.from_square, last.to_square):
                neurons = self.square_to_sensory[square]
                external[neurons] += CHANGE_BOOST

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

    def _captured_value(self, board: chess.Board, move: chess.Move) -> float:
        if not board.is_capture(move):
            return 0.0
        if board.is_en_passant(move):
            return PIECE_VALUES[chess.PAWN]
        captured = board.piece_at(move.to_square)
        return PIECE_VALUES[captured.piece_type] if captured else 0.0

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

    def _score_move(self, board: chess.Board, move: chess.Move, motor_activity: np.ndarray,
                     own_recent_pairs: list[frozenset]) -> dict:
        code = self._move_features(board, move) @ self.readout
        code_norm = np.linalg.norm(code)
        if code_norm > 0:
            code = code / code_norm
        connectome_score = float(motor_activity @ code)

        # The random readout has no notion of material at all - "capture"
        # is just one arbitrary feature bit in a random projection, not a
        # weighted preference - so without this the brain never takes even a
        # free queen (verified empirically). Deliberate, disclosed
        # material-awareness term layered on top of the connectome score.
        capture_bonus = CAPTURE_WEIGHT * (self._captured_value(board, move) / 9.0)

        # Steer away from repeating a position (e.g. shuffling one piece back
        # and forth forever) unless every legal move repeats one.
        board.push(move)
        repetition_penalty = REPETITION_PENALTY if board.is_repetition(2) else 0.0
        board.pop()

        # The untrained random readout genuinely doesn't discriminate finely
        # between similar-looking positions (verified empirically: two boards
        # differing by one far-away opening move gave >0.95 cosine-similar
        # motor activity regardless of stimulus tuning) - a real limitation of
        # a fixed reservoir this size, not something a magnitude tweak fixes.
        # Left alone this reliably degenerates into shuffling one piece back
        # and forth. Clearly-labeled behavioral guard, not a claim that the
        # brain itself "noticed" the loop.
        shuffle_count = own_recent_pairs.count(frozenset((move.from_square, move.to_square)))
        shuffle_penalty = SHUFFLE_PENALTY * shuffle_count

        total = connectome_score + capture_bonus - repetition_penalty - shuffle_penalty
        return {
            "uci": move.uci(),
            "san": board.san(move),
            "connectome_score": round(connectome_score, 4),
            "capture_bonus": round(capture_bonus, 4),
            "repetition_penalty": round(repetition_penalty, 4),
            "shuffle_penalty": round(shuffle_penalty, 4),
            "total": round(total, 4),
        }

    def think(self, board: chess.Board):
        """Generator yielding live progress events; the last event is the
        decision. Mutates `board` in place (pushes the chosen move), exactly
        like choose_move, so callers should not also call choose_move on it."""
        legal = list(board.legal_moves)
        if not legal:
            raise ValueError("No legal moves available")

        n_occupied = sum(1 for sq in chess.SQUARES if board.piece_at(sq))
        yield {"stage": "encode",
               "message": f"Encoding {n_occupied} occupied squares into sensory-neuron stimulus "
                          f"({len(self.sensory_idx)} real afferent neurons available)."}
        external = self._encode_board(board)
        if board.move_stack:
            last_san_sq = chess.square_name(board.move_stack[-1].to_square)
            yield {"stage": "encode_done",
                   "message": f"Applied change-sensitive boost around the last move (...{last_san_sq})."}
        else:
            yield {"stage": "encode_done", "message": "No prior move this game - no change-boost applied."}

        v = np.zeros(self.n, dtype=np.float32)
        spikes = np.zeros(self.n, dtype=np.float32)
        motor_accum = np.zeros(self.n_motor, dtype=np.float32)
        for step in range(N_STEPS):
            input_current = self.W @ spikes + external
            v = DECAY * v + input_current
            fired = v >= THRESHOLD
            v = np.where(fired, 0.0, v)
            spikes = fired.astype(np.float32)
            motor_accum += spikes[self.motor_idx]
            yield {"stage": "sim_step", "step": step + 1, "n_steps": N_STEPS,
                   "neurons_firing": int(spikes.sum()), "motor_firing": int(spikes[self.motor_idx].sum())}

        norm = np.linalg.norm(motor_accum)
        motor_activity = motor_accum / norm if norm > 0 else motor_accum
        yield {"stage": "sim_done",
               "message": f"{int((motor_accum > 0).sum())}/{self.n_motor} real descending/motor "
                          f"neurons fired at least once over {N_STEPS} steps."}

        yield {"stage": "scoring", "message": f"Scoring {len(legal)} legal moves against motor activity..."}
        own_recent = [board.move_stack[i] for i in range(len(board.move_stack) - 2, -1, -2)][:SHUFFLE_HISTORY]
        own_recent_pairs = [frozenset((m.from_square, m.to_square)) for m in own_recent]

        scored = [self._score_move(board, m, motor_activity, own_recent_pairs) for m in legal]
        scored.sort(key=lambda r: -r["total"])
        yield {"stage": "candidates", "top": scored[:6]}

        best = scored[0]
        move = chess.Move.from_uci(best["uci"])
        san = board.san(move)
        board.push(move)
        yield {
            "stage": "decision",
            "move_uci": move.uci(),
            "move_san": san,
            "fen_after": board.fen(),
            "is_check": board.is_check(),
            "is_checkmate": board.is_checkmate(),
            "is_stalemate": board.is_stalemate(),
            "is_game_over": board.is_game_over(),
            "brain": self.name,
            "score_breakdown": best,
        }

    def choose_move(self, board: chess.Board) -> chess.Move:
        legal = list(board.legal_moves)
        if not legal:
            raise ValueError("No legal moves available")

        external = self._encode_board(board)
        motor_activity = self._simulate(external)
        norm = np.linalg.norm(motor_activity)
        if norm > 0:
            motor_activity = motor_activity / norm

        own_recent = [board.move_stack[i] for i in range(len(board.move_stack) - 2, -1, -2)][:SHUFFLE_HISTORY]
        own_recent_pairs = [frozenset((m.from_square, m.to_square)) for m in own_recent]

        scored = [self._score_move(board, m, motor_activity, own_recent_pairs) for m in legal]
        scored.sort(key=lambda r: -r["total"])
        return chess.Move.from_uci(scored[0]["uci"])


_singleton: ConnectomeFlyBrain | None = None


def get_connectome_brain() -> ConnectomeFlyBrain:
    global _singleton
    if _singleton is None:
        _singleton = ConnectomeFlyBrain()
    return _singleton
