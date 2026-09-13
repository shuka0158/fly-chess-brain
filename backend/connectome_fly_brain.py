"""
The real fly-brain chess opponent.

Architecture: for every legal move, show the brain what the board would look
like *after* that move (as sensory stimulus), run a leaky integrate-and-fire
(LIF) spiking simulation of the real FAFB connectome on all candidates at
once, and read out real DAN (dopaminergic, PAM-cluster) neuron activity for
each - these are the fly's actual reward/valence teaching-signal neurons,
extensively studied in real associative-learning neuroscience. The move whose
resulting-position stimulus produces the most reward-neuron activity is the
one "the fly's brain picks."

This replaces an earlier version that scored moves with an untrained random
projection - a bigger, more honest change: the decision signal is now a real,
named, biologically meaningful population (real reward neurons), not an
arbitrary hash. It's still not literal insect cognition - flies have no
notion of chess, and there is no training signal that could ever make real
reward neurons "know" chess is good to win - but the readout is now a
genuine neuroscience concept, not a coincidence of random numbers.
"""
from __future__ import annotations

import time
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
import chess

from fly_brain import FlyBrain

GRAPH_DIR = Path(__file__).resolve().parent.parent / "data" / "graph"

# --- Simulation hyperparameters ---
# Fewer steps than the old single-position version (25) because we now
# simulate every legal move's resulting position at once (batched matrix
# sim) - cost scales with steps x candidate moves, so this keeps a typical
# ~30-legal-move position responding in single-digit seconds.
N_STEPS = 12
DECAY = 0.8
THRESHOLD = 0.35
STIMULUS_MAGNITUDE = 1.0
CHANGE_BOOST = 4.0  # extra stimulus on squares touched by the move being evaluated
REPETITION_PENALTY = 1000.0  # steer away from repeating a position when not forced
SHUFFLE_HISTORY = 6  # how many of the brain's own past moves to look back over
SHUFFLE_PENALTY = 40.0  # per past occurrence of this same piece-pair shuffle
CAPTURE_WEIGHT = 0.5  # small material-awareness safety net, see _score_move
PIECE_VALUES = {
    chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
    chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 4,
}
RNG_SEED = 42


class ConnectomeFlyBrain(FlyBrain):
    name = "fafb-connectome-dan"

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
        self.dan_idx = np.load(GRAPH_DIR / "dan_idx.npy")
        self.n = self.W.shape[0]

        # Per-DAN-neuron identity, for breaking valence down by real
        # mushroom-body compartment (PAM01-PAM15, PPL101-etc.) and individual
        # real neuron instead of one opaque aggregate number. Of the 331 real
        # DAN neurons, 307 are PAM cluster (reward-signaling) and 24 are PPL
        # cluster (punishment/aversive-signaling) - real, distinct, opposite-
        # valence populations in fly associative-learning neuroscience, not
        # an assumption: this was discovered from the annotation data, not
        # designed in (an earlier version of this app wrongly assumed "DAN"
        # meant "all reward" from only checking the 15 most common types).
        meta = pd.read_parquet(GRAPH_DIR / "neuron_meta.parquet")
        self.dan_root_id = meta["root_id"].to_numpy()[self.dan_idx]
        self.dan_cell_type = meta["cell_type"].fillna("?").to_numpy()[self.dan_idx]
        self.dan_is_pam = np.array([ct.startswith("PAM") for ct in self.dan_cell_type])
        self.dan_is_ppl = np.array([ct.startswith("PPL") for ct in self.dan_cell_type])
        print(f"[fly-brain]   DAN split: {self.dan_is_pam.sum()} PAM (reward), "
              f"{self.dan_is_ppl.sum()} PPL (punishment/aversive)")

        # Fixed, deterministic mapping: each of the 64 board squares gets its
        # own cluster of sensory neurons to stimulate (arbitrary but stable -
        # a real fly has no retina for a chessboard, this is a necessary
        # simplification to get board state into the network at all).
        rng = np.random.RandomState(RNG_SEED)
        shuffled_sensory = self.sensory_idx.copy()
        rng.shuffle(shuffled_sensory)
        self.square_to_sensory = np.array_split(shuffled_sensory, 64)

        print(f"[fly-brain] ready: {self.n} neurons, {self.W.nnz} synapses, "
              f"{len(self.sensory_idx)} sensory, {len(self.motor_idx)} motor, "
              f"{len(self.dan_idx)} DAN (reward)")

    # -- board encoding --------------------------------------------------
    def _encode_board(self, board: chess.Board, perspective: chess.Color) -> np.ndarray:
        """Stimulus for `board` as seen by `perspective` (True=white). Always
        pass the fly's own color explicitly - do not rely on board.turn,
        since we encode hypothetical *resulting* positions (after a
        candidate move has already been pushed, flipping board.turn)."""
        external = np.zeros(self.n, dtype=np.float32)
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is None:
                continue
            value = PIECE_VALUES[piece.piece_type]
            sign = 1.0 if piece.color == perspective else -1.0
            mag = STIMULUS_MAGNITUDE * (0.4 + 0.6 * value / 9.0) * sign
            external[self.square_to_sensory[square]] += mag

        # Change-sensitive boost on the squares touched by whichever move
        # produced this position - real sensory systems are far more driven
        # by what just moved than by the (mostly static) rest of the board.
        if board.move_stack:
            last = board.move_stack[-1]
            for square in (last.from_square, last.to_square):
                external[self.square_to_sensory[square]] += CHANGE_BOOST

        return external

    def _captured_value(self, board: chess.Board, move: chess.Move) -> float:
        if not board.is_capture(move):
            return 0.0
        if board.is_en_passant(move):
            return PIECE_VALUES[chess.PAWN]
        captured = board.piece_at(move.to_square)
        return PIECE_VALUES[captured.piece_type] if captured else 0.0

    def _safety_net(self, board: chess.Board, move: chess.Move,
                     own_recent_pairs: list[frozenset]) -> dict:
        """Small, disclosed adjustments layered on top of the real DAN
        valence signal - see think()'s docstring notes on why these exist."""
        capture_bonus = CAPTURE_WEIGHT * (self._captured_value(board, move) / 9.0)

        board.push(move)
        repetition_penalty = REPETITION_PENALTY if board.is_repetition(2) else 0.0
        board.pop()

        shuffle_count = own_recent_pairs.count(frozenset((move.from_square, move.to_square)))
        shuffle_penalty = SHUFFLE_PENALTY * shuffle_count

        return {
            "capture_bonus": round(capture_bonus, 4),
            "repetition_penalty": round(repetition_penalty, 4),
            "shuffle_penalty": round(shuffle_penalty, 4),
        }

    def _dan_breakdown(self, dan_spikes_for_move: np.ndarray) -> dict:
        """Per-real-mushroom-body-compartment and per-individual-real-neuron
        detail for one candidate's DAN response, instead of one opaque sum."""
        by_type: dict[str, int] = {}
        for ct, s in zip(self.dan_cell_type, dan_spikes_for_move):
            if s <= 0:
                continue
            by_type[ct] = by_type.get(ct, 0) + int(s)
        compartments = sorted(
            [{"cell_type": ct, "spikes": s, "group": "reward" if ct.startswith("PAM") else "punishment"}
             for ct, s in by_type.items()],
            key=lambda r: -r["spikes"],
        )
        top_neuron_order = np.argsort(-dan_spikes_for_move)[:5]
        top_neurons = [
            {"root_id": int(self.dan_root_id[j]), "cell_type": self.dan_cell_type[j],
             "spikes": int(dan_spikes_for_move[j]),
             "group": "reward" if self.dan_cell_type[j].startswith("PAM") else "punishment"}
            for j in top_neuron_order if dan_spikes_for_move[j] > 0
        ]
        return {"compartments": compartments, "top_neurons": top_neurons}

    # -- core: present every legal move's resulting position, batched ----
    def think(self, board: chess.Board):
        t_start = time.perf_counter()

        def elapsed_ms() -> int:
            return int((time.perf_counter() - t_start) * 1000)

        legal = list(board.legal_moves)
        if not legal:
            raise ValueError("No legal moves available")

        mover_color = board.turn  # the fly's own color - fixed perspective for all candidates
        yield {"stage": "present_options", "elapsed_ms": elapsed_ms(),
               "message": f"Presenting {len(legal)} candidate futures (the board after each legal "
                          f"move) to {len(self.dan_idx)} real reward-signaling (DAN) neurons at once."}

        t_encode = time.perf_counter()
        external_cols = []
        for move in legal:
            board.push(move)
            external_cols.append(self._encode_board(board, perspective=mover_color))
            board.pop()
        external_matrix = np.stack(external_cols, axis=1)  # (n, M)
        M = len(legal)
        yield {"stage": "encode_done", "elapsed_ms": elapsed_ms(),
               "message": f"Encoded {M} candidate boards in {int((time.perf_counter()-t_encode)*1000)}ms."}

        yield {"stage": "simulate_batch", "elapsed_ms": elapsed_ms(),
               "message": f"Running a {N_STEPS}-step spiking simulation across all {M} candidates "
                          f"in parallel over the real synaptic wiring ({self.W.nnz:,} synapses)..."}

        t_sim = time.perf_counter()
        v = np.zeros((self.n, M), dtype=np.float32)
        spikes = np.zeros((self.n, M), dtype=np.float32)
        dan_spike_matrix = np.zeros((len(self.dan_idx), M), dtype=np.float32)
        motor_accum = np.zeros(M, dtype=np.float32)
        for step in range(N_STEPS):
            t_step = time.perf_counter()
            input_current = self.W @ spikes + external_matrix
            v = DECAY * v + input_current
            fired = v >= THRESHOLD
            v = np.where(fired, 0.0, v)
            spikes = fired.astype(np.float32)
            dan_spike_matrix += spikes[self.dan_idx, :]
            motor_accum += spikes[self.motor_idx, :].sum(axis=0)
            yield {"stage": "sim_step", "step": step + 1, "n_steps": N_STEPS,
                   "elapsed_ms": elapsed_ms(), "step_ms": int((time.perf_counter() - t_step) * 1000),
                   "total_firing": int(spikes.sum()),
                   "reward_firing_total": int(spikes[self.dan_idx, :].sum()),
                   "motor_firing_total": int(spikes[self.motor_idx, :].sum())}

        dan_accum = dan_spike_matrix.sum(axis=0)
        yield {"stage": "valence_done", "elapsed_ms": elapsed_ms(),
               "message": f"Simulation done in {int((time.perf_counter()-t_sim)*1000)}ms. "
                          f"Reward-neuron response computed for all {M} candidates "
                          f"(range {int(dan_accum.min())}-{int(dan_accum.max())} spikes)."}

        yield {"stage": "scoring", "elapsed_ms": elapsed_ms(),
               "message": "Combining real reward-vs-punishment neuron valence with material safety net..."}
        own_recent = [board.move_stack[i] for i in range(len(board.move_stack) - 2, -1, -2)][:SHUFFLE_HISTORY]
        own_recent_pairs = [frozenset((m.from_square, m.to_square)) for m in own_recent]

        # Net valence = mean firing rate of real reward (PAM) neurons minus
        # mean firing rate of real punishment/aversive (PPL) neurons, each
        # normalized by their own population size so the much-smaller PPL
        # group (24 neurons) is compared fairly against PAM (307 neurons)
        # rather than being swamped by raw count. This is the actual
        # biological contrast (approach vs. avoidance teaching signal), not
        # just "more dopaminergic activity is automatically better."
        pam_rate = dan_spike_matrix[self.dan_is_pam, :].mean(axis=0) / N_STEPS
        ppl_rate = dan_spike_matrix[self.dan_is_ppl, :].mean(axis=0) / N_STEPS
        net_valence = pam_rate - ppl_rate  # roughly in [-1, 1]

        max_motor_possible = N_STEPS * max(len(self.motor_idx), 1)
        scored = []
        for i, move in enumerate(legal):
            valence = float(net_valence[i])
            net = self._safety_net(board, move, own_recent_pairs)
            total = valence + net["capture_bonus"] - net["repetition_penalty"] - net["shuffle_penalty"]
            scored.append({
                "uci": move.uci(),
                "san": board.san(move),
                "reward_spikes": int(dan_accum[i]),
                "reward_rate": round(float(pam_rate[i]), 4),
                "punishment_rate": round(float(ppl_rate[i]), 4),
                "valence": round(valence, 4),
                # Real descending/motor neuron activity for this candidate -
                # informational only, not used in scoring (this app doesn't
                # simulate an actual body for the fly to move), but it's real
                # data we already have loaded, so it's surfaced rather than
                # thrown away.
                "motor_activity": round(float(motor_accum[i]) / max_motor_possible, 4),
                **net,
                "total": round(total, 4),
                "_i": i,
            })
        scored.sort(key=lambda r: -r["total"])

        # Only compute the (slightly heavier) per-compartment/per-neuron
        # breakdown for the candidates we'll actually display.
        SHOWN = min(len(scored), 12)
        for r in scored[:SHOWN]:
            r.update(self._dan_breakdown(dan_spike_matrix[:, r.pop("_i")]))
        for r in scored[SHOWN:]:
            r.pop("_i", None)
        yield {"stage": "candidates", "elapsed_ms": elapsed_ms(), "top": scored[:SHOWN],
               "total_legal_moves": len(scored)}

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
            "total_elapsed_ms": elapsed_ms(),
        }

    def choose_move(self, board: chess.Board) -> chess.Move:
        """Like think(), but silent and leaves `board` unmodified (matches
        the FlyBrain interface other callers, e.g. /api/fly-move, expect)."""
        for event in self.think(board):
            pass
        move = board.pop()  # think() left its chosen move pushed; undo it
        return move


_singleton: ConnectomeFlyBrain | None = None


def get_connectome_brain() -> ConnectomeFlyBrain:
    global _singleton
    if _singleton is None:
        _singleton = ConnectomeFlyBrain()
    return _singleton
