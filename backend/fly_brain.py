"""
The "fly brain" chess opponent.

This module defines the interface for move selection so the app is playable
immediately with a placeholder, and the real FAFB-connectome-driven spiking
engine can be dropped in later without touching the API layer.
"""
from __future__ import annotations

import random
import chess


class FlyBrain:
    """Base interface: given a board, pick one of the legal moves."""

    name = "placeholder"

    def choose_move(self, board: chess.Board) -> chess.Move:
        raise NotImplementedError


class RandomFlyBrain(FlyBrain):
    """Temporary stand-in used until the real connectome engine is wired in.

    Picks uniformly among legal moves, but mildly prefers captures/checks so
    games aren't completely aimless while the real brain is being built.
    """

    name = "placeholder-random"

    def choose_move(self, board: chess.Board) -> chess.Move:
        legal = list(board.legal_moves)
        if not legal:
            raise ValueError("No legal moves available")

        def score(m: chess.Move) -> float:
            s = random.random()
            if board.is_capture(m):
                s += 1.5
            board.push(m)
            if board.is_check():
                s += 1.0
            board.pop()
            return s

        return max(legal, key=score)


def get_active_brain() -> FlyBrain:
    """Returns whichever brain implementation is currently active.

    Uses the real FAFB-connectome spiking engine if its cached graph data is
    present (see scripts/build_graph.py); falls back to the placeholder
    otherwise so the app still runs without the ~1GB of connectome data.
    """
    from pathlib import Path
    graph_dir = Path(__file__).resolve().parent.parent / "data" / "graph"
    if (graph_dir / "weights.npz").exists():
        from connectome_fly_brain import get_connectome_brain
        return get_connectome_brain()
    return RandomFlyBrain()
