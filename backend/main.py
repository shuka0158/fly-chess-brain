from __future__ import annotations

import chess
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from fly_brain import get_active_brain

app = FastAPI(title="Fly Brain Chess")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class MoveRequest(BaseModel):
    fen: str


class MoveResponse(BaseModel):
    move_uci: str
    move_san: str
    fen_after: str
    is_check: bool
    is_checkmate: bool
    is_stalemate: bool
    is_game_over: bool
    brain: str


@app.get("/api/health")
def health():
    brain = get_active_brain()
    return {"status": "ok", "brain": brain.name}


@app.post("/api/fly-move", response_model=MoveResponse)
def fly_move(req: MoveRequest):
    try:
        board = chess.Board(req.fen)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid FEN: {e}")

    if board.is_game_over():
        raise HTTPException(status_code=400, detail="Game is already over")

    brain = get_active_brain()
    move = brain.choose_move(board)
    san = board.san(move)
    board.push(move)

    return MoveResponse(
        move_uci=move.uci(),
        move_san=san,
        fen_after=board.fen(),
        is_check=board.is_check(),
        is_checkmate=board.is_checkmate(),
        is_stalemate=board.is_stalemate(),
        is_game_over=board.is_game_over(),
        brain=brain.name,
    )


# Serve the frontend
app.mount("/", StaticFiles(directory="../frontend", html=True), name="frontend")
