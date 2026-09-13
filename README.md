---
title: Chess vs The Fly Brain
emoji: 🪰
colorFrom: purple
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# Chess vs. The Fly Brain

**🪰 Live: [fly-chess-brain.onrender.com](https://fly-chess-brain.onrender.com)**

> **Not 24/7 always-on.** This runs on Render's free tier, which sleeps the
> app after ~15min with no traffic to save resources. The link is permanent,
> but if nobody's used it recently the first visit triggers a ~30-60s cold
> start while it wakes back up - normal, not broken. Once warm, expect
> ~15-45s per move regardless (this is a genuinely heavy simulation running
> on shared free-tier CPU). A real always-on deploy would need a paid tier.

Play chess against an opponent whose moves are chosen by real **DAN (reward)
neuron** activity in a leaky integrate-and-fire (LIF) spiking simulation of
the real FAFB fly connectome (139,255 proofread neurons, ~15M aggregated
synaptic weights, from [FlyWire/Codex](https://codex.flywire.ai/?dataset=fafb)).

**What this is, honestly:** a fruit fly cannot reason about chess. This
engine shows the brain the resulting board of every legal move and reads out
activity from its real DAN (dopaminergic) neurons — of the 331 real DAN
neurons in this connectome, 307 are **PAM cluster** (reward-signaling) and 24
are **PPL cluster** (punishment/aversive-signaling): two real, distinct,
opposite-valence populations from actual fly associative-learning
neuroscience (PAM reinforces approach, PPL1 reinforces avoidance). Net
valence = mean PAM firing rate minus mean PPL firing rate. The move whose
resulting position produces the most net-positive reward-vs-punishment
response is played. This is a genuine, named biological concept — not an
arbitrary formula — but there is still no training signal that could ever
make real reward/punishment neurons "know" chess is good to win, so don't
expect strong play.

## How the fly picks a move

1. **Present every option**: for each legal move, compute the board that
   would result from playing it, and encode it as stimulus into real
   *sensory* (afferent) neurons — each of the 64 squares maps to a fixed
   cluster, signed by whose piece it is and scaled by piece value, with an
   extra boost on the squares that move touches.
2. **Simulate all candidates at once**: propagate every candidate's stimulus
   in parallel (one batched matrix simulation, not one run per move — this is
   what keeps a ~30-legal-move position responding in single-digit seconds)
   through the real synaptic weight matrix for 12 steps of LIF dynamics.
   Excitatory/inhibitory sign per edge comes from real predicted
   neurotransmitter probabilities.
3. **Read real reward-vs-punishment activity**: for each candidate, take the
   mean firing rate of the real PAM (reward) neurons minus the mean firing
   rate of the real PPL (punishment/aversive) neurons - this is the fly's
   actual approach/avoidance teaching-signal response to that outcome.
4. **Score**: that net valence plus a small, disclosed safety net -
   material-awareness (the raw valence signal has no inherent notion of
   piece value) and anti-repetition/anti-shuffle guards (the valence signal
   doesn't reliably discriminate between very similar-looking positions on
   its own; left alone this degenerates into shuffling one piece back and
   forth). The highest-scoring legal move is played.

The live-thinking panel also exposes, per candidate: the breakdown by real
mushroom-body compartment (e.g. PAM08, PPL101), the individual real neurons
(actual root IDs) that fired most, informational real descending/motor
neuron activity (not used in scoring), and per-stage timing.

## Running it

```bash
cd backend
../.venv/bin/uvicorn main:app --host 127.0.0.1 --port 8770
```

Then open http://127.0.0.1:8770 in a browser. You play White; the fly plays
Black. The right-hand panel streams the brain's live thinking: board
encoding, all 12 simulation steps (with real firing counts), reward-neuron
valence per candidate, and the final scored move table.

## Rebuilding the connectome graph

Data (not committed, ~900MB) lives in `data/`:
- `proofread_connections_783.feather` — from [Zenodo record 10676866](https://zenodo.org/records/10676866)
- `neuron_annotations.tsv` — from [flyconnectome/flywire_annotations](https://github.com/flyconnectome/flywire_annotations)

```bash
cd scripts
../.venv/bin/python3 build_graph.py
```

This writes `data/graph/{weights.npz, root_ids.npy, sensory_idx.npy,
motor_idx.npy, dan_idx.npy, neuron_meta.parquet}`, which
`backend/connectome_fly_brain.py` loads at startup. `dan_idx.npy` is the 331
real neurons annotated `cell_class == "DAN"` (all PAM-cluster in this
dataset) that the whole decision mechanism reads out from.

## Project layout

```
backend/
  main.py                  FastAPI app: /api/fly-move, /api/fly-move-stream, /api/health
  fly_brain.py              FlyBrain interface + placeholder + brain selection
  connectome_fly_brain.py   the real spiking engine (batched-simulation + DAN readout)
frontend/
  index.html                chess.js + chessboard.js UI + live-thinking panel
scripts/
  build_graph.py             connectome -> cached sparse graph + classification
data/                        raw downloads + data/graph/ cache (gitignored)
```

## Known limitations / honest caveats

- The board→neuron mapping (64 squares → sensory neuron clusters) is an
  arbitrary but fixed assignment, not a biologically real retinotopic map —
  the fly doesn't "see" a chessboard.
- Real DAN neurons signal reward for actual fly behaviors (odor/reward
  association in the mushroom body) — repurposing their simulated response
  to an artificial chess-board stimulus as "move quality" is a deliberate
  analogy, not literal insect chess evaluation.
- The material-awareness and anti-repetition/anti-shuffle terms are
  hand-written safety nets layered on top of the raw reward-neuron signal,
  disclosed in the live-thinking panel's score breakdown (capture bonus /
  penalty columns) rather than hidden.
- ~8-10s per move (batched sparse matvec over 15M weights × 12 steps ×
  ~20-40 simultaneous candidate simulations, single-threaded).
