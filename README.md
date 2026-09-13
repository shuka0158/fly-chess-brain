# Chess vs. The Fly Brain

Play chess against an opponent whose moves are chosen by a leaky
integrate-and-fire (LIF) spiking simulation running on the **real FAFB fly
connectome** (139,255 proofread neurons, ~15M aggregated synaptic weights,
from [FlyWire/Codex](https://codex.flywire.ai/?dataset=fafb)).

**What this is, honestly:** a fruit fly cannot reason about chess. This is a
*reservoir-computing* novelty engine — the reservoir (fixed synaptic weights)
is 100% real connectome data; the readout (mapping motor-neuron activity to a
chess move score) is an untrained, fixed random projection, since there's no
training signal that would make an insect brain "want" to play chess well.
Don't expect strong play — expect genuinely fly-wiring-driven play.

## How the fly picks a move

1. **Encode the board**: each of the 64 squares maps to a fixed cluster of
   real *sensory* (afferent) neurons. Occupied squares inject a stimulus
   current, signed by whose piece it is and scaled by piece value.
2. **Simulate**: propagate that stimulus through the real synaptic weight
   matrix for 25 timesteps of LIF dynamics (excitatory/inhibitory sign comes
   from each edge's predicted neurotransmitter probabilities — ACh/octopamine/
   serotonin/dopamine ≈ excitatory, GABA/glutamate ≈ inhibitory).
3. **Read out**: sum spikes at real *descending/motor* neurons over the
   simulation → a motor activity vector.
4. **Score moves**: each legal move gets a fixed random "motor code" (from a
   seeded projection of its from/to squares, piece type, capture/promotion/
   check flags). The move whose code best matches the motor activity vector
   is played.

## Running it

```bash
cd backend
../.venv/bin/uvicorn main:app --host 127.0.0.1 --port 8770
```

Then open http://127.0.0.1:8770 in a browser. You play White; the fly plays
Black.

## Rebuilding the connectome graph

Data (not committed, ~900MB) lives in `data/`:
- `proofread_connections_783.feather` — from [Zenodo record 10676866](https://zenodo.org/records/10676866)
- `neuron_annotations.tsv` — from [flyconnectome/flywire_annotations](https://github.com/flyconnectome/flywire_annotations)

```bash
cd scripts
../.venv/bin/python3 build_graph.py
```

This writes `data/graph/{weights.npz, root_ids.npy, sensory_idx.npy,
motor_idx.npy, neuron_meta.parquet}`, which `backend/connectome_fly_brain.py`
loads at startup.

## Project layout

```
backend/
  main.py                  FastAPI app: /api/fly-move, /api/health
  fly_brain.py              FlyBrain interface + placeholder + brain selection
  connectome_fly_brain.py   the real spiking engine
frontend/
  index.html                chess.js + chessboard.js UI
scripts/
  build_graph.py             connectome -> cached sparse graph + classification
data/                        raw downloads + data/graph/ cache (gitignored)
```

## Known limitations / honest caveats

- The readout is untrained, so don't expect coherent strategy — expect
  plausible-looking but not "good" moves, closer to a strong bias for certain
  move types than genuine evaluation.
- ~1.5s per move (single-threaded sparse matvec over 15M weights × 25 steps).
- The board→neuron mapping (64 squares → sensory neuron clusters) is an
  arbitrary but fixed assignment, not a biologically real retinotopic map —
  the fly doesn't "see" a chessboard.
