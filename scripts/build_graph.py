"""
Build the FAFB connectome into a cached sparse weighted graph + neuron
classification, ready for the spiking simulation.

Inputs (in ../data):
  - proofread_connections_783.feather  (pre/post/neuropil/syn_count/NT-probs)
  - neuron_annotations.tsv              (flow, super_class, cell_type, side, ...)

Outputs (in ../data/graph):
  - weights.npz          scipy sparse CSR, W[post_idx, pre_idx] = signed weight
  - root_ids.npy         int64 array: index -> root_id
  - sensory_idx.npy      int array: indices classified as sensory input (afferent)
  - motor_idx.npy        int array: indices classified as motor output (descending/motor)
  - neuron_meta.parquet  per-index metadata (root_id, super_class, cell_type, side)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
OUT = DATA / "graph"
OUT.mkdir(exist_ok=True)


def main():
    print("Loading annotations...")
    ann = pd.read_csv(DATA / "neuron_annotations.tsv", sep="\t", low_memory=False)
    ann = ann.drop_duplicates(subset="root_id").set_index("root_id")
    print(f"  {len(ann)} annotated neurons")

    print("Loading connections (this is the big one)...")
    conn = pd.read_feather(DATA / "proofread_connections_783.feather")
    print(f"  {len(conn)} region-level edge rows")

    print("Aggregating to one weight per (pre, post) pair...")
    nt_cols = ["gaba_avg", "ach_avg", "glut_avg", "oct_avg", "ser_avg", "da_avg"]
    # weight NT probabilities by syn_count of that region-row before summing
    for c in nt_cols:
        conn[c] = conn[c] * conn["syn_count"]

    agg = conn.groupby(["pre_pt_root_id", "post_pt_root_id"], sort=False).agg(
        syn_count=("syn_count", "sum"),
        **{c: (c, "sum") for c in nt_cols},
    ).reset_index()
    for c in nt_cols:
        agg[c] = agg[c] / agg["syn_count"]

    excitatory = agg["ach_avg"] + agg["oct_avg"] + agg["ser_avg"] + agg["da_avg"]
    inhibitory = agg["gaba_avg"] + agg["glut_avg"]
    agg["signed_weight"] = agg["syn_count"] * (excitatory - inhibitory)
    print(f"  {len(agg)} unique directed neuron-pair edges")

    print("Building index over neurons that appear in the connectome...")
    all_ids = pd.unique(pd.concat([agg["pre_pt_root_id"], agg["post_pt_root_id"]], ignore_index=True))
    all_ids.sort()
    root_ids = all_ids
    id_to_idx = pd.Series(np.arange(len(root_ids)), index=root_ids)
    n = len(root_ids)
    print(f"  {n} unique neurons in graph")

    pre_idx = id_to_idx.loc[agg["pre_pt_root_id"]].to_numpy()
    post_idx = id_to_idx.loc[agg["post_pt_root_id"]].to_numpy()
    weights = agg["signed_weight"].to_numpy(dtype=np.float32)

    print("Building sparse CSR matrix W[post, pre] = weight ...")
    W = sp.csr_matrix((weights, (post_idx, pre_idx)), shape=(n, n))
    sp.save_npz(OUT / "weights.npz", W)
    np.save(OUT / "root_ids.npy", root_ids)
    print(f"  saved weights.npz ({W.nnz} nonzeros) and root_ids.npy")

    print("Classifying sensory (afferent) and motor (descending/motor) populations...")
    meta = ann.reindex(root_ids)
    is_sensory = (meta["flow"] == "afferent").to_numpy()
    is_motor = meta["super_class"].isin(["descending", "motor"]).to_numpy()

    sensory_idx = np.nonzero(is_sensory)[0]
    motor_idx = np.nonzero(is_motor)[0]
    np.save(OUT / "sensory_idx.npy", sensory_idx)
    np.save(OUT / "motor_idx.npy", motor_idx)
    print(f"  sensory: {len(sensory_idx)}  motor: {len(motor_idx)}")

    meta_out = pd.DataFrame({
        "root_id": root_ids,
        "super_class": meta["super_class"].to_numpy(),
        "cell_type": meta["cell_type"].to_numpy(),
        "side": meta["side"].to_numpy(),
        "flow": meta["flow"].to_numpy(),
    })
    meta_out.to_parquet(OUT / "neuron_meta.parquet")
    print("  saved neuron_meta.parquet")
    print("Done.")


if __name__ == "__main__":
    main()
