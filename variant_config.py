import os

# ---------------------------------------------------------------------------
# Variant selection — driven by environment variables so that each sbatch job
# carries its own variant without touching this file.
#
# Set these env vars in your sbatch script before srun:
#   export CBM_GRAPH_VARIANT="v1p2"
#   export CBM_MODEL_VARIANT="v1p1"
#   export CBM_TRAIN_VARIANT="v1p1"
#
# If the env vars are not set, all three default to "v1" (original behaviour).
# This file never needs to be edited manually again.
# ---------------------------------------------------------------------------

VARIANTS = {
    "graph":       os.environ.get("CBM_GRAPH_VARIANT",  "v1"),
    "model":       os.environ.get("CBM_MODEL_VARIANT",  "v1"),
    "train_model": os.environ.get("CBM_TRAIN_VARIANT",  "v1"),
}

REGISTRY = {
    "graph": {
        "v1":   "graph_v1",   # original: weighted CNN features, no concept gate
        "v2":   "graph_v2",   # summary statistics
        "v3":   "graph_v3",   # co-occurrence + scalars
        "v4":   "graph_v4",   # cosine similarity (raw, no hand-crafted stats)
        "v1p2": "graph_v1p2", # NEW (primary): z_c thresholded gate on h^(0) + metadata JSON
                              #   + z_soft/z_onehot saved in .dgl for concept head ablation
                              #   PRIMARY fix for reviewer BaF4: concepts now gate node features
    },
    "model": {
        "v1":   "model_v1",   # original: GAT classifier — use with v1p2 graph for main result
        "v2":   "model_v2",   # Recall + val_bal_acc
        "v1p1": "model_v1p1", # NEW (ablation probe): GAT + linear concept head + learnable lambda
                              #   ABLATION only — not the primary model
        "v1p2": "model_v1p2", # NEW (ablation): MLP frontend replacing GAT — proves GAT is necessary
    },
    "train_model": {
        "v1":   "train_model_v1",   # original: val_loss, no class weights
        "v2":   "train_model_v2",   # val_bal_acc, class weights, per-class metrics
        "v1p1": "train_model_v1p1", # NEW: trains model_v1p1 — dual-head, 3 eval modes, lambda log
    },
}
