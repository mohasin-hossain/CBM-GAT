import os

# ---------------------------------------------------------------------------
# Variant selection — driven by environment variables so that each sbatch job
# carries its own variant without touching this file.
#
# Set these env vars in your sbatch script before srun. v3 defaults shown:
#   export CBM_GRAPH_VARIANT="v1"
#   export CBM_MODEL_VARIANT="v1"
#   export CBM_TRAIN_VARIANT="v1"
#
# Set CBM_GRAPH_VARIANT="v1_threshold" if you also want the per-patch
# sim-threshold gate at build time . Pair it with `--sim-threshold tau`
# on `build_concept_graphs.py`. `graph_v4` already supports the same
# knob natively, so there is no separate `v4_threshold` registry entry.
#
# Frontend pairing:
#   FRONTEND=gat    → model v1  + train_model v1    (GAT)
#   FRONTEND=cb_mlp / cb_linear → concept bottleneck z only (single-node K-dim feat);
#       graph concept_bottleneck_mlp_linear + model/train concept_bottleneck_{mlp,linear}
#
# If the env vars are not set, all three default to "v1" (original behaviour).
# ---------------------------------------------------------------------------

VARIANTS = {
    "graph":       os.environ.get("CBM_GRAPH_VARIANT",  "v1"),
    "model":       os.environ.get("CBM_MODEL_VARIANT",  "v1"),
    "train_model": os.environ.get("CBM_TRAIN_VARIANT",  "v1"),
}

REGISTRY = {
    "graph": {
        "v1":           "graph_v1",            # original: weighted CNN features
        "v4":           "graph_v4",            # cosine similarity (supports sim_threshold)
        "v1_threshold": "graph_v1_threshold",  # graph_v1 + per-patch sim-threshold gate 
        # K-dim bottleneck z only (single-node graphs); search token concept_bottleneck_mlp_linear
        "concept_bottleneck_mlp_linear": "graph_concept_bottleneck_mlp_linear",
    },
    "model": {
        "v1":   "model_v1",   # GAT classifier — primary model 
        "concept_bottleneck_mlp": "model_concept_bottleneck_mlp",     # MLP on z (FRONTEND=cb_mlp)
        "concept_bottleneck_linear": "model_concept_bottleneck_linear",  # linear on z (FRONTEND=cb_linear)
    },
    "train_model": {
        "v1":   "train_model_v1",   # GAT trainer 
        "concept_bottleneck_mlp": "train_model_concept_bottleneck_mlp",
        "concept_bottleneck_linear": "train_model_concept_bottleneck_linear",
    },
}
