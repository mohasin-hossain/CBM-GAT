VARIANTS = {
    "graph": "v4",
    "model": "v1",
    "train_model": "v1",
}

REGISTRY = {
    "graph": {
        "v1": "graph_v1",          # original: CNN features
        "v2": "graph_v2",          # summary statistics
        "v3": "graph_v3",          # co-occurrence + scalars
        "v4": "graph_v4",          # cosine similarity (raw, no hand-crafted stats)
    },
    "model": {
        "v1": "model_v1",          # original: Accuracy only
        "v2": "model_v2",          # Recall + val_bal_acc
    },
    "train_model": {
        "v1": "train_model_v1",    # original: val_loss, no class weights
        "v2": "train_model_v2",    # val_bal_acc, class weights, per-class metrics
    },
}
