"""
eval_threshold_sweep_v2.py — F1 / AUC vs τ sweep over the threshold
graph variants (`graph_v1_threshold` and `graph_v4` with
`sim_threshold`).

F1/AUC vs sim-threshold sweep evaluator
(left on disk but no longer referenced by the active research plan,
see `RESEARCH_PLAN.md`). For each (dataset, τ) it:

  1. Loads the chosen data split (train / val / test; default **val**) via the
     dataset registry.
  2. Rebuilds that split's concept graphs in memory using
     `graph_v1_threshold` (resp. `graph_v4` with `sim_threshold`) at the
     requested τ.
  3. Forwards them through the ALREADY-TRAINED R1 (resp. R3) GAT
     checkpoint — no retraining.
  4. Records {f1, auc, acc, balanced_acc, mean_active_concepts,
     mean_active_edges, n_graphs} per τ.

Outputs (under ``--out-dir``; default under ``<run_root>/``):
  - ``threshold_sweep/`` when ``--split val`` (default)
  - ``threshold_sweep_test/`` or ``threshold_sweep_train/`` for other splits
  - threshold_sweep_results.json   (consolidated, all datasets)
  - threshold_sweep_<dataset>.csv  (one row per τ, for plotting)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from dgl.dataloading import GraphDataLoader
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score, roc_auc_score)

from concepts import build_model_parts, load_craft_and_attach
from config import DATASETS, get_dataset_params
from utils import _set_seed
from model_v1 import EGATClassifier, GAT_LightningModule


# ---------------------------------------------------------------------------
# Variant resolution
# ---------------------------------------------------------------------------

def _import_graph_module(graph_variant: str):
    """Resolve the python module that implements the requested graph
    variant. `v4_threshold` reuses `graph_v4` directly because
    `graph_v4.ConceptGraphDataset` already accepts `sim_threshold`."""
    if graph_variant == "v1_threshold":
        import graph_v1_threshold as gm
    elif graph_variant == "v4_threshold":
        import graph_v4 as gm
    else:
        raise ValueError(f"Unsupported graph variant: {graph_variant}")
    return gm


def _craft_path(run_root: str, dataset: str) -> str:
    return os.path.join(
        run_root, dataset, "craft", dataset, f"craft_{dataset}.dill")


def _checkpoint_path(run_root: str, dataset: str) -> str:
    return os.path.join(
        run_root, dataset, "models", dataset, f"{dataset}_best_model.ckpt")


def _infer_backbone_from_run_root(run_root: str) -> str:
    """``run_root`` is ``.../<run_id>/concept_graph_data``. Pick the CRAFT CNN
    backbone from ``run_id`` so NMF patch dimensions match the serialized
    ``craft_*.dill`` (ResNet-50: 2048 ch, DenseNet-201: 1920, MobileNet-V2:
    1280)."""
    rid = os.path.basename(os.path.dirname(os.path.abspath(run_root)))
    if "bbmobilenet_v2" in rid:
        return "mobilenet_v2"
    if "bbdensenet201" in rid:
        return "densenet201"
    return "resnet50"


# ---------------------------------------------------------------------------
# In-memory threshold rebuild
# ---------------------------------------------------------------------------

def _build_split_graphs(graph_variant: str, dataset_key: str,
                        run_root: str, device: str,
                        patch_size: int, stride_r: float,
                        sim_threshold: float, split: str,
                        craft_root: str | None = None,
                        backbone: str = "resnet50"):
    """Rebuild concept graphs for one dataset split at the requested
    threshold. ``split`` must be ``train``, ``val``, or ``test`` (registry
    convention). Returns a list of (graph, label)."""
    spec = DATASETS[dataset_key]
    tdict = spec.build_transforms()
    paths = spec.resolve_paths()
    images, labels, _ = spec.load_split(paths, tdict, split=split)
    if images.numel() == 0:
        return []

    g, h = build_model_parts(backbone, device=device, pretrained=True)
    craft = load_craft_and_attach(
        _craft_path(craft_root or run_root, dataset_key), g, h)

    gm = _import_graph_module(graph_variant)
    if graph_variant == "v1_threshold":
        ds = gm.ConceptGraphDataset(
            images=images.to(device), y=labels.to(device), masks=None,
            patch_size=patch_size, craft_xai=craft, ignore_list=[],
            device=device, stride_r=stride_r, coverage_threshold=0.0,
            seed=42, requires_grad=False, sim_threshold=sim_threshold,
        )
    else:  # v4_threshold
        ds = gm.ConceptGraphDataset(
            images=images.to(device), y=labels.to(device), masks=None,
            patch_size=patch_size, craft_xai=craft, ignore_list=[],
            device=device, stride_r=stride_r, coverage_threshold=0.0,
            seed=42, requires_grad=False, sim_threshold=sim_threshold,
        )
    ds.process()
    return list(zip(ds.graphs, ds.labels))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def _evaluate(model, graph_label_pairs, device, num_classes):
    model.eval()
    ys, probs = [], []
    n_active_concepts: List[float] = []
    n_active_edges: List[float] = []

    # Use a GraphDataLoader for batching; build a tiny dataset wrapper.
    class _DS:
        def __init__(self, items):
            self.items = items
        def __getitem__(self, i):
            return self.items[i]
        def __len__(self):
            return len(self.items)

    loader = GraphDataLoader(
        _DS(graph_label_pairs), batch_size=32, shuffle=False, drop_last=False)

    for bg, y in loader:
        bg = bg.to(device)
        x = bg.ndata["feat"].float().to(device)
        logits, _, _ = model(bg, x)
        probs.append(F.softmax(logits, dim=1).cpu())
        ys.append(y.cpu())

        # active = any non-zero feature value on the node
        feat = x.detach()
        active_per_node = (feat.abs().sum(dim=1) > 0).float()
        # split per graph using batch_num_nodes
        sizes = bg.batch_num_nodes().cpu().tolist()
        offset = 0
        for s in sizes:
            n_active_concepts.append(
                float(active_per_node[offset:offset + s].sum().item()))
            offset += s
        # edges per graph
        for s in bg.batch_num_edges().cpu().tolist():
            n_active_edges.append(float(s))

    if not ys:
        return {
            "acc": float("nan"), "f1": float("nan"),
            "auc": float("nan"), "balanced_acc": float("nan"),
            "mean_active_concepts": 0.0, "mean_active_edges": 0.0,
            "n_graphs": 0,
        }

    y_true = torch.cat(ys).numpy()
    y_prob = torch.cat(probs).numpy()
    y_pred = y_prob.argmax(axis=1)
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, average="weighted"))
    bal = float(balanced_accuracy_score(y_true, y_pred))
    try:
        auc = float(
            roc_auc_score(y_true, y_prob[:, 1], average="macro")
            if num_classes == 2
            else roc_auc_score(y_true, y_prob, multi_class="ovr",
                               average="macro"))
    except Exception:
        auc = float("nan")

    return {
        "acc": acc, "f1": f1, "auc": auc, "balanced_acc": bal,
        "mean_active_concepts": float(np.mean(n_active_concepts)),
        "mean_active_edges": float(np.mean(n_active_edges)),
        "n_graphs": int(len(y_true)),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        "eval_threshold_sweep_v2 — F1/AUC vs tau over threshold variants")
    ap.add_argument("--run-root", required=True,
                    help="Path to the trained run, e.g. "
                         "/netscratch/.../<run_id>/concept_graph_data")
    ap.add_argument("--datasets", nargs="+",
                    default=["ham10000", "ph2", "derm7pt", "imagenet"],
                    choices=list(DATASETS.keys()))
    ap.add_argument("--graph-variant", nargs="+",
                    default=["v1_threshold", "v4_threshold"],
                    choices=["v1_threshold", "v4_threshold"],
                    help="One or more threshold-aware graph variants. "
                         "Each is evaluated independently across all "
                         "datasets and τ values.")
    ap.add_argument("--taus", nargs="+", type=float,
                    default=[0.1, 0.2, 0.3, 0.4, 0.5,
                             0.6, 0.7, 0.8, 0.9, 1.0])
    ap.add_argument("--patch-size", type=int, default=70)
    ap.add_argument("--stride-r", type=float, default=0.5)
    ap.add_argument("--out-dir", default=None,
                    help="Output directory. Default: "
                         "<run_root>/threshold_sweep/ when --split val, "
                         "else <run_root>/threshold_sweep_<split>/")
    ap.add_argument(
        "--split", default="val", choices=["train", "val", "test"],
        help="Which data split to rebuild graphs on and evaluate (registry "
             "keys: train, val, test). Default: val (validation).")
    ap.add_argument("--craft-root", default=None,
                    help="Root containing craft .dill files. Defaults to "
                         "--run-root. Set to the base (no-seed) run when "
                         "evaluating a seeded checkpoint whose run folder "
                         "has no craft .dill.")
    ap.add_argument(
        "--backbone", default="auto",
        choices=["auto", "resnet50", "densenet201", "mobilenet_v2"],
        help="CNN encoder for CRAFT patch features — must match the backbone "
             "used when the craft .dill was built. "
             "'auto' infers from the parent run folder name "
             "(_bbdensenet201 / _bbmobilenet_v2 vs ResNet-50 default).")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--num-heads", type=int, default=None)
    args = ap.parse_args()

    device = (args.device
              if (args.device.startswith("cuda") and torch.cuda.is_available())
              else "cpu")
    _set_seed(args.seed)
    craft_root = args.craft_root or args.run_root
    backbone = (
        _infer_backbone_from_run_root(args.run_root)
        if args.backbone == "auto"
        else args.backbone)
    print(f"Backbone (CRAFT patch encoder): {backbone}")

    if args.out_dir is not None:
        out_dir = args.out_dir
    else:
        sub = ("threshold_sweep" if args.split == "val"
               else f"threshold_sweep_{args.split}")
        out_dir = os.path.join(args.run_root, sub)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Split: {args.split}  |  Output directory: {out_dir}")

    consolidated = {}

    for ds in args.datasets:
        consolidated[ds] = {}
        ds_params = get_dataset_params(ds) or {}
        num_heads = (args.num_heads
                     if args.num_heads is not None
                     else ds_params.get("num_heads"))
        hidden_dim = (args.hidden_dim
                      if args.hidden_dim is not None
                      else ds_params.get("hidden_dim", args.hidden_dim))

        ckpt = _checkpoint_path(args.run_root, ds)
        if not os.path.isfile(ckpt):
            print(f"[skip dataset {ds}] checkpoint not found: {ckpt}")
            continue
        if not os.path.isfile(_craft_path(craft_root, ds)):
            print(f"[skip dataset {ds}] craft not found in {craft_root}")
            continue

        for variant in args.graph_variant:
            print(f"\n==================== {ds} | {variant} | split={args.split} "
                  f"====================")
            rows = []
            csv_path = os.path.join(
                out_dir, f"threshold_sweep_{ds}_{variant}.csv")
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "tau", "f1", "auc", "acc", "balanced_acc",
                        "mean_active_concepts", "mean_active_edges",
                        "n_graphs"])
                writer.writeheader()

                # Build the trained GAT once per (dataset, variant);
                # in_dim is determined by the FIRST τ build.
                gat = None
                for tau in args.taus:
                    print(f"  τ = {tau:.2f}")
                    pairs = _build_split_graphs(
                        graph_variant=variant, dataset_key=ds,
                        run_root=args.run_root, device=device,
                        patch_size=args.patch_size,
                        stride_r=args.stride_r, sim_threshold=tau,
                        split=args.split, craft_root=craft_root,
                        backbone=backbone)
                    if not pairs:
                        print(f"    [skip] no graphs at τ={tau}")
                        continue

                    if gat is None:
                        in_dim = pairs[0][0].ndata["feat"].shape[1]
                        # Prefer the canonical class count from the
                        # dataset registry; fall back to label max+1.
                        spec = DATASETS[ds]
                        cn = getattr(spec, "class_names", None) or []
                        if cn:
                            num_classes = len(cn)
                        else:
                            num_classes = (
                                int(max(int(p[1].item())
                                        for p in pairs)) + 1)

                        model = EGATClassifier(
                            in_feats=in_dim, out_feats=hidden_dim,
                            num_heads=num_heads, out_dim=num_classes,
                            feat_drop=0.0, node_drop=0.0)
                        gat = GAT_LightningModule.load_from_checkpoint(
                            ckpt, model=model).model.eval().to(device)

                    metrics = _evaluate(gat, pairs, device, num_classes)
                    row = {"tau": float(tau), **metrics}
                    rows.append(row)
                    writer.writerow(row)
                    print(f"    f1={metrics['f1']:.4f}  "
                          f"auc={metrics['auc']:.4f}  "
                          f"active_concepts="
                          f"{metrics['mean_active_concepts']:.2f}")

            consolidated[ds][variant] = rows
            print(f"  -> {csv_path}")

    json_path = os.path.join(out_dir, "threshold_sweep_results.json")
    with open(json_path, "w") as f:
        json.dump(consolidated, f, indent=2)
    print(f"\nConsolidated JSON: {json_path}")


if __name__ == "__main__":
    main()
