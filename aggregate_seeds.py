"""
aggregate_seeds.py — compute mean ± std across multi-seed runs for GCBM
(all backbone variants), GCBM with fixed thresholds (Phase 11b), and CNN
baselines, producing a LaTeX-ready results table.

Run patterns discovered in netscratch
--------------------------------------
  GCBM (ResNet-50):
    <RUNS_ROOT>/a_run_gv1_mv1_ps70_sr0.5_gat_s<SEED>/
  GCBM (DenseNet-201):
    <RUNS_ROOT>/a_run_gv1_mv1_ps70_sr0.5_gat_bbdensenet201_s<SEED>/
  GCBM (MobileNet-V2):
    <RUNS_ROOT>/a_run_gv1_mv1_ps70_sr0.5_gat_bbmobilenet_v2_s<SEED>/
  GCBM τ=0.5, PH2-only (ResNet-50):
    <RUNS_ROOT>/a_run_gv1_mv1_ps70_sr0.5_gat_tau0.5_s<SEED>/
  GCBM τ=0.5, PH2-only (DenseNet-201):
    <RUNS_ROOT>/a_run_gv1_mv1_ps70_sr0.5_gat_tau0.5_bbdensenet201_s<SEED>/
  GCBM τ=0.2, HAM10000-only (ResNet-50):
    <RUNS_ROOT>/a_run_gv1_mv1_ps70_sr0.5_gat_tau0.2_s<SEED>/
  CNN baseline:
    <RUNS_ROOT>/baseline_cnn_<DS>_<BACKBONE>_s<SEED>/

Usage
-----
  # All GCBM backbones + all CNN baselines (recommended — produces full paper table)
  python aggregate_seeds.py \\
      --gcbm-prefix a_run_gv1_mv1_ps70_sr0.5_gat \\
      --gcbm-backbones resnet50 densenet201 mobilenet_v2 \\
      --cnn-backbones resnet50 densenet201 mobilenet_v2 \\
      --seeds 42 123 456 \\
      --all-metrics \\
      --out-csv 0_seed_aggregated_results/all_results.csv

  # Include thresholded rows (Phase 11b):
  python aggregate_seeds.py \\
      --gcbm-prefix a_run_gv1_mv1_ps70_sr0.5_gat \\
      --gcbm-tau-prefix a_run_gv1_threshold_mv1_ps70_sr0.5_gat \\
      --gcbm-backbones resnet50 densenet201 mobilenet_v2 \\
      --gcbm-thresholds 0.5:ph2 0.2:ham10000 \\
      --seeds 42 123 456 \\
      --all-metrics

  # GCBM only (ResNet-50 default backbone)
  python aggregate_seeds.py \\
      --gcbm-prefix a_run_gv1_mv1_ps70_sr0.5_gat \\
      --seeds 42 123 456

  # CNN baselines only
  python aggregate_seeds.py \\
      --cnn-backbones resnet50 densenet201 mobilenet_v2 \\
      --seeds 42 123 456

Outputs
-------
  - Console: formatted mean ± std table per (config, dataset)
  - Optional --out-csv: writes the same table as a CSV file

Notes
-----
  GCBM run-ID construction (from sbatch_train_gcbm.sh lines 98-105):
    <PREFIX>
      [_tau<TAU>]          ← appended when SIM_THRESHOLD > 0
      [_bb<BACKBONE>]      ← appended when backbone != resnet50
      [_s<SEED>]           ← appended when SEED is set

  IMPORTANT: threshold runs are submitted with CBM_GRAPH_VARIANT=v1_threshold,
  which changes the _g<variant>_ segment in the run ID from _gv1_ to _gv1_threshold_.
  Therefore the base prefix for threshold runs differs from --gcbm-prefix:
    Standard:  a_run_gv1_mv1_ps70_sr0.5_gat
    Threshold: a_run_gv1_threshold_mv1_ps70_sr0.5_gat
  Use --gcbm-tau-prefix to supply the correct threshold prefix explicitly.

  GCBM metrics path:
    <RUNS_ROOT>/<RUN_ID>/concept_graph_data/<DS>/models/<DS>/metrics.json

  CNN baseline metrics path:
    <RUNS_ROOT>/baseline_cnn_<DS>_<BACKBONE>_s<SEED>/concept_graph_data/
        <DS>/models_cnn/<DS>/metrics_cnn_<BACKBONE>.json

  All metrics files contain: test.{acc, f1, auc}
  balanced_acc is included in METRICS list but may not be present in all files
  (missing keys are silently skipped and shown as "—").

  Thresholded runs (--gcbm-thresholds) are dataset-specific:
    τ=0.5 was only submitted for ph2   → all other dataset cells show "—"
    τ=0.2 was only submitted for ham10000 → all other dataset cells show "—"
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np

RUNS_ROOT = "/netscratch/mhossain/cbm_gat/runs"
DATASETS  = ["ham10000", "ph2", "derm7pt", "imagenet"]
METRICS   = ["auc", "f1", "acc", "balanced_acc"]

# Human-readable display names for backbone identifiers
BACKBONE_DISPLAY = {
    "resnet50":     "ResNet-50",
    "densenet201":  "DenseNet-201",
    "mobilenet_v2": "MobileNet-V2",
}

_DS_DISPLAY = {
    "ham10000": "HAM10000",
    "ph2":      "PH2",
    "derm7pt":  "Derm7pt",
    "imagenet": "ImageNet",
}


def _backbone_display(bb: str) -> str:
    return BACKBONE_DISPLAY.get(bb, bb)


def _load_json_metric(path: str, metric: str) -> float | None:
    """Load a single test metric from a metrics.json file. Returns None if
    the file is missing, the key is absent, or the value is NaN."""
    if not os.path.isfile(path):
        print(f"  [missing] {path}", file=sys.stderr)
        return None
    with open(path) as f:
        data = json.load(f)
    val = data.get("test", {}).get(metric)
    if val is None:
        return None
    if isinstance(val, float) and val != val:  # NaN guard
        return None
    return float(val)


def _build_gcbm_run_id(base_prefix: str, tau: float | None,
                       backbone: str, seed: int) -> str:
    """Construct a GCBM run-ID following the same logic as sbatch_train_gcbm.sh:
      <base_prefix>[_tau<tau>][_bb<backbone>][_s<seed>]
    tau is omitted when None or 0; backbone suffix omitted for resnet50.
    """
    run_id = base_prefix
    if tau:
        run_id += f"_tau{tau}"
    if backbone != "resnet50":
        run_id += f"_bb{backbone}"
    run_id += f"_s{seed}"
    return run_id


def _load_gcbm(base_prefix: str, backbone: str, seeds: List[int],
               ds: str, metric: str) -> List[float]:
    """Load GCBM test metrics across seeds for a given backbone (no threshold)."""
    values = []
    for seed in seeds:
        run_id = _build_gcbm_run_id(base_prefix, None, backbone, seed)
        path = os.path.join(
            RUNS_ROOT, run_id, "concept_graph_data",
            ds, "models", ds, "metrics.json")
        val = _load_json_metric(path, metric)
        if val is not None:
            values.append(val)
    return values


def _load_gcbm_tau(base_prefix: str, tau: float, backbone: str,
                   seeds: List[int], target_ds: str,
                   ds: str, metric: str) -> List[float]:
    """Load thresholded GCBM metrics across seeds.

    These runs were only submitted for *target_ds* (e.g. 'ph2' for τ=0.5).
    For any other dataset the function returns [] so the cell shows "—".
    """
    if ds != target_ds:
        return []
    values = []
    for seed in seeds:
        run_id = _build_gcbm_run_id(base_prefix, tau, backbone, seed)
        path = os.path.join(
            RUNS_ROOT, run_id, "concept_graph_data",
            ds, "models", ds, "metrics.json")
        val = _load_json_metric(path, metric)
        if val is not None:
            values.append(val)
    return values


def _load_cnn(backbone: str, seeds: List[int], ds: str,
              metric: str) -> List[float]:
    """Load CNN baseline test metrics across seeds."""
    values = []
    for seed in seeds:
        run_id = f"baseline_cnn_{ds}_{backbone}_s{seed}"
        path = os.path.join(
            RUNS_ROOT, run_id, "concept_graph_data",
            ds, "models_cnn", ds, f"metrics_cnn_{backbone}.json")
        val = _load_json_metric(path, metric)
        if val is not None:
            values.append(val)
    return values


def _fmt(values: List[float]) -> str:
    if not values:
        return "—"
    if len(values) == 1:
        return f"{values[0]:.3f}"
    return f"{np.mean(values):.3f} ± {np.std(values):.3f}  (n={len(values)})"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Aggregate multi-seed results into mean ± std table.")
    ap.add_argument("--gcbm-prefix", default=None,
                    help="Base RUN_ID prefix for GCBM runs, without any "
                         "_bb<backbone> or _s<seed> suffix. "
                         "E.g. 'a_run_gv1_mv1_ps70_sr0.5_gat'")
    ap.add_argument("--gcbm-backbones", nargs="*", default=["resnet50"],
                    metavar="BACKBONE",
                    help="GCBM backbone variants to aggregate. "
                         "Use 'resnet50' for the default run (no _bb suffix). "
                         "Default: resnet50. "
                         "E.g. --gcbm-backbones resnet50 densenet201 mobilenet_v2")
    ap.add_argument("--gcbm-tau-prefix", default=None,
                    help="Base RUN_ID prefix for threshold runs (Phase 11b). "
                         "These runs use CBM_GRAPH_VARIANT=v1_threshold so the graph "
                         "variant segment in the run ID is 'gv1_threshold' instead of 'gv1'. "
                         "Default: auto-derived from --gcbm-prefix by replacing '_gv1_' "
                         "with '_gv1_threshold_'. "
                         "E.g. 'a_run_gv1_threshold_mv1_ps70_sr0.5_gat'")
    ap.add_argument("--gcbm-thresholds", nargs="*", default=[],
                    metavar="TAU:DATASET",
                    help="Fixed-threshold GCBM rows to add (Phase 11b). "
                         "Each entry is '<tau>:<dataset>', e.g. '0.5:ph2 0.2:ham10000'. "
                         "For each entry one row per backbone (--gcbm-backbones) is added; "
                         "only the specified dataset will have data, others show '—'.")
    ap.add_argument("--cnn-backbones", nargs="*", default=[],
                    metavar="BACKBONE",
                    help="CNN baseline backbone names to aggregate. "
                         "E.g. --cnn-backbones resnet50 densenet201 mobilenet_v2")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456],
                    help="Seeds to aggregate over (default: 42 123 456)")
    ap.add_argument("--datasets", nargs="*", default=DATASETS,
                    help="Datasets to include (default: all four)")
    ap.add_argument("--metric", default="auc",
                    choices=METRICS,
                    help="Primary metric to display (default: auc)")
    ap.add_argument("--all-metrics", action="store_true",
                    help="Print all metrics instead of just --metric")
    ap.add_argument("--out-csv", default=None,
                    help="Optional path to write CSV output.")
    args = ap.parse_args()

    metrics_to_show = METRICS if args.all_metrics else [args.metric]
    datasets = args.datasets

    # Derive tau prefix if not explicitly given.
    # Threshold runs use CBM_GRAPH_VARIANT=v1_threshold which turns "_gv1_"
    # into "_gv1_threshold_" in the run ID.
    if args.gcbm_tau_prefix:
        tau_prefix = args.gcbm_tau_prefix
    elif args.gcbm_prefix:
        tau_prefix = args.gcbm_prefix.replace("_gv1_", "_gv1_threshold_", 1)
    else:
        tau_prefix = None

    # Build configs list: (display_name, loader_fn)
    # Loaders have signature (seeds, ds, metric) -> List[float]
    configs: List[Tuple[str, object]] = []

    if args.gcbm_prefix:
        for bb in args.gcbm_backbones:
            disp = _backbone_display(bb)
            configs.append((
                f"GCBM ({disp})",
                lambda s, d, m, pfx=args.gcbm_prefix, b=bb:
                    _load_gcbm(pfx, b, s, d, m)
            ))

        # Thresholded rows (Phase 11b) — one row per (tau, dataset, backbone)
        if args.gcbm_thresholds and tau_prefix is None:
            print("[warn] --gcbm-thresholds requested but --gcbm-prefix is not set; "
                  "cannot derive tau prefix — skipping threshold rows.", file=sys.stderr)
        elif args.gcbm_thresholds:
            print(f"[tau rows] using prefix: {tau_prefix}", file=sys.stderr)
        for spec in args.gcbm_thresholds:
            if ":" not in spec:
                print(f"[warn] --gcbm-thresholds entry '{spec}' must be "
                      f"'<tau>:<dataset>' — skipping", file=sys.stderr)
                continue
            tau_str, target_ds = spec.split(":", 1)
            try:
                tau = float(tau_str)
            except ValueError:
                print(f"[warn] cannot parse tau '{tau_str}' in '{spec}' "
                      f"— skipping", file=sys.stderr)
                continue
            for bb in args.gcbm_backbones:
                disp = _backbone_display(bb)
                ds_disp = _DS_DISPLAY.get(target_ds, target_ds.upper())
                row_label = f"GCBM τ={tau} [{ds_disp}] ({disp})"
                configs.append((
                    row_label,
                    lambda s, d, m,
                           pfx=tau_prefix, t=tau, b=bb, tds=target_ds:
                        _load_gcbm_tau(pfx, t, b, s, tds, d, m)
                ))

    for bb in args.cnn_backbones:
        disp = _backbone_display(bb)
        configs.append((
            f"CNN baseline ({disp})",
            lambda s, d, m, b=bb: _load_cnn(b, s, d, m)
        ))

    if not configs:
        print("Nothing to aggregate — pass --gcbm-prefix and/or "
              "--cnn-backbones.", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------ table
    # col_w must accommodate the longest threshold row label, e.g.
    # "GCBM τ=0.5 [HAM10000] (DenseNet-201)"  ≈ 40 chars
    col_w = max(40, *(len(n) for n, _ in configs)) + 2
    ds_w  = 22
    header = f"{'Config':<{col_w}}" + "".join(
        f"  {ds:<{ds_w}}" for ds in datasets)
    separator = "-" * len(header)

    rows_csv: List[Dict[str, str]] = []

    for metric in metrics_to_show:
        print(f"\n{'='*len(header)}")
        print(f"  Metric: {metric.upper()}")
        print(header)
        print(separator)

        for name, loader in configs:
            row: Dict[str, str] = {"config": name, "metric": metric}
            cells = []
            for ds in datasets:
                vals = loader(args.seeds, ds, metric)
                cell = _fmt(vals)
                cells.append(cell)
                row[ds] = cell
            print(f"{name:<{col_w}}" + "".join(
                f"  {c:<{ds_w}}" for c in cells))
            rows_csv.append(row)

        print(separator)

    # ------------------------------------------------------------------ CSV
    if args.out_csv and rows_csv:
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        fieldnames = ["config", "metric"] + list(datasets)
        with open(args.out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows_csv)
        print(f"\nCSV written to {args.out_csv}")


if __name__ == "__main__":
    main()
