"""
aggregate_per_dataset_tau_seeds.py — mean ± std across seeds for Phase 2e runs:
`graph_v1_threshold` + `INCLUDE_DATASET_IN_RUN_ID=1` (see train_gcbm_per_tau_multiseed.sh).

  Omit `_bb…` when `BACKBONE=resnet50` (see `build_run_id` in this module).
  resnet50:      ham10000 0.2, ph2 0.5, derm7pt 0.1, imagenet 0.1
                 (same τ table as Phase 3 concept-bottleneck runs; see train_cb_ml_per_tau_multiseed.sh)
  densenet201:  ham10000 0.3, ph2 0.3, derm7pt 0.1, imagenet 0.1
  mobilenet_v2: ham10000 0.2, ph2 0.4, derm7pt 0.1, imagenet 0.2

Metrics are read from:
  <RUNS_ROOT>/<RUN_ID>/concept_graph_data/<DS>/models/<DS>/metrics.json
  → test.{auc,f1,acc,balanced_acc}

Usage
-----
  python aggregate_per_dataset_tau_seeds.py \\
      --out-csv /path/to/per_ds_tau_multiseed.csv \\
      --out-json /path/to/per_ds_tau_multiseed.json

  python aggregate_per_dataset_tau_seeds.py \\
      --tau-prefix a_run_gv1_threshold_mv1_ps70_sr0.5_gat \\
      --seeds 42 123 456

Concept-bottleneck z (Phase 3, FRONTEND=cb_mlp / cb_linear) uses the same per-backbone
τ tables as G-CBM (resnet50 / densenet201 / mobilenet_v2). Example --tau-prefix values:
  ``a_run_ggcbml_mmcbmlp_ps70_sr0.5_cb_mlp`` / ``a_run_ggcbml_mmcblin_ps70_sr0.5_cb_linear``
(omit _tau*_bb*_ds*_s* suffixes; see markdowns/TESTING_EXECUTION.md Phase 3a–3b).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from glob import glob
from typing import Any, Dict, List, Optional

import numpy as np

RUNS_ROOT = "/netscratch/mhossain/cbm_gat/runs"
DATASETS = ["ham10000", "ph2", "derm7pt", "imagenet"]
METRICS = ["auc", "f1", "acc", "balanced_acc"]

# Must stay in sync with scripts/train_gcbm_per_tau_multiseed.sh (dense/mobilenet)
# and scripts/train_cb_ml_per_tau_multiseed.sh (ResNet concept-bottleneck z).
TAU_BY_BACKBONE: Dict[str, Dict[str, float]] = {
    "resnet50": {
        "ham10000": 0.2,
        "ph2": 0.5,
        "derm7pt": 0.1,
        "imagenet": 0.1,
    },
    "densenet201": {
        "ham10000": 0.3,
        "ph2": 0.3,
        "derm7pt": 0.1,
        "imagenet": 0.1,
    },
    "mobilenet_v2": {
        "ham10000": 0.2,
        "ph2": 0.4,
        "derm7pt": 0.1,
        "imagenet": 0.2,
    },
}

_BACKBONE_LABEL = {
    "resnet50": "ResNet-50",
    "densenet201": "DenseNet-201",
    "mobilenet_v2": "MobileNet-V2",
}


def _default_tau_prefix(gcbm_prefix: str) -> str:
    """Match aggregate_seeds.py: v1 → v1_threshold in run id."""
    return gcbm_prefix.replace("_gv1_", "_gv1_threshold_", 1)


def _metrics_path(runs_root: str, run_id: str, ds: str) -> str:
    return os.path.join(
        runs_root,
        run_id,
        "concept_graph_data",
        ds,
        "models",
        ds,
        "metrics.json",
    )


def _load_test_metric(path: str, metric: str) -> Optional[float]:
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        data = json.load(f)
    val = data.get("test", {}).get(metric)
    if val is None:
        return None
    v = float(val)
    if v != v:  # NaN
        return None
    return v


def build_run_id(
    tau_prefix: str,
    tau: float,
    backbone: str,
    dataset: str,
    seed: int,
) -> str:
    rid = f"{tau_prefix}_tau{tau}"
    if backbone != "resnet50":
        rid += f"_bb{backbone}"
    rid += f"_ds{dataset}"
    rid += f"_s{seed}"
    return rid


def _resolve_metrics_path(
    runs_root: str,
    tau_prefix: str,
    tau: float,
    backbone: str,
    dataset: str,
    seed: int,
) -> Optional[str]:
    """Exact run id first; if missing, try a glob for τ string variants."""
    rid = build_run_id(tau_prefix, tau, backbone, dataset, seed)
    p = _metrics_path(runs_root, rid, dataset)
    if os.path.isfile(p):
        return p
    bb_part = f"_bb{backbone}" if backbone != "resnet50" else ""
    pat = os.path.join(
        runs_root,
        f"{tau_prefix}_tau*{bb_part}_ds{dataset}_s{seed}",
    )
    for d in sorted(glob(pat)):
        if not os.path.isdir(d):
            continue
        mp = _metrics_path(runs_root, os.path.basename(d), dataset)
        if os.path.isfile(mp):
            return mp
    return None


def collect_seed_values(
    runs_root: str,
    tau_prefix: str,
    backbone: str,
    dataset: str,
    tau: float,
    seeds: List[int],
    metric: str,
    *,
    warn_metric: str = "auc",
) -> List[float]:
    values: List[float] = []
    for seed in seeds:
        p = _resolve_metrics_path(
            runs_root, tau_prefix, tau, backbone, dataset, seed
        )
        if p is None:
            if metric == warn_metric:
                rid = build_run_id(tau_prefix, tau, backbone, dataset, seed)
                print(
                    f"  [missing] {_metrics_path(runs_root, rid, dataset)}",
                    file=sys.stderr,
                )
            continue
        val = _load_test_metric(p, metric)
        if val is not None:
            values.append(val)
    return values


def fmt_mean_std(values: List[float]) -> str:
    if not values:
        return "—"
    if len(values) == 1:
        return f"{values[0]:.3f}"
    return f"{np.mean(values):.3f} ± {np.std(values, ddof=1):.3f}  (n={len(values)})"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Aggregate Phase 2e per-dataset τ multi-seed GCBM metrics."
    )
    ap.add_argument(
        "--tau-prefix",
        default=None,
        help="Run id segment before _tau… (default: derive from --gcbm-prefix "
        "or use a_run_gv1_threshold_mv1_ps70_sr0.5_gat)",
    )
    ap.add_argument(
        "--gcbm-prefix",
        default="a_run_gv1_mv1_ps70_sr0.5_gat",
        help="Non-threshold GCBM prefix; used only to derive --tau-prefix when "
             "--tau-prefix is omitted (default: a_run_gv1_mv1_ps70_sr0.5_gat)",
    )
    ap.add_argument(
        "--runs-root",
        default=RUNS_ROOT,
        help=f"Runs root (default: {RUNS_ROOT})",
    )
    ap.add_argument(
        "--backbones",
        nargs="+",
        default=["densenet201", "mobilenet_v2"],
        metavar="BACKBONE",
        choices=list(TAU_BY_BACKBONE.keys()),
        help="Backbone key (sets per-dataset τ table). Use resnet50 for Phase 3 "
             "concept-bottleneck (cb_mlp / cb_linear) runs.",
    )
    ap.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 123, 456],
    )
    ap.add_argument(
        "--out-csv",
        default=None,
        help="Optional CSV path (wide table: one row per backbone × metric block).",
    )
    ap.add_argument(
        "--out-json",
        default=None,
        help="Optional JSON path with nested means/stds/values per backbone/dataset/metric.",
    )
    args = ap.parse_args()

    tau_prefix: str
    if args.tau_prefix:
        tau_prefix = args.tau_prefix
    else:
        tau_prefix = _default_tau_prefix(args.gcbm_prefix)

    print(f"[aggregate Phase 2e] tau_prefix = {tau_prefix}", file=sys.stderr)
    print(f"[aggregate Phase 2e] seeds     = {args.seeds}", file=sys.stderr)
    print(f"[aggregate Phase 2e] runs_root = {args.runs_root}", file=sys.stderr)

    json_out: Dict[str, Any] = {
        "meta": {
            "tau_prefix": tau_prefix,
            "seeds": args.seeds,
            "runs_root": args.runs_root,
            "run_id_pattern": (
                f"{tau_prefix}_tau<τ>[_bb<backbone>_]ds<dataset>_s<seed>"
            ),
        },
        "tau_table": {bb: TAU_BY_BACKBONE[bb] for bb in args.backbones},
        "results": {},
    }

    csv_rows: List[Dict[str, Any]] = []

    for backbone in args.backbones:
        bb_json: Dict[str, Any] = {}
        tau_map = TAU_BY_BACKBONE[backbone]
        label = _BACKBONE_LABEL.get(backbone, backbone)
        row_label = f"G-CBM ({label}) per-dataset τ"

        for metric in METRICS:
            line: Dict[str, str] = {"config": row_label, "metric": metric}
            for ds in DATASETS:
                tau = tau_map[ds]
                vals = collect_seed_values(
                    args.runs_root,
                    tau_prefix,
                    backbone,
                    ds,
                    tau,
                    args.seeds,
                    metric,
                )
                line[ds] = fmt_mean_std(vals)

                entry = bb_json.setdefault(ds, {})
                mblock = entry.setdefault(metric, {})
                mblock["tau"] = tau
                mblock["n"] = len(vals)
                mblock["mean"] = float(np.mean(vals)) if vals else None
                if len(vals) >= 2:
                    mblock["std"] = float(np.std(vals, ddof=1))
                elif len(vals) == 1:
                    mblock["std"] = 0.0
                else:
                    mblock["std"] = None
                mblock["values"] = vals
                mblock["run_ids"] = [
                    build_run_id(tau_prefix, tau, backbone, ds, s)
                    for s in args.seeds
                ]

            # print table row
            col_w = 48
            ds_w = 28
            if metric == METRICS[0]:
                print("", file=sys.stderr)
                print(
                    f"{label} — τ: "
                    + ", ".join(f"{ds}={tau_map[ds]}" for ds in DATASETS),
                    file=sys.stderr,
                )
            header = f"{'Config':<{col_w}}" + "".join(
                f"  {ds:<{ds_w}}" for ds in DATASETS
            )
            if metric == METRICS[0]:
                print(header)
                print("-" * len(header))
            print(
                f"{row_label + ' [' + metric.upper() + ']':<{col_w}}"
                + "".join(f"  {line[ds]:<{ds_w}}" for ds in DATASETS)
            )

            csv_rows.append(line)

        json_out["results"][backbone] = bb_json

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(json_out, f, indent=2)
        print(f"\nJSON written to {args.out_json}")

    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        fieldnames = ["config", "metric"] + DATASETS
        with open(args.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            for row in csv_rows:
                w.writerow(row)
        print(f"CSV written to {args.out_csv}")


if __name__ == "__main__":
    main()
