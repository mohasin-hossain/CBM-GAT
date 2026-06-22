"""
aggregate_threshold_sweep_seeds.py — compute mean ± std across multi-seed
threshold sweep runs, producing aggregated CSVs ready for plotting with
plot_threshold_sweep.py --aggregated-root.

For each (dataset, variant) it reads
  <RUNS_ROOT>/<BASE_RUN_ID>_s<SEED>/concept_graph_data/
      <threshold_subdir>/threshold_sweep_<DS>_<VARIANT>.csv
across all seeds and emits a single aggregated CSV per (dataset, variant):
  tau, f1_mean, f1_std, auc_mean, auc_std,
  acc_mean, acc_std, balanced_acc_mean, balanced_acc_std,
  mean_active_concepts_mean, mean_active_concepts_std,
  mean_active_edges_mean, mean_active_edges_std,
  n_seeds

Usage
-----
  # ResNet-50 multi-seed threshold sweep
  python aggregate_threshold_sweep_seeds.py \\
      --base-run-id a_run_gv1_mv1_ps70_sr0.5_gat \\
      --seeds 42 123 456 \\
      --variants v1_threshold \\
      --datasets ham10000 ph2 derm7pt imagenet \\
      --out-dir /netscratch/mhossain/cbm_gat/runs/threshold_sweep_aggregated/resnet50/

  # DenseNet-201
  python aggregate_threshold_sweep_seeds.py \\
      --base-run-id a_run_gv1_mv1_ps70_sr0.5_gat_bbdensenet201 \\
      --seeds 42 123 456 \\
      --variants v1_threshold \\
      --out-dir /netscratch/.../threshold_sweep_aggregated/densenet201/

Outputs (in --out-dir):
  threshold_sweep_<DS>_<VARIANT>.csv   aggregated CSV (one row per tau)
  aggregation_summary.json             metadata (seeds used, missing files)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List

import numpy as np

RUNS_ROOT = "/netscratch/mhossain/cbm_gat/runs"

SCALAR_COLS = [
    "f1", "auc", "acc", "balanced_acc",
    "mean_active_concepts", "mean_active_edges",
]


def _csv_path(base_run_id: str, seed: int, ds: str, variant: str,
              threshold_subdir: str) -> str:
    return os.path.join(
        RUNS_ROOT,
        f"{base_run_id}_s{seed}",
        "concept_graph_data",
        threshold_subdir,
        f"threshold_sweep_{ds}_{variant}.csv",
    )


def _read_seed_csv(path: str) -> Dict[str, List[float]]:
    """Read one seed's threshold-sweep CSV. Returns dict col -> list of values
    (one per tau row, in file order)."""
    rows: Dict[str, List[float]] = defaultdict(list)
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for col in ["tau"] + SCALAR_COLS:
                val = row.get(col, "")
                try:
                    rows[col].append(float(val))
                except (ValueError, TypeError):
                    rows[col].append(float("nan"))
    return dict(rows)


def aggregate(base_run_id: str, seeds: List[int], ds: str, variant: str,
              threshold_subdir: str) -> tuple[list[dict], list[str]]:
    """Read per-seed CSVs for (ds, variant) and return aggregated rows +
    a list of missing-file paths."""
    seed_data: List[Dict[str, List[float]]] = []
    missing: List[str] = []
    for seed in seeds:
        path = _csv_path(base_run_id, seed, ds, variant, threshold_subdir)
        if not os.path.isfile(path):
            missing.append(path)
            print(f"  [missing] {path}", file=sys.stderr)
            continue
        seed_data.append(_read_seed_csv(path))

    if not seed_data:
        return [], missing

    # Verify all seeds have the same tau sequence.
    tau_ref = seed_data[0]["tau"]
    for d in seed_data[1:]:
        if d["tau"] != tau_ref:
            print(f"  [warn] tau mismatch for {ds}/{variant} — "
                  "seeds may have different tau grids; aligning by position.",
                  file=sys.stderr)
            break

    n_tau = len(tau_ref)
    rows = []
    for i in range(n_tau):
        row: dict = {"tau": tau_ref[i], "n_seeds": len(seed_data)}
        for col in SCALAR_COLS:
            vals = [d[col][i] for d in seed_data
                    if i < len(d.get(col, [])) and not np.isnan(d[col][i])]
            if vals:
                row[f"{col}_mean"] = float(np.mean(vals))
                row[f"{col}_std"]  = float(np.std(vals))
            else:
                row[f"{col}_mean"] = float("nan")
                row[f"{col}_std"]  = float("nan")
        rows.append(row)
    return rows, missing


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Aggregate multi-seed threshold sweep CSVs into mean±std.")
    ap.add_argument("--base-run-id", required=True,
                    help="Base run ID (no _s<seed> suffix), e.g. "
                         "a_run_gv1_mv1_ps70_sr0.5_gat_bbdensenet201")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456],
                    help="Seeds to aggregate over (default: 42 123 456)")
    ap.add_argument("--variants", nargs="+",
                    default=["v1_threshold"],
                    choices=["v1_threshold", "v4_threshold"],
                    help="Graph variants to aggregate (default: v1_threshold)")
    ap.add_argument("--datasets", nargs="+",
                    default=["ham10000", "ph2", "derm7pt", "imagenet"],
                    help="Datasets to aggregate")
    ap.add_argument("--threshold-subdir", default="threshold_sweep",
                    help="Subdirectory inside each seed run's concept_graph_data "
                         "that holds the threshold sweep CSVs "
                         "(default: threshold_sweep)")
    ap.add_argument("--out-dir", required=True,
                    help="Directory to write aggregated CSVs and summary JSON.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    fieldnames = (
        ["tau", "n_seeds"]
        + [f"{c}_{s}" for c in SCALAR_COLS for s in ("mean", "std")]
    )

    summary: dict = {
        "base_run_id": args.base_run_id,
        "seeds": args.seeds,
        "variants": args.variants,
        "datasets": args.datasets,
        "threshold_subdir": args.threshold_subdir,
        "missing_files": [],
        "outputs": [],
    }

    for ds in args.datasets:
        for variant in args.variants:
            print(f"\n--- {ds} | {variant} ---")
            rows, missing = aggregate(
                args.base_run_id, args.seeds, ds, variant,
                args.threshold_subdir)
            summary["missing_files"].extend(missing)

            if not rows:
                print(f"  [skip] no data for {ds}/{variant}")
                continue

            out_path = os.path.join(
                args.out_dir,
                f"threshold_sweep_{ds}_{variant}.csv")
            with open(out_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames,
                                        extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)
            print(f"  -> {out_path}  ({len(rows)} tau rows, "
                  f"{rows[0]['n_seeds']} seeds)")
            summary["outputs"].append(out_path)

    summary_path = os.path.join(args.out_dir, "aggregation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary: {summary_path}")
    if summary["missing_files"]:
        print(f"[warn] {len(summary['missing_files'])} missing file(s) — "
              "check summary JSON for details.", file=sys.stderr)


if __name__ == "__main__":
    main()
