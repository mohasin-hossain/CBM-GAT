"""
plot_concept_di.py — render per-concept D_i discriminativeness charts
from `eval_concept_di.py` output.

Reads:
  <run_root>/concept_di_scores.json
    {dataset_key: {best_k, num_images, q, theta, concepts: [{concept_id,
                   D_i, dominant_class}, ...], avg_D_i, num_discriminative}}

Emits:
  <out_dir>/concept_di_<ds>.png        # per-concept bar (sorted by D_i)
  <out_dir>/concept_di_summary.png     # avg D_i + #discriminative across
                                       # datasets, single grouped chart
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams.update({
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.labelsize":   12,
    "xtick.labelsize":  11,
    "ytick.labelsize":  11,
    "legend.fontsize":  11,
    "figure.titlesize": 14,
})

_DS_DISPLAY = {
    "ham10000": "HAM10000",
    "ph2":      "PH2",
    "derm7pt":  "Derm7pt",
    "imagenet": "ImageNet",
    "cub":      "CUB-200",
}


def _plot_one(ds: str, payload: dict, out_dir: str) -> str:
    concepts = payload.get("concepts", [])
    if not concepts:
        return ""
    theta = float(payload.get("theta", 0.6))

    sorted_c = sorted(concepts, key=lambda c: c["D_i"], reverse=True)
    ids = [str(c["concept_id"]) for c in sorted_c]
    di = np.asarray([c["D_i"] for c in sorted_c])
    colors = ["#27ae60" if d >= theta else "#bdc3c7" for d in di]

    fig, ax = plt.subplots(figsize=(max(7, 0.5 * len(ids) + 2), 5.0))
    ax.bar(ids, di, color=colors, edgecolor="black", linewidth=0.4)
    ax.axhline(theta, color="#c0392b", linestyle="--", linewidth=1.4,
               label=f"Discriminative threshold θ = {theta}")
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Concept (sorted by discriminativeness)")
    ax.set_ylabel(r"Discriminativeness Score $D_i$")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best", frameon=True)

    avg = payload.get("avg_D_i", float("nan"))
    n_disc = payload.get("num_discriminative", "?")
    n_total = payload.get("best_k", len(concepts))
    ax.text(0.99, 0.97,
            f"avg $D_i$ = {avg:.3f}   discriminative = {n_disc}/{n_total}",
            transform=ax.transAxes, fontsize=10, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc",
                      alpha=0.8))

    ds_name = _DS_DISPLAY.get(ds, ds)
    ax.set_title(f"Concept Discriminativeness — {ds_name}", fontweight="bold")
    fig.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"concept_di_{ds}.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def _plot_summary(all_results: Dict[str, dict], out_dir: str) -> str:
    if not all_results:
        return ""
    datasets = list(all_results.keys())
    avg_di = [all_results[ds].get("avg_D_i", np.nan) for ds in datasets]
    n_disc = [all_results[ds].get("num_discriminative", 0) for ds in datasets]
    n_total = [all_results[ds].get("best_k", 0) for ds in datasets]
    theta = float(next(iter(all_results.values())).get("theta", 0.6))
    ds_labels = [_DS_DISPLAY.get(ds, ds) for ds in datasets]

    fig, axes = plt.subplots(1, 2, figsize=(max(9, len(datasets) * 1.8), 5.0))
    ax_avg, ax_disc = axes
    x = np.arange(len(datasets))

    ax_avg.bar(x, avg_di, color="#2980b9", edgecolor="black", linewidth=0.4)
    ax_avg.axhline(theta, color="#c0392b", linestyle="--", linewidth=1.4,
                   label=f"Discriminative threshold θ = {theta}")
    ax_avg.set_ylim(0.0, 1.05)
    ax_avg.set_ylabel(r"Mean Discriminativeness $D_i$")
    ax_avg.set_title(r"Mean $D_i$ per Dataset")
    ax_avg.set_xticks(x)
    ax_avg.set_xticklabels(ds_labels, rotation=20, ha="right")
    ax_avg.grid(True, axis="y", alpha=0.3)
    ax_avg.legend(loc="best", frameon=True)

    frac = [d / max(t, 1) for d, t in zip(n_disc, n_total)]
    ax_disc.bar(x, frac, color="#27ae60", edgecolor="black", linewidth=0.4)
    for i, (d, t) in enumerate(zip(n_disc, n_total)):
        ax_disc.text(x[i], frac[i] + 0.02, f"{d}/{t}",
                     ha="center", va="bottom", fontsize=11)
    ax_disc.set_ylim(0.0, 1.1)
    ax_disc.set_ylabel(f"Discriminative fraction ($D_i$ ≥ {theta})")
    ax_disc.set_title("Discriminative Concepts per Dataset")
    ax_disc.set_xticks(x)
    ax_disc.set_xticklabels(ds_labels, rotation=20, ha="right")
    ax_disc.grid(True, axis="y", alpha=0.3)

    fig.suptitle(r"Concept Discriminativeness ($D_i$) Summary",
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "concept_di_summary.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser(
        "plot_concept_di — render per-concept D_i charts")
    ap.add_argument("--run-root", required=True,
                    help="Path to <run_id>/concept_graph_data containing "
                         "concept_di_scores.json.")
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="Subset of datasets to plot. Default = all keys "
                         "found in the JSON.")
    ap.add_argument("--out-dir", default=None,
                    help="Output directory. Default: <run_root>/figures/")
    args = ap.parse_args()

    json_path = os.path.join(args.run_root, "concept_di_scores.json")
    if not os.path.isfile(json_path):
        print(f"[error] {json_path} not found. Did Phase 9 finish?")
        return

    with open(json_path) as f:
        all_results = json.load(f)

    if args.datasets:
        all_results = {k: v for k, v in all_results.items()
                       if k in args.datasets}
    if not all_results:
        print("[error] no datasets to plot.")
        return

    out_dir = args.out_dir or os.path.join(args.run_root, "figures")
    written: List[str] = []
    for ds, payload in all_results.items():
        path = _plot_one(ds, payload, out_dir)
        if path:
            written.append(path)
            print(f"  wrote {path}")

    summary = _plot_summary(all_results, out_dir)
    if summary:
        written.append(summary)
        print(f"  wrote {summary}")

    print(f"\nDone. {len(written)} figure(s) under {out_dir}")


if __name__ == "__main__":
    main()
