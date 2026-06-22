"""
plot_clip_labels.py — render per-concept CLIP-agreement charts from
`eval_clip_label_concepts.py` output.

Reads:
  <run_root>/concept_clip_labels.json
    {dataset_key: {vocabulary, num_labels_in_vocab, agreement_threshold,
                   num_high_agreement,
                   concepts: [{concept_id, label, agreement,
                               clip_confidence, num_crops,
                               D_i?, dominant_class?}, ...]}}

Emits:
  <out_dir>/clip_labels_<ds>.png        # per-concept agreement + clip
                                        # confidence (sorted by agreement),
                                        # x-tick labels = predicted CLIP
                                        # phrase (truncated)
  <out_dir>/clip_labels_summary.png     # high-agreement fraction across
                                        # datasets
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
    "xtick.labelsize":  10,
    "ytick.labelsize":  11,
    "legend.fontsize":  11,
    "figure.titlesize": 14,
})

_DS_DISPLAY = {
    "ham10000": "HAM10000",
    "ph2":      "PH2",
    "derm7pt":  "Derm7pt",
    "imagenet": "ImageNet",
}

_MAX_LABEL_LEN = 22


def _trunc(s: str) -> str:
    return s if len(s) <= _MAX_LABEL_LEN else s[:_MAX_LABEL_LEN - 1] + "…"


def _plot_one(ds: str, payload: dict, out_dir: str) -> str:
    concepts = payload.get("concepts", [])
    if not concepts:
        return ""
    threshold = float(payload.get("agreement_threshold", 0.5))

    sorted_c = sorted(concepts, key=lambda c: c["agreement"], reverse=True)
    ids = [str(c["concept_id"]) for c in sorted_c]
    agreement = np.asarray([c["agreement"] for c in sorted_c])
    clip_conf = np.asarray([c.get("clip_confidence", np.nan)
                            for c in sorted_c])
    labels = [_trunc(c.get("label", "?")) for c in sorted_c]

    fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(ids) + 2), 5.5))
    x = np.arange(len(ids))
    bar_w = 0.42

    colors = ["#2980b9" if a >= threshold else "#bdc3c7" for a in agreement]
    ax.bar(x - bar_w / 2, agreement, bar_w, color=colors, edgecolor="black",
           linewidth=0.4, label="Agreement (top-k crops)")
    ax.bar(x + bar_w / 2, clip_conf, bar_w, color="#f39c12", edgecolor="black",
           linewidth=0.4, label="CLIP confidence", alpha=0.9)
    ax.axhline(threshold, color="#c0392b", linestyle="--", linewidth=1.4,
               label=f"Agreement threshold = {threshold}")

    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{i}\n{lab}" for i, lab in zip(ids, labels)],
                       rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Agreement score")
    ax.set_xlabel("Concept / CLIP-predicted label")

    n_high = payload.get("num_high_agreement", "?")
    n_total = len(concepts)
    ax.text(0.99, 0.97,
            f"High-agreement: {n_high}/{n_total}",
            transform=ax.transAxes, fontsize=10, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc",
                      alpha=0.8))

    ds_name = _DS_DISPLAY.get(ds, ds)
    ax.set_title(f"Concept Semantic Labels — {ds_name}", fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"clip_labels_{ds}.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def _plot_summary(all_results: Dict[str, dict], out_dir: str) -> str:
    if not all_results:
        return ""
    datasets = list(all_results.keys())
    n_high = [all_results[ds].get("num_high_agreement", 0) for ds in datasets]
    n_total = [len(all_results[ds].get("concepts", [])) for ds in datasets]
    threshold = float(next(iter(all_results.values()))
                      .get("agreement_threshold", 0.5))
    frac = [h / max(t, 1) for h, t in zip(n_high, n_total)]
    ds_labels = [_DS_DISPLAY.get(ds, ds) for ds in datasets]

    fig, ax = plt.subplots(figsize=(max(7, len(datasets) * 1.8), 5.0))
    x = np.arange(len(datasets))
    ax.bar(x, frac, color="#2980b9", edgecolor="black", linewidth=0.4)
    for i, (h, t) in enumerate(zip(n_high, n_total)):
        ax.text(x[i], frac[i] + 0.02, f"{h}/{t}",
                ha="center", va="bottom", fontsize=11)
    ax.set_ylim(0.0, 1.1)
    ax.set_ylabel(f"Fraction of high-agreement concepts (≥ {threshold})")
    ax.set_title("CLIP Label Agreement per Dataset", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(ds_labels, rotation=20, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "clip_labels_summary.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser(
        "plot_clip_labels — render CLIP-agreement charts")
    ap.add_argument("--run-root", required=True,
                    help="Path to <run_id>/concept_graph_data containing "
                         "concept_clip_labels.json.")
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="Subset of datasets to plot. Default = all keys "
                         "in the JSON.")
    ap.add_argument("--out-dir", default=None,
                    help="Output directory. Default: <run_root>/figures/")
    args = ap.parse_args()

    json_path = os.path.join(args.run_root, "concept_clip_labels.json")
    if not os.path.isfile(json_path):
        print(f"[error] {json_path} not found. Did Phase 10 finish?")
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
