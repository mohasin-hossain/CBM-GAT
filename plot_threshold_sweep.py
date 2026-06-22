"""
plot_threshold_sweep.py — render the headline F1/AUC vs τ figures from the
CSVs produced by `eval_threshold_sweep_v2.py`.

For each dataset it reads:
  <run_root>/<threshold-subdir>/threshold_sweep_<ds>_v1_threshold.csv
  <run_root>/<threshold-subdir>/threshold_sweep_<ds>_v4_threshold.csv

``--threshold-subdir`` defaults to ``threshold_sweep`` (Phase 8 default
split is **val**, so CSVs live there). Use ``threshold_sweep_test`` when
Phase 8 was run with ``--split test``.

and emits:
  <out_dir>/threshold_sweep_<ds>.png       # 2-panel: F1-vs-τ + AUC-vs-τ
  <out_dir>/threshold_sweep_summary.png    # 4-dataset grid (1 row per
                                           # metric, 1 column per dataset)
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
from typing import Dict, List

import matplotlib as mpl
import matplotlib.ticker
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
}

_VARIANT_STYLES = {
    "v1_threshold": {"color": "#c0392b", "label": "GCBM"},
    "v4_threshold": {"color": "#2c3e50", "label": "Concept-only baseline"},
}

# Per-dataset colours used when overlaying multiple datasets on one figure.
_DS_COLORS = {
    "ham10000": "#2980b9",
    "ph2":      "#c0392b",
    "derm7pt":  "#27ae60",
    "imagenet": "#8e44ad",
}


def _savefig(fig: plt.Figure, out_path: str, **kwargs) -> None:
    """Save figure as PNG (dpi=300) and SVG to the same directory."""
    fig.savefig(out_path, dpi=300, **kwargs)
    svg_path = out_path.replace(".png", ".svg")
    fig.savefig(svg_path, format="svg", **kwargs)
    print(f"  [svg] {svg_path}")


def _read_csv(path: str) -> Dict[str, np.ndarray]:
    rows = {"tau": [], "f1": [], "auc": [], "acc": [], "balanced_acc": [],
            "mean_active_concepts": [], "mean_active_edges": []}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k in rows:
                rows[k].append(float(row[k]) if row.get(k) not in (None, "")
                               else float("nan"))
    return {k: np.asarray(v) for k, v in rows.items()}


def _load_dataset(run_root: str, ds: str, variants: List[str],
                  threshold_subdir: str = "threshold_sweep",
                  ) -> Dict[str, Dict[str, np.ndarray]]:
    base = os.path.join(run_root, threshold_subdir)
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for variant in variants:
        path = os.path.join(base, f"threshold_sweep_{ds}_{variant}.csv")
        if os.path.isfile(path):
            out[variant] = _read_csv(path)
    return out


def _autodiscover_datasets(run_root: str, variants: List[str],
                           threshold_subdir: str = "threshold_sweep",
                           ) -> List[str]:
    found = set()
    base = os.path.join(run_root, threshold_subdir)
    for variant in variants:
        for path in glob.glob(
                os.path.join(base, f"threshold_sweep_*_{variant}.csv")):
            stem = os.path.basename(path).rsplit(".csv", 1)[0]
            ds = stem[len("threshold_sweep_"):-(len(variant) + 1)]
            found.add(ds)
    return sorted(found)


def _plot_one(ds: str, per_variant: Dict[str, Dict[str, np.ndarray]],
              out_dir: str) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax_f1, ax_auc = axes

    # Collect mean_active_concepts across variants to determine right-axis range.
    # Use the first variant that has the data; all variants share the same K and
    # thresholding logic so their concept-count curves are effectively identical.
    concept_tau: np.ndarray | None = None
    concept_counts: np.ndarray | None = None
    for data in per_variant.values():
        mac = data.get("mean_active_concepts")
        if mac is not None and not np.all(np.isnan(mac)):
            concept_tau = data["tau"]
            concept_counts = mac
            break

    for variant, data in per_variant.items():
        style = _VARIANT_STYLES.get(
            variant, {"color": "#888", "label": variant})
        ax_f1.plot(data["tau"], data["f1"], "o-",
                   color=style["color"], label=style["label"], linewidth=2.0)
        ax_auc.plot(data["tau"], data["auc"], "o-",
                    color=style["color"], label=style["label"], linewidth=2.0)

    # Panel labels merged into titles — sits above the plot area, no data overlap
    ax_f1.set_title("(a)  Weighted F1", loc="left", fontweight="bold")
    ax_auc.set_title("(b)  ROC-AUC", loc="left", fontweight="bold")
    for ax, ylab in zip(axes, ("Weighted F1", "AUC (macro)")):
        ax.set_xlabel("Concept activation threshold τ")
        ax.set_ylabel(ylab)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.3)

    # Secondary y-axis: mean active concept nodes (actual count, not normalised)
    if concept_counts is not None:
        k_max = float(np.nanmax(concept_counts))
        for ax in (ax_f1, ax_auc):
            ax2 = ax.twinx()
            ax2.plot(concept_tau, concept_counts, "s--",
                     color="#95a5a6", linewidth=1.6, markersize=5,
                     label="Active concept nodes", zorder=1)
            ax2.set_ylim(0, max(k_max * 1.15, 1))
            ax2.set_ylabel("Active concept nodes (mean)", color="#636e72")
            ax2.tick_params(axis="y", labelcolor="#636e72")
            ax2.yaxis.set_major_locator(
                mpl.ticker.MaxNLocator(integer=True, nbins=5))
            # Keep secondary grid off to avoid visual noise
            ax2.grid(False)
            # Collect handles from both axes for a unified legend
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, loc="lower left", frameon=True)
    else:
        for ax in axes:
            ax.legend(loc="best", frameon=True)

    ds_name = _DS_DISPLAY.get(ds, ds)
    fig.suptitle(f"Concept Sparsity vs. Classification Performance — {ds_name}",
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"threshold_sweep_{ds}.png")
    _savefig(fig, out_path)
    plt.close(fig)
    return out_path


def _plot_grid(loaded: Dict[str, Dict[str, Dict[str, np.ndarray]]],
               datasets: List[str], variants: List[str], out_dir: str) -> str:
    if not datasets:
        return ""
    n_cols = len(datasets)
    fig, axes = plt.subplots(2, n_cols, figsize=(max(9, n_cols * 3.2), 7),
                             sharex=True)
    if n_cols == 1:
        axes = np.asarray([[axes[0]], [axes[1]]])

    for j, ds in enumerate(datasets):
        per_variant = loaded.get(ds, {})
        for variant, data in per_variant.items():
            style = _VARIANT_STYLES.get(
                variant, {"color": "#888", "label": variant})
            axes[0, j].plot(data["tau"], data["f1"], "o-",
                            color=style["color"], label=style["label"],
                            linewidth=1.8)
            axes[1, j].plot(data["tau"], data["auc"], "o-",
                            color=style["color"], label=style["label"],
                            linewidth=1.8)

        axes[0, j].set_title(_DS_DISPLAY.get(ds, ds))
        for i in (0, 1):
            axes[i, j].set_xlim(0.0, 1.0)
            axes[i, j].set_ylim(0.0, 1.0)
            axes[i, j].grid(True, alpha=0.3)
        axes[1, j].set_xlabel("τ")

    axes[0, 0].set_ylabel("Weighted F1")
    axes[1, 0].set_ylabel("AUC (macro)")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center",
                   ncol=len(handles), fontsize=11,
                   bbox_to_anchor=(0.5, 0.98))

    fig.suptitle("Concept Activation Threshold Sensitivity",
                 fontweight="bold", y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "threshold_sweep_summary.png")
    _savefig(fig, out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_two_datasets(loaded: Dict[str, Dict[str, Dict[str, np.ndarray]]],
                       ds_pair: List[str], variants: List[str],
                       out_dir: str) -> str:
    """2-panel figure (F1 | AUC) overlaying two datasets on the same axes.
    Each dataset gets its own colour; a shared grey secondary y-axis shows the
    mean number of active concept nodes (dashed, per-dataset colour)."""
    available = [ds for ds in ds_pair if loaded.get(ds)]
    if not available:
        print(f"[two-dataset] no data for {ds_pair}")
        return ""

    # Pick the first variant with data for at least one of the two datasets.
    primary_variant = None
    for v in variants:
        if any(loaded.get(ds, {}).get(v) for ds in available):
            primary_variant = v
            break
    if primary_variant is None:
        print("[two-dataset] no variant with data found")
        return ""

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax_f1, ax_auc = axes

    # Collect all concept-count arrays to set a shared right-axis range.
    all_counts: list[np.ndarray] = []

    for ds in available:
        data = loaded.get(ds, {}).get(primary_variant)
        if data is None:
            print(f"[two-dataset] no data for {ds} / {primary_variant}")
            continue
        color = _DS_COLORS.get(ds, "#888")
        label = _DS_DISPLAY.get(ds, ds)
        ax_f1.plot(data["tau"], data["f1"], "o-",
                   color=color, label=label, linewidth=2.0)
        ax_auc.plot(data["tau"], data["auc"], "o-",
                    color=color, label=label, linewidth=2.0)
        mac = data.get("mean_active_concepts")
        if mac is not None and not np.all(np.isnan(mac)):
            all_counts.append(mac)

    # Secondary y-axis: mean active concept nodes per dataset (same colour, dashed)
    if all_counts:
        k_max = float(max(np.nanmax(c) for c in all_counts))
        for ax in (ax_f1, ax_auc):
            ax2 = ax.twinx()
            ax2.set_ylim(0, max(k_max * 1.15, 1))
            ax2.set_ylabel("Active concept nodes (mean)", color="#636e72")
            ax2.tick_params(axis="y", labelcolor="#636e72")
            ax2.yaxis.set_major_locator(
                mpl.ticker.MaxNLocator(integer=True, nbins=5))
            ax2.grid(False)
            # One dashed concept-count line per dataset
            for ds in available:
                data = loaded.get(ds, {}).get(primary_variant)
                if data is None:
                    continue
                mac = data.get("mean_active_concepts")
                if mac is None or np.all(np.isnan(mac)):
                    continue
                color = _DS_COLORS.get(ds, "#888")
                label = f"{_DS_DISPLAY.get(ds, ds)} — concepts"
                ax2.plot(data["tau"], mac, "s--", color=color,
                         linewidth=1.5, markersize=4, alpha=0.6, label=label)
            # Unified legend (performance lines + concept lines)
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, loc="lower left",
                      frameon=True, fontsize=9)
    else:
        for ax in axes:
            ax.legend(loc="best", frameon=True)

    ax_f1.set_title("(a)  Weighted F1", loc="left", fontweight="bold")
    ax_auc.set_title("(b)  ROC-AUC",    loc="left", fontweight="bold")
    for ax, ylab in zip(axes, ("Weighted F1", "AUC (macro)")):
        ax.set_xlabel("Concept activation threshold τ")
        ax.set_ylabel(ylab)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.3)

    name_a = _DS_DISPLAY.get(available[0], available[0])
    name_b = _DS_DISPLAY.get(available[1], available[1]) if len(available) > 1 else ""
    title_suffix = f"{name_a} vs. {name_b}" if name_b else name_a
    fig.suptitle(f"Concept Activation Threshold Sensitivity — {title_suffix}",
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    os.makedirs(out_dir, exist_ok=True)
    tag = "_".join(available)
    out_path = os.path.join(out_dir, f"threshold_sweep_{tag}_comparison.png")
    _savefig(fig, out_path)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Aggregated (multi-seed mean ± std) CSV helpers and plot functions
# ---------------------------------------------------------------------------

def _read_aggregated_csv(path: str) -> Dict[str, np.ndarray]:
    """Read an aggregated CSV produced by aggregate_threshold_sweep_seeds.py.
    Returns dict with keys like tau, f1_mean, f1_std, auc_mean, auc_std,
    mean_active_concepts_mean, mean_active_concepts_std, etc."""
    rows: Dict[str, List] = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                rows.setdefault(k, [])
                try:
                    rows[k].append(float(v))
                except (ValueError, TypeError):
                    rows[k].append(float("nan"))
    return {k: np.asarray(v) for k, v in rows.items()}


def _load_aggregated_dataset(agg_root: str, ds: str,
                              variants: List[str],
                              ) -> Dict[str, Dict[str, np.ndarray]]:
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for variant in variants:
        path = os.path.join(
            agg_root, f"threshold_sweep_{ds}_{variant}.csv")
        if os.path.isfile(path):
            out[variant] = _read_aggregated_csv(path)
    return out


def _autodiscover_aggregated_datasets(agg_root: str,
                                       variants: List[str]) -> List[str]:
    found = set()
    for variant in variants:
        for path in glob.glob(
                os.path.join(agg_root,
                             f"threshold_sweep_*_{variant}.csv")):
            stem = os.path.basename(path).rsplit(".csv", 1)[0]
            ds = stem[len("threshold_sweep_"):-(len(variant) + 1)]
            found.add(ds)
    return sorted(found)


def _plot_one_aggregated(ds: str,
                          per_variant: Dict[str, Dict[str, np.ndarray]],
                          out_dir: str) -> str:
    """Like _plot_one but draws mean lines with ±1 std shaded bands.
    Reads *_mean / *_std columns from aggregated CSVs."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax_f1, ax_auc = axes

    concept_tau: np.ndarray | None = None
    concept_mean: np.ndarray | None = None
    concept_std: np.ndarray | None = None

    for variant, data in per_variant.items():
        style = _VARIANT_STYLES.get(
            variant, {"color": "#888", "label": variant})
        color = style["color"]
        label = style["label"]
        tau = data["tau"]

        for ax, metric in zip((ax_f1, ax_auc), ("f1", "auc")):
            mean = data.get(f"{metric}_mean")
            std  = data.get(f"{metric}_std")
            if mean is None:
                continue
            ax.plot(tau, mean, "o-", color=color, label=label, linewidth=2.0)
            if std is not None and not np.all(np.isnan(std)):
                ax.fill_between(tau,
                                np.clip(mean - std, 0, 1),
                                np.clip(mean + std, 0, 1),
                                color=color, alpha=0.15)

        mac_mean = data.get("mean_active_concepts_mean")
        mac_std  = data.get("mean_active_concepts_std")
        if mac_mean is not None and not np.all(np.isnan(mac_mean)):
            concept_tau   = tau
            concept_mean  = mac_mean
            concept_std   = mac_std

    ax_f1.set_title("(a)  Weighted F1", loc="left", fontweight="bold")
    ax_auc.set_title("(b)  ROC-AUC",    loc="left", fontweight="bold")
    for ax, ylab in zip(axes, ("Weighted F1", "AUC (macro)")):
        ax.set_xlabel("Concept activation threshold τ")
        ax.set_ylabel(ylab)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.3)

    if concept_mean is not None:
        k_max = float(np.nanmax(concept_mean))
        for ax in (ax_f1, ax_auc):
            ax2 = ax.twinx()
            ax2.plot(concept_tau, concept_mean, "s--",
                     color="#95a5a6", linewidth=1.6, markersize=5,
                     label="Active concept nodes (mean)", zorder=1)
            if concept_std is not None and not np.all(np.isnan(concept_std)):
                ax2.fill_between(
                    concept_tau,
                    np.clip(concept_mean - concept_std, 0, None),
                    concept_mean + concept_std,
                    color="#95a5a6", alpha=0.15, zorder=0)
            ax2.set_ylim(0, max(k_max * 1.15, 1))
            ax2.set_ylabel("Active concept nodes (mean)", color="#636e72")
            ax2.tick_params(axis="y", labelcolor="#636e72")
            ax2.yaxis.set_major_locator(
                mpl.ticker.MaxNLocator(integer=True, nbins=5))
            ax2.grid(False)
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, loc="lower left", frameon=True)
    else:
        for ax in axes:
            ax.legend(loc="best", frameon=True)

    ds_name = _DS_DISPLAY.get(ds, ds)
    fig.suptitle(
        f"Concept Sparsity vs. Classification Performance — {ds_name}"
        "  (mean ± 1σ, multi-seed)",
        fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"threshold_sweep_{ds}.png")
    _savefig(fig, out_path)
    plt.close(fig)
    return out_path


def _plot_two_datasets_aggregated(
        loaded: Dict[str, Dict[str, Dict[str, np.ndarray]]],
        ds_pair: List[str],
        variants: List[str],
        out_dir: str) -> str:
    """Like _plot_two_datasets but with ±1 std shaded bands from aggregated CSVs."""
    available = [ds for ds in ds_pair if loaded.get(ds)]
    if not available:
        return ""

    primary_variant = None
    for v in variants:
        if any(loaded.get(ds, {}).get(v) for ds in available):
            primary_variant = v
            break
    if primary_variant is None:
        return ""

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax_f1, ax_auc = axes

    all_mac_means: list[np.ndarray] = []

    for ds in available:
        data = loaded.get(ds, {}).get(primary_variant)
        if data is None:
            continue
        color = _DS_COLORS.get(ds, "#888")
        label = _DS_DISPLAY.get(ds, ds)
        tau = data["tau"]

        for ax, metric in zip((ax_f1, ax_auc), ("f1", "auc")):
            mean = data.get(f"{metric}_mean")
            std  = data.get(f"{metric}_std")
            if mean is None:
                continue
            ax.plot(tau, mean, "o-", color=color, label=label, linewidth=2.0)
            if std is not None and not np.all(np.isnan(std)):
                ax.fill_between(tau,
                                np.clip(mean - std, 0, 1),
                                np.clip(mean + std, 0, 1),
                                color=color, alpha=0.15)

        mac_mean = data.get("mean_active_concepts_mean")
        if mac_mean is not None and not np.all(np.isnan(mac_mean)):
            all_mac_means.append(mac_mean)

    if all_mac_means:
        k_max = float(max(np.nanmax(c) for c in all_mac_means))
        for ax in (ax_f1, ax_auc):
            ax2 = ax.twinx()
            ax2.set_ylim(0, max(k_max * 1.15, 1))
            ax2.set_ylabel("Active concept nodes (mean)", color="#636e72")
            ax2.tick_params(axis="y", labelcolor="#636e72")
            ax2.yaxis.set_major_locator(
                mpl.ticker.MaxNLocator(integer=True, nbins=5))
            ax2.grid(False)
            for ds in available:
                data = loaded.get(ds, {}).get(primary_variant)
                if data is None:
                    continue
                mac_mean = data.get("mean_active_concepts_mean")
                mac_std  = data.get("mean_active_concepts_std")
                if mac_mean is None or np.all(np.isnan(mac_mean)):
                    continue
                color = _DS_COLORS.get(ds, "#888")
                lbl = f"{_DS_DISPLAY.get(ds, ds)} — concepts"
                ax2.plot(data["tau"], mac_mean, "s--", color=color,
                         linewidth=1.5, markersize=4, alpha=0.6, label=lbl)
                if mac_std is not None and not np.all(np.isnan(mac_std)):
                    ax2.fill_between(
                        data["tau"],
                        np.clip(mac_mean - mac_std, 0, None),
                        mac_mean + mac_std,
                        color=color, alpha=0.10)
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, loc="lower left",
                      frameon=True, fontsize=9)
    else:
        for ax in axes:
            ax.legend(loc="best", frameon=True)

    ax_f1.set_title("(a)  Weighted F1", loc="left", fontweight="bold")
    ax_auc.set_title("(b)  ROC-AUC",    loc="left", fontweight="bold")
    for ax, ylab in zip(axes, ("Weighted F1", "AUC (macro)")):
        ax.set_xlabel("Concept activation threshold τ")
        ax.set_ylabel(ylab)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.3)

    name_a = _DS_DISPLAY.get(available[0], available[0])
    name_b = (_DS_DISPLAY.get(available[1], available[1])
              if len(available) > 1 else "")
    title_suffix = f"{name_a} vs. {name_b}" if name_b else name_a
    fig.suptitle(
        f"Concept Activation Threshold Sensitivity — {title_suffix}"
        "  (mean ± 1σ, multi-seed)",
        fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    os.makedirs(out_dir, exist_ok=True)
    tag = "_".join(available)
    out_path = os.path.join(out_dir, f"threshold_sweep_{tag}_comparison.png")
    _savefig(fig, out_path)
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser(
        "plot_threshold_sweep — render F1/AUC vs τ figures")
    ap.add_argument("--run-root", default=None,
                    help="Path to <run_id>/concept_graph_data. "
                         "Required unless --aggregated-root is set.")
    ap.add_argument("--aggregated-root", default=None,
                    help="Directory containing aggregated CSVs produced by "
                         "aggregate_threshold_sweep_seeds.py. When set, "
                         "plots mean lines with ±1 std shaded bands instead "
                         "of raw single-seed curves. --run-root is not "
                         "required when this flag is used.")
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="Datasets to plot. Default: auto-discover.")
    ap.add_argument("--variants", nargs="+",
                    default=["v1_threshold", "v4_threshold"],
                    help="Graph variants to overlay on each panel.")
    ap.add_argument("--out-dir", default=None,
                    help="Output directory. Default: <run_root>/figures/ or "
                         "<aggregated_root>/figures/ depending on mode.")
    ap.add_argument(
        "--threshold-subdir", default="threshold_sweep",
        help="Subdirectory of run_root containing Phase-8 CSVs. Default "
             "threshold_sweep (matches default Phase-8 --split val). Use "
             "threshold_sweep_test after eval with --split test. "
             "Ignored when --aggregated-root is set.")
    args = ap.parse_args()

    # ------------------------------------------------------------------ mode
    if args.aggregated_root:
        # Aggregated multi-seed mode — mean ± std shaded bands.
        agg_root = args.aggregated_root
        out_dir = args.out_dir or os.path.join(agg_root, "figures")

        if args.datasets:
            datasets = args.datasets
        else:
            datasets = _autodiscover_aggregated_datasets(
                agg_root, args.variants)
            if not datasets:
                print("[error] no aggregated CSVs found under "
                      f"{agg_root}/. Did aggregation finish?")
                return
            print(f"[auto-discover] datasets = {datasets}")

        loaded: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
        for ds in datasets:
            loaded[ds] = _load_aggregated_dataset(
                agg_root, ds, args.variants)

        written = []
        for ds in datasets:
            if not loaded.get(ds):
                print(f"[skip] {ds} — no aggregated CSV on disk")
                continue
            out_path = _plot_one_aggregated(ds, loaded[ds], out_dir)
            written.append(out_path)
            print(f"  wrote {out_path}")

        two_ds_path = _plot_two_datasets_aggregated(
            loaded, ["ph2", "ham10000"], args.variants, out_dir)
        if two_ds_path:
            written.append(two_ds_path)
            print(f"  wrote {two_ds_path}")

        print(f"\nDone. {len(written)} figure(s) under {out_dir}")
        return

    # ------------------------------------------------------------------ raw
    if not args.run_root:
        print("[error] provide --run-root (raw mode) or "
              "--aggregated-root (multi-seed aggregated mode).")
        return

    out_dir = args.out_dir or os.path.join(args.run_root, "figures")
    ts = args.threshold_subdir

    if args.datasets:
        datasets = args.datasets
    else:
        datasets = _autodiscover_datasets(args.run_root, args.variants, ts)
        if not datasets:
            print("[error] no threshold-sweep CSVs found under "
                  f"{os.path.join(args.run_root, ts)}/. Did Phase 8 finish?")
            return
        print(f"[auto-discover] datasets = {datasets}")

    loaded = {}
    for ds in datasets:
        loaded[ds] = _load_dataset(args.run_root, ds, args.variants, ts)

    written = []
    for ds in datasets:
        if not loaded.get(ds):
            print(f"[skip] {ds} — no CSV on disk")
            continue
        out_path = _plot_one(ds, loaded[ds], out_dir)
        written.append(out_path)
        print(f"  wrote {out_path}")

    grid_path = _plot_grid(loaded, datasets, args.variants, out_dir)
    if grid_path:
        written.append(grid_path)
        print(f"  wrote {grid_path}")

    two_ds_path = _plot_two_datasets(
        loaded, ["ph2", "ham10000"], args.variants, out_dir)
    if two_ds_path:
        written.append(two_ds_path)
        print(f"  wrote {two_ds_path}")

    print(f"\nDone. {len(written)} figure(s) under {out_dir}")


if __name__ == "__main__":
    main()
