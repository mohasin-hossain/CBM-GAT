"""
plot_fidelity.py — render Figure-3 style fidelity curves from the JSON/CSV
produced by `eval_fidelity_v2.py`.

For each (dataset, graph_variant) it reads:
  <run_root>/fidelity/<variant>/<ds>_topk_grad.json
  <run_root>/fidelity/<variant>/<ds>_random.json

and emits (each figure saved as .png and .svg):
  <out_dir>/fidelity_<ds>_<variant>.png        # 2-panel: deletion + insertion
  <out_dir>/fidelity_ph2_ham10000_<variant>.png  # PH2 vs HAM10000, 4 curves/panel
  <out_dir>/fidelity_auc_summary.png          # bar chart, AUC del/ins per
                                              # (variant, strategy, dataset)

Faithfulness reads as:
  - deletion: solid lines, lower curve (and lower AUC) = better.
  - insertion: dashed lines, higher curve (and higher AUC) = better.
  - MRF should beat random on both metrics for a faithful model.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List, Tuple

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

_STRATEGY_STYLES = {
    "topk_grad": {"color": "#c0392b", "label": "Most-relevant first (MRF)"},
    "random":    {"color": "#7f8c8d", "label": "Random order"},
}

# Distinct markers per strategy so curves remain distinguishable in grayscale print.
_STRATEGY_MARKERS = {
    "topk_grad": "o",
    "random":    "s",
}

# Combined PH2 + HAM10000: strategy → colour (MRF green, random red);
# dataset → marker (PH2 circles, HAM squares); MRF solid, random dashed.
_PH2_HAM_DATASET_MARKERS = {"ph2": "o", "ham10000": "s"}
_PH2_HAM_STRATEGY_COLOR = {
    "topk_grad": "#2e7d32",  # MRF — green
    "random":    "#c62828",  # Random — red
}

_BAND_ALPHA = 0.22


def _plot_mean_std_band(
    ax: plt.Axes,
    fr: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray | None,
    color: str,
    linestyle: str,
    marker: str,
    label: str,
    *,
    markersize: float = 6.0,
) -> None:
    """Mean line with optional ±1 std shading (across test images)."""
    if y_std is not None and np.any(np.isfinite(y_std)):
        lo = np.clip(y_mean - y_std, 0.0, 1.0)
        hi = np.clip(y_mean + y_std, 0.0, 1.0)
        ax.fill_between(fr, lo, hi, color=color, alpha=_BAND_ALPHA,
                          linewidth=0.0, antialiased=True)
    n = len(fr)
    markevery = 1 if n <= 16 else max(1, n // 10)
    ax.plot(fr, y_mean, linestyle=linestyle, color=color, label=label,
            linewidth=2.0, marker=marker, markersize=markersize,
            markevery=markevery, markeredgecolor="white",
            markeredgewidth=0.6)


def _savefig(fig: plt.Figure, out_path: str, **kwargs) -> None:
    """Save figure as PNG (dpi=300) and SVG to the same directory."""
    fig.savefig(out_path, dpi=300, **kwargs)
    svg_path = out_path.replace(".png", ".svg")
    fig.savefig(svg_path, format="svg", **kwargs)
    print(f"  [svg] {svg_path}")


def _load_jsons(run_root: str, variant: str, datasets: List[str]
                ) -> Dict[str, Dict[str, dict]]:
    """Return {dataset: {strategy: payload}} for the requested variant."""
    base = os.path.join(run_root, "fidelity", variant)
    out: Dict[str, Dict[str, dict]] = {}
    if not os.path.isdir(base):
        return out
    for ds in datasets:
        per_strategy: Dict[str, dict] = {}
        for strat in ("topk_grad", "random"):
            path = os.path.join(base, f"{ds}_{strat}.json")
            if os.path.isfile(path):
                with open(path) as f:
                    per_strategy[strat] = json.load(f)
        if per_strategy:
            out[ds] = per_strategy
    return out


def _autodiscover_datasets(run_root: str, variants: List[str]) -> List[str]:
    """Sniff which datasets actually have JSONs on disk."""
    found = set()
    for v in variants:
        base = os.path.join(run_root, "fidelity", v)
        for path in glob.glob(os.path.join(base, "*_topk_grad.json")):
            ds = os.path.basename(path).rsplit("_topk_grad.json", 1)[0]
            found.add(ds)
        for path in glob.glob(os.path.join(base, "*_random.json")):
            ds = os.path.basename(path).rsplit("_random.json", 1)[0]
            found.add(ds)
    return sorted(found)


def _plot_one(ds: str, variant: str, payloads: Dict[str, dict],
              out_dir: str) -> str:
    """Render the deletion + insertion two-panel figure for one dataset."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    ax_del, ax_ins = axes

    for strat in ("topk_grad", "random"):
        if strat not in payloads:
            continue
        p = payloads[strat]
        fr = np.asarray(p["fracs"], dtype=np.float64)
        del_y = np.asarray(p["mean_deletion_probs"], dtype=np.float64)
        ins_y = np.asarray(p["mean_insertion_probs"], dtype=np.float64)
        style = _STRATEGY_STYLES[strat]
        mk = _STRATEGY_MARKERS[strat]

        _plot_mean_std_band(
            ax_del, fr, del_y, None, style["color"], "-", mk,
            f"{style['label']}  (AUC={p['mean_AUC_del']:.3f})",
        )
        _plot_mean_std_band(
            ax_ins, fr, ins_y, None, style["color"], "--", mk,
            f"{style['label']}  (AUC={p['mean_AUC_ins']:.3f})",
        )

    # Panel labels merged into titles — sits above the plot area, no data overlap
    ax_del.set_title("(a)  Deletion ↓", loc="left", fontweight="bold")
    ax_ins.set_title("(b)  Insertion ↑", loc="left", fontweight="bold")
    for ax, xlabel in zip(axes,
                          ("Fraction of concept nodes removed",
                           "Fraction of concept nodes inserted")):
        ax.set_xlabel(xlabel)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", frameon=True)
    ax_del.set_ylabel("P(correct class)")

    ds_name = _DS_DISPLAY.get(ds, ds)
    fig.suptitle(f"Concept Node Faithfulness — {ds_name}", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"fidelity_{ds}_{variant}.png")
    _savefig(fig, out_path)
    plt.close(fig)
    return out_path


def _plot_ph2_ham_combined(
    per_variant: Dict[str, Dict[str, dict]],
    variant: str,
    out_dir: str,
) -> str:
    """
    One row, two columns: (a) Deletion, (b) Insertion.
    Four series per panel: PH2/HAM10000 × MRF/random.
    No std bands. Colours: MRF green, random red. Markers: PH2 ○, HAM □.
    Linestyles: MRF solid, random dashed (same in both panels).
    """
    for ds in ("ph2", "ham10000"):
        if ds not in per_variant:
            return ""
        for strat in ("topk_grad", "random"):
            if strat not in per_variant[ds]:
                return ""

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    ax_del, ax_ins = axes

    for ds in ("ph2", "ham10000"):
        marker = _PH2_HAM_DATASET_MARKERS[ds]
        ds_name = _DS_DISPLAY.get(ds, ds)
        for strat, ls in (("topk_grad", "-"), ("random", "--")):
            p = per_variant[ds][strat]
            fr = np.asarray(p["fracs"], dtype=np.float64)
            del_y = np.asarray(p["mean_deletion_probs"], dtype=np.float64)
            ins_y = np.asarray(p["mean_insertion_probs"], dtype=np.float64)
            color = _PH2_HAM_STRATEGY_COLOR[strat]
            strat_label = "MRF" if strat == "topk_grad" else "Random"
            # Legend matches linestyle: MRF solid, Random dashed; colours green/red.
            lab_del = (
                f"{strat_label}, {ds_name}  "
                f"(AUC={p['mean_AUC_del']:.3f})"
            )
            lab_ins = (
                f"{strat_label}, {ds_name}  "
                f"(AUC={p['mean_AUC_ins']:.3f})"
            )
            _plot_mean_std_band(
                ax_del, fr, del_y, None, color, ls, marker, lab_del,
            )
            _plot_mean_std_band(
                ax_ins, fr, ins_y, None, color, ls, marker, lab_ins,
            )

    ax_del.set_title("(a)  Deletion ↓", loc="left", fontweight="bold")
    ax_ins.set_title("(b)  Insertion ↑", loc="left", fontweight="bold")
    for ax, xlabel in zip(
        axes,
        ("Fraction of concept nodes removed",
         "Fraction of concept nodes inserted"),
    ):
        ax.set_xlabel(xlabel)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", frameon=True, fontsize=9)
    ax_del.set_ylabel("P(correct class)")

    fig.suptitle(
        f"Concept Node Faithfulness — PH2 vs HAM10000  ({variant})",
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"fidelity_ph2_ham10000_{variant}.png")
    _savefig(fig, out_path)
    plt.close(fig)
    return out_path


def _plot_auc_summary(loaded: Dict[str, Dict[str, Dict[str, dict]]],
                      datasets: List[str], variants: List[str],
                      out_dir: str) -> str:
    """One grouped bar chart: AUC_del + AUC_ins per (variant, strategy, dataset)."""
    rows: List[Tuple[str, str, str, float, float]] = []
    for variant in variants:
        for ds in datasets:
            per_strat = loaded.get(variant, {}).get(ds, {})
            for strat, payload in per_strat.items():
                rows.append((variant, ds, strat,
                             float(payload["mean_AUC_del"]),
                             float(payload["mean_AUC_ins"])))
    if not rows:
        print("[summary] no rows to plot")
        return ""

    n_groups = len(datasets)
    bar_w = 0.18
    fig, axes = plt.subplots(1, 2, figsize=(max(9, n_groups * 2.5), 5.0),
                             sharey=False)
    ax_del, ax_ins = axes
    x = np.arange(n_groups)

    series = [(v, s) for v in variants for s in ("topk_grad", "random")]
    cmap = {
        ("v1_threshold", "topk_grad"): ("#c0392b", "GCBM — MRF"),
        ("v1_threshold", "random"):    ("#e6a8a0", "GCBM — random"),
        ("v4_threshold", "topk_grad"): ("#2c3e50", "Concept-only — MRF"),
        ("v4_threshold", "random"):    ("#95a5a6", "Concept-only — random"),
        ("v1",           "topk_grad"): ("#c0392b", "GCBM — MRF"),
        ("v1",           "random"):    ("#e6a8a0", "GCBM — random"),
        ("v4",           "topk_grad"): ("#2c3e50", "Concept-only — MRF"),
        ("v4",           "random"):    ("#95a5a6", "Concept-only — random"),
    }

    for i, (variant, strat) in enumerate(series):
        del_vals, ins_vals = [], []
        for ds in datasets:
            payload = loaded.get(variant, {}).get(ds, {}).get(strat)
            del_vals.append(payload["mean_AUC_del"] if payload else np.nan)
            ins_vals.append(payload["mean_AUC_ins"] if payload else np.nan)
        color, label = cmap.get((variant, strat), ("#888", f"{variant}/{strat}"))
        offset = (i - (len(series) - 1) / 2) * bar_w
        ax_del.bar(x + offset, del_vals, bar_w, color=color, label=label)
        ax_ins.bar(x + offset, ins_vals, bar_w, color=color, label=label)

    ax_del.set_title("Faithfulness AUC — Deletion ↓")
    ax_ins.set_title("Faithfulness AUC — Insertion ↑")
    ds_labels = [_DS_DISPLAY.get(ds, ds) for ds in datasets]
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(ds_labels, rotation=20, ha="right")
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_ylim(0.0, 1.0)
        ax.legend(loc="best", frameon=True)
    ax_del.set_ylabel("AUC")

    fig.suptitle("Faithfulness AUC Summary", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "fidelity_auc_summary.png")
    _savefig(fig, out_path)
    plt.close(fig)
    return out_path


def _plot_all_datasets_overlay(loaded: Dict[str, Dict[str, Dict[str, dict]]],
                               datasets: List[str], variants: List[str],
                               out_dir: str) -> str:
    """2-panel figure: all datasets overlaid on the same deletion panel (left)
    and the same insertion panel (right).  Only the MRF (topk_grad) strategy
    is shown — one coloured line per dataset — to keep the figure readable.
    Uses the first variant that has data."""
    if not datasets:
        return ""

    _DS_COLORS_FID = {
        "ham10000": "#2980b9",
        "ph2":      "#c0392b",
        "derm7pt":  "#27ae60",
        "imagenet": "#8e44ad",
        "cub":      "#e67e22",
    }
    # Line styles cycle for extra visual distinction (solid, dashed, dotted, dashdot)
    _DS_LINESTYLES = {
        "ham10000": "-",
        "ph2":      "--",
        "derm7pt":  ":",
        "imagenet": "-.",
        "cub":      (0, (3, 1, 1, 1)),
    }

    # Pick the first variant with any data.
    primary_variant = None
    for v in variants:
        if any(loaded.get(v, {}).get(ds) for ds in datasets):
            primary_variant = v
            break
    if primary_variant is None:
        print("[all-datasets overlay] no data found for any variant")
        return ""

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    ax_del, ax_ins = axes

    for ds in datasets:
        payloads = loaded.get(primary_variant, {}).get(ds, {})
        p = payloads.get("topk_grad")
        if p is None:
            print(f"[all-datasets overlay] skip {ds} — no topk_grad data")
            continue
        fr    = np.asarray(p["fracs"])
        del_y = np.asarray(p["mean_deletion_probs"])
        ins_y = np.asarray(p["mean_insertion_probs"])
        color = _DS_COLORS_FID.get(ds, "#888")
        ls    = _DS_LINESTYLES.get(ds, "-")
        label = f"{_DS_DISPLAY.get(ds, ds)}  (del AUC={p['mean_AUC_del']:.3f})"
        label_ins = f"{_DS_DISPLAY.get(ds, ds)}  (ins AUC={p['mean_AUC_ins']:.3f})"
        ax_del.plot(fr, del_y, ls, color=color, label=label,     linewidth=2.0)
        ax_ins.plot(fr, ins_y, ls, color=color, label=label_ins, linewidth=2.0)

    ax_del.set_title("(a)  Deletion ↓",  loc="left", fontweight="bold")
    ax_ins.set_title("(b)  Insertion ↑", loc="left", fontweight="bold")
    for ax, xlabel in zip(axes, ("Fraction of concept nodes removed",
                                 "Fraction of concept nodes inserted")):
        ax.set_xlabel(xlabel)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", frameon=True, fontsize=10)
    ax_del.set_ylabel("P(correct class)")

    fig.suptitle("Concept Node Faithfulness — All Datasets (MRF strategy)",
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "fidelity_all_datasets.png")
    _savefig(fig, out_path)
    plt.close(fig)
    return out_path


def _plot_all_datasets_grid(loaded: Dict[str, Dict[str, Dict[str, dict]]],
                            datasets: List[str], variants: List[str],
                            out_dir: str) -> str:
    """2-row × N-col grid: deletion (row 0) and insertion (row 1) per-dataset columns,
    with both MRF and Random strategies shown.  Saved as fidelity_all_datasets_grid.png."""
    if not datasets:
        return ""

    primary_variant = None
    for v in variants:
        if any(loaded.get(v, {}).get(ds) for ds in datasets):
            primary_variant = v
            break
    if primary_variant is None:
        print("[all-datasets grid] no data found for any variant")
        return ""

    n_cols = len(datasets)
    fig, axes = plt.subplots(2, n_cols,
                             figsize=(max(10, n_cols * 3.5), 7),
                             sharey="row")
    if n_cols == 1:
        axes = np.asarray([[axes[0]], [axes[1]]])

    for j, ds in enumerate(datasets):
        payloads = loaded.get(primary_variant, {}).get(ds, {})
        for strat in ("topk_grad", "random"):
            if strat not in payloads:
                continue
            p = payloads[strat]
            fr    = np.asarray(p["fracs"])
            del_y = np.asarray(p["mean_deletion_probs"])
            ins_y = np.asarray(p["mean_insertion_probs"])
            style = _STRATEGY_STYLES[strat]
            axes[0, j].plot(fr, del_y, "-",  color=style["color"],
                            label=style["label"], linewidth=1.8)
            axes[1, j].plot(fr, ins_y, "--", color=style["color"],
                            label=style["label"], linewidth=1.8)

        for i in (0, 1):
            axes[i, j].set_xlim(0.0, 1.0)
            axes[i, j].set_ylim(0.0, 1.05)
            axes[i, j].grid(True, alpha=0.3)
        axes[1, j].set_xlabel("Fraction removed / inserted")

    axes[0, 0].set_ylabel("P(correct class)")
    axes[1, 0].set_ylabel("P(correct class)")
    axes[0, 0].set_title(f"(a)  Deletion ↓\n{_DS_DISPLAY.get(datasets[0], datasets[0])}",
                         loc="left", fontweight="bold")
    for j, ds in enumerate(datasets[1:], start=1):
        axes[0, j].set_title(_DS_DISPLAY.get(ds, ds))
    axes[1, 0].set_title("(b)  Insertion ↑", loc="left", fontweight="bold")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center",
                   ncol=len(handles), fontsize=11,
                   bbox_to_anchor=(0.5, 0.98))

    fig.suptitle("Concept Node Faithfulness — All Datasets",
                 fontweight="bold", y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "fidelity_all_datasets_grid.png")
    _savefig(fig, out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _load_cnn_gradcam_jsons(cnn_fidelity_root: str,
                            datasets: List[str]) -> Dict[str, Dict[str, dict]]:
    """Return {dataset: {strategy: payload}} from the CNN Grad-CAM fidelity dir.

    Expects files at: <cnn_fidelity_root>/<dataset>/<dataset>_<strategy>.json
    This is the format written by eval_fidelity_cnn_gradcam.py.
    """
    out: Dict[str, Dict[str, dict]] = {}
    for ds in datasets:
        per_strategy: Dict[str, dict] = {}
        for strat in ("topk_grad", "random"):
            path = os.path.join(cnn_fidelity_root, ds, f"{ds}_{strat}.json")
            if os.path.isfile(path):
                with open(path) as fh:
                    per_strategy[strat] = json.load(fh)
        if per_strategy:
            out[ds] = per_strategy
    return out


def _plot_gcbm_vs_cnn_overlay(
    gcbm_loaded: Dict[str, Dict[str, dict]],   # {dataset: {strategy: payload}}
    cnn_loaded:  Dict[str, Dict[str, dict]],   # {dataset: {strategy: payload}}
    datasets: List[str],
    variant: str,
    out_dir: str,
) -> str:
    """
    2-row × N-col grid comparing G-CBM (concept-node) and CNN (Grad-CAM)
    fidelity curves on the same axes.

    Row 0 — Deletion ↓  (lower AUC = more faithful)
    Row 1 — Insertion ↑ (higher AUC = more faithful)

    Per panel four series:
      G-CBM MRF    solid red    (existing topk_grad from eval_fidelity_v2)
      G-CBM Random solid grey   (existing random from eval_fidelity_v2)
      CNN MRF      dashed blue  (new topk_grad from eval_fidelity_cnn_gradcam)
      CNN Random   dashed tan   (new random from eval_fidelity_cnn_gradcam)

    Only datasets that have data in BOTH loaders are plotted.
    """
    # Restrict to datasets with both G-CBM and CNN data.
    avail = [ds for ds in datasets
             if gcbm_loaded.get(ds) and cnn_loaded.get(ds)]
    if not avail:
        print("[gcbm-vs-cnn overlay] no datasets with both G-CBM and CNN data")
        return ""

    _GCBM_MRF   = {"color": "#c0392b", "ls": "-",  "label": "G-CBM MRF"}
    _GCBM_RND   = {"color": "#7f8c8d", "ls": "-",  "label": "G-CBM Random"}
    _CNN_MRF    = {"color": "#2471a3", "ls": "--", "label": "CNN (ResNet-50) MRF"}
    _CNN_RND    = {"color": "#a9cce3", "ls": "--", "label": "CNN (ResNet-50) Random"}

    n_cols = len(avail)
    fig, axes = plt.subplots(
        2, n_cols,
        figsize=(max(10, n_cols * 3.5), 7),
        sharey="row",
    )
    if n_cols == 1:
        axes = np.asarray([[axes[0]], [axes[1]]])

    for j, ds in enumerate(avail):
        gcbm = gcbm_loaded.get(ds, {})
        cnn  = cnn_loaded.get(ds, {})
        ds_name = _DS_DISPLAY.get(ds, ds)

        series = [
            ("topk_grad", gcbm, _GCBM_MRF),
            ("random",    gcbm, _GCBM_RND),
            ("topk_grad", cnn,  _CNN_MRF),
            ("random",    cnn,  _CNN_RND),
        ]

        for strat, loader, style in series:
            p = loader.get(strat)
            if p is None:
                continue
            fr    = np.asarray(p["fracs"])
            del_y = np.asarray(p["mean_deletion_probs"])
            ins_y = np.asarray(p["mean_insertion_probs"])
            del_std = np.asarray(p.get("std_deletion_probs") or
                                 [0.0] * len(del_y))
            ins_std = np.asarray(p.get("std_insertion_probs") or
                                 [0.0] * len(ins_y))

            lab_del = (f"{style['label']}  "
                       f"(AUC={p['mean_AUC_del']:.3f})")
            lab_ins = (f"{style['label']}  "
                       f"(AUC={p['mean_AUC_ins']:.3f})")

            _plot_mean_std_band(
                axes[0, j], fr, del_y, del_std,
                style["color"], style["ls"], "none", lab_del,
                markersize=0,
            )
            _plot_mean_std_band(
                axes[1, j], fr, ins_y, ins_std,
                style["color"], style["ls"], "none", lab_ins,
                markersize=0,
            )

        axes[0, j].set_title(ds_name, fontsize=11)
        for row in (0, 1):
            axes[row, j].set_xlim(0.0, 1.0)
            axes[row, j].set_ylim(0.0, 1.05)
            axes[row, j].grid(True, alpha=0.3)
        axes[1, j].set_xlabel("Fraction removed / inserted", fontsize=10)

    # Row labels
    axes[0, 0].set_ylabel("P(correct class)")
    axes[1, 0].set_ylabel("P(correct class)")

    # Panel labels on leftmost column
    axes[0, 0].set_title(
        f"(a) Deletion ↓\n{_DS_DISPLAY.get(avail[0], avail[0])}",
        loc="left", fontweight="bold")
    axes[1, 0].set_title("(b) Insertion ↑", loc="left", fontweight="bold")

    # Shared legend at the top
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center",
                   ncol=min(len(handles), 4), fontsize=10,
                   bbox_to_anchor=(0.5, 0.99), frameon=True)

    fig.suptitle(
        f"Faithfulness: G-CBM (concept nodes) vs CNN ResNet-50 (Grad-CAM)  "
        f"[{variant}]",
        fontweight="bold", y=1.04,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"fidelity_gcbm_vs_cnn_{variant}.png")
    _savefig(fig, out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser(
        "plot_fidelity — render fidelity figures from eval_fidelity_v2 output")
    ap.add_argument("--run-root", required=True,
                    help="Path to <run_id>/concept_graph_data — same value "
                         "passed to eval_fidelity_v2.")
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="Datasets to plot. Default: auto-discover.")
    ap.add_argument("--variants", nargs="+",
                    default=["v1_threshold", "v4_threshold"],
                    help="Graph variants to look up under "
                         "<run_root>/fidelity/<variant>/.")
    ap.add_argument("--out-dir", default=None,
                    help="Output directory. Default: <run_root>/figures/")
    ap.add_argument(
        "--cnn-fidelity-root", default=None,
        help="Path to the output directory produced by "
             "eval_fidelity_cnn_gradcam.py "
             "(i.e. the --out-dir passed to that script, "
             "default: <cnn-runs-base>/cnn_gradcam_fidelity). "
             "When supplied, generates an additional "
             "'fidelity_gcbm_vs_cnn_<variant>.png' overlay figure "
             "for each variant that compares G-CBM concept-node curves "
             "with CNN Grad-CAM pixel-level curves.")
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(args.run_root, "figures")

    if args.datasets:
        datasets = args.datasets
    else:
        datasets = _autodiscover_datasets(args.run_root, args.variants)
        if not datasets:
            print("[error] no fidelity JSONs found under "
                  f"{args.run_root}/fidelity/. Did Phase 7 finish?")
            return
        print(f"[auto-discover] datasets = {datasets}")

    loaded: Dict[str, Dict[str, Dict[str, dict]]] = {}
    for variant in args.variants:
        loaded[variant] = _load_jsons(args.run_root, variant, datasets)

    written = []
    for variant in args.variants:
        for ds in datasets:
            payloads = loaded.get(variant, {}).get(ds, {})
            if not payloads:
                print(f"[skip] {ds} / {variant} — no JSON on disk")
                continue
            out_path = _plot_one(ds, variant, payloads, out_dir)
            written.append(out_path)
            print(f"  wrote {out_path}")

    summary_path = _plot_auc_summary(loaded, datasets, args.variants, out_dir)
    if summary_path:
        written.append(summary_path)
        print(f"  wrote {summary_path}")

    overlay_path = _plot_all_datasets_overlay(loaded, datasets, args.variants, out_dir)
    if overlay_path:
        written.append(overlay_path)
        print(f"  wrote {overlay_path}")

    grid_path = _plot_all_datasets_grid(loaded, datasets, args.variants, out_dir)
    if grid_path:
        written.append(grid_path)
        print(f"  wrote {grid_path}")

    for variant in args.variants:
        per = loaded.get(variant, {})
        if "ph2" not in per or "ham10000" not in per:
            continue
        combo = _plot_ph2_ham_combined(per, variant, out_dir)
        if combo:
            written.append(combo)
            print(f"  wrote {combo}")
        else:
            print(f"[skip] PH2+HAM combined / {variant} — need both datasets "
                  f"with topk_grad and random JSONs")

    # ---- G-CBM vs CNN overlay (requires --cnn-fidelity-root) ----
    if args.cnn_fidelity_root:
        print(f"\n[cnn overlay] loading CNN Grad-CAM fidelity from "
              f"{args.cnn_fidelity_root}")
        cnn_loaded = _load_cnn_gradcam_jsons(args.cnn_fidelity_root, datasets)
        if not cnn_loaded:
            print("  [warning] no CNN fidelity JSONs found — "
                  "did eval_fidelity_cnn_gradcam.py finish?")
        else:
            for variant in args.variants:
                gcbm_for_variant = loaded.get(variant, {})
                overlay = _plot_gcbm_vs_cnn_overlay(
                    gcbm_for_variant, cnn_loaded, datasets, variant, out_dir)
                if overlay:
                    written.append(overlay)
                    print(f"  wrote {overlay}")

    print(f"\nDone. {len(written)} figure(s) under {out_dir}")


if __name__ == "__main__":
    main()
