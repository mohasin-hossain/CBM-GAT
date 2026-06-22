"""
eval_fidelity_cnn_gradcam.py — pixel/patch-level faithfulness evaluator for
CNN baselines (ResNet-50) using Grad-CAM spatial rankings.

Applies the same MRF-vs-random deletion/insertion protocol as
eval_fidelity_v2.py, but at the image-patch level:

  1. Load the trained ResNet-50 from its .pt state-dict file.
  2. Run Grad-CAM on model.layer4[-1] to obtain a 7×7 importance map.
  3. Rank the 49 spatial cells by Grad-CAM score (MRF) or at random.
  4. Deletion: progressively Gaussian-blur the top-k cells (highest first).
  5. Insertion: start from a fully blurred image, progressively restore
     top-k cells from the original (highest first).
  6. Record P(true class) at frk_step+1 evenly-spaced fraction steps.
  7. Compute AUC_del / AUC_ins and emit JSON + CSV in the same format as
     eval_fidelity_v2.py — so plot_fidelity.py can load both without changes.

Backbone: ResNet-50 only.
  G-CBM uses ResNet-50 as its CRAFT backbone, so fixing the CNN baseline to
  the same architecture isolates the explanation method (concept-node ranking
  vs Grad-CAM spatial-cell ranking) as the sole experimental variable.

CNN weights are located at:
  <cnn-runs-base>/baseline_cnn_<dataset>_resnet50/concept_graph_data/
      <dataset>/models_cnn/<dataset>/<dataset>_resnet50_cnn.pt

Output per (dataset, strategy):
  <out-dir>/<dataset>/<dataset>_<strategy>.json   — aggregated curves + AUC
  <out-dir>/<dataset>/<dataset>_<strategy>.csv    — per-image AUC rows

JSON keys are identical to eval_fidelity_v2.py output:
  n_images, fracs, mean_deletion_probs, mean_insertion_probs,
  std_deletion_probs, std_insertion_probs,
  mean_AUC_del, mean_AUC_ins, std_AUC_del, std_AUC_ins
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur

from config import DATASETS
from train_cnn import build_backbone
from utils import _set_seed


BACKBONE = "resnet50"

_GRID_H = _GRID_W = 7   # ResNet-50 layer4 output spatial size
_N_CELLS = _GRID_H * _GRID_W       # 49 cells
_CELL_SIZE = 224 // _GRID_H         # 32 pixels per cell side (224/7 = 32)


# ---------------------------------------------------------------------------
# Grad-CAM for ResNet-50
# ---------------------------------------------------------------------------

class _GradCAM:
    """Hook-based Grad-CAM for model.layer4[-1] of a ResNet-50."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self._fmap: torch.Tensor | None = None
        self._grad: torch.Tensor | None = None
        target = model.layer4[-1]
        self._fwd_handle = target.register_forward_hook(self._on_fwd)
        self._bwd_handle = target.register_full_backward_hook(self._on_bwd)

    def _on_fwd(self, module, inp, out):
        # Keep feature map in the computational graph for backward to work.
        self._fmap = out

    def _on_bwd(self, module, grad_in, grad_out):
        # grad_out[0] is ∂loss/∂(layer output): shape (1, C, H, W)
        self._grad = grad_out[0]

    def compute(self, x: torch.Tensor, class_idx: int) -> torch.Tensor:
        """Return a (7, 7) heat map for a single image (1, 3, 224, 224).

        Must be called inside a torch.enable_grad() context.
        """
        self.model.zero_grad()
        logits = self.model(x)          # triggers forward hook
        logits[0, class_idx].backward() # triggers backward hook

        fmap = self._fmap.detach()      # (1, 2048, 7, 7)
        grad = self._grad.detach()      # (1, 2048, 7, 7)

        alpha = grad.mean(dim=(2, 3), keepdim=True)  # (1, 2048, 1, 1)
        cam = F.relu((alpha * fmap).sum(dim=1)).squeeze(0)  # (7, 7)

        cam_max = cam.max()
        if cam_max > 1e-8:
            cam = cam / cam_max
        return cam

    def remove_hooks(self) -> None:
        self._fwd_handle.remove()
        self._bwd_handle.remove()


# ---------------------------------------------------------------------------
# Image perturbation helpers
# ---------------------------------------------------------------------------

def _make_blurred(img: torch.Tensor, blur_sigma: float) -> torch.Tensor:
    """Return a Gaussian-blurred copy of img (C, H, W), CPU tensor."""
    ks = max(3, int(6 * blur_sigma + 1) | 1)  # smallest odd integer ≥ 6σ
    return gaussian_blur(img, kernel_size=[ks, ks], sigma=[blur_sigma, blur_sigma])


def _apply_cells(
    base: torch.Tensor,         # (C, H, W) — background image
    donor: torch.Tensor,        # (C, H, W) — source of replacement patches
    cell_order: List[int],      # cell indices in preference order
    k: int,                     # how many cells to copy from donor
) -> torch.Tensor:
    """Copy the first k cells (by cell_order) from donor into a clone of base."""
    out = base.clone()
    for cell_idx in cell_order[:k]:
        row = cell_idx // _GRID_W
        col = cell_idx % _GRID_W
        r0, c0 = row * _CELL_SIZE, col * _CELL_SIZE
        r1, c1 = r0 + _CELL_SIZE, c0 + _CELL_SIZE
        out[:, r0:r1, c0:c1] = donor[:, r0:r1, c0:c1]
    return out


# ---------------------------------------------------------------------------
# Per-image fidelity curve
# ---------------------------------------------------------------------------

def _fidelity_curve_single(
    model: nn.Module,
    gradcam: _GradCAM,
    img: torch.Tensor,          # (C, H, W) on CPU, float32
    device: str,
    strategy: str,
    frk_step: int,
    blur_sigma: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Return (fracs, del_probs, ins_probs, auc_del, auc_ins) for one image."""

    x = img.unsqueeze(0).to(device)

    # ---- predict on original (no grad needed) ----
    with torch.no_grad():
        y_hat = model(x)[0].argmax().item()

    # ---- Grad-CAM on the predicted class ----
    with torch.enable_grad():
        cam = gradcam.compute(x, y_hat)      # (7, 7) on device

    if strategy == "topk_grad":
        # most-relevant first: highest cam value = index 0 in order
        order: List[int] = cam.flatten().argsort(descending=True).cpu().tolist()
    elif strategy == "random":
        order = rng.permutation(_N_CELLS).tolist()
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}")

    img_cpu = img.cpu()
    blurred = _make_blurred(img_cpu, blur_sigma)

    # ---- precompute all (2 * frk_step+1) perturbed images in one batch ----
    imgs: List[torch.Tensor] = []
    for i in range(frk_step + 1):
        k = int(round(i * _N_CELLS / frk_step))
        # deletion: original → progressively blurred
        imgs.append(_apply_cells(img_cpu, blurred, order, k))
        # insertion: blurred → progressively restored
        imgs.append(_apply_cells(blurred, img_cpu, order, k))

    batch = torch.stack(imgs, dim=0).to(device)   # (2*(frk+1), C, H, W)
    with torch.no_grad():
        probs = F.softmax(model(batch), dim=1)[:, y_hat].cpu().numpy()

    del_probs = probs[0::2]          # shape (frk_step+1,)
    ins_probs = probs[1::2]
    fracs = np.linspace(0.0, 1.0, frk_step + 1)
    auc_del = float(np.trapz(del_probs, fracs))
    auc_ins = float(np.trapz(ins_probs, fracs))

    return fracs, del_probs, ins_probs, auc_del, auc_ins


# ---------------------------------------------------------------------------
# Per-dataset evaluation
# ---------------------------------------------------------------------------

def _evaluate_dataset(
    model: nn.Module,
    gradcam: _GradCAM,
    X_test: torch.Tensor,       # (N, C, H, W)
    strategy: str,
    frk_step: int,
    blur_sigma: float,
    device: str,
    seed: int,
) -> Tuple[dict, list]:
    rng = np.random.default_rng(seed)
    grid = np.linspace(0.0, 1.0, frk_step + 1)
    curves_del: List[np.ndarray] = []
    curves_ins: List[np.ndarray] = []
    per_image: List[dict] = []

    for i in range(len(X_test)):
        img = X_test[i]
        try:
            _, delp, insp, aucd, auci = _fidelity_curve_single(
                model, gradcam, img, device,
                strategy, frk_step, blur_sigma, rng,
            )
            curves_del.append(delp)
            curves_ins.append(insp)
            per_image.append({"image_index": i, "AUC_del": aucd, "AUC_ins": auci})
        except Exception as e:
            print(f"  [skip image {i}]: {e}")

        if (i + 1) % 50 == 0:
            print(f"    processed {i + 1}/{len(X_test)}")

    n_ok = len(curves_del)
    _empty = {
        "n_images":            0,
        "fracs":               grid.tolist(),
        "mean_deletion_probs": [],
        "mean_insertion_probs": [],
        "std_deletion_probs":  [],
        "std_insertion_probs":  [],
        "mean_AUC_del": float("nan"),
        "mean_AUC_ins": float("nan"),
        "std_AUC_del":  float("nan"),
        "std_AUC_ins":  float("nan"),
    }
    if n_ok == 0:
        return _empty, per_image

    stack_del = np.stack(curves_del, axis=0)
    stack_ins = np.stack(curves_ins, axis=0)
    aucs_del  = np.array([r["AUC_del"] for r in per_image])
    aucs_ins  = np.array([r["AUC_ins"] for r in per_image])

    return {
        "n_images":            n_ok,
        "fracs":               grid.tolist(),
        "mean_deletion_probs": stack_del.mean(0).tolist(),
        "mean_insertion_probs": stack_ins.mean(0).tolist(),
        "std_deletion_probs":  stack_del.std(0, ddof=0).tolist(),
        "std_insertion_probs":  stack_ins.std(0, ddof=0).tolist(),
        "mean_AUC_del": float(aucs_del.mean()),
        "mean_AUC_ins": float(aucs_ins.mean()),
        "std_AUC_del":  float(aucs_del.std()),
        "std_AUC_ins":  float(aucs_ins.std()),
    }, per_image


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        "eval_fidelity_cnn_gradcam — pixel-level fidelity for CNN baselines "
        "(ResNet-50, Grad-CAM ranking)")
    ap.add_argument(
        "--cnn-runs-base",
        default="/netscratch/mhossain/cbm_gat/runs",
        help="Parent directory of all run folders. The CNN weights for each "
             "dataset are expected at: "
             "<cnn-runs-base>/baseline_cnn_<dataset>_resnet50/"
             "concept_graph_data/<dataset>/models_cnn/<dataset>/"
             "<dataset>_resnet50_cnn.pt")
    ap.add_argument(
        "--out-dir", default=None,
        help="Output root directory. Default: "
             "<cnn-runs-base>/cnn_gradcam_fidelity/")
    ap.add_argument(
        "--datasets", nargs="+",
        default=["ham10000", "ph2", "derm7pt", "imagenet"],
        choices=list(DATASETS.keys()),
        help="Datasets to evaluate (default: all four).")
    ap.add_argument(
        "--strategies", nargs="+",
        default=["topk_grad", "random"],
        choices=["topk_grad", "random"],
        help="Ranking strategies. topk_grad = MRF (Grad-CAM score), "
             "random = reviewer-critical baseline.")
    ap.add_argument(
        "--frk-step", type=int, default=10,
        help="Number of fraction steps; total grid = frk_step+1 points "
             "(default 10 → 11 points, matching eval_fidelity_v2.py).")
    ap.add_argument(
        "--blur-sigma", type=float, default=10.0,
        help="Gaussian blur σ for patch perturbation (default 10).")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = (args.device
              if args.device.startswith("cuda") and torch.cuda.is_available()
              else "cpu")
    print(f"device: {device}")
    _set_seed(args.seed)

    out_root = (args.out_dir
                or os.path.join(args.cnn_runs_base, "cnn_gradcam_fidelity"))

    for ds in args.datasets:
        print(f"\n==================== DATASET: {ds} ====================")

        # ---- locate weights ----
        weights_path = os.path.join(
            args.cnn_runs_base,
            f"baseline_cnn_{ds}_{BACKBONE}",
            "concept_graph_data",
            ds, "models_cnn", ds,
            f"{ds}_{BACKBONE}_cnn.pt",
        )
        if not os.path.isfile(weights_path):
            print(f"  [skip] weights not found: {weights_path}")
            continue

        # ---- load model ----
        ckpt = torch.load(weights_path, map_location="cpu")
        num_classes = ckpt["num_classes"]
        model = build_backbone(BACKBONE, num_classes)
        model.load_state_dict(ckpt["state_dict"])
        model = model.eval().to(device)
        print(f"  loaded {BACKBONE} ({num_classes} classes) from {weights_path}")

        # ---- load test images ----
        ds_spec = DATASETS[ds]
        tdict = ds_spec.build_transforms()
        paths = ds_spec.resolve_paths()
        X_test, _, _ = ds_spec.load_split(paths, tdict, "test")
        print(f"  test images: {len(X_test)}")

        # ---- build Grad-CAM ----
        gradcam = _GradCAM(model)

        ds_out_dir = os.path.join(out_root, ds)
        os.makedirs(ds_out_dir, exist_ok=True)

        for strategy in args.strategies:
            print(f"\n  --- strategy: {strategy} ---")
            agg, per_image = _evaluate_dataset(
                model, gradcam, X_test, strategy,
                args.frk_step, args.blur_sigma, device, args.seed,
            )

            payload = {
                "dataset":    ds,
                "backbone":   BACKBONE,
                "strategy":   strategy,
                "frk_step":   args.frk_step,
                "blur_sigma": args.blur_sigma,
                "seed":       args.seed,
                **agg,
            }

            json_path = os.path.join(ds_out_dir, f"{ds}_{strategy}.json")
            csv_path  = os.path.join(ds_out_dir, f"{ds}_{strategy}.csv")

            with open(json_path, "w") as fh:
                json.dump(payload, fh, indent=2)

            with open(csv_path, "w", newline="") as fh:
                writer = csv.DictWriter(
                    fh, fieldnames=["image_index", "AUC_del", "AUC_ins"])
                writer.writeheader()
                writer.writerows(per_image)

            print(f"    n_images    : {agg['n_images']}")
            print(f"    mean AUC del: {agg['mean_AUC_del']:.4f} "
                  f"± {agg['std_AUC_del']:.4f}")
            print(f"    mean AUC ins: {agg['mean_AUC_ins']:.4f} "
                  f"± {agg['std_AUC_ins']:.4f}")
            print(f"    -> {json_path}")
            print(f"    -> {csv_path}")

        gradcam.remove_hooks()

    print("\nDone.")


if __name__ == "__main__":
    main()
