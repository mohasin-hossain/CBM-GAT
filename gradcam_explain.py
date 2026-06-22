import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import argparse
from typing import List, Optional
import numpy as np
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

mpl.rcParams.update({
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.labelsize":   12,
    "xtick.labelsize":  11,
    "ytick.labelsize":  11,
    "legend.fontsize":  10,
    "figure.titlesize": 14,
})
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from config import DATASETS, default_output_dir, get_dataset_params, get_class_label


# ---------------------------------------------------------------------------
# Grad-CAM (self-contained, no extra dependency)
# ---------------------------------------------------------------------------

class GradCAM:
    """Gradient-weighted Class Activation Mapping on a target conv layer."""

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model.eval()
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    @torch.enable_grad()
    def generate(self, input_tensor: torch.Tensor,
                 target_class: Optional[int] = None,
                 img_size: int = 224):
        """
        Returns (cam_np, predicted_class, confidence).
        cam_np is a float32 array of shape (img_size, img_size) in [0, 1].
        """
        self.model.zero_grad()
        logits = self.model(input_tensor)
        probs = F.softmax(logits, dim=1)

        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())
        confidence = float(probs[0, target_class].item())

        score = logits[0, target_class]
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # [1, 1, H, W]
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(img_size, img_size),
                            mode="bilinear", align_corners=False)
        cam = cam.squeeze()
        cam_min, cam_max = cam.min(), cam.max()
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        return cam.cpu().numpy(), target_class, confidence


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_eval_transform(dataset_key: str):
    """Get the dataset-specific eval transform from config."""
    tdict = DATASETS[dataset_key].build_transforms()
    return tdict["eval"]


def load_resnet50(device: str = "cuda") -> models.ResNet:
    """Load full pretrained ResNet-50 (same weights used by CRAFT)."""
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights).to(device).eval()
    return model


def load_resnet50_finetuned(ckpt_path: str,
                            device: str = "cuda") -> nn.Module:
    """Load a fine-tuned CNN saved by train_cnn.py.

    The checkpoint dict has keys: ``state_dict``, ``num_classes``, ``backbone``.
    The backbone key determines which architecture to reconstruct — no external
    argument needed (avoids mismatch when the caller uses a different backbone name).
    Falls back to resnet50 if the key is absent (old checkpoints).
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    num_classes = ckpt["num_classes"]
    backbone = ckpt.get("backbone", "resnet50")
    if backbone == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif backbone == "densenet201":
        model = models.densenet201(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif backbone == "mobilenet_v2":
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unsupported backbone in checkpoint: {backbone!r}")
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()


def prepare_image(image_path: str, dataset_key: str, device: str):
    """
    Returns:
      input_tensor  – (1, 3, 224, 224) normalised tensor on device
      image_pil     – 224x224 PIL image for display
    """
    image_pil = Image.open(image_path).convert("RGB").resize((224, 224), Image.BICUBIC)
    eval_tfm = load_eval_transform(dataset_key)
    input_tensor = eval_tfm(image_pil).unsqueeze(0).to(device)
    return input_tensor, image_pil


def overlay_heatmap(image_pil: Image.Image, cam: np.ndarray, alpha: float = 0.5):
    """Alpha-blend a jet heatmap onto a PIL image, return as np array."""
    cmap = plt.cm.jet
    heatmap_rgba = cmap(cam)[:, :, :3]  # drop alpha channel
    heatmap_uint8 = (heatmap_rgba * 255).astype(np.uint8)
    img_arr = np.array(image_pil).astype(np.float32)
    blended = (1 - alpha) * img_arr + alpha * heatmap_uint8.astype(np.float32)
    return np.clip(blended, 0, 255).astype(np.uint8)


def patch_importance_to_heatmap(patch_importance: torch.Tensor,
                                num_patches_h: int,
                                num_patches_w: int,
                                img_size: int = 224) -> np.ndarray:
    """
    Convert a 1D patch importance vector (length num_patches_h*num_patches_w)
    into a dense (img_size, img_size) heatmap in [0, 1].
    """
    if patch_importance.ndim != 1:
        patch_importance = patch_importance.view(-1)

    expected = int(num_patches_h * num_patches_w)
    if patch_importance.numel() != expected:
        raise ValueError(
            f"patch_importance has {patch_importance.numel()} values but expected {expected} "
            f"({num_patches_h}x{num_patches_w})."
        )

    grid = patch_importance.detach().float().view(1, 1, num_patches_h, num_patches_w)
    grid = grid - grid.min()
    grid = grid / (grid.max() + 1e-8)
    dense = F.interpolate(grid, size=(img_size, img_size), mode="bilinear", align_corners=False)
    return dense.squeeze().cpu().numpy()


def annotate_concept_patch_panel(
    image_pil_cbm: Image.Image,
    patches_U: np.ndarray,
    patch_importance_np: np.ndarray,
    top_concepts: List[int],
    colors: np.ndarray,
    stride: int,
    num_patches_w: int,
    patch_size: int,
    sorted_patch_idx: List[int],
    top_k: int,
) -> Image.Image:
    """
    Draw exactly one patch-level box per top concept: argmax of
    (concept activation × patch importance). Falls back to top patch indices if needed.
    """
    img = image_pil_cbm.copy()
    draw = ImageDraw.Draw(img)
    num_concepts_total = patches_U.shape[1]

    selected_indices = []
    for concept_id in top_concepts:
        if concept_id >= num_concepts_total:
            continue
        concept_activations = patches_U[:, concept_id]
        scores = concept_activations * patch_importance_np
        best_idx = int(np.argmax(scores))
        selected_indices.append((concept_id, best_idx))
    if not selected_indices:
        selected_indices = [(top_concepts[0], idx) for idx in sorted_patch_idx[:top_k]]

    for concept_id, idx in selected_indices:
        row = idx // num_patches_w
        col = idx % num_patches_w
        x, y = col * stride, row * stride
        c_index = top_concepts.index(concept_id) if concept_id in top_concepts else 0
        outline_color = tuple((colors[c_index] * 255).astype(int))
        draw.rectangle(
            [x, y, x + patch_size, y + patch_size],
            outline=tuple(outline_color),
            width=3,
        )
    return img


def _greedy_distinct_patch_indices(
    scores_1d: np.ndarray,
    num_patches_h: int,
    num_patches_w: int,
    max_boxes: int,
    min_sep: int,
    floor_frac: float,
) -> List[int]:
    """
    Greedy non-maximum suppression on the patch grid: take high-scoring patches in order,
    skipping any whose Chebyshev distance to an already chosen patch is < ``min_sep``.
    Candidates below ``floor_frac * max(scores)`` are never selected (except fallback).
    """
    scores_1d = np.asarray(scores_1d, dtype=np.float64).reshape(-1)
    n = int(num_patches_h * num_patches_w)
    if scores_1d.size < n:
        pad = np.zeros(n, dtype=np.float64)
        pad[: scores_1d.size] = scores_1d
        scores_1d = pad
    elif scores_1d.size > n:
        scores_1d = scores_1d[:n]

    smax = float(scores_1d.max())
    if smax <= 1e-12:
        return [int(np.argmax(scores_1d))]

    flat_order = np.argsort(-scores_1d)
    chosen: List[int] = []
    for idx in flat_order:
        idx = int(idx)
        if scores_1d[idx] < floor_frac * smax:
            break
        r, c = idx // num_patches_w, idx % num_patches_w
        ok = True
        for j in chosen:
            r2 = j // num_patches_w
            c2 = j % num_patches_w
            if max(abs(r - r2), abs(c - c2)) < min_sep:
                ok = False
                break
        if ok:
            chosen.append(idx)
            if len(chosen) >= max_boxes:
                break
    if not chosen:
        chosen = [int(np.argmax(scores_1d))]
    return chosen


def save_concept_heatmaps_row(
    out_basename: str,
    top_concepts: List[int],
    colors: np.ndarray,
    patches_U: np.ndarray,
    patch_importance: torch.Tensor,
    num_patches_h: int,
    num_patches_w: int,
    img_size: int,
    image_pil: Image.Image,
    alpha: float,
) -> str:
    """
    One row of per-concept patch heatmaps overlaid on the same CBM image (``--alpha`` blend).
    Saves ``{out_basename}_concept_heatmaps.png`` and ``.svg`` in cwd. Returns the PNG path.
    """
    patch_importance_np = patch_importance.detach().cpu().numpy()
    top_k = len(top_concepts)
    fig_w = max(12, 4 * top_k)
    fig, axes = plt.subplots(1, top_k, figsize=(fig_w, 4.5))
    if top_k == 1:
        axes = [axes]
    for i, concept_id in enumerate(top_concepts):
        concept_activations = patches_U[:, concept_id]
        scores = concept_activations * patch_importance_np
        hm = patch_importance_to_heatmap(
            torch.from_numpy(scores.astype(np.float32)),
            num_patches_h,
            num_patches_w,
            img_size=img_size,
        )
        blended = overlay_heatmap(image_pil, hm, alpha)
        ax = axes[i]
        ax.imshow(blended)
        ax.set_title(
            f"Concept {concept_id}",
            fontsize=14, fontweight="bold",
            color=tuple(colors[i][:3]),
            loc="center",
        )
        ax.axis("off")
    fig.suptitle("Per-concept activation heatmaps", fontsize=14,
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    out_path = os.path.join(os.getcwd(), f"{out_basename}_concept_heatmaps.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.25)
    svg_path = out_path.replace(".png", ".svg")
    plt.savefig(svg_path, format="svg", bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"[INFO] Per-concept heatmap row saved to {out_path}  |  {svg_path}")
    return out_path


def save_concept_heatmaps_row_multi_boxes(
    out_basename: str,
    top_concepts: List[int],
    colors: np.ndarray,
    patches_U: np.ndarray,
    patch_importance: torch.Tensor,
    num_patches_h: int,
    num_patches_w: int,
    img_size: int,
    image_pil: Image.Image,
    alpha: float,
    stride: int,
    patch_size: int,
    max_distinct_patches: int,
    min_patch_separation: int,
    floor_score_frac: float,
) -> str:
    """
    Same layout as ``save_concept_heatmaps_row``, but draws multiple patch boxes per concept
    via :func:`_greedy_distinct_patch_indices`. Saves
    ``{out_basename}_concept_heatmaps_multi_boxes.png`` and ``.svg``.
    """
    patch_importance_np = patch_importance.detach().cpu().numpy()
    top_k = len(top_concepts)
    fig_w = max(12, 4 * top_k)
    fig, axes = plt.subplots(1, top_k, figsize=(fig_w, 4.5))
    if top_k == 1:
        axes = [axes]
    for i, concept_id in enumerate(top_concepts):
        concept_activations = patches_U[:, concept_id]
        scores = concept_activations * patch_importance_np
        hm = patch_importance_to_heatmap(
            torch.from_numpy(scores.astype(np.float32)),
            num_patches_h,
            num_patches_w,
            img_size=img_size,
        )
        blended = overlay_heatmap(image_pil, hm, alpha)
        pil_img = Image.fromarray(blended)
        draw = ImageDraw.Draw(pil_img)
        outline_color = tuple((colors[i] * 255).astype(int))
        for idx in _greedy_distinct_patch_indices(
            scores,
            num_patches_h,
            num_patches_w,
            max_distinct_patches,
            min_patch_separation,
            floor_score_frac,
        ):
            row = idx // num_patches_w
            col = idx % num_patches_w
            x0 = col * stride
            y0 = row * stride
            draw.rectangle(
                [x0, y0, x0 + patch_size, y0 + patch_size],
                outline=outline_color,
                width=3,
            )
        ax = axes[i]
        ax.imshow(np.asarray(pil_img))
        ax.set_title(
            f"Concept {concept_id}",
            fontsize=14, fontweight="bold",
            color=tuple(colors[i][:3]),
            loc="center",
        )
        ax.axis("off")
    fig.suptitle("Per-concept activation heatmaps with localised regions",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_path = os.path.join(os.getcwd(), f"{out_basename}_concept_heatmaps_multi_boxes.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.25)
    svg_path = out_path.replace(".png", ".svg")
    plt.savefig(svg_path, format="svg", bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"[INFO] Per-concept heatmap row (multi-box) saved to {out_path}  |  {svg_path}")
    return out_path


def save_concept_multi_boxes_on_image_row(
    out_basename: str,
    top_concepts: List[int],
    colors: np.ndarray,
    patches_U: np.ndarray,
    patch_importance: torch.Tensor,
    num_patches_h: int,
    num_patches_w: int,
    image_pil: Image.Image,
    stride: int,
    patch_size: int,
    max_distinct_patches: int,
    min_patch_separation: int,
    floor_score_frac: float,
) -> str:
    """
    One row per top concept: **raw CBM image only** (no heatmap overlay) with the same
    greedy multi-patch boxes as :func:`save_concept_heatmaps_row_multi_boxes`.
    Saves ``{out_basename}_concept_multi_boxes_image.png`` and ``.svg``.
    """
    patch_importance_np = patch_importance.detach().cpu().numpy()
    top_k = len(top_concepts)
    fig_w = max(12, 4 * top_k)
    fig, axes = plt.subplots(1, top_k, figsize=(fig_w, 4.5))
    if top_k == 1:
        axes = [axes]
    for i, concept_id in enumerate(top_concepts):
        concept_activations = patches_U[:, concept_id]
        scores = concept_activations * patch_importance_np
        pil_img = image_pil.copy()
        draw = ImageDraw.Draw(pil_img)
        outline_color = tuple((colors[i] * 255).astype(int))
        for idx in _greedy_distinct_patch_indices(
            scores,
            num_patches_h,
            num_patches_w,
            max_distinct_patches,
            min_patch_separation,
            floor_score_frac,
        ):
            row = idx // num_patches_w
            col = idx % num_patches_w
            x0 = col * stride
            y0 = row * stride
            draw.rectangle(
                [x0, y0, x0 + patch_size, y0 + patch_size],
                outline=outline_color,
                width=3,
            )
        ax = axes[i]
        ax.imshow(np.asarray(pil_img))
        ax.set_title(
            f"Concept {concept_id}",
            fontsize=14, fontweight="bold",
            color=tuple(colors[i][:3]),
            loc="center",
        )
        ax.axis("off")
    fig.suptitle("Most active patches per concept",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_path = os.path.join(os.getcwd(), f"{out_basename}_concept_multi_boxes_image.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.25)
    svg_path = out_path.replace(".png", ".svg")
    plt.savefig(svg_path, format="svg", bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"[INFO] Per-concept multi-box row (image only) saved to {out_path}  |  {svg_path}")
    return out_path


# ---------------------------------------------------------------------------
# Standalone Grad-CAM visualisations
# ---------------------------------------------------------------------------

def gradcam_standalone_spatial(dataset_key: str, image_path: str, device: str,
                               backbone: str, target_class: Optional[int],
                               alpha: float):
    """Standalone spatial Grad-CAM using ImageNet-pretrained ResNet-50."""
    model = load_resnet50(device)
    gradcam = GradCAM(model, target_layer=model.layer4)

    input_tensor, image_pil = prepare_image(image_path, dataset_key, device)
    cam, pred_cls, conf = gradcam.generate(input_tensor, target_class)
    blended = overlay_heatmap(image_pil, cam, alpha)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(np.array(image_pil))
    axes[0].set_title("Original Image", fontsize=13)
    axes[0].axis("off")

    axes[1].imshow(cam, cmap="jet")
    axes[1].set_title("ResNet-50 Grad-CAM (ImageNet)", fontsize=13)
    axes[1].axis("off")

    axes[2].imshow(blended)
    axes[2].set_title(f"Overlay  —  {get_class_label(dataset_key, pred_cls)} ({conf*100:.1f}%)", fontsize=13)
    axes[2].axis("off")

    plt.tight_layout()
    out_path = os.path.join(os.getcwd(), "output_gradcam_spatial.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)
    print(f"[INFO] Spatial Grad-CAM figure saved to {out_path}")
    print(f"ResNet-50 predicted: {get_class_label(dataset_key, pred_cls)} ({conf*100:.2f}%)")


def gradcam_standalone_medical(dataset_key: str, image_path: str, device: str,
                               backbone: str, alpha: float, output_root: str,
                               patch_size: int, stride_r: float,
                               top_k_max: int, min_concept_weight: float):
    """Standalone CBM-GAT Grad-CAM (medical decision) – no concepts."""
    from graph import load_split, infer_dims
    from explain_image import (
        load_craft, load_trained_gat, build_graph_from_single_image,
    )

    # Build CBM-GAT pipeline (same as in comparison)
    train_ds = load_split(output_root, dataset_key, "train", device=device)
    in_dim, num_classes, _ = infer_dims(train_ds)

    craft, _ = load_craft(dataset_key, device, output_root, backbone=backbone)
    gat_model = load_trained_gat(dataset_key, device, output_root, in_dim, num_classes)

    graph, patches_U, image_pil_cbm = build_graph_from_single_image(
        dataset_key, image_path, device, craft, patch_size, stride_r)

    node_f = graph.ndata["feat"].float().to(device).requires_grad_(True)
    logits, _, _ = gat_model(graph, node_f)
    probs = F.softmax(logits[0], dim=0)
    pred_idx = int(torch.argmax(probs).item())
    pred_conf = float(probs[pred_idx].item())

    target_prob = probs[pred_idx]
    grads = torch.autograd.grad(target_prob, node_f, create_graph=False)[0]
    node_importance = grads.abs().sum(dim=1)
    node_importance = node_importance / (node_importance.sum() + 1e-8)

    U = torch.tensor(patches_U, device=node_importance.device, dtype=node_importance.dtype)
    patch_importance = torch.matmul(U, node_importance)
    patch_importance = patch_importance / (patch_importance.sum() + 1e-8)

    stride = int(patch_size * stride_r)
    num_patches_w = (image_pil_cbm.width - patch_size) // stride + 1
    num_patches_h = (image_pil_cbm.height - patch_size) // stride + 1

    medical_heatmap = patch_importance_to_heatmap(
        patch_importance=patch_importance,
        num_patches_h=num_patches_h,
        num_patches_w=num_patches_w,
        img_size=image_pil_cbm.width,
    )
    blended_medical = overlay_heatmap(image_pil_cbm, medical_heatmap, alpha)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(np.array(image_pil_cbm))
    axes[0].set_title("Original Image", fontsize=13)
    axes[0].axis("off")

    axes[1].imshow(medical_heatmap, cmap="jet")
    axes[1].set_title("CBM-GAT Grad-CAM (medical)", fontsize=13)
    axes[1].axis("off")

    axes[2].imshow(blended_medical)
    axes[2].set_title(f"Overlay  —  GAT: {get_class_label(dataset_key, pred_idx)} ({pred_conf*100:.1f}%)", fontsize=13)
    axes[2].axis("off")

    plt.tight_layout()
    out_path = os.path.join(os.getcwd(), "output_gradcam_medical.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)
    print(f"[INFO] Medical Grad-CAM figure saved to {out_path}")
    print(f"GAT predicted: {get_class_label(dataset_key, pred_idx)} ({pred_conf*100:.2f}%)")


def _get_gradcam_target_layer(model: nn.Module, backbone: str) -> nn.Module:
    """Return the last convolutional block for Grad-CAM given the backbone name."""
    if backbone == "resnet50":
        return model.layer4
    elif backbone == "densenet201":
        return model.features.denseblock4
    elif backbone == "mobilenet_v2":
        return model.features[-1]
    else:
        raise ValueError(f"No Grad-CAM target layer defined for backbone: {backbone!r}")


def gradcam_standalone_cnn(dataset_key: str, image_path: str, device: str,
                           backbone: str, target_class: Optional[int],
                           alpha: float, output_root: str):
    """Standalone Grad-CAM using the fine-tuned CNN baseline (auto-loads checkpoint)."""
    ckpt_path = os.path.join(
        output_root, dataset_key, "models_cnn", dataset_key,
        f"{dataset_key}_{backbone}_cnn.pt",
    )
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"CNN baseline checkpoint not found at {ckpt_path}. "
            "Run train_cnn.py first."
        )

    model = load_resnet50_finetuned(ckpt_path, device)
    # Backbone name is read back from the checkpoint by load_resnet50_finetuned;
    # pass it here explicitly so the target layer selection is consistent.
    actual_backbone = torch.load(ckpt_path, map_location="cpu",
                                 weights_only=False).get("backbone", "resnet50")
    gradcam = GradCAM(model, target_layer=_get_gradcam_target_layer(model, actual_backbone))

    input_tensor, image_pil = prepare_image(image_path, dataset_key, device)
    cam, pred_cls, conf = gradcam.generate(input_tensor, target_class)
    blended = overlay_heatmap(image_pil, cam, alpha)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(np.array(image_pil))
    axes[0].set_title("Original Image", fontsize=13)
    axes[0].axis("off")

    axes[1].imshow(cam, cmap="jet")
    axes[1].set_title(f"{actual_backbone} Grad-CAM (fine-tuned)", fontsize=13)
    axes[1].axis("off")

    axes[2].imshow(blended)
    axes[2].set_title(f"Overlay  —  {get_class_label(dataset_key, pred_cls)} ({conf*100:.1f}%)", fontsize=13)
    axes[2].axis("off")

    plt.tight_layout()
    out_path = os.path.join(os.getcwd(), "output_gradcam_cnn.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)
    print(f"[INFO] CNN baseline Grad-CAM figure saved to {out_path}")
    print(f"Fine-tuned ResNet-50 predicted: {get_class_label(dataset_key, pred_cls)} ({conf*100:.2f}%)")


# ---------------------------------------------------------------------------
# Caption helpers
# ---------------------------------------------------------------------------

def _disp_label(dataset_key: str, idx: int) -> str:
    """Short display label — removes verbose prefixes for cleaner captions."""
    raw = get_class_label(dataset_key, idx)
    raw = raw.replace("Atypical/", "").replace("Common ", "")
    return raw


def _label_color(label: str) -> str:
    """Green for benign/nevus classes, red for malignant/melanoma classes."""
    l = label.lower()
    if any(w in l for w in ("nevus", "benign", "melanocytic", "normal")):
        return "#2ecc71"   # green
    return "#e74c3c"       # red


def _fmt_backbone(backbone: str) -> str:
    """Human-readable backbone name: resnet50 -> ResNet-50, etc."""
    mapping = {
        "resnet50":     "ResNet-50",
        "densenet201":  "DenseNet-201",
        "mobilenet_v2": "MobileNet-V2",
    }
    return mapping.get(backbone.lower(), backbone)


def _two_tone_caption(ax, y: float, prefix: str, label: str, label_color: str,
                      conf: Optional[float] = None, fontsize: int = 11):
    """
    Render a single-line caption:  "prefix  Label  (xx.x%)"
    where the prefix and percentage are black and the class label is colored.

    All three segments meet on the same horizontal line:
      • "prefix " — black, right-aligned at x=0.5
      • "Label"   — colored, left-aligned at x=0.5 (immediately after prefix)
      • " (xx.x%)"— black, left-aligned at x = 0.5 + estimated label width

    The label width is estimated from its character count so the percentage
    sits flush after the label without needing a figure renderer.
    """
    ax.text(0.5, y, f"{prefix} ",
            ha="right", va="top", fontsize=fontsize, fontweight="bold",
            color="black", transform=ax.transAxes)
    ax.text(0.5, y, label,
            ha="left", va="top", fontsize=fontsize, fontweight="bold",
            color=label_color, transform=ax.transAxes)
    if conf is not None:
        # Approximate axes-fraction width of one character at this fontsize.
        # 0.00164 * fontsize ≈ 0.018 at fontsize=11, which is a good match for
        # typical subplot widths (≈ 5–6 inches) and the default sans-serif font.
        char_w = fontsize * 0.0026
        x_pct = 0.5 + (len(label) + 2.0) * char_w
        ax.text(x_pct, y, f"({conf*100:.1f}%)",
                ha="left", va="top", fontsize=fontsize, fontweight="bold",
                color="black", transform=ax.transAxes)


# ---------------------------------------------------------------------------
# Comparison modes
# ---------------------------------------------------------------------------

def _compute_cbm_concepts(dataset_key: str, image_path: str, device: str,
                          backbone: str, output_root: str,
                          patch_size: int, stride_r: float,
                          top_k_max: int, min_concept_weight: float):
    """
    Shared CBM-GAT pipeline returning everything needed for concept visualisation.
    """
    from graph import load_split, infer_dims
    from explain_image import (
        load_craft, load_trained_gat, build_graph_from_single_image,
        argmax_safe,
    )

    train_ds = load_split(output_root, dataset_key, "train", device=device)
    in_dim, num_classes, _ = infer_dims(train_ds)

    craft, craft_dir = load_craft(dataset_key, device, output_root, backbone=backbone)
    gat_model = load_trained_gat(dataset_key, device, output_root, in_dim, num_classes)

    graph, patches_U, image_pil_cbm = build_graph_from_single_image(
        dataset_key, image_path, device, craft, patch_size, stride_r)

    node_f = graph.ndata["feat"].float().to(device).requires_grad_(True)
    logits, _, h = gat_model(graph, node_f)
    probs = F.softmax(logits[0], dim=0)
    pred_idx = int(torch.argmax(probs).item())
    pred_conf = float(probs[pred_idx].item())

    target_prob = probs[pred_idx]
    grads = torch.autograd.grad(target_prob, node_f, create_graph=False)[0]
    node_importance = grads.abs().sum(dim=1)
    node_importance = node_importance / (node_importance.sum() + 1e-8)

    sorted_values, sorted_indices = torch.sort(node_importance, descending=True)
    concept_importance_values = sorted_values.tolist()
    concept_ranking = sorted_indices.tolist()

    top_vals = [v for v in concept_importance_values if v > min_concept_weight]
    top_k = min(len(top_vals), top_k_max) if top_vals else min(top_k_max, len(concept_ranking))
    top_concepts = concept_ranking[:top_k]
    top_values = concept_importance_values[:top_k]

    U = torch.tensor(patches_U, device=node_importance.device, dtype=node_importance.dtype)
    patch_importance = torch.matmul(U, node_importance)
    patch_importance = patch_importance / (patch_importance.sum() + 1e-8)
    _, sorted_patch_idx = torch.sort(patch_importance, descending=True)
    sorted_patch_idx = sorted_patch_idx.tolist()

    patches_C = argmax_safe(patches_U, top_concepts)

    bar_colors = plt.cm.tab10(np.arange(10))
    # top_concepts are graph node ids, not 0..K-1; color by rank in the top-k list.
    colors = bar_colors[np.arange(len(top_concepts)) % 10]

    stride = int(patch_size * stride_r)
    num_patches_w = (image_pil_cbm.width - patch_size) // stride + 1
    num_patches_h = (image_pil_cbm.height - patch_size) // stride + 1

    return (image_pil_cbm, craft_dir, gat_model,
            node_importance, patch_importance, patches_U,
            top_concepts, top_values, sorted_patch_idx, patches_C,
            colors, stride, num_patches_w, num_patches_h,
            pred_idx, pred_conf)


def gradcam_vs_concepts_medical(dataset_key: str, image_path: str, device: str,
                                backbone: str, alpha: float, output_root: str,
                                patch_size: int, stride_r: float,
                                top_k_max: int, min_concept_weight: float,
                                *,
                                true_class: int = -1,
                                save_concept_heatmaps: bool = False,
                                save_concept_heatmap_multi_boxes: bool = False,
                                save_concept_multi_boxes_on_image: bool = False,
                                max_distinct_patches_per_concept: int = 4,
                                min_patch_separation: int = 3,
                                floor_score_frac: float = 0.15):
    """
    Left half  : CBM-GAT spatial heatmap (medical decision) + prediction
    Right half : CBM-GAT concept patches, importance bars, concept examples
    """
    (image_pil_cbm, craft_dir, gat_model,
     node_importance, patch_importance, patches_U,
     top_concepts, top_values, sorted_patch_idx, patches_C,
     colors, stride, num_patches_w, num_patches_h,
     pred_idx, pred_conf) = _compute_cbm_concepts(
        dataset_key, image_path, device, backbone, output_root,
        patch_size, stride_r, top_k_max, min_concept_weight
    )

    top_k = len(top_concepts)

    # ---- CBM-GAT spatial heatmap (medical decision) ----
    medical_heatmap = patch_importance_to_heatmap(
        patch_importance=patch_importance,
        num_patches_h=num_patches_h,
        num_patches_w=num_patches_w,
        img_size=image_pil_cbm.width,
    )
    blended_medical = overlay_heatmap(image_pil_cbm, medical_heatmap, alpha)

    patch_importance_np = patch_importance.detach().cpu().numpy()
    image_pil_cbm_annot = annotate_concept_patch_panel(
        image_pil_cbm,
        patches_U,
        patch_importance_np,
        top_concepts,
        colors,
        stride,
        num_patches_w,
        patch_size,
        sorted_patch_idx,
        top_k,
    )

    # ---- Caption helpers ----
    pred_short  = _disp_label(dataset_key, pred_idx)
    pred_color  = _label_color(pred_short)
    show_gt     = true_class >= 0
    if show_gt:
        gt_short = _disp_label(dataset_key, true_class)
        gt_color = _label_color(gt_short)

    def _draw_caption_gt(ax):
        """Render Ground-truth-only caption under panel (a)."""
        ax.axis("off")
        if show_gt:
            _two_tone_caption(ax, 0.80, "Ground truth:", gt_short, gt_color)

    def _draw_caption_pred(ax):
        """Render Predicted-only caption under panel (b)."""
        ax.axis("off")
        _two_tone_caption(ax, 0.80, "Predicted:", pred_short, pred_color,
                          conf=pred_conf)

    # ---- Build the 3-panel main figure (spatial decision map removed) ----
    # 4-column outer layout: [patches | spacer | bars | exemplars]
    # Each real column uses a 3-row sub-gridspec: [title | content | caption]
    cap_ratio = 3
    fig = plt.figure(figsize=(14, 6))
    outer_gs = gridspec.GridSpec(
        1, 4,
        width_ratios=[1, 0.10, 1, 1],
        wspace=0.12,
        left=0.02, right=0.98, top=0.94, bottom=0.04,
    )

    def _col_subgs(col_idx):
        return gridspec.GridSpecFromSubplotSpec(
            3, 1, subplot_spec=outer_gs[0, col_idx],
            height_ratios=[2, 28, cap_ratio], hspace=0.04)

    # ---- Panel (a) — concept localisation ----
    gs_a = _col_subgs(0)
    ax_a_title = fig.add_subplot(gs_a[0, 0]); ax_a_title.axis("off")
    ax_a_title.text(0.5, 0.5, "Concept Localisation",
                    ha="center", va="center", fontsize=13, fontweight="bold",
                    transform=ax_a_title.transAxes)
    ax_cbm = fig.add_subplot(gs_a[1, 0])
    ax_cbm.imshow(np.array(image_pil_cbm_annot)); ax_cbm.axis("off")
    _draw_caption_gt(fig.add_subplot(gs_a[2, 0]))

    # ---- Panel (b) — concept importance vertical bars ----
    gs_b = _col_subgs(2)
    ax_b_title = fig.add_subplot(gs_b[0, 0]); ax_b_title.axis("off")
    ax_b_title.text(0.5, 0.5, "Top 3 Concept Activations",
                    ha="center", va="center", fontsize=13, fontweight="bold",
                    transform=ax_b_title.transAxes)
    ax_bar = fig.add_subplot(gs_b[1, 0])
    x_pos = np.arange(top_k)
    bars = ax_bar.bar(x_pos, top_values, color=colors[:top_k], align="center", width=0.55)
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels([f"Concept {c}" for c in top_concepts], fontsize=11)
    ax_bar.set_ylabel("Gradient importance weight", fontsize=12)
    ax_bar.set_box_aspect(1)
    if top_k:
        ax_bar.set_ylim(0, max(top_values) * 1.25)
        for i, b in enumerate(bars):
            ax_bar.text(b.get_x() + b.get_width() / 2,
                        b.get_height() + max(top_values) * 0.015,
                        f"{top_values[i]:.2f}",
                        ha="center", va="bottom", fontsize=12, fontweight="bold")
    _draw_caption_pred(fig.add_subplot(gs_b[2, 0]))

    # ---- Panel (c) — concept exemplar thumbnails ----
    gs_c = _col_subgs(3)
    ax_c_title = fig.add_subplot(gs_c[0, 0]); ax_c_title.axis("off")
    ax_c_title.text(0.5, 0.5, "Top 3 Concept Exemplars",
                    ha="center", va="center", fontsize=13, fontweight="bold",
                    transform=ax_c_title.transAxes)
    right_gs = gridspec.GridSpecFromSubplotSpec(
        top_k, 2, subplot_spec=gs_c[1, 0],
        width_ratios=[0.1, 1.0], wspace=0.0, hspace=0.12)
    for i in range(top_k):
        concept_id = top_concepts[i]
        c_color = colors[i]
        ax_c = fig.add_subplot(right_gs[i, 1])
        ax_c.axis("off")
        thumb = os.path.join(craft_dir, "concept_examples", f"concept_{concept_id}.png")
        if os.path.isfile(thumb):
            im = Image.open(thumb).convert("RGB")
            ax_c.imshow(im)
            ax_c.add_patch(mpatches.Rectangle(
                (0, 0), 1, 1, transform=ax_c.transAxes,
                linewidth=8, edgecolor=c_color, facecolor="none"))
        else:
            ax_c.text(0.5, 0.5, f"(no example for {concept_id})",
                      ha="center", va="center", fontsize=11)
        ax_c.text(0.5, 1.07, f"Concept {concept_id}",
                  ha="center", va="bottom", fontsize=12, fontweight="bold",
                  color=c_color, transform=ax_c.transAxes)
        ax_c.text(0.5, -0.05, f"Importance: {top_values[i]*100:.1f}%",
                  ha="center", va="top", fontsize=11, transform=ax_c.transAxes)
    fig.add_subplot(gs_c[2, 0]).axis("off")

    out_path = os.path.join(os.getcwd(), "output_gradcam_vs_concepts_medical.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.2)
    svg_path = out_path.replace(".png", ".svg")
    plt.savefig(svg_path, format="svg", bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    print(f"[INFO] 3-panel concept figure saved to {out_path}")

    if save_concept_heatmaps:
        save_concept_heatmaps_row(
            "output_gradcam_vs_concepts_medical",
            top_concepts,
            colors,
            patches_U,
            patch_importance,
            num_patches_h,
            num_patches_w,
            img_size=image_pil_cbm.width,
            image_pil=image_pil_cbm,
            alpha=alpha,
        )
    if save_concept_heatmap_multi_boxes:
        save_concept_heatmaps_row_multi_boxes(
            "output_gradcam_vs_concepts_medical",
            top_concepts,
            colors,
            patches_U,
            patch_importance,
            num_patches_h,
            num_patches_w,
            img_size=image_pil_cbm.width,
            image_pil=image_pil_cbm,
            alpha=alpha,
            stride=stride,
            patch_size=patch_size,
            max_distinct_patches=max_distinct_patches_per_concept,
            min_patch_separation=min_patch_separation,
            floor_score_frac=floor_score_frac,
        )
    if save_concept_multi_boxes_on_image:
        save_concept_multi_boxes_on_image_row(
            "output_gradcam_vs_concepts_medical",
            top_concepts,
            colors,
            patches_U,
            patch_importance,
            num_patches_h,
            num_patches_w,
            image_pil=image_pil_cbm,
            stride=stride,
            patch_size=patch_size,
            max_distinct_patches=max_distinct_patches_per_concept,
            min_patch_separation=min_patch_separation,
            floor_score_frac=floor_score_frac,
        )
    print(f"CBM-GAT   -> {get_class_label(dataset_key, pred_idx)} ({pred_conf*100:.2f}%)")
    print(f"Top concepts: {', '.join(str(c) for c in top_concepts)}")


def gradcam_vs_concepts_spatial(dataset_key: str, image_path: str, device: str,
                                backbone: str, target_class: Optional[int],
                                alpha: float, output_root: str,
                                patch_size: int, stride_r: float,
                                top_k_max: int, min_concept_weight: float,
                                *,
                                save_concept_heatmaps: bool = False,
                                save_concept_heatmap_multi_boxes: bool = False,
                                save_concept_multi_boxes_on_image: bool = False,
                                max_distinct_patches_per_concept: int = 4,
                                min_patch_separation: int = 3,
                                floor_score_frac: float = 0.15):
    """
    Left half  : ImageNet-pretrained ResNet-50 spatial Grad-CAM
    Right half : CBM-GAT concept patches, importance bars, concept examples
    """
    # Left: spatial Grad-CAM (ImageNet)
    model = load_resnet50(device)
    gradcam = GradCAM(model, target_layer=model.layer4)
    input_tensor, image_pil_gc = prepare_image(image_path, dataset_key, device)
    cam, gc_cls, gc_conf = gradcam.generate(input_tensor, target_class)
    blended_spatial = overlay_heatmap(image_pil_gc, cam, alpha)

    # Right: CBM-GAT concepts (no medical heatmap)
    (image_pil_cbm, craft_dir, gat_model,
     node_importance, patch_importance, patches_U,
     top_concepts, top_values, sorted_patch_idx, patches_C,
     colors, stride, num_patches_w, num_patches_h,
     pred_idx, pred_conf) = _compute_cbm_concepts(
        dataset_key, image_path, device, backbone, output_root,
        patch_size, stride_r, top_k_max, min_concept_weight
    )

    top_k = len(top_concepts)
    patch_importance_np = patch_importance.detach().cpu().numpy()
    image_pil_cbm_annot = annotate_concept_patch_panel(
        image_pil_cbm,
        patches_U,
        patch_importance_np,
        top_concepts,
        colors,
        stride,
        num_patches_w,
        patch_size,
        sorted_patch_idx,
        top_k,
    )

    # Build figure: [spatial Grad-CAM | CBM-GAT patches | concept bars | examples]
    fig = plt.figure(figsize=(32, 8), constrained_layout=True)
    outer_gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 1])

    # Title for rightmost column (training patches for top concepts)
    strips_title_ax = fig.add_subplot(outer_gs[0, 3])
    strips_title_ax.axis("off")
    strips_title_ax.text(
        0.5, 1.0,
        "Top 3 Concept Image Patches",
        ha="center", va="bottom",
        fontsize=13, fontweight="bold",
        transform=strips_title_ax.transAxes,
    )

    # Panel 1 – spatial Grad-CAM
    gc_gs = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=outer_gs[0, 0], height_ratios=[30, 2], hspace=0.1)
    ax_gc = fig.add_subplot(gc_gs[0, 0])
    ax_gc.imshow(blended_spatial)
    ax_gc.set_title("ResNet-50 Grad-CAM (ImageNet)", fontsize=13, fontweight="bold")
    ax_gc.axis("off")
    ax_gc_cap = fig.add_subplot(gc_gs[1, 0])
    ax_gc_cap.axis("off")
    ax_gc_cap.text(
        0.5, 0.5,
        f"ResNet-50 prediction: {get_class_label(dataset_key, gc_cls)} ({gc_conf*100:.1f}%)",
        ha="center", va="center",
        fontsize=12,
        fontweight="bold",
    )

    # Panel 2 – CBM-GAT concept patches
    cbm_gs = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=outer_gs[0, 1], height_ratios=[30, 2], hspace=0.1)
    ax_cbm = fig.add_subplot(cbm_gs[0, 0])
    ax_cbm.imshow(np.array(image_pil_cbm_annot))
    ax_cbm.set_title("CBM-GAT (concept patches)", fontsize=13, fontweight="bold")
    ax_cbm.axis("off")
    ax_cbm_cap = fig.add_subplot(cbm_gs[1, 0])
    ax_cbm_cap.axis("off")
    ax_cbm_cap.text(
        0.5, 0.5,
        f"GAT prediction: {get_class_label(dataset_key, pred_idx)} ({pred_conf*100:.1f}%)",
        ha="center", va="center",
        fontsize=12,
        fontweight="bold",
    )

    # Panel 3 – concept importance bars (aligned vertically with other panels)
    bar_gs = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=outer_gs[0, 2], height_ratios=[30, 2], hspace=0.1
    )
    ax_bar = fig.add_subplot(bar_gs[0, 0])
    y_pos = np.arange(top_k)
    bars = ax_bar.barh(y_pos, top_values, color=colors[:top_k], align="center")
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels([f"Concept {c}" for c in top_concepts])
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("Importance", fontsize=12)
    ax_bar.xaxis.labelpad = 14
    ax_bar.set_title(f"Top {top_k} Concept Activations", fontsize=13, fontweight="bold")
    if top_k:
        ax_bar.set_xlim(0, max(top_values) * 1.3)
        for i, b in enumerate(bars):
            ax_bar.text(b.get_width() + max(top_values) * 0.01,
                        b.get_y() + b.get_height() / 2,
                        f"{top_values[i]:.3f}", va="center", fontsize=10)

    # small caption axis for alignment (can remain empty)
    ax_bar_cap = fig.add_subplot(bar_gs[1, 0])
    ax_bar_cap.axis("off")

    # Panel 4 – concept example thumbnails
    right_gs = gridspec.GridSpecFromSubplotSpec(
        top_k, 2, subplot_spec=outer_gs[0, 3],
        width_ratios=[0.1, 1.0], wspace=0.0, hspace=0.10)
    for i in range(top_k):
        concept_id = top_concepts[i]
        c_color = colors[i]

        ax_c = fig.add_subplot(right_gs[i, 1])
        ax_c.axis("off")
        thumb = os.path.join(craft_dir, "concept_examples", f"concept_{concept_id}.png")
        if os.path.isfile(thumb):
            im = Image.open(thumb).convert("RGB")
            ax_c.imshow(im)
            border = mpatches.Rectangle(
                (0, 0), 1, 1, transform=ax_c.transAxes,
                linewidth=8, edgecolor=c_color, facecolor="none")
            ax_c.add_patch(border)
        else:
            ax_c.text(0.5, 0.5, f"(no example for {concept_id})",
                      ha="center", va="center", fontsize=11)

        # concept label above the strip
        ax_c.text(
            0.5, 1.07,
            f"Concept {concept_id}",
            ha="center", va="bottom",
            fontsize=12,
            fontweight="bold",
            color=c_color,
            transform=ax_c.transAxes,
        )

        # importance percentage below the strip
        ax_c.text(
            0.5, -0.05,
            f"Importance: {top_values[i]*100:.1f}%",
            ha="center", va="top",
            fontsize=10,
            transform=ax_c.transAxes,
        )

    out_path = os.path.join(os.getcwd(), "output_gradcam_vs_concepts_spatial.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    print(f"[INFO] Spatial comparison figure saved to {out_path}")
    if save_concept_heatmaps:
        save_concept_heatmaps_row(
            "output_gradcam_vs_concepts_spatial",
            top_concepts,
            colors,
            patches_U,
            patch_importance,
            num_patches_h,
            num_patches_w,
            img_size=image_pil_cbm.width,
            image_pil=image_pil_cbm,
            alpha=alpha,
        )
    if save_concept_heatmap_multi_boxes:
        save_concept_heatmaps_row_multi_boxes(
            "output_gradcam_vs_concepts_spatial",
            top_concepts,
            colors,
            patches_U,
            patch_importance,
            num_patches_h,
            num_patches_w,
            img_size=image_pil_cbm.width,
            image_pil=image_pil_cbm,
            alpha=alpha,
            stride=stride,
            patch_size=patch_size,
            max_distinct_patches=max_distinct_patches_per_concept,
            min_patch_separation=min_patch_separation,
            floor_score_frac=floor_score_frac,
        )
    if save_concept_multi_boxes_on_image:
        save_concept_multi_boxes_on_image_row(
            "output_gradcam_vs_concepts_spatial",
            top_concepts,
            colors,
            patches_U,
            patch_importance,
            num_patches_h,
            num_patches_w,
            image_pil=image_pil_cbm,
            stride=stride,
            patch_size=patch_size,
            max_distinct_patches=max_distinct_patches_per_concept,
            min_patch_separation=min_patch_separation,
            floor_score_frac=floor_score_frac,
        )
    print(f"Fine-tuned CNN -> {get_class_label(dataset_key, gc_cls)} ({gc_conf*100:.2f}%)")
    print(f"CBM-GAT        -> {get_class_label(dataset_key, pred_idx)} ({pred_conf*100:.2f}%)")
    print(f"Top concepts: {', '.join(str(c) for c in top_concepts)}")


def gradcam_vs_concepts_cnn(dataset_key: str, image_path: str, device: str,
                            backbone: str, target_class: Optional[int],
                            alpha: float, output_root: str,
                            patch_size: int, stride_r: float,
                            top_k_max: int, min_concept_weight: float,
                            *,
                            save_concept_heatmaps: bool = False,
                            save_concept_heatmap_multi_boxes: bool = False,
                            save_concept_multi_boxes_on_image: bool = False,
                            max_distinct_patches_per_concept: int = 4,
                            min_patch_separation: int = 3,
                            floor_score_frac: float = 0.15):
    """
    Left half  : Fine-tuned CNN Grad-CAM (backbone determined by --backbone arg)
    Right half : CBM-GAT concept patches, importance bars, concept examples
    """
    # Left: fine-tuned CNN Grad-CAM
    ckpt_path = os.path.join(
        output_root, dataset_key, "models_cnn", dataset_key,
        f"{dataset_key}_{backbone}_cnn.pt",
    )
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"CNN baseline checkpoint not found at {ckpt_path}. "
            "Run train_cnn.py first."
        )

    model = load_resnet50_finetuned(ckpt_path, device)
    actual_backbone = torch.load(ckpt_path, map_location="cpu",
                                 weights_only=False).get("backbone", "resnet50")
    gradcam = GradCAM(model, target_layer=_get_gradcam_target_layer(model, actual_backbone))
    input_tensor, image_pil_gc = prepare_image(image_path, dataset_key, device)
    cam, gc_cls, gc_conf = gradcam.generate(input_tensor, target_class)
    blended_cnn = overlay_heatmap(image_pil_gc, cam, alpha)

    # Right: CBM-GAT concepts
    (image_pil_cbm, craft_dir, gat_model,
     node_importance, patch_importance, patches_U,
     top_concepts, top_values, sorted_patch_idx, patches_C,
     colors, stride, num_patches_w, num_patches_h,
     pred_idx, pred_conf) = _compute_cbm_concepts(
        dataset_key, image_path, device, backbone, output_root,
        patch_size, stride_r, top_k_max, min_concept_weight
    )

    top_k = len(top_concepts)
    patch_importance_np = patch_importance.detach().cpu().numpy()
    image_pil_cbm_annot = annotate_concept_patch_panel(
        image_pil_cbm,
        patches_U,
        patch_importance_np,
        top_concepts,
        colors,
        stride,
        num_patches_w,
        patch_size,
        sorted_patch_idx,
        top_k,
    )

    # Build figure: [CNN Grad-CAM | CBM-GAT patches | concept bars | examples]
    fig = plt.figure(figsize=(32, 8), constrained_layout=True)
    outer_gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 1])

    strips_title_ax = fig.add_subplot(outer_gs[0, 3])
    strips_title_ax.axis("off")
    strips_title_ax.text(
        0.5, 1.0,
        "Top 3 Concept Image Patches",
        ha="center", va="bottom",
        fontsize=13, fontweight="bold",
        transform=strips_title_ax.transAxes,
    )

    # Panel 1 – fine-tuned CNN Grad-CAM
    gc_gs = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=outer_gs[0, 0], height_ratios=[30, 2], hspace=0.1)
    ax_gc = fig.add_subplot(gc_gs[0, 0])
    ax_gc.imshow(blended_cnn)
    ax_gc.set_title("ResNet-50 Grad-CAM (fine-tuned)", fontsize=13, fontweight="bold")
    ax_gc.axis("off")
    ax_gc_cap = fig.add_subplot(gc_gs[1, 0])
    ax_gc_cap.axis("off")
    ax_gc_cap.text(
        0.5, 0.5,
        f"CNN prediction: {get_class_label(dataset_key, gc_cls)} ({gc_conf*100:.1f}%)",
        ha="center", va="center",
        fontsize=12,
        fontweight="bold",
    )

    # Panel 2 – CBM-GAT concept patches
    cbm_gs = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=outer_gs[0, 1], height_ratios=[30, 2], hspace=0.1)
    ax_cbm = fig.add_subplot(cbm_gs[0, 0])
    ax_cbm.imshow(np.array(image_pil_cbm_annot))
    ax_cbm.set_title("CBM-GAT (concept patches)", fontsize=13, fontweight="bold")
    ax_cbm.axis("off")
    ax_cbm_cap = fig.add_subplot(cbm_gs[1, 0])
    ax_cbm_cap.axis("off")
    ax_cbm_cap.text(
        0.5, 0.5,
        f"GAT prediction: {get_class_label(dataset_key, pred_idx)} ({pred_conf*100:.1f}%)",
        ha="center", va="center",
        fontsize=12,
        fontweight="bold",
    )

    # Panel 3 – concept importance bars
    bar_gs = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=outer_gs[0, 2], height_ratios=[30, 2], hspace=0.1
    )
    ax_bar = fig.add_subplot(bar_gs[0, 0])
    y_pos = np.arange(top_k)
    bars = ax_bar.barh(y_pos, top_values, color=colors[:top_k], align="center")
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels([f"Concept {c}" for c in top_concepts])
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("Importance", fontsize=12)
    ax_bar.xaxis.labelpad = 14
    ax_bar.set_title(f"Top {top_k} Concept Activations", fontsize=13, fontweight="bold")
    if top_k:
        ax_bar.set_xlim(0, max(top_values) * 1.3)
        for i, b in enumerate(bars):
            ax_bar.text(b.get_width() + max(top_values) * 0.01,
                        b.get_y() + b.get_height() / 2,
                        f"{top_values[i]:.3f}", va="center", fontsize=10)

    ax_bar_cap = fig.add_subplot(bar_gs[1, 0])
    ax_bar_cap.axis("off")

    # Panel 4 – concept example thumbnails
    right_gs = gridspec.GridSpecFromSubplotSpec(
        top_k, 2, subplot_spec=outer_gs[0, 3],
        width_ratios=[0.1, 1.0], wspace=0.0, hspace=0.10)
    for i in range(top_k):
        concept_id = top_concepts[i]
        c_color = colors[i]

        ax_c = fig.add_subplot(right_gs[i, 1])
        ax_c.axis("off")
        thumb = os.path.join(craft_dir, "concept_examples", f"concept_{concept_id}.png")
        if os.path.isfile(thumb):
            im = Image.open(thumb).convert("RGB")
            ax_c.imshow(im)
            border = mpatches.Rectangle(
                (0, 0), 1, 1, transform=ax_c.transAxes,
                linewidth=8, edgecolor=c_color, facecolor="none")
            ax_c.add_patch(border)
        else:
            ax_c.text(0.5, 0.5, f"(no example for {concept_id})",
                      ha="center", va="center", fontsize=11)

        ax_c.text(
            0.5, 1.07,
            f"Concept {concept_id}",
            ha="center", va="bottom",
            fontsize=12,
            fontweight="bold",
            color=c_color,
            transform=ax_c.transAxes,
        )

        ax_c.text(
            0.5, -0.05,
            f"Importance: {top_values[i]*100:.1f}%",
            ha="center", va="top",
            fontsize=10,
            transform=ax_c.transAxes,
        )

    out_path = os.path.join(os.getcwd(), "output_gradcam_vs_concepts_cnn.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    print(f"[INFO] CNN vs concepts comparison figure saved to {out_path}")
    if save_concept_heatmaps:
        save_concept_heatmaps_row(
            "output_gradcam_vs_concepts_cnn",
            top_concepts,
            colors,
            patches_U,
            patch_importance,
            num_patches_h,
            num_patches_w,
            img_size=image_pil_cbm.width,
            image_pil=image_pil_cbm,
            alpha=alpha,
        )
    if save_concept_heatmap_multi_boxes:
        save_concept_heatmaps_row_multi_boxes(
            "output_gradcam_vs_concepts_cnn",
            top_concepts,
            colors,
            patches_U,
            patch_importance,
            num_patches_h,
            num_patches_w,
            img_size=image_pil_cbm.width,
            image_pil=image_pil_cbm,
            alpha=alpha,
            stride=stride,
            patch_size=patch_size,
            max_distinct_patches=max_distinct_patches_per_concept,
            min_patch_separation=min_patch_separation,
            floor_score_frac=floor_score_frac,
        )
    if save_concept_multi_boxes_on_image:
        save_concept_multi_boxes_on_image_row(
            "output_gradcam_vs_concepts_cnn",
            top_concepts,
            colors,
            patches_U,
            patch_importance,
            num_patches_h,
            num_patches_w,
            image_pil=image_pil_cbm,
            stride=stride,
            patch_size=patch_size,
            max_distinct_patches=max_distinct_patches_per_concept,
            min_patch_separation=min_patch_separation,
            floor_score_frac=floor_score_frac,
        )
    print(f"Fine-tuned CNN -> {get_class_label(dataset_key, gc_cls)} ({gc_conf*100:.2f}%)")
    print(f"CBM-GAT        -> {get_class_label(dataset_key, pred_idx)} ({pred_conf*100:.2f}%)")
    print(f"Top concepts: {', '.join(str(c) for c in top_concepts)}")


# ---------------------------------------------------------------------------
# Figure B — compare_heatmaps: CNN Grad-CAM vs GCBM spatial map
# ---------------------------------------------------------------------------

def gradcam_compare_heatmaps(
    dataset_key: str,
    image_path: str,
    device: str,
    backbone: str,
    target_class: Optional[int],
    alpha: float,
    output_root: str,
    cnn_root: str,
    patch_size: int,
    stride_r: float,
    top_k_max: int,
    min_concept_weight: float,
    *,
    true_class: int = -1,
):
    """
    3-panel comparison figure for paper Figure B:

      (a) Original image
      (b) Fine-tuned CNN Grad-CAM heatmap (backbone-matched checkpoint)
      (c) GCBM spatial decision map (patch-importance heatmap)

    Saved to ``output_heatmap_comparison.png``.

    Parameters
    ----------
    cnn_root : str
        Root dir that contains the fine-tuned CNN checkpoint at the path
        ``<cnn_root>/<dataset_key>/models_cnn/<dataset_key>/<dataset_key>_<backbone>_cnn.pt``.
        Defaults to ``output_root`` when not specified.
    """
    from graph import load_split, infer_dims
    from explain_image import (
        load_craft, load_trained_gat, build_graph_from_single_image,
    )

    # ── CNN Grad-CAM ─────────────────────────────────────────────────────────
    ckpt_path = os.path.join(
        cnn_root, dataset_key, "models_cnn", dataset_key,
        f"{dataset_key}_{backbone}_cnn.pt",
    )
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"CNN checkpoint not found: {ckpt_path}\n"
            "Pass --cnn-root pointing to the run that has the fine-tuned CNN, "
            "or use --file-root if the CNN lives in the same run as the GCBM."
        )
    cnn_model = load_resnet50_finetuned(ckpt_path, device)
    actual_backbone = torch.load(ckpt_path, map_location="cpu",
                                 weights_only=False).get("backbone", backbone)
    cnn_gradcam = GradCAM(cnn_model, _get_gradcam_target_layer(cnn_model, actual_backbone))
    input_tensor, image_pil = prepare_image(image_path, dataset_key, device)
    cnn_cam, cnn_cls, cnn_conf = cnn_gradcam.generate(input_tensor, target_class)
    cnn_blended = overlay_heatmap(image_pil, cnn_cam, alpha)

    # ── GCBM spatial decision map ─────────────────────────────────────────────
    train_ds = load_split(output_root, dataset_key, "train", device=device)
    in_dim, num_classes, _ = infer_dims(train_ds)
    craft, _ = load_craft(dataset_key, device, output_root, backbone=backbone)
    gat_model = load_trained_gat(dataset_key, device, output_root, in_dim, num_classes)
    graph, patches_U, image_pil_cbm = build_graph_from_single_image(
        dataset_key, image_path, device, craft, patch_size, stride_r)

    node_f = graph.ndata["feat"].float().to(device).requires_grad_(True)
    logits, _, _ = gat_model(graph, node_f)
    probs = torch.nn.functional.softmax(logits[0], dim=0)
    pred_idx = int(torch.argmax(probs).item())
    pred_conf = float(probs[pred_idx].item())
    grads = torch.autograd.grad(probs[pred_idx], node_f, create_graph=False)[0]
    node_importance = grads.abs().sum(dim=1)
    node_importance = node_importance / (node_importance.sum() + 1e-8)
    U = torch.tensor(patches_U, device=node_importance.device, dtype=node_importance.dtype)
    patch_imp = torch.matmul(U, node_importance)
    patch_imp = patch_imp / (patch_imp.sum() + 1e-8)
    stride = int(patch_size * stride_r)
    num_patches_w = (image_pil_cbm.width - patch_size) // stride + 1
    num_patches_h = (image_pil_cbm.height - patch_size) // stride + 1
    gcbm_heatmap = patch_importance_to_heatmap(patch_imp, num_patches_h, num_patches_w,
                                               img_size=image_pil_cbm.width)
    gcbm_blended = overlay_heatmap(image_pil_cbm, gcbm_heatmap, alpha)

    # ── Build figure ──────────────────────────────────────────────────────────
    pred_short = _disp_label(dataset_key, pred_idx)
    pred_color = _label_color(pred_short)
    cnn_short  = _disp_label(dataset_key, cnn_cls)
    cnn_color  = _label_color(cnn_short)
    show_gt    = true_class >= 0
    if show_gt:
        gt_short = _disp_label(dataset_key, true_class)
        gt_color = _label_color(gt_short)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # (a) Original image
    axes[0].imshow(np.array(image_pil))
    axes[0].axis("off")
    axes[0].set_title("Original Image", fontsize=13, fontweight="bold",
                      loc="center", pad=6)
    if show_gt:
        _two_tone_caption(axes[0], -0.06, "Ground truth:", gt_short, gt_color)

    # (b) Baseline CNN Grad-CAM
    axes[1].imshow(np.array(cnn_blended))
    axes[1].axis("off")
    axes[1].set_title(f"Baseline ({_fmt_backbone(actual_backbone)})",
                      fontsize=13, fontweight="bold", loc="center", pad=6)
    _two_tone_caption(axes[1], -0.06, "Predicted:", cnn_short, cnn_color,
                      conf=cnn_conf)

    # (c) GCBM spatial decision map overlay
    axes[2].imshow(np.array(gcbm_blended))
    axes[2].axis("off")
    axes[2].set_title("GCBM Spatial Decision Map",
                      fontsize=13, fontweight="bold", loc="center", pad=6)
    _two_tone_caption(axes[2], -0.06, "Predicted:", pred_short, pred_color,
                      conf=pred_conf)

    plt.tight_layout()
    out_path = os.path.join(os.getcwd(), "output_heatmap_comparison.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.25)
    svg_path = out_path.replace(".png", ".svg")
    plt.savefig(svg_path, format="svg", bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"[INFO] Heatmap comparison figure saved to {out_path}")
    print(f"CNN ({actual_backbone})  -> {cnn_short} ({cnn_conf*100:.2f}%)")
    print(f"GCBM                    -> {pred_short} ({pred_conf*100:.2f}%)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Grad-CAM heatmap explanation (standalone or side-by-side with CBM-GAT concepts)")
    ap.add_argument("--dataset", choices=list(DATASETS.keys()), required=True)
    ap.add_argument("--image_path", required=True, type=str)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--backbone", default="resnet50")
    ap.add_argument("--target-class", type=int, default=None,
                    help="Target class index for Grad-CAM (default: predicted class)")
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="Heatmap overlay transparency (0=image only, 1=heatmap only)")

    ap.add_argument(
        "--mode",
        choices=[
            "standalone_spatial",
            "standalone_cnn",
            "standalone_medical",
            "compare_spatial_concepts",
            "compare_cnn_concepts",
            "compare_medical_concepts",
            "compare_heatmaps",
        ],
        help="Visualization mode. If omitted, falls back to --compare flag for backward compatibility.",
        default=None,
    )

    cmp = ap.add_argument_group("comparison mode (requires trained CBM-GAT)")
    cmp.add_argument("--compare", action="store_true",
                     help="Also run CBM-GAT and produce side-by-side figure")
    cmp.add_argument("--file-root", default=default_output_dir,
                     help="Root dir containing craft / graphs / models")
    cmp.add_argument("--patch-size", type=int, default=70)
    cmp.add_argument("--stride-r", type=float, default=0.5)
    cmp.add_argument("--top-k-max", type=int, default=3)
    cmp.add_argument("--min-concept-weight", type=float, default=0.01)
    cmp.add_argument(
        "--true-class", type=int, default=-1,
        help="Ground-truth class index for the image (0-based). "
             "When provided, the caption shows a colour-coded 'Ground truth:' line "
             "above the prediction. Default -1 = unknown (caption shows prediction only).",
    )
    cmp.add_argument(
        "--save-concept-heatmaps",
        action="store_true",
        help="In compare_* modes, also save a second PNG: one row of per-concept patch heatmaps "
        "(jet overlay on the CBM image; blend strength matches --alpha).",
    )
    cmp.add_argument(
        "--save-concept-heatmap-multi-boxes",
        action="store_true",
        help="In compare_* modes, also save an additional PNG: same layout as per-concept heatmaps "
        "but with multiple distinct patch boxes per concept (greedy NMS on patch grid).",
    )
    cmp.add_argument(
        "--save-concept-multi-boxes-on-image",
        action="store_true",
        help="In compare_* modes, also save an additional PNG: same multi-box selection as "
        "--save-concept-heatmap-multi-boxes but boxes drawn on the raw CBM image (no jet overlay).",
    )
    cmp.add_argument(
        "--max-distinct-patches-per-concept",
        type=int,
        default=4,
        help="Cap for multi-box heatmap figure (greedy selection).",
    )
    cmp.add_argument(
        "--min-patch-separation",
        type=int,
        default=3,
        help="Minimum Chebyshev distance (in patch cells) between boxes for the same concept.",
    )
    cmp.add_argument(
        "--floor-score-frac",
        type=float,
        default=0.15,
        help="Ignore patch candidates below this fraction of the max per-concept score.",
    )
    cmp.add_argument(
        "--cnn-root",
        default=None,
        help="Root dir containing the fine-tuned CNN checkpoint for compare_heatmaps mode. "
             "Defaults to --file-root when not set.",
    )

    args = ap.parse_args()

    # Backward compatibility: if --mode not given, infer from --compare
    if args.mode is None:
        if args.compare:
            mode = "compare_medical_concepts"
        else:
            mode = "standalone_spatial"
    else:
        mode = args.mode

    if mode == "standalone_spatial":
        gradcam_standalone_spatial(
            dataset_key=args.dataset,
            image_path=args.image_path,
            device=args.device,
            backbone=args.backbone,
            target_class=args.target_class,
            alpha=args.alpha,
        )
    elif mode == "standalone_cnn":
        gradcam_standalone_cnn(
            dataset_key=args.dataset,
            image_path=args.image_path,
            device=args.device,
            backbone=args.backbone,
            target_class=args.target_class,
            alpha=args.alpha,
            output_root=args.file_root,
        )
    elif mode == "standalone_medical":
        gradcam_standalone_medical(
            dataset_key=args.dataset,
            image_path=args.image_path,
            device=args.device,
            backbone=args.backbone,
            alpha=args.alpha,
            output_root=args.file_root,
            patch_size=args.patch_size,
            stride_r=args.stride_r,
            top_k_max=args.top_k_max,
            min_concept_weight=args.min_concept_weight,
        )
    elif mode == "compare_spatial_concepts":
        gradcam_vs_concepts_spatial(
            dataset_key=args.dataset,
            image_path=args.image_path,
            device=args.device,
            backbone=args.backbone,
            target_class=args.target_class,
            alpha=args.alpha,
            output_root=args.file_root,
            patch_size=args.patch_size,
            stride_r=args.stride_r,
            top_k_max=args.top_k_max,
            min_concept_weight=args.min_concept_weight,
            save_concept_heatmaps=args.save_concept_heatmaps,
            save_concept_heatmap_multi_boxes=args.save_concept_heatmap_multi_boxes,
            save_concept_multi_boxes_on_image=args.save_concept_multi_boxes_on_image,
            max_distinct_patches_per_concept=args.max_distinct_patches_per_concept,
            min_patch_separation=args.min_patch_separation,
            floor_score_frac=args.floor_score_frac,
        )
    elif mode == "compare_cnn_concepts":
        gradcam_vs_concepts_cnn(
            dataset_key=args.dataset,
            image_path=args.image_path,
            device=args.device,
            backbone=args.backbone,
            target_class=args.target_class,
            alpha=args.alpha,
            output_root=args.file_root,
            patch_size=args.patch_size,
            stride_r=args.stride_r,
            top_k_max=args.top_k_max,
            min_concept_weight=args.min_concept_weight,
            save_concept_heatmaps=args.save_concept_heatmaps,
            save_concept_heatmap_multi_boxes=args.save_concept_heatmap_multi_boxes,
            save_concept_multi_boxes_on_image=args.save_concept_multi_boxes_on_image,
            max_distinct_patches_per_concept=args.max_distinct_patches_per_concept,
            min_patch_separation=args.min_patch_separation,
            floor_score_frac=args.floor_score_frac,
        )
    elif mode == "compare_medical_concepts":
        gradcam_vs_concepts_medical(
            dataset_key=args.dataset,
            image_path=args.image_path,
            device=args.device,
            backbone=args.backbone,
            alpha=args.alpha,
            output_root=args.file_root,
            patch_size=args.patch_size,
            stride_r=args.stride_r,
            top_k_max=args.top_k_max,
            min_concept_weight=args.min_concept_weight,
            true_class=args.true_class,
            save_concept_heatmaps=args.save_concept_heatmaps,
            save_concept_heatmap_multi_boxes=args.save_concept_heatmap_multi_boxes,
            save_concept_multi_boxes_on_image=args.save_concept_multi_boxes_on_image,
            max_distinct_patches_per_concept=args.max_distinct_patches_per_concept,
            min_patch_separation=args.min_patch_separation,
            floor_score_frac=args.floor_score_frac,
        )
    elif mode == "compare_heatmaps":
        gradcam_compare_heatmaps(
            dataset_key=args.dataset,
            image_path=args.image_path,
            device=args.device,
            backbone=args.backbone,
            target_class=args.target_class,
            alpha=args.alpha,
            output_root=args.file_root,
            cnn_root=args.cnn_root if args.cnn_root else args.file_root,
            patch_size=args.patch_size,
            stride_r=args.stride_r,
            top_k_max=args.top_k_max,
            min_concept_weight=args.min_concept_weight,
            true_class=args.true_class,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")


if __name__ == "__main__":
    main()
