import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import argparse
from typing import List, Optional
import numpy as np
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from config import DATASETS, default_output_dir, get_dataset_params


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


# ---------------------------------------------------------------------------
# Standalone Grad-CAM visualisation
# ---------------------------------------------------------------------------

def gradcam_standalone(dataset_key: str, image_path: str, device: str,
                       backbone: str, target_class: Optional[int],
                       alpha: float):
    """Produce a 3-panel figure: original | heatmap | overlay."""
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
    axes[1].set_title("Grad-CAM Heatmap", fontsize=13)
    axes[1].axis("off")

    axes[2].imshow(blended)
    axes[2].set_title(f"Overlay  —  class {pred_cls} ({conf*100:.1f}%)", fontsize=13)
    axes[2].axis("off")

    plt.tight_layout()
    out_path = os.path.join(os.getcwd(), "output_gradcam.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)
    print(f"[INFO] Grad-CAM figure saved to {out_path}")
    print(f"Predicted class: {pred_cls} ({conf*100:.2f}%)")


# ---------------------------------------------------------------------------
# Comparison mode: Grad-CAM  vs  CBM-GAT concepts  (side-by-side)
# ---------------------------------------------------------------------------

def gradcam_vs_concepts(dataset_key: str, image_path: str, device: str,
                        backbone: str, target_class: Optional[int],
                        alpha: float, output_root: str,
                        patch_size: int, stride_r: float,
                        top_k_max: int, min_concept_weight: float):
    """
    Left half  : CBM-GAT spatial heatmap (medical decision) + prediction
    Right half : CBM-GAT concept patches, importance bars, concept examples
    """
    from concepts import build_model_parts, load_craft_and_attach
    from graph import ConceptGraphDataset, load_split, infer_dims
    from model import EGATClassifier, GAT_LightningModule
    from explain_image import (
        load_craft, load_trained_gat, build_graph_from_single_image,
        argmax_safe,
    )

    # ---- CBM-GAT side (mirrors explain_image.explain_image) ----
    train_ds = load_split(output_root, dataset_key, "train", device=device)
    in_dim, num_classes = infer_dims(train_ds)

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
    colors = bar_colors[top_concepts]

    stride = int(patch_size * stride_r)
    num_patches_w = (image_pil_cbm.width - patch_size) // stride + 1
    num_patches_h = (image_pil_cbm.height - patch_size) // stride + 1

    # ---- CBM-GAT spatial heatmap (medical decision) ----
    medical_heatmap = patch_importance_to_heatmap(
        patch_importance=patch_importance,
        num_patches_h=num_patches_h,
        num_patches_w=num_patches_w,
        img_size=image_pil_cbm.width,
    )
    blended_medical = overlay_heatmap(image_pil_cbm, medical_heatmap, alpha)

    # Select exactly one top patch per top concept (clear color–concept link)
    selected_indices = []
    patch_importance_np = patch_importance.detach().cpu().numpy()
    # patches_U is numpy [num_patches, K]; use its columns directly
    num_patches, num_concepts_total = patches_U.shape

    for concept_id in top_concepts:
        if concept_id >= num_concepts_total:
            continue
        concept_activations = patches_U[:, concept_id]  # numpy [num_patches]
        # score: how important this patch is for this concept and for the decision
        scores = concept_activations * patch_importance_np
        best_idx = int(np.argmax(scores))
        selected_indices.append((concept_id, best_idx))

    # fallback: if no concept produced a valid index, keep previous behavior
    if not selected_indices:
        selected_indices = [(top_concepts[0], idx) for idx in sorted_patch_idx[:top_k]]

    draw = ImageDraw.Draw(image_pil_cbm)
    for concept_id, idx in selected_indices:
        row = idx // num_patches_w
        col = idx % num_patches_w
        x, y = col * stride, row * stride
        c_index = top_concepts.index(concept_id) if concept_id in top_concepts else 0
        outline_color = tuple((colors[c_index] * 255).astype(int))
        draw.rectangle([x, y, x + patch_size, y + patch_size],
                       outline=tuple(outline_color), width=3)

    # ---- Build the combined figure ----
    # Layout: [GradCAM overlay | CBM-GAT patches | concept bars | concept examples]
    fig = plt.figure(figsize=(32, 8), constrained_layout=True)
    outer_gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 1])

    # Panel 1 – CBM-GAT spatial heatmap (medical decision)
    gc_gs = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=outer_gs[0, 0], height_ratios=[30, 2], hspace=0.1)
    ax_gc = fig.add_subplot(gc_gs[0, 0])
    ax_gc.imshow(blended_medical)
    ax_gc.set_title("CBM-GAT: where it looks (medical)", fontsize=13, fontweight="bold")
    ax_gc.axis("off")
    ax_gc_cap = fig.add_subplot(gc_gs[1, 0])
    ax_gc_cap.axis("off")
    ax_gc_cap.text(0.5, 0.5,
                   f"GAT prediction: class {pred_idx} ({pred_conf*100:.1f}%)",
                   ha="center", va="center", fontsize=11)

    # Panel 2 – CBM-GAT concept patches on image
    cbm_gs = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=outer_gs[0, 1], height_ratios=[30, 2], hspace=0.1)
    ax_cbm = fig.add_subplot(cbm_gs[0, 0])
    ax_cbm.imshow(np.array(image_pil_cbm))
    ax_cbm.set_title("CBM-GAT (concept patches)", fontsize=13, fontweight="bold")
    ax_cbm.axis("off")
    ax_cbm_cap = fig.add_subplot(cbm_gs[1, 0])
    ax_cbm_cap.axis("off")
    ax_cbm_cap.text(0.5, 0.5,
                    f"GAT prediction: class {pred_idx} ({pred_conf*100:.1f}%)",
                    ha="center", va="center", fontsize=11)

    # Panel 3 – concept importance bars
    ax_bar = fig.add_subplot(outer_gs[0, 2])
    y_pos = np.arange(top_k)
    bars = ax_bar.barh(y_pos, top_values, color=colors[:top_k], align="center")
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels([f"Concept {c}" for c in top_concepts])
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("Importance")
    ax_bar.set_title(f"Top {top_k} Concept IDs", fontsize=13, fontweight="bold")
    if top_k:
        ax_bar.set_xlim(0, max(top_values) * 1.3)
        for i, b in enumerate(bars):
            ax_bar.text(b.get_width() + max(top_values) * 0.01,
                        b.get_y() + b.get_height() / 2,
                        f"{top_values[i]:.3f}", va="center", fontsize=10)

    # Panel 4 – concept example thumbnails
    right_gs = gridspec.GridSpecFromSubplotSpec(
        top_k, 2, subplot_spec=outer_gs[0, 3],
        width_ratios=[0.3, 1.0], wspace=0.0, hspace=0.4)
    for i in range(top_k):
        concept_id = top_concepts[i]
        c_color = colors[i]
        fig.add_subplot(right_gs[i, 0]).axis("off")
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

    out_path = os.path.join(os.getcwd(), "output_gradcam_vs_concepts.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    print(f"[INFO] Comparison figure saved to {out_path}")
    print(f"CBM-GAT   -> class {pred_idx} ({pred_conf*100:.2f}%)")
    print(f"Top concepts: {', '.join(str(c) for c in top_concepts)}")


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

    cmp = ap.add_argument_group("comparison mode (requires trained CBM-GAT)")
    cmp.add_argument("--compare", action="store_true",
                     help="Also run CBM-GAT and produce side-by-side figure")
    cmp.add_argument("--file-root", default=default_output_dir,
                     help="Root dir containing craft / graphs / models")
    cmp.add_argument("--patch-size", type=int, default=70)
    cmp.add_argument("--stride-r", type=float, default=0.5)
    cmp.add_argument("--top-k-max", type=int, default=3)
    cmp.add_argument("--min-concept-weight", type=float, default=0.01)

    args = ap.parse_args()

    if args.compare:
        gradcam_vs_concepts(
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
        )
    else:
        gradcam_standalone(
            dataset_key=args.dataset,
            image_path=args.image_path,
            device=args.device,
            backbone=args.backbone,
            target_class=args.target_class,
            alpha=args.alpha,
        )


if __name__ == "__main__":
    main()
