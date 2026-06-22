import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # silence TensorFlow logging
import argparse
from typing import List
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import torch
import torch.nn.functional as F
from config import DATASETS, default_output_dir, get_dataset_params, get_class_label
from graph import ConceptGraphDataset, load_split, infer_dims          # graph utils (infer dims from saved graphs)
from concepts import build_model_parts, load_craft_and_attach          # attach Craft's g/h
from model import EGATClassifier, GAT_LightningModule                  # GAT + Lightning wrapper


# small helpers here
def argmax_safe(patches_U: np.ndarray, search_list: List[int]) -> np.ndarray:
    """Row-wise argmax restricted to indices in search_list (first valid)."""
    sorted_idx = np.argsort(-patches_U, axis=1)  # desc
    out = np.zeros(patches_U.shape[0], dtype=int)
    allow = set(search_list)
    for i in range(patches_U.shape[0]):
        for j in sorted_idx[i]:
            if j in allow:
                out[i] = j
                break
    return out


def load_eval_transform(dataset_key: str):
    """Get the dataset-specific eval transform from config."""
    tdict = DATASETS[dataset_key].build_transforms()
    return tdict["eval"]


def load_craft(dataset_key: str, device: str, output_root: str, backbone: str = "resnet50"):
    """Rebuild g/h and attach to the saved light Craft."""
    g, h = build_model_parts(backbone_name=backbone, device=device, pretrained=True)
    craft_dir = os.path.join(output_root, dataset_key, "craft", dataset_key)
    craft_path = os.path.join(craft_dir, f"craft_{dataset_key}.dill")
    if not os.path.isfile(craft_path):
        raise FileNotFoundError(f"Craft not found: {craft_path}")
    craft = load_craft_and_attach(craft_path, g, h)
    return craft, craft_dir


def load_trained_gat(dataset_key: str, device: str, output_root: str, in_dim: int, num_classes: int):
    """
    Build the EGATClassifier with correct dims, then load Lightning checkpoint.
    We infer in_dim/num_classes from saved graphs (train split).
    """

    # dataset-specific heads/hidden
    ds_params = get_dataset_params(dataset_key) or {}
    num_heads = ds_params.get("num_heads", 4)
    hidden_dim = ds_params.get("hidden_dim", 128)

    model = EGATClassifier(
        in_feats=in_dim,
        out_feats=hidden_dim,
        num_heads=num_heads,
        out_dim=num_classes,
        feat_drop=0.0,
        node_drop=0.0,
    )

    ckpt = os.path.join(output_root, dataset_key, "models", dataset_key, f"{dataset_key}_best_model.ckpt")
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    lightning = GAT_LightningModule.load_from_checkpoint(ckpt, model=model)
    gat = lightning.model.eval().to(device)
    return gat


@torch.no_grad()
def build_graph_from_single_image(dataset_key: str, image_path: str, device: str,
                                  craft, patch_size: int, stride_r: float):
    """
    Returns:
      graph (DGLGraph on device),
      patches_U (np.ndarray [#patches, K]),
      image_pil_resized (for overlay),
      stride (int), patch_size (int)
    """
    # eval transform from config
    eval_tfm = load_eval_transform(dataset_key)

    # load + resize image only for display; the model sees transformed tensor
    image_pil = Image.open(image_path).convert("RGB").resize((224, 224), Image.BICUBIC)
    x = eval_tfm(image_pil).unsqueeze(0)  # (1,3,224,224)

    # make a tiny ConceptGraphDataset with a dummy label
    ds = ConceptGraphDataset(
        images=x.to(device),
        y=torch.tensor([0], dtype=torch.long, device=device),  # dummy
        masks=None,
        patch_size=patch_size,
        craft_xai=craft,
        ignore_list=[],
        device=device,
        stride_r=stride_r,
        coverage_threshold=0.0,
        seed=42,
        requires_grad=False,
    )
    ds.process()  # populates graphs, patches_U, etc.

    graph = ds[0][0].to(device)
    patches_U = ds.patches_U  # numpy array
    return graph, patches_U, image_pil


def explain_image(dataset_key: str,
                  image_path: str,
                  device: str = "cuda",
                  output_root: str = default_output_dir,
                  backbone: str = "resnet50",
                  in_dim: int = 2048,
                  num_classes: int = 2,
                  patch_size: int = 70,
                  stride_r: float = 0.5,
                  top_k_max: int = 3,
                  min_concept_weight: float = 0.01):
    # 1. craft (attached) + model
    craft, craft_dir = load_craft(dataset_key, device, output_root, backbone=backbone)
    gat_model = load_trained_gat(
        dataset_key, device, output_root, in_dim, num_classes)

    # 2. build graph from the single image
    graph, patches_U, image_pil = build_graph_from_single_image(
        dataset_key, image_path, device, craft, patch_size, stride_r)

    # 3. forward for prediction + attention + node features
    node_f = graph.ndata["feat"].float().to(device).requires_grad_(True)
    logits, attn_scr1, h = gat_model(graph, node_f)
    probs = F.softmax(logits[0], dim=0)
    pred_idx = int(torch.argmax(probs).item())
    pred_conf = float(probs[pred_idx].item())

    # 4. node-grad importance L1 norm of p/node_f
    target_prob = probs[pred_idx]
    grads = torch.autograd.grad(target_prob, node_f, create_graph=False)[0]  
    node_importance = grads.abs().sum(dim=1)                               

    # combine (you used pure grad; keep it simple and stable)
    graph_node_importance = node_importance
    graph_node_importance = graph_node_importance / (graph_node_importance.sum() + 1e-8)

    # 5. rank concepts and patches
    sorted_values, sorted_indices = torch.sort(graph_node_importance, descending=True)
    concept_importance_values = sorted_values.tolist()
    concept_ranking = sorted_indices.tolist()

    # keep top concepts above threshold
    top_vals = [v for v in concept_importance_values if v > min_concept_weight]
    top_k = min(len(top_vals), top_k_max) if top_vals else min(top_k_max, len(concept_ranking))
    top_concepts = concept_ranking[:top_k]
    top_values = concept_importance_values[:top_k]

    # patch importance via U * node-importance
    U = torch.tensor(patches_U, device=graph_node_importance.device, dtype=graph_node_importance.dtype)  # [#patches, K]
    patch_importance = torch.matmul(U, graph_node_importance)  # [#patches]
    patch_importance = patch_importance / (patch_importance.sum() + 1e-8)
    _, sorted_patch_idx = torch.sort(patch_importance, descending=True)
    sorted_patch_idx = sorted_patch_idx.tolist()

    # to color patches by the concept they map to (restricted argmax)
    patches_C = argmax_safe(patches_U, top_concepts)

    # 6. visualization
    bar_colors = plt.cm.tab10(np.arange(10))  # up to 10 distinct concept colors
    colors = bar_colors[top_concepts]

    stride = int(patch_size * stride_r)
    num_patches_w = (image_pil.width - patch_size) // stride + 1

    # draw top patches
    draw = ImageDraw.Draw(image_pil)
    for idx in sorted_patch_idx[:top_k]:
        row = idx // num_patches_w
        col = idx % num_patches_w
        x, y = col * stride, row * stride
        # color from concept membership
        c_index = top_concepts.index(patches_C[idx]) if patches_C[idx] in top_concepts else 0
        outline_color = tuple((colors[c_index] * 255).astype(int))
        draw.rectangle([x, y, x + patch_size, y + patch_size], outline=tuple(outline_color), width=3)

    # make figure
    fig = plt.figure(figsize=(24, 7), constrained_layout=True)  # keep constrained layout ON
    outer_gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])

    # left column: image + centered caption
    left_gs = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=outer_gs[0, 0], height_ratios=[30, 2], hspace=0.1
    )
    ax_img = fig.add_subplot(left_gs[0, 0])
    ax_caption = fig.add_subplot(left_gs[1, 0])

    # image
    ax_img.imshow(np.array(image_pil))
    ax_img.axis("off")

    # centered caption (no extra top padding)
    ax_caption.axis("off")
    ax_caption.text(
        0.5, 0.5,
        f"Predicted: {get_class_label(dataset_key, pred_idx)} ({pred_conf*100:.2f}%)",
        ha="center", va="center", fontsize=12
    )

    # middle: top concept bars (unchanged)
    ax_bar = fig.add_subplot(outer_gs[0, 1])
    y_pos = np.arange(top_k)
    bars = ax_bar.barh(y_pos, top_values, color=colors[:top_k], align="center")
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels([f"Concept: {c}" for c in top_concepts])
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("Importance")
    ax_bar.set_title(f"Top {top_k} Concept IDs")
    if top_k:
        ax_bar.set_xlim(0, max(top_values) * 1.3)
        for i, b in enumerate(bars):
            ax_bar.text(
                b.get_width() + max(top_values) * 0.01,
                b.get_y() + b.get_height() / 2,
                f"{top_values[i]:.3f}",
                va="center", fontsize=10
            )

    # right: concept examples (unchanged)
    right_gs = gridspec.GridSpecFromSubplotSpec(
        top_k, 2, subplot_spec=outer_gs[0, 2],
        width_ratios=[0.3, 1.0], wspace=0.0, hspace=0.4
    )
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
            border = patches.Rectangle((0, 0), 1, 1, transform=ax_c.transAxes,
                                    linewidth=8, edgecolor=c_color, facecolor="none")
            ax_c.add_patch(border)
        else:
            ax_c.text(0.5, 0.5, f"(no example for {concept_id})",
                    ha="center", va="center", fontsize=11)

    # plt.tight_layout()
    out_path = os.path.join(os.getcwd(), "output_explanation.png")
    plt.savefig(out_path, dpi=150)
    print(f"[INFO] Figure saved to {out_path}")

    # print prediction + important concepts
    print(f"Predicted label: {get_class_label(dataset_key, pred_idx)} ({pred_conf*100:.2f}%)")
    print("Important concepts:", ", ".join(str(c) for c in top_concepts))

    plt.show()
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser("predict + explain a single image with concept graphs")
    ap.add_argument("--dataset", choices=list(DATASETS.keys()))
    ap.add_argument("--image_path", required=True, type=str)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--file-root", default=default_output_dir)
    ap.add_argument("--backbone", default="resnet50")
    ap.add_argument("--patch-size", type=int, default=70)
    ap.add_argument("--stride-r", type=float, default=0.5)
    ap.add_argument("--top-k-max", type=int, default=3)
    ap.add_argument("--min-concept-weight", type=float, default=0.01)
    args = ap.parse_args()

    explain_image(
        dataset_key=args.dataset,
        image_path=args.image_path,
        device=args.device,
        output_root=args.file_root,
        backbone=args.backbone,
        patch_size=args.patch_size,
        stride_r=args.stride_r,
        top_k_max=args.top_k_max,
        min_concept_weight=args.min_concept_weight,
    )


if __name__ == "__main__":
    main()
