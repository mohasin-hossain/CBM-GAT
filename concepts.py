import os
import dill
import json
from typing import List, Tuple, Dict, Optional
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from collections import Counter
from craft.craft_torch import Craft, torch_to_numpy # use the original Craft implementation


def build_model_parts(backbone_name: str = "resnet50",
                      device: str = "cuda",
                      pretrained: bool = True) -> Tuple[nn.Module, nn.Module]:
    """
    Returns (g, h) where:
      g: input -> last conv feature map (before avgpool/fc)  [B, C, H', W']
      h: feature map -> logits (global avg-pool then fc)

    Supported backbones and their feature dimensions:
      resnet18      -> 512-dim spatial features (pytorchcv CUB-pretrained; CUB-only)
      resnet50      -> 2048-dim spatial features
      densenet201   -> 1920-dim spatial features
      mobilenet_v2  -> 1280-dim spatial features
    """
    backbone_name = backbone_name.lower()
    if backbone_name == "resnet18":
        from pytorchcv.model_provider import get_model as ptcv_get_model

        ptcv_root = os.path.join(
            os.environ.get("TORCH_HOME", os.path.expanduser("~/.torch")),
            "pytorchcv",
        )
        net = ptcv_get_model("resnet18_cub", pretrained=pretrained, root=ptcv_root)
        g = nn.Sequential(*list(net.features.children())[:-1]).to(device).eval()
        fc = net.output
        h = lambda x, _fc=fc: _fc(torch.mean(x, (2, 3)))
        return g, h

    elif backbone_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        g = nn.Sequential(*list(model.children())[:-2]).to(device).eval()
        # keep fc alive for h; discard the rest
        fc = model.fc
        h = lambda x, _fc=fc: _fc(torch.mean(x, (2, 3)))
        return g, h

    elif backbone_name == "densenet201":
        import torch.nn.functional as _F
        weights = models.DenseNet201_Weights.DEFAULT if pretrained else None
        model = models.densenet201(weights=weights)
        # model.features ends with BatchNorm2d (no ReLU), so activations can be
        # negative — CRAFT's NMF requires non-negative inputs.  Append ReLU to g
        # so both g output and h input see the same positive feature map.
        g = nn.Sequential(model.features, nn.ReLU(inplace=False)).to(device).eval()
        classifier = model.classifier
        h = lambda x, _clf=classifier: _clf(x.mean([2, 3]))
        return g, h

    elif backbone_name == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        # model.features outputs [B, 1280, H', W']
        g = model.features.to(device).eval()
        classifier = model.classifier
        h = lambda x, _clf=classifier: _clf(x.mean([2, 3]))
        return g, h

    else:
        raise ValueError(
            f"Unsupported backbone for Craft: {backbone_name!r}. "
            "Choose from: resnet18, resnet50, densenet201, mobilenet_v2"
        )

# craft fitting and scoring
def fit_craft_for_k(images: torch.Tensor,
                    k: int,
                    patch_size: int,
                    batch_size: int,
                    device: str,
                    g: nn.Module,
                    h: nn.Module) -> Tuple[Craft, torch.Tensor]:
    """
    Fits Craft for a given number of concepts k on the provided images.
    Returns the fitted craft instance (with g,h attached) and crops_u (patch activations, [num_patches, k]).
    """
    craft = Craft(
        input_to_latent=g.to(device),
        latent_to_logit=h,
        number_of_concepts=k,
        patch_size=patch_size,
        batch_size=batch_size,
        device=device,
    )
    crops, crops_u, w = craft.fit(images.to(device)) # crops_u is torch.Tensor [num_patches, k]
    crops = np.moveaxis(torch_to_numpy(crops), 1, -1)
    return craft, crops, crops_u

def score_concepts_from_u(crops_u: torch.Tensor,
                          labels: torch.Tensor,
                          q: float = 0.1,
                          theta: float = 0.6,
                          lambda_weight: float = 1.0) -> Tuple[float, Dict]:
    """
    Implements your discriminativeness scoring using crops_u.
    We infer patches_per_image from crops_u.shape[0] // N_images.
    """


    if crops_u.size == 0:
        return -1.0, {"Avg D_i": 0.0, "Penalty": 1.0, "Num Discriminative Concepts": 0, "Class Split": "None"}

    U = crops_u
    num_patches, num_concepts = U.shape
    labels_np = labels.detach().cpu().numpy()
    N_images = len(labels_np)
    patches_per_image = max(1, num_patches // max(1, N_images))
    num_classes = len(set(labels_np)) if len(labels_np) > 0 else 0

    results = []
    for i in range(num_concepts):
        u_i = U[:, i]
        tau_i = np.quantile(u_i, 1 - q)
        top_patch_indices = np.where(u_i >= tau_i)[0]
        top_image_indices = top_patch_indices // patches_per_image
        top_class_labels = labels_np[top_image_indices]

        class_counts = Counter(top_class_labels)
        total = len(top_class_labels) if len(top_class_labels) > 0 else 1
        R_ic = [class_counts.get(c, 0) / total for c in range(num_classes)]
        D_i = max(R_ic) if R_ic else 0.0
        dominant_class = int(np.argmax(R_ic)) if R_ic else 0

        results.append({
            "concept": i,
            "D_i": D_i,
            "dominant_class": dominant_class if D_i >= theta else None
        })

    discriminative = [r for r in results if r["D_i"] >= theta]
    d = len(discriminative)

    if d > 0 and num_classes > 0:
        avg_Di = float(np.mean([r["D_i"] for r in discriminative]))
        class_assignments = [r["dominant_class"] for r in discriminative]
        class_counts = Counter(class_assignments)
        penalty = sum([abs(class_counts.get(c, 0) / d - 1 / num_classes) for c in range(num_classes)]) / num_classes
        score = avg_Di - lambda_weight * penalty
    else:
        avg_Di = 0.0
        penalty = 1.0
        score = -lambda_weight * penalty

    summary = {
        "Avg D_i": round(avg_Di, 4),
        "Penalty": round(penalty, 4),
        'Class Split': [class_counts.get(c, 0) for c in range(num_classes)],
        "Num Discriminative Concepts": d
    }
    return float(score), summary

def auto_select_k(images: torch.Tensor,
                  labels: torch.Tensor,
                  candidates: List[int],
                  patch_size: int,
                  batch_size: int,
                  device: str,
                  g: nn.Module,
                  h: nn.Module) -> Tuple[int, List[Dict], Craft]:
    best_k, best_score = None, -1e9
    table = []
    for k in candidates:
        _, _, crops_u = fit_craft_for_k(images, k, patch_size, batch_size, device, g, h)
        score, summary = score_concepts_from_u(crops_u, labels)
        rec = {"Num Concepts": k, **summary, "Score": round(score, 4)}
        table.append(rec)
        if score > best_score:
            best_score = score
            best_k = k
    return best_k, table


# save/load a lightweight Craft
def save_craft_light(craft: Craft, out_path: str):
    """
    Save Craft without the heavy model objects (input_to_latent / latent_to_logit).
    We temporarily remove them, dump, then restore (in-memory).
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    g = getattr(craft, "input_to_latent", None)
    h = getattr(craft, "latent_to_logit", None)
    try:
        craft.input_to_latent = None
        craft.latent_to_logit = None
        with open(out_path, "wb") as f:
            dill.dump(craft, f)
    finally:
        craft.input_to_latent = g
        craft.latent_to_logit = h

def load_craft_and_attach(path: str, g: nn.Module, h: nn.Module) -> Craft:
    with open(path, "rb") as f:
        craft = dill.load(f)
    # reattach models
    craft.input_to_latent = g
    craft.latent_to_logit = h
    return craft


def write_best_k(craft_dir: str, best_k: int, patch_size: int, stride_r: float):
    """
    Write a single JSON file to U_meta/nmf_best_k.json containing only:
    best_k, n_components, patch_size, stride_r.
    """
    u_meta = os.path.join(craft_dir, "U_meta")
    os.makedirs(u_meta, exist_ok=True)
    payload = {
        "best_k": int(best_k),
        "patch_size": int(patch_size),
        "stride_r": float(stride_r),
    }
    out_path = os.path.join(u_meta, "nmf_best_k.json")
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)