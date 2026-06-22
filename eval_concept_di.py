"""
eval_concept_di.py — Phase G: Per-Concept D_i Discriminativeness Scores

Addresses reviewer B2: "Concepts are free non-negative vectors without clear connection
to visual concepts."

What is D_i?
  For concept c, take the top q% of patches with highest U[:, c] activation.
  D_i = fraction of those patches whose source image belongs to the dominant class.
  D_i = 1.0  →  perfectly class-specific (e.g. all melanoma patches)
  D_i = 0.5  →  class-agnostic (random split between classes)

Why not just use concept_search.json?
  concept_search.json stores average D_i across all K concepts for each K candidate.
  This script computes per-concept D_i — one score per concept — needed for the paper table
  "Concept 3 (CLIP label: blue-white veil) has D_i = 0.81, dominant class: Melanoma."

Logic adapted from score_concepts_from_u() in concepts.py — DO NOT modify concepts.py.

Per-concept output fields (new fields added):
  dominant_class        — int index of the majority class in the top-q patches (always set)
  dominant_class_name   — human-readable name from config.class_names (e.g. "Melanoma")
  is_discriminative     — bool, True when D_i >= theta
  class_distribution    — list[float], fraction of top-q patches per class (all classes)
  class_names           — list[str] matching class_distribution indices

Outputs:
  {output_root}/concept_di_scores.json
  {output_root}/concept_di_summary.txt   (human-readable table, one per dataset)
"""

import os
import json
import argparse

import numpy as np
import torch
import dill

from concepts import load_craft_and_attach, build_model_parts
from config import DATASETS, get_class_label
from utils import _set_seed


# ---------------------------------------------------------------------------
# D_i computation (per-concept, patch-level, adapted from concepts.py)
# ---------------------------------------------------------------------------

def compute_per_concept_di(patches_U: np.ndarray,
                           patch_labels: np.ndarray,
                           q: float = 0.1,
                           theta: float = 0.6,
                           ds_key: str = None):
    """
    Compute per-concept D_i discriminativeness scores.

    Args:
        patches_U:    [total_patches, K] NMF activations (numpy)
        patch_labels: [total_patches]    class label for each patch (replicated from image label)
        q:            top-q fraction used to define "top patches"
        theta:        D_i threshold to count a concept as discriminative
        ds_key:       dataset key for resolving human-readable class names (optional)

    Returns:
        list of dicts with fields:
          concept_id, D_i, is_discriminative,
          dominant_class (int, always set),
          dominant_class_name (str, always set),
          class_distribution (list[float], one per class),
          class_names (list[str])
        avg_Di: mean D_i over all concepts
        num_discriminative: count with D_i >= theta
    """
    patches_U = np.asarray(patches_U)
    if patches_U.ndim != 2:
        patches_U = patches_U.reshape(-1, patches_U.shape[-1])
    num_patches, K = patches_U.shape
    num_classes = int(patch_labels.max()) + 1

    # Resolve class names from config once
    ds_spec = DATASETS.get(ds_key) if ds_key else None
    class_names = (ds_spec.class_names or []) if ds_spec else []
    if not class_names:
        class_names = [str(i) for i in range(num_classes)]

    results = []
    for c in range(K):
        u_c = patches_U[:, c]
        top_q_thresh = np.quantile(u_c, 1 - q)
        top_mask = u_c >= top_q_thresh
        top_labels = patch_labels[top_mask]

        if len(top_labels) == 0:
            results.append({
                "concept_id":          c,
                "D_i":                 0.5,
                "is_discriminative":   False,
                "dominant_class":      0,
                "dominant_class_name": class_names[0] if class_names else "0",
                "class_distribution":  [1.0 / num_classes] * num_classes,
                "class_names":         class_names[:num_classes],
            })
            continue

        counts = np.bincount(top_labels.astype(int), minlength=num_classes)
        total  = counts.sum()
        dominant   = int(counts.argmax())
        D_i        = float(counts[dominant] / total)
        class_dist = [round(float(counts[i] / total), 4) for i in range(num_classes)]
        dom_name   = class_names[dominant] if dominant < len(class_names) else str(dominant)

        results.append({
            "concept_id":          c,
            "D_i":                 round(D_i, 4),
            "is_discriminative":   bool(D_i >= theta),
            "dominant_class":      dominant,
            "dominant_class_name": dom_name,
            "class_distribution":  class_dist,
            "class_names":         class_names[:num_classes],
        })

    discriminative = [r for r in results if r["is_discriminative"]]
    avg_Di = float(np.mean([r["D_i"] for r in results]))
    return results, round(avg_Di, 4), len(discriminative)


# ---------------------------------------------------------------------------
# Per-dataset processing
# ---------------------------------------------------------------------------

def process_dataset(ds_key: str, output_root: str, device: str,
                    q: float, theta: float):
    """
    Load craft, run NMF split images through input_to_latent + reducer.transform,
    accumulate patch-level activations and labels, compute D_i per concept.
    """
    ds_spec = DATASETS[ds_key]
    run_id  = ds_key
    craft_dir  = os.path.join(output_root, ds_key, "craft", run_id)
    craft_path = os.path.join(craft_dir, f"craft_{ds_key}.dill")
    best_k_path = os.path.join(craft_dir, "U_meta", "nmf_best_k.json")

    if not os.path.isfile(craft_path):
        print(f"  [SKIP] Craft file not found: {craft_path}")
        return None

    print(f"  Loading craft: {craft_path}")
    g, h = build_model_parts(device=device, pretrained=False)
    craft = load_craft_and_attach(craft_path, g, h)

    # Read best_k from U_meta
    best_k = None
    if os.path.isfile(best_k_path):
        with open(best_k_path) as f:
            best_k = json.load(f).get("best_k")
    if best_k is None:
        # Fall back to craft's internal reducer
        best_k = craft.reducer.n_components_ if hasattr(craft.reducer, "n_components_") else None
        if best_k is None:
            best_k = craft.reducer.components_.shape[0]
    K = int(best_k)
    print(f"  K = {K}")

    # Load NMF split images + labels
    print(f"  Loading NMF split images...")
    tdict = ds_spec.build_transforms()
    paths = ds_spec.resolve_paths()
    X, Y, _ = ds_spec.load_split(paths, tdict, split="nmf")
    # X: [N, C, H, W] tensor  Y: [N] tensor (class labels)
    N = X.shape[0]
    print(f"  N images = {N}")

    # Run each image through input_to_latent → collect activations
    # craft.reducer.transform() expects numpy [total_patches, latent_dim]
    g_model = craft.input_to_latent.to(device).eval()
    batch_size = 32

    all_patches_U = []   # [total_patches, K]
    all_patch_labels = []  # [total_patches]

    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            x_batch = X[start:end].to(device)   # [B, C, H, W]
            y_batch = Y[start:end].numpy()        # [B]

            # Extract patch crops from each image using craft's get_crops logic
            # craft.transform returns crops_u [n_patches_in_batch * B, K]
            try:
                crops_u = craft.transform(x_batch)   # [n_patches * B, K] or [B, P, K]
                crops_u_np = crops_u.detach().cpu().numpy() if torch.is_tensor(crops_u) else np.array(crops_u)
            except Exception as e:
                print(f"  [WARN] craft.transform failed for batch {start}:{end}: {e}. Skipping.")
                continue

            B = len(y_batch)
            # CRAFT may return [B, P, K] (per-image patch grids) instead of raveled [B*P, K].
            if crops_u_np.ndim == 3:
                if crops_u_np.shape[0] != B:
                    raise ValueError(
                        f"craft.transform 3D output first dim {crops_u_np.shape[0]} != batch size {B}"
                    )
                P = crops_u_np.shape[1]
                crops_u_np = crops_u_np.reshape(B * P, crops_u_np.shape[2])
                patches_per_image = P
                n_total_patches = crops_u_np.shape[0]
            elif crops_u_np.ndim == 4:
                # [B, H, W, K] feature maps before ravel
                if crops_u_np.shape[0] != B:
                    raise ValueError(
                        f"craft.transform 4D output first dim {crops_u_np.shape[0]} != batch size {B}"
                    )
                _, H, W, Kdim = crops_u_np.shape
                crops_u_np = crops_u_np.reshape(B * H * W, Kdim)
                patches_per_image = H * W
                n_total_patches = crops_u_np.shape[0]
            elif crops_u_np.ndim == 2:
                n_total_patches = crops_u_np.shape[0]
                patches_per_image = max(1, n_total_patches // B)
            else:
                raise ValueError(f"Unexpected craft.transform output shape {crops_u_np.shape}")

            # Replicate label for each patch of its source image
            for img_idx, label in enumerate(y_batch):
                p_start = img_idx * patches_per_image
                p_end = min(p_start + patches_per_image, n_total_patches)
                n_p = p_end - p_start
                if n_p > 0:
                    all_patches_U.append(crops_u_np[p_start:p_end])
                    all_patch_labels.extend([int(label)] * n_p)

            if (start // batch_size) % 10 == 0:
                print(f"  Processed {end}/{N} images, patches so far: {sum(len(u) for u in all_patches_U)}")

    if not all_patches_U:
        print(f"  [SKIP] No patches collected for {ds_key}.")
        return None

    patches_U    = np.vstack(all_patches_U)      # [total_patches, K]
    patch_labels = np.array(all_patch_labels)    # [total_patches]
    print(f"  Total patches: {patches_U.shape[0]}, K={patches_U.shape[1]}")

    # Compute per-concept D_i
    concepts_di, avg_Di, num_disc = compute_per_concept_di(
        patches_U, patch_labels, q=q, theta=theta, ds_key=ds_key
    )

    return {
        "best_k": K,
        "num_images": N,
        "q": q,
        "theta": theta,
        "concepts": concepts_di,
        "avg_D_i": avg_Di,
        "num_discriminative": num_disc,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser("Per-concept D_i discriminativeness scores (Phase G)")
    ap.add_argument("--datasets",    nargs="+",
                    default=["ham10000", "ph2", "derm7pt", "imagenet"],
                    help="Datasets to process. Keys must be in config.py DATASETS.")
    ap.add_argument("--output-root", required=True,
                    help="Absolute netscratch path (same as used for build_concept_graphs.py)")
    ap.add_argument("--device",      default="cuda")
    ap.add_argument("--theta",       type=float, default=0.6,
                    help="D_i threshold for 'discriminative' count (default: 0.6)")
    ap.add_argument("--q",           type=float, default=0.1,
                    help="Top-q quantile for D_i computation (default: 0.1 = top 10%%)")
    ap.add_argument("--seed",        type=int, default=42)
    args = ap.parse_args()

    _set_seed(args.seed)
    device = args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu"

    all_results = {}
    summary_lines = []

    for ds_key in args.datasets:
        if ds_key not in DATASETS:
            print(f"[SKIP] Unknown dataset key: {ds_key}. Valid keys: {list(DATASETS.keys())}")
            continue

        print(f"\n{'='*60}")
        print(f"Dataset: {ds_key}")
        print(f"{'='*60}")

        result = process_dataset(ds_key, args.output_root, device,
                                 q=args.q, theta=args.theta)
        if result is None:
            continue

        all_results[ds_key] = result

        # Print summary table
        K = result["best_k"]
        header = (f"\n  {'Concept':>8}  {'D_i':>6}  {'Dominant class':<22}"
                  f"  {'Dist':>20}  {'Disc':>4}")
        print(header)
        print("  " + "-" * 68)
        for c_info in result["concepts"]:
            disc = "YES" if c_info["is_discriminative"] else "no"
            dom  = f"{c_info['dominant_class']} ({c_info['dominant_class_name']})"
            dist = " / ".join(f"{v:.2f}" for v in c_info["class_distribution"])
            print(f"  {c_info['concept_id']:>8}  {c_info['D_i']:>6.3f}  "
                  f"{dom:<22}  [{dist:>20}]  {disc:>4}")
        print(f"\n  avg D_i = {result['avg_D_i']:.4f}  |  "
              f"discriminative (≥{args.theta}): {result['num_discriminative']}/{K}")

        summary_lines.append(
            f"{ds_key}: avg_D_i={result['avg_D_i']:.4f}, "
            f"discriminative={result['num_discriminative']}/{K}"
        )

    if not all_results:
        print("\nNo results produced. Check that craft files exist under --output-root.")
        return

    # Save JSON
    json_path = os.path.join(args.output_root, "concept_di_scores.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {json_path}")

    # Save human-readable summary
    txt_path = os.path.join(args.output_root, "concept_di_summary.txt")
    with open(txt_path, "w") as f:
        f.write("Per-concept D_i discriminativeness scores\n")
        f.write(f"theta={args.theta}, q={args.q}\n\n")
        for ds_key, result in all_results.items():
            concepts = result["concepts"]
            class_names = concepts[0]["class_names"] if concepts else []
            dist_header = " / ".join(class_names) if class_names else "class distribution"
            f.write(f"\n=== {ds_key} (K={result['best_k']}) ===\n")
            f.write(f"{'Concept':>8}  {'D_i':>6}  {'Dominant class':<22}  "
                    f"[{dist_header}]  {'Flag':>5}\n")
            f.write("-" * 70 + "\n")
            for c_info in concepts:
                flag = "DISC" if c_info["is_discriminative"] else ""
                dom  = f"{c_info['dominant_class']} ({c_info['dominant_class_name']})"
                dist = " / ".join(f"{v:.2f}" for v in c_info["class_distribution"])
                f.write(f"{c_info['concept_id']:>8}  {c_info['D_i']:>6.3f}  "
                        f"{dom:<22}  [{dist}]  {flag:>5}\n")
            f.write(f"\n  avg D_i={result['avg_D_i']:.4f}  "
                    f"discriminative={result['num_discriminative']}/{result['best_k']}\n")
    print(f"Saved: {txt_path}")

    print("\n=== SUMMARY ===")
    for line in summary_lines:
        print(" ", line)


if __name__ == "__main__":
    main()
