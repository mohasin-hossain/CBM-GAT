import os
import shutil
import argparse
import json
import numpy as np
import torch
from PIL import Image as PILImage
from utils import _save_concepts, _reverse_preprocess
from concepts import build_model_parts, fit_craft_for_k, auto_select_k, save_craft_light, write_best_k
from config import DATASETS, default_output_dir

def parse_args():
    p = argparse.ArgumentParser(description="craft-based concept discovery + genereating dataset for graph with DGL")
    p.add_argument("--dataset", required=True, choices=list(DATASETS.keys()))
    p.add_argument("--steps", nargs="+", default=["gen_concepts", "build_graphs"],
                   choices=["gen_concepts", "build_graphs"])
    p.add_argument("--output-root", default=default_output_dir)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--backbone", default="resnet50")
    # concept discovery params
    p.add_argument("--n-components", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64, help="craft fitting batch size")
    p.add_argument("--auto-n-components", action="store_true", help="select best #concepts using discriminativeness score")
    p.add_argument("--candidates", type=int, nargs="*", default=[6,7,8,9,10])
    # sliding window params
    p.add_argument("--patch-size", type=int, default=70)  # your default in examples
    p.add_argument("--stride-r", type=float, default=0.8)
    # per-patch sim threshold (honoured by graph_v4 natively and by graph_v1_threshold)
    p.add_argument("--sim-threshold", type=float, default=0.0,
                   help="Per-patch concept-activation threshold. Values below this are zeroed "
                        "before node-feature aggregation. Honoured by graph_v4 and "
                        "graph_v1_threshold; ignored by graph_v1. 0.0 = no thresholding.")
    # reuse craft if already fitted
    p.add_argument("--craft-path", default=None, help="If provided, load this craft .dill and skip fitting")
    # individual crop saving (needed by Phase H CLIP labelling)
    p.add_argument("--save-individual-crops", action="store_true",
                   help="Save top-5 individual patch crops per concept as separate PNGs for CLIP labelling")
    p.add_argument(
        "--concept-bottleneck-mlp-linear",
        action="store_true",
        help="Use graph_concept_bottleneck_mlp_linear (K-dim z, single-node graphs). "
             "Sets CBM_GRAPH_VARIANT for this process and writes under "
             "graphs_concept_bottleneck_mlp_linear/<run_id>/.",
    )
    return p.parse_args()


def save_individual_concept_crops(crops: np.ndarray,
                                   crops_u: torch.Tensor,
                                   concept_examples_dir: str,
                                   top_k: int = 5):
    """
    Save top-k individual patch crops per concept as separate PNG files.

    crops:              np.ndarray [num_patches, H, W, C] (float, un-normalised, values 0..1 after reverse-preprocess)
    crops_u:            torch.Tensor [num_patches, K] — NMF activation scores
    concept_examples_dir: directory where concept_N.png composites are already saved
    top_k:              number of individual crops to save per concept

    Saves: concept_{c}_crop_{rank}.png  (rank = 0 is the strongest activating patch)
    These are used by Phase H (CLIP labelling) for per-crop agreement scoring.
    """
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    crops_u_np = crops_u.numpy() if hasattr(crops_u, 'numpy') else np.array(crops_u)
    K = crops_u_np.shape[1]

    os.makedirs(concept_examples_dir, exist_ok=True)

    for c in range(K):
        scores = crops_u_np[:, c]
        top_indices = scores.argsort()[::-1][:top_k]

        for rank, patch_global_idx in enumerate(top_indices):
            crop = np.array(crops[patch_global_idx])  # [H, W, C] float

            # Reverse the ImageNet normalisation so the saved PNG looks correct
            img_arr = _reverse_preprocess(crop, mean, std)  # values in [0, 1]
            img_arr = (img_arr * 255).clip(0, 255).astype(np.uint8)

            pil_img = PILImage.fromarray(img_arr)
            save_path = os.path.join(concept_examples_dir, f"concept_{c}_crop_{rank}.png")
            pil_img.save(save_path)

    print(f"  Saved individual crops ({top_k} per concept, {K} concepts) to: {concept_examples_dir}")


def main():
    args = parse_args()
    if getattr(args, "concept_bottleneck_mlp_linear", False):
        os.environ["CBM_GRAPH_VARIANT"] = "concept_bottleneck_mlp_linear"
    ds_spec = DATASETS[args.dataset]

    tdict = ds_spec.build_transforms()
    paths = ds_spec.resolve_paths()

    # step 1: concept discovery with NMF using Craft
    craft_path = args.craft_path
    run_id = f"{args.dataset}"

    current_dir = os.getcwd()
    args.output_root = os.path.join(current_dir, args.output_root)

    craft_dir = os.path.join(args.output_root, args.dataset, "craft", run_id)
    os.makedirs(craft_dir, exist_ok=True)
    default_craft_file = os.path.join(craft_dir, f"craft_{args.dataset}.dill")
    
    concept_example_save_dir = os.path.join(craft_dir, "concept_examples")
    if os.path.exists(concept_example_save_dir):
        shutil.rmtree(concept_example_save_dir)
    os.makedirs(concept_example_save_dir)

    if "gen_concepts" in args.steps and craft_path is None:
        print("Concept discovery with Craft (NMF split)")
        images_nmf, labels_nmf, _ = ds_spec.load_split(paths, tdict, split="nmf")
        if images_nmf.numel() == 0:
            raise RuntimeError("Loaded 0 images for NMF. Check CSV paths in dataset_registry.py.")
        g, h = build_model_parts(args.backbone, device=args.device, pretrained=True)

        if args.auto_n_components and args.candidates:
            best_k, table = auto_select_k(
                images=images_nmf,
                labels=labels_nmf,
                candidates=args.candidates,
                patch_size=args.patch_size,
                batch_size=args.batch_size,
                device=args.device,
                g=g, h=h
            )
            with open(os.path.join(craft_dir, "concept_search.json"), "w") as f:
                json.dump(table, f, indent=2)
            print(f"Selected best k={best_k}")
            k = best_k
            write_best_k(craft_dir, best_k, args.patch_size,args.stride_r)
        else:
            k = args.n_components

        craft, crops, crops_u = fit_craft_for_k(
            images=images_nmf,
            k=k,
            patch_size=args.patch_size,
            batch_size=args.batch_size,
            device=args.device,
            g=g, h=h)

        save_craft_light(craft, default_craft_file)
        _save_concepts(crops, crops_u, reverse=True, start=0, nb_crops=5, save=True, save_dir=concept_example_save_dir)
        print(f"Saved Craft (light) to: {default_craft_file}")

        if args.save_individual_crops:
            save_individual_concept_crops(
                crops=crops,
                crops_u=crops_u,
                concept_examples_dir=concept_example_save_dir,
                top_k=5,
            )

        craft_path = default_craft_file
    elif craft_path is None:
        if not os.path.isfile(default_craft_file):
            raise FileNotFoundError("craft file not found. Provide --craft-path or run gen_concepts first.")
        craft_path = default_craft_file

    # Step 2: build graphs per split (reuse the same craft)
    if "build_graphs" in args.steps:
        import graph as graph_mod

        build_and_save_graphs_per_split = graph_mod.build_and_save_graphs_per_split
        if getattr(args, "concept_bottleneck_mlp_linear", False):
            from graph_concept_bottleneck_mlp_linear import GRAPHS_SUBDIR

            graphs_dir = os.path.join(
                args.output_root, args.dataset, GRAPHS_SUBDIR, run_id
            )
        else:
            graphs_dir = os.path.join(args.output_root, args.dataset, "graphs", run_id)
        os.makedirs(graphs_dir, exist_ok=True)

        for split in ["train", "val", "test"]:
            print(f"Building graphs for split: {split}")
            images_s, labels_s, _ = ds_spec.load_split(paths, tdict, split=split)
            if images_s.numel() == 0:
                print(f"No images for split: {split}")
                continue

            out_file = os.path.join(graphs_dir, f"concept_graphs_{'validation' if split=='val' else split}.dgl")
            out_file, n_graphs = build_and_save_graphs_per_split(
                images=images_s,
                labels=labels_s,
                device=args.device,
                backbone_name=args.backbone,
                craft_path=craft_path,
                out_path=out_file,
                patch_size=args.patch_size,
                stride_r=args.stride_r,
                ignore_list=[],
                coverage_threshold=0.0,
                sim_threshold=args.sim_threshold,
            )
            print(f"  Saved {n_graphs} graphs to: {out_file}")


if __name__ == "__main__":
    main()

# python concept_data_generator.py --dataset ham10000 --steps gen_concepts build_graphs --auto-n-components --candidates 6 7 8 9 10 12 16 --patch-size 70 --stride-r 0.8 --batch-size 64 --device cuda --output-root /mnt/sda/anar-data/concept_graph_data