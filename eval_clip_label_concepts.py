"""
eval_clip_label_concepts.py — Phase H: CLIP-Based Concept Semantic Labelling

Addresses reviewer: "Concepts are free non-negative vectors without clear connection
to visual concepts."

Protocol:
  For each concept c, load its top-activating patch crops (individual files from Phase A+B),
  or fall back to the composite concept_c.png if individual crops are not present.
  Encode each crop with CLIP and compute cosine similarity to a vocabulary of descriptors.
  Assign the best-matching label. Measure "agreement" = fraction of crops that vote for
  the winning label.

Improvements over baseline:
  - Prompt ensembling: each vocabulary phrase is encoded as the mean of multiple
    templates (e.g. "a dermoscopy image showing {label}").  Controlled by --ensemble
    (default: on).  Typically adds +3–8% to zero-shot accuracy on clinical benchmarks.
  - Top-3 labels: each concept entry now contains a top_labels list with the three
    best-matching vocabulary terms and their mean similarity scores, exposing compound
    or ambiguous concepts.
  - Rank-weighted agreement: crops are ranked by NMF activation strength (crop_0 =
    strongest).  Agreement is weighted so stronger crops count more.
  - Unknown escape: each vocabulary contains a "no clear feature" catch-all.  Concepts
    whose best label is the escape label are flagged as label_reliability="low".
  - Duplicate clustering: per-dataset output includes a duplicate_concepts dict grouping
    concept IDs that share the same top-1 CLIP label.
  - Class name propagation: all D_i fields from Phase G (dominant_class_name,
    class_distribution, is_discriminative) are propagated verbatim into the output.

Vocabularies — dataset-specific, grounded in established clinical diagnostic criteria:
  - ham10000 : ABCDE criteria + HAM10000 paper (Tschandl et al. 2018)
  - ph2                            : PH2 annotation schema (Mendonça et al. 2013)
  - derm7pt                        : 7-Point Checklist (Kawahara et al. 2019)
  - imagenet                       : 10 general visual terms

Inputs:
  - {output_root}/{ds}/craft/{ds}/concept_examples/concept_{c}_crop_0.png .. crop_4.png
  - {output_root}/concept_di_scores.json  (from Phase G — merged into output if present)

Output:
  {output_root}/concept_clip_labels.json
"""

import os
import json
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from config import DATASETS
from utils import _set_seed


# ---------------------------------------------------------------------------
# Vocabularies — dataset-specific, grounded in clinical diagnostic literature
# ---------------------------------------------------------------------------

# HAM10000 (Tschandl et al. 2018) — ABCDE criteria + lesion morphology descriptors
# Source: "The HAM10000 dataset, a large collection of multi-source dermatoscopic images
#  of common pigmented skin lesions", Tschandl et al., Scientific Data 2018.
HAM10000_LABELS = [
    # ABCDE criteria
    "asymmetric lesion shape",
    "irregular border",
    "multiple colors",
    "large diameter lesion",
    # Dermoscopy morphology
    "blue-white veil",
    "atypical pigment network",
    "regression structures",
    "dotted or globular vessels",
    "streaks or pseudopods",
    "milia-like cysts",
    "brown globules",
    "dark homogeneous pigmentation",
    "uniform pink or red area",
    "light uniform skin background",
    "vascular structures",
    "hair follicles or skin texture",
    # Escape label — chosen when no clear dermoscopic feature dominates
    "no clear dermoscopic feature",
]

# PH2 (Mendonça et al. 2013) — PH2 annotation schema (8 visual features)
# Source: "PH2 — A dermoscopic image database for research and benchmarking",
#  Mendonça et al., EMBC 2013.
PH2_LABELS = [
    # Pigment network
    "typical pigment network",
    "atypical pigment network",
    "absent pigment network",
    # Dots and globules
    "regular dots and globules",
    "irregular dots and globules",
    # Streaks
    "regular streaks",
    "irregular streaks",
    # Regression areas
    "white scar-like areas",
    "blue-grey peppering",
    # Blue-whitish veil
    "blue-whitish veil",
    # Vascular structures
    "regular vascular structures",
    "irregular vascular structures",
    # Background
    "uniform skin background",
    "hair and follicle structures",
    # Escape label
    "no clear dermoscopic feature",
]

# Derm7pt (Kawahara et al. 2019) — 7-Point Checklist criteria (Argenziano et al. 1998)
# Source: "Seven-Point Checklist and Skin Lesion Classification using Multi-Task
#  Multi-Modal Neural Nets", Kawahara et al., IEEE JBHI 2019.
# The 7 criteria are: atypical pigment network, blue-whitish veil, atypical vascular
# pattern, irregular pigmentation, irregular dots/globules, irregular streaks,
# regression structures — mapped to visual patch descriptions for CLIP.
DERM7PT_LABELS = [
    # The 7 checklist criteria (as visual descriptions for CLIP)
    "atypical irregular pigment network",
    "blue-whitish veil structure",
    "atypical vascular pattern",
    "irregular pigmentation blotches",
    "irregular dots and globules",
    "irregular streaks or pseudopods",
    "regression structures white and grey",
    # Additional common dermoscopy features
    "regular symmetric pigment network",
    "homogeneous brown pigmentation",
    "uniform vascular pattern",
    "light uniform skin background",
    "milia-like cysts",
    "hair follicle openings",
    # Escape label
    "no clear dermoscopic feature",
]

# ImageNet — general visual descriptors (not disease-specific)
IMAGENET_LABELS = [
    "animal fur texture",
    "geometric pattern",
    "background sky",
    "vegetation leaves",
    "water surface",
    "object edge boundary",
    "fine granular texture",
    "circular or round shape",
    "metallic or shiny surface",
    "human body part",
    # Escape label
    "no dominant visual feature",
]

# CUB-200-2011 — short bird visual lexicon (general prompts; not species names)
CUB_LABELS = [
    "feather texture and plumage pattern",
    "beak shape and color",
    "wing and tail feathers",
    "eye ring or facial markings",
    "crown or head coloration",
    "breast and belly plumage",
    "leg and foot color",
    "perching on branch",
    "foliage or tree background",
    "sky or open background",
    "water or wetland background",
    "fine-grained bird silhouette",
    # Escape label
    "no dominant visual feature",
]

# Vocabulary dispatch — each dataset maps to its own literature-grounded vocabulary
_VOCAB_MAP = {
    "ham10000":            (HAM10000_LABELS, "ham10000_abcde"),
    "ph2":                 (PH2_LABELS,      "ph2_annotation_schema"),
    "derm7pt":             (DERM7PT_LABELS,  "derm7pt_7point_checklist"),
    "imagenet":            (IMAGENET_LABELS, "imagenet_visual"),
    "cub":                 (CUB_LABELS,      "cub_bird_visual"),
}

# Escape labels — the final entry in each vocabulary list.  When CLIP's argmax
# lands on the escape label the concept is marked label_reliability="low".
_ESCAPE_LABELS = {
    "no clear dermoscopic feature",
    "no dominant visual feature",
}

# Prompt templates for ensembling.  Medical datasets use dermatology-specific templates;
# imagenet uses generic photo templates.  When --ensemble is on (default), CLIP text
# features are the mean of all applicable templates, which typically gives +3–8% zero-shot.
_PROMPT_TEMPLATES_DERM = [
    "a dermoscopy image showing {label}",
    "a skin lesion patch with {label}",
    "a close-up of {label} under dermoscopy",
    "dermatology: {label}",
]
_PROMPT_TEMPLATES_GENERAL = [
    "a photo of {label}",
    "an image showing {label}",
    "a picture of {label}",
    "{label}",
]
_PROMPT_TEMPLATES_MAP = {
    "ham10000":            _PROMPT_TEMPLATES_DERM,
    "ph2":                 _PROMPT_TEMPLATES_DERM,
    "derm7pt":             _PROMPT_TEMPLATES_DERM,
    "imagenet":            _PROMPT_TEMPLATES_GENERAL,
    "cub":                 _PROMPT_TEMPLATES_GENERAL,
}


def get_vocabulary(ds_key: str):
    """
    Return (labels, vocab_name) for a dataset.
    Each medical dataset uses the vocabulary grounded in its own diagnostic schema:
      ham10000 → ABCDE criteria + HAM10000 paper (Tschandl et al. 2018)
      ph2      → PH2 annotation schema (Mendonça et al. 2013)
      derm7pt  → 7-Point Checklist (Kawahara et al. 2019 / Argenziano et al. 1998)
      imagenet → general visual terms
    """
    if ds_key in _VOCAB_MAP:
        return _VOCAB_MAP[ds_key]
    return HAM10000_LABELS, "ham10000_abcde"


# ---------------------------------------------------------------------------
# CLIP helpers
# ---------------------------------------------------------------------------

def load_clip(model_name: str, device: str):
    """Load CLIP from HuggingFace transformers."""
    from transformers import CLIPModel, CLIPProcessor
    print(f"Loading CLIP: {model_name} ...")
    clip_model = CLIPModel.from_pretrained(model_name).to(device).eval()
    clip_proc  = CLIPProcessor.from_pretrained(model_name)
    return clip_model, clip_proc


@torch.no_grad()
def _encode_texts_raw(clip_model, clip_proc, texts: list, device: str):
    """Encode a list of text strings → normalised feature matrix [N, D]."""
    inputs = clip_proc(text=texts, return_tensors="pt", padding=True).to(device)
    feats  = clip_model.get_text_features(**inputs)
    return feats / feats.norm(dim=-1, keepdim=True)


@torch.no_grad()
def encode_texts_ensembled(clip_model, clip_proc, labels: list,
                           templates: list, device: str):
    """
    Encode vocabulary labels using prompt ensembling.

    For each template, fill in every label and encode the resulting phrases.
    Average the normalised feature vectors across templates, then re-normalise.
    This is the standard CLIP zero-shot protocol and typically gives +3–8%
    over bare-phrase encoding on clinical benchmarks.

    Args:
        labels:    list of vocabulary phrases (e.g. ["blue-white veil", ...])
        templates: list of f-string templates with {label} placeholder
    Returns:
        [len(labels), D] normalised tensor
    """
    all_feats = []
    for tpl in templates:
        prompts = [tpl.format(label=lbl) for lbl in labels]
        feats   = _encode_texts_raw(clip_model, clip_proc, prompts, device)  # [L, D]
        all_feats.append(feats)
    stacked = torch.stack(all_feats, dim=0)   # [T, L, D]
    avg     = stacked.mean(dim=0)              # [L, D]
    return avg / avg.norm(dim=-1, keepdim=True)


@torch.no_grad()
def encode_images(clip_model, clip_proc, images: list, device: str):
    """
    Encode a list of PIL images → normalised feature matrix [N, D].
    images: list of PIL.Image
    """
    inputs = clip_proc(images=images, return_tensors="pt").to(device)
    feats  = clip_model.get_image_features(**inputs)
    return feats / feats.norm(dim=-1, keepdim=True)


# ---------------------------------------------------------------------------
# Load concept crops
# ---------------------------------------------------------------------------

def load_concept_crops(concept_examples_dir: str, concept_id: int, n_crops: int = 5):
    """
    Try loading individual crop files: concept_{c}_crop_0.png .. crop_{n-1}.png
    Fall back to composite: concept_{c}.png
    Returns list of PIL images (may be length 1 if only composite exists).
    """
    crops = []
    for rank in range(n_crops):
        p = os.path.join(concept_examples_dir, f"concept_{concept_id}_crop_{rank}.png")
        if os.path.isfile(p):
            crops.append(Image.open(p).convert("RGB"))

    if not crops:
        composite = os.path.join(concept_examples_dir, f"concept_{concept_id}.png")
        if os.path.isfile(composite):
            crops = [Image.open(composite).convert("RGB")]

    return crops


# ---------------------------------------------------------------------------
# Label one concept
# ---------------------------------------------------------------------------

def label_concept(clip_model, clip_proc, text_features,
                  crops: list, vocab: list, device: str,
                  top_k: int = 3):
    """
    Label a single concept given its crop images.

    Crops are assumed to be ordered strongest-first (crop_0 = highest NMF score),
    which is how build_concept_graphs.py saves them.  Agreement is computed in
    two ways:
      - agreement         : uniform fraction of crops voting for the best label
      - weighted_agreement: rank-weighted fraction (crop_0 weight = n, crop_n-1 = 1)

    Returns dict with:
      label              (str)         — best-matching vocabulary term
      top_labels         (list[dict])  — top_k labels with scores
      agreement          (float)       — uniform crop agreement
      weighted_agreement (float)       — rank-weighted crop agreement
      clip_confidence    (float)       — mean similarity for best label
      per_crop_labels    (list[str])   — per-crop winning label
      label_reliability  (str)         — "low" when best label is an escape label
    """
    if not crops:
        return {
            "label": None, "top_labels": [], "agreement": 0.0,
            "weighted_agreement": 0.0, "clip_confidence": 0.0,
            "per_crop_labels": [], "label_reliability": "low",
        }

    n_crops = len(crops)
    img_features = encode_images(clip_model, clip_proc, crops, device)  # [n_crops, D]
    sims = (img_features @ text_features.T).cpu().numpy()               # [n_crops, n_labels]

    # Per-crop winning label (uniform)
    per_crop_idx    = sims.argmax(axis=1)                               # [n_crops]
    per_crop_labels = [vocab[i] for i in per_crop_idx]

    # Best label via mean similarity across all crops
    mean_sims = sims.mean(axis=0)                                       # [n_labels]
    best_idx  = int(mean_sims.argmax())
    best_label = vocab[best_idx]
    clip_conf  = float(mean_sims[best_idx])

    # Uniform agreement
    agreement = float((per_crop_idx == best_idx).mean())

    # Rank-weighted agreement — crop_0 (strongest) has highest weight
    rank_weights = np.arange(n_crops, 0, -1, dtype=float)  # [n, n-1, ..., 1]
    rank_weights /= rank_weights.sum()
    weighted_agreement = float(rank_weights[per_crop_idx == best_idx].sum())

    # Top-k labels
    top_idx = mean_sims.argsort()[::-1][:top_k]
    top_labels = [
        {"label": vocab[int(i)], "score": round(float(mean_sims[i]), 4)}
        for i in top_idx
    ]

    # Escape label check
    reliability = "low" if best_label in _ESCAPE_LABELS else "high"

    return {
        "label":              best_label,
        "top_labels":         top_labels,
        "agreement":          round(agreement, 3),
        "weighted_agreement": round(weighted_agreement, 3),
        "clip_confidence":    round(clip_conf, 4),
        "per_crop_labels":    per_crop_labels,
        "label_reliability":  reliability,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser("CLIP concept labelling (Phase H)")
    ap.add_argument("--datasets",            nargs="+",
                    default=["ham10000", "ph2", "derm7pt", "imagenet"])
    ap.add_argument("--output-root",         required=True,
                    help="Absolute netscratch path (same as build_concept_graphs.py)")
    ap.add_argument("--device",              default="cuda")
    ap.add_argument("--agreement-threshold", type=float, default=0.5,
                    help="Agreement threshold for 'high-agreement' count (default: 0.5)")
    ap.add_argument("--clip-model",          default="openai/clip-vit-base-patch32")
    ap.add_argument("--ensemble",            type=int, default=1,
                    help="Use prompt ensembling (1=on, 0=off). Default: 1.")
    ap.add_argument("--top-k-labels",        type=int, default=3,
                    help="Number of top CLIP labels to store per concept. Default: 3.")
    ap.add_argument("--seed",                type=int, default=42)
    args = ap.parse_args()

    _set_seed(args.seed)
    device = args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
    use_ensemble = bool(args.ensemble)

    # Load CLIP once and reuse for all datasets
    clip_model, clip_proc = load_clip(args.clip_model, device)
    print(f"  Prompt ensembling: {'ON' if use_ensemble else 'OFF'}")

    # Load Phase G D_i scores if available (merge into output)
    di_path = os.path.join(args.output_root, "concept_di_scores.json")
    di_scores = {}
    if os.path.isfile(di_path):
        with open(di_path) as f:
            di_scores = json.load(f)
        print(f"Loaded D_i scores from: {di_path}")
    else:
        print(f"[INFO] concept_di_scores.json not found at {di_path} — D_i will not be merged.")

    all_results = {}

    for ds_key in args.datasets:
        if ds_key not in DATASETS:
            print(f"[SKIP] Unknown dataset: {ds_key}")
            continue

        vocab, vocab_name = get_vocabulary(ds_key)
        run_id = ds_key
        concept_examples_dir = os.path.join(
            args.output_root, ds_key, "craft", run_id, "concept_examples"
        )

        if not os.path.isdir(concept_examples_dir):
            print(f"\n[SKIP] {ds_key}: concept_examples dir not found: {concept_examples_dir}")
            continue

        print(f"\n{'='*60}")
        print(f"Dataset: {ds_key}  |  vocabulary: {vocab_name}  ({len(vocab)} labels)")
        print(f"Concept examples: {concept_examples_dir}")

        # Encode text vocabulary once per dataset (with or without ensembling)
        templates = _PROMPT_TEMPLATES_MAP.get(ds_key, _PROMPT_TEMPLATES_GENERAL)
        if use_ensemble:
            text_features = encode_texts_ensembled(
                clip_model, clip_proc, vocab, templates, device
            )
            print(f"  Prompt templates ({len(templates)}): {templates}")
        else:
            text_features = _encode_texts_raw(clip_model, clip_proc, vocab, device)

        # Determine K from existing concept files
        existing = [f for f in os.listdir(concept_examples_dir)
                    if f.startswith("concept_") and f.endswith(".png")]
        if not existing:
            print(f"  [SKIP] No concept images in {concept_examples_dir}")
            continue

        # Collect concept IDs (from either crop or composite files)
        concept_ids = set()
        for fname in existing:
            parts = fname.replace(".png", "").split("_")
            try:
                concept_ids.add(int(parts[1]))
            except (IndexError, ValueError):
                pass
        concept_ids = sorted(concept_ids)
        K = len(concept_ids)
        print(f"  Found {K} concept IDs: {concept_ids[:10]}{'...' if K > 10 else ''}")

        # D_i lookup for this dataset (keyed by concept_id int)
        ds_di = {c["concept_id"]: c for c in di_scores.get(ds_key, {}).get("concepts", [])}

        ds_concepts = []
        for c_id in concept_ids:
            crops = load_concept_crops(concept_examples_dir, c_id, n_crops=5)
            if not crops:
                print(f"  [SKIP] No crops for concept {c_id}")
                continue

            result = label_concept(
                clip_model, clip_proc, text_features, crops, vocab, device,
                top_k=args.top_k_labels,
            )

            entry = {
                "concept_id":         c_id,
                "label":              result["label"],
                "top_labels":         result["top_labels"],
                "agreement":          result["agreement"],
                "weighted_agreement": result["weighted_agreement"],
                "clip_confidence":    result["clip_confidence"],
                "per_crop_labels":    result["per_crop_labels"],
                "num_crops":          len(crops),
                "label_reliability":  result["label_reliability"],
            }

            # Propagate ALL D_i fields from Phase G (dominant_class is now always set)
            if c_id in ds_di:
                di = ds_di[c_id]
                entry["D_i"]                = di["D_i"]
                entry["is_discriminative"]  = di.get("is_discriminative",
                                                      di["D_i"] >= 0.6)
                entry["dominant_class"]     = di["dominant_class"]
                entry["dominant_class_name"]= di.get("dominant_class_name",
                                                      str(di["dominant_class"]))
                entry["class_distribution"] = di.get("class_distribution", [])
                entry["class_names"]        = di.get("class_names", [])

            ds_concepts.append(entry)

            dom_str = (f" dom={entry.get('dominant_class_name','—')}"
                       f"(D_i={entry.get('D_i','—')})")
            rel = f" [{result['label_reliability']}]"
            print(f"  Concept {c_id:>3}: {result['label']:<40}"
                  f"  agree={result['agreement']:.2f}"
                  f"  w_agree={result['weighted_agreement']:.2f}"
                  f"{dom_str}{rel}")

        # Summary stats for this dataset
        n_high_agree = sum(1 for c in ds_concepts
                           if c["agreement"] >= args.agreement_threshold)
        n_low_rel = sum(1 for c in ds_concepts
                        if c["label_reliability"] == "low")
        print(f"\n  High-agreement (≥{args.agreement_threshold}): "
              f"{n_high_agree}/{len(ds_concepts)}")
        print(f"  Low reliability (escape label): {n_low_rel}/{len(ds_concepts)}")

        # Cluster duplicate concepts by top-1 CLIP label
        label_to_ids: dict = defaultdict(list)
        for c in ds_concepts:
            if c["label"] and c["label_reliability"] == "high":
                label_to_ids[c["label"]].append(c["concept_id"])
        duplicate_concepts = {lbl: ids for lbl, ids in label_to_ids.items()
                              if len(ids) > 1}
        if duplicate_concepts:
            print(f"  Duplicate concepts (same top-1 label):")
            for lbl, ids in duplicate_concepts.items():
                print(f"    '{lbl}' → concepts {ids}")

        all_results[ds_key] = {
            "vocabulary":          vocab_name,
            "num_labels_in_vocab": len(vocab),
            "agreement_threshold": args.agreement_threshold,
            "ensemble":            use_ensemble,
            "clip_model":          args.clip_model,
            "concepts":            ds_concepts,
            "num_high_agreement":  n_high_agree,
            "num_low_reliability": n_low_rel,
            "duplicate_concepts":  duplicate_concepts,
        }

    if not all_results:
        print("\nNo results produced. Check concept_examples directories exist.")
        return

    # Save JSON
    out_path = os.path.join(args.output_root, "concept_clip_labels.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Print final summary
    print("\n=== SUMMARY ===")
    for ds_key, result in all_results.items():
        K   = len(result["concepts"])
        ha  = result["num_high_agreement"]
        lr  = result["num_low_reliability"]
        dup = sum(len(v) for v in result["duplicate_concepts"].values())
        print(f"  {ds_key}: {K} concepts  |  high-agree={ha}/{K}"
              f"  |  low-rel={lr}/{K}  |  duplicate-concept-ids={dup}")

        # Highlight discriminative + labelled concepts (paper table candidates)
        disc_with_label = [
            c for c in result["concepts"]
            if c.get("is_discriminative", False)
            and c["agreement"] >= args.agreement_threshold
            and c["label_reliability"] == "high"
        ]
        if disc_with_label:
            print(f"    Paper-table candidates (discriminative + high-agree + reliable): "
                  f"{len(disc_with_label)}/{K}")
            for c in disc_with_label:
                print(f"      c{c['concept_id']:>2}: '{c['label']}'  "
                      f"D_i={c.get('D_i','?'):.3f}  "
                      f"dom={c.get('dominant_class_name','—')}  "
                      f"agree={c['agreement']:.2f}")


if __name__ == "__main__":
    main()
