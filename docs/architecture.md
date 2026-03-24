# CBM-GAT Architecture Guide

## Table of Contents

1. [Repository Overview](#1-repository-overview)
2. [Key Technologies](#2-key-technologies)
3. [Code Organization](#3-code-organization)
4. [High-Level Architecture](#4-high-level-architecture)
5. [Pipeline A — Concept Discovery (CRAFT + NMF)](#5-pipeline-a--concept-discovery-craft--nmf)
6. [Pipeline B — Concept Graph Construction](#6-pipeline-b--concept-graph-construction)
7. [Pipeline C — GAT Training](#7-pipeline-c--gat-training)
8. [Pipeline D — Single-Image Explanation](#8-pipeline-d--single-image-explanation)
9. [Grad-CAM Explanation Modes](#9-grad-cam-explanation-modes)
10. [CNN Baseline](#10-cnn-baseline)
11. [Evaluation Pipelines](#11-evaluation-pipelines)
12. [Data Layout and Transforms](#12-data-layout-and-transforms)
13. [Design Decisions and Rationale](#13-design-decisions-and-rationale)
14. [End-to-End Data Flow](#14-end-to-end-data-flow)
15. [Configuration Reference](#15-configuration-reference)

---

## 1. Repository Overview

**CBM-GAT** is a master's thesis project (conducted at DFKI) that builds an *explainable image-classification framework* by combining two complementary ideas:

| Idea | What it solves |
|------|---------------|
| **Visually grounded concept discovery** via Non-negative Matrix Factorisation (NMF) through CRAFT | Avoids costly human concept annotations and removes reliance on textual labels |
| **Graph Attention Network (GAT)** classifier over a *concept graph* | Captures inter-concept relationships while keeping the model shallow and interpretable |

The result is a pipeline that (a) discovers `k` visual concepts automatically from training images, (b) encodes each image as a small graph whose nodes are those concepts, and (c) classifies the graph while producing human-interpretable attention weights and gradient-based concept-importance scores.

---

## 2. Key Technologies

| Technology | Version | Role |
|---|---|---|
| **Python** | 3.10.x | Runtime |
| **PyTorch** | 2.4.0 | Tensor operations, CNN backbone, autograd |
| **PyTorch Lightning** | 2.5.0 | Training loop, checkpointing, early stopping |
| **DGL** (Deep Graph Library) | 2.4.0+cu118 | Graph construction, GAT layers, graph-level pooling |
| **torch-geometric** | 2.6.1 | `GraphNorm` normalisation layer used inside the GAT |
| **torchvision** | 0.19.0 | ResNet-50 backbone, image transforms |
| **CRAFT-xai** | 0.0.3 | CRAFT concept-factorisation framework (NMF core) |
| **scikit-learn** | 1.6.1 | Evaluation metrics (accuracy, F1, AUC) |
| **torchmetrics** | 1.6.1 | Online accuracy tracking during training |
| **NumPy / Pandas** | 1.26.4 / 2.2.3 | Numerical operations, CSV handling |
| **Matplotlib / Pillow** | 3.10.0 / 11.0.0 | Visualisation and image I/O |
| **dill** | latest | Serialisation of CRAFT objects (supports lambdas/closures) |
| **TensorFlow** | 2.15.x | Required internally by CRAFT-xai |

---

## 3. Code Organization

```
CBM-GAT/
│
├── config.py                  # Dataset registry, transforms, model hyperparameters
├── concepts.py                # CRAFT fitting, NMF scoring, auto-k selection, save/load helpers
├── graph.py                   # ConceptGraphDataset builder + LoadConceptGraphDataset loader
├── model.py                   # EGATClassifier (GAT) + GAT_LightningModule (trainer)
├── utils.py                   # ImageDataset, tensor helpers, concept-crop saving
│
├── build_concept_graphs.py    # CLI: Pipeline A + B (concept discovery + graph construction)
├── train_model.py             # CLI: Pipeline C (GAT training + evaluation)
├── train_cnn.py               # CLI: ResNet-50 CNN baseline training
├── explain_image.py           # CLI: Pipeline D (single-image concept explanation)
├── gradcam_explain.py         # CLI: Grad-CAM heatmaps and side-by-side comparisons
│
├── eval_benchmark.py          # CLI: End-to-end multi-run benchmark (A+B+C aggregated)
├── eval_fidelity.py           # CLI: Insertion/deletion fidelity curves
├── eval_concept_quality.py    # CLI: Concept quality metrics (disentanglement)
│
├── datasets/                  # CSV split files (train/val/test/nmf) for all datasets
│   ├── ph2dataset/
│   ├── ham10000/
│   ├── derm7pt/
│   └── imagenet/
│
├── concept_graph_data/        # Default output root for trained artifacts
│   └── <dataset>/
│       ├── craft/<dataset>/
│       │   ├── craft_<dataset>.dill        # Serialised CRAFT/NMF object (lightweight)
│       │   ├── U_meta/nmf_best_k.json      # Best k, patch_size, stride_r
│       │   ├── concept_examples/           # Top-5 patch crops per concept (PNG)
│       │   └── concept_search.json         # k-candidate scoring table
│       ├── graphs/<dataset>/
│       │   ├── concept_graphs_train.dgl
│       │   ├── concept_graphs_validation.dgl
│       │   └── concept_graphs_test.dgl
│       └── models/<dataset>/
│           ├── <dataset>_best_model.ckpt   # Lightning checkpoint (weights only)
│           └── metrics.json               # acc / f1 / auc for train/val/test
│
├── results/                   # Benchmark aggregated results
│   ├── results.csv
│   └── results.json
│
├── assets/                    # Example explanation images
└── requirements.txt
```

### Layer map

| Layer | Files | Responsibility |
|---|---|---|
| **Data** | `config.py`, `utils.py` | Dataset registry, transforms, CSV loading |
| **Concept** | `concepts.py` | ResNet-50 feature extraction, NMF via CRAFT, auto-k selection |
| **Graph** | `graph.py` | Patch-to-concept mapping, DGL graph building/saving/loading |
| **Model** | `model.py` | GAT architecture, Lightning training wrapper |
| **Pipelines** | `build_concept_graphs.py`, `train_model.py`, `explain_image.py`, `gradcam_explain.py` | Orchestration scripts |
| **Evaluation** | `eval_benchmark.py`, `eval_fidelity.py`, `eval_concept_quality.py` | Reproducibility and metrics |

---

## 4. High-Level Architecture

```
                 ┌─────────────────────────────────────────────────┐
                 │                  Raw Images                      │
                 └───────────────────────┬─────────────────────────┘
                                         │
                         ┌───────────────▼───────────────┐
                         │    ResNet-50  (frozen)         │
                         │    input_to_latent (g)         │
                         │    [B, 2048, 7, 7] feature maps│
                         └───────────────┬───────────────┘
                                         │  spatial avg
                         ┌───────────────▼───────────────┐
                         │    CRAFT NMF Reducer           │
                         │    (k components)              │
                         │    patches_U  [P, k]           │
                         └───────────────┬───────────────┘
                                         │  per-image
                         ┌───────────────▼───────────────┐
                         │    Concept Graph (DGL)         │
                         │    k nodes, fully connected    │
                         │    node_feat [k, 2048]         │
                         └───────────────┬───────────────┘
                                         │
                         ┌───────────────▼───────────────┐
                         │    GATConv  (1 layer)          │
                         │    + GraphNorm + ELU           │
                         └───────────────┬───────────────┘
                                         │  mean-node readout
                         ┌───────────────▼───────────────┐
                         │    Linear  →  class logits     │
                         └───────────────┬───────────────┘
                                         │
                         ┌───────────────▼───────────────┐
                         │  CrossEntropyLoss + (opt L1)   │
                         └─────────────────────────────────┘
```

---

## 5. Pipeline A — Concept Discovery (CRAFT + NMF)

**Entry point:** `build_concept_graphs.py --steps gen_concepts`  
**Core logic:** `concepts.py`

### What it does

1. **Load NMF split images** – the `_all_balanced.csv` or `nmf.csv` split, loaded via `config.py`.
2. **Build CNN parts** (`build_model_parts`) – split ResNet-50 into two halves:
   - `g`: layers up to (but not including) the final avg-pool/FC → outputs feature maps `[B, 2048, H, W]`.
   - `h`: average pooling + FC → outputs logits (used internally by CRAFT for gradient attribution).
3. **Extract patches** – a sliding window of size `patch_size` with stride `patch_size × stride_r` is applied to every image.
4. **Run CRAFT** (`fit_craft_for_k`) – CRAFT passes each patch through `g`, spatially averages the activations to `[2048]`, then fits an NMF model with `k` components. The output `crops_u` has shape `[num_patches, k]` — the coefficient of each patch under each concept basis vector.
5. **(Optional) Auto-select k** (`auto_select_k`) – CRAFT is fitted for each candidate `k` in `--candidates`. The **discriminativeness score** picks the `k` that maximises:
   ```
   score = avg_D_i  −  λ · penalty
   ```
   where `D_i = max_c R_ic` (the maximum fraction of concept-`i` patches belonging to a single class), and `penalty` measures how unevenly the discriminative concepts are distributed across classes.
6. **Save artifacts**:
   - `craft_<dataset>.dill` – a lightweight serialised CRAFT object (CNN parts stripped to save space, re-attached at load time).
   - `U_meta/nmf_best_k.json` – `{ best_k, patch_size, stride_r }`.
   - `concept_examples/concept_<c>.png` – top-5 patch crops visualising each concept.

### Flow diagram

```
NMF-split images
      │
      ▼
 sliding window → patches [P, C, H, W]
      │
      ▼
   g(patch) → feature map → spatial avg → [P, 2048]
      │
      ▼
  NMF.fit()  → W [P, k], H [k, 2048]
              crops_u = W  (patch concept coefficients)
      │
      ▼
  discriminativeness score (per candidate k)
      │
      ▼
  best k selected → craft saved
```

---

## 6. Pipeline B — Concept Graph Construction

**Entry point:** `build_concept_graphs.py --steps build_graphs`  
**Core logic:** `graph.py` → `ConceptGraphDataset`

### What it does

For each data split (train / val / test):

1. **Load images** via `config.py` (eval transform, no augmentation).
2. **Re-attach CNN** to the saved CRAFT object (`load_craft_and_attach`).
3. **Extract patches** from each image with the same `patch_size` / `stride_r` as in Pipeline A.
4. **Compute patch activations** – pass patches through `g`, spatially average → `[P, 2048]`.
5. **Project onto concept basis** – `patches_U = NMF.transform(activations)` → `[P, k]`. Each row gives the concept loading for one patch.
6. **Build a complete DGL graph** with `k` nodes (one per concept):
   - Edges: all `k × k` pairs (including self-loops), giving a fully connected concept graph.
   - **Node feature** for concept `c`:
     ```python
     node_feature[c] = GELU( mean_over_patches( activation[p] × patches_U[p, c] ) )
     ```
     This is a weighted average of patch activations, where the weight of patch `p` is its NMF coefficient for concept `c`. Shape: `[2048]`.
7. **Z-score normalise** node features across all graphs in the split using the training set statistics.
8. **Save** graphs as `concept_graphs_{split}.dgl` using `dgl.save_graphs`.

### Graph structure

| Property | Value |
|---|---|
| Nodes | `k` (one per discovered concept) |
| Edges | `k × k` (fully connected, including self-loops) |
| Node feature dim | `2048` (ResNet-50 channel depth) |
| Node feature semantics | Weighted mean activation for that concept across all patches of the image |

### Flow diagram

```
Image (eval-transformed)
      │
      ▼
 sliding window → patches [P, 3, pH, pH]
      │
      ▼
  g(patches) → [P, 2048, h, w] → spatial avg → [P, 2048]
      │
      ▼
  NMF.transform() → patches_U [P, k]
      │
      ▼
  For each concept c:
    node_feat[c] = GELU( mean_p( act[p] × U[p,c] ) )   shape [2048]
      │
      ▼
  DGL graph: k nodes, k×k edges, ndata['feat'] = node_feat
      │
      ▼
  dgl.save_graphs(concept_graphs_{split}.dgl)
```

---

## 7. Pipeline C — GAT Training

**Entry point:** `train_model.py`  
**Core logic:** `model.py` → `EGATClassifier` + `GAT_LightningModule`

### Model architecture (`EGATClassifier`)

```
Input: DGL batched graph, node features [N_total, 2048]
  │
  ▼
GATConv(in=2048, out=hidden_dim, heads=num_heads)
  │    returns (h, attn)   shapes: [N, heads, hidden_dim], [E, heads, 1]
  │
  ▼
mean over heads → [N, hidden_dim]
  │
  ▼
GraphNorm([N, hidden_dim])      # normalise per graph
  │
  ▼
ELU activation → [N, hidden_dim]
  │
  ▼
dgl.mean_nodes()               # graph-level readout → [B, hidden_dim]
  │
  ▼
Linear(hidden_dim → num_classes)  # classification head
  │
  ▼
logits [B, num_classes]
```

### Training wrapper (`GAT_LightningModule`)

| Component | Choice | Reason |
|---|---|---|
| Loss | `CrossEntropyLoss` (+ optional L1 on node features) | Standard multi-class loss; L1 encourages sparse node representations |
| Optimiser | `AdamW` | Weight decay built-in, stable on graph data |
| LR schedule | `CosineAnnealingLR(T_max=epochs, eta_min=lr/50)` | Smooth decay, avoids sharp LR drops |
| Early stopping | `patience` epochs on `val_loss` | Prevents overfitting on small medical datasets |
| Checkpoint | `ModelCheckpoint` monitors `val_loss`, saves weights only | Smallest checkpoint size |

### Dataset-specific hyperparameters (from `config.py`)

| Dataset | `num_heads` | `hidden_dim` | `batch_size` |
|---|---|---|---|
| PH2 | 4 | 128 | 64 |
| Derm7pt | 4 | 128 | 32 |
| HAM10000 | 6 | 128 | 128 |
| ImageNet subset | 6 | 128 | 128 |

### Training flow

```
concept_graphs_train.dgl  ──►  GraphDataLoader(shuffle=True)
concept_graphs_validation.dgl ► GraphDataLoader(shuffle=False)
                                        │
                              pl.Trainer.fit(max_epochs, patience)
                                        │
                               Best checkpoint (val_loss)
                                        │
                              evaluate() on train / val / test
                                        │
                              metrics.json  {acc, f1, auc}
```

---

## 8. Pipeline D — Single-Image Explanation

**Entry point:** `explain_image.py`  
**Core logic:** inline in `explain_image.py`, calling `graph.py` and `model.py`

### Steps

1. **Load artifacts** – CRAFT (via `load_craft`), trained GAT (via `load_trained_gat`).
2. **Build graph** from the single image using `build_graph_from_single_image`:
   - Apply eval transform → build `ConceptGraphDataset` with a dummy label.
   - Returns the DGL graph and `patches_U [P, k]`.
3. **Forward pass** – run `gat_model(graph, node_f)` to get `logits`, attention scores `attn_scr1`, and node embeddings `h`.
4. **Gradient-based node importance**:
   ```python
   target_prob = softmax(logits)[pred_idx]
   grads = autograd.grad(target_prob, node_f)
   node_importance = grads.abs().sum(dim=1)   # L1 norm of gradient per node
   node_importance /= node_importance.sum()   # normalise to sum to 1
   ```
5. **Rank concepts** – sort nodes by importance; keep top `top_k_max` nodes above `min_concept_weight`.
6. **Patch importance** – `patch_importance = patches_U @ node_importance`; sort patches.
7. **Visualise** – three-panel figure:
   - Left: image with top-`k` patch bounding boxes coloured by concept.
   - Middle: horizontal bar chart of concept importance scores.
   - Right: example patches for each top concept (loaded from `concept_examples/`).

### Output

`output_explanation.png` saved in the current working directory.

### Explanation flow

```
image.jpg
    │
    ▼
eval_transform → tensor [1,3,224,224]
    │
    ▼
ConceptGraphDataset.process() → DGL graph + patches_U
    │
    ▼
gat_model(graph, node_f.requires_grad_())
    │
    ├─► logits → softmax → predicted class + confidence
    │
    └─► autograd.grad(prob[pred], node_f)
              │
              ▼
         node_importance = |grad|.sum(dim=1)
              │
              ├─► top concepts (sorted by importance)
              │
              └─► patch_importance = patches_U @ node_importance
                       │
                       ▼
                  coloured patch boxes on image  +  bar chart  +  example patches
```

---

## 9. Grad-CAM Explanation Modes

**Entry point:** `gradcam_explain.py --mode <mode>`

Grad-CAM provides spatial explanations (*where* the CNN looks), complementing CBM-GAT's concept explanations (*which concepts* matter).

### Available modes

| Mode | CNN source | CBM-GAT shown | Panels |
|---|---|---|---|
| `standalone_spatial` | ImageNet pretrained ResNet-50 | No | 3 (original, heatmap, overlay) |
| `standalone_cnn` | Fine-tuned CNN baseline | No | 3 |
| `standalone_medical` | CBM-GAT patch-importance heatmap | Yes (patch scores) | 3 |
| `compare_spatial_concepts` | ImageNet pretrained | Yes (concept bar + patches) | 4 |
| `compare_cnn_concepts` | Fine-tuned CNN baseline | Yes | 4 |
| `compare_medical_concepts` | CBM-GAT patch-importance | Yes | 4 |

### Output file naming

```
output_gradcam_spatial.png
output_gradcam_cnn.png
output_gradcam_medical.png
output_gradcam_vs_concepts_spatial.png
output_gradcam_vs_concepts_cnn.png
output_gradcam_vs_concepts_medical.png
```

Optional flags:
- `--save-concept-heatmaps` — per-concept jet overlays
- `--save-concept-heatmap-multi-boxes` — multiple distinct patch boxes per concept (greedy selection with minimum spacing)
- `--save-concept-multi-boxes-on-image` — multi-box selection drawn on the raw image

---

## 10. CNN Baseline

**Entry point:** `train_cnn.py`

A ResNet-50 classifier trained **directly on images** (no concept graphs, no GAT). Used as a performance and Grad-CAM comparison baseline.

- Same train/val/test CSV splits as in `config.py`.
- Output: `metrics_cnn.json` and `{dataset}_resnet50_cnn.pt` checkpoint.
- Checkpoint format: `{"state_dict": ..., "num_classes": N}`.
- The `.pt` file is loaded automatically by `gradcam_explain.py` for `standalone_cnn` / `compare_cnn_concepts` modes.

---

## 11. Evaluation Pipelines

### 11.1 End-to-end benchmark (`eval_benchmark.py`)

Runs the full Pipeline A → B → C for each dataset, repeated `--n-runs` times, then aggregates mean ± std.

```
for dataset in [ph2, ham10000, derm7pt, imagenet]:
    for run in range(n_runs):
        build_concept_graphs.py  (gen_concepts + build_graphs)
        train_model.py
        read metrics.json
    aggregate mean/std over runs
save results.csv + results.json
clean scratch directories
```

**Output columns:** `{split}_{metric}_{mean|std}` for `split ∈ {train, val, test}`, `metric ∈ {acc, f1, auc}`.

### 11.2 Fidelity curves (`eval_fidelity.py`)

Measures **insertion** and **deletion** AUC:
- **Insertion**: progressively add the most important patches; measure how quickly the model recovers its prediction.
- **Deletion**: progressively remove the most important patches; measure how quickly the prediction degrades.

```bash
python eval_fidelity.py --output-dir concept_graph_data --device cuda
```

### 11.3 Concept quality (`eval_concept_quality.py`)

Measures the **disentanglement** quality of discovered concepts using a subset of ImageNet images.

```bash
python eval_concept_quality.py \
  --img_folder_dir datasets/imagenet/train \
  --num_concepts 25 \
  --patch_size 80 \
  --stride_r 0.8
```

---

## 12. Data Layout and Transforms

### Supported datasets

| Key | Name | Classes | Images root | CSV prefix |
|---|---|---|---|---|
| `ph2` | PH2 | 2 | `PH2Dataset/trainx` | `PH2_` |
| `ham10000` | HAM10000 | 7 | `ham10000/` | (no prefix) |
| `derm7pt` | Derm7pt (7-point checklist) | multi | `derm7pt/images` | `derm7pt_` |
| `imagenet` | ImageNet subset | 2 | `imagenet/` | – |

CSV files live in `datasets/<dataset>/` inside the repository (paths are anchored to `config.py`'s directory so SLURM jobs work from any working directory).

### Transforms

**Medical datasets (PH2, HAM10000, Derm7pt)**

| Split | Transform chain |
|---|---|
| `nmf` / `train` | Resize(270, BILINEAR) → CenterCrop(224) → RandomRotate90 → RandomHorizontalFlip(0.5) → ToTensor → Normalize(ImageNet stats) |
| `val` / `test` | Resize(270, BILINEAR) → CenterCrop(224) → ToTensor → Normalize(ImageNet stats) |

**ImageNet subset**

| Split | Transform chain |
|---|---|
| `nmf` | Resize(256) → CenterCrop(224) → ToTensor → Normalize |
| `train` | RandomResizedCrop(224, scale 0.6-1.0) → RandomHorizontalFlip → ColorJitter → ToTensor → Normalize |
| `val` / `test` | Resize(256) → CenterCrop(224) → ToTensor → Normalize |

**Why eval transform for graph construction?**  
Graphs are built with the *eval* transform (no augmentation) so that the concept activations are deterministic and reproducible across runs.

**Class balancing**  
For imbalanced medical datasets, `_all_balanced.csv` / `train_balanced.csv` include augmented copies of minority-class samples. These copies use the training augmentation pipeline (rotation + flipping) so they are not exact duplicates.

---

## 13. Design Decisions and Rationale

### 13.1 Why NMF / CRAFT for concept discovery?

- NMF produces **non-negative, parts-based** decompositions that map naturally to visual concepts (e.g. colour regions, textures).
- CRAFT wraps NMF with a CNN backbone and provides ready-made patch-level coefficients `U` that directly feed the graph builder.
- Avoids manual concept annotation or text supervision, making the framework applicable to any image classification domain.

### 13.2 Why a fully connected concept graph?

- With only `k = 6–16` concept nodes, a complete graph has at most 256 edges — trivially small.
- A complete graph lets the GAT learn *all pairwise concept interactions* without any structural prior, so the attention mechanism alone determines which relationships matter.
- Self-loops are included to allow each node to also attend to itself, which is standard in GAT formulations.

### 13.3 Why a single GATConv layer?

- Deeper GNNs can over-smooth node features (all nodes converge to the same representation). With only `k` nodes, one attention layer is sufficient to capture global concept interactions.
- Keeps the model interpretable: the attention weights from a single layer directly reflect concept co-importance.

### 13.4 Why `mean_nodes` for graph readout?

- `mean_nodes` is permutation-invariant and produces a fixed-size graph embedding regardless of `k`.
- More robust than `sum` when `k` varies between runs (auto-k selection may pick different values).

### 13.5 Why GraphNorm instead of BatchNorm?

- `BatchNorm` normalises over the batch dimension, which can be unstable when graph sizes vary.
- `GraphNorm` normalises within each graph, decoupling the normalisation from batch composition.

### 13.6 Why node features = weighted-mean activations?

The node feature for concept `c` is:
```
node_feat[c] = GELU( mean_p( g(patch_p) × U[p, c] ) )
```
This is a **soft aggregation**: each patch contributes to concept `c` in proportion to its NMF coefficient for that concept. It preserves the full 2048-dimensional activation space while weighting it by concept membership, giving the GAT a rich, graded signal rather than a hard assignment.

### 13.7 Why gradient-based node importance for explanation?

- Attention weights in the GAT reflect *structural relevance* (how much one concept attends to another), not *predictive importance*.
- Gradient of the predicted class probability with respect to node features directly measures how much perturbing a concept's activation changes the prediction — a natural importance score.
- This is consistent with attribution methods like Integrated Gradients and GradCAM in the CNN literature.

### 13.8 Why save CRAFT without CNN parts (lightweight)?

- The ResNet-50 backbone is ~100 MB. Saving it per dataset would be wasteful.
- At load time, `build_model_parts` re-creates the frozen ResNet-50 and `load_craft_and_attach` re-attaches `g` and `h` to the deserialized NMF object.

### 13.9 Why AdamW + CosineAnnealingLR?

- AdamW decouples weight decay from the gradient update, which is better calibrated than L2 regularisation via the loss.
- Cosine annealing provides a smooth, gradual learning rate reduction to `lr/50`, preventing sharp drops that can destabilise training on small graph datasets.

---

## 14. End-to-End Data Flow

### Full training pipeline

```
[Dataset CSVs]
      │  config.py  →  load_split()
      ▼
[Raw image tensors]  (NMF split for Pipeline A; eval transform for B)
      │
      ├─── Pipeline A ───────────────────────────────────────────────►
      │    concepts.py                                                │
      │    ResNet-50 (g) → patches → NMF.fit(k)                      │
      │    → craft_<ds>.dill  +  nmf_best_k.json                     │
      │                                                              │
      ├─── Pipeline B ───────────────────────────────────────────────►
      │    graph.py                                                   │
      │    ResNet-50 (g) → patch activations → NMF.transform()       │
      │    → DGL graphs  →  concept_graphs_{split}.dgl               │
      │                                                              │
      └─── Pipeline C ───────────────────────────────────────────────►
           model.py  +  train_model.py
           GraphDataLoader → GATConv → GraphNorm → ELU
           → mean_nodes → Linear → CrossEntropyLoss
           → best checkpoint  +  metrics.json
```

### Inference / explanation pipeline

```
[Single image]
      │  eval_transform
      ▼
[Patch extraction]  (same patch_size / stride_r as training)
      │
      ▼
[CRAFT: g(patches) → activations → NMF.transform() → patches_U]
      │
      ▼
[Build DGL graph: k nodes, weighted-mean node features]
      │
      ▼
[GAT forward pass: logits, attn_weights, node_embeddings]
      │
      ├─► Predicted class + confidence
      │
      └─► autograd.grad → node_importance
                │
                ├─► Top concept IDs  +  importance bar chart
                │
                └─► patch_importance = patches_U @ node_importance
                          │
                          └─► Coloured patch boxes on image
```

---

## 15. Configuration Reference

All dataset-level and model-level configuration lives in `config.py`.

### `DatasetSpec` dataclass

| Field | Type | Description |
|---|---|---|
| `name` | `str` | Human-readable dataset name |
| `build_transforms` | `Callable` | Returns `{"nmf", "train", "eval"}` transform dict |
| `resolve_paths` | `Callable` | Returns `{"images_root", "nmf_csv", "train_csv", ...}` |
| `load_split` | `Callable` | Loads and returns `(X, Y, masks)` tensors for a split |

### `MODEL_CFG` (per-dataset hyperparameters)

```python
MODEL_CFG = {
    "ph2":      {"num_heads": 4, "hidden_dim": 128, "batch_size":  64},
    "derm7pt":  {"num_heads": 4, "hidden_dim": 128, "batch_size":  32},
    "ham10000": {"num_heads": 6, "hidden_dim": 128, "batch_size": 128},
    "imagenet": {"num_heads": 6, "hidden_dim": 128, "batch_size": 128},
}
```

### Default paths

| Variable | Default value | Description |
|---|---|---|
| `default_output_dir` | `{cwd}/concept_graph_data` | CRAFT + graph + model artifacts |
| `default_eval_dir` | `{cwd}/results` | Benchmark result CSVs |
| `default_datasets_dir` | `{repo_root}/datasets` | CSV split files (anchored to repo, not cwd) |
