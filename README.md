# Concept-Based Explainability with Graph Attention Networks
This repository contains the code implementation for my master’s thesis project conducted at DFKI.

While Concept Bottleneck Models (CBMs) show promise for interpretable AI, they face major challenges: defining clinically meaningful concepts, expensive annotations, reliance on heatmaps, and spurious alignment between visual features and textual labels. To overcome these issues, we focus on visually grounded concepts and avoid text supervision. Unlike prior visually grounded approaches that provide only global, class-specific, post-hoc explanations, our framework (a) discovers concepts via Non-negative Matrix Factorisation (NMF) and (b) captures their interactions through a shallow Graph Attention Network (GAT), balancing model expressiveness with interpretability. The proposed architecture is as follows:

<img src="https://github.com/anaramirli/gatCBM_msc_thesis/blob/main/assets/thesis.png" alt="Thesis Illustration" width="600">

## Overview  
This section provides a general overview of the repository contents.

### Main Components  
- **Concept Discovery**: Extracts high-level concepts using *Non-negative Matrix Factorization (NMF)*.  
- **Concept Graph Construction**: Builds graphs that capture relationships between discovered concepts.  
- **Model Training**: Trains a **Graph Attention Network (GAT)** classifier on concept graphs.  
- **Evaluation**: Measures **accuracy, fidelity, and concept quality** across multiple datasets.  

### Explanations  
- The framework also supports **explanation generation** for individual predictions.
- Once the model is trained, you can run the following command to make predictions and generate visual explanations.  
- **Note**  
> The `--file-root` argument is required to load the pretrained GAT model and NMF composer.  
> This repository already includes the trained models and NMF composers, with their default location specified in `config.py`.  
> You can override this path manually by providing a custom value for `--file-root`.


```bash
python explain_image.py \
  --dataset <datasetname> \ 
  --image_path /path/to/image.jpg
```

- `<datasetname>` can be one of ph2, derm7pt, imagenet, and ham10000.

- Example visual explanations can be found at `/assets` folder

#### Example visual explanations for a medical image:
![Explanation 2](https://github.com/anaramirli/gatCBM_msc_thesis/blob/main/assets/output_explanation_2.png)

#### Example visual explanations for a general image:
![Explanation 1](https://github.com/anaramirli/gatCBM_msc_thesis/blob/main/assets/output_explanation.png)

---

## Grad-CAM (`gradcam_explain.py`)

Spatial **Grad-CAM** heatmaps complement concept-based explanations: they show *where* a CNN focuses; CBM-GAT shows *which concepts* matter. The script supports **six modes** via `--mode`.

### Modes

| Mode | Description | Needs `--file-root`? |
|------|-------------|----------------------|
| `standalone_spatial` | ResNet-50 **ImageNet** pretrained only — 3 panels (original, heatmap, overlay). | **No** |
| `standalone_cnn` | ResNet-50 **fine-tuned** on your dataset (CNN baseline) — 3 panels. | **Yes** — loads checkpoint under `file-root` (see below). |
| `standalone_medical` | CBM-GAT **patch-importance** heatmap (from GAT gradients), no ImageNet/CNN spatial Grad-CAM. | **Yes** |
| `compare_spatial_concepts` | ImageNet ResNet-50 Grad-CAM **vs** CBM-GAT concepts (4 columns). | **Yes** |
| `compare_cnn_concepts` | Fine-tuned CNN Grad-CAM **vs** CBM-GAT concepts (4 columns). | **Yes** |
| `compare_medical_concepts` | CBM-GAT medical heatmap **vs** CBM-GAT concepts (4 columns). | **Yes** |

If `--mode` is omitted, behavior matches older scripts: without `--compare` → `standalone_spatial`; with `--compare` → `compare_medical_concepts`.

### CLI example (local)

```bash
# ImageNet backbone only (no trained CBM-GAT artifacts)
python gradcam_explain.py \
  --mode standalone_spatial \
  --dataset ham10000 \
  --image_path /path/to/image.jpg

# Modes that need CRAFT + GAT + (for CNN modes) CNN baseline checkpoint
python gradcam_explain.py \
  --mode compare_cnn_concepts \
  --dataset ham10000 \
  --image_path /path/to/image.jpg \
  --file-root /path/to/concept_graph_data
```

Common flags: `--device`, `--alpha` (overlay blend), `--backbone resnet50`, `--target-class` (optional class index for Grad-CAM). For CBM-GAT panels: `--patch-size`, `--stride-r`, `--top-k-max`, `--min-concept-weight`.

**Compare modes only** (`compare_*`): the concept patch column draws **one patch-sized box per top concept** (argmax of concept activation × patch importance), matching the bar colors.

- `--save-concept-heatmaps` — also save a second figure: one **row** of per-concept patch heatmaps as **jet overlays on the same CBM image** (blend strength uses `--alpha`, same as other overlays).

**Output PNGs** (written to the current working directory): e.g. `output_gradcam_spatial.png`, `output_gradcam_cnn.png`, `output_gradcam_medical.png`, `output_gradcam_vs_concepts_spatial.png`, `output_gradcam_vs_concepts_cnn.png`, `output_gradcam_vs_concepts_medical.png`. With `--save-concept-heatmaps`, compare modes also write `output_gradcam_vs_concepts_{spatial,cnn,medical}_concept_heatmaps.png`.

### CNN baseline checkpoint (for `standalone_cnn` / `compare_cnn_concepts`)

Train a ResNet-50 directly on images (no graphs) with `train_cnn.py` (see below). The checkpoint is saved as a dict `{"state_dict", "num_classes"}` at:

`{file_root}/{dataset}/models_cnn/{dataset}/{dataset}_resnet50_cnn.pt`

Grad-CAM loads it automatically — **no manual class count or path** beyond `--file-root`.

### Cluster (SLURM)

- `scripts/sbatch_gradcam.sh` — set `MODE` and `RUN_ROOT` (path to your `concept_graph_data`). For compare modes only, optional `GRADCAM_EXTRA_ARGS` (e.g. `--save-concept-heatmaps`) is appended to the Python command; standalone modes ignore it. Then `sbatch`.
- `scripts/sbatch_train_cnn.sh` — trains the CNN baseline; writes metrics and the `.pt` file above under your `--output-root`.

---

## CNN-only baseline (`train_cnn.py`)

A **ResNet-50** classifier trained **directly** on image tensors (same train/val/test CSV splits as in `config.py`) — **no** concept graphs, **no** GAT. Used as a performance and Grad-CAM baseline.

```bash
python train_cnn.py \
  --dataset ham10000 \
  --output-root concept_graph_data \
  --device cuda \
  --epochs 300 --patience 50 \
  --lr 1e-3 --weight-decay 2e-4 \
  --save-model
```

- **Train / val / test**: training uses train + val (Lightning); **test** is only evaluated after training for reported metrics (same pattern as `train_model.py`).
- **Outputs**: `metrics_cnn.json` and `{dataset}_resnet50_cnn.pt` under `{output_root}/{dataset}/models_cnn/{dataset}/`.

Dataset CSV paths are resolved relative to the **repository** `datasets/` folder (see `config.py`), so jobs work even when the process `cwd` is a SLURM work directory.

---

## Setting up Environment

```bash
# create env (Python 3.10.x)
conda create -n <env> python=3.10.16 -y
conda activate <env>

# install deps
pip install -r requirements.txt
```

> Tip: this code uses CUDA when available. Make sure your PyTorch/DGL wheels match your CUDA version.

---

## Folder structure

```
+-- datasets
+-- concept_graph_data
+--- results
``` 

- `datasets` — stores the available datasets, currently includes CSV files for train, test, validation, and nmf splits.
- `concept_graph_data` — currently stores the trained GAT model and the fitted CRAFT NMF components (enough for directly running explain_image.py). Graph datasets need to be created from the dataset splits.
- `results` — stores `train_/val_/test_{acc,f1,auc}_{mean,std}` results for **ph2**, **ham10000**, **7pt**, **imagenet** from multiple runs.

- For more details about the contents of these folders, please refer to the **Outputs** section below in  
  1. *Build Concept Basis & Graphs* and  
  2. *Train the GAT*  

#### Reproducibility of the results
Download the required datasets:
- PH2: https://www.kaggle.com/datasets/spacesurfer/ph2-dataset
- Derm7pt: https://derm.cs.sfu.ca/Welcome.html
- HAM10000: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T
- ImageNet subset: `wget https://image-net.org/data/winter21_whole/<wnid>.tar` (use `n04065272` for recreational vehicle, `n02701002` for ambulance)

## File definitions

- `build_concept_graphs.py` — discover concepts & build graphs, can be used for each separately too.
- `train_model.py` — train GAT and write metrics, also used for eval.
- `train_cnn.py` — train ResNet-50 CNN baseline on images only (no graphs); saves CNN checkpoint for Grad-CAM modes.
- `gradcam_explain.py` — Grad-CAM heatmaps and side-by-side comparisons with CBM-GAT concepts (`--mode`; see **Grad-CAM** section above).
- `explain_image.py` — concept-based explanation for a single image.
- `eval_benchmark.py` — run build→train N times across datasets, aggregate mean/std.
- `eval_fidelity.py` — insertion/deletion AUC curves.
- `eval_concept_quality.py` — concept quality metrics.
- `config.py` — dataset registry & transforms (CSV paths are anchored to the repo `datasets/` directory).
- `graph.py` — concept graph dataset loaders/builders.
- `concepts.py` — NMF utilities using CRAFT.
- `model.py` — EGAT classifier + Lightning training wrapper.
- `utils.py` — helpers (e.g., save concept crops).
- `scripts/sbatch_gradcam.sh`, `scripts/sbatch_train_cnn.sh`, `scripts/sbatch_train_gcbm.sh` — example SLURM jobs (paths may need adjustment for your cluster).

## Data layout & registry

Datasets and transforms are defined in `config.py`. Adjust paths if your data lives elsewhere.

- **PH2**
  - images_root: `/datasets/ph2_dataset/trainx`
  - CSVs: `PH2_all_balanced.csv`, `PH2_train_balanced.csv`, `PH2_validation.csv`, `PH2_test.csv`
- **HAM10000**
  - images_root: `/datasets/ham10000/`
  - CSVs: `all_balanced.csv`, `train_balanced.csv`, `validation.csv`, `test.csv`
- **Derm7pt (7pt)**
  - images_root: `/datasets/derm7pt/images`
  - CSVs: `derm7pt_all_balanced.csv`, `derm7pt_train_balanced.csv`, `derm7pt_validation.csv`, `derm7pt_test.csv`
- **ImageNet subset**
  - images_root: `/datasets/imagenet`
  - CSVs: `nmf.csv`, `train.csv`, `val.csv`, `test.csv`

Please ensure that all downloaded datasets are placed in the designated directory before running the code, and the images_root matches the dataset root.

### Transforms
- Medical sets (`ph2`, `ham10000`, `derm7pt`): same nmf/train transform (resize→center crop→90° rotate→flip) + eval transform (resize→center crop).
- ImageNet: standard `RandomResizedCrop/HorizontalFlip/Jitter` for train; resize+center crop for eval.

(See `config.py` for exact definitions. The config file also contains additional parameters for model training and specifies output directories.)

Since the PH2, Derm7pt, and HAM10000 datasets have imbalanced class distributions, we trained on a balanced version of the data. To achieve this, we included extra copies of samples from the underrepresented classes. However, these copies were augmented with transformations from the training pipeline, so they are not exact duplicates.

`<ds>_all_balanced.csv` or `nmf.csv` files are used for concept generation with NMF.

---

## 1) Build concept basis & graphs

Use `build_concept_graphs.py` to (a) discover concepts with using CRAFT based on NMF on the NMF split, optionally auto-selecting `k`, and (b) build per-split concept graphs (`train/val/test`) using the *same* fitted reducer.

### example (auto-select best concept number `k`)

```bash
python build_concept_graphs.py \
  --dataset <datasetname> \
  --steps gen_concepts build_graphs \
  --auto-n-components --candidates 6 7 8 9 10 12 16 \
  --patch-size 70 --stride-r 0.8 \
  --batch-size 64 \
  --device cuda \
  --output-root /concept_graph_data
```

- Running `gen_concepts` together with `build_graphs` will fit the best concept basis and create concept graphs from it.  
- Running only `gen_concepts` will create the NMF composer.  
- Running only `build_graphs` will generate the graph dataset, assuming an NMF composer already exists.
- `--auto-n-components --candidates 6 7 8 9 10 12 16` selects the best number of concepts `k` based on the disentanglement score. If not enabled, the model uses `--n-components`, which defaults to 10.

### Outputs (inside `--output-root/<dataset>/`)

The structure of the output folder is as follows. The default `--output-root` is set to `/concept_graph_data` (see `config.py`):

- `craft/<dataset>/craft_<dataset>.dill` — **lightweight Craft** (heavy torch parts detached)
- `U_meta/nmf_best_k.json` with `{best_k, patch_size, stride_r}`
- `concept_examples/` with top crops per concept (visual examples of concept generation).
- `graphs/<dataset>/concept_graphs_train.dgl`, `concept_graphs_validation.dgl`, `concept_graphs_test.dgl`.

---

## 2) Train the GAT

Once the concept basis is computed and concept graphs are formed, the model can be trained.  
This is done with `train_model.py`, which can be used for both training and evaluation only.  
By default, the best checkpoint is deleted after evaluation unless `--save-model` is set.

```bash
python train_model.py \
  --dataset <datasetname> \
  --output-root concept_graph_data \
  --device cuda \
  --epochs 300 --patience 50 \
  --lr 1e-3 --weight-decay 2e-4
```

**Outputs** (under `--output-root/<dataset>/models/<dataset>/`):

- emmits `metrics.json` with **accuracy**, **weighted F1**, and **AUC** for **train/val/test**.
- If `--save-model` is provided, then the best mode checkpoint will be saved at `<dataset>_best_model.ckpt`.
- If the `<dataset>_best_model.ckpt` file is provided, the model will directly switch to eval mode. Otherwise, it'll train from scratch.

---

## 3) One-shot benchmark (build → train → aggregate)

Automate multi-run evaluation across all datasets with `eval_benchmark.py`:

- The default `--output-root` here is set to `/results` (see default_eval_dir in `config.py`).  

```bash
python eval_benchmark.py \
  --output-root results \
  --n-runs 5 \
  --device cuda \
  --patch-size 70 --stride-r 0.8 \
  --candidates 6 7 8 9 10 \
  --batch-size-fit 64 \
  --epochs 300 --patience 50 \
  --hidden-dim 128 \
  --lr 1e-3 --weight-decay 2e-4
```

**Results**: `--output-root/results.csv` and `results.json` with `train_/val_/test_{acc,f1,auc}_{mean,std}` for **ph2**, **ham10000**, **7pt**, **imagenet**.

---

## 4) Optional: fidelity & concept quality

### Fidelity insertion/deletion curves

```bash
python eval_fidelity.py --output-dir concept_graph_data --device cuda
```

### Concept quality

```bash
python eval_concept_quality.py \
  --img_folder_dir datasets/imagenet/train \
  --num_concepts 25 \
  --patch_size 80 \
  --stride_r 0.8
```
