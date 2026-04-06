import torch
import dgl
from dgl.data import DGLDataset
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
import os
from utils import _safe_argmax
from typing import Optional, List
from concepts import build_model_parts, load_craft_and_attach


class ConceptGraphDataset(DGLDataset):
    """
    Converts images into concept graphs using cosine similarity.

    Each image becomes one graph where:
      - Nodes = concepts (K nodes, one per NMF concept)
      - Node features = cosine similarity profile across all patches (P-dim vector)
      - Edges = fully connected (every concept connected to every other)

    Unlike v1/v2/v3 which compute hand-crafted statistics per concept,
    this version passes the raw cosine similarity values so that the
    graph network (GAT) can learn its own patterns from the data.

    Key parameter:
      sim_threshold: cosine similarity values below this are zeroed out.
                     Set to 0.0 to keep all values (no thresholding).
    """

    def __init__(self, images, y, masks, patch_size, craft_xai, ignore_list,
                 device, stride_r=0.8, coverage_threshold=0.5, seed=42,
                 requires_grad=False, sim_threshold=0.0):
        self.images = images
        self.y = y
        self.masks = masks
        self.patch_size = patch_size
        self.craft_xai = craft_xai
        self.ignore_list = ignore_list
        self.device = device
        self.stride_r = stride_r
        self.seed = seed
        self.coverage_threshold = coverage_threshold
        self.requires_grad = requires_grad
        # cosine similarity threshold: values below this are zeroed out
        self.sim_threshold = sim_threshold

        super().__init__(name='concept_graph_dataset')

    def _batch_inference(self, model, x, resize=None, device='cuda'):
        """Run a forward pass through the CNN backbone without computing gradients."""
        with torch.no_grad():
            x = x.clone().detach()
            x = x.to(device)
            if resize:
                x = torch.nn.functional.interpolate(
                    x, size=resize, mode='bicubic', align_corners=False)
            activation = model(x).cpu()
        return activation

    def process(self):
        self.graphs = []
        self.labels = []

        # stride in pixels between adjacent patches
        strides = int(self.patch_size * self.stride_r)

        if self.masks == None:
            self.masks = [None] * self.images.shape[0]

        for img, y, mask in zip(self.images, self.y, self.masks):
            img = img.unsqueeze(0)  # add batch dimension: [1, C, H, W]
            image_size = img.shape[2]

            # --- Step 1: Extract patches from the image ---
            if mask == None:
                patches = torch.nn.functional.unfold(
                    img, kernel_size=self.patch_size, stride=strides)
                patches = patches.transpose(1, 2).contiguous().view(
                    -1, img.shape[1], self.patch_size, self.patch_size)
            else:
                mask = mask.unsqueeze(0)
                img_patches = torch.nn.functional.unfold(
                    img, kernel_size=self.patch_size, stride=strides)
                img_patches = img_patches.transpose(1, 2).contiguous().view(
                    -1, 3, self.patch_size, self.patch_size)

                mask_patches = torch.nn.functional.unfold(
                    mask, kernel_size=self.patch_size, stride=strides)
                mask_patches = mask_patches.transpose(1, 2).contiguous().view(
                    -1, 1, self.patch_size, self.patch_size)

                # only keep patches where enough pixels fall inside the mask
                coverage = mask_patches.float().mean(dim=(1, 2, 3))
                keep_indices = coverage >= self.coverage_threshold
                patches = img_patches[keep_indices]

            if patches.shape[0] != 0:

                # --- Step 2: Get CNN features for each patch ---
                # patch_activations shape: [P, 2048] (P = number of patches)
                self.craft_xai.device = self.device
                patch_activations = self._batch_inference(
                    self.craft_xai.input_to_latent, patches,
                    resize=image_size, device=self.device)

                # if CNN output is spatial (4D), average-pool to get [P, 2048]
                if len(patch_activations.shape) == 4:
                    patch_activations = torch.mean(patch_activations, dim=(2, 3))

                # --- Step 3: NMF transform (still needed for patches_U / patches_C) ---
                # patches_U shape: [P, K] -- NMF activation of each concept at each patch
                W_dtype = self.craft_xai.reducer.components_.dtype
                patches_U = self.craft_xai.reducer.transform(
                    np.array(patch_activations, dtype=W_dtype))
                patches_C = _safe_argmax(patches_U, self.ignore_list)
                self.patches_U = patches_U
                self.patches_C = patches_C

                patches_U = torch.tensor(
                    patches_U, dtype=torch.float32, device=self.device)

                if self.requires_grad:
                    patch_activations = patch_activations.clone().detach().to(
                        torch.float32).to(self.device).requires_grad_()
                    self.patch_activations = patch_activations
                else:
                    patch_activations = patch_activations.clone().detach().to(
                        torch.float32).to(self.device)

                # which concept indices are we actually using?
                valid_nodes = [i for i in range(patches_U.shape[1])
                               if i not in self.ignore_list]
                num_nodes = len(valid_nodes)

                if num_nodes > 1:

                    # --- Step 4: Build fully-connected graph ---
                    # every concept node is connected to every other concept node
                    src, dst = [], []
                    for i in range(num_nodes):
                        for j in range(num_nodes):
                            src.append(i)
                            dst.append(j)
                    graph = dgl.graph(
                        (torch.tensor(src), torch.tensor(dst))).to(self.device)

                    # --- Step 5: Compute cosine similarity node features ---
                    # W = NMF concept directions, shape [K, 2048]
                    W = torch.tensor(
                        self.craft_xai.reducer.components_,
                        dtype=torch.float32, device=self.device)

                    # normalize patch_activations to unit vectors: [P, 2048]
                    A_norm = patch_activations / (
                        patch_activations.norm(dim=1, keepdim=True) + 1e-8)

                    # normalize concept directions to unit vectors: [K, 2048]
                    W_norm = W / (W.norm(dim=1, keepdim=True) + 1e-8)

                    # cosine similarity between every patch and every concept
                    # sim_matrix shape: [P, K]
                    # each entry tells us: how similar is patch p to concept k?
                    sim_matrix = A_norm @ W_norm.T

                    # apply threshold: zero out weak similarities
                    if self.sim_threshold > 0:
                        sim_matrix = sim_matrix * (
                            sim_matrix > self.sim_threshold).float()

                    # transpose to [K, P]: each concept node gets the full
                    # similarity profile across all patches as its feature vector
                    # this is done as a single matrix operation -- no per-concept loop
                    node_features = sim_matrix.T  # [K, P]

                    # keep only the valid (non-ignored) concept rows
                    valid_indices = torch.tensor(
                        valid_nodes, dtype=torch.long, device=self.device)
                    node_features = node_features[valid_indices]  # [num_nodes, P]

                    if self.requires_grad:
                        graph.ndata['feat'] = node_features.requires_grad_()
                    else:
                        graph.ndata['feat'] = node_features

                    self.graphs.append(graph)
                    self.labels.append(y)

    def node_z_score_normalize(self, global_mean=None, global_std=None):
        """Apply Z-score normalization to node features across all graphs."""

        assert hasattr(self, 'graphs') and len(self.graphs) > 0, \
            "No graphs found for normalization."

        if global_mean is None or global_std is None:
            all_feats = torch.cat(
                [g.ndata['feat'] for g in self.graphs], dim=0)
            self.global_mean = all_feats.mean(dim=0)
            self.global_std = all_feats.std(dim=0) + 1e-8
        else:
            self.global_mean = global_mean
            self.global_std = global_std

        for graph in self.graphs:
            feats = graph.ndata['feat']
            feats = (feats - self.global_mean) / self.global_std
            graph.ndata['feat'] = feats

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]

    def __len__(self):
        return len(self.graphs)


def build_and_save_graphs_per_split(images: torch.Tensor,
                                    labels: torch.Tensor,
                                    device: str,
                                    backbone_name: str,
                                    craft_path: str,
                                    out_path: str,
                                    patch_size: int,
                                    stride_r: float,
                                    ignore_list: Optional[List[int]] = None,
                                    coverage_threshold: float = 0.0,
                                    sim_threshold: float = 0.0):
    """
    Builds concept graphs for one data split and saves them to disk.
    Uses cosine similarity between patch CNN features and NMF concept
    directions as node features (no hand-crafted statistics).
    """
    ignore_list = ignore_list or []
    # rebuild the CNN backbone and attach it to the saved craft object
    g, h = build_model_parts(backbone_name, device=device, pretrained=True)
    craft = load_craft_and_attach(craft_path, g, h)

    ds = ConceptGraphDataset(
        images=images.to(device),
        y=labels.to(device),
        masks=None,
        patch_size=patch_size,
        craft_xai=craft,
        ignore_list=ignore_list,
        device=device,
        stride_r=stride_r,
        coverage_threshold=coverage_threshold,
        seed=42,
        requires_grad=False,
        sim_threshold=sim_threshold,
    )
    ds.process()

    graphs = ds.graphs
    labels_out = (torch.stack(ds.labels)
                  if isinstance(ds.labels[0], torch.Tensor)
                  else torch.tensor(ds.labels))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    dgl.save_graphs(out_path, graphs, {"labels": labels_out})
    return out_path, len(graphs)


# ---- Loading pre-built graphs from disk ----

class LoadConceptGraphDataset(DGLDataset):
    def __init__(self, file_path=None, efeats=True, device='cuda'):
        self.file_path = file_path
        self.device = device
        self.efeats = efeats

        super().__init__(name='concept_graph_dataset')

    def load(self):
        self.graphs, metadata = dgl.load_graphs(self.file_path)
        self.labels = metadata['labels']
        print(f"Loaded {len(self.graphs)} graphs from {self.file_path}, "
              f"moved to {self.device}.")

    def process(self):
        if self.file_path:
            self.load()
        else:
            self.graphs = []
            self.labels = []

    def node_z_score_normalize(self, global_mean=None, global_std=None):
        """Apply Z-score normalization to node features across all graphs."""

        assert hasattr(self, 'graphs') and len(self.graphs) > 0, \
            "No graphs found for normalization."

        if global_mean is None or global_std is None:
            all_feats = torch.cat(
                [g.ndata['feat'] for g in self.graphs], dim=0)
            self.global_mean = all_feats.mean(dim=0)
            self.global_std = all_feats.std(dim=0) + 1e-8
        else:
            self.global_mean = global_mean
            self.global_std = global_std

        for graph in self.graphs:
            feats = graph.ndata['feat']
            feats = (feats - self.global_mean) / self.global_std
            graph.ndata['feat'] = feats

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]

    def __len__(self):
        return len(self.graphs)


def load_split(output_root: str, dataset: str, split: str,
               device: str = "cuda"):
    path = os.path.join(
        output_root, dataset, "graphs", dataset,
        f"concept_graphs_{split}.dgl")
    ds = LoadConceptGraphDataset(file_path=path, device=device)
    ds.load()
    return ds


def infer_dims(ds: LoadConceptGraphDataset):
    in_dim = ds.graphs[0].ndata["feat"].shape[1]
    num_classes = int(ds.labels.max().item()) + 1
    return in_dim, num_classes
