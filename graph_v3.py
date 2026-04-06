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
    def __init__(self, images, y, masks, patch_size, craft_xai, ignore_list, device, stride_r=0.8, coverage_threshold=0.5, seed=42, requires_grad=False):
        self.images = images
        self.y = y
        self.masks = masks
        self.patch_size = patch_size
        self.craft_xai = craft_xai
        self.ignore_list = ignore_list
        self.device = device
        self.stride_r = stride_r
        self.seed=seed
        self.coverage_threshold = coverage_threshold
        self.requires_grad=requires_grad

        super().__init__(name='concept_graph_dataset')

    def _batch_inference(self, model, x, resize=None, device='cuda'):
        with torch.no_grad():
            x = x.clone().detach()
            x = x.to(device)
            if resize:
                x = torch.nn.functional.interpolate(x, size=resize, mode='bicubic', align_corners=False)
            activation = model(x).cpu()
        return activation


    def process(self):
        self.graphs = []
        self.labels = []

        strides = int(self.patch_size * self.stride_r)

        if self.masks == None:
            self.masks = [None]*self.images.shape[0]

        for img, y, mask in zip(self.images, self.y, self.masks):
            img = img.unsqueeze(0)
            image_size = img.shape[2]

            if mask == None:
                patches = torch.nn.functional.unfold(img, kernel_size=self.patch_size, stride=strides)
                patches = patches.transpose(1, 2).contiguous().view(-1, img.shape[1], self.patch_size, self.patch_size)
            else:
                mask = mask.unsqueeze(0)
                img_patches = torch.nn.functional.unfold(img, kernel_size=self.patch_size, stride=strides)
                img_patches = img_patches.transpose(1, 2).contiguous().view(-1, 3, self.patch_size, self.patch_size)

                mask_patches = torch.nn.functional.unfold(mask, kernel_size=self.patch_size, stride=strides)
                mask_patches = mask_patches.transpose(1, 2).contiguous().view(-1, 1, self.patch_size, self.patch_size)
                coverage = mask_patches.float().mean(dim=(1,2,3))

                keep_indices = coverage >= self.coverage_threshold

                patches = img_patches[keep_indices]

            if patches.shape[0]!=0:

                self.craft_xai.device = self.device
                patch_activations = self._batch_inference(self.craft_xai.input_to_latent, patches, resize=image_size, device=self.device)

                if len(patch_activations.shape) == 4:
                    patch_activations = torch.mean(patch_activations, dim=(2, 3))

                W_dtype = self.craft_xai.reducer.components_.dtype
                patches_U = self.craft_xai.reducer.transform(np.array(patch_activations, dtype=W_dtype))
                patches_C = _safe_argmax(patches_U, self.ignore_list)
                self.patches_U = patches_U
                self.patches_C = patches_C

                patches_U = torch.tensor(patches_U, dtype=torch.float32, device=self.device)
                if self.requires_grad:
                    patch_activations = patch_activations.clone().detach().to(torch.float32).to(self.device).requires_grad_()
                    self.patch_activations = patch_activations
                else:
                    patch_activations = patch_activations.clone().detach().to(torch.float32).to(self.device)

                valid_nodes = [i for i in range(patches_U.shape[1]) if i not in self.ignore_list]
                num_nodes = len(valid_nodes)

                if num_nodes>1:
                    src, dst = [], []
                    for i in range(num_nodes):
                        for j in range(num_nodes):
                            src.append(i)
                            dst.append(j)

                    graph = dgl.graph((torch.tensor(src), torch.tensor(dst))).to(self.device)

                    # Approach 3: concept-specific co-occurrence profile + summary statistics
                    node_features = []

                    for c in valid_nodes:
                        u_c = patches_U[:, c]
                        weights = u_c / (u_c.sum() + 1e-8)

                        co_occurrence = (weights.unsqueeze(-1) * patches_U).sum(dim=0)  # [K]

                        scalars = torch.tensor([
                            u_c.mean().item(),
                            u_c.max().item(),
                            (u_c > 0.1).sum().item() / len(u_c),
                        ], dtype=torch.float32, device=self.device)

                        node_feature = torch.cat([co_occurrence, scalars])  # [K + 3]
                        node_features.append(node_feature)

                    if self.requires_grad:
                        graph.ndata['feat'] = torch.stack(node_features).requires_grad_()
                    else:
                        graph.ndata['feat'] = torch.stack(node_features)

                    self.graphs.append(graph)
                    self.labels.append(y)


    def node_z_score_normalize(self, global_mean=None, global_std=None):
        """Apply Z-score normalization to node features across all graphs."""

        assert hasattr(self, 'graphs') and len(self.graphs) > 0, "No graphs found for normalization."

        if global_mean is None or global_std is None:
            all_feats = torch.cat([g.ndata['feat'] for g in self.graphs], dim=0)
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
                                    **kwargs):
    """
    builds graphs for a single split and saves to out_path using the NMF object.
    """
    ignore_list = ignore_list or []
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
    )
    ds.process()

    graphs = ds.graphs
    labels_out = torch.stack(ds.labels) if isinstance(ds.labels[0], torch.Tensor) else torch.tensor(ds.labels)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    dgl.save_graphs(out_path, graphs, {"labels": labels_out})
    return out_path, len(graphs)


class LoadConceptGraphDataset(DGLDataset):
    def __init__(self, file_path=None, efeats=True, device='cuda'):
        self.file_path = file_path
        self.device = device
        self.efeats = efeats

        super().__init__(name='concept_graph_dataset')

    def load(self):
        """
        Load the graphs and labels from a file and move them to the specified device.
        """
        self.graphs, metadata = dgl.load_graphs(self.file_path)
        self.labels = metadata['labels']

        print(f"Loaded {len(self.graphs)} graphs from {self.file_path}, moved to {self.device}.")

    def process(self):
        if self.file_path:
            self.load()
        else:
            self.graphs = []
            self.labels = []

    def node_z_score_normalize(self, global_mean=None, global_std=None):
        """Apply Z-score normalization to node features across all graphs."""

        assert hasattr(self, 'graphs') and len(self.graphs) > 0, "No graphs found for normalization."

        if global_mean is None or global_std is None:
            all_feats = torch.cat([g.ndata['feat'] for g in self.graphs], dim=0)
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


def load_split(output_root: str, dataset: str, split: str, device: str = "cuda"):
    path = os.path.join(output_root, dataset, "graphs", dataset, f"concept_graphs_{split}.dgl")
    ds = LoadConceptGraphDataset(file_path=path, device=device)
    ds.load()
    return ds


def infer_dims(ds: LoadConceptGraphDataset):
    in_dim = ds.graphs[0].ndata["feat"].shape[1]
    num_classes = int(ds.labels.max().item()) + 1
    return in_dim, num_classes
