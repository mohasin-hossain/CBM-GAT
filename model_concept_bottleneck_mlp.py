"""
model_concept_bottleneck_mlp — MLP on K-dim concept-bottleneck vector z.

Part of **concept_bottleneck_mlp_linear**: each graph is one node with
``feat`` shape ``[1, K]`` (τ-masked max-pooled CRAFT/NMF scores per concept).
``mean_nodes`` yields ``[batch, K]`` → 2-layer MLP → logits.

``num_heads`` is unused (API parity with ``EGATClassifier``).
"""

import torch
import torch.nn as nn
import dgl


class ConceptBottleneckMLP(nn.Module):
    def __init__(
        self,
        in_feats: int,
        out_feats: int,
        num_heads: int = None,
        out_dim: int = 2,
        feat_drop: float = 0.0,
        node_drop: float = 0.0,
    ):
        super().__init__()
        self.feat_drop = feat_drop
        self.node_drop = node_drop
        self.mlp = nn.Sequential(
            nn.Linear(in_feats, out_feats),
            nn.ELU(),
            nn.Linear(out_feats, out_feats),
            nn.ELU(),
        )
        self.classify = nn.Linear(out_feats, out_dim)

    def forward(self, graph, nfeats):
        with graph.local_scope():
            graph.ndata["h"] = nfeats
            hg = dgl.mean_nodes(graph, "h")
        h = self.mlp(hg)
        logits = self.classify(h)
        attn_placeholder = torch.zeros(logits.shape[0], device=logits.device)
        return logits, attn_placeholder, h
