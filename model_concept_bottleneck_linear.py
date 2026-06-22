"""
model_concept_bottleneck_linear — Linear head on K-dim concept-bottleneck z.

**concept_bottleneck_mlp_linear** — same graph layout as MLP variant
(1 node, ``feat`` ``[1, K]``).
"""

import torch
import torch.nn as nn
import dgl


class ConceptBottleneckLinear(nn.Module):
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
        self.classify = nn.Linear(in_feats, out_dim)

    def forward(self, graph, nfeats):
        with graph.local_scope():
            graph.ndata["h"] = nfeats
            hg = dgl.mean_nodes(graph, "h")
        logits = self.classify(hg)
        attn_placeholder = torch.zeros(logits.shape[0], device=logits.device)
        return logits, attn_placeholder, hg
