import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GraphNorm
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import dgl
import dgl.function as fn
from dgl.nn import EdgeGATConv, EGATConv, GATConv
from torchmetrics.classification import Accuracy, Recall
from dgl.nn.pytorch.glob import GlobalAttentionPooling


class EGATClassifier(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 out_dim=1,
                 feat_drop=0.0,
                 node_drop=0.0,
                 ):
        super(EGATClassifier, self).__init__()

        self.feat_drop = feat_drop
        self.node_drop = node_drop

        self.layer1 = GATConv(
            in_feats=in_feats,
            out_feats=out_feats,
            num_heads=num_heads,
            bias=True
        )

        self.classify = nn.Linear(out_feats, out_dim)

        self.node_gn1 = GraphNorm(out_feats)


    def forward(self, graph, nfeats):
        node_batch = torch.repeat_interleave(graph.batch_num_nodes())

        h,  attn1 = self.layer1(graph, nfeats, get_attention=True)
        h = h.mean(dim=1)
        h = self.node_gn1(h, node_batch)
        h = F.elu(h)

        with graph.local_scope():
            graph.ndata['h'] = h
            hg = dgl.mean_nodes(graph, 'h')
            logits = self.classify(hg)
            return logits, attn1.mean(dim=1).mean(dim=1), h


class GAT_LightningModule(pl.LightningModule):
    def __init__(self, model, lr=0.01, weight_decay=2e-4, max_epochs=1000,
                 num_classes=5, class_weights=None, l1_loss_alpha=0.0):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.l1_loss_alpha = l1_loss_alpha

        self.save_hyperparameters(ignore=['model'])

        if class_weights is not None and not torch.is_tensor(class_weights):
            class_weights = torch.tensor(class_weights, dtype=torch.float32)

        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc   = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_recall_per_class = Recall(task="multiclass", num_classes=num_classes, average="none")
        self.num_classes = num_classes

    def forward(self, graph, node_f):
        return self.model(graph, node_f)

    def get_loss(self, batch, mode="train"):
        graph = batch[0].to(self.device)
        y = batch[1].long().to(self.device)
        node_f = graph.ndata['feat'].float()
        preds, _, h = self.model(graph, node_f)

        l1_loss = torch.norm(h, p=1)
        loss = self.criterion(preds, y) + self.l1_loss_alpha * l1_loss

        pred_labels = torch.argmax(preds, dim=1)
        if mode == "train":
            acc = self.train_acc(pred_labels, y)
        else:
            acc = self.val_acc(pred_labels, y)
            per_class_recall = self.val_recall_per_class(pred_labels, y)
            bal_acc = per_class_recall.mean()
            self.log("val_bal_acc", bal_acc, prog_bar=True)
        self.log(f"{mode}_loss", loss, prog_bar=True)
        self.log(f"{mode}_acc",  acc,  prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.get_loss(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.get_loss(batch, "val")

    def configure_optimizers(self):
        opt = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.max_epochs, eta_min=self.lr/50)
        return [opt], [sch]
