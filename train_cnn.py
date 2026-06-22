import os
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torchvision.models import (
    resnet50,      ResNet50_Weights,
    densenet201,   DenseNet201_Weights,
    mobilenet_v2,  MobileNet_V2_Weights,
)

from config import DATASETS, default_output_dir
from utils import _set_seed


class ResNetLightning(pl.LightningModule):
    def __init__(self, model, lr=1e-3, weight_decay=1e-4, max_epochs=300,
                 num_classes=2):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.num_classes = num_classes
        self.save_hyperparameters(ignore=["model"])
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch, stage: str):
        x, y = batch
        logits = self.model(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log(f"{stage}_loss", loss, prog_bar=True)
        self.log(f"{stage}_acc", acc, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr,
                                weight_decay=self.weight_decay)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.max_epochs, eta_min=self.lr / 50
        )
        return [opt], [sch]


@torch.no_grad()
def evaluate(model, loader, device: str, num_classes: int):
    model.eval()
    ys, probs = [], []
    for x, y in loader:
        x = x.to(device)
        ys.append(y.cpu())
        logits = model(x)
        probs.append(F.softmax(logits, dim=1).cpu())
    y_true = torch.cat(ys).numpy()
    y_prob = torch.cat(probs).numpy()
    y_pred = y_prob.argmax(axis=1)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    try:
        auc = (roc_auc_score(y_true, y_prob[:, 1], average="macro")
               if num_classes == 2
               else roc_auc_score(y_true, y_prob, multi_class="ovr",
                                  average="macro"))
    except Exception:
        auc = float("nan")
    return {"acc": acc, "f1": f1, "auc": auc}


def build_backbone(backbone: str, num_classes: int) -> nn.Module:
    """Construct a pretrained backbone with its final classifier replaced."""
    if backbone == "resnet50":
        base = resnet50(weights=ResNet50_Weights.DEFAULT)
        base.fc = nn.Linear(base.fc.in_features, num_classes)
    elif backbone == "densenet201":
        base = densenet201(weights=DenseNet201_Weights.DEFAULT)
        base.classifier = nn.Linear(base.classifier.in_features, num_classes)
    elif backbone == "mobilenet_v2":
        base = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        base.classifier[1] = nn.Linear(base.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unknown backbone: {backbone!r}. "
                         f"Choose from resnet50, densenet201, mobilenet_v2.")
    return base


def main():
    ap = argparse.ArgumentParser("Train CNN baseline on images")
    ap.add_argument("--dataset",  required=True, choices=list(DATASETS.keys()))
    ap.add_argument("--backbone", default="resnet50",
                    choices=["resnet50", "densenet201", "mobilenet_v2"],
                    help="Backbone architecture (default: resnet50)")
    ap.add_argument("--output-root", type=str, default=default_output_dir)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--checkpoint-path", default=None,
                    help="Resume from existing checkpoint (skip training)")
    ap.add_argument("--save-model", action="store_true",
                    help="Keep the best Lightning checkpoint after training")
    args = ap.parse_args()

    _set_seed(args.seed)
    device = (args.device
              if args.device.startswith("cuda") and torch.cuda.is_available()
              else "cpu")

    current_dir = os.getcwd()
    args.output_root = os.path.join(current_dir, args.output_root)

    ds_spec = DATASETS[args.dataset]
    tdict = ds_spec.build_transforms()
    paths = ds_spec.resolve_paths()

    X_train, Y_train, _ = ds_spec.load_split(paths, tdict, "train")
    X_val,   Y_val,   _ = ds_spec.load_split(paths, tdict, "val")
    X_test,  Y_test,  _ = ds_spec.load_split(paths, tdict, "test")

    train_loader = DataLoader(TensorDataset(X_train, Y_train),
                              batch_size=args.batch_size, shuffle=True,
                              drop_last=False)
    val_loader = DataLoader(TensorDataset(X_val, Y_val),
                            batch_size=args.batch_size, shuffle=False,
                            drop_last=False)
    test_loader = DataLoader(TensorDataset(X_test, Y_test),
                             batch_size=args.batch_size, shuffle=False,
                             drop_last=False)

    num_classes = int(Y_train.max().item()) + 1

    base = build_backbone(args.backbone, num_classes)

    model_dir = os.path.join(args.output_root, args.dataset,
                             "models_cnn", args.dataset)
    os.makedirs(model_dir, exist_ok=True)

    lightning_model = ResNetLightning(
        model=base,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=args.epochs,
        num_classes=num_classes,
    )

    if args.checkpoint_path is None:
        ckpt_cb = ModelCheckpoint(
            monitor="val_loss",
            dirpath=model_dir,
            filename=f"{args.dataset}_{args.backbone}_cnn_best",
            mode="min",
            save_weights_only=True,
            save_top_k=1,
        )
        es_cb = EarlyStopping(monitor="val_loss", patience=args.patience,
                              mode="min", verbose=True)
        lr_cb = LearningRateMonitor(logging_interval="epoch")

        trainer = pl.Trainer(
            max_epochs=args.epochs,
            callbacks=[ckpt_cb, lr_cb, es_cb],
            enable_progress_bar=True,
            accelerator="gpu" if device.startswith("cuda") else "cpu",
            devices=1,
            log_every_n_steps=1,
        )

        trainer.fit(lightning_model, train_loader, val_loader)

        best_ckpt_path = ckpt_cb.best_model_path
        best_model = ResNetLightning.load_from_checkpoint(
            best_ckpt_path, model=build_backbone(args.backbone, num_classes))

        if args.save_model:
            print(f"\nBest checkpoint: {best_ckpt_path}")
        else:
            try:
                os.remove(best_ckpt_path)
            except OSError as e:
                print(f"Warning: Could not delete checkpoint {best_ckpt_path}: {e}")
    else:
        best_ckpt_path = args.checkpoint_path
        best_model = ResNetLightning.load_from_checkpoint(
            best_ckpt_path, model=build_backbone(args.backbone, num_classes))

    cnn = best_model.model.to(device)

    train_m = evaluate(cnn, train_loader, device, num_classes)
    val_m = evaluate(cnn, val_loader, device, num_classes)
    test_m = evaluate(cnn, test_loader, device, num_classes)

    print(f"\n=== FINAL CNN BASELINE ({args.backbone}) ===")
    print(f"Train: acc={train_m['acc']:.4f}  f1={train_m['f1']:.4f}  auc={train_m['auc']:.4f}")
    print(f"Val:   acc={val_m['acc']:.4f}  f1={val_m['f1']:.4f}  auc={val_m['auc']:.4f}")
    print(f"Test:  acc={test_m['acc']:.4f}  f1={test_m['f1']:.4f}  auc={test_m['auc']:.4f}")

    metrics_path = os.path.join(model_dir, f"metrics_cnn_{args.backbone}.json")
    with open(metrics_path, "w") as f:
        json.dump({"backbone": args.backbone, "train": train_m, "val": val_m, "test": test_m},
                  f, indent=2)
    print(f"Metrics saved: {metrics_path}")

    state_dict_path = os.path.join(model_dir, f"{args.dataset}_{args.backbone}_cnn.pt")
    torch.save({
        "state_dict": cnn.state_dict(),
        "backbone":   args.backbone,
        "num_classes": num_classes,
    }, state_dict_path)
    print(f"CNN weights saved: {state_dict_path}")


if __name__ == "__main__":
    main()
