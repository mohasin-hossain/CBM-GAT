import os, json, argparse, random
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from dgl.dataloading import GraphDataLoader
from graph import load_split, infer_dims
from model import EGATClassifier, GAT_LightningModule
from config import default_output_dir, get_dataset_params, DATASETS
from utils import _set_seed

@torch.no_grad()
def evaluate(model, loader, device: str, num_classes: int):
    model.eval()
    ys, probs = [], []
    for bg, y in loader:
        bg = bg.to(device)
        ys.append(y.cpu())
        x = bg.ndata["feat"].float().to(device)
        logits, _, _ = model(bg, x)
        probs.append(F.softmax(logits, dim=1).cpu())
    y_true = torch.cat(ys).numpy()
    y_prob = torch.cat(probs).numpy()
    y_pred = y_prob.argmax(axis=1)
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="weighted")
    try:
        auc = (roc_auc_score(y_true, y_prob[:, 1], average="macro")
               if num_classes == 2
               else roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
    except Exception:
        auc = float("nan")
    return {"acc": acc, "f1": f1, "auc": auc}


def main():
    ap = argparse.ArgumentParser("train our model on selected datasets")
    ap.add_argument("--dataset", required=True, choices=list(DATASETS.keys()))
    ap.add_argument("--output-root", type=str, default=default_output_dir)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--num-heads", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--checkpoint-path", default=None, help="No checkpoint given, training from scratch.")
    ap.add_argument("--save-model", action="store_true", help="When true, save the model checkpoints.")
    args = ap.parse_args()

    # get dataset-specific parameters
    ds_params = get_dataset_params(args.dataset) if args.dataset else {}

    # num_head and batch size I got dirrectly from config if the relevant args are none
    # only assign if dataset and key exists in config
    if ds_params:
        if "num_heads" in ds_params and args.num_heads==None:
            args.num_heads = ds_params["num_heads"]

        if "batch_size" in ds_params and args.batch_size==None:
            args.batch_size = ds_params["batch_size"]

    _set_seed(args.seed)
    device = args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu"

    current_dir = os.getcwd()
    args.output_root = os.path.join(current_dir, args.output_root)

    # load data
    train_ds = load_split(args.output_root, args.dataset, "train", device)
    val_ds   = load_split(args.output_root, args.dataset, "validation", device)
    test_ds  = load_split(args.output_root, args.dataset, "test", device)

    train_loader = GraphDataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  drop_last=False)
    val_loader   = GraphDataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, drop_last=False)
    test_loader  = GraphDataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, drop_last=False)

    in_dim, num_classes = infer_dims(train_ds)

    # GAT model
    gat_model = EGATClassifier(
        in_feats=in_dim,
        out_feats=args.hidden_dim,
        num_heads=args.num_heads,
        out_dim=num_classes,
        feat_drop=0.0,
        node_drop=0.0,
    )
    # Traning model
    lightning_model = GAT_LightningModule(
        model=gat_model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=args.epochs,
        num_classes=num_classes,
        class_weights=None,
        l1_loss_alpha=0.0,
    )

    # checkpoints / logs dir define
    run_id = args.dataset
    model_dir = os.path.join(args.output_root, args.dataset, "models", run_id)
    os.makedirs(model_dir, exist_ok=True)

    # train / resume
    if args.checkpoint_path is None:

        checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=model_dir,
        filename="{}_best_model".format(run_id),
        mode="min",
        save_weights_only=True,
        save_top_k=1)

        early_stop_callback = EarlyStopping(monitor="val_loss", patience=args.patience, mode="min", verbose=True)
        lr_monitor = LearningRateMonitor(logging_interval="epoch")

        trainer = pl.Trainer(
            max_epochs=args.epochs,
            callbacks=[checkpoint_callback, lr_monitor, early_stop_callback],
            enable_progress_bar=True,
            accelerator="gpu" if device.startswith("cuda") else "cpu",
            devices=1,
            log_every_n_steps=1,
        )

        trainer.fit(lightning_model, train_loader, val_loader)

        best_ckpt_path = checkpoint_callback.best_model_path
        best_model = GAT_LightningModule.load_from_checkpoint(best_ckpt_path, model=gat_model)
        
        if args.save_model==True:
            print(f"\nBest checkpoint: {best_ckpt_path}") 
        else:
            try:
                os.remove(best_ckpt_path) # remove the check-point
            except OSError as e:
                print(f"Warning: Could not delete checkpoint {best_ckpt_path}: {e}")

    else:
        best_ckpt_path = args.checkpoint_path
        best_model = GAT_LightningModule.load_from_checkpoint(best_ckpt_path, model=gat_model)

    # eval (ACC/AUC/F1 on train/val/test)
    best_model.eval().to(device)
    train_metrics = evaluate(best_model.model.to(device), train_loader, device, num_classes)
    val_metrics   = evaluate(best_model.model.to(device),   val_loader, device, num_classes)
    test_metrics  = evaluate(best_model.model.to(device),  test_loader, device, num_classes)

    print("\n=== FINAL ===")
    print(f"Train: acc={train_metrics['acc']:.4f}  f1={train_metrics['f1']:.4f}  auc={train_metrics['auc']:.4f}")
    print(f"Val:   acc={val_metrics['acc']:.4f}  f1={val_metrics['f1']:.4f}  auc={val_metrics['auc']:.4f}")
    print(f"Test:  acc={test_metrics['acc']:.4f}  f1={test_metrics['f1']:.4f}  auc={test_metrics['auc']:.4f}")

    with open(os.path.join(model_dir, "metrics.json"), "w") as f:
        json.dump({"train": train_metrics, "val": val_metrics, "test": test_metrics}, f, indent=2)
    print(f"Metrics saved:   {os.path.join(model_dir, 'metrics.json')}")


if __name__ == "__main__":
    main()