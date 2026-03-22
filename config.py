import os, random
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple, Any, List
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F
from torchvision import transforms

from utils import ImageDataset, _tensors_from_loader

current_dir = os.getcwd()
# Dataset CSVs live in the repo; use config's location so paths work when cwd is elsewhere (e.g. SLURM workdir)
_config_dir = os.path.dirname(os.path.abspath(__file__))
default_output_dir = os.path.join(current_dir, "concept_graph_data")
default_eval_dir = os.path.join(current_dir, "results")
default_datasets_dir = os.path.join(_config_dir, "datasets")



@dataclass
class DatasetSpec:
    name: str
    build_transforms: Callable[[], Dict[str, transforms.Compose]]
    resolve_paths: Callable[[], Dict[str, str]]
    # load_split takes mode ("nmf"/"train"/"val"/"test") and returns tensors for that specific split
    load_split: Callable[[Dict[str, str], Dict[str, transforms.Compose], str], Tuple[torch.Tensor, torch.Tensor, Optional[List[Any]]]]


# custom small transform used in 90° rotation ---
class RandomRotate90:
    def __call__(self, img):
        rotations = random.choice([0, 1, 2, 3])
        return F.rotate(img, angle=rotations * 90)

def _shared_medical_data_transforms():
    nmf_train = transforms.Compose([
        transforms.Resize(270, interpolation=InterpolationMode.BILINEAR, antialias=True),
        transforms.CenterCrop(224),
        RandomRotate90(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    ev = transforms.Compose([
        transforms.Resize(270, interpolation=InterpolationMode.BILINEAR, antialias=True),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    return {"nmf": nmf_train, "train": nmf_train, "eval": ev}

def _load_one_csv(images_root: str, csv_path: str, tfm, batch_size: int = 64):
    ds = ImageDataset(csv_file=csv_path, root_dir=images_root, transform=tfm)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=False)
    return _tensors_from_loader(dl)

# ---------- PH2 ----------
def _ph2_build_transforms():
    return _shared_medical_data_transforms()

def _ph2_resolve_paths():
    base = "/ds-iml/cbm-gat"
    return {
        "images_root": os.path.join(base, "PH2Dataset/trainx"),
        "nmf_csv": os.path.join(default_datasets_dir, "ph2dataset/PH2_all_balanced.csv"),
        "train_csv": os.path.join(default_datasets_dir, "ph2dataset/PH2_train_balanced.csv"),
        "val_csv": os.path.join(default_datasets_dir, "ph2dataset/PH2_validation.csv"),
        "test_csv": os.path.join(default_datasets_dir, "ph2dataset/PH2_test.csv"),
    }

def _ph2_load_split(paths, tdict, split):
    if split == "nmf":
        tfm = tdict["nmf"]
        csv_path = paths["nmf_csv"]
    elif split in ("train","val","test"):
        tfm = tdict["eval"]
        csv_path = paths[f"{split}_csv"]
    else:
        raise ValueError(f"Unknown split: {split}")
    X, Y = _load_one_csv(paths["images_root"], csv_path, tfm)
    return X, Y, None

# ---------- HAM10000 ----------
def _ham_build_transforms():
    return _shared_medical_data_transforms()

def _ham_resolve_paths():
    base = "/ds-iml/cbm-gat"
    return {
        "images_root": os.path.join(base, "ham10000"),
        "nmf_csv":   os.path.join(default_datasets_dir, "ham10000/all_balanced.csv"),
        "train_csv": os.path.join(default_datasets_dir, "ham10000/train_balanced.csv"),
        "val_csv":   os.path.join(default_datasets_dir, "ham10000/validation.csv"),
        "test_csv":  os.path.join(default_datasets_dir, "ham10000/test.csv"),
    }

def _ham_load_split(paths, tdict, split):
    if split == "nmf":
        tfm = tdict["nmf"]
        csv_path = paths["nmf_csv"]
    elif split in ("train","val","test"):
        tfm = tdict["eval"]
        csv_path = paths[f"{split}_csv"]
    else:
        raise ValueError(f"Unknown split: {split}")
    X, Y = _load_one_csv(paths["images_root"], csv_path, tfm)
    return X, Y, None

# ---------- Derm7pt (P7) ----------
def _derm7pt_build_transforms():
    return _shared_medical_data_transforms()

def _derm7pt_resolve_paths():
    base = "/ds-iml/cbm-gat"
    return {
        "images_root": os.path.join(base, "derm7pt/images"),
        "nmf_csv":   os.path.join(default_datasets_dir, "derm7pt/derm7pt_all_balanced.csv"),
        "train_csv": os.path.join(default_datasets_dir, "derm7pt/derm7pt_train_balanced.csv"),
        "val_csv":   os.path.join(default_datasets_dir, "derm7pt/derm7pt_validation.csv"),
        "test_csv":  os.path.join(default_datasets_dir, "derm7pt/derm7pt_test.csv"),
    }

def _derm7pt_load_split(paths, tdict, split):
    if split == "nmf":
        tfm = tdict["nmf"]
        csv_path = paths["nmf_csv"]
    elif split in ("train","val","test"):
        tfm = tdict["eval"]
        csv_path = paths[f"{split}_csv"]
    else:
        raise ValueError(f"Unknown split: {split}")
    X, Y = _load_one_csv(paths["images_root"], csv_path, tfm)
    return X, Y, None

# ---------- ImageNet ----------
def _imagenet_build_transforms():
    
    train = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0), ratio=(3/4, 4/3)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    ev = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return {"nmf": ev, "train": train, "eval": ev}

def _imagenet_resolve_paths():
    base = "/ds-iml/cbm-gat"
    return {
        "images_root": os.path.join(base, "imagenet"),
        "nmf_csv":   os.path.join(default_datasets_dir, "imagenet/nmf.csv"),
        "train_csv": os.path.join(default_datasets_dir, "imagenet/train.csv"),
        "val_csv":   os.path.join(default_datasets_dir, "imagenet/val.csv"),
        "test_csv":  os.path.join(default_datasets_dir, "imagenet/test.csv"),
    }

def _imagenet_load_split(paths, tdict, split):
    if split == "nmf":
        tfm = tdict["nmf"]
        csv_path = paths["nmf_csv"]
    elif split in ("train","val","test"):
        tfm = tdict["eval"]
        csv_path = paths[f"{split}_csv"]
    else:
        raise ValueError(f"Unknown split: {split}")
    X, Y = _load_one_csv(paths["images_root"], csv_path, tfm)
    return X, Y, None

# dataset keys, transformer, and meta-data handler
DATASETS: Dict[str, DatasetSpec] = {
    "ph2": DatasetSpec("PH2", _ph2_build_transforms, _ph2_resolve_paths, _ph2_load_split),
    "ham10000": DatasetSpec("HAM10000", _ham_build_transforms, _ham_resolve_paths, _ham_load_split),
    "derm7pt": DatasetSpec("Derm7pt", _derm7pt_build_transforms, _derm7pt_resolve_paths, _derm7pt_load_split),
    "imagenet": DatasetSpec("ImageNetSubset", _imagenet_build_transforms, _imagenet_resolve_paths, _imagenet_load_split),
}

# model configurations
MODEL_CFG = {
    "derm7pt": {
        "num_heads": 4,
        "hidden_dim": 128,
        "batch_size": 32,
    },
    "ph2": {
        "num_heads": 4,
        "hidden_dim": 128,
        "batch_size": 64,
    },
    "ham10000": {
        "num_heads": 6,
        "hidden_dim": 128,
        "batch_size": 128,
    },
    "imagenet": {
        "num_heads": 6,
        "hidden_dim": 128,
        "batch_size": 128,
    },
}

def get_dataset_params(dataset: str):
    """Return num_heads and batch_size for a given dataset, or None if not found."""
    return MODEL_CFG.get(dataset, {})
