"""
Generate CUB-200-2011 dataset CSVs for the GCBM pipeline.

Reads official metadata from:
  /ds-iml/cbm-gat/CUB_200_2011/CUB_200_2011/CUB_200_2011/

Writes to:
  <repo>/datasets/cub/
    train.csv          — 90% of official train (stratified per class, seed 42)
    validation.csv     — 10% of official train
    test.csv           — official test split (unchanged)
    nmf.csv            — all official train images (concept discovery)
    all.csv            — all images with split column
    class_names.txt    — one species name per line, label index = line number

CSV format matches other datasets:
  image_path,labels   (train / validation / test / nmf)
  image_path,labels,split  (all.csv)

Labels are 0-indexed (official CUB class ids 1..200 → 0..199).
"""

from __future__ import annotations

import csv
import os
import random
from collections import defaultdict
from typing import Dict, List, Tuple

random.seed(42)

CUB_META_ROOT = "/ds-iml/cbm-gat/CUB_200_2011/CUB_200_2011/CUB_200_2011"
CUB_IMAGES_ROOT = os.path.join(CUB_META_ROOT, "images")
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(_SCRIPT_DIR, "cub")

VAL_RATIO = 0.10
EXPECTED_TRAIN = 5994
EXPECTED_TEST = 5794
EXPECTED_TOTAL = 11788


def _parse_id_map(path: str, value_fn=str) -> Dict[int, str]:
    """Parse 'id value' lines; value may contain spaces (only first token is id)."""
    out: Dict[int, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                raise ValueError(f"Bad line in {path}: {line!r}")
            out[int(parts[0])] = value_fn(parts[1])
    return out


def _load_class_names() -> List[str]:
    """Return 200 human-readable names, index = label 0..199."""
    raw = _parse_id_map(os.path.join(CUB_META_ROOT, "classes.txt"))
    names = [""] * len(raw)
    for class_id, folder_name in raw.items():
        idx = class_id - 1
        # e.g. 001.Black_footed_Albatross → Black footed Albatross
        name = folder_name.split(".", 1)[-1].replace("_", " ")
        names[idx] = name
    if any(n == "" for n in names):
        raise RuntimeError("Incomplete class name table from classes.txt")
    return names


def _load_records() -> List[Tuple[str, int, bool]]:
    """Return list of (rel_image_path, label, is_official_train)."""
    images = _parse_id_map(os.path.join(CUB_META_ROOT, "images.txt"))
    labels = {
        int(k): int(v)
        for k, v in _parse_id_map(
            os.path.join(CUB_META_ROOT, "image_class_labels.txt")
        ).items()
    }
    splits = {
        int(k): int(v)
        for k, v in _parse_id_map(
            os.path.join(CUB_META_ROOT, "train_test_split.txt")
        ).items()
    }

    if set(images) != set(labels) or set(images) != set(splits):
        raise RuntimeError("images / labels / split id sets do not match")

    records: List[Tuple[str, int, bool]] = []
    missing = 0
    for img_id in sorted(images):
        rel_path = images[img_id]
        abs_path = os.path.join(CUB_IMAGES_ROOT, rel_path)
        if not os.path.isfile(abs_path):
            missing += 1
        label = labels[img_id] - 1
        if label < 0 or label >= 200:
            raise ValueError(f"Unexpected class id {labels[img_id]} for image {img_id}")
        is_train = splits[img_id] == 1
        records.append((rel_path, label, is_train))

    if missing:
        print(
            f"  WARNING: {missing}/{len(records)} image paths not found under "
            f"{CUB_IMAGES_ROOT} (CSVs still written from metadata)."
        )

    return records


def _stratified_train_val(
    train_records: List[Tuple[str, int]],
    val_ratio: float,
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    by_class: Dict[int, List[Tuple[str, int]]] = defaultdict(list)
    for rec in train_records:
        by_class[rec[1]].append(rec)

    train_out: List[Tuple[str, int]] = []
    val_out: List[Tuple[str, int]] = []
    for label in sorted(by_class):
        items = by_class[label][:]
        random.shuffle(items)
        n = len(items)
        n_val = max(1, int(round(n * val_ratio))) if n > 1 else 0
        if n_val >= n:
            n_val = n - 1
        val_out.extend(items[:n_val])
        train_out.extend(items[n_val:])

    random.shuffle(train_out)
    random.shuffle(val_out)
    return train_out, val_out


def _write_csv(path: str, rows: List[Tuple[str, int]], header: str = "image_path,labels") -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header.split(","))
        for img_path, label in rows:
            writer.writerow([img_path, label])


def _class_hist(rows: List[Tuple[str, int]]) -> Dict[int, int]:
    hist: Dict[int, int] = defaultdict(int)
    for _, label in rows:
        hist[label] += 1
    return hist


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"CUB meta root:   {CUB_META_ROOT}")
    print(f"CUB images root: {CUB_IMAGES_ROOT}")
    print(f"Output dir:      {OUT_DIR}")

    class_names = _load_class_names()
    records = _load_records()
    print(f"Loaded {len(records)} images")

    official_train = [(p, l) for p, l, is_tr in records if is_tr]
    official_test = [(p, l) for p, l, is_tr in records if not is_tr]

    if len(official_train) != EXPECTED_TRAIN or len(official_test) != EXPECTED_TEST:
        raise RuntimeError(
            f"Unexpected official split sizes: train={len(official_train)}, "
            f"test={len(official_test)} (expected {EXPECTED_TRAIN}/{EXPECTED_TEST})"
        )

    train_rows, val_rows = _stratified_train_val(official_train, VAL_RATIO)
    nmf_rows = official_train[:]
    random.shuffle(nmf_rows)

    # Sanity: partitions
    train_paths = {p for p, _ in train_rows}
    val_paths = {p for p, _ in val_rows}
    test_paths = {p for p, _ in official_test}
    nmf_paths = {p for p, _ in nmf_rows}
    assert train_paths.isdisjoint(val_paths)
    assert train_paths.isdisjoint(test_paths)
    assert val_paths.isdisjoint(test_paths)
    assert nmf_paths == train_paths | val_paths

    print("\nWriting CSVs...")
    _write_csv(os.path.join(OUT_DIR, "train.csv"), train_rows)
    _write_csv(os.path.join(OUT_DIR, "validation.csv"), val_rows)
    _write_csv(os.path.join(OUT_DIR, "test.csv"), official_test)
    _write_csv(os.path.join(OUT_DIR, "nmf.csv"), nmf_rows)

    all_rows = (
        [(p, l, "train") for p, l in train_rows]
        + [(p, l, "validation") for p, l in val_rows]
        + [(p, l, "test") for p, l in official_test]
    )
    with open(os.path.join(OUT_DIR, "all.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "labels", "split"])
        for img_path, label, split in all_rows:
            writer.writerow([img_path, label, split])

    with open(os.path.join(OUT_DIR, "class_names.txt"), "w", encoding="utf-8") as f:
        for name in class_names:
            f.write(name + "\n")

    print(f"  train.csv:       {len(train_rows)}")
    print(f"  validation.csv:  {len(val_rows)}")
    print(f"  test.csv:        {len(official_test)}")
    print(f"  nmf.csv:         {len(nmf_rows)}")
    print(f"  all.csv:         {len(all_rows)}")
    print(f"  class_names.txt: {len(class_names)} names")

    total = len(train_rows) + len(val_rows) + len(official_test)
    if total != EXPECTED_TOTAL:
        raise RuntimeError(f"Split total {total} != {EXPECTED_TOTAL}")

    for name, rows in [
        ("train", train_rows),
        ("validation", val_rows),
        ("test", official_test),
    ]:
        hist = _class_hist(rows)
        counts = list(hist.values())
        print(
            f"  {name}: {len(hist)} classes present, "
            f"min={min(counts)} max={max(counts)} images/class"
        )

    # Spot-check one train image if present on disk
    sample_path, sample_label = train_rows[0]
    sample_abs = os.path.join(CUB_IMAGES_ROOT, sample_path)
    if os.path.isfile(sample_abs):
        print(f"\nSample: label={sample_label} ({class_names[sample_label]})")
        print(f"        path={sample_path}")
    else:
        print(f"\nSample (metadata): label={sample_label} ({class_names[sample_label]})")
        print(f"                   path={sample_path}  [file not on disk yet]")
    print("\nDone.")


if __name__ == "__main__":
    main()
