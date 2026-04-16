"""
Generate ham10000_multiclass dataset CSVs (3-class: nv=0, mel=1, bkl=2).

Reads:
  /ds-iml/cbm-gat/ham10000/HAM10000_metadata
  /ds-iml/cbm-gat/ham10000/HAM10000_images_part_1/
  /ds-iml/cbm-gat/ham10000/HAM10000_images_part_2/

Writes to:
  /home/mhossain/projects/CBM-GAT/datasets/ham10000_multiclass/
    all.csv                  — all 3-class images, no header split column
    all_balanced.csv         — balanced by oversampling mel+bkl to match nv count
    train.csv                — 70% stratified
    validation.csv           — 15% stratified
    test.csv                 — 15% stratified
    train_balanced.csv       — balanced train split

Format matches existing binary CSVs:
  image_path,labels   (train/val/test)
  image_path,labels,split  (all.csv / all_balanced.csv)
"""

import os
import csv
import random
from collections import defaultdict

random.seed(42)

METADATA_PATH = "/ds-iml/cbm-gat/ham10000/HAM10000_metadata"
IMG_PART1 = "/ds-iml/cbm-gat/ham10000/HAM10000_images_part_1"
IMG_PART2 = "/ds-iml/cbm-gat/ham10000/HAM10000_images_part_2"
OUT_DIR = "/home/mhossain/projects/CBM-GAT/datasets/ham10000_multiclass"

CLASS_MAP = {"nv": 0, "mel": 1, "bkl": 2}
CLASS_NAMES = {0: "Nevi", 1: "Melanoma", 2: "Benign Keratosis"}

os.makedirs(OUT_DIR, exist_ok=True)

# Build image_id -> relative path mapping
print("Scanning image folders...")
image_lookup = {}
for fname in os.listdir(IMG_PART1):
    if fname.endswith(".jpg"):
        image_id = fname.replace(".jpg", "")
        image_lookup[image_id] = f"HAM10000_images_part_1/{fname}"
for fname in os.listdir(IMG_PART2):
    if fname.endswith(".jpg"):
        image_id = fname.replace(".jpg", "")
        image_lookup[image_id] = f"HAM10000_images_part_2/{fname}"

print(f"  Found {len(image_lookup)} images total")

# Parse metadata — skip duplicate lesion rows (same lesion, multiple images)
# HAM10000 has multiple images per lesion_id. We keep one image per lesion
# to avoid data leakage across splits.
print("Parsing metadata...")
seen_lesions = set()
records = []  # list of (image_path, label)

with open(METADATA_PATH, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        dx = row["dx"].strip()
        if dx not in CLASS_MAP:
            continue
        lesion_id = row["lesion_id"].strip()
        image_id = row["image_id"].strip()
        if image_id not in image_lookup:
            continue
        # One image per lesion to avoid leakage
        if lesion_id in seen_lesions:
            continue
        seen_lesions.add(lesion_id)
        records.append((image_lookup[image_id], CLASS_MAP[dx]))

print(f"  Total unique-lesion records: {len(records)}")

# Show per-class count
class_counts = defaultdict(int)
for _, label in records:
    class_counts[label] += 1
for label, count in sorted(class_counts.items()):
    print(f"  Class {label} ({CLASS_NAMES[label]}): {count}")

# Group by class for stratified splitting
by_class = defaultdict(list)
for rec in records:
    by_class[rec[1]].append(rec)

for label in by_class:
    random.shuffle(by_class[label])

# Stratified split: 70/15/15
def split_class(items, train_frac=0.70, val_frac=0.15):
    n = len(items)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    return items[:n_train], items[n_train:n_train + n_val], items[n_train + n_val:]

train_all, val_all, test_all = [], [], []
for label in sorted(by_class.keys()):
    tr, va, te = split_class(by_class[label])
    train_all.extend(tr)
    val_all.extend(va)
    test_all.extend(te)
    print(f"  Class {label} splits: train={len(tr)}, val={len(va)}, test={len(te)}")

random.shuffle(train_all)
random.shuffle(val_all)
random.shuffle(test_all)

# Balanced training set: oversample minority classes to match majority
max_train_count = max(
    sum(1 for _, l in train_all if l == c) for c in CLASS_MAP.values()
)
print(f"\nBalancing train set to {max_train_count} per class...")

balanced_train = []
for label in sorted(by_class.keys()):
    class_train = [(p, l) for p, l in train_all if l == label]
    while len(class_train) < max_train_count:
        class_train += class_train
    class_train = class_train[:max_train_count]
    balanced_train.extend(class_train)
random.shuffle(balanced_train)

# Write helper
def write_csv(path, rows, include_split_col=False, split_name=None):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        if include_split_col:
            writer.writerow(["image_path", "labels", "split"])
            for img_path, label in rows:
                writer.writerow([img_path, label, split_name])
        else:
            writer.writerow(["image_path", "labels"])
            for img_path, label in rows:
                writer.writerow([img_path, label])
    print(f"  Wrote {len(rows)} rows to {os.path.basename(path)}")

print("\nWriting CSVs...")
write_csv(f"{OUT_DIR}/train.csv", train_all)
write_csv(f"{OUT_DIR}/validation.csv", val_all)
write_csv(f"{OUT_DIR}/test.csv", test_all)
write_csv(f"{OUT_DIR}/train_balanced.csv", balanced_train)

# all.csv = all splits with split column
all_rows_with_split = (
    [(p, l, "train") for p, l in train_all]
    + [(p, l, "val") for p, l in val_all]
    + [(p, l, "test") for p, l in test_all]
)
with open(f"{OUT_DIR}/all.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image_path", "labels", "split"])
    for img_path, label, split in all_rows_with_split:
        writer.writerow([img_path, label, split])
print(f"  Wrote {len(all_rows_with_split)} rows to all.csv")

# all_balanced.csv = balanced train + val + test
all_balanced_rows = (
    [(p, l, "train") for p, l in balanced_train]
    + [(p, l, "val") for p, l in val_all]
    + [(p, l, "test") for p, l in test_all]
)
with open(f"{OUT_DIR}/all_balanced.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image_path", "labels", "split"])
    for img_path, label, split in all_balanced_rows:
        writer.writerow([img_path, label, split])
print(f"  Wrote {len(all_balanced_rows)} rows to all_balanced.csv")

print("\nDone. Summary:")
print(f"  Train:          {len(train_all)}")
print(f"  Train balanced: {len(balanced_train)}")
print(f"  Validation:     {len(val_all)}")
print(f"  Test:           {len(test_all)}")
print(f"\nClass names: {CLASS_NAMES}")
print(f"Output dir:  {OUT_DIR}")
