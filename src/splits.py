"""Build a leakage-safe train/val/test manifest (data/splits.csv).

The original Kaggle split has a 16-image validation set. Here the original
train+val images are pooled and re-split into train/val with a stratified,
patient-grouped split (no patient appears in both). The original 624-image test
set is kept untouched. Images are not moved; the manifest records the split.

Usage: python -m src.splits
"""
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MANIFEST = DATA_DIR / "splits.csv"
CLASSES = ["NORMAL", "PNEUMONIA"]  # NORMAL=0, PNEUMONIA=1 (same as ImageFolder)

_PNEUMONIA_ID = re.compile(r"person(\d+)_(bacteria|virus)", re.I)
_NORMAL_ID = re.compile(r"((?:NORMAL2-)?IM-\d+)", re.I)


def patient_id(path: Path, label: str) -> str:
    """Kaggle filenames encode the patient: personN_{bacteria|virus}_* (pneumonia), IM-XXXX-* (normal)."""
    if label == "PNEUMONIA":
        m = _PNEUMONIA_ID.search(path.name)
    else:
        m = _NORMAL_ID.search(path.name)
    if not m:
        return f"{label}:{path.stem}"
    # personN numbering is reused across pathogen types, so the type is part of the patient key
    return f"{label}:{'_'.join(m.groups()).upper()}"


def collect(orig_split):
    rows = []
    for label in CLASSES:
        for p in sorted((DATA_DIR / orig_split / label).glob("*.jp*g")):
            rows.append({"path": p.relative_to(DATA_DIR).as_posix(), "label": CLASSES.index(label),
                         "orig_split": orig_split, "group": patient_id(p, label)})
    return rows


def build(val_folds=6, seed=42):
    pool = pd.DataFrame(collect("train") + collect("val"))
    test = pd.DataFrame(collect("test"))

    # Drop pool images from patients that also appear in the test set (patient-level leakage).
    overlap = set(pool["group"]) & set(test["group"])
    if overlap:
        print(f"Removing {int((pool['group'].isin(overlap)).sum())} pool images from "
              f"{len(overlap)} patients that also appear in the test set")
        pool = pool[~pool["group"].isin(overlap)].reset_index(drop=True)

    sgkf = StratifiedGroupKFold(n_splits=val_folds, shuffle=True, random_state=seed)
    _, val_idx = next(sgkf.split(pool, pool["label"], pool["group"]))
    pool["split"] = "train"
    pool.loc[pool.index[val_idx], "split"] = "val"
    test["split"] = "test"

    df = pd.concat([pool, test], ignore_index=True)
    df.to_csv(MANIFEST, index=False)
    return df


def summarize(df):
    t = df.groupby(["split", "label"]).size().unstack(fill_value=0)
    t.columns = CLASSES
    t["total"] = t.sum(axis=1)
    t["PNEUMONIA:NORMAL"] = (t["PNEUMONIA"] / t["NORMAL"]).round(2)
    print(t.loc[["train", "val", "test"]])
    g = df.groupby("group")["split"].nunique()
    print(f"Patients in >1 split: {int((g > 1).sum())}")


if __name__ == "__main__":
    df = build()
    summarize(df)
    print(f"Saved {MANIFEST}")
