"""Evaluate a trained model ONCE on the held-out test split, with bootstrap 95% CIs.

Usage: python -m src.evaluate [--weights resnet18_pneumonia.pth]
"""
import argparse
import json

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score

from .data_loader import ManifestDataset, test_transforms
from .splits import ROOT
from .train import MODEL_DIR, build_model
from .utils import device_name, get_device

from torch.utils.data import DataLoader


def metrics(y, p, thr=0.5):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(auc=roc_auc_score(y, p), sensitivity=tp / (tp + fn), specificity=tn / (tn + fp),
                accuracy=(tp + tn) / len(y), tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))


def bootstrap_ci(y, p, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    keys = ("auc", "sensitivity", "specificity", "accuracy")
    draws = {k: [] for k in keys}
    for _ in range(n_boot):  # stratified resampling keeps both classes present
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        m = metrics(y[idx], p[idx])
        for k in keys:
            draws[k].append(m[k])
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in draws.items()}


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        ps.append(torch.softmax(model(x.to(device)).float(), 1)[:, 1].cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="resnet18_pneumonia.pth")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    device = get_device(args.device)
    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(MODEL_DIR / args.weights, map_location="cpu", weights_only=True))
    model.to(device)

    ds = ManifestDataset("test", test_transforms)
    y, p = predict(model, DataLoader(ds, args.batch_size, shuffle=False, num_workers=4), device)
    m, ci = metrics(y, p), bootstrap_ci(y, p)

    print(f"Test set: {len(y)} images ({(y == 0).sum()} NORMAL / {(y == 1).sum()} PNEUMONIA) on {device_name(device)}")
    for k in ("auc", "sensitivity", "specificity", "accuracy"):
        print(f"  {k:<12} {m[k]:.4f}  (95% CI {ci[k][0]:.4f}-{ci[k][1]:.4f})")
    print(f"  confusion: TN={m['tn']} FP={m['fp']} FN={m['fn']} TP={m['tp']}")

    out = ROOT / "outputs" / "test_metrics.json"
    out.write_text(json.dumps(dict(weights=args.weights, n_test=int(len(y)), metrics=m, ci95=ci), indent=2))
    np.savez(ROOT / "outputs" / "test_predictions.npz", y=y, p=p)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
