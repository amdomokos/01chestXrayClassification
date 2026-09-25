"""Train ResNet18 on the leakage-safe splits. Model selection uses the VAL split only.

Usage: python -m src.train [--device auto|xpu|cuda|cpu] [--epochs 15] [--batch-size 32]
"""
import argparse
import json
import time
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from sklearn.metrics import roc_auc_score

from .data_loader import make_loaders
from .splits import ROOT
from .utils import device_name, get_device, set_seed

MODEL_DIR = ROOT / "outputs" / "models"


def build_model(num_classes=2, pretrained=True):
    weights = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
    model = torchvision.models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


@torch.no_grad()
def run_eval(model, loader, criterion, device):
    model.eval()
    loss_sum, correct, n = 0.0, 0, 0
    labels_all, probs_all = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss_sum += criterion(out, y).item() * x.size(0)
        correct += out.argmax(1).eq(y).sum().item()
        n += x.size(0)
        labels_all.append(y.cpu())
        probs_all.append(torch.softmax(out.float(), 1)[:, 1].cpu())
    auc = roc_auc_score(torch.cat(labels_all).numpy(), torch.cat(probs_all).numpy())
    return loss_sum / n, correct / n, auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--amp", action="store_true", help="bfloat16 autocast (XPU/CUDA)")
    ap.add_argument("--out", default="resnet18_pneumonia.pth")
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    print(f"Device: {device} ({device_name(device)})")

    train_loader, val_loader, _, (train_ds, val_ds, test_ds) = make_loaders(args.batch_size, args.workers)
    print(f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} (test is not touched here)")

    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    use_amp = args.amp and device.type in ("xpu", "cuda")

    history, best_auc, best_state, start = [], -1.0, None, time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum, correct, n = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                out = model(x)
                loss = criterion(out.float(), y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * x.size(0)
            correct += out.argmax(1).eq(y).sum().item()
            n += x.size(0)
        val_loss, val_acc, val_auc = run_eval(model, val_loader, criterion, device)
        rec = dict(epoch=epoch, train_loss=loss_sum / n, train_acc=correct / n,
                   val_loss=val_loss, val_acc=val_acc, val_auc=val_auc)
        history.append(rec)
        print(f"[{epoch:02d}/{args.epochs}] train loss {rec['train_loss']:.4f} acc {rec['train_acc']:.4f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.4f} auc {val_auc:.4f} ({time.time() - start:.0f}s)")
        if val_auc > best_auc:
            best_auc, best_state = val_auc, deepcopy(model.state_dict())
            print("  -> new best (val AUC)")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    cpu_state = {k: v.cpu() for k, v in best_state.items()}
    torch.save(cpu_state, MODEL_DIR / args.out)
    (MODEL_DIR / (args.out.replace(".pth", "_history.json"))).write_text(
        json.dumps(dict(args=vars(args), device=device_name(device), best_val_auc=best_auc,
                        history=history), indent=2))
    print(f"Saved best model (val AUC {best_auc:.4f}) -> {MODEL_DIR / args.out}")


if __name__ == "__main__":
    main()
