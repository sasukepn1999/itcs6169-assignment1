"""Re-run the starter-notebook baseline (TNet, grayscale 64x64, Adam 2e-3, 20 epochs) as a script.

Same seed, split and model as the notebook; additionally saves training curves and
per-class validation accuracy of the best-validation checkpoint.
"""
import argparse
import copy
import json
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="data")
parser.add_argument("--figs", default="../Report/figs")
parser.add_argument("--out", default="results/baseline_starter.json")
args = parser.parse_args()

SEED, IMG_SIZE, BATCH_SIZE, VAL_FRACTION, EPOCHS = 0, 64, 64, 0.20, 20


def set_random_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_random_seed()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])
full = datasets.ImageFolder(os.path.join(args.data, "train"), transform=tf)
test = datasets.ImageFolder(os.path.join(args.data, "test"), transform=tf)
classes = full.classes
val_size = int(round(len(full) * VAL_FRACTION))
train_ds, val_ds = random_split(full, [len(full) - val_size, val_size],
                                generator=torch.Generator().manual_seed(SEED))
kw = dict(batch_size=BATCH_SIZE, num_workers=2, pin_memory=True)
train_loader = DataLoader(train_ds, shuffle=True, **kw)
val_loader = DataLoader(val_ds, shuffle=False, **kw)
test_loader = DataLoader(test, shuffle=False, **kw)


class TNet(nn.Module):
    def __init__(self, num_classes=16):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(1, 16, 3), nn.ReLU(inplace=True), nn.MaxPool2d(4, 4))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(16 * 15 * 15, num_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


@torch.inference_mode()
def evaluate(model, loader):
    model.eval()
    crit = nn.CrossEntropyLoss(reduction="sum")
    loss, preds, labels = 0.0, [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss += crit(logits, y).item()
        preds.append(logits.argmax(1).cpu())
        labels.append(y.cpu())
    preds, labels = torch.cat(preds), torch.cat(labels)
    return loss / len(labels), (preds == labels).float().mean().item(), preds, labels


set_random_seed()
model = TNet(len(classes)).to(device)
opt = optim.Adam(model.parameters(), lr=0.002)
crit = nn.CrossEntropyLoss()
hist = {"train_loss": [], "val_loss": [], "val_acc": []}
best_acc, best_state, best_epoch = 0.0, None, 0
for epoch in range(1, EPOCHS + 1):
    model.train()
    tot, seen = 0.0, 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        loss = crit(model(x), y)
        loss.backward()
        opt.step()
        tot += loss.item() * x.size(0)
        seen += x.size(0)
    vl, va, _, _ = evaluate(model, val_loader)
    hist["train_loss"].append(tot / seen)
    hist["val_loss"].append(vl)
    hist["val_acc"].append(va)
    if va > best_acc:
        best_acc, best_state, best_epoch = va, copy.deepcopy(model.state_dict()), epoch
    print(f"epoch {epoch:02d} train {tot / seen:.4f} val_loss {vl:.4f} val_acc {va:.4f}")

model.load_state_dict(best_state)
_, va, vp, vy = evaluate(model, val_loader)
_, ta, _, _ = evaluate(model, test_loader)
per_class = {c: (vp[vy == i] == i).float().mean().item() for i, c in enumerate(classes)}
print(f"best val acc {best_acc:.4f} (epoch {best_epoch}) | test acc {ta:.4f}")
for c, a in sorted(per_class.items(), key=lambda kv: kv[1]):
    print(f"  {c:13s} {a:.3f}  (n={(vy == classes.index(c)).sum().item()})")

os.makedirs(os.path.dirname(args.out), exist_ok=True)
json.dump({"history": hist, "best_val_acc": best_acc, "best_epoch": best_epoch,
           "test_acc": ta, "per_class_val_acc": per_class,
           "torch": torch.__version__}, open(args.out, "w"), indent=2)

fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.4))
ep = range(1, EPOCHS + 1)
axes[0].plot(ep, hist["train_loss"], label="train")
axes[0].plot(ep, hist["val_loss"], label="validation")
axes[0].set_xlabel("epoch"); axes[0].set_ylabel("cross-entropy"); axes[0].legend(fontsize=7)
names = sorted(per_class, key=per_class.get)
axes[1].barh(names, [per_class[n] for n in names], color="tab:blue")
axes[1].set_xlabel("val accuracy (best ckpt)"); axes[1].tick_params(axis="y", labelsize=6)
axes[1].set_xlim(0, 1)
fig.tight_layout()
os.makedirs(args.figs, exist_ok=True)
fig.savefig(os.path.join(args.figs, "baseline.png"), dpi=200)
print("saved", os.path.join(args.figs, "baseline.png"))
