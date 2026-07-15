"""Training script for GRACE stripe-noise denoising U-Net.

Usage:
    python -m python_tools.training.train         # full train
    python -m python_tools.training.train --epochs 5 --dry  # quick test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure proj root on path
_PROJ = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJ))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from python_tools.training.dataset import GRACEDenoiseDataset
from python_tools.training.model import UNet

# ── config ────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = _PROJ / "python_tools" / "training" / "runs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_CFG = dict(
    batch_size    = 32,
    epochs        = 200,
    lr            = 1e-3,
    weight_decay  = 1e-5,
    patch_size    = 64,
    patches_per   = 32,         # patches per grid per epoch
    base_filters  = 32,
    val_year      = 2014,       # years >= this → validation
)


# ── training loop ─────────────────────────────────────────────────

def train_one_epoch(model, loader, optim, criterion):
    model.train()
    total_loss = 0.0
    for raw, noise in loader:
        raw, noise = raw.to(DEVICE), noise.to(DEVICE)
        optim.zero_grad()
        pred = model(raw)
        loss = criterion(pred, noise)
        loss.backward()
        optim.step()
        total_loss += loss.item() * raw.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    for raw, noise in loader:
        raw, noise = raw.to(DEVICE), noise.to(DEVICE)
        pred = model(raw)
        total_loss += criterion(pred, noise).item() * raw.size(0)
    return total_loss / len(loader.dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",   type=int, default=DEFAULT_CFG["epochs"])
    parser.add_argument("--lr",       type=float, default=DEFAULT_CFG["lr"])
    parser.add_argument("--batch",    type=int, default=DEFAULT_CFG["batch_size"])
    parser.add_argument("--patch",    type=int, default=DEFAULT_CFG["patch_size"])
    parser.add_argument("--dry",      action="store_true", help="2 epochs only")
    parser.add_argument("--cpu",      action="store_true", help="Force CPU")
    args = parser.parse_args()

    device = torch.device("cpu") if args.cpu else DEVICE
    print(f"Device: {device}")
    print(f"Config: epochs={args.epochs}, lr={args.lr}, batch={args.batch}, "
          f"patch={args.patch}")

    # ── datasets ──────────────────────────────────────────────────
    print("\nBuilding datasets ...")
    train_ds = GRACEDenoiseDataset(
        split="train", val_year=DEFAULT_CFG["val_year"],
        patch_size=args.patch, patches_per_grid=DEFAULT_CFG["patches_per"],
    )
    val_ds = GRACEDenoiseDataset(
        split="val", val_year=DEFAULT_CFG["val_year"],
        patch_size=args.patch, patches_per_grid=DEFAULT_CFG["patches_per"],
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=0, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                              num_workers=0)

    # ── model ─────────────────────────────────────────────────────
    model = UNet(in_channels=1, out_channels=1,
                 base_filters=DEFAULT_CFG["base_filters"])
    model = model.to(device)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr,
                             weight_decay=DEFAULT_CFG["weight_decay"])
    criterion = nn.MSELoss()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters")

    # ── train ─────────────────────────────────────────────────────
    best_loss = float("inf")
    history = {"train": [], "val": []}

    n_epochs = 2 if args.dry else args.epochs
    for epoch in range(1, n_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optim, criterion)
        val_loss   = validate(model, val_loader, criterion)

        history["train"].append(train_loss)
        history["val"].append(val_loss)

        marker = ""
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), OUT_DIR / "best_model.pt")
            marker = " *"

        print(f"  Epoch {epoch:3d}/{n_epochs}  "
              f"train={train_loss:.6f}  val={val_loss:.6f}{marker}")

    # ── save ──────────────────────────────────────────────────────
    torch.save(model.state_dict(), OUT_DIR / "last_model.pt")
    np.savez(OUT_DIR / "history.npz", train=history["train"], val=history["val"])
    print(f"\nBest val loss: {best_loss:.6f}")
    print(f"Models saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
