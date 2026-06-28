import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class SCSEDataset(Dataset):
    def __init__(self, x_path, y_path):
        self.x = np.load(x_path, mmap_mode="r")
        self.y = np.load(y_path, mmap_mode="r")
        assert self.x.shape == self.y.shape, (self.x.shape, self.y.shape)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        x = self.x[idx].astype("float32") / 255.0
        y = self.y[idx].astype("int64")
        x = torch.from_numpy(x)[None, :, :]
        y = torch.from_numpy(y)
        return x, y


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, padding=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.05):
        super().__init__()
        self.net = nn.Sequential(
            ConvBNReLU(in_ch, out_ch),
            ConvBNReLU(out_ch, out_ch),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x):
        return self.net(x)


class MultiScaleFusion(nn.Module):
    def __init__(self, in_chs, out_ch, dropout=0.05):
        super().__init__()
        self.proj = nn.ModuleList(
            [ConvBNReLU(in_ch, out_ch, k=1, padding=0) for in_ch in in_chs]
        )
        self.fuse = ConvBlock(out_ch * len(in_chs), out_ch, dropout)

    def forward(self, features, size):
        projected = []
        for feat, proj in zip(features, self.proj):
            feat = proj(feat)
            if feat.shape[-2:] != size:
                feat = F.interpolate(feat, size=size, mode="bilinear", align_corners=False)
            projected.append(feat)
        return self.fuse(torch.cat(projected, dim=1))


class MUNet(nn.Module):
    """
    MU-Net-style baseline with U-Net encoder-decoder, skip connections, and
    multi-level feature fusion at each decoder stage.
    """

    def __init__(self, in_ch=1, num_classes=3, base=32, dropout=0.05):
        super().__init__()
        b = base

        self.enc1 = ConvBlock(in_ch, b, dropout)
        self.enc2 = ConvBlock(b, b * 2, dropout)
        self.enc3 = ConvBlock(b * 2, b * 4, dropout)
        self.enc4 = ConvBlock(b * 4, b * 8, dropout)
        self.center = ConvBlock(b * 8, b * 16, dropout)

        self.dec4 = MultiScaleFusion([b * 16, b * 8, b * 4, b * 2], b * 8, dropout)
        self.dec3 = MultiScaleFusion([b * 8, b * 4, b * 2, b], b * 4, dropout)
        self.dec2 = MultiScaleFusion([b * 4, b * 2, b], b * 2, dropout)
        self.dec1 = MultiScaleFusion([b * 2, b], b, dropout)

        self.out = nn.Conv2d(b, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))
        c = self.center(F.max_pool2d(e4, 2))

        d4 = self.dec4([c, e4, e3, e2], size=e4.shape[-2:])
        d3 = self.dec3([d4, e3, e2, e1], size=e3.shape[-2:])
        d2 = self.dec2([d3, e2, e1], size=e2.shape[-2:])
        d1 = self.dec1([d2, e1], size=e1.shape[-2:])

        return self.out(d1)


def weighted_dice_loss(logits, target):
    probs = torch.softmax(logits, dim=1)
    onehot = F.one_hot(target, num_classes=3).permute(0, 3, 1, 2).float()

    dims = (0, 2, 3)
    inter = torch.sum(probs * onehot, dims)
    denom = torch.sum(probs + onehot, dims)
    dice = (2.0 * inter + 1e-6) / (denom + 1e-6)

    # class order: 0 non-eddy, 1 anti-cyclonic, 2 cyclonic
    weights = torch.tensor([0.03, 0.35, 0.62], device=logits.device)
    return 1.0 - torch.sum(weights * dice) / torch.sum(weights)


def combined_loss(logits, target, ce_loss):
    return 0.5 * ce_loss(logits, target) + 0.5 * weighted_dice_loss(logits, target)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    tp = torch.zeros(3, dtype=torch.float64)
    pred_sum = torch.zeros(3, dtype=torch.float64)
    true_sum = torch.zeros(3, dtype=torch.float64)

    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        pred = torch.argmax(logits, dim=1)

        correct += (pred == y).sum().item()
        total += y.numel()

        pred_cpu = pred.cpu()
        y_cpu = y.cpu()

        for c in range(3):
            pc = pred_cpu == c
            yc = y_cpu == c
            tp[c] += torch.logical_and(pc, yc).sum().item()
            pred_sum[c] += pc.sum().item()
            true_sum[c] += yc.sum().item()

    dice = (2 * tp / (pred_sum + true_sum + 1e-12)).numpy()
    iou = (tp / (pred_sum + true_sum - tp + 1e-12)).numpy()
    acc = correct / total

    return {
        "non_eddy_dice": float(dice[0]),
        "anti_dice": float(dice[1]),
        "cycl_dice": float(dice[2]),
        "mean_dice": float(dice.mean()),
        "non_eddy_iou": float(iou[0]),
        "anti_iou": float(iou[1]),
        "cycl_iou": float(iou[2]),
        "mean_iou": float(iou.mean()),
        "global_acc": float(acc),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data/SCSE_clean")
    parser.add_argument("--out_dir", type=str, default="runs/MUNet/scse_fullmap_base32_e80")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--base", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_x = data_root / "source_npy" / "filtered_SSH_train_data.npy"
    train_y = data_root / "source_npy" / "train_groundtruth_Segmentation.npy"
    val_x = data_root / "source_npy" / "filtered_SSH_vali_data.npy"
    val_y = data_root / "source_npy" / "vali_groundtruth_Segmentation.npy"

    train_ds = SCSEDataset(train_x, train_y)
    val_ds = SCSEDataset(val_x, val_y)

    expected_train_shape = (4750, 184, 302)
    expected_val_shape = (730, 184, 302)
    assert train_ds.x.shape == expected_train_shape, (train_ds.x.shape, expected_train_shape)
    assert train_ds.y.shape == expected_train_shape, (train_ds.y.shape, expected_train_shape)
    assert val_ds.x.shape == expected_val_shape, (val_ds.x.shape, expected_val_shape)
    assert val_ds.y.shape == expected_val_shape, (val_ds.y.shape, expected_val_shape)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    print("train:", train_ds.x.shape, train_ds.y.shape)
    print("val:", val_ds.x.shape, val_ds.y.shape)

    model = MUNet(base=args.base, dropout=args.dropout).to(device)

    # inverse-frequency CE weights from your SCSE train counts, normalized
    ce_weights = torch.tensor([0.46, 2.29, 2.45], dtype=torch.float32, device=device)
    ce_loss = nn.CrossEntropyLoss(weight=ce_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_mean_dice = -1.0
    metrics_path = out_dir / "metrics.csv"

    fieldnames = [
        "epoch",
        "train_loss",
        "non_eddy_dice",
        "anti_dice",
        "cycl_dice",
        "mean_dice",
        "non_eddy_iou",
        "anti_iou",
        "cycl_iou",
        "mean_iou",
        "global_acc",
        "lr",
    ]

    with open(metrics_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits = model(x)
                loss = combined_loss(logits, y, ce_loss)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            losses.append(loss.item())

        scheduler.step()

        val_metrics = evaluate(model, val_loader, device)
        train_loss = float(np.mean(losses))
        lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            **val_metrics,
            "lr": lr,
        }

        print(
            f"Epoch {epoch:03d} | "
            f"loss={train_loss:.5f} | "
            f"anti={val_metrics['anti_dice']:.4f} | "
            f"cycl={val_metrics['cycl_dice']:.4f} | "
            f"non={val_metrics['non_eddy_dice']:.4f} | "
            f"mean={val_metrics['mean_dice']:.4f} | "
            f"miou={val_metrics['mean_iou']:.4f} | "
            f"acc={val_metrics['global_acc']:.4f}"
        )

        with open(metrics_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(row)

        if val_metrics["mean_dice"] > best_mean_dice:
            best_mean_dice = val_metrics["mean_dice"]
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "args": vars(args),
                    "metrics": val_metrics,
                },
                out_dir / "best_model.pt",
            )
            with open(out_dir / "best_metrics.json", "w") as f:
                json.dump({"epoch": epoch, **val_metrics}, f, indent=2)
            print("  saved best_model.pt")

    print("best mean dice:", best_mean_dice)
    print("outputs:", out_dir)


if __name__ == "__main__":
    main()
