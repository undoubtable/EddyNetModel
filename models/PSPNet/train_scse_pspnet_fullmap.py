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
    def __init__(self, in_ch, out_ch, k=3, stride=1, padding=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                in_ch,
                out_ch,
                k,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, dropout=0.05):
        super().__init__()
        self.body = nn.Sequential(
            ConvBNReLU(in_ch, out_ch, stride=stride),
            ConvBNReLU(out_ch, out_ch),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )
        if in_ch != out_ch or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.body(x) + self.shortcut(x))


class PyramidPoolingModule(nn.Module):
    def __init__(self, in_ch, pool_ch, scales=(1, 2, 3, 6)):
        super().__init__()
        self.paths = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(scale),
                    ConvBNReLU(in_ch, pool_ch, k=1, padding=0),
                )
                for scale in scales
            ]
        )

    def forward(self, x):
        size = x.shape[-2:]
        feats = [x]
        for path in self.paths:
            pooled = path(x)
            pooled = F.interpolate(pooled, size=size, mode="bilinear", align_corners=False)
            feats.append(pooled)
        return torch.cat(feats, dim=1)


class PSPNet(nn.Module):
    """
    PSPNet baseline with a compact encoder, pyramid pooling module, and decoder.
    Pyramid pooling uses scales 1, 2, 3, and 6, then logits are upsampled to the
    original SCSE full-map size.
    """

    def __init__(self, in_ch=1, num_classes=3, base=32, dropout=0.05):
        super().__init__()
        b = base

        self.enc1 = nn.Sequential(
            ConvBNReLU(in_ch, b),
            ConvBNReLU(b, b),
        )
        self.enc2 = ResidualBlock(b, b * 2, stride=2, dropout=dropout)
        self.enc3 = ResidualBlock(b * 2, b * 4, stride=2, dropout=dropout)
        self.enc4 = ResidualBlock(b * 4, b * 8, stride=2, dropout=dropout)
        self.context = nn.Sequential(
            ResidualBlock(b * 8, b * 8, dropout=dropout),
            ResidualBlock(b * 8, b * 8, dropout=dropout),
        )

        self.ppm = PyramidPoolingModule(b * 8, b * 2, scales=(1, 2, 3, 6))
        self.ppm_project = nn.Sequential(
            ConvBNReLU(b * 16, b * 4, k=1, padding=0),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )

        self.low_project = ConvBNReLU(b, b, k=1, padding=0)
        self.decoder = nn.Sequential(
            ConvBNReLU(b * 5, b * 2),
            ConvBNReLU(b * 2, b * 2),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )
        self.out = nn.Conv2d(b * 2, num_classes, 1)

    def forward(self, x):
        input_size = x.shape[-2:]

        low = self.enc1(x)
        x = self.enc2(low)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.context(x)
        x = self.ppm_project(self.ppm(x))

        x = F.interpolate(x, size=low.shape[-2:], mode="bilinear", align_corners=False)
        low = self.low_project(low)
        x = self.decoder(torch.cat([x, low], dim=1))
        x = self.out(x)
        return F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)


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
    parser.add_argument("--out_dir", type=str, default="runs/PSPNet/scse_fullmap_base32_e80")
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

    model = PSPNet(base=args.base, dropout=args.dropout).to(device)

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
