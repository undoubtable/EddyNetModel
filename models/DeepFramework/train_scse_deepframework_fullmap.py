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
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.net(x)


class SepConvBN(nn.Module):
    """
    PyTorch equivalent of FrameWork.py SepConv_BN:
    ReLU -> depthwise conv -> BN -> pointwise conv -> BN, with optional
    depth_activation behavior for exit flow, ASPP, and decoder layers.
    """

    def __init__(
        self,
        in_ch,
        out_ch,
        stride=1,
        dilation=1,
        depth_activation=False,
        pre_activation=True,
    ):
        super().__init__()
        self.pre_activation = pre_activation
        self.depth_activation = depth_activation
        self.relu = nn.ReLU(inplace=False)
        self.depthwise = nn.Conv2d(
            in_ch,
            in_ch,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            groups=in_ch,
            bias=False,
        )
        self.depth_bn = nn.BatchNorm2d(in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.point_bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        if self.pre_activation:
            x = self.relu(x)
        x = self.depthwise(x)
        x = self.depth_bn(x)
        if self.depth_activation:
            x = self.relu(x)
        x = self.pointwise(x)
        x = self.point_bn(x)
        if self.depth_activation:
            x = self.relu(x)
        return x


class XceptionBlock(nn.Module):
    """
    Equivalent migration of FrameWork.py _xception_block.
    skip_connection_type is one of: "conv", "sum", "none".
    return_skip exposes the second separable-conv feature used by the decoder.
    """

    def __init__(
        self,
        in_ch,
        depth_list,
        skip_connection_type,
        stride,
        rate=1,
        depth_activation=False,
        return_skip=False,
    ):
        super().__init__()
        c1, c2, c3 = depth_list
        self.return_skip = return_skip
        self.skip_connection_type = skip_connection_type

        self.sep1 = SepConvBN(
            in_ch,
            c1,
            dilation=rate,
            depth_activation=depth_activation,
            pre_activation=not depth_activation,
        )
        self.sep2 = SepConvBN(
            c1,
            c2,
            dilation=rate,
            depth_activation=depth_activation,
            pre_activation=not depth_activation,
        )
        self.sep3 = SepConvBN(
            c2,
            c3,
            stride=stride,
            dilation=rate,
            depth_activation=depth_activation,
            pre_activation=not depth_activation,
        )

        if skip_connection_type == "conv":
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, c3, 1, stride=stride, bias=False),
                nn.BatchNorm2d(c3),
            )
        elif skip_connection_type == "sum":
            self.shortcut = nn.Identity()
        elif skip_connection_type == "none":
            self.shortcut = None
        else:
            raise ValueError(f"Unknown skip_connection_type: {skip_connection_type}")

    def forward(self, x):
        residual = self.sep1(x)
        residual = self.sep2(residual)
        skip = residual
        residual = self.sep3(residual)

        if self.skip_connection_type == "conv":
            out = residual + self.shortcut(x)
        elif self.skip_connection_type == "sum":
            out = residual + x
        else:
            out = residual

        if self.return_skip:
            return out, skip
        return out


class ASPP(nn.Module):
    """
    FrameWork.py Deeplabv3 ASPP: image pooling, 1x1 branch, and three atrous
    separable-conv branches with rates 6, 12, and 18.
    """

    def __init__(self, in_ch, out_ch, atrous_rates=(6, 12, 18), dropout=0.2):
        super().__init__()
        self.image_pooling = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBNReLU(in_ch, out_ch, k=1, padding=0),
        )
        self.aspp0 = ConvBNReLU(in_ch, out_ch, k=1, padding=0)
        self.aspp1 = SepConvBN(
            in_ch,
            out_ch,
            dilation=atrous_rates[0],
            depth_activation=True,
            pre_activation=False,
        )
        self.aspp2 = SepConvBN(
            in_ch,
            out_ch,
            dilation=atrous_rates[1],
            depth_activation=True,
            pre_activation=False,
        )
        self.aspp3 = SepConvBN(
            in_ch,
            out_ch,
            dilation=atrous_rates[2],
            depth_activation=True,
            pre_activation=False,
        )
        self.concat_projection = nn.Sequential(
            ConvBNReLU(out_ch * 5, out_ch, k=1, padding=0),
            nn.Dropout2d(dropout),
        )

    def forward(self, x):
        size = x.shape[-2:]
        b4 = self.image_pooling(x)
        b4 = F.interpolate(b4, size=size, mode="bilinear", align_corners=True)
        b0 = self.aspp0(x)
        b1 = self.aspp1(x)
        b2 = self.aspp2(x)
        b3 = self.aspp3(x)
        return self.concat_projection(torch.cat([b4, b0, b1, b2, b3], dim=1))


class DeepFramework(nn.Module):
    """
    PyTorch migration of the official EddyData FrameWork.py detection network.

    The official Keras model is Deeplabv3+ with Xception backbone: entry flow,
    16 middle-flow blocks, exit flow, ASPP, low-level feature projection,
    decoder separable convolutions, and final semantic logits.
    """

    def __init__(self, in_ch=1, num_classes=3, base=32, dropout=0.05):
        super().__init__()
        b = base

        c32 = b
        c64 = b * 2
        c128 = b * 4
        c256 = b * 8
        c728 = b * 16
        c1024 = b * 24
        c1536 = b * 32
        c2048 = b * 32
        aspp_ch = b * 8
        low_ch = max(b + b // 2, 8)

        # Entry flow conv stem: entry_flow_conv1_1 and entry_flow_conv1_2.
        self.entry_flow_conv1_1 = nn.Sequential(
            nn.Conv2d(in_ch, c32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c32),
            nn.ReLU(inplace=False),
        )
        self.entry_flow_conv1_2 = ConvBNReLU(c32, c64, k=3, padding=1)

        self.entry_flow_block1 = XceptionBlock(
            c64,
            [c128, c128, c128],
            skip_connection_type="conv",
            stride=2,
            depth_activation=False,
        )
        self.entry_flow_block2 = XceptionBlock(
            c128,
            [c256, c256, c256],
            skip_connection_type="conv",
            stride=2,
            depth_activation=False,
            return_skip=True,
        )
        self.entry_flow_block3 = XceptionBlock(
            c256,
            [c728, c728, c728],
            skip_connection_type="conv",
            stride=2,
            depth_activation=False,
        )

        # FrameWork.py uses 16 middle_flow_unit_* Xception sum blocks.
        self.middle_flow = nn.Sequential(
            *[
                XceptionBlock(
                    c728,
                    [c728, c728, c728],
                    skip_connection_type="sum",
                    stride=1,
                    rate=1,
                    depth_activation=False,
                )
                for _ in range(16)
            ]
        )

        self.exit_flow_block1 = XceptionBlock(
            c728,
            [c728, c1024, c1024],
            skip_connection_type="conv",
            stride=1,
            rate=1,
            depth_activation=False,
        )
        self.exit_flow_block2 = XceptionBlock(
            c1024,
            [c1536, c1536, c2048],
            skip_connection_type="none",
            stride=1,
            rate=2,
            depth_activation=True,
        )

        self.aspp = ASPP(c2048, aspp_ch, atrous_rates=(6, 12, 18), dropout=0.2)
        self.feature_projection0 = ConvBNReLU(c256, low_ch, k=1, padding=0)
        self.decoder_conv0 = SepConvBN(
            aspp_ch + low_ch,
            aspp_ch,
            depth_activation=True,
            pre_activation=False,
        )
        self.decoder_conv1 = SepConvBN(
            aspp_ch,
            aspp_ch,
            depth_activation=True,
            pre_activation=False,
        )
        self.decoder_dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.logits_semantic = nn.Conv2d(aspp_ch, num_classes, 1)

    def forward(self, x):
        input_size = x.shape[-2:]

        x = self.entry_flow_conv1_1(x)
        x = self.entry_flow_conv1_2(x)
        x = self.entry_flow_block1(x)
        x, skip1 = self.entry_flow_block2(x)
        x = self.entry_flow_block3(x)

        x = self.middle_flow(x)
        x = self.exit_flow_block1(x)
        x = self.exit_flow_block2(x)

        x = self.aspp(x)
        x = F.interpolate(x, size=skip1.shape[-2:], mode="bilinear", align_corners=True)
        skip1 = self.feature_projection0(skip1)
        x = torch.cat([x, skip1], dim=1)
        x = self.decoder_conv0(x)
        x = self.decoder_conv1(x)
        x = self.decoder_dropout(x)
        x = self.logits_semantic(x)
        return F.interpolate(x, size=input_size, mode="bilinear", align_corners=True)


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
    parser.add_argument("--out_dir", type=str, default="runs/DeepFramework/scse_fullmap_base32_e80")
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

    model = DeepFramework(base=args.base, dropout=args.dropout).to(device)

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
