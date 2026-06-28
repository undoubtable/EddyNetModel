import argparse
import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


PAPER_MIOU = {
    "PSPNet": 84.31,
    "EddyNet": 86.16,
    "DeepFramework": 84.73,
    "AutoDetectionAttention": 88.16,
    "MUNet": 88.47,
    "DCNN": 88.67,
    "DUNet": 89.66,
}


def load_metrics(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {"model": row["model"], "rank": int(row["rank"]), "epoch": int(row["epoch"])}
            for key, value in row.items():
                if key not in {"rank", "model", "epoch"}:
                    parsed[key] = float(value)
            rows.append(parsed)
    rows.sort(key=lambda item: item["rank"])
    return rows


def set_style():
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.dpi": 400,
            "savefig.dpi": 400,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "xtick.major.size": 3.2,
            "ytick.major.size": 3.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig, png_path):
    png_path = Path(png_path)
    pdf_path = png_path.with_suffix(".pdf")
    fig.savefig(png_path, bbox_inches="tight", dpi=400)
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {png_path}")
    print(f"saved: {pdf_path}")


def style_axes(ax):
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.grid(axis="y", linestyle="--", linewidth=0.45, color="#c7c7c7", alpha=0.45)
    ax.set_axisbelow(True)


def rotate_xticklabels(ax):
    ax.tick_params(axis="x", rotation=35)
    for label in ax.get_xticklabels():
        label.set_ha("right")


def annotate_bars(ax, bars, fmt="{:.4f}", y_offset=0.003):
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + y_offset,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=0,
        )


def metric_ylim(values, pad_low=0.018, pad_high=0.018, upper=1.0):
    ymin = max(0.0, min(values) - pad_low)
    ymax = min(upper, max(values) + pad_high)
    return ymin, ymax


def save_metric_bar(rows, metric, title, ylabel, out_path):
    models = [row["model"] for row in rows]
    values = [row[metric] for row in rows]

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=400)
    bars = ax.bar(
        models,
        values,
        width=0.62,
        color="#5b7894",
        edgecolor="#2f3f4f",
        linewidth=0.55,
    )
    annotate_bars(ax, bars, fmt="{:.4f}", y_offset=0.0025)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel("Model")
    ax.set_ylabel(ylabel)
    ax.set_ylim(*metric_ylim(values))
    style_axes(ax)
    rotate_xticklabels(ax)
    save_figure(fig, out_path)


def save_class_dice(rows, out_path):
    models = [row["model"] for row in rows]
    x = np.arange(len(models))
    width = 0.22

    non_eddy = [row["non_eddy_dice"] for row in rows]
    anti = [row["anti_dice"] for row in rows]
    cycl = [row["cycl_dice"] for row in rows]

    fig, ax = plt.subplots(figsize=(7.8, 4.6), dpi=400)
    ax.bar(
        x - width,
        non_eddy,
        width,
        label="Non-eddy",
        color="#4f6f8f",
        edgecolor="#2f3f4f",
        linewidth=0.45,
    )
    ax.bar(
        x,
        anti,
        width,
        label="Anti-cyclonic",
        color="#b88a44",
        edgecolor="#5d4828",
        linewidth=0.45,
    )
    ax.bar(
        x + width,
        cycl,
        width,
        label="Cyclonic",
        color="#5f8a68",
        edgecolor="#344f3a",
        linewidth=0.45,
    )

    ax.set_title("Class-wise Dice on SCSE-Eddy", loc="left", fontweight="bold")
    ax.set_xlabel("Model")
    ax.set_ylabel("Dice")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=35, ha="right")
    all_values = non_eddy + anti + cycl
    ax.set_ylim(max(0.0, min(all_values) - 0.045), 1.0)
    style_axes(ax)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.20))
    save_figure(fig, out_path)


def save_paper_vs_ours(rows, out_path):
    row_by_model = {row["model"]: row for row in rows}
    models = [row["model"] for row in rows if row["model"] in PAPER_MIOU]
    x = np.arange(len(models))
    width = 0.34

    paper = [PAPER_MIOU[model] for model in models]
    ours = [row_by_model[model]["mean_iou"] * 100.0 for model in models]

    fig, ax = plt.subplots(figsize=(7.8, 4.6), dpi=400)
    bars1 = ax.bar(
        x - width / 2,
        paper,
        width,
        label="Paper-reported",
        color="#6f7f8f",
        edgecolor="#35424f",
        linewidth=0.45,
    )
    bars2 = ax.bar(
        x + width / 2,
        ours,
        width,
        label="Reproduced",
        color="#b7605c",
        edgecolor="#6b3431",
        linewidth=0.45,
    )

    annotate_bars(ax, bars1, fmt="{:.2f}", y_offset=0.20)
    annotate_bars(ax, bars2, fmt="{:.2f}", y_offset=0.20)

    ax.set_title("Paper-reported vs Reproduced Mean IoU", loc="left", fontweight="bold")
    ax.set_xlabel("Model")
    ax.set_ylabel("Mean IoU (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=35, ha="right")
    ax.set_ylim(min(min(paper), min(ours)) - 6, max(max(paper), max(ours)) + 3.5)
    style_axes(ax)
    ax.legend(frameon=False, loc="upper left", ncol=2)
    save_figure(fig, out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_csv", type=str, default="results/baseline_metrics.csv")
    parser.add_argument("--out_dir", type=str, default="Figures/03_metric_comparison")
    args = parser.parse_args()

    set_style()
    rows = load_metrics(Path(args.metrics_csv))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_metric_bar(
        rows,
        metric="mean_dice",
        title="Mean Dice on SCSE-Eddy",
        ylabel="Mean Dice",
        out_path=out_dir / "mean_dice_ranking.png",
    )
    save_metric_bar(
        rows,
        metric="mean_iou",
        title="Mean IoU on SCSE-Eddy",
        ylabel="Mean IoU",
        out_path=out_dir / "mean_iou_ranking.png",
    )
    save_class_dice(rows, out_dir / "class_dice_comparison.png")
    save_paper_vs_ours(rows, out_dir / "paper_vs_ours_miou.png")


if __name__ == "__main__":
    main()
