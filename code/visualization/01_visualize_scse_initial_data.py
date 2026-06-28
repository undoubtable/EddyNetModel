import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def set_paper_style():
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
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "axes.linewidth": 0.9,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def lon_formatter(x, pos):
    return f"{int(round(x))}°E"


def lat_formatter(y, pos):
    return f"{int(round(y))}°N"


def load_scse_sample(data_root, split, index):
    source_dir = Path(data_root) / "source_npy"
    if split == "train":
        ssh_path = source_dir / "filtered_SSH_train_data.npy"
        gt_path = source_dir / "train_groundtruth_Segmentation.npy"
        expected_shape = (4750, 184, 302)
    elif split == "val":
        ssh_path = source_dir / "filtered_SSH_vali_data.npy"
        gt_path = source_dir / "vali_groundtruth_Segmentation.npy"
        expected_shape = (730, 184, 302)
    else:
        raise ValueError(f"Unsupported split: {split}")

    ssh_all = np.load(ssh_path, mmap_mode="r")
    gt_all = np.load(gt_path, mmap_mode="r")
    assert ssh_all.shape == expected_shape, (ssh_all.shape, expected_shape)
    assert gt_all.shape == expected_shape, (gt_all.shape, expected_shape)

    if index < 0 or index >= ssh_all.shape[0]:
        raise IndexError(f"index={index} is out of range for split={split} with n={ssh_all.shape[0]}")

    ssh = ssh_all[index].astype(np.float32)
    gt = gt_all[index].astype(np.int64)
    return ssh, gt


def plot_scse_initial_data(
    ssh,
    gt,
    out_file,
    sample_idx,
    split,
    lon_min=105.5,
    lon_max=150.0,
    lat_min=4.0,
    lat_max=30.0,
):
    set_paper_style()

    ssh_cmap = "turbo"
    gt_colors = [
        "#24105a",  # Non-eddy
        "#ffd43b",  # Anti-cyclonic eddy
        "#1f77b4",  # Cyclonic eddy
    ]
    gt_cmap = ListedColormap(gt_colors)
    gt_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], gt_cmap.N)
    extent = [lon_min, lon_max, lat_min, lat_max]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13.2, 5.2),
        dpi=300,
        constrained_layout=False,
    )
    ax1, ax2 = axes

    im1 = ax1.imshow(
        ssh,
        origin="lower",
        extent=extent,
        cmap=ssh_cmap,
        aspect="auto",
        interpolation="nearest",
    )
    ax1.set_title("(a) SSH intensity", loc="left", fontweight="bold")

    cax1 = inset_axes(
        ax1,
        width="82%",
        height="6%",
        loc="lower center",
        bbox_to_anchor=(0.0, -0.24, 1.0, 1.0),
        bbox_transform=ax1.transAxes,
        borderpad=0,
    )
    cbar1 = fig.colorbar(im1, cax=cax1, orientation="horizontal")
    cbar1.set_label("SSH intensity", labelpad=4)
    cbar1.ax.tick_params(labelsize=8)

    ax2.imshow(
        gt,
        origin="lower",
        extent=extent,
        cmap=gt_cmap,
        norm=gt_norm,
        aspect="auto",
        interpolation="nearest",
    )
    ax2.set_title("(b) Ground truth", loc="left", fontweight="bold")

    sm = ScalarMappable(cmap=gt_cmap, norm=gt_norm)
    sm.set_array([])
    cax2 = inset_axes(
        ax2,
        width="82%",
        height="6%",
        loc="lower center",
        bbox_to_anchor=(0.0, -0.24, 1.0, 1.0),
        bbox_transform=ax2.transAxes,
        borderpad=0,
    )
    cbar2 = fig.colorbar(
        sm,
        cax=cax2,
        orientation="horizontal",
        boundaries=[-0.5, 0.5, 1.5, 2.5],
        ticks=[0, 1, 2],
        spacing="proportional",
    )
    cbar2.ax.set_xticklabels(
        [
            "Non-eddy",
            "Anti-cyclonic eddy",
            "Cyclonic eddy",
        ]
    )
    cbar2.ax.tick_params(labelsize=8, length=0, pad=2)
    cbar2.set_label("Ground-truth class", labelpad=4)

    xticks = [110, 115, 120, 125, 130, 135, 140, 145, 150]
    yticks = [5, 10, 15, 20, 25, 30]

    for ax in [ax1, ax2]:
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.xaxis.set_major_formatter(FuncFormatter(lon_formatter))
        ax.yaxis.set_major_formatter(FuncFormatter(lat_formatter))
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude", labelpad=6)
        ax.grid(True, linestyle="--", linewidth=0.45, color="#c7c7c7", alpha=0.55)
        ax.set_facecolor("white")

        for spine in ax.spines.values():
            spine.set_linewidth(0.9)
            spine.set_color("black")

    ax2.set_ylabel("")
    fig.suptitle(
        f"SCSE-Eddy visualization, sample {sample_idx}",
        y=0.98,
        fontsize=13,
    )
    fig.subplots_adjust(
        left=0.055,
        right=0.985,
        bottom=0.22,
        top=0.84,
        wspace=0.10,
    )

    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"saved: {out_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data/SCSE_clean")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"])
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="Figures/01_dataset_visualization")
    args = parser.parse_args()

    ssh, gt = load_scse_sample(args.data_root, args.split, args.index)
    out_file = Path(args.out_dir) / f"scse_initial_data_idx{args.index:04d}.png"
    plot_scse_initial_data(
        ssh=ssh,
        gt=gt,
        out_file=out_file,
        sample_idx=args.index,
        split=args.split,
    )


if __name__ == "__main__":
    main()
