import argparse
import importlib.util
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.ticker import FuncFormatter


MODEL_SPECS = {
    "DUNet": {
        "folder": "DUNet",
        "script": "train_scse_dunet_fullmap.py",
        "class_name": "DUNet",
    },
    "EddyNet": {
        "folder": "EddyNet",
        "script": "train_scse_eddynet_fullmap.py",
        "class_name": "EddyNet",
    },
    "PSPNet": {
        "folder": "PSPNet",
        "script": "train_scse_pspnet_fullmap.py",
        "class_name": "PSPNet",
    },
    "MU-Net": {
        "folder": "MUNet",
        "script": "train_scse_munet_fullmap.py",
        "class_name": "MUNet",
    },
    "DCNN": {
        "folder": "DCNN",
        "script": "train_scse_dcnn_fullmap.py",
        "class_name": "DCNN",
    },
    "AutoDetectionAttention": {
        "folder": "AutoDetectionAttention",
        "script": "train_scse_auto_detection_attention_fullmap.py",
        "class_name": "AutoDetectionAttentionNet",
    },
    "DeepFramework": {
        "folder": "DeepFramework",
        "script": "train_scse_deepframework_fullmap.py",
        "class_name": "DeepFramework",
    },
}

PANEL_ORDER = [
    "Ground truth",
    "DUNet",
    "EddyNet",
    "PSPNet",
    "MU-Net",
    "DCNN",
    "AutoDetectionAttention",
    "DeepFramework",
]


def set_paper_style():
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.dpi": 300,
            "savefig.dpi": 400,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def lon_formatter(x, pos):
    return f"{int(round(x))}°E"


def lat_formatter(y, pos):
    return f"{int(round(y))}°N"


def load_module_from_path(path):
    spec = importlib.util.spec_from_file_location(path.stem, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_logits(output):
    # DUNet returns (logits_final, logits1, logits2).
    # Other models return logits directly.
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def get_split_paths(data_root, split):
    source_dir = data_root / "source_npy"

    if split == "train":
        x_path = source_dir / "filtered_SSH_train_data.npy"
        y_path = source_dir / "train_groundtruth_Segmentation.npy"
    elif split == "val":
        x_path = source_dir / "filtered_SSH_vali_data.npy"
        y_path = source_dir / "vali_groundtruth_Segmentation.npy"
    else:
        raise ValueError(f"Unsupported split: {split}")

    return x_path, y_path


def load_one_sample(data_root, split, index):
    x_path, y_path = get_split_paths(data_root, split)

    if not x_path.exists():
        raise FileNotFoundError(f"Missing SSH file: {x_path}")
    if not y_path.exists():
        raise FileNotFoundError(f"Missing GT file: {y_path}")

    x_all = np.load(x_path, mmap_mode="r")
    y_all = np.load(y_path, mmap_mode="r")

    if index < 0 or index >= x_all.shape[0]:
        raise IndexError(
            f"index={index} is out of range for split={split}. "
            f"Available range: 0..{x_all.shape[0] - 1}"
        )

    x = x_all[index].astype("float32") / 255.0
    y = y_all[index].astype("int64")

    return x, y


def load_model(repo_root, model_name, run_tag, device):
    spec = MODEL_SPECS[model_name]

    model_script = repo_root / "models" / spec["folder"] / spec["script"]
    ckpt_path = repo_root / "runs" / spec["folder"] / run_tag / "best_model.pt"

    if not model_script.exists():
        raise FileNotFoundError(f"Missing model script: {model_script}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    module = load_module_from_path(model_script)
    model_cls = getattr(module, spec["class_name"])

    ckpt = torch.load(ckpt_path, map_location="cpu")

    ckpt_args = ckpt.get("args", {})
    base = int(ckpt_args.get("base", 32))
    dropout = float(ckpt_args.get("dropout", 0.05))

    model = model_cls(base=base, dropout=dropout)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    return model


@torch.no_grad()
def predict_one_model(model, x_np, device):
    x = torch.from_numpy(x_np)[None, None, :, :].to(device)
    output = model(x)
    logits = get_logits(output)
    pred = torch.argmax(logits, dim=1)[0]
    return pred.cpu().numpy().astype(np.uint8)


def style_geo_axis(
    ax,
    lon_min,
    lon_max,
    lat_min,
    lat_max,
    show_ylabel=False,
):
    xticks = [110, 115, 120, 125, 130, 135, 140, 145, 150]
    yticks = [5, 10, 15, 20, 25, 30]

    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)

    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

    ax.xaxis.set_major_formatter(FuncFormatter(lon_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(lat_formatter))

    ax.set_xlabel("Longitude")

    if show_ylabel:
        ax.set_ylabel("Latitude", labelpad=6)
    else:
        ax.set_ylabel("")

    ax.grid(
        True,
        linestyle="--",
        linewidth=0.42,
        color="#c7c7c7",
        alpha=0.50,
    )

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("black")


def plot_comparison(
    gt,
    predictions,
    out_png,
    split,
    index,
    lon_min,
    lon_max,
    lat_min,
    lat_max,
):
    set_paper_style()

    class_colors = [
        "#24105a",  # 0: Non-eddy
        "#ffd43b",  # 1: Anti-cyclonic eddy
        "#1f77b4",  # 2: Cyclonic eddy
    ]

    cmap = ListedColormap(class_colors)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    extent = [lon_min, lon_max, lat_min, lat_max]

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(19.5, 7.6),
        dpi=300,
        constrained_layout=False,
    )

    panel_data = {"Ground truth": gt}
    panel_data.update(predictions)

    panel_labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)"]

    for i, name in enumerate(PANEL_ORDER):
        ax = axes.flat[i]
        mask = panel_data[name]

        ax.imshow(
            mask,
            origin="lower",
            extent=extent,
            cmap=cmap,
            norm=norm,
            aspect="auto",
            interpolation="nearest",
        )

        ax.set_title(
            f"{panel_labels[i]} {name}",
            loc="left",
            fontweight="bold",
        )

        show_ylabel = i in [0, 4]
        style_geo_axis(
            ax,
            lon_min=lon_min,
            lon_max=lon_max,
            lat_min=lat_min,
            lat_max=lat_max,
            show_ylabel=show_ylabel,
        )

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar_ax = fig.add_axes([0.25, 0.055, 0.50, 0.035])
    cbar = fig.colorbar(
        sm,
        cax=cbar_ax,
        orientation="horizontal",
        boundaries=[-0.5, 0.5, 1.5, 2.5],
        ticks=[0, 1, 2],
        spacing="proportional",
        extend="both",
        extendfrac=0.08,
    )

    cbar.ax.set_xticklabels(
        [
            "Non-eddy",
            "Anti-cyclonic eddy",
            "Cyclonic eddy",
        ]
    )
    cbar.ax.tick_params(labelsize=9, length=0, pad=3)
    cbar.set_label("Class", labelpad=7, fontsize=11)

    fig.suptitle(
        f"SCSE-Eddy qualitative comparison, {split} sample {index}",
        y=0.965,
        fontsize=14,
    )

    fig.subplots_adjust(
        left=0.055,
        right=0.985,
        bottom=0.17,
        top=0.90,
        wspace=0.22,
        hspace=0.34,
    )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_pdf = out_png.with_suffix(".pdf")

    fig.savefig(out_png, bbox_inches="tight", dpi=400)
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"saved: {out_png}")
    print(f"saved: {out_pdf}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--repo_root", type=str, default=".")
    parser.add_argument("--data_root", type=str, default="data/SCSE_clean")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--index", type=int, default=0)

    parser.add_argument(
        "--run_tag",
        type=str,
        default="scse_fullmap_base32_e80",
        help="Subfolder name under runs/<ModelName>/.",
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default="Figures/04_qualitative_comparison",
    )

    parser.add_argument("--lon_min", type=float, default=105.5)
    parser.add_argument("--lon_max", type=float, default=150.0)
    parser.add_argument("--lat_min", type=float, default=4.0)
    parser.add_argument("--lat_max", type=float, default=30.0)

    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    data_root = repo_root / args.data_root
    out_dir = repo_root / args.out_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    x_np, gt = load_one_sample(
        data_root=data_root,
        split=args.split,
        index=args.index,
    )

    predictions = {}

    for model_name in MODEL_SPECS.keys():
        print(f"predicting: {model_name}")
        model = load_model(
            repo_root=repo_root,
            model_name=model_name,
            run_tag=args.run_tag,
            device=device,
        )
        pred = predict_one_model(model, x_np, device)
        predictions[model_name] = pred

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out_png = out_dir / f"scse_qualitative_{args.split}_idx{args.index:04d}.png"

    plot_comparison(
        gt=gt,
        predictions=predictions,
        out_png=out_png,
        split=args.split,
        index=args.index,
        lon_min=args.lon_min,
        lon_max=args.lon_max,
        lat_min=args.lat_min,
        lat_max=args.lat_max,
    )


if __name__ == "__main__":
    main()