import argparse
import csv
import json
from pathlib import Path


MODEL_SPECS = [
    ("DUNet", "runs/DUNet/scse_fullmap_base32_e80/best_metrics.json"),
    ("EddyNet", "runs/EddyNet/scse_fullmap_base32_e80/best_metrics.json"),
    ("PSPNet", "runs/PSPNet/scse_fullmap_base32_e80/best_metrics.json"),
    ("MUNet", "runs/MUNet/scse_fullmap_base32_e80/best_metrics.json"),
    ("DCNN", "runs/DCNN/scse_fullmap_base32_e80/best_metrics.json"),
    (
        "AutoDetectionAttention",
        "runs/AutoDetectionAttention/scse_fullmap_base32_e80/best_metrics.json",
    ),
    (
        "DeepFramework",
        "runs/DeepFramework/scse_fullmap_base32_e80/best_metrics.json",
    ),
]

FIELDS = [
    "rank",
    "model",
    "epoch",
    "non_eddy_dice",
    "anti_dice",
    "cycl_dice",
    "mean_dice",
    "non_eddy_iou",
    "anti_iou",
    "cycl_iou",
    "mean_iou",
    "global_acc",
]

METRIC_FIELDS = [
    "non_eddy_dice",
    "anti_dice",
    "cycl_dice",
    "mean_dice",
    "non_eddy_iou",
    "anti_iou",
    "cycl_iou",
    "mean_iou",
    "global_acc",
]


def load_rows(repo_root):
    rows = []
    for model, rel_path in MODEL_SPECS:
        path = repo_root / rel_path
        if not path.exists():
            raise FileNotFoundError(f"Missing metrics file for {model}: {path}")

        with open(path) as f:
            metrics = json.load(f)

        row = {
            "model": model,
            "epoch": int(metrics["epoch"]),
        }
        for field in METRIC_FIELDS:
            row[field] = float(metrics[field])
        rows.append(row)

    rows.sort(key=lambda item: item["mean_dice"], reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def write_csv(rows, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in FIELDS})


def fmt_value(field, value):
    if field in {"rank", "epoch"}:
        return str(value)
    if field == "model":
        return str(value)
    return f"{float(value):.4f}"


def write_markdown(rows, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("| " + " | ".join(FIELDS) + " |")
    lines.append("| " + " | ".join(["---"] * len(FIELDS)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(fmt_value(field, row[field]) for field in FIELDS) + " |")
    out_path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_root", type=str, default=".")
    parser.add_argument("--out_dir", type=str, default="results")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir

    rows = load_rows(repo_root)
    write_csv(rows, out_dir / "baseline_metrics.csv")
    write_markdown(rows, out_dir / "baseline_metrics.md")

    print(f"saved: {out_dir / 'baseline_metrics.csv'}")
    print(f"saved: {out_dir / 'baseline_metrics.md'}")


if __name__ == "__main__":
    main()
