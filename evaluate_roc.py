"""Evaluate image-level toy-car confidence ROC on the labeled validation set."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/home/intern/datasets/CV/data"))
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--weights", type=Path, default=ROOT / "runs/baseline10_gpu1/weights/best.pt")
    parser.add_argument("--class-id", type=int, default=1, help="toy_car is class 1")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="1")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-csv", type=Path, default=ROOT / "runs/roc/roc_curve.csv")
    parser.add_argument("--output-plot", type=Path, default=ROOT / "runs/roc/roc_curve.png")
    return parser.parse_args()


def label_has_class(label_path: Path, class_id: int) -> bool:
    if not label_path.exists():
        return False
    for line in label_path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if fields and int(fields[0]) == class_id:
            return True
    return False


def main() -> None:
    args = parse_args()
    image_dir = args.data_root / "images" / args.split
    label_dir = args.data_root / "labels" / args.split
    image_paths = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    if not image_paths:
        raise SystemExit(f"No images found: {image_dir}")
    if not args.weights.exists():
        raise SystemExit(f"Weights not found: {args.weights}")

    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise SystemExit("Ultralytics is required.") from error
    model = YOLO(str(args.weights))
    scores: list[float] = []
    labels: list[int] = []
    for start in range(0, len(image_paths), args.batch_size):
        batch = image_paths[start : start + args.batch_size]
        results = model.predict(
            source=[str(path) for path in batch],
            imgsz=args.imgsz,
            conf=0.001,
            device=args.device,
            batch=len(batch),
            save=False,
            verbose=False,
        )
        for image_path, result in zip(batch, results):
            score = 0.0
            if result.boxes is not None and len(result.boxes):
                classes = result.boxes.cls.detach().cpu().numpy().astype(int)
                confidences = result.boxes.conf.detach().cpu().numpy()
                class_scores = confidences[classes == args.class_id]
                if len(class_scores):
                    score = float(class_scores.max())
            scores.append(score)
            labels.append(int(label_has_class(label_dir / f"{image_path.stem}.txt", args.class_id)))
        print(f"processed {min(start + args.batch_size, len(image_paths))}/{len(image_paths)} images")

    scores_array = np.asarray(scores, dtype=np.float64)
    labels_array = np.asarray(labels, dtype=np.int32)
    positives = int(labels_array.sum())
    negatives = int(len(labels_array) - positives)
    if not positives or not negatives:
        raise SystemExit("ROC needs both positive and negative images in the selected split.")

    rows = []
    for threshold in np.linspace(0.0, 1.0, 1001):
        predicted = scores_array >= threshold
        tp = int(np.sum(predicted & (labels_array == 1)))
        fp = int(np.sum(predicted & (labels_array == 0)))
        fn = positives - tp
        tn = negatives - fp
        tpr = tp / positives
        fpr = fp / negatives
        rows.append({"threshold": float(threshold), "tpr": tpr, "fpr": fpr, "tp": tp, "fp": fp, "tn": tn, "fn": fn})
    curve = sorted(rows, key=lambda row: (row["fpr"], row["tpr"]))
    auc = float(np.trapezoid([row["tpr"] for row in curve], [row["fpr"] for row in curve]))
    best = max(rows, key=lambda row: row["tpr"] - row["fpr"])
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    try:
        import matplotlib.pyplot as plt
        args.output_plot.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(6, 6))
        plt.plot([row["fpr"] for row in curve], [row["tpr"] for row in curve], label=f"toy_car AUC={auc:.4f}")
        plt.plot([0, 1], [0, 1], "--", color="gray")
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.title("Toy-car image-level ROC")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.output_plot, dpi=160)
        plt.close()
    except ImportError:
        print("matplotlib not installed; CSV was still written.")

    print(f"images={len(image_paths)} positives={positives} negatives={negatives}")
    print(f"AUC={auc:.6f}")
    print(f"best_youden_threshold={best['threshold']:.3f} TPR={best['tpr']:.4f} FPR={best['fpr']:.4f}")
    print(f"Saved ROC CSV: {args.output_csv}")
    if args.output_plot.exists():
        print(f"Saved ROC plot: {args.output_plot}")


if __name__ == "__main__":
    main()
