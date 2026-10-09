"""Evaluate Homography reprojection error and optional static-car ground truth."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, default=ROOT / "calibration.json")
    parser.add_argument("--tracking-jsonl", type=Path, default=None)
    parser.add_argument("--ground-truth", type=Path, default=None,
                        help="CSV with frame_index,car_id,x_mm,y_mm and optional theta_deg")
    return parser.parse_args()


def angle_error(now: float, truth: float) -> float:
    return abs((now - truth + 180.0) % 360.0 - 180.0)


def evaluate_calibration(data: dict[str, Any]) -> None:
    image = np.asarray(data["image_points_px"], dtype=np.float32)
    world = np.asarray(data["world_points_xy"], dtype=np.float32)
    homography = np.asarray(data["homography_image_to_world"], dtype=np.float64)
    projected = cv2.perspectiveTransform(image.reshape(-1, 1, 2), homography).reshape(-1, 2)
    errors = np.linalg.norm(projected - world, axis=1)
    inliers = np.asarray(data.get("homography_inliers", [True] * len(errors)), dtype=bool)
    units = data.get("world_units", "world units")
    print(f"control_points={len(errors)} inliers={int(inliers.sum())}/{len(errors)} units={units}")
    print(f"all_points mean={errors.mean():.4f} max={errors.max():.4f} {units}")
    if inliers.any():
        print(f"inliers mean={errors[inliers].mean():.4f} max={errors[inliers].max():.4f} {units}")
    for index, error in enumerate(errors, start=1):
        print(f"point {index}: {'inlier' if inliers[index - 1] else 'outlier'} error={error:.4f} {units}")


def load_ground_truth(path: Path) -> dict[tuple[int, int], dict[str, float]]:
    result: dict[tuple[int, int], dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            frame = int(row["frame_index"])
            car_id = int(row["car_id"])
            result[(frame, car_id)] = {
                key: float(row[key])
                for key in ("x_mm", "y_mm", "theta_deg")
                if key in row and row[key] not in ("", None)
            }
    return result


def evaluate_tracking(tracking_path: Path, ground_truth_path: Path) -> None:
    truth = load_ground_truth(ground_truth_path)
    position_errors = []
    orientation_errors = []
    with tracking_path.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            key = (int(row["frame_index"]), int(row["car_id"]))
            reference = truth.get(key)
            if reference is None or not row.get("detected", False):
                continue
            position_errors.append(math.hypot(row["world_xyz"][0] - reference["x_mm"], row["world_xyz"][1] - reference["y_mm"]))
            if "theta_deg" in reference and row["theta"] > -999:
                orientation_errors.append(angle_error(row["theta"], reference["theta_deg"]))
    if not position_errors:
        print("No matching detected frames found in ground truth.")
        return
    print(f"static-car position samples={len(position_errors)}")
    print(f"position mean={np.mean(position_errors):.4f} mm max={np.max(position_errors):.4f} mm")
    if orientation_errors:
        print(f"orientation mean={np.mean(orientation_errors):.4f} deg max={np.max(orientation_errors):.4f} deg")


def main() -> None:
    args = parse_args()
    data = json.loads(args.calibration.read_text(encoding="utf-8"))
    evaluate_calibration(data)
    if args.tracking_jsonl and args.ground_truth:
        evaluate_tracking(args.tracking_jsonl, args.ground_truth)
    elif args.tracking_jsonl or args.ground_truth:
        raise SystemExit("Use --tracking-jsonl and --ground-truth together.")


if __name__ == "__main__":
    main()
