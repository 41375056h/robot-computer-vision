"""Crop a small-object video at source resolution before YOLO inference.

The pipeline never downsizes the complete source frame before the initial
search. It uses overlapping source-resolution tiles to bootstrap/recover an
ROI, then runs the detector on that ROI until a boundary or recovery trigger
requires another tile search.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
CLASS_NAMES = {0: "red_cone", 1: "toy_car"}
TARGET_CLASSES = frozenset(CLASS_NAMES)
# OpenCV uses BGR colors: red cone -> red, toy car -> blue.
BOX_COLORS = {0: (0, 0, 255), 1: (255, 0, 0)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "IMG_5985.MOV")
    parser.add_argument("--weights", type=Path, default=ROOT / "runs/baseline10_gpu1/weights/best.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/video_crop/cropped.mp4")
    parser.add_argument("--metadata", type=Path, default=ROOT / "runs/video_crop/frames.jsonl")
    parser.add_argument("--debug-video", type=Path, default=ROOT / "runs/video_crop/debug.mp4")
    parser.add_argument("--device", default="1")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--tile-size", type=int, default=640)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--margin", type=float, default=0.30)
    parser.add_argument("--min-margin-px", type=int, default=48)
    parser.add_argument("--boundary-px", type=int, default=64)
    parser.add_argument("--rescan-interval", type=int, default=10)
    parser.add_argument("--max-missed", type=int, default=8)
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--max-frames", type=int, default=None)
    return parser.parse_args()


def clamp_box(box: tuple[float, float, float, float], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    x1 = max(0, min(width - 1, int(round(x1))))
    y1 = max(0, min(height - 1, int(round(y1))))
    x2 = max(x1 + 1, min(width, int(round(x2))))
    y2 = max(y1 + 1, min(height, int(round(y2))))
    return x1, y1, x2, y2


def tile_starts(length: int, tile_size: int, overlap: float) -> list[int]:
    if tile_size >= length:
        return [0]
    step = max(1, int(round(tile_size * (1.0 - overlap))))
    starts = list(range(0, length - tile_size + 1, step))
    final = length - tile_size
    if starts[-1] != final:
        starts.append(final)
    return starts


def make_tiles(frame: np.ndarray, tile_size: int, overlap: float) -> list[tuple[np.ndarray, int, int]]:
    height, width = frame.shape[:2]
    xs = tile_starts(width, tile_size, overlap)
    ys = tile_starts(height, tile_size, overlap)
    tiles = []
    for y in ys:
        for x in xs:
            tile = frame[y : min(y + tile_size, height), x : min(x + tile_size, width)]
            tiles.append((tile, x, y))
    return tiles


def result_detections(result: Any, offset_x: int = 0, offset_y: int = 0) -> list[dict[str, Any]]:
    detections: list[dict[str, Any]] = []
    if result.boxes is None or len(result.boxes) == 0:
        return detections
    boxes = result.boxes.xyxy.detach().cpu().numpy()
    classes = result.boxes.cls.detach().cpu().numpy().astype(int)
    confidences = result.boxes.conf.detach().cpu().numpy()
    for box, class_id, confidence in zip(boxes, classes, confidences):
        if class_id not in TARGET_CLASSES:
            continue
        x1, y1, x2, y2 = box.tolist()
        detections.append(
            {
                "class_id": int(class_id),
                "class_name": CLASS_NAMES[int(class_id)],
                "confidence": float(confidence),
                "xyxy": [x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y],
            }
        )
    return detections


def detect_tiles(model: Any, frame: np.ndarray, args: argparse.Namespace) -> list[dict[str, Any]]:
    tiles = make_tiles(frame, args.tile_size, args.overlap)
    results = model.predict(
        source=[tile[0] for tile in tiles],
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        batch=min(16, len(tiles)),
        save=False,
        verbose=False,
    )
    detections = []
    for result, (_, offset_x, offset_y) in zip(results, tiles):
        detections.extend(result_detections(result, offset_x, offset_y))
    return detections


def detect_crop(model: Any, frame: np.ndarray, roi: tuple[int, int, int, int], args: argparse.Namespace) -> list[dict[str, Any]]:
    x1, y1, x2, y2 = roi
    crop = frame[y1:y2, x1:x2]
    result = model.predict(
        source=crop,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        batch=1,
        save=False,
        verbose=False,
    )[0]
    return result_detections(result, x1, y1)


def expanded_union(
    detections: list[dict[str, Any]],
    frame_shape: tuple[int, int, int],
    margin_ratio: float,
    min_margin_px: int,
) -> tuple[int, int, int, int] | None:
    if not detections:
        return None
    width = max(float(d["xyxy"][2]) - float(d["xyxy"][0]) for d in detections)
    height = max(float(d["xyxy"][3]) - float(d["xyxy"][1]) for d in detections)
    x1 = min(float(d["xyxy"][0]) for d in detections)
    y1 = min(float(d["xyxy"][1]) for d in detections)
    x2 = max(float(d["xyxy"][2]) for d in detections)
    y2 = max(float(d["xyxy"][3]) for d in detections)
    margin_x = max(min_margin_px, width * margin_ratio)
    margin_y = max(min_margin_px, height * margin_ratio)
    return clamp_box((x1 - margin_x, y1 - margin_y, x2 + margin_x, y2 + margin_y), frame_shape[1], frame_shape[0])


def smooth_roi(
    previous: tuple[int, int, int, int] | None,
    candidate: tuple[int, int, int, int] | None,
    detections: list[dict[str, Any]],
    frame_shape: tuple[int, int, int],
    margin_ratio: float,
    min_margin_px: int,
) -> tuple[int, int, int, int] | None:
    if candidate is None:
        return previous
    if previous is None:
        return candidate
    alpha = 0.30
    blended = tuple((1 - alpha) * old + alpha * new for old, new in zip(previous, candidate))
    result = clamp_box(blended, frame_shape[1], frame_shape[0])
    required = expanded_union(detections, frame_shape, margin_ratio, min_margin_px)
    if required is None:
        return result
    return clamp_box(
        (
            min(result[0], required[0]),
            min(result[1], required[1]),
            max(result[2], required[2]),
            max(result[3], required[3]),
        ),
        frame_shape[1],
        frame_shape[0],
    )


def near_boundary(detections: list[dict[str, Any]], roi: tuple[int, int, int, int], boundary_px: int) -> bool:
    if not detections:
        return False
    x1, y1, x2, y2 = roi
    for detection in detections:
        dx1, dy1, dx2, dy2 = detection["xyxy"]
        if dx1 - x1 < boundary_px or dy1 - y1 < boundary_px or x2 - dx2 < boundary_px or y2 - dy2 < boundary_px:
            return True
    return False


def crop_for_output(
    frame: np.ndarray,
    roi: tuple[int, int, int, int],
    size: int,
    detections: list[dict[str, Any]] | None = None,
) -> np.ndarray:
    x1, y1, x2, y2 = roi
    crop = frame[y1:y2, x1:x2]
    height, width = crop.shape[:2]
    scale = min(size / width, size / height)
    resized = cv2.resize(crop, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    top = (size - resized.shape[0]) // 2
    left = (size - resized.shape[1]) // 2
    canvas[top : top + resized.shape[0], left : left + resized.shape[1]] = resized
    for detection in detections or []:
        box = detection.get("xyxy_original", detection.get("xyxy"))
        if box is None:
            continue
        dx1, dy1, dx2, dy2 = box
        px1 = int(round((dx1 - x1) * scale + left))
        py1 = int(round((dy1 - y1) * scale + top))
        px2 = int(round((dx2 - x1) * scale + left))
        py2 = int(round((dy2 - y1) * scale + top))
        color = BOX_COLORS.get(int(detection.get("class_id", -1)), (0, 220, 0))
        cv2.rectangle(canvas, (px1, py1), (px2, py2), color, 2)
        label = f"{detection.get('class_name', 'object')} {float(detection.get('confidence', 0.0)):.2f}"
        cv2.putText(canvas, label, (px1, max(18, py1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2, cv2.LINE_AA)
    return canvas


def draw_debug(frame: np.ndarray, roi: tuple[int, int, int, int], detections: list[dict[str, Any]], status: str, frame_index: int) -> np.ndarray:
    output = frame.copy()
    x1, y1, x2, y2 = roi
    cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 255), 4)
    for detection in detections:
        dx1, dy1, dx2, dy2 = [int(round(v)) for v in detection["xyxy"]]
        label = f"{detection['class_name']} {detection['confidence']:.2f}"
        color = BOX_COLORS.get(int(detection["class_id"]), (0, 180, 0))
        cv2.rectangle(output, (dx1, dy1), (dx2, dy2), color, 3)
        cv2.putText(output, label, (dx1, max(24, dy1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    cv2.putText(output, f"frame={frame_index} status={status}", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, cv2.LINE_AA)
    return output


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise SystemExit(f"Input video not found: {args.input}")
    if not args.weights.exists():
        raise SystemExit(f"Weights not found: {args.weights}")

    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise SystemExit("Ultralytics is not installed in the selected environment.") from error

    capture = cv2.VideoCapture(str(args.input))
    if not capture.isOpened():
        raise SystemExit(f"Unable to open video: {args.input}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_limit = min(total_frames, args.max_frames) if args.max_frames else total_frames

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.debug_video.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    crop_writer = cv2.VideoWriter(str(args.output), fourcc, fps, (args.imgsz, args.imgsz))
    debug_writer = cv2.VideoWriter(str(args.debug_video), fourcc, fps, (width, height))
    if not crop_writer.isOpened() or not debug_writer.isOpened():
        raise SystemExit("Unable to create output video writers.")

    model = YOLO(str(args.weights))
    roi: tuple[int, int, int, int] | None = None
    missed = 0
    force_rescan = True
    started = time.perf_counter()

    with args.metadata.open("w", encoding="utf-8") as metadata_file:
        for frame_index in range(frame_limit):
            ok, frame = capture.read()
            if not ok:
                break

            do_rescan = force_rescan or roi is None or frame_index % args.rescan_interval == 0
            if do_rescan:
                detections = detect_tiles(model, frame, args)
                candidate = expanded_union(detections, frame.shape, args.margin, args.min_margin_px)
                if candidate is not None:
                    roi = smooth_roi(roi, candidate, detections, frame.shape, args.margin, args.min_margin_px)
                    missed = 0
                    status = "bootstrap" if frame_index == 0 else "periodic_rescan"
                elif roi is None:
                    roi = (0, 0, width, height)
                    status = "fallback_full_frame"
                    missed += 1
                else:
                    status = "rescan_keep_previous"
                    missed += 1
                force_rescan = False
            else:
                detections = detect_crop(model, frame, roi, args)
                candidate = expanded_union(detections, frame.shape, args.margin, args.min_margin_px)
                roi = smooth_roi(roi, candidate, detections, frame.shape, args.margin, args.min_margin_px)
                if detections:
                    missed = 0
                    status = "tracked_crop"
                else:
                    missed += 1
                    status = "tracked_crop_no_detection"
                force_rescan = near_boundary(detections, roi, args.boundary_px) or missed >= args.max_missed

            crop_writer.write(crop_for_output(frame, roi, args.imgsz, detections))
            debug_writer.write(draw_debug(frame, roi, detections, status, frame_index))
            record = {
                "frame": frame_index,
                "timestamp_sec": frame_index / fps,
                "source_size": [width, height],
                "crop_xyxy": list(roi),
                "status": status,
                "detections": [
                    {
                        "class_id": d["class_id"],
                        "class_name": d["class_name"],
                        "confidence": d["confidence"],
                        "xyxy_original": d["xyxy"],
                    }
                    for d in detections
                ],
            }
            metadata_file.write(json.dumps(jsonable(record), ensure_ascii=False) + "\n")

            if (frame_index + 1) % 100 == 0 or frame_index + 1 == frame_limit:
                elapsed = time.perf_counter() - started
                print(f"processed {frame_index + 1}/{frame_limit} frames; {elapsed:.1f}s elapsed; status={status}")

    capture.release()
    crop_writer.release()
    debug_writer.release()
    print(f"Saved crop video: {args.output}")
    print(f"Saved metadata: {args.metadata}")
    print(f"Saved debug video: {args.debug_video}")


if __name__ == "__main__":
    main()
