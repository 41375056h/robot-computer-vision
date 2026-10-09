"""Two-car tracking, world-coordinate velocity, heading, and UDP output.

Pipeline:
    source-resolution tiles -> YOLO car detections -> lightweight ID tracker
    -> background-subtraction mask -> heading/omega/arrow/UDP

The heading branch only reads the current car detection and mask. It never
feeds heading or arrows back into the tracker.
"""

from __future__ import annotations

import argparse
import json
import math
import socket
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from video_crop import clamp_box, detect_tiles, result_detections


ROOT = Path(__file__).resolve().parent
CAR_CLASS_ID = 1
CAR_CLASS_NAME = "toy_car"
INVALID = -1000.0
COLORS = {1: (0, 255, 0), 2: (255, 0, 255)}  # car 1 green, car 2 magenta; BGR


@dataclass
class CarState:
    car_id: int
    box: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    confidence: float = 0.0
    velocity_px: tuple[float, float] = (0.0, 0.0)
    last_detection_box: list[float] | None = None
    last_detection_timestamp: float | None = None
    missed: int = 0
    active: bool = True
    prev_theta: float | None = None
    prev_timestamp: float | None = None
    prev_world: tuple[float, float] | None = None
    history: list[tuple[float, float]] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=ROOT / "IMG_5985.MOV")
    parser.add_argument("--weights", type=Path, default=ROOT / "runs/baseline10_gpu1/weights/best.pt")
    parser.add_argument("--calibration", type=Path, default=ROOT / "calibration.json")
    parser.add_argument("--output-video", type=Path, default=ROOT / "runs/tracking/tracking_overlay.mp4")
    parser.add_argument("--output-jsonl", type=Path, default=ROOT / "runs/tracking/tracking.jsonl")
    parser.add_argument("--udp-host", default="127.0.0.1")
    parser.add_argument("--udp-port", type=int, default=5000)
    parser.add_argument("--no-udp", action="store_true")
    parser.add_argument("--device", default="1")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--tile-size", type=int, default=640)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--conf", type=float, default=0.50)
    parser.add_argument("--detect-interval", type=int, default=6,
                        help="run YOLO every N video frames; use 1 for maximum detection quality")
    parser.add_argument("--max-cars", type=int, default=2)
    parser.add_argument("--max-missed", type=int, default=8)
    parser.add_argument("--full-rescan-interval", type=int, default=36,
                        help="full source-tile recovery interval while cars are already tracked")
    parser.add_argument("--field-width-mm", type=float, default=1500.0)
    parser.add_argument("--field-height-mm", type=float, default=2500.0)
    parser.add_argument("--bg-learning-rate", type=float, default=0.001)
    parser.add_argument("--min-mask-pixels", type=int, default=20)
    parser.add_argument("--arrow-length", type=int, default=70)
    parser.add_argument("--no-video", action="store_true", help="disable overlay video for maximum runtime FPS")
    parser.add_argument("--max-frames", type=int, default=None)
    return parser.parse_args()


def center(box: list[float]) -> tuple[float, float]:
    return (0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3]))


def box_iou(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def distance(a: list[float], b: list[float]) -> float:
    ax, ay = center(a)
    bx, by = center(b)
    return float(math.hypot(ax - bx, ay - by))


def deduplicate_cars(detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for detection in sorted(detections, key=lambda item: item["confidence"], reverse=True):
        if detection["class_id"] != CAR_CLASS_ID:
            continue
        if any(box_iou(detection["xyxy"], old["xyxy"]) > 0.30 for old in kept):
            continue
        kept.append(detection)
    return kept


def car_search_roi(track: CarState, frame_shape: tuple[int, int, int], size: int, timestamp: float) -> tuple[int, int, int, int]:
    """Create a source-resolution search window around a predicted car."""
    predicted = predicted_box(track, timestamp)
    cx, cy = center(predicted)
    half = max(float(size), predicted[2] - predicted[0], predicted[3] - predicted[1]) / 2.0
    return clamp_box(
        (cx - half, cy - half, cx + half, cy + half),
        frame_shape[1],
        frame_shape[0],
    )


def detect_car_windows(
    model: Any,
    frame: np.ndarray,
    tracks: list[CarState],
    timestamp: float,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """Detect cars in one source-resolution window per active track."""
    active = [track for track in tracks if track.active and track.last_detection_timestamp is not None]
    if not active:
        return []
    windows = [car_search_roi(track, frame.shape, args.tile_size, timestamp) for track in active]
    crops = [frame[y1:y2, x1:x2] for x1, y1, x2, y2 in windows]
    results = model.predict(
        source=crops,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        batch=min(16, len(crops)),
        save=False,
        verbose=False,
    )
    detections: list[dict[str, Any]] = []
    for result, roi in zip(results, windows):
        detections.extend(
            detection
            for detection in result_detections(result, roi[0], roi[1])
            if detection["class_id"] == CAR_CLASS_ID
        )
    return detections


def predicted_box(track: CarState, timestamp: float) -> list[float]:
    if track.last_detection_timestamp is None:
        return list(track.box)
    dt = max(0.0, timestamp - track.last_detection_timestamp)
    dx, dy = track.velocity_px
    return [track.box[0] + dx * dt, track.box[1] + dy * dt,
            track.box[2] + dx * dt, track.box[3] + dy * dt]


def associate_detections(
    tracks: list[CarState],
    detections: list[dict[str, Any]],
    timestamp: float,
    max_cars: int,
) -> dict[int, dict[str, Any]]:
    active = [track for track in tracks if track.active and track.last_detection_timestamp is not None]
    candidates = []
    for track in active:
        reference = predicted_box(track, timestamp)
        size = max(reference[2] - reference[0], reference[3] - reference[1])
        for detection_index, detection in enumerate(detections):
            overlap = box_iou(reference, detection["xyxy"])
            gap = distance(reference, detection["xyxy"])
            if overlap >= 0.02 or gap <= max(160.0, 3.0 * size):
                score = overlap + 0.01 * detection["confidence"] - 1e-6 * gap
                candidates.append((score, track.car_id, detection_index))
    matches: dict[int, dict[str, Any]] = {}
    used_tracks: set[int] = set()
    used_detections: set[int] = set()
    for _score, car_id, detection_index in sorted(candidates, reverse=True):
        if car_id in used_tracks or detection_index in used_detections:
            continue
        matches[car_id] = detections[detection_index]
        used_tracks.add(car_id)
        used_detections.add(detection_index)

    free_ids = [index for index in range(1, max_cars + 1) if index not in used_tracks]
    for detection_index, detection in enumerate(detections):
        if detection_index in used_detections or not free_ids:
            continue
        car_id = free_ids.pop(0)
        matches[car_id] = detection
        used_tracks.add(car_id)
        used_detections.add(detection_index)
    return matches


def update_track(track: CarState, detection: dict[str, Any], timestamp: float) -> None:
    new_box = [float(value) for value in detection["xyxy"]]
    new_center = center(new_box)
    if track.last_detection_timestamp is not None:
        dt = timestamp - track.last_detection_timestamp
        if dt > 0:
            old_center = center(track.last_detection_box or track.box)
            track.velocity_px = ((new_center[0] - old_center[0]) / dt, (new_center[1] - old_center[1]) / dt)
    track.box = new_box
    track.last_detection_box = list(new_box)
    track.confidence = float(detection["confidence"])
    track.last_detection_timestamp = timestamp
    track.missed = 0
    track.active = True
    track.history.append(new_center)
    track.history = track.history[-30:]


def reset_measurement_state(track: CarState) -> None:
    """Reset temporal angle/velocity state after a missed detection."""
    track.prev_theta = None
    track.prev_timestamp = None
    track.prev_world = None


def make_object_mask(
    foreground: np.ndarray,
    box: list[float],
    min_pixels: int,
) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    height, width = foreground.shape[:2]
    x1 = max(0, min(width - 1, int(round(box[0]))))
    y1 = max(0, min(height - 1, int(round(box[1]))))
    x2 = max(x1 + 1, min(width, int(round(box[2]))))
    y2 = max(y1 + 1, min(height, int(round(box[3]))))
    roi = foreground[y1:y2, x1:x2]
    roi = cv2.threshold(roi, 128, 255, cv2.THRESH_BINARY)[1]
    kernel = np.ones((3, 3), np.uint8)
    roi = cv2.morphologyEx(roi, cv2.MORPH_OPEN, kernel)
    roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(roi, 8)
    if count <= 1:
        return None
    roi_center = np.array([(x2 - x1) / 2.0, (y2 - y1) / 2.0])
    choices = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_pixels:
            continue
        component_center = centroids[label]
        component_distance = float(np.linalg.norm(component_center - roi_center))
        choices.append((area / (1.0 + 0.02 * component_distance), label))
    if not choices:
        return None
    label = max(choices)[1]
    return (labels == label).astype(np.uint8) * 255, (x1, y1, x2, y2)


def angle_diff(now: float, previous: float) -> float:
    return (now - previous + 180.0) % 360.0 - 180.0


def heading_from_mask(
    frame: np.ndarray,
    object_mask: tuple[np.ndarray, tuple[int, int, int, int]] | None,
    previous_theta: float | None,
) -> tuple[float, bool, tuple[float, float] | None]:
    """Return math-angle theta, validity, and mask centroid.

    Image coordinates use +y downward. Therefore the PCA vector is converted
    with atan2(-dy, dx). PCA has a 180-degree ambiguity. We resolve it by
    comparing brightness on the two ends of the principal axis, then by
    continuity with the previous theta. If neither is available, a fixed 0
    degree direction is used deliberately and marked in this comment.
    """
    if object_mask is None:
        # No usable foreground mask: keep a deterministic fixed direction.
        # This avoids random 180-degree flips until a better mask is available.
        return 0.0, True, None
    mask, (x1, y1, _x2, _y2) = object_mask
    ys, xs = np.where(mask > 0)
    if len(xs) < 5:
        return 0.0, True, None
    points = np.column_stack((xs.astype(np.float64), ys.astype(np.float64)))
    centroid = points.mean(axis=0)
    covariance = np.cov(points - centroid, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    if not np.isfinite(axis).all() or np.linalg.norm(axis) < 1e-6:
        return 0.0, True, (float(centroid[0] + x1), float(centroid[1] + y1))
    axis = axis / np.linalg.norm(axis)
    projection = (points - centroid) @ axis
    gray = cv2.cvtColor(frame[y1 : y1 + mask.shape[0], x1 : x1 + mask.shape[1]], cv2.COLOR_BGR2GRAY)
    brightness = gray[ys, xs].astype(np.float64)
    extent = max(float(np.max(np.abs(projection))), 1.0)
    positive = projection > 0.25 * extent
    negative = projection < -0.25 * extent
    direction_sign = None
    if positive.sum() >= 3 and negative.sum() >= 3:
        brightness_difference = float(brightness[positive].mean() - brightness[negative].mean())
        if abs(brightness_difference) >= 5.0:
            direction_sign = 1.0 if brightness_difference > 0 else -1.0
    candidate_positive = math.degrees(math.atan2(-axis[1], axis[0])) % 360.0
    candidate_negative = (candidate_positive + 180.0) % 360.0
    if direction_sign is not None:
        theta = candidate_positive if direction_sign > 0 else candidate_negative
    elif previous_theta is not None:
        theta = candidate_positive if abs(angle_diff(candidate_positive, previous_theta)) <= 90.0 else candidate_negative
    else:
        # No reliable front cue: use a deterministic fixed sign instead of
        # randomly flipping every frame. This is the documented fallback.
        theta = candidate_positive if axis[0] >= 0 else candidate_negative
    return theta, True, (float(centroid[0] + x1), float(centroid[1] + y1))


def project_world(image_uv: tuple[float, float], homography: np.ndarray) -> tuple[float, float]:
    source = np.asarray([[image_uv]], dtype=np.float32)
    result = cv2.perspectiveTransform(source, homography)[0, 0]
    return float(result[0]), float(result[1])


def world_in_field(world_xy: tuple[float, float], width_mm: float, height_mm: float) -> bool:
    return 0.0 <= world_xy[0] <= width_mm and 0.0 <= world_xy[1] <= height_mm


def make_packet(
    timestamp_us: int,
    car_id: int,
    world_xy: tuple[float, float],
    theta: float,
    vx: float,
    vy: float,
    omega: float,
    image_uv: tuple[float, float],
) -> str:
    return (
        f'{timestamp_us},"{car_id}",{world_xy[0]:.3f},{world_xy[1]:.3f},'
        f'{theta:.3f},{vx:.3f},{vy:.3f},{omega:.3f},'
        f'{image_uv[0]:.3f},{image_uv[1]:.3f}\n'
    )


def draw_arrow(
    frame: np.ndarray,
    car_id: int,
    uv: tuple[float, float],
    theta: float,
    length: int,
) -> None:
    x, y = round(uv[0]), round(uv[1])
    radians = math.radians(theta)
    color = COLORS.get(car_id, (0, 255, 255))
    if theta > INVALID / 2:
        endpoint = (round(x + length * math.cos(radians)), round(y - length * math.sin(radians)))
        cv2.arrowedLine(frame, (x, y), endpoint, color, 3, cv2.LINE_AA, tipLength=0.25)
    cv2.circle(frame, (x, y), 4, color, -1)
    cv2.putText(frame, f"car {car_id}", (x + 8, max(24, y + 24)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2, cv2.LINE_AA)


def draw_orientation_panel(
    frame: np.ndarray,
    values: list[tuple[int, float, float, tuple[float, float, float]]],
) -> None:
    """Draw theta/omega in a separate fixed panel, away from each car."""
    if not values:
        return
    line_height = 32
    panel_height = 18 + line_height * len(values)
    panel_width = 850
    overlay = frame.copy()
    cv2.rectangle(overlay, (12, 58), (12 + panel_width, 58 + panel_height), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
    cv2.putText(frame, "orientation", (24, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    for index, (car_id, theta, omega, world_xyz) in enumerate(values):
        theta_text = f"{theta:.1f}" if theta > INVALID / 2 else "NA"
        omega_text = f"{omega:.1f}" if omega > INVALID / 2 else "NA"
        x_mm, y_mm, z_mm = world_xyz
        xyz_text = f"({x_mm:.0f},{y_mm:.0f},{z_mm:.0f}) mm" if x_mm > INVALID / 2 and y_mm > INVALID / 2 else "INVALID"
        color = COLORS.get(car_id, (0, 255, 255))
        label = f"car {car_id}: xyz={xyz_text}   theta={theta_text} deg   omega={omega_text} deg/s"
        cv2.putText(frame, label, (24, 112 + index * line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2, cv2.LINE_AA)


def draw_fps_panel(frame: np.ndarray, video_fps: float, processing_fps: float) -> None:
    label = f"video FPS: {video_fps:.2f}   processing FPS: {processing_fps:.2f}"
    height, width = frame.shape[:2]
    origin = (width - 470, 34)
    cv2.rectangle(frame, (origin[0] - 10, 8), (width - 10, 48), (20, 20, 20), -1)
    cv2.putText(frame, label, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 255), 2, cv2.LINE_AA)


def main() -> None:
    args = parse_args()
    for path in (args.video, args.weights, args.calibration):
        if not path.exists():
            raise SystemExit(f"File not found: {path}")
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    homography = np.asarray(calibration["homography_image_to_world"], dtype=np.float64)

    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise SystemExit("Ultralytics is required to initialize the car detections.") from error

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise SystemExit(f"Unable to open video: {args.video}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_limit = min(total_frames, args.max_frames) if args.max_frames else total_frames
    args.output_video.parent.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    if not args.no_video:
        writer = cv2.VideoWriter(str(args.output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            raise SystemExit(f"Unable to create output video: {args.output_video}")

    model = YOLO(str(args.weights))
    background = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=24, detectShadows=False)
    udp = None if args.no_udp else socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    tracks = [CarState(car_id=index) for index in range(1, args.max_cars + 1)]
    args.conf = max(args.conf, 0.01)
    started = time.perf_counter()

    with args.output_jsonl.open("w", encoding="utf-8") as log_file:
        for frame_index in range(frame_limit):
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = frame_index / fps
            timestamp_us = int(round(timestamp * 1_000_000.0))
            detections: list[dict[str, Any]] = []
            detection_frame = frame_index % max(1, args.detect_interval) == 0
            matches: dict[int, dict[str, Any]] = {}
            foreground = None
            if detection_frame:
                foreground = background.apply(frame, learningRate=args.bg_learning_rate)
                detection_args = argparse.Namespace(
                    tile_size=args.tile_size,
                    overlap=args.overlap,
                    imgsz=args.imgsz,
                    conf=args.conf,
                    device=args.device,
                )
                tracked_cars = [track for track in tracks if track.active and track.last_detection_timestamp is not None]
                full_rescan = (
                    not tracked_cars
                    or frame_index % max(1, args.full_rescan_interval) == 0
                )
                if full_rescan:
                    detections = deduplicate_cars(detect_tiles(model, frame, detection_args))
                else:
                    detections = deduplicate_cars(
                        detect_car_windows(model, frame, tracked_cars, timestamp, args)
                    )
                matches = associate_detections(tracks, detections, timestamp, args.max_cars)
                for track in tracks:
                    detection = matches.get(track.car_id)
                    if detection is None:
                        if track.last_detection_timestamp is not None:
                            track.missed += 1
                            reset_measurement_state(track)
                            if track.missed > args.max_missed:
                                track.active = False
                    else:
                        update_track(track, detection, timestamp)
            else:
                # Between YOLO frames, use the tracker's constant-velocity
                # prediction. This keeps output at the video frame rate while
                # reducing expensive tile inference calls.
                for track in tracks:
                    if track.active and track.last_detection_timestamp is not None:
                        track.box = list(clamp_box(
                            tuple(predicted_box(track, timestamp)),
                            width,
                            height,
                        ))

            orientation_values: list[tuple[int, float, float, tuple[float, float, float]]] = []
            for track in tracks:
                detected_this_frame = track.car_id in matches if detection_frame else track.active and track.last_detection_timestamp is not None
                if track.active and detected_this_frame:
                    box = track.box
                    if foreground is not None:
                        object_mask = make_object_mask(foreground, box, args.min_mask_pixels)
                        theta, theta_valid, mask_uv = heading_from_mask(frame, object_mask, track.prev_theta)
                    else:
                        # No expensive full-frame mask between detector frames.
                        # Keep the last heading while the tracker predicts the
                        # position; the next detector frame refreshes PCA.
                        theta = track.prev_theta if track.prev_theta is not None else 0.0
                        theta_valid = True
                        mask_uv = center(box)
                    u, v = ((mask_uv if mask_uv is not None else center(box)))
                    projected_world = project_world((u, v), homography)
                    world_valid = world_in_field(
                        projected_world,
                        args.field_width_mm,
                        args.field_height_mm,
                    )
                    world_xy = projected_world if world_valid else (INVALID, INVALID)
                    if world_valid and track.prev_world is not None and track.prev_timestamp is not None:
                        dt = timestamp - track.prev_timestamp
                        if dt > 0:
                            vx = (world_xy[0] - track.prev_world[0]) / dt
                            vy = (world_xy[1] - track.prev_world[1]) / dt
                        else:
                            vx = vy = INVALID
                    else:
                        vx = vy = INVALID
                    if theta_valid and track.prev_theta is not None and track.prev_timestamp is not None:
                        dt = timestamp - track.prev_timestamp
                        omega = angle_diff(theta, track.prev_theta) / dt if dt > 0 else INVALID
                    else:
                        omega = INVALID
                    theta_output = theta if theta_valid else INVALID
                    if not args.no_video:
                        draw_arrow(
                            frame,
                            track.car_id,
                            (u, v),
                            theta_output,
                            args.arrow_length,
                        )
                    orientation_values.append((track.car_id, theta_output, omega, (world_xy[0], world_xy[1], 0.0)))
                    packet = make_packet(timestamp_us, track.car_id, world_xy, theta_output,
                                         vx, vy, omega, (u, v))
                    track.prev_theta = theta if theta_valid else None
                    track.prev_timestamp = timestamp if theta_valid else None
                    track.prev_world = world_xy if world_valid else None
                    record = {
                        "frame_index": frame_index,
                        "timestamp_us": timestamp_us,
                        "car_id": track.car_id,
                        "detected": True,
                        "image_uv": [u, v],
                        "world_xyz": [world_xy[0], world_xy[1], 0.0],
                        "theta": theta_output,
                        "vx_mm_s": vx,
                        "vy_mm_s": vy,
                        "omega_deg_s": omega,
                        "udp": packet.rstrip("\n"),
                    }
                else:
                    # The assignment requires a fixed per-car packet even
                    # when a car is absent from the current frame.
                    reset_measurement_state(track)
                    packet = make_packet(
                        timestamp_us,
                        track.car_id,
                        (INVALID, INVALID),
                        INVALID,
                        INVALID,
                        INVALID,
                        INVALID,
                        (INVALID, INVALID),
                    )
                    record = {
                        "frame_index": frame_index,
                        "timestamp_us": timestamp_us,
                        "car_id": track.car_id,
                        "detected": False,
                        "image_uv": [INVALID, INVALID],
                        "world_xyz": [INVALID, INVALID, 0.0],
                        "theta": INVALID,
                        "vx_mm_s": INVALID,
                        "vy_mm_s": INVALID,
                        "omega_deg_s": INVALID,
                        "udp": packet.rstrip("\n"),
                    }
                if udp is not None:
                    udp.sendto(packet.encode("utf-8"), (args.udp_host, args.udp_port))
                log_file.write(json.dumps(record, ensure_ascii=False) + "\n")

            if not args.no_video:
                draw_orientation_panel(frame, orientation_values)
                elapsed = time.perf_counter() - started
                draw_fps_panel(frame, fps, (frame_index + 1) / max(elapsed, 1e-6))
            if writer is not None:
                writer.write(frame)
            if (frame_index + 1) % 100 == 0 or frame_index + 1 == frame_limit:
                elapsed = time.perf_counter() - started
                print(f"processed {frame_index + 1}/{frame_limit}; processing_fps={(frame_index + 1) / max(elapsed, 1e-6):.2f}")

    capture.release()
    if writer is not None:
        writer.release()
    if udp is not None:
        udp.close()
    if writer is not None:
        print(f"Saved tracking video: {args.output_video}")
    else:
        print("Overlay video disabled by --no-video")
    print(f"Saved tracking log: {args.output_jsonl}")
    if args.no_udp:
        print("UDP disabled by --no-udp")
    else:
        print(f"UDP destination: {args.udp_host}:{args.udp_port}")


if __name__ == "__main__":
    main()
