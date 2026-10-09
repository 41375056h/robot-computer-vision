"""Create independent source-resolution ROI videos for each tracked object."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np

from video_crop import (
    BOX_COLORS,
    CLASS_NAMES,
    TARGET_CLASSES,
    clamp_box,
    crop_for_output,
    detect_tiles,
    expanded_union,
    result_detections,
)


ROOT = Path(__file__).resolve().parent


@dataclass
class Track:
    track_id: int
    class_id: int
    class_name: str
    roi: tuple[int, int, int, int]
    last_box: list[float]
    missed: int = 0
    active: bool = True
    writer: Any = None
    start_frame: int = 0
    tracker_id: int | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "IMG_5985.MOV")
    parser.add_argument("--weights", type=Path, default=ROOT / "runs/baseline10_gpu1/weights/best.pt")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/video_multi_crop/rois")
    parser.add_argument("--metadata", type=Path, default=ROOT / "runs/video_multi_crop/frames.jsonl")
    parser.add_argument("--debug-video", type=Path, default=ROOT / "runs/video_multi_crop/debug.mp4")
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
    parser.add_argument("--car-conf", type=float, default=0.50)
    parser.add_argument("--cone-conf", type=float, default=0.15)
    parser.add_argument("--track-car", action=argparse.BooleanOptionalAction, default=True,
                        help="Track toy_car objects with ByteTrack; red_cone is never passed to the tracker.")
    parser.add_argument("--track-buffer", type=int, default=30)
    parser.add_argument("--track-high-thresh", type=float, default=0.25)
    parser.add_argument("--track-low-thresh", type=float, default=0.10)
    parser.add_argument("--track-match-thresh", type=float, default=0.80)
    parser.add_argument("--max-frames", type=int, default=None)
    return parser.parse_args()


def iou(a: list[float], b: list[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def center_distance(a: list[float], b: list[float]) -> float:
    ax, ay = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
    bx, by = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return float(np.hypot(ax - bx, ay - by))


def deduplicate(detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for detection in sorted(detections, key=lambda d: d["confidence"], reverse=True):
        if any(detection["class_id"] == old["class_id"] and iou(detection["xyxy"], old["xyxy"]) > 0.30 for old in kept):
            continue
        kept.append(detection)
    return kept


def filter_by_class_confidence(detections: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    return [
        detection
        for detection in detections
        if detection["confidence"] >= (args.car_conf if detection["class_id"] == 1 else args.cone_conf)
    ]


def make_car_tracker(args: argparse.Namespace) -> Any:
    """Create ByteTrack for cars only.

    The detector still runs on source-resolution tiles/crops. Only toy_car
    detections are fed into this tracker, so red_cone never receives a
    tracker ID.
    """
    from ultralytics.trackers.byte_tracker import BYTETracker

    tracker_args = SimpleNamespace(
        track_high_thresh=args.track_high_thresh,
        track_low_thresh=args.track_low_thresh,
        new_track_thresh=args.track_high_thresh,
        track_buffer=args.track_buffer,
        match_thresh=args.track_match_thresh,
        fuse_score=True,
    )
    return BYTETracker(tracker_args)


def update_car_tracker(
    tracker: Any,
    detections: list[dict[str, Any]],
    frame: np.ndarray,
) -> dict[int, dict[str, Any]]:
    """Update ByteTrack and return ``tracker_id -> matched detection``."""
    from ultralytics.engine.results import Boxes

    car_detections = [d for d in detections if d["class_id"] == 1]
    if car_detections:
        data = np.asarray(
            [d["xyxy"] + [d["confidence"], d["class_id"]] for d in car_detections],
            dtype=np.float32,
        )
    else:
        data = np.empty((0, 6), dtype=np.float32)
    outputs = tracker.update(Boxes(data, frame.shape[:2]), frame)
    matched: dict[int, dict[str, Any]] = {}
    for row in outputs:
        tracker_id = int(round(float(row[4])))
        detection_index = int(round(float(row[7])))
        if not 0 <= detection_index < len(car_detections):
            continue
        detection = dict(car_detections[detection_index])
        detection["xyxy"] = [float(value) for value in row[:4]]
        detection["tracker_id"] = tracker_id
        matched[tracker_id] = detection
    return matched


def associate(tracks: list[Track], detections: list[dict[str, Any]]) -> tuple[dict[int, dict[str, Any]], set[int]]:
    candidates = []
    for track_index, track in enumerate(tracks):
        if not track.active:
            continue
        for detection_index, detection in enumerate(detections):
            if track.class_id != detection["class_id"]:
                continue
            old_box = track.last_box
            new_box = detection["xyxy"]
            distance_limit = max(160.0, 2.5 * max(old_box[2] - old_box[0], old_box[3] - old_box[1]))
            overlap = iou(old_box, new_box)
            distance = center_distance(old_box, new_box)
            if overlap >= 0.02 or distance <= distance_limit:
                candidates.append((overlap + detection["confidence"] * 0.01 - distance * 1e-6, track_index, detection_index))
    matches: dict[int, dict[str, Any]] = {}
    used_detections: set[int] = set()
    used_tracks: set[int] = set()
    for _, track_index, detection_index in sorted(candidates, reverse=True):
        if track_index in used_tracks or detection_index in used_detections:
            continue
        matches[tracks[track_index].track_id] = detections[detection_index]
        used_tracks.add(track_index)
        used_detections.add(detection_index)
    return matches, used_detections


def car_search_roi(track: Track, frame_shape: tuple[int, int, int], size: int) -> tuple[int, int, int, int]:
    """Return a source-resolution search window centered on the last car box."""
    height, width = frame_shape[:2]
    x1, y1, x2, y2 = track.last_box
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    half = max(float(size), x2 - x1, y2 - y1) / 2.0
    return clamp_box((cx - half, cy - half, cx + half, cy + half), width, height)


def detect_track_crops(model: Any, frame: np.ndarray, tracks: list[Track], args: argparse.Namespace) -> dict[int, dict[str, Any] | None]:
    active = [track for track in tracks if track.active]
    if not active:
        return {}
    search_rois = [
        car_search_roi(track, frame.shape, args.tile_size) if track.class_id == 1 else track.roi
        for track in active
    ]
    crops = [frame[roi[1] : roi[3], roi[0] : roi[2]] for roi in search_rois]
    results = model.predict(
        source=crops,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        batch=min(16, len(crops)),
        save=False,
        verbose=False,
    )
    selected: dict[int, dict[str, Any] | None] = {}
    for track, search_roi, result in zip(active, search_rois, results):
        detections = [
            d
            for d in filter_by_class_confidence(result_detections(result, search_roi[0], search_roi[1]), args)
            if d["class_id"] == track.class_id
        ]
        if not detections:
            selected[track.track_id] = None
            continue
        selected[track.track_id] = max(
            detections,
            key=lambda d: (iou(track.last_box, d["xyxy"]), d["confidence"]),
        )
    return selected


def update_track(track: Track, detection: dict[str, Any] | None, frame_shape: tuple[int, int, int], args: argparse.Namespace) -> None:
    if detection is None:
        track.missed += 1
        if track.missed > args.max_missed:
            track.active = False
        return
    track.last_box = detection["xyxy"]
    track.roi = expanded_union([detection], frame_shape, args.margin, args.min_margin_px) or track.roi
    track.missed = 0


def draw_debug(frame: np.ndarray, tracks: list[Track], detections: dict[int, dict[str, Any] | None], frame_index: int) -> np.ndarray:
    output = frame.copy()
    for track in tracks:
        if not track.active:
            continue
        color = BOX_COLORS.get(track.class_id, (0, 255, 255))
        x1, y1, x2, y2 = track.roi
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        detection = detections.get(track.track_id)
        if detection is not None:
            dx1, dy1, dx2, dy2 = [int(round(v)) for v in detection["xyxy"]]
            cv2.rectangle(output, (dx1, dy1), (dx2, dy2), color, 3)
            identity = f"car ID {track.tracker_id}" if track.tracker_id is not None else f"ROI {track.track_id}"
            label = f"{identity} {track.class_name} {detection['confidence']:.2f}"
        else:
            identity = f"car ID {track.tracker_id}" if track.tracker_id is not None else f"ROI {track.track_id}"
            label = f"{identity} {track.class_name} missed={track.missed}"
        cv2.putText(output, label, (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    cv2.putText(output, f"frame={frame_index} independent_rois={sum(t.active for t in tracks)}", (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, cv2.LINE_AA)
    return output


def open_track_writer(track: Track, output_dir: Path, fps: float, size: int) -> None:
    if track.tracker_id is not None:
        filename = output_dir / f"car_track_{track.tracker_id:03d}_{track.class_name}.mp4"
    else:
        filename = output_dir / f"roi_{track.track_id:03d}_{track.class_name}.mp4"
    track.writer = cv2.VideoWriter(str(filename), cv2.VideoWriter_fourcc(*"mp4v"), fps, (size, size))
    if not track.writer.isOpened():
        raise SystemExit(f"Unable to create ROI video: {filename}")


def main() -> None:
    args = parse_args()
    if not args.input.exists() or not args.weights.exists():
        raise SystemExit("Input video or weights not found.")
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
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.debug_video.parent.mkdir(parents=True, exist_ok=True)
    debug_writer = cv2.VideoWriter(str(args.debug_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    model = YOLO(str(args.weights))
    car_tracker = make_car_tracker(args) if args.track_car else None
    tracks: list[Track] = []
    car_tracks: dict[int, Track] = {}
    next_id = 0
    force_rescan = True
    started = time.perf_counter()

    with args.metadata.open("w", encoding="utf-8") as metadata_file:
        for frame_index in range(frame_limit):
            ok, frame = capture.read()
            if not ok:
                break
            active_tracks = [track for track in tracks if track.active]
            do_rescan = force_rescan or not active_tracks or frame_index % args.rescan_interval == 0
            current: dict[int, dict[str, Any] | None] = {}

            if do_rescan:
                detections = deduplicate(filter_by_class_confidence(detect_tiles(model, frame, args), args))
                # ByteTrack is intentionally fed toy_car detections only.
                car_matches = update_car_tracker(car_tracker, detections, frame) if car_tracker else {}
                for tracker_id, track in list(car_tracks.items()):
                    if track.active:
                        detection = car_matches.get(tracker_id)
                        update_track(track, detection, frame.shape, args)
                        current[track.track_id] = detection
                for tracker_id, detection in car_matches.items():
                    track = car_tracks.get(tracker_id)
                    if track is None or not track.active:
                        roi = expanded_union([detection], frame.shape, args.margin, args.min_margin_px)
                        track = Track(
                            next_id,
                            detection["class_id"],
                            detection["class_name"],
                            roi,
                            detection["xyxy"],
                            start_frame=frame_index,
                            tracker_id=tracker_id,
                        )
                        car_tracks[tracker_id] = track
                        tracks.append(track)
                        open_track_writer(track, args.output_dir, fps, args.imgsz)
                        next_id += 1
                    update_track(track, detection, frame.shape, args)
                    current[track.track_id] = detection

                # Red cones are not passed to ByteTrack. They retain independent
                # ROI continuity through the crop association only.
                cone_tracks = [track for track in tracks if track.class_id == 0 and track.active]
                cone_detections = [detection for detection in detections if detection["class_id"] == 0]
                cone_matches, used = associate(cone_tracks, cone_detections)
                for track in cone_tracks:
                    detection = cone_matches.get(track.track_id)
                    update_track(track, detection, frame.shape, args)
                    current[track.track_id] = detection
                for detection_index, detection in enumerate(cone_detections):
                    if detection_index in used:
                        continue
                    roi = expanded_union([detection], frame.shape, args.margin, args.min_margin_px)
                    track = Track(
                        next_id,
                        detection["class_id"],
                        detection["class_name"],
                        roi,
                        detection["xyxy"],
                        start_frame=frame_index,
                    )
                    open_track_writer(track, args.output_dir, fps, args.imgsz)
                    tracks.append(track)
                    current[track.track_id] = detection
                    next_id += 1
                force_rescan = False
            else:
                crop_detections = detect_track_crops(model, frame, active_tracks, args)
                current = {}

                # Run ByteTrack on car detections from their individual crops.
                car_crop_detections = [
                    detection
                    for track in active_tracks
                    if track.class_id == 1
                    for detection in [crop_detections.get(track.track_id)]
                    if detection is not None
                ]
                car_matches = update_car_tracker(car_tracker, car_crop_detections, frame) if car_tracker else {}
                for track in active_tracks:
                    if track.class_id == 1:
                        detection = car_matches.get(track.tracker_id) if track.tracker_id is not None else None
                    else:
                        detection = crop_detections.get(track.track_id)
                    update_track(track, detection, frame.shape, args)
                    current[track.track_id] = detection

                # A new car appearing inside an existing crop gets its own ROI.
                for tracker_id, detection in car_matches.items():
                    if tracker_id in car_tracks:
                        continue
                    roi = expanded_union([detection], frame.shape, args.margin, args.min_margin_px)
                    track = Track(
                        next_id,
                        detection["class_id"],
                        detection["class_name"],
                        roi,
                        detection["xyxy"],
                        start_frame=frame_index,
                        tracker_id=tracker_id,
                    )
                    car_tracks[tracker_id] = track
                    tracks.append(track)
                    open_track_writer(track, args.output_dir, fps, args.imgsz)
                    current[track.track_id] = detection
                    next_id += 1
                force_rescan = any(
                    current.get(track.track_id) is not None
                    and (
                        current[track.track_id]["xyxy"][0] - track.roi[0] < args.boundary_px
                        or current[track.track_id]["xyxy"][1] - track.roi[1] < args.boundary_px
                        or track.roi[2] - current[track.track_id]["xyxy"][2] < args.boundary_px
                        or track.roi[3] - current[track.track_id]["xyxy"][3] < args.boundary_px
                    )
                    or track.missed >= args.max_missed
                    for track in active_tracks
                )

            for track in tracks:
                if track.writer is None or not track.active:
                    continue
                detection = current.get(track.track_id)
                track.writer.write(crop_for_output(frame, track.roi, args.imgsz, [detection] if detection else []))

            debug_writer.write(draw_debug(frame, tracks, current, frame_index))
            record = {
                "frame": frame_index,
                "timestamp_sec": frame_index / fps,
                "source_size": [width, height],
                "tracks": [
                    {
                        "track_id": track.track_id,
                        "tracker_id": track.tracker_id,
                        "class_id": track.class_id,
                        "class_name": track.class_name,
                        "active": track.active,
                        "roi_xyxy": list(track.roi),
                        "missed": track.missed,
                        "detection": current.get(track.track_id),
                    }
                    for track in tracks
                ],
            }
            metadata_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            if (frame_index + 1) % 100 == 0 or frame_index + 1 == frame_limit:
                elapsed = time.perf_counter() - started
                print(f"processed {frame_index + 1}/{frame_limit} frames; active_rois={sum(t.active for t in tracks)}; {elapsed:.1f}s")

    capture.release()
    debug_writer.release()
    for track in tracks:
        if track.writer is not None:
            track.writer.release()
    print(f"Saved independent ROI videos under: {args.output_dir}")
    print(f"Saved metadata: {args.metadata}")
    print(f"Saved debug video: {args.debug_video}")


if __name__ == "__main__":
    main()
