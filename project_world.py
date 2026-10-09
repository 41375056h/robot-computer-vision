"""Project detections from the selected debug video into field coordinates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent


def project_point(point_xy: tuple[float, float], homography: np.ndarray) -> tuple[float, float]:
    source = np.asarray([[point_xy]], dtype=np.float32)
    projected = cv2.perspectiveTransform(source, homography)[0, 0]
    return float(projected[0]), float(projected[1])


def bottom_center(detection: dict[str, Any]) -> tuple[float, float]:
    x1, y1, x2, y2 = detection["xyxy"]
    return (0.5 * (float(x1) + float(x2)), float(y2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=ROOT / "runs/video_multi_crop/debug.mp4")
    parser.add_argument("--metadata", type=Path, default=ROOT / "runs/video_multi_crop/frames.jsonl")
    parser.add_argument("--calibration", type=Path, default=ROOT / "runs/calibration/calibration.json")
    parser.add_argument("--output-video", type=Path, default=ROOT / "runs/calibration/world_overlay.mp4")
    parser.add_argument("--output-jsonl", type=Path, default=ROOT / "runs/calibration/world_tracks.jsonl")
    parser.add_argument("--class-name", default="toy_car")
    parser.add_argument("--ground-z", type=float, default=0.0, help="ground-plane Z coordinate")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    homography = np.asarray(calibration["homography_image_to_world"], dtype=np.float64)
    world_units = calibration.get("world_units", "m")

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise SystemExit(f"Unable to open video: {args.video}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    args.output_video.parent.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise SystemExit(f"Unable to create output video: {args.output_video}")

    with args.metadata.open(encoding="utf-8") as metadata_file, args.output_jsonl.open("w", encoding="utf-8") as output_file:
        for frame_index, line in enumerate(metadata_file):
            row = json.loads(line)
            ok, frame = capture.read()
            if not ok:
                break
            world_tracks = []
            for track in row.get("tracks", []):
                detection = track.get("detection")
                if detection is None or track.get("class_name") != args.class_name:
                    continue
                image_xy = bottom_center(detection)
                world_xy = project_point(image_xy, homography)
                world_xyz = [world_xy[0], world_xy[1], args.ground_z]
                world_track = {
                    "track_id": track.get("track_id"),
                    "tracker_id": track.get("tracker_id"),
                    "class_name": track.get("class_name"),
                    "image_uv": list(image_xy),
                    "world_xyz": world_xyz,
                    "x": world_xyz[0],
                    "y": world_xyz[1],
                    "z": world_xyz[2],
                    "world_units": world_units,
                }
                world_tracks.append(world_track)
                label = f"ID {track.get('tracker_id', track.get('track_id'))}: ({world_xyz[0]:.1f}, {world_xyz[1]:.1f}, {world_xyz[2]:.1f}) {world_units}"
                x, y = round(image_xy[0]), round(image_xy[1])
                cv2.circle(frame, (x, y), 6, (0, 255, 255), -1)
                cv2.putText(frame, label, (max(5, x - 240), max(24, y - 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
            output_file.write(json.dumps({"frame": row.get("frame", frame_index), "tracks": world_tracks}) + "\n")
            writer.write(frame)

    capture.release()
    writer.release()
    print(f"Saved world-coordinate video: {args.output_video}")
    print(f"Saved world-coordinate metadata: {args.output_jsonl}")


if __name__ == "__main__":
    main()
