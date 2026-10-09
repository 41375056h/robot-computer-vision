"""Manually calibrate the selected video and map image points to field coordinates.

The homography path needs four or more field-plane correspondences:
image pixel (u, v) <-> field coordinate (X, Y).  The optional camera path
uses cv.calibrateCamera() from one or more views containing known 3-D points.
For a flat field, use Z=0 for all manually measured field points.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_VIDEO = ROOT / "runs/video_multi_crop/debug.mp4"
DEFAULT_POINTS = ROOT / "runs/calibration/points.json"
DEFAULT_CALIBRATION = ROOT / "runs/calibration/calibration.json"


def load_frame(video: Path, frame_index: int) -> tuple[np.ndarray, float]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise SystemExit(f"Unable to open video: {video}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    capture.release()
    if not ok:
        raise SystemExit(f"Unable to read frame {frame_index} from {video}")
    return frame, fps


def select_image_points(frame: np.ndarray, window: str = "manual calibration") -> list[list[float]]:
    """Select points on a resized display while retaining source coordinates."""
    height, width = frame.shape[:2]
    max_width, max_height = 1100, 1200
    scale = min(1.0, max_width / width, max_height / height)
    display_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    points: list[list[float]] = []

    def redraw() -> None:
        canvas = cv2.resize(frame, display_size, interpolation=cv2.INTER_AREA)
        for index, (x, y) in enumerate(points):
            dx, dy = round(x * scale), round(y * scale)
            cv2.circle(canvas, (dx, dy), 7, (0, 0, 255), -1)
            cv2.putText(canvas, str(index + 1), (dx + 10, dy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(canvas, "left click: add | u/right click: undo | s: save | esc: cancel", (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(window, canvas)

    def on_mouse(event: int, x: int, y: int, _flags: int, _userdata: Any) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append([round(x / scale, 3), round(y / scale, 3)])
            redraw()
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()
            redraw()

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, display_size[0], display_size[1])
    cv2.setMouseCallback(window, on_mouse)
    redraw()
    while True:
        key = cv2.waitKey(30) & 0xFF
        if key in (27, ord("q")):
            cv2.destroyWindow(window)
            raise SystemExit("Manual marking cancelled.")
        if key == ord("u") and points:
            points.pop()
            redraw()
        if key == ord("s"):
            if len(points) < 4:
                print("Select at least four points before saving.")
                continue
            cv2.destroyWindow(window)
            return points


def prompt_world_points(count: int) -> list[list[float]]:
    print("Enter the matching field coordinates in the same order as the image points.")
    print("Use one consistent coordinate system, normally metres; e.g. X Y.")
    points = []
    for index in range(count):
        while True:
            raw = input(f"world point {index + 1} (X Y): ").strip().replace(",", " ")
            values = raw.split()
            if len(values) == 2:
                try:
                    points.append([float(values[0]), float(values[1])])
                    break
                except ValueError:
                    pass
            print("Please enter exactly two numbers, for example: 1.25 3.00")
    return points


def solve_homography(
    image_points: list[list[float]],
    world_points: list[list[float]],
    ransac_threshold: float = 3.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image = np.asarray(image_points, dtype=np.float32)
    world = np.asarray(world_points, dtype=np.float32)
    if image.shape != world.shape or image.ndim != 2 or image.shape[0] < 4 or image.shape[1] != 2:
        raise ValueError("Homography requires matching Nx2 image/world points with N >= 4.")
    homography, inlier_mask = cv2.findHomography(image, world, cv2.RANSAC, ransac_threshold)
    if homography is None:
        raise ValueError("cv.findHomography() failed; check that the points are not collinear.")
    projected = cv2.perspectiveTransform(image.reshape(-1, 1, 2), homography).reshape(-1, 2)
    errors = np.linalg.norm(projected - world, axis=1)
    return homography, inlier_mask.reshape(-1).astype(bool), errors


def calibrate_camera(views: list[dict[str, Any]], image_size: tuple[int, int]) -> dict[str, Any]:
    """Run cv.calibrateCamera() from manually measured 3-D points over views."""
    object_points = [np.asarray(view["world_points_3d"], dtype=np.float32) for view in views]
    image_points = [np.asarray(view["image_points_px"], dtype=np.float32) for view in views]
    if len(views) < 2:
        print("Warning: camera calibration is much more reliable with multiple views.")
    if any(points.ndim != 2 or points.shape[1] != 3 or len(points) < 6 for points in object_points):
        raise ValueError("Each camera-calibration view needs at least 6 Nx3 world points.")
    if any(points.shape != objects[:, :2].shape for points, objects in zip(image_points, object_points)):
        raise ValueError("Each view must have matching image and world point counts.")
    rms, camera_matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None,
    )
    return {
        "rms_reprojection_error_px": float(rms),
        "camera_matrix": camera_matrix.tolist(),
        "distortion_coefficients": distortion.reshape(-1).tolist(),
        "rvecs": [vector.reshape(-1).tolist() for vector in rvecs],
        "tvecs": [vector.reshape(-1).tolist() for vector in tvecs],
        "image_size": list(image_size),
        "views": len(views),
    }


def mark(args: argparse.Namespace) -> None:
    frame, fps = load_frame(args.video, args.frame)
    image_points = select_image_points(frame)
    world_points = prompt_world_points(len(image_points))
    args.points.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "video": str(args.video),
        "frame_index": args.frame,
        "fps": fps,
        "image_size": [int(frame.shape[1]), int(frame.shape[0])],
        "image_points_px": image_points,
        "world_points_xy": world_points,
        "world_units": args.units,
    }
    args.points.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved manual points: {args.points}")


def solve(args: argparse.Namespace) -> None:
    data = json.loads(args.points.read_text(encoding="utf-8"))
    homography, mask, errors = solve_homography(
        data["image_points_px"],
        data["world_points_xy"],
        args.ransac_threshold,
    )
    result: dict[str, Any] = {
        "video": data.get("video", str(args.video)),
        "frame_index": data.get("frame_index"),
        "image_size": data["image_size"],
        "world_units": data.get("world_units", args.units),
        "image_points_px": data["image_points_px"],
        "world_points_xy": data["world_points_xy"],
        "homography_image_to_world": homography.tolist(),
        "homography_inliers": mask.tolist(),
        "homography_reprojection_errors": errors.tolist(),
        "homography_rmse": float(np.sqrt(np.mean(errors[mask] ** 2))) if mask.any() else None,
    }
    if args.camera_views:
        camera_data = json.loads(args.camera_views.read_text(encoding="utf-8"))
        result["camera_calibration"] = calibrate_camera(camera_data["views"], tuple(data["image_size"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved calibration: {args.output}")
    print(f"Homography inliers: {int(mask.sum())}/{len(mask)}")
    print(f"Homography RMSE: {result['homography_rmse']:.4f} {result['world_units']}")
    for index, (inlier, error) in enumerate(zip(mask, errors), start=1):
        print(f"  point {index}: {'inlier' if inlier else 'outlier'}, error={error:.4f} {result['world_units']}")
    if "camera_calibration" in result:
        print(f"Camera RMS error: {result['camera_calibration']['rms_reprojection_error_px']:.4f} px")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    mark_parser = subparsers.add_parser("mark", help="click image points and enter matching field coordinates")
    mark_parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    mark_parser.add_argument("--frame", type=int, default=800)
    mark_parser.add_argument("--points", type=Path, default=DEFAULT_POINTS)
    mark_parser.add_argument("--units", default="m")
    mark_parser.set_defaults(func=mark)

    solve_parser = subparsers.add_parser("solve", help="run findHomography and optionally calibrateCamera")
    solve_parser.add_argument("--points", type=Path, default=DEFAULT_POINTS)
    solve_parser.add_argument("--output", type=Path, default=DEFAULT_CALIBRATION)
    solve_parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    solve_parser.add_argument("--units", default="m")
    solve_parser.add_argument("--ransac-threshold", type=float, default=3.0,
                              help="RANSAC reprojection threshold in the selected world units")
    solve_parser.add_argument("--camera-views", type=Path, default=None,
                              help="JSON with multiple views for cv.calibrateCamera()")
    solve_parser.set_defaults(func=solve)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    arguments = parse_args()
    arguments.func(arguments)
