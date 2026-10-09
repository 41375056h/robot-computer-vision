# Robot Computer Vision - Toy Race Car

This repository is the runnable submission for Assignment 1. It detects and tracks toy race cars, maps image coordinates to field coordinates in millimetres, estimates orientation and motion, and sends one UTF-8 UDP packet per car per video frame.

## Contents

```text
tracking.py              main detector, tracker, mapping, visualization, and UDP pipeline
video_crop.py            source-resolution tiled YOLO helpers
video_multi_crop.py      independent ROI preprocessing/tracking utility
calibrate_world.py       manual points and homography generation
project_world.py         world-coordinate projection utility
evaluate_roc.py          ROC evaluation
evaluate_mapping.py      homography and static-car evaluation
udp_receiver.py          local UDP receiver test
verify_calibration.py    calibration validation
calibration.json         selected image-to-field homography
points.json              manually selected calibration points
weights/best.pt          trained YOLO checkpoint
assets/roc_curve.png     ROC figure
assets/demo_frame.png    final overlay example
```

The dataset and source video are not included. The dataset path in `data.yaml` must be changed if the dataset is stored elsewhere.

## Installation

```bash
python -m pip install ultralytics opencv-python numpy matplotlib
```

The verified run used CUDA device 1. Select another device with `--device`.

## Run the tracker

Create a demonstration overlay video:

```bash
python tracking.py \
  --video /path/to/IMG_5985.MOV \
  --weights weights/best.pt \
  --calibration calibration.json \
  --device 1 \
  --output-video runs/tracking_overlay.mp4 \
  --output-jsonl runs/tracking.jsonl
```

For runtime UDP operation, use `--no-video` to avoid high-resolution video encoding:

```bash
python tracking.py \
  --video /path/to/video.MOV \
  --weights weights/best.pt \
  --calibration calibration.json \
  --device 1 \
  --no-video \
  --udp-host 127.0.0.1 \
  --udp-port 5000 \
  --output-jsonl runs/tracking.jsonl
```

The default detector cadence is every 6 video frames, with full source-tile recovery every 36 frames. The tested source video is 29.984 FPS; runtime processing measured approximately 47 FPS in `--no-video` mode on GPU 1.

## UDP format

The default UDP port is 5000 and can be changed with `--udp-port`. Packets are UTF-8 text with this fixed order:

```text
timestamp_us,"car_id",x_mm,y_mm,theta,vx_mm_s,vy_mm_s,omega_deg_s,u_px,v_px
```

Unavailable measurements use `-1000.0`. Timestamps use video time, not processing time. Car IDs 1 and 2 are maintained independently; only toy-car class 1 is sent to the tracker.

Test the receiver with:

```bash
python udp_receiver.py --host 127.0.0.1 --port 5000 --count 60
```

The verified local test received 60/60 packets.

## Calibration

The field is treated as a plane. `calibration.json` stores a homography generated with `cv.findHomography()`. It maps image `(u,v)` to field `(x,y)` in millimetres and uses `z=0`.

To create a new calibration:

```bash
python calibrate_world.py mark --video /path/to/video.mp4 --frame 800 --points points.json
python calibrate_world.py solve --points points.json --output calibration.json --units mm
```

The selected calibration has 5/8 RANSAC inliers, an inlier mean reprojection error of 0.6798 mm, and an inlier maximum error of 1.1083 mm.

## Evaluation

ROC evaluation:

```bash
python evaluate_roc.py \
  --data-root /path/to/CV/data \
  --weights weights/best.pt \
  --device 1 \
  --output-dir runs/roc
```

Verified result: AUC `0.999904`, best threshold `0.809`, TPR `0.9976`, and FPR `0.0000`. This is an image-level toy-car presence ROC based on the maximum confidence per image.

Mapping evaluation requires a physically measured CSV with columns `frame_index,car_id,x_mm,y_mm,theta_deg`:

```bash
python evaluate_mapping.py \
  --calibration calibration.json \
  --tracking-jsonl runs/tracking.jsonl \
  --ground-truth static_car_ground_truth.csv
```

The current `(1800,1200)` mm / 90-degree CSV is preliminary only; those values were not independently measured in the corresponding video frames. The final report documents this limitation.

## Final artifacts and limitations

The full demo video, preview video, JSONL log, ROC CSV/PNG, and Word report are kept outside this code repository under `/home/intern/jeremy/CVHW/runs/` because videos are large.

- The source video is 29.984 FPS, so a true 60 FPS camera demonstration requires a higher-frame-rate input.
- Orientation uses foreground-mask PCA and can be ambiguous for symmetric masks.
- Homography accuracy outside the manually marked field region is uncertain.
- A physically measured static-car ground truth is still required for the final mapping-error table.
