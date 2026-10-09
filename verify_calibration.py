import cv2
import json
import numpy as np

# 讀校正檔
cal = json.load(open("calibration.json"))
H = np.array(cal["homography_image_to_world"], np.float64)

# 從 world_points_xy 找場地範圍
world = np.array(cal["world_points_xy"], np.float64)
world_w = world[:, 0].max()
world_h = world[:, 1].max()

# 輸出解析度：每 mm 幾個像素
scale = 0.5
out_w = int(world_w * scale)
out_h = int(world_h * scale)

# image -> world(mm) -> output pixel
S = np.array([[scale, 0, 0],
              [0, scale, 0],
              [0, 0, 1]], np.float64)
H_out = S @ H

# 讀第 800 幀
cap = cv2.VideoCapture("debug.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 800)
ok, frame = cap.read()
cap.release()
if not ok:
    raise SystemExit("cannot read frame")

# 透視變換
warped = cv2.warpPerspective(frame, H_out, (out_w, out_h))

# 畫 600 mm 格線
for x in range(0, int(world_w) + 1, 600):
    px = int(x * scale)
    cv2.line(warped, (px, 0), (px, out_h), (0, 255, 0), 1)
for y in range(0, int(world_h) + 1, 600):
    py = int(y * scale)
    cv2.line(warped, (0, py), (out_w, py), (0, 255, 0), 1)

# 畫邊界
cv2.rectangle(warped, (0, 0), (out_w - 1, out_h - 1), (0, 0, 255), 2)

cv2.imwrite("topdown.jpg", warped)
print("saved topdown.jpg")

cv2.imshow("top-down", warped)
cv2.waitKey(0)
cv2.destroyAllWindows()