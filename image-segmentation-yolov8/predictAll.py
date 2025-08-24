from ultralytics import YOLO
import cv2
import numpy as np

model_path = '/home/gyrobot/code/segment/image-segmentation-yolov8/model/best.pt'
image_path = '/home/gyrobot/code/segment/image-segmentation-yolov8/test/1754706254161_A330MM014030100_0_rgb.png'

# Load image
img = cv2.imread(image_path)
H, W = img.shape[:2]

# Inference
model = YOLO(model_path)
results = model(img)

# Prepare a single color layer for all masks (green in BGR)
green = (0, 255, 0)
alpha = 0.2  # transparency for masks
mask_layer = np.zeros_like(img, dtype=np.uint8)  # where we paint all masks

# We will draw numbers on this layer so they don't get double-faded
label_layer = img.copy()

mask_idx = 0
for result in results:
    if result.masks is None:
        continue

    for m in result.masks.data:
        mask_idx += 1
        m = m.detach().cpu().numpy()         # 0/1 float mask
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)

        # Paint green on mask_layer where mask is True
        mask_bool = m > 0.5
        mask_layer[mask_bool] = green

        # Compute centroid and number it
        ys, xs = np.where(mask_bool)
        if xs.size > 0:
            cx, cy = int(xs.mean()), int(ys.mean())
            # Put a small black outline for readability
            cv2.putText(label_layer, str(mask_idx), (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(label_layer, str(mask_idx), (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)  # red number

# Blend once: original + green masks
overlay = cv2.addWeighted(mask_layer, alpha, img, 1 - alpha, 0)

# Now draw the numbers on top (not faded)
final = overlay.copy()
# Copy only number pixels from label_layer where they differ (simple way: just re-draw)
# (we already drew text on label_layer; draw again onto final)
mask_idx = 0
for result in results:
    if result.masks is None:
        continue
    for m in result.masks.data:
        mask_idx += 1
        m = m.detach().cpu().numpy()
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        ys, xs = np.where(m > 0.5)
        if xs.size > 0:
            cx, cy = int(xs.mean()), int(ys.mean())
            cv2.putText(final, str(mask_idx), (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(final, str(mask_idx), (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

cv2.imwrite('./output_overlay_numbered_green.png', final)
print("Saved ./output_overlay_numbered_green.png")
