import cv2
import numpy as np
from ultralytics import YOLO
import open3d as o3d

# === 1. 相機內參 ===
fx, fy = 1804.693061772938, 1805.035577584851
cx, cy = 726.6635218310454, 545.9205568446745
K = (fx, fy, cx, cy)

# === 2. 載入影像與深度 ===
rgb_path = "pcData/1754706254161_A330MM014030100_0_rgb.png"
depth_path = "pcData/1754706254161_A330MM014030100_0_depth.png"

rgb = cv2.imread(rgb_path)
depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)

if depth.max() > 100:
    depth = depth / 1000.0  # mm -> m

H, W = depth.shape

# === 3. YOLOv8 segmentation ===
model = YOLO("model/best.pt")
results = model(rgb)

# === 4. Depth -> Point Cloud ===
def depth_to_points(depth, K):
    fx, fy, cx, cy = K
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    X = (u - cx) * depth / fx
    Y = (v - cy) * depth / fy
    Z = depth
    return np.stack((X, Y, Z), axis=-1)

points = depth_to_points(depth, K)

# === 5. 建立上色版本 (透明綠色) ===
alpha = 0.4  # 透明度
overlay_color = np.array([0, 255, 0], dtype=np.float32)  # 綠色

colored_rgb = rgb.copy().astype(np.float32)

inst_id = 0
for result in results:
    if result.masks is None:
        continue
    for mask in result.masks.data.cpu().numpy():
        mask = cv2.resize(mask, (W, H))
        mask = mask > 0.5
        # 混合透明綠色
        colored_rgb[mask] = (1 - alpha) * colored_rgb[mask] + alpha * overlay_color
        inst_id += 1

# === 6. 建立完整 point cloud，帶 instance overlay ===
points_flat = points.reshape(-1, 3)
colors_flat = colored_rgb.reshape(-1, 3) / 255.0  # normalize [0,1]

mask_valid = points_flat[:, 2] > 0
points_valid = points_flat[mask_valid]
colors_valid = colors_flat[mask_valid]

cloud = o3d.geometry.PointCloud()
cloud.points = o3d.utility.Vector3dVector(points_valid)
cloud.colors = o3d.utility.Vector3dVector(colors_valid)

# === 7. 存成 .ply ===
o3d.io.write_point_cloud("segmented_green_overlay.ply", cloud)
print(f"Saved segmented_green_overlay.ply, {len(points_valid)} points")
