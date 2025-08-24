import cv2
import numpy as np
from ultralytics import YOLO
import open3d as o3d  # 用來存/顯示點雲

# === 1. 相機內參 (要換成你自己 RealSense 的數值) ===
fx, fy = 1804.693061772938, 1805.035577584851
cx, cy = 726.6635218310454, 545.9205568446745
K = (fx, fy, cx, cy)

# === 2. 載入影像與深度 ===
rgb_path = "pcData/1754706254161_A330MM014030100_0_rgb.png"
depth_path = "pcData/1754706254161_A330MM014030100_0_depth.png"  # 16bit PNG (單位 mm 或 m，看相機API)

rgb = cv2.imread(rgb_path)
depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)

# 如果深度單位是 mm → 轉成 m
if depth.max() > 100:
    depth = depth / 1000.0

H, W = depth.shape

# === 3. YOLOv8 segmentation 推論 ===
model = YOLO("model/best.pt")
results = model(rgb)

# === 4. Depth → 3D point cloud ===
def depth_to_points(depth, K):
    fx, fy, cx, cy = K
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    X = (u - cx) * depth / fx
    Y = (v - cy) * depth / fy
    Z = depth
    return np.stack((X, Y, Z), axis=-1)  # H x W x 3

points = depth_to_points(depth, K)  # shape [H, W, 3]

# === 5. 依 mask 裁切 instance 點雲 ===
instance_id = 0
for result in results:
    if result.masks is None:
        continue

    for mask in result.masks.data.cpu().numpy():
        mask = cv2.resize(mask, (W, H))  # YOLO mask resize 到 depth 大小
        mask = mask > 0.5  # binarize

        masked_points = points[mask]
        masked_colors = rgb[mask]

        # === 6. 存成 PCD (用 Open3D) ===
        ply = o3d.geometry.PointCloud()
        ply.points = o3d.utility.Vector3dVector(masked_points)
        ply.colors = o3d.utility.Vector3dVector(masked_colors.astype(np.float32) / 255.0)

        filename = f"instance_{instance_id}.ply"
        o3d.io.write_point_cloud(filename, ply)
        print(f"Saved {filename}, {len(masked_points)} points")

        instance_id += 1
