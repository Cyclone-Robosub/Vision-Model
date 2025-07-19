# 11.5 15.5 81

import cv2
import numpy as np

camera_matrix = np.array([[1060.7, 0, 960],
                          [0, 1060.7, 540],
                          [0,   0,   1]], dtype=np.float32)

dist_coeffs = np.zeros((5, 1))  # Assume no distortion for simplicity

# === Known size of the real-world rectangle (in cm) ===
rectangle_width = 30
rectangle_height = 30

# === 3D real-world coordinates of rectangle corners (Z=0 plane) ===
object_points = np.array([
    [0, 0, 0],
    [rectangle_width, 0, 0],
    [rectangle_width, rectangle_height, 0],
    [0, rectangle_height, 0]
], dtype=np.float32)

# === Load image ===
image = cv2.imread('xy1.jpg')
if image is None:
    raise ValueError("Image not found. Check the path!")

# === Manually defined 2D image coordinates of the rectangle corners ===
# Example format: [top-left, top-right, bottom-right, bottom-left]
image_points = np.array([
    [817, 350],
    [1080, 372],
    [1060, 670],
    [787, 650]
], dtype=np.float32)

# === Estimate pose ===
success, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)
if not success:
    raise RuntimeError("Pose estimation failed.")

# === Draw the axes on the image ===
cv2.drawFrameAxes(image, camera_matrix, dist_coeffs, rvec, tvec, 10)

# === Annotate translation vector ===
x, y, z = tvec.ravel()
cv2.putText(image, f"Position: X={x:.1f}cm Y={y:.1f}cm Z={z:.1f}cm",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# === Draw rectangle outline ===
cv2.polylines(image, [image_points.astype(np.int32).reshape(-1, 1, 2)], isClosed=True, color=(0, 255, 0), thickness=2)

# === Show result ===
cv2.imshow("Pose Estimation", image)
while(True):
    key = cv2.waitKey(1)
    if key == 32:
        cv2.destroyAllWindows()