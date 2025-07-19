import cv2
import numpy as np

camera_matrix = np.array([[458, 0, 296],
                          [0, 458, 251],
                          [0,   0,   1]], dtype=np.float32)

dist_coeffs = np.array([-0.02572274, 0.01135277, 0.00600643, -0.00230132, 0.14919308], dtype=np.float32)

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
image = cv2.imread('0_5_m.jpg')
if image is None:
    raise ValueError("Image not found. Check the path!")

# === Manually defined 2D image coordinates of the rectangle corners ===
# Clockwise from top left conrer
image_points = np.array([
    [383, 319],
    [872, 280],
    [910, 768],
    [419, 796]
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