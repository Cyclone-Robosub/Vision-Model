import cv2
import numpy as np

# === Replace with your real camera matrix and distortion ===
# Example values for 640x480 webcam
camera_matrix = np.array([[458, 0, 296],
                          [0, 458, 251],
                          [0,   0,   1]], dtype=np.float32)

dist_coeffs = np.array([-0.02572274, 0.01135277, 0.00600643, -0.00230132, 0.14919308], dtype=np.float32)
 
# Known real-world size of the rectangle (in cm)
rectangle_width = 23.5
rectangle_height = 36

# 3D object points in real-world (origin at top-left corner)
object_points = np.array([
    [0, 0, 0],
    [rectangle_width, 0, 0],
    [rectangle_width, rectangle_height, 0],
    [0, rectangle_height, 0]
], dtype=np.float32)

# Open camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to gray and blur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edged = cv2.Canny(blurred, 20, 100)

    out_frame = cv2.cvtColor(edged, cv2.COLOR_GRAY2BGR)

    # Find contours
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        if len(approx) == 4 and cv2.isContourConvex(approx):
            area = cv2.contourArea(approx)
            if area > 1000:
                approx = approx.reshape(4, 2)

                # Sort points in consistent order: top-left, top-right, bottom-right, bottom-left
                def order_points(pts):
                    rect = np.zeros((4, 2), dtype="float32")
                    s = pts.sum(axis=1)
                    rect[0] = pts[np.argmin(s)]
                    rect[2] = pts[np.argmax(s)]
                    diff = np.diff(pts, axis=1)
                    rect[1] = pts[np.argmin(diff)]
                    rect[3] = pts[np.argmax(diff)]
                    return rect

                image_points = order_points(approx).astype(np.float32)

                # Estimate pose
                success, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)
                if success:
                    # Draw axes
                    cv2.drawFrameAxes(out_frame, camera_matrix, dist_coeffs, rvec, tvec, 5)

                    # Show translation vector (position)
                    x, y, z = tvec.ravel()
                    cv2.putText(out_frame, f"Position: X={x:.1f}cm Y={y:.1f}cm Z={z:.1f}cm",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

                    # Draw rectangle
                    cv2.polylines(out_frame, [approx.reshape(-1, 1, 2)], isClosed=True, color=(0, 255, 0), thickness=2)

    cv2.imshow("3D Pose Estimation", out_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
