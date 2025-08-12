import cv2
import numpy as np
import os

# Calibration settings
chessboard_size = (9, 6)  # Number of inner corners per a chessboard row and column
square_size = 2.4  # Set the size of a square in your defined unit (e.g., meters, centimeters)

# Prepare object points (0,0,0), (1,0,0), ..., (8,5,0)
objp = np.zeros((chessboard_size[0]*chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

# Arrays to store object points and image points
objpoints = []
imgpoints = []

# Start video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FPS, 30)
print("Press SPACE to capture, ESC to quit and calibrate")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret_corners, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret_corners:
        cv2.drawChessboardCorners(frame, chessboard_size, corners, ret_corners)

    cv2.imshow('Calibration', frame)
    key = cv2.waitKey(1)

    if key == 27:  # ESC
        break
    elif key == 32 and ret_corners:  # SPACE
        print("Captured")
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1),
                                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        imgpoints.append(corners2)

cap.release()
cv2.destroyAllWindows()

if len(objpoints) < 10:
    print("Not enough samples for calibration.")
else:
    print("Calibrating...")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    print("\nCalibration Successful:")
    print("Camera Matrix:\n", mtx)
    print("Distortion Coefficients:\n", dist.ravel())

    # Save to file
    np.savez("calibration_data.npz", camera_matrix=mtx, dist_coeffs=dist, rvecs=rvecs, tvecs=tvecs)
    print("\nSaved calibration data to 'calibration_data.npz'")
