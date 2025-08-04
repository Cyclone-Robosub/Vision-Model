from ultralytics import YOLO
import cv2 as cv
import numpy as np

model = YOLO("yolo11n.pt")

# Camera calibration parameters (from calibration data)
K = np.array([[1060.7, 0, 960],
              [0, 1060.7, 540],
              [0, 0, 1]], dtype=np.float32)
D = np.array([0, 0, 0, 0, 0], dtype=np.float32)  # Assuming minimal distortion
Z = 50.0  # Fixed depth in cm

def pixel_to_world_coords(u, v, K, D, Z):
    """Convert pixel coordinates to real-world coordinates"""
    pixel = np.array([[u, v]], dtype=np.float32)
    
    # Step 1: Undistort and normalize
    undistorted = cv.undistortPoints(pixel, K, D)  # shape: (1,1,2)
    x_n, y_n = undistorted[0][0]
    
    # Step 2: Scale by depth to get real-world coords
    X = x_n * Z
    Y = y_n * Z
    
    return X, Y, Z

cap = cv.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.1, verbose=False, classes=[66]) # use a trained model for keyboard detection
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy()
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box[:4])
            confidence = confidences[i]
            class_id = int(class_ids[i])
            class_name = result.names[class_id]
            
            # Calculate center of bounding box
            center_u = (x1 + x2) // 2
            center_v = (y1 + y2) // 2
            
            # Convert to real-world coordinates if it's a keyboard
            if class_name.lower() == 'keyboard':
                X, Y, Z_coord = pixel_to_world_coords(center_u, center_v, K, D, Z)
                print(f"Keyboard center - Pixel: ({center_u}, {center_v}), "
                      f"World: X={X:.2f}cm, Y={Y:.2f}cm, Z={Z_coord:.2f}cm")
            
            cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv.putText(frame, f"{class_name} {confidence:.2f}", 
                       (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Draw line from center of the frame to center of bounding box
            cv.line(frame, (frame.shape[1]//2, frame.shape[0]//2), (center_u, center_v), (255, 0, 0), 2)
            
            # Draw center point for keyboards
            if class_name.lower() == 'keyboard':
                cv.circle(frame, (center_u, center_v), 5, (0, 0, 255), -1)

    cv.imshow("YOLO Detection", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()