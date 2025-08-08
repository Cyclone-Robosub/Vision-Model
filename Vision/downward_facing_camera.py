from ultralytics import YOLO

import cv2 as cv
import numpy as np
import json

# Camera calibration parameters (from calibration data)
K = np.array([[1060.7, 0, 960],
              [0, 1060.7, 540],
              [0, 0, 1]], dtype=np.float32)
D = np.array([0, 0, 0, 0, 0], dtype=np.float32)  # Assuming minimal distortion
Z = 50.0  # Fixed depth in cm


class ObjectDetector:

    # Camera calibration parameters (from calibration data)
    K = np.array([[1060.7, 0, 960],
                [0, 1060.7, 540],
                [0, 0, 1]], dtype=np.float32)
    D = np.array([0, 0, 0, 0, 0], dtype=np.float32)  # Assuming minimal distortion
    
    def __init__(self, model_path="yolo11n.pt"):
        self.model = YOLO(model_path)

    @staticmethod
    def pixel_to_world_coords(u, v, K, D, Z):
        """Convert pixel coordinates to real-world coordinates"""
        pixel = np.array([[u, v]], dtype=np.float32)
        
        # Step 1: Undistort and normalize
        undistorted = cv.undistortPoints(pixel, K, D)  # shape: (1,1,2)
        x_n, y_n = undistorted[0][0]
        
        # Step 2: Scale by depth to get real-world coords
        x = x_n * Z
        y = y_n * Z

        return x, y


    def get_object_center(self, frame, z=50.0):
        results = self.model(frame, conf=0.1, verbose=False, classes=[66]) # use a trained model for keyboard detection
        objects_centers = []
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

                x, y = ObjectDetector.pixel_to_world_coords(center_u, center_v, self.K, self.D, Z)
                objects_centers.append({
                    "class_name": class_name,
                    "confidence": float(confidence),
                    "pixel_coords": {
                        "x": center_u,
                        "y": center_v
                    },
                    "world_coords": {
                        "x": float(x),
                        "y": float(y)
                    }
                })

                # Visualization for debugging
                cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv.putText(frame, f"{class_name} {confidence:.2f}", 
                           (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                # Draw line from center of the frame to center of bounding box
                cv.line(frame, (frame.shape[1]//2, frame.shape[0]//2), (center_u, center_v), (255, 0, 0), 2)
                cv.circle(frame, (center_u, center_v), 5, (0, 0, 255), -1)
                
        # Optional: show frame for debugging
        cv.imshow("Camera", frame)
        cv.waitKey(1)
                
        return json.dumps(objects_centers)
