import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import cv2

from ultralytics import YOLO

model = YOLO("last.pt")

class ObjectDetectionNode(Node):
    def __init__(self):
        super().__init__('object_detector')
        self.publisher_ = self.create_publisher(String, 'detections', 10)
        self.cap = cv2.VideoCapture(0)  # Adjust if using USB camera index or file
        self.get_logger().info('Camera opened. Starting detection...')

    def run(self):
        while rclpy.ok():
            ret, frame = self.cap.read()
            if not ret:
                self.get_logger().warn("Failed to read frame.")
                continue

            # === Your detection logic here ===
            ok, detection_result = self.detect_white_bin(frame)

            # === Publish result ===
            msg = String()
            msg.data = detection_result

            
            self.publisher_.publish(msg)

            # Optional: show frame for debugging
            cv2.imshow("Camera", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def detect_white_bin(self, frame):
        """Example: white bin in image"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model.predict(rgb_frame, conf=0.7)
        if len(results[0].boxes) > 0:
            # draw bounding boxes
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f'{box.cls[0]} {box.conf[0]:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            return True, "White bin detected"
        return False, "No white bin detected"


def main():
    rclpy.init()
    node = ObjectDetectionNode()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()