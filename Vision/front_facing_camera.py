from rclpy.node import Node
from std_msgs.msg import String
import rclpy
import cv2

from camera import ObjectDetector

class FrontVision(Node):
    def __init__(self):
        super().__init__('FrontVision')
        self.publisher_ = self.create_publisher(String, 'shark_seen', 10)

        self.cap = cv2.VideoCapture(0)  # Adjust if using USB camera index or file

        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        self.detector = ObjectDetector("yolo11n.pt")  # a different model for front detection
        self.get_logger().info('Camera opened. Starting detection...')

    def run(self):
        while rclpy.ok():
            ret, frame = self.cap.read()
            print(f"Frame shape: {frame.shape}")
            if not ret:
                self.get_logger().warn("Failed to read frame.")
                continue

            # === Use ObjectDetector to get formatted JSON results ===
            detection_json = self.detector.get_object_names(frame)
            print(f"Detection JSON: {detection_json}")

            # === Publish result ===
            msg = String()
            msg.data = detection_json

            self.publisher_.publish(msg)


def main():
    rclpy.init()
    node = FrontVision()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()