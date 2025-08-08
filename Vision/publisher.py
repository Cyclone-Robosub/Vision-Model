import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import cv2

from downward_facing_camera import ObjectDetector

class ObjectDetectionNode(Node):
    def __init__(self):
        super().__init__('object_detector')
        self.publisher_ = self.create_publisher(String, 'detections', 10)
        self.cap = cv2.VideoCapture(0)  # Adjust if using USB camera index or file
        self.detector = ObjectDetector("yolo11n.pt")  # Initialize ObjectDetector
        self.fixed_z = 50.0  # Fixed depth in cm, can be updated from other sources later
        self.get_logger().info('Camera opened. Starting detection...')

    def run(self):
        while rclpy.ok():
            ret, frame = self.cap.read()
            if not ret:
                self.get_logger().warn("Failed to read frame.")
                continue

            # === Use ObjectDetector to get object centers ===
            objects_centers = self.detector.get_object_center(frame, z=self.fixed_z)

            # === Publish result ===
            msg = String()
            if objects_centers:
                # Format the detection results as a string
                detection_data = []
                for obj in objects_centers:
                    detection_data.append(
                        f"Class: {obj['class_name']}, "
                        f"Confidence: {obj['confidence']:.2f}, "
                        f"Pixel: ({obj['pixel_coords'][0]}, {obj['pixel_coords'][1]}), "
                        f"World: ({obj['world_coords'][0]:.2f}, {obj['world_coords'][1]:.2f}, {obj['world_coords'][2]:.2f})"
                    )
                msg.data = f"Objects detected: {len(objects_centers)} | " + " | ".join(detection_data)
            else:
                msg.data = "No objects detected"

            self.publisher_.publish(msg)

            # Frame display is handled in get_object_center method
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def update_depth(self, new_z):
        """Update the fixed depth value for coordinate conversion"""
        self.fixed_z = new_z
        self.get_logger().info(f'Updated depth to: {new_z} cm')


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