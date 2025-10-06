
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

class RoadMarker(Node):
    def __init__(self, param=0):
        super().__init__('road_marker_node')
        self.image_sub_name = "/camera/image_marker"
        self.show = int(param)
        self.bridge = CvBridge()

        # Subscriber to camera feed
        self.image_sub = self.create_subscription(
            Image,
            self.image_sub_name,
            self.callback,
            10
        )

        # Publisher for processed images
        self.marker_pub = self.create_publisher(Image, "/camera/markers", 10)

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            self.get_logger().error(str(e))
            return

        h, w, _ = cv_image.shape

        # Example: dotted yellow centerline
        for y in range(50, h, 40):
            cv2.line(cv_image, (w // 2, y), (w // 2, y + 10), (0, 255, 255), 3)

        # Example: blue lane lines
        cv2.line(cv_image, (w // 3, h), (w // 3, h // 2), (255, 0, 0), 2)
        cv2.line(cv_image, (2 * w // 3, h), (2 * w // 3, h // 2), (255, 0, 0), 2)

        if self.show == 1:
            cv2.imshow("Road Markers", cv_image)
            cv2.waitKey(10)

        # Publish modified image
        out_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
        self.marker_pub.publish(out_msg)

def main(args=None):
    rclpy.init(args=args)
    param = 0
    if len(args) > 1:
        param = args[1]
    node = RoadMarker(param)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
