

#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import sys

class BezierFromCamera(Node):
    def __init__(self, show_debug: bool = True):
        super().__init__('bezier_from_camera')
        self.bridge = CvBridge()
        self.show_debug = show_debug

        # Sub to Gazebo camera (make sure this matches your camera topic)
        self.image_sub = self.create_subscription(
            Image, "/camera/image", self.image_callback, 10
        )

        # Pub smoothed path
        self.path_pub = self.create_publisher(Path, "/bezier_path", 10)

        self.get_logger().info("BezierFromCamera node started, listening to /camera/image_raw")

    def image_callback(self, msg):
        try:
            # Convert ROS → OpenCV
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f"cv_bridge error: {e}")
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Edge detection
        edges = cv2.Canny(gray, 80, 150)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(contours) < 1:
            self.get_logger().warn("No contours found in image")
            return

        # Pick largest contour (most likely road/wall)
        contour = max(contours, key=cv2.contourArea)
        pts = contour.squeeze()
        if pts.ndim != 2 or pts.shape[0] < 3:
            self.get_logger().warn("Not enough points in contour for Bezier curve")
            return

        # Pick 3 points for quadratic Bezier (start, mid, end of contour)
        P0 = pts[0]
        P1 = pts[len(pts)//2]
        P2 = pts[-1]

        # Generate quadratic Bezier curve
        t = np.linspace(0, 1, 50)
        bez_x = (1-t)**2 * P0[0] + 2*(1-t)*t*P1[0] + t**2 * P2[0]
        bez_y = (1-t)**2 * P0[1] + 2*(1-t)*t*P1[1] + t**2 * P2[1]

        # Publish as Path
        path_msg = Path()
        path_msg.header.frame_id = "camera_link"   # or base_link depending on your TF tree
        for x, y in zip(bez_x, bez_y):
            pose = PoseStamped()
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            path_msg.poses.append(pose)
        self.path_pub.publish(path_msg)

        # Debug visualization
        if self.show_debug:
            debug = frame.copy()
            cv2.drawContours(debug, [contour], -1, (0,255,0), 1)
            for x, y in zip(bez_x.astype(int), bez_y.astype(int)):
                cv2.circle(debug, (x,y), 2, (0,0,255), -1)

            cv2.imshow("Bezier Curve from Camera", debug)
            cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    show_debug = True
    if len(sys.argv) > 1 and sys.argv[1] == "0":
        show_debug = False

    node = BezierFromCamera(show_debug=show_debug)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

