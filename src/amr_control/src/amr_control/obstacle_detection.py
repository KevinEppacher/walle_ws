#!/usr/bin/env python3

import rospy
import numpy as np
import cv2
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Float32MultiArray
from dynamic_reconfigure.server import Server
from amr_control.cfg import ObstacleDetectionConfig  # Importiere die .cfg-Datei
from amr_control.visualizer import Visualizer
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import tf

class ObstacleDetection:
    def __init__(self):
        rospy.init_node('obstacle_detection', anonymous=True)
        self.tf_listener = tf.TransformListener()

        # Load parameters from YAML configuration
        self.search_radius = rospy.get_param('trajectory_planner/prediction_distance', 2.0)
        self.safety_corner_diam = rospy.get_param('obstacle_detection/safety_corner_diam', 0.1)
        self.quality_level = rospy.get_param('obstacle_detection/quality_level', 0.01)
        self.max_corners = rospy.get_param('obstacle_detection/max_corners', 20)
        self.min_distance = rospy.get_param('obstacle_detection/min_distance', 20)
        self.loop_rate = rospy.get_param('obstacle_detection/loop_rate', 1)
        self.max_objects = rospy.get_param('nmpc_controller/max_obstacles', 10)
        self.search_angle = rospy.get_param('obstacle_detection/search_angle', 90)

        # Initialize ROS publishers and subscribers
        self.local_costmap_sub = rospy.Subscriber('/move_base/local_costmap/costmap', OccupancyGrid, self.costmap_callback)
        self.obstacle_pub = rospy.Publisher('/detected_obstacles', Float32MultiArray, queue_size=10)
        self.detected_obstacles = Float32MultiArray()
        self.latest_costmap_data = None
        print("Obstacle detection node initialized.")

        self.visualizer = Visualizer()

    def costmap_callback(self, data):
        """Speichert die neuesten Costmap-Daten zur späteren Verarbeitung."""
        self.latest_costmap_data = data

    def process_costmap_data(self, data):
        """Verarbeitet die Costmap-Daten und erkennt Kreise (Objekte)."""
        costmap_array = np.array(data.data).reshape((data.info.height, data.info.width))
        resolution = data.info.resolution
        origin_x = data.info.origin.position.x
        origin_y = data.info.origin.position.y

        # Konvertiere die Costmap in ein Bild
        costmap_image = np.zeros(costmap_array.shape, dtype=np.uint8)
        costmap_image[costmap_array == 100] = 255  # Belege belegte Zellen mit Weiß (255)

        blurred_image = cv2.GaussianBlur(costmap_image, (9, 9), 2)
        
        blurred_image = cv2.resize(blurred_image, None, fx=5, fy=5, interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Costmap", blurred_image)
        cv2.waitKey(1)

        # Erkenne Kreise in der Costmap (Rundungen der Objekte)
        circles = cv2.HoughCircles(blurred_image, cv2.HOUGH_GRADIENT, dp=1.2, minDist=self.min_distance,
                                   param1=50, param2=30, minRadius=5, maxRadius=50)

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            self.publish_circles_as_obstacles(circles, origin_x, origin_y, resolution)

    def publish_circles_as_obstacles(self, circles, origin_x, origin_y, resolution):
        """Veröffentlicht erkannte Kreise als Hindernisse."""
        obstacles_msg = Float32MultiArray()

        obstacles = []

        # Hole die Transformation vom Roboter (/base_footprint) zum globalen Frame (/map)
        try:
            (trans, rot) = self.tf_listener.lookupTransform('/map', '/base_footprint', rospy.Time(0))
            robot_x, robot_y = trans[0], trans[1]
            yaw = self.get_yaw_from_quaternion(rot)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logwarn("Unable to get robot pose from TF")
            return

        distances = []
        search_angle_rad = np.radians(self.search_angle / 2)

        for circle in circles:
            center_x = circle[0]
            center_y = circle[1]
            radius = circle[2] * resolution

            # Konvertiere die Kreiskoordinaten in den globalen Frame
            obstacle_x = origin_x + center_x * resolution
            obstacle_y = origin_y + center_y * resolution

            transformed_x = robot_x + (obstacle_x * np.cos(yaw) - obstacle_y * np.sin(yaw))
            transformed_y = robot_y + (obstacle_x * np.sin(yaw) + obstacle_y * np.cos(yaw))

            # Berechne den Winkel relativ zur Bewegungsrichtung des Roboters
            angle_to_obstacle = np.arctan2(obstacle_y, obstacle_x)

            # Filter: nur Hindernisse innerhalb des Suchkegels berücksichtigen
            if abs(angle_to_obstacle) <= search_angle_rad:
                distance = np.sqrt(obstacle_x ** 2 + obstacle_y ** 2)
                distances.append((distance, transformed_x, transformed_y, radius))

        sorted_distances = sorted(distances, key=lambda d: d[0])
        sorted_distances = sorted_distances[:self.max_objects]

        for _, transformed_x, transformed_y, radius in sorted_distances:
            obstacles_msg.data.extend([transformed_x, transformed_y, radius])
            obstacles.append([transformed_x, transformed_y, radius])

        # Veröffentliche die erkannten Kreise
        self.obstacle_pub.publish(obstacles_msg)
        
        print(f"Detected obstacles: {obstacles}")
        rospy.loginfo(f"Published {len(sorted_distances)} nearest circles within search cone.")

    def get_yaw_from_quaternion(self, quaternion):
        """Konvertiert Quaternion in Yaw-Winkel."""
        norm = np.linalg.norm(quaternion)
        quaternion = [x / norm for x in quaternion]
        _, _, yaw = euler_from_quaternion(quaternion)
        return yaw

    def run(self):
        rate = rospy.Rate(self.loop_rate)
        while not rospy.is_shutdown():
            if self.latest_costmap_data:
                print("Processing costmap data...")
                self.process_costmap_data(self.latest_costmap_data)
            rate.sleep()

if __name__ == '__main__':
    try:
        print("Starting obstacle detection node...")
        obstacle_detector = ObstacleDetection()
        obstacle_detector.run()
    except rospy.ROSInterruptException:
        pass
