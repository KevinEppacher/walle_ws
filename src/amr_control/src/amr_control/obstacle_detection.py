#!/usr/bin/env python3

import rospy
import numpy as np
import cv2
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray
import tf
from tf.transformations import euler_from_quaternion

class ObstacleDetection:
    def __init__(self):
        rospy.init_node('obstacle_detection', anonymous=True)
        self.tf_listener = tf.TransformListener()

        # Load parameters
        self.search_radius = rospy.get_param('trajectory_planner/prediction_distance', 2.0)
        self.safety_corner_diam = rospy.get_param('obstacle_detection/safety_corner_diam', 0.1)
        self.quality_level = rospy.get_param('obstacle_detection/quality_level', 0.01)
        self.max_corners = rospy.get_param('obstacle_detection/max_corners', 20)
        self.min_distance = rospy.get_param('obstacle_detection/min_distance', 20)
        self.loop_rate = rospy.get_param('obstacle_detection/loop_rate', 1)
        self.max_objects = rospy.get_param('nmpc_controller/max_obstacles', 10)
        self.search_angle = rospy.get_param('obstacle_detection/search_angle', 90)  # Der komplette Winkel, z.B. 90° bedeutet +/- 45°

        # Initialize ROS publishers and subscribers
        self.lidar_sub = rospy.Subscriber('/scan', LaserScan, self.lidar_callback)
        self.obstacle_pub = rospy.Publisher('/detected_obstacles', Float32MultiArray, queue_size=10)
        self.latest_scan_data = None

    def lidar_callback(self, data):
        """Speichert die neuesten Lidar-Daten zur späteren Verarbeitung."""
        self.latest_scan_data = data

    def process_lidar_data(self, data):
        """Verarbeitet die Lidar-Daten und erkennt Hindernisse."""
        lidar_ranges = np.array(data.ranges)
        lidar_angles = np.linspace(data.angle_min, data.angle_max, len(lidar_ranges))

        # Berechne den Suchwinkel: +/- der Hälfte des angegebenen Suchwinkels
        search_angle_half = np.radians(self.search_angle / 2)

        # Filter out obstacles within the search radius and within the desired angle ranges
        indices = np.where((lidar_ranges < self.search_radius) & 
                        ((lidar_angles <= search_angle_half) | (lidar_angles >= np.radians(360) - search_angle_half)))

        filtered_ranges = lidar_ranges[indices]
        filtered_angles = lidar_angles[indices]

        # Convert filtered ranges and angles to x, y coordinates
        x_points = filtered_ranges * np.cos(filtered_angles)
        y_points = filtered_ranges * np.sin(filtered_angles)

        # Create an image from the filtered points
        lidar_image = self.create_lidar_image(x_points, y_points)
        
        # Detect corners in the image
        corners = self.detect_corners(lidar_image)

        # Publish corners if any
        if corners is not None:
            self.publish_corners_as_circles(corners)

    def create_lidar_image(self, x_points, y_points):
        max_range = int(self.search_radius * 100)
        lidar_image = np.zeros((2 * max_range, 2 * max_range), dtype=np.uint8)

        scaled_x = np.int32((x_points * 100) + max_range)
        scaled_y = np.int32((y_points * 100) + max_range)

        lidar_image[scaled_y, scaled_x] = 255

        # Apply Gaussian blur to smoothen the image
        blurred_image = cv2.GaussianBlur(lidar_image, (9, 9), 5)
        return blurred_image

    def detect_corners(self, image):
        """Erkennt die markantesten Ecken im Bild mit Shi-Tomasi Corner Detection."""
        corners = cv2.goodFeaturesToTrack(image, maxCorners=self.max_corners, qualityLevel=self.quality_level, minDistance=self.min_distance)
        
        if corners is not None:
            corners = np.intp(corners)
        return corners

    def publish_corners_as_circles(self, corners):
        """Veröffentlicht erkannte Ecken als Kreise mit Koordinaten relativ zum /map-Frame."""
        obstacles_msg = Float32MultiArray()
        max_range = int(self.search_radius * 100)

        obstacles = []
        
        # Hole die Transformation vom Roboter (/base_footprint) zum globalen Frame (/map)
        try:
            (trans, rot) = self.tf_listener.lookupTransform('/map', '/base_footprint', rospy.Time(0))
            robot_x, robot_y = trans[0], trans[1]
            yaw = self.get_yaw_from_quaternion(rot)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logwarn("Unable to get robot pose from TF")
            return

        # Berechne die Entfernung und den Winkel relativ zur Bewegungsrichtung für jede Ecke
        distances = []
        for corner in corners:
            x, y = corner.ravel()
            obstacle_x = (x - max_range) / 100.0  # Convert to meters
            obstacle_y = (y - max_range) / 100.0  # Convert to meters

            # Transform the obstacle position relative to the /map frame
            transformed_x = robot_x + (obstacle_x * np.cos(yaw) - obstacle_y * np.sin(yaw))
            transformed_y = robot_y + (obstacle_x * np.sin(yaw) + obstacle_y * np.cos(yaw))

            # Calculate distance and store
            distance = np.sqrt(obstacle_x ** 2 + obstacle_y ** 2)
            distances.append((distance, transformed_x, transformed_y))

        # Sort obstacles by distance and limit the number of obstacles
        sorted_distances = sorted(distances, key=lambda d: d[0])[:self.max_objects]

        # Prepare obstacle data for publishing
        for _, transformed_x, transformed_y in sorted_distances:
            obstacles_msg.data.extend([transformed_x, transformed_y, self.safety_corner_diam])

        # Publish the detected obstacles
        self.obstacle_pub.publish(obstacles_msg)

    def get_yaw_from_quaternion(self, quaternion):
        """Konvertiert Quaternion in Yaw-Winkel."""
        norm = np.linalg.norm(quaternion)
        quaternion = [x / norm for x in quaternion]
        _, _, yaw = euler_from_quaternion(quaternion)
        return yaw

    def run(self):
        rate = rospy.Rate(self.loop_rate)
        while not rospy.is_shutdown():
            if self.latest_scan_data:
                self.process_lidar_data(self.latest_scan_data)
            rate.sleep()

if __name__ == '__main__':
    try:
        obstacle_detector = ObstacleDetection()
        obstacle_detector.run()
    except rospy.ROSInterruptException:
        pass
