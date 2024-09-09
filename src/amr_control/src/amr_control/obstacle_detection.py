#!/usr/bin/env python3

import rospy
import numpy as np
import cv2
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray
from dynamic_reconfigure.server import Server
from amr_control.cfg import ObstacleDetectionConfig  # Importiere die .cfg-Datei
from amr_control.visualizer import Visualizer

class ObstacleDetection:
    def __init__(self):
        rospy.init_node('obstacle_detection', anonymous=True)

        # Load parameters from YAML configuration
        self.search_radius = rospy.get_param('trajectory_planner/prediction_distance', 2.0)  # Prediction distance
        self.safety_corner_radius = rospy.get_param('obstacle_detection/safety_corner_radius', 0.1)  # Prediction distance
        self.quality_level = rospy.get_param('obstacle_detection/quality_level', 0.01)  # Prediction distance
        self.max_corners = rospy.get_param('obstacle_detection/max_corners', 20)  # Prediction distance
        self.min_distance = rospy.get_param('obstacle_detection/min_distance', 20)  # Prediction distance
        self.loop_rate = rospy.get_param('obstacle_detection/loop_rate', 1)  # Prediction distance
        self.max_objects = rospy.get_param('nmpc_controller/max_obstacles', 10)  # Maximum number of obstacles
        
        # Initialize ROS publishers and subscribers
        self.lidar_sub = rospy.Subscriber('/scan', LaserScan, self.lidar_callback)
        self.obstacle_pub = rospy.Publisher('/detected_obstacles', Float32MultiArray, queue_size=10)
        self.detected_obstacles = Float32MultiArray()
        self.latest_scan_data = None
        
        self.visualizer = Visualizer()

        # Initialisiere den Dynamic Reconfigure Server
        self.server = Server(ObstacleDetectionConfig, self.dynamic_reconfigure_callback)

    def dynamic_reconfigure_callback(self, config, level):
        """Callback, der aufgerufen wird, wenn der Dynamic Reconfigure Parameter geändert wird."""
        rospy.loginfo(f"Reconfigure Request: Quality Level: {config['qualityLevel']}")
        self.quality_level = config['qualityLevel']
        return config

    def lidar_callback(self, data):
        """Speichert die neuesten Lidar-Daten zur späteren Verarbeitung."""
        self.latest_scan_data = data

    def process_lidar_data(self, data):
        """Verarbeitet die Lidar-Daten und erkennt Hindernisse."""
        lidar_ranges = np.array(data.ranges)
        lidar_angles = np.linspace(data.angle_min, data.angle_max, len(lidar_ranges))
        
        indices = np.where(lidar_ranges < self.search_radius)
        filtered_ranges = lidar_ranges[indices]
        filtered_angles = lidar_angles[indices]

        x_points = filtered_ranges * np.cos(filtered_angles)
        y_points = filtered_ranges * np.sin(filtered_angles)

        lidar_image = self.create_lidar_image(x_points, y_points)
        
        corners = self.detect_corners(lidar_image)

        if corners is not None:
            self.publish_corners_as_circles(corners)


    def create_lidar_image(self, x_points, y_points):
        max_range = int(self.search_radius * 100)
        lidar_image = np.zeros((2 * max_range, 2 * max_range), dtype=np.uint8)

        scaled_x = np.int32((x_points * 100) + max_range)
        scaled_y = np.int32((y_points * 100) + max_range)

        lidar_image[scaled_y, scaled_x] = 255

        blurred_image = cv2.GaussianBlur(lidar_image, (9, 9), 5)
        return blurred_image

    def detect_corners(self, image):
        """Erkennt die markantesten Ecken im Bild mit Shi-Tomasi Corner Detection."""
        # Shi-Tomasi Corner Detection anwenden
        corners = cv2.goodFeaturesToTrack(image, maxCorners=self.max_corners, qualityLevel=self.quality_level, minDistance=self.min_distance)
        
        if corners is not None:
            corners = np.intp(corners)
        return corners

    def publish_corners_as_circles(self, corners):
        """Veröffentlicht erkannte Ecken als Kreise mit festem Durchmesser."""
        obstacles_msg = Float32MultiArray()
        max_range = int(self.search_radius * 100)
        
        obstacles = []
        
        # Begrenze die Anzahl der zu veröffentlichenden Objekte
        corners_to_publish = corners[:self.max_objects]  # Nimm nur die ersten max_objects Ecken

        for corner in corners_to_publish:
            x, y = corner.ravel()
            obstacle_x = (x - max_range) / 100.0  # Umrechnung in Meter
            obstacle_y = (y - max_range) / 100.0  # Umrechnung in Meter
            obstacle_diameter = 2 * self.safety_corner_radius  # Roboter-Durchmesser als Kreis-Durchmesser

            # Füge die Daten des Hindernisses hinzu
            obstacles_msg.data.extend([obstacle_x, obstacle_y, obstacle_diameter])
            
            obstacles.append([obstacle_x, obstacle_y, obstacle_diameter])

        # Visualisiere alle Kreise im Array
        self.visualizer.create_marker_array(obstacles)

        # Veröffentliche die erkannten Ecken als Hindernisse
        self.obstacle_pub.publish(obstacles_msg)
        rospy.loginfo(f"Published {len(corners_to_publish)} corners as circles.")

    def run(self):
        rate = rospy.Rate(self.loop_rate)  # Setze die Veröffentlichungsrate auf 2 Hz
        while not rospy.is_shutdown():
            if self.latest_scan_data:
                # Verarbeite die neuesten Daten
                self.process_lidar_data(self.latest_scan_data)
            self.visualizer.search_radius_marker(self.search_radius)
            rate.sleep()  # Kontrolliere die Publikationsfrequenz


if __name__ == '__main__':
    try:
        obstacle_detector = ObstacleDetection()
        obstacle_detector.run()
    except rospy.ROSInterruptException:
        pass
