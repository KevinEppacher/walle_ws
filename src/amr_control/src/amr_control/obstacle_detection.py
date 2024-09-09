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
    def __init__(self, search_radius=2.0):
        rospy.init_node('obstacle_detection', anonymous=True)

        self.search_radius = search_radius
        self.robot_radius = 0.1  # Roboter-Durchmesser wird als Kreis-Durchmesser verwendet
        self.max_objects = 10  # Maximale Anzahl von Kreisen (Ecken), die veröffentlicht werden
        self.lidar_sub = rospy.Subscriber('/scan', LaserScan, self.lidar_callback)
        self.obstacle_pub = rospy.Publisher('/detected_obstacles', Float32MultiArray, queue_size=10)
        self.detected_obstacles = Float32MultiArray()
        
        self.visualizer = Visualizer()

        # Initialisiere den Dynamic Reconfigure Server
        self.server = Server(ObstacleDetectionConfig, self.dynamic_reconfigure_callback)

        # Initiale Parameter
        self.quality_level = 0.95  # Standardwert für Quality Level

    def dynamic_reconfigure_callback(self, config, level):
        """Callback, der aufgerufen wird, wenn der Dynamic Reconfigure Parameter geändert wird."""
        rospy.loginfo(f"Reconfigure Request: Quality Level: {config['qualityLevel']}")
        self.quality_level = config['qualityLevel']
        return config

    def lidar_callback(self, data):
        # Die Callback-Funktion verarbeitet die neuesten Nachrichten direkt
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
            self.display_corners_on_image(lidar_image, corners)
            self.publish_corners_as_circles(corners)

        cv2.imshow("Lidar Image with Corners", lidar_image)
        cv2.waitKey(1)


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
        max_corners = 20  # Maximal erkennbare Ecken
        min_distance = 20  # Mindestabstand zwischen den erkannten Ecken

        # Shi-Tomasi Corner Detection anwenden
        corners = cv2.goodFeaturesToTrack(image, maxCorners=max_corners, qualityLevel=self.quality_level, minDistance=min_distance)
        
        if corners is not None:
            corners = np.int0(corners)
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
            obstacle_diameter = 2 * self.robot_radius  # Roboter-Durchmesser als Kreis-Durchmesser

            # Füge die Daten des Hindernisses hinzu
            obstacles_msg.data.extend([obstacle_x, obstacle_y, obstacle_diameter])
            
            obstacles.append([obstacle_x, obstacle_y, obstacle_diameter])

        # Visualisiere alle Kreise im Array
        self.visualizer.create_marker_array(obstacles)

        # Veröffentliche die erkannten Ecken als Hindernisse
        self.obstacle_pub.publish(obstacles_msg)
        rospy.loginfo(f"Published {len(corners_to_publish)} corners as circles.")

    def display_corners_on_image(self, image, corners):
        """Zeichnet die erkannten Ecken als Kreise auf das Bild."""
        # Konvertiere das Bild in Farbe, um die Ecken hervorzuheben
        image_with_corners = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        for corner in corners:
            x, y = corner.ravel()
            cv2.circle(image_with_corners, (x, y), 5, (0, 255, 0), -1)  # Zeichne grüne Kreise auf die Ecken

        # Zeige das Bild mit den markanten Ecken an
        cv2.imshow("Detected Corners", image_with_corners)

    def run(self):
        rospy.spin()  # spin_once existiert nicht in ROS, nutze spin



if __name__ == '__main__':
    try:
        obstacle_detector = ObstacleDetection(search_radius=2.0)
        obstacle_detector.run()
    except rospy.ROSInterruptException:
        pass
