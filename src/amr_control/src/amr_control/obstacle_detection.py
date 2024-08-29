#!/usr/bin/env python3

import rospy
import numpy as np
import cv2
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Pose, Point, Quaternion, Vector3

class ObstacleDetection:
    def __init__(self, search_radius=3.0):
        rospy.init_node('obstacle_detection', anonymous=True)
        
        self.search_radius = search_radius
        self.lidar_sub = rospy.Subscriber('/scan', LaserScan, self.lidar_callback)
        self.obstacle_pub = rospy.Publisher('/detected_obstacles', Float32MultiArray, queue_size=10)
        self.marker_pub = rospy.Publisher('/visualization_marker', Marker, queue_size=10)
        self.detected_obstacles = Float32MultiArray()
        self.marker_id = 0

    def lidar_callback(self, data):
        # Begrenzen des Lidar-Suchradius
        lidar_ranges = np.array(data.ranges)
        lidar_angles = np.linspace(data.angle_min, data.angle_max, len(lidar_ranges))
        
        # Filtern der Daten im definierten Suchradius
        indices = np.where(lidar_ranges < self.search_radius)
        filtered_ranges = lidar_ranges[indices]
        filtered_angles = lidar_angles[indices]

        # Konvertieren in kartesische Koordinaten
        x_points = filtered_ranges * np.cos(filtered_angles)
        y_points = filtered_ranges * np.sin(filtered_angles)

        # Erstellen eines Bildes für die Hough-Transformation
        lidar_image = self.create_lidar_image(x_points, y_points)
        
        # Erkennen von Kreisen (Hindernissen)
        circles = self.detect_circles(lidar_image)

        if circles is not None:
            self.publish_obstacles(circles, x_points, y_points)

    def create_lidar_image(self, x_points, y_points):
        """Konvertiert die Lidar-Punkte in ein Bild für die Hough-Transformation."""
        max_range = int(self.search_radius * 100)  # Umwandlung in cm für Bildskalierung
        lidar_image = np.zeros((2 * max_range, 2 * max_range), dtype=np.uint8)

        # Skalierung und Verschiebung der Punkte für die Bilddarstellung
        scaled_x = np.int32((x_points * 100) + max_range)
        scaled_y = np.int32((y_points * 100) + max_range)

        # Zeichnen der Punkte in das Bild
        lidar_image[scaled_y, scaled_x] = 255

        # Gaussian Blur für bessere Hough-Erkennung
        blurred_image = cv2.GaussianBlur(lidar_image, (7, 7), 2)
        return blurred_image

    def detect_circles(self, image):
        """Erkennt Kreise in einem gegebenen Bild."""
        circles = cv2.HoughCircles(
            image, 
            cv2.HOUGH_GRADIENT, 
            dp=1, 
            minDist=20, 
            param1=50, 
            param2=15, 
            minRadius=5, 
            maxRadius=50
        )
        if circles is not None:
            circles = np.uint16(np.around(circles))
        return circles

    def publish_obstacles(self, circles, x_points, y_points):
        """Berechnet die Position und den Durchmesser der erkannten Kreise und veröffentlicht diese."""
        obstacles_msg = Float32MultiArray()
        max_range = int(self.search_radius * 100)

        for circle in circles[0, :]:
            # Extrahiere die Kreisparameter
            center_x, center_y, radius = circle
            obstacle_x = (center_x - max_range) / 100.0
            obstacle_y = (center_y - max_range) / 100.0
            obstacle_diameter = 2 * (radius / 100.0)

            # Füge Hindernisposition und Durchmesser zur Nachricht hinzu
            obstacles_msg.data.extend([obstacle_x, obstacle_y, obstacle_diameter])

            # Erstelle einen Visual Marker für das Hindernis
            self.create_marker(obstacle_x, obstacle_y, obstacle_diameter)

        # Veröffentliche die erkannten Hindernisse
        self.obstacle_pub.publish(obstacles_msg)
        rospy.loginfo(f"Detected obstacles: {obstacles_msg.data}")

    def create_marker(self, x, y, diameter):
        """Erstellt einen Visual Marker für ein erkanntes Hindernis."""
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "obstacle_detection"
        marker.id = self.marker_id
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD

        # Setze die Position des Markers
        marker.pose = Pose()
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = diameter / 2.0  # Setzt die Höhe des Zylinders
        marker.pose.orientation = Quaternion(0, 0, 0, 1)

        # Setze die Skalierung des Markers (Durchmesser und Höhe)
        marker.scale = Vector3(diameter, diameter, diameter)
        
        # Farbe des Markers
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 0.8

        # Lebensdauer des Markers
        marker.lifetime = rospy.Duration(1)  # Bleibt für 1 Sekunde sichtbar

        # Publiziere den Marker
        self.marker_pub.publish(marker)
        self.marker_id += 1

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        obstacle_detector = ObstacleDetection(search_radius=2.0)
        obstacle_detector.run()
    except rospy.ROSInterruptException:
        pass
