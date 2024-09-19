#!/usr/bin/env python3

import rospy
import numpy as np
import cv2
from nav_msgs.msg import OccupancyGrid

class ObstacleAvoidance:
    def __init__(self):
        rospy.init_node('costmap_obstacle_avoidance', anonymous=True)

        # Subscriber für die local costmap
        self.local_costmap_sub = rospy.Subscriber('/move_base/local_costmap/costmap', OccupancyGrid, self.costmap_callback)
        self.latest_costmap_data = None  # Speichert die neuesten Costmap-Daten

        # Skalierungsfaktor für das Bild
        self.scale_factor = 10  # Erhöhe den Skalierungsfaktor für größere Darstellung

        print("Costmap Obstacle Avoidance Node initialized.")

    def costmap_callback(self, data):
        """Callback-Funktion, die die neuesten Costmap-Daten speichert."""
        self.latest_costmap_data = data

    def process_costmap(self):
        """Verarbeitet die Costmap-Daten, konvertiert sie in ein OpenCV-Bild und extrahiert Kreise."""
        if self.latest_costmap_data is None:
            return None

        # Extrahiere Costmap-Daten
        costmap_array = np.array(self.latest_costmap_data.data).reshape((self.latest_costmap_data.info.height, self.latest_costmap_data.info.width))

        # Filtern wir die "blauen" Bereiche heraus. Angenommen, Blau wird durch Zellenwerte zwischen 50 und 70 dargestellt.
        # Du kannst diesen Wertebereich anpassen, um genau das "Blau" zu repräsentieren, das du in RViz siehst.
        blue_mask = np.logical_and(costmap_array >= 50, costmap_array <= 70)

        # Konvertiere die gefilterte Costmap in ein Bild
        costmap_image = np.zeros(costmap_array.shape, dtype=np.uint8)
        costmap_image[blue_mask] = 255  # Die blauen Zellen werden weiß dargestellt

        # Verwende findContours, um die äußersten Konturen der "blauen" Bereiche zu extrahieren
        contours, _ = cv2.findContours(costmap_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Erstelle ein leeres Bild zum Zeichnen der Konturen
        contour_image = np.zeros_like(costmap_image)

        # Zeichne die Konturen auf das Bild
        cv2.drawContours(contour_image, contours, -1, (255, 255, 255), 2)

        # Verwende HoughCircles, um Kreise in den Konturen zu erkennen
        circles = cv2.HoughCircles(contour_image, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20, param1=50, param2=10, minRadius=1, maxRadius=25)

        # Skaliere das Bild für eine bessere Sichtbarkeit
        contour_image_resized = cv2.resize(contour_image, None, fx=self.scale_factor, fy=self.scale_factor, interpolation=cv2.INTER_NEAREST)

        # Zeichne die erkannten Kreise auf das Bild
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                # Zeichne den Kreis in rot (BGR: (0, 0, 255))
                cv2.circle(contour_image_resized, (x * self.scale_factor, y * self.scale_factor), r * self.scale_factor, (255, 255, 255), 2)
                # Optional: Zeichne das Zentrum des Kreises in rot
                cv2.circle(contour_image_resized, (x * self.scale_factor, y * self.scale_factor), 2, (255, 255, 255), 3)
            rospy.loginfo(f"Detected {len(circles)} circles.")

        return contour_image_resized


    def run(self):
        """Hauptschleife, die das OpenCV-Bild anzeigt."""
        rate = rospy.Rate(10)  # 10 Hz
        while not rospy.is_shutdown():
            image = self.process_costmap()
            if image is not None:
                # Zeige das Bild mit den Kreisen und Konturen an
                print("Showing costmap contours with detected circles...")
                print("Press any key to close the window.")
                cv2.imshow("Costmap Contours and Circles", image)
                cv2.waitKey(1)  # 1 ms Verzögerung

            rate.sleep()

if __name__ == '__main__':
    try:
        obstacle_avoidance = ObstacleAvoidance()
        obstacle_avoidance.run()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()
