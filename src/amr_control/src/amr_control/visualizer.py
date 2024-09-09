#!/usr/bin/env python3
import rospy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PoseArray, Pose
from tf.transformations import euler_from_quaternion
import tf.transformations

class Visualizer:
    def __init__(self):
        # Entferne den zweiten Aufruf von rospy.init_node
        self.predicted_pose_array_pub = rospy.Publisher('/predicted_trajectory', PoseArray, queue_size=10)
        self.refrence_pose_array_pub = rospy.Publisher('/refrence_trajectory', PoseArray, queue_size=10)
        self.marker_publisher = rospy.Publisher('/obstacle_marker_array', MarkerArray, queue_size=10)
        self.search_radius_pub = rospy.Publisher('/search_radius', Marker, queue_size=10)
        
        self.marker_id = 0
        
    def publish_predicted_trajectory(self, predicted_trajectory):
        pose_array = PoseArray()
        pose_array.header.stamp = rospy.Time.now()
        pose_array.header.frame_id = "map"

        for state in predicted_trajectory:
            pose = Pose()
            pose.position.x = state[0]
            pose.position.y = state[1]
            quaternion = tf.transformations.quaternion_from_euler(0, 0, state[2])
            pose.orientation.x = quaternion[0]
            pose.orientation.y = quaternion[1]
            pose.orientation.z = quaternion[2]
            pose.orientation.w = quaternion[3]
            pose_array.poses.append(pose)

        self.predicted_pose_array_pub.publish(pose_array)
        
    def create_marker_array(self, obstacles):
        self.delete_obstacle_markers()
        
        """Erstellt einen MarkerArray für mehrere Hindernisse."""
        marker_array = MarkerArray()

        for obstacle in obstacles:
            marker = Marker()
            marker.header.frame_id = "base_link"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "obstacle_detection"
            marker.id = self.marker_id
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD

            # Setze die Position des Markers
            marker.pose.position.x = obstacle[0]
            marker.pose.position.y = obstacle[1]
            marker.pose.position.z = obstacle[2] / 2.0  # Hindernis-Höhe
            marker.pose.orientation.w = 1.0

            # Setze die Skalierung des Markers (Durchmesser und Höhe)
            marker.scale.x = obstacle[2]
            marker.scale.y = obstacle[2]
            marker.scale.z = 1.0  # Setzt die Höhe

            # Setze die Farbe des Markers
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 0.8

            # Füge den Marker dem MarkerArray hinzu
            marker_array.markers.append(marker)
            self.marker_id += 1

        # Veröffentliche den MarkerArray
        self.marker_publisher.publish(marker_array)

    def delete_obstacle_markers(self):
        """Löscht alle bisher veröffentlichten Hindernis-Marker."""
        marker_array = MarkerArray()

        marker = Marker()
        marker.action = Marker.DELETEALL  # Lösche alle Marker
        marker_array.markers.append(marker)
        
        self.marker_publisher.publish(marker_array)

    def publish_refrence_trajectory(self, refrence_trajectory):
        pose_array = PoseArray()
        pose_array.header.stamp = rospy.Time.now()
        pose_array.header.frame_id = "map"

        for pose in refrence_trajectory:
            new_pose = Pose()

            if isinstance(pose, Pose):
                # Wenn das Element bereits eine Pose ist, direkt übernehmen
                new_pose.position.x = pose.position.x
                new_pose.position.y = pose.position.y

                new_pose.orientation.x = pose.orientation.x
                new_pose.orientation.y = pose.orientation.y
                new_pose.orientation.z = pose.orientation.z
                new_pose.orientation.w = pose.orientation.w

            else:
                # Wenn es sich um eine Liste [x, y, yaw] handelt, in Pose umwandeln
                new_pose.position.x = pose[0]
                new_pose.position.y = pose[1]

                # Konvertiere den yaw-Wert in ein Quaternion
                quaternion = tf.transformations.quaternion_from_euler(0, 0, pose[2])
                new_pose.orientation.x = quaternion[0]
                new_pose.orientation.y = quaternion[1]
                new_pose.orientation.z = quaternion[2]
                new_pose.orientation.w = quaternion[3]

            pose_array.poses.append(new_pose)

        self.refrence_pose_array_pub.publish(pose_array)
        
    def create_search_radius_marker(self, search_radius):
        """Erstellt und veröffentlicht einen Marker für den Suchradius."""
        marker = Marker()
        marker.header.frame_id = "base_link"  # Verankere den Suchradius an der Mitte des Roboters
        marker.header.stamp = rospy.Time.now()
        marker.ns = "search_radius"
        marker.id = 0
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD

        # Positioniere den Marker in der Mitte des Roboters (base_link)
        marker.pose.position.x = 0.0
        marker.pose.position.y = 0.0
        marker.pose.position.z = 0.0
        marker.pose.orientation.w = 1.0

        # Setze die Skalierung des Markers (Durchmesser und Höhe)
        marker.scale.x = search_radius * 2  # Der Durchmesser ist doppelt so groß wie der Radius
        marker.scale.y = search_radius * 2
        marker.scale.z = 0.1  # Höhe des Zylinders (flach)

        # Setze die Farbe des Markers
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0  # Blau
        marker.color.a = 0.5  # Transparenz

        # Veröffentliche den Marker
        self.search_radius_pub.publish(marker)

    def publish_obstacle_marker(self, obstacle):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "obstacle_detection"
        marker.id = 0
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        marker.pose.position.x = obstacle[0]
        marker.pose.position.y = obstacle[1]
        marker.pose.position.z = obstacle[2] / 2.0  # Hindernis-Höhe
        marker.pose.orientation.w = 1.0
        marker.scale.x = obstacle[2]
        marker.scale.y = obstacle[2]
        marker.scale.z = 1.0  # Hindernis-Höhe

        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 0.8

        self.marker_publisher.publish(marker)