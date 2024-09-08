#!/usr/bin/env python3
import rospy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PoseArray, Pose
from tf.transformations import euler_from_quaternion
import tf.transformations

# from amr_control import Obstacle

class Visualizer:
    def __init__(self):
        rospy.init_node('nmpc_node', anonymous=True)
        self.predicted_pose_array_pub = rospy.Publisher('/predicted_trajectory', PoseArray, queue_size=10)
        self.refrence_pose_array_pub = rospy.Publisher('/refrence_trajectory', PoseArray, queue_size=10)
        self.marker_publisher = rospy.Publisher('/obstacle_marker', Marker, queue_size=10)

        
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
        
    def publish_obstacle_marker(self, obstacle):
        # Obstacle parameters
        self.obs_x = obstacle[0]
        self.obs_y = obstacle[1]
        self.obs_diam = obstacle[2]
        self.obs_height = 1
        
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "obstacle"
        marker.id = 0
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        marker.pose.position.x = self.obs_x
        marker.pose.position.y = self.obs_y
        marker.pose.position.z = self.obs_height / 2.0
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = self.obs_diam
        marker.scale.y = self.obs_diam
        marker.scale.z = self.obs_height
        marker.color.a = 0.8
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        self.marker_publisher.publish(marker)
        
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
                new_pose.position.z = pose.position.z

                new_pose.orientation.x = pose.orientation.x
                new_pose.orientation.y = pose.orientation.y
                new_pose.orientation.z = pose.orientation.z
                new_pose.orientation.w = pose.orientation.w

            else:
                # Wenn es sich um eine Liste [x, y, yaw] handelt, in Pose umwandeln
                new_pose.position.x = pose[0]
                new_pose.position.y = pose[1]
                new_pose.position.z = 0.0  # Standard z-Wert

                # Konvertiere den yaw-Wert in ein Quaternion
                quaternion = tf.transformations.quaternion_from_euler(0, 0, pose[2])
                new_pose.orientation.x = quaternion[0]
                new_pose.orientation.y = quaternion[1]
                new_pose.orientation.z = quaternion[2]
                new_pose.orientation.w = quaternion[3]

            pose_array.poses.append(new_pose)

        self.refrence_pose_array_pub.publish(pose_array)
