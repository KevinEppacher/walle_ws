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
        self.pose_array_publisher = rospy.Publisher('/predicted_trajectory', PoseArray, queue_size=10)
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

        self.pose_array_publisher.publish(pose_array)
        
    def publish_obstacle_marker(self, obstacle):
        # Obstacle parameters
        self.obs_x = obstacle.x
        self.obs_y = obstacle.y
        self.obs_diam = obstacle.diam
        self.obs_height = obstacle.height
        
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
    

