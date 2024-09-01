#!/usr/bin/env python3

# ROS1 imports
import rospy
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped, PoseArray, Pose
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray
from nav_msgs.msg import Odometry, Path
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import tf
from visualization_msgs.msg import Marker, MarkerArray
import time

# Casadi imports
import casadi as ca
from casadi.tools import *

# Other imports
import numpy as np
import math

from amr_control.visualizer import Visualizer
from amr_control.obstacle import Obstacle
from amr_control.controller import nMPC
from amr_control.robot_model import RobotModel

class TrajectoryPlanner:
    def __init__(self):
        rospy.init_node('nmpc_node', anonymous=True)
        self.tf_listener = tf.TransformListener()
        self.cmd_vel_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.global_plan_sub = rospy.Subscriber('/move_base/NavfnROS/plan', Path, self.global_plan_callback)
        self.viz = Visualizer()
        self.T = 0.1
        self.ref_traj = []
        self.timer = rospy.Timer(rospy.Duration(self.T), self.controller_loop)

        # Zusätzliche Variablen für die Zeitmessung
        self.total_time = 0.0
        self.loop_count = 0
        
        self.model = RobotModel()
        self.controller = nMPC(self.model, 3)
        self.current_state = np.array([0.0, 0.0, 0.0])  # Initialisiere mit einer Standardpose

    def get_robot_pose(self):
        try:
            (trans, rot) = self.tf_listener.lookupTransform('/map', '/base_footprint', rospy.Time(0))
            yaw = self.get_yaw_from_quaternion(rot)
            return np.array([trans[0], trans[1], yaw])
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logwarn("Unable to get robot pose from TF")
            return None

    def global_plan_callback(self, msg):
        self.ref_traj = self.adjust_waypoint_orientations(msg.poses)
        size_global_plan = len(msg.poses)
        target_pose = self.ref_traj[size_global_plan-1].pose
        yaw = self.get_yaw_from_quaternion([
            target_pose.orientation.x,
            target_pose.orientation.y,
            target_pose.orientation.z,
            target_pose.orientation.w
        ])
        self.target_state = np.array([target_pose.position.x, target_pose.position.y, yaw])

    def adjust_waypoint_orientations(self, poses):
        if len(poses) < 2:
            return poses  # Wenn es weniger als 2 Punkte gibt, gibt es nichts zu tun

        for i in range(len(poses) - 1):
            # Aktueller und nächster Punkt
            current_pose = poses[i].pose
            next_pose = poses[i + 1].pose

            # Berechne den Yaw-Winkel basierend auf der Richtung zwischen den beiden Punkten
            dx = next_pose.position.x - current_pose.position.x
            dy = next_pose.position.y - current_pose.position.y
            yaw = math.atan2(dy, dx)
            yaw = self.normalize_angle(yaw)
            # Konvertiere den Yaw-Winkel in ein Quaternion
            quaternion = quaternion_from_euler(0, 0, yaw)

            # Aktualisiere die Orientierung des aktuellen Waypoints
            current_pose.orientation.x = quaternion[0]
            current_pose.orientation.y = quaternion[1]
            current_pose.orientation.z = quaternion[2]
            current_pose.orientation.w = quaternion[3]

        return poses
    
    def normalize_angle(self, angle):
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def get_yaw_from_quaternion(self, quaternion):
        norm = math.sqrt(sum([x * x for x in quaternion]))
        quaternion = [x / norm for x in quaternion]
        _, _, yaw = euler_from_quaternion(quaternion)
        return yaw

    def controller_loop(self, event):
        start_time = rospy.Time.now()  # Startzeit mit ROS-Zeitstempel

        # Holen Sie sich die aktuelle Pose des Roboters
        robot_pose = self.get_robot_pose()
        if robot_pose is not None:
            self.current_state = robot_pose

        size_ref_traj = len(self.ref_traj)
        if size_ref_traj > 0:
            self.compute_control_input()
        else:
            rospy.loginfo("No global plan available")

        end_time = rospy.Time.now()  # Endzeit mit ROS-Zeitstempel
        loop_time = (end_time - start_time).to_sec()  # Taktzeit berechnen in Sekunden

        # Aktualisierung der Gesamtzeit und des Schleifenzählers
        self.total_time += loop_time
        self.loop_count += 1

        # Berechnung der kumulierten durchschnittlichen Taktzeit
        average_loop_time = self.total_time / self.loop_count

        # Ausgabe der Taktzeit und der durchschnittlichen Taktzeit
        rospy.loginfo(f"Taktzeit: {loop_time:.4f} Sekunden")
        rospy.loginfo(f"Kumulierte durchschnittliche Taktzeit: {average_loop_time:.4f} Sekunden")

    def compute_control_input(self):
        if np.linalg.norm(self.current_state - self.target_state, 2) > 1e-2:
            obstacles = [
                [3, 0, 0.3],
                [-2, 1, 0.5]
                # [4, 0, 0.2],
                # [-4, 4, 0.7]
            ]
            u = self.controller.solve_mpc(self.current_state, self.ref_traj, self.target_state, obstacles)
            self.publish_cmd_vel(u)
        else:
            u = [0, 0]
            self.publish_cmd_vel(u)
            rospy.loginfo("Target reached")        

    def publish_cmd_vel(self, u):
        cmd_vel_msg = Twist()
        cmd_vel_msg.linear.x = u[0]
        cmd_vel_msg.angular.z = u[1]
        self.cmd_vel_publisher.publish(cmd_vel_msg)

if __name__ == "__main__":
    planner = TrajectoryPlanner()
    rospy.spin()
