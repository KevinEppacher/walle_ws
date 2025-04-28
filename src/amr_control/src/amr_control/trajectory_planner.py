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
from scipy.interpolate import interp1d

from amr_control.visualizer import Visualizer
from amr_control.obstacle import Obstacle
from amr_control.controller import nMPC
from amr_control.robot_model import RobotModel

class TrajectoryPlanner:
    def __init__(self):
        rospy.init_node('nmpc_node', anonymous=True)
        
        # Load parameters from the YAML configuration
        self.loop_rate = rospy.get_param('trajectory_planner/controller_loop_rate', 100)  # Loop rate in Hz
        self.feed_forward_scaling = rospy.get_param('trajectory_planner/feed_forward_scaling', 0.8)  # Feed-forward scaling
        self.prediction_distance = rospy.get_param('trajectory_planner/prediction_distance', 2.0)  # Prediction distance
        self.goal_tolerance = rospy.get_param('trajectory_planner/goal_tolerance', 0.1)  # Goal tolerance
        self.current_state = np.array(rospy.get_param('nmpc_controller/init_position', [0.0, 0.0, 0.0]))
        
        self.u = [0, 0]
        self.viz = Visualizer()
        
        self.tf_listener = tf.TransformListener()
        self.cmd_vel_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.global_plan_sub = rospy.Subscriber('/move_base/NavfnROS/plan', Path, self.global_plan_callback)
        self.obstacle_sub = rospy.Subscriber('/detected_obstacles', Float32MultiArray, self.obstacle_callback)
        self.ref_traj = []

        # Additional variables for time measurement
        self.total_time = 0.0
        self.loop_count = 0.0
                
        self.model = RobotModel()
        self.controller = nMPC(self.model)
        self.T = 1 / self.loop_rate
        self.timer = rospy.Timer(rospy.Duration(self.T), self.controller_loop)
        
        self.obstacles = []
        self.search_radius_marker = 0

    def get_robot_pose(self):
        try:
            (trans, rot) = self.tf_listener.lookupTransform('/map', '/base_footprint', rospy.Time(0))
            yaw = self.get_yaw_from_quaternion(rot)
            return np.array([trans[0], trans[1], yaw])
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logwarn("Unable to get robot pose from TF")
            return None
        
    def obstacle_callback(self, msg):
        # Extract obstacles from the received message (msg.data is a flat list)
        obstacles = np.array(msg.data).reshape(-1, 3)
        
        # Convert the numpy array into a regular list
        self.obstacles = obstacles.tolist()
    
    def global_plan_callback(self, msg):
        # Check if there are waypoints in the global plan
        if len(msg.poses) == 0:
            rospy.logwarn("Received empty global plan. Skipping.")
            return

        # Extract (x, y, yaw) for each pose in the global plan message
        self.ref_traj = self.adjust_waypoint_orientations(msg.poses)

        # Check if the resulting ref_traj is empty after adjustment
        if len(self.ref_traj) == 0:
            rospy.logwarn("Adjusted reference trajectory is empty. Skipping.")
            return

        # Set the target to the last point in the trajectory
        self.target_state = self.ref_traj[-1]
        
        self.ref_traj, self.T = self.interpolate_trajectory(self.ref_traj, self.prediction_distance, self.controller.N, self.controller.v_max)

    def adjust_waypoint_orientations(self, poses):
        """
        Calculates the orientation (yaw) for each pose based on the direction to the next point.
        
        :param poses: List of poses (geometry_msgs/Pose)
        :return: Array of waypoints [(x, y, yaw)] with calculated orientation
        """
        if len(poses) < 2:
            return np.array([])  # Nothing to do if there are fewer than 2 points

        waypoints_with_yaw = []

        for i in range(len(poses) - 1):
            # Current and next pose
            current_pose = poses[i].pose
            next_pose = poses[i + 1].pose

            # Calculate yaw based on the direction between the two points
            dx = next_pose.position.x - current_pose.position.x
            dy = next_pose.position.y - current_pose.position.y
            yaw = math.atan2(dy, dx)
            yaw = self.normalize_search_radius_marker(yaw)

            # Add the current point (x, y, yaw) to the list
            waypoints_with_yaw.append([current_pose.position.x, current_pose.position.y, yaw])

        # Add the last point with the same orientation as the second last
        last_pose = poses[-1].pose
        last_yaw = waypoints_with_yaw[-1][2]
        waypoints_with_yaw.append([last_pose.position.x, last_pose.position.y, last_yaw])

        return np.array(waypoints_with_yaw)
    
    def interpolate_trajectory(self, ref_traj, total_prediction_distance, N, robot_maximum_speed):
        """
        Interpolates or reduces the given trajectory to exactly N points, including orientation (Yaw).
        Only keeps waypoints within the total prediction distance.
        
        :param ref_traj: list of waypoints (x, y, yaw)
        :param total_prediction_distance: total distance for MPC prediction
        :param N: number of prediction points for MPC
        :param robot_maximum_speed: maximum speed of the robot
        :return: Interpolated or reduced reference trajectory with orientation
        """
        ref_traj_array = np.array(ref_traj)
        if ref_traj_array.ndim == 1:
            raise ValueError("Ref_traj has only one dimension, expected (x, y, yaw) format.")

        ref_points = ref_traj_array[:, :2]
        yaw_search_radius_markers = ref_traj_array[:, 2]
        distances = np.sqrt(np.sum(np.diff(ref_points, axis=0) ** 2, axis=1))
        cumulative_distances = np.insert(np.cumsum(distances), 0, 0)

        within_distance_mask = cumulative_distances <= total_prediction_distance
        ref_points_within_distance = ref_points[within_distance_mask]
        yaw_within_distance = yaw_search_radius_markers[within_distance_mask]
        cumulative_distances_within = cumulative_distances[within_distance_mask]

        if len(ref_points_within_distance) < 2:
            ref_points_within_distance = np.vstack([ref_points_within_distance, ref_points[-1]])
            yaw_within_distance = np.append(yaw_within_distance, yaw_search_radius_markers[-1])
            cumulative_distances_within = np.append(cumulative_distances_within, total_prediction_distance)

        total_ref_distance = cumulative_distances_within[-1]

        prediction_distance = total_prediction_distance / N
        T = prediction_distance / robot_maximum_speed

        T = T * self.feed_forward_scaling

        new_distances = np.linspace(0, total_ref_distance, N)

        interp_x = interp1d(cumulative_distances_within, ref_points_within_distance[:, 0], kind='linear', fill_value="extrapolate")
        interp_y = interp1d(cumulative_distances_within, ref_points_within_distance[:, 1], kind='linear', fill_value="extrapolate")

        new_points_x = interp_x(new_distances)
        new_points_y = interp_y(new_distances)

        interp_yaw = interp1d(cumulative_distances_within, yaw_within_distance, kind='linear', fill_value="extrapolate")
        new_yaw_search_radius_markers = interp_yaw(new_distances)

        new_trajectory = np.column_stack((new_points_x, new_points_y, new_yaw_search_radius_markers))
        
        return new_trajectory, T

    def normalize_search_radius_marker(self, search_radius_marker):
        return (search_radius_marker + math.pi) % (2 * math.pi) - math.pi

    def get_yaw_from_quaternion(self, quaternion):
        norm = math.sqrt(sum([x * x for x in quaternion]))
        quaternion = [x / norm for x in quaternion]
        _, _, yaw = euler_from_quaternion(quaternion)
        return yaw

    def controller_loop(self, event):
        start_time = rospy.Time.now()

        # Get the current pose of the robot
        robot_pose = self.get_robot_pose()
        if robot_pose is not None:
            self.current_state = robot_pose

        size_ref_traj = len(self.ref_traj)
        if size_ref_traj > 0:
            self.compute_control_input()
        else:
            rospy.loginfo("No global plan available")

        end_time = rospy.Time.now()
        loop_time = (end_time - start_time).to_sec()

        if loop_time > 0.1:
            rospy.logwarn(f"Cycle Time: {loop_time:.4f} Seconds")
        else:
            rospy.loginfo(f"Cycle Time: {loop_time:.4f} Seconds")
            
    def compute_control_input(self):
        if np.linalg.norm(self.current_state - self.target_state, 2) > self.goal_tolerance:
            self.viz.create_marker_array(self.obstacles)
            self.u = self.controller.solve_mpc(self.current_state, self.ref_traj, self.target_state, self.T, self.obstacles)
            self.publish_cmd_vel(self.u)
        else:
            self.u = [0, 0]
            self.publish_cmd_vel(self.u)
            rospy.loginfo("Target reached")        

    def publish_cmd_vel(self, u):
        cmd_vel_msg = Twist()
        cmd_vel_msg.linear.x = u[0]
        cmd_vel_msg.angular.z = u[1]
        self.cmd_vel_publisher.publish(cmd_vel_msg)

if __name__ == "__main__":
    planner = TrajectoryPlanner()
    rospy.spin()
