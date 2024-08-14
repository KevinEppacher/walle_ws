#!/usr/bin/env python3

# ROS1 imports
import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Twist, Point, PoseWithCovarianceStamped, PoseArray, Pose
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray
from nav_msgs.msg import Odometry, Path
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import tf.transformations
from visualization_msgs.msg import Marker, MarkerArray

# Casadi imports
import casadi as ca
from casadi.tools import *

# Other imports
import numpy as np
import math

from amr_control.visualizer import Visualizer
from amr_control.obstacle import Obstacle

class TrajectoryPlanner:
    def __init__(self, model, controller, initial_state, target_state):
        rospy.init_node('nmpc_node', anonymous=True)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.pose_callback)
        self.cmd_vel_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.global_plan_sub = rospy.Subscriber('/move_base/NavfnROS/plan', Path, self.global_plan_callback)
        self.viz = Visualizer()

        self.model = model
        self.controller = controller
        self.current_state = np.array(initial_state)
        self.target_state = np.array(target_state)
        self.u0 = np.zeros((self.controller.N, 2))
        self.ref_traj = []

        # Obstacle parameters
        self.obs_x = self.controller.obstacle.x
        self.obs_y = self.controller.obstacle.y
        self.obs_diam = self.controller.obstacle.diam
        self.obs_height = 1.0  # Height of the obstacle for visualization

        # Initialize t as an instance variable
        self.t = np.array([rospy.get_time()])  # Use ROS time in ROS environment

        self.timer = rospy.Timer(rospy.Duration(self.controller.T), self.controller_loop)

    def pose_callback(self, msg):
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        yaw = self.get_yaw_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])
        self.current_state = np.array([position.x, position.y, yaw])
        
    def global_plan_callback(self, msg):
        self.ref_traj = self.adjust_waypoint_orientations(msg.poses)

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
        roll, pitch, yaw = euler_from_quaternion(quaternion)
        return yaw

    def controller_loop(self, event):
        if np.linalg.norm(self.current_state - self.target_state, 2) > 1e-1:
            N = self.controller.N
            n_states = self.model.n_states
            n_controls = self.model.n_controls
            current_time = rospy.get_time()
            T = self.controller.T
            ref_traj = self.ref_traj

            v_min = -0.2
            v_max = 0.2
            omega_min = -0.2
            omega_max = 0.2

            args = {
                'lbg': np.concatenate((np.zeros((3 * (N + 1), 1)), np.full((N + 1, 1), -np.inf))),
                'ubg': np.concatenate((np.zeros((3 * (N + 1), 1)), np.full((N + 1, 1), 0))),
                'lbx': np.full((3 * (N + 1) + 2 * N, 1), -ca.inf),
                'ubx': np.full((3 * (N + 1) + 2 * N, 1), ca.inf),
                'p': np.zeros((n_states + N * (n_states + n_controls),))  # Initialize 'p'
            }
            
            # State bounds
            args['lbx'][0:3 * (N + 1):3] = -20
            args['ubx'][0:3 * (N + 1):3] = 20
            args['lbx'][1:3 * (N + 1):3] = -20
            args['ubx'][1:3 * (N + 1):3] = 20
            args['lbx'][2:3 * (N + 1):3] = -ca.inf
            args['ubx'][2:3 * (N + 1):3] = ca.inf

            # Control bounds
            args['lbx'][3 * (N + 1)::2] = v_min
            args['ubx'][3 * (N + 1)::2] = v_max
            args['lbx'][3 * (N + 1) + 1::2] = omega_min
            args['ubx'][3 * (N + 1) + 1::2] = omega_max
            
            args['p'][0:3] = self.current_state  # Initial condition
            
            size_ref_traj = len(ref_traj)
            ref_traj_array = []
                        
            # Set the reference trajectory
            for k in range(N):           
                if k < size_ref_traj:
                    ref_traj_array.append(ref_traj[k].pose)
                    # Use reference trajectory from ref_traj if available
                    x_ref = ref_traj[k].pose.position.x
                    y_ref = ref_traj[k].pose.position.y
                    theta_ref = self.get_yaw_from_quaternion([
                        ref_traj[k].pose.orientation.x,
                        ref_traj[k].pose.orientation.y,
                        ref_traj[k].pose.orientation.z,
                        ref_traj[k].pose.orientation.w
                    ])
                    theta_ref = self.normalize_angle(theta_ref)
                    u_ref = 0.5
                    omega_ref = 0
                else:
                    # Fallback values if k is beyond available ref_traj
                    x_ref = self.current_state[0]
                    y_ref = self.current_state[1]
                    theta_ref = self.current_state[2]
                             
                # if size_ref_traj < N:
                #     # Use reference trajectory from ref_traj if available
                #     x_ref = ref_traj[size_ref_traj].pose.position.x
                #     y_ref = ref_traj[size_ref_traj].pose.position.y
                #     theta_ref = self.get_yaw_from_quaternion([
                #         ref_traj[size_ref_traj].pose.orientation.x,
                #         ref_traj[size_ref_traj].pose.orientation.y,
                #         ref_traj[size_ref_traj].pose.orientation.z,
                #         ref_traj[size_ref_traj].pose.orientation.w
                #     ])
                #     theta_ref = self.normalize_angle(theta_ref)
                #     u_ref = 0
                #     omega_ref = 0
                #     print("ref_traj[size_ref_traj].pose.position.x", ref_traj[size_ref_traj].pose.position.x)
                    

                print("ref theta: ", theta_ref)
                print("current theta: ", self.current_state[2])


                # Set the reference values in p
                args['p'][n_states + k * (n_states + n_controls): n_states + (k + 1) * (n_states + n_controls) - 2] = [x_ref, y_ref, theta_ref]
                args['p'][n_states + (k + 1) * (n_states + n_controls) - 2: n_states + (k + 1) * (n_states + n_controls)] = [u_ref, omega_ref]

            self.viz.publish_refrence_trajectory(ref_traj_array)

            u0 = np.zeros((N, 2))
            X0 = np.tile(self.current_state, (N + 1, 1))

            args['x0'] = np.concatenate((X0.T.flatten(), u0.T.flatten()))

            sol = self.controller.solver(x0=args['x0'], lbx=args['lbx'], ubx=args['ubx'],
                                         lbg=args['lbg'], ubg=args['ubg'], p=args['p'])

            u = np.reshape(sol['x'][3 * (N + 1):].full(), (N, 2))
            
            u = u[0]
            
            self.publish_cmd_vel(u)

            predicted_states = []
            predicted_states.append(np.reshape(sol['x'][:3 * (N + 1)].full(), (N + 1, 3)))

            predicted_states = np.array(predicted_states).squeeze()

            self.viz.publish_predicted_trajectory(predicted_states)
            # rospy.loginfo("Calculating optimal control input")
            # rospy.loginfo("Control input: {}".format(u))

        else:
            u = [0,0]
            self.publish_cmd_vel(u)
            rospy.loginfo("Target reached")
            
        self.viz.publish_obstacle_marker(self.controller.obstacle)
        
    def publish_cmd_vel(self, u):
        cmd_vel_msg = Twist()
        cmd_vel_msg.linear.x = u[0]
        cmd_vel_msg.angular.z = u[1]
        self.cmd_vel_publisher.publish(cmd_vel_msg)

if __name__ == '__main__':
    try:
        planner = TrajectoryPlanner(model=None, controller=None, initial_state=[0, 0, 0], target_state=[1, 1, 0])
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
