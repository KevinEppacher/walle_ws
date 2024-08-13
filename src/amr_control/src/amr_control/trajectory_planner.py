#!/usr/bin/env python3

# ROS1 imports
import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Twist, Point, PoseWithCovarianceStamped, PoseArray, Pose
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray  # Assuming you need multiple floats for position
from nav_msgs.msg import Odometry, Path  # Import Odometry message type
from tf.transformations import euler_from_quaternion
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

        # Initialize `t` as an instance variable
        self.t = np.array([rospy.get_time()])  # Use ROS time in ROS environment

        self.timer = rospy.Timer(rospy.Duration(self.controller.T), self.controller_loop)

    def pose_callback(self, msg):
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        yaw = self.get_yaw_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])
        self.current_state = np.array([position.x, position.y, yaw])
        
    def global_plan_callback(self, msg):
        self.ref_traj = msg.poses

    def get_yaw_from_quaternion(self, quaternion):
        norm = math.sqrt(sum([x * x for x in quaternion]))
        quaternion = [x / norm for x in quaternion]
        roll, pitch, yaw = euler_from_quaternion(quaternion)
        return yaw

    def controller_loop(self, event):
        if np.linalg.norm(self.current_state - self.target_state, 2) > 1e-1:
            args = self.get_args()

            # Solve the optimization problem
            sol = self.controller.solver(x0=args['x0'], lbx=args['lbx'], ubx=args['ubx'],
                                         lbg=args['lbg'], ubg=args['ubg'], p=args['p'])

            u = np.reshape(sol['x'][3 * (self.controller.N + 1):].full(), (self.controller.N, 2))
            u = u[0]

            self.publish_cmd_vel(u)

            predicted_states = np.reshape(sol['x'][:3 * (self.controller.N + 1)].full(), (self.controller.N + 1, 3))
            predicted_states = np.array(predicted_states).squeeze()

            self.viz.publish_predicted_trajectory(predicted_states)
        else:
            u = [0, 0]
            self.publish_cmd_vel(u)
            
        self.viz.publish_obstacle_marker(self.controller.obstacle)

    def get_args(self):
        """Prepare the arguments for the NMPC solver."""
        N = self.controller.N

        state_bounds = self.get_state_bounds(N)
        control_bounds = self.get_control_bounds(N)
        general_bounds = self.get_general_bounds(N)
        parameters = self.get_parameters()
        initial_guess = self.get_initial_guess(N)

        # Combine all parts into args
        args = {**state_bounds, **control_bounds, **general_bounds, **parameters, **initial_guess}
        return args

    def get_state_bounds(self, N):
        """Define and return the state bounds."""
        lbx = np.full((3 * (N + 1) + 2 * N, 1), -ca.inf)
        ubx = np.full((3 * (N + 1) + 2 * N, 1), ca.inf)
        
        # Set state bounds for x, y, and theta
        lbx[0:3 * (N + 1):3] = -20
        ubx[0:3 * (N + 1):3] = 20
        lbx[1:3 * (N + 1):3] = -20
        ubx[1:3 * (N + 1):3] = 20
        lbx[2:3 * (N + 1):3] = -ca.inf
        ubx[2:3 * (N + 1):3] = ca.inf
        
        return {'lbx': lbx, 'ubx': ubx}

    def get_control_bounds(self, N):
        """Define and return the control bounds."""
        v_min, v_max = -0.2, 0.2
        omega_min, omega_max = -0.2, 0.2

        lbx = np.full((3 * (N + 1) + 2 * N, 1), -ca.inf)
        ubx = np.full((3 * (N + 1) + 2 * N, 1), ca.inf)

        lbx[3 * (N + 1)::2] = v_min
        ubx[3 * (N + 1)::2] = v_max
        lbx[3 * (N + 1) + 1::2] = omega_min
        ubx[3 * (N + 1) + 1::2] = omega_max

        return {'lbx': lbx, 'ubx': ubx}

    def get_general_bounds(self, N):
        """Define and return the general constraints bounds."""
        lbg = np.concatenate((np.zeros((3 * (N + 1), 1)), np.full((N + 1, 1), -np.inf)))
        ubg = np.concatenate((np.zeros((3 * (N + 1), 1)), np.full((N + 1, 1), 0)))
        return {'lbg': lbg, 'ubg': ubg}

    def get_parameters(self):
        """Concatenate the current state and the target state as parameters."""
        p = np.concatenate((self.current_state, self.target_state))
        print(p.shape)
        return {'p': p}

    def get_initial_guess(self, N):
        """Define and return the initial guess for the solver."""
        u0 = np.zeros((N, 2))
        X0 = np.tile(self.current_state, (N + 1, 1))
        x0 = np.concatenate((X0.T.flatten(), u0.T.flatten()))
        return {'x0': x0}

    def publish_cmd_vel(self, u):
        """Publish the command velocity to the /cmd_vel topic."""
        cmd_vel_msg = Twist()
        cmd_vel_msg.linear.x = u[0]
        cmd_vel_msg.angular.z = u[1]
        self.cmd_vel_publisher.publish(cmd_vel_msg)