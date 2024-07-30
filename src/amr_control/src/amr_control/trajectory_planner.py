#!/usr/bin/env python3

# ROS1 imports
import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Twist, Point, PoseWithCovarianceStamped, PoseArray, Pose
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray  # Assuming you need multiple floats for position
from nav_msgs.msg import Odometry  # Import Odometry message type
from tf.transformations import euler_from_quaternion
import tf.transformations
from visualization_msgs.msg import Marker, MarkerArray

# Casadi imports
import casadi as ca
from casadi.tools import *

# Other imports
import numpy as np
import math

class TrajectoryPlanner:
    def __init__(self, model, controller, initial_state, target_state):
        rospy.init_node('nmpc_node', anonymous=True)
        self.subscription = rospy.Subscriber('/odom', Odometry, self.pose_callback)
        self.cmd_vel_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.pose_array_publisher = rospy.Publisher('/predicted_trajectory', PoseArray, queue_size=10)

        self.model = model
        self.controller = controller
        self.current_state = np.array(initial_state)
        self.target_state = np.array(target_state)
        self.u0 = np.zeros((self.controller.N, 2))
        self.xx1 = []
        self.u_cl = []

        # Initialisieren Sie `t` als Instanzvariable
        self.t = np.array([rospy.get_time()])  # Verwenden Sie ROS-Zeit, wenn in ROS-Umgebung

        self.timer = rospy.Timer(rospy.Duration(self.controller.T), self.controller_loop)

    def pose_callback(self, msg):
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        yaw = self.get_yaw_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])
        self.current_state = np.array([position.x, position.y, yaw])

    def get_yaw_from_quaternion(self, quaternion):
        norm = math.sqrt(sum([x * x for x in quaternion]))
        quaternion = [x / norm for x in quaternion]
        roll, pitch, yaw = euler_from_quaternion(quaternion)
        return yaw

    def controller_loop(self, event):
        if np.linalg.norm(self.current_state - self.target_state, 2) > 1e-2:
            N = self.controller.N

            v_min = -0.2
            v_max = 0.2
            omega_min = -0.2
            omega_max = 0.2
            
            args = {'lbg': np.zeros((3 * (N + 1), 1)),
            'ubg': np.zeros((3 * (N + 1), 1)),
            'lbx': np.full((3 * (N + 1) + 2 * N, 1), -ca.inf),
            'ubx': np.full((3 * (N + 1) + 2 * N, 1), ca.inf)}
                        
            # State bounds
            args['lbx'][0:3 * (N + 1):3] = -2
            args['ubx'][0:3 * (N + 1):3] = 2
            args['lbx'][1:3 * (N + 1):3] = -2
            args['ubx'][1:3 * (N + 1):3] = 2
            args['lbx'][2:3 * (N + 1):3] = -ca.inf
            args['ubx'][2:3 * (N + 1):3] = ca.inf
            
            # Control bounds
            args['lbx'][3 * (N + 1)::2] = v_min
            args['ubx'][3 * (N + 1)::2] = v_max
            args['lbx'][3 * (N + 1) + 1::2] = omega_min
            args['ubx'][3 * (N + 1) + 1::2] = omega_max
            # print(args['p'])

            args['p'] = np.concatenate((self.current_state, self.target_state))
            
            u0 = np.zeros((N, 2))
            X0 = np.tile(self.current_state, (N + 1, 1))

            args['x0'] = np.concatenate((X0.T.flatten(), u0.T.flatten()))
                        
            sol = self.controller.solver(x0=args['x0'], lbx=args['lbx'], ubx=args['ubx'],
                                         lbg=args['lbg'], ubg=args['ubg'], p=args['p'])
            
            xx = np.zeros((3, 1))
            xx[:, 0] = self.current_state
            
            u = np.reshape(sol['x'][3 * (N + 1):].full(), (N, 2))
            
            predicted_states = []
            predicted_states.append(np.reshape(sol['x'][:3 * (N + 1)].full(), (N + 1, 3)))
            
            predicted_states = np.array(predicted_states).squeeze()
            print(predicted_states)

            # Konvertieren Sie predicted_states in ein PoseArray und veröffentlichen Sie es
            pose_array = PoseArray()
            pose_array.header.stamp = rospy.Time.now()
            pose_array.header.frame_id = "map"  # oder einen anderen geeigneten frame_id

            for state in predicted_states:  # Iteriere durch Spalten, wenn states Zeilen von Zuständen sind
                pose = Pose()
                pose.position.x = state[0]
                pose.position.y = state[1]
                print(state[0], state[1], state[2])
                quaternion = tf.transformations.quaternion_from_euler(0, 0, state[2])
                pose.orientation.x = quaternion[0]
                pose.orientation.y = quaternion[1]
                pose.orientation.z = quaternion[2]
                pose.orientation.w = quaternion[3]
                pose_array.poses.append(pose)

            self.pose_array_publisher.publish(pose_array)   
        
                                        
            # Konvertiere den ersten Steuerbefehl in eine ROS-Nachricht
            cmd_vel_msg = Twist()
            cmd_vel_msg.linear.x = u[0, 0]  # v
            cmd_vel_msg.angular.z = u[0, 1]  # omega

            # Veröffentliche den Steuerbefehl
            self.cmd_vel_publisher.publish(cmd_vel_msg)
            
            print(u[0,0], u[0,1])
            
            self.current_state, self.u0 = self.shift(u)  # Aktualisieren des Zustands und der Steuerung
            
        else:
            cmd_vel_msg = Twist()
            cmd_vel_msg.linear.x = 0  # v
            cmd_vel_msg.angular.z = 0 # omega
            self.cmd_vel_publisher.publish(cmd_vel_msg)


    def shift(self, u):
        st = self.current_state
        con = u[0, :]
        f_value = self.model.f(st, con)
        st_next = st + (self.controller.T * f_value.full().flatten())
        u0 = np.vstack((u[1:, :], u[-1, :]))
        return st_next, u0
