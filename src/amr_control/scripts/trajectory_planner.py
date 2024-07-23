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
            t0 = rospy.get_time()
            self.xx = np.zeros((3, 1))
            self.xx[:, 0] = self.current_state
            
            # Konfigurieren der Argumente für den Solver
            args = {
                'lbg': -2, 'ubg': 2,
                'lbx': ca.DM.zeros((2 * self.controller.N, 1)),
                'ubx': ca.DM.zeros((2 * self.controller.N, 1)),
                'p': np.concatenate((self.current_state, self.target_state))
            }
            # print(args['p'])

            
            args['x0'] = self.u0.reshape(2 * self.controller.N, 1)
                        
            sol = self.controller.solver(x0=args['x0'], lbx=args['lbx'], ubx=args['ubx'],
                                         lbg=args['lbg'], ubg=args['ubg'], p=args['p'])
            
            u = sol['x'].full().reshape(self.controller.N, 2)
                        
            print(u[0,0])
            
            # Konvertiere den ersten Steuerbefehl in eine ROS-Nachricht
            cmd_vel_msg = Twist()
            cmd_vel_msg.linear.x = u[0, 0]  # v
            cmd_vel_msg.angular.z = u[0, 1]  # omega

            # Veröffentliche den Steuerbefehl
            self.cmd_vel_publisher.publish(cmd_vel_msg)
            
            ff_value = self.controller.ff(u.T, args['p'])
            self.xx1.append(ff_value.full())
            self.u_cl.append(u[0, :])
            
            # Aktualisieren von `t` und Zuständen
            self.t = np.vstack((self.t, t0))  # Aktualisieren der Zeitstempel
            self.current_state, self.u0 = self.shift(u)  # Aktualisieren des Zustands und der Steuerung

    def shift(self, u):
        st = self.current_state
        con = u[0, :]
        f_value = self.model.f(st, con)
        st_next = st + (self.controller.T * f_value.full().flatten())
        u0 = np.vstack((u[1:, :], u[-1, :]))
        return st_next, u0
