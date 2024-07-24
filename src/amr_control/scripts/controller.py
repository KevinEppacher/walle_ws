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


from amr_control.scripts.robot_model import RobotModel

class nMPC:
    def __init__(self, model, N=50, Q=np.diag([1, 5, 0.1]), R=np.diag([0.5, 0.05]), T=0.1):
        print("Controller initialized")
        self.model = model
        self.N = N
        self.Q = Q
        self.R = R
        self.T = T
        self.define_optimization_problem()

    def define_optimization_problem(self):
        U = ca.SX.sym('U', self.model.n_controls, self.N)
        P = ca.SX.sym('P', self.model.n_states + self.model.n_states)
        X = ca.SX.sym('X', self.model.n_states, self.N + 1)
        X[:, 0] = P[0:3]

        # Berechne den nächsten Zustand für jeden Schritt im Prädiktionshorizont
        for k in range(self.N):
            st = X[:, k]
            con = U[:, k]
            f_value = self.model.f(st, con)
            st_next = st + (self.T * f_value)
            X[:, k + 1] = st_next

        # Definiere die Funktion ff, um die vollständige Trajektorie berechnen zu können
        self.ff = ca.Function('ff', [U, P], [X])

        # Definiere die Kostenfunktion
        obj = 0
        g = []
        for k in range(self.N):
            st = X[:, k]
            con = U[:, k]
            obj += ca.mtimes([(st - P[3:6]).T, self.Q, (st - P[3:6])]) + ca.mtimes([con.T, self.R, con])
        g = X[0:2, :].T.reshape((-1, 1))

        # Definiere das NLP-Problem
        self.OPT_variables = ca.reshape(U, 2 * self.N, 1)
        nlp_prob = {'f': obj, 'x': self.OPT_variables, 'g': g, 'p': P}
        opts = {'ipopt.max_iter': 2000, 'ipopt.print_level': 0, 'print_time': 0, 'ipopt.acceptable_tol': 1e-8, 'ipopt.acceptable_obj_change_tol': 1e-6}
        self.solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)

