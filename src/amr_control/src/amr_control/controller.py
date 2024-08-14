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

# from amr_control.robot_model import RobotModel

class nMPC:
    def __init__(self, model, obstacle, N=50, Q=np.diag([1, 5, 0.1]), R=np.diag([0.5, 0.05]), T=0.1):
        print("Controller initialized")
        self.model = model
        self.obstacle = obstacle
        self.N = N
        self.Q = Q
        self.R = R
        self.T = T
        self.define_optimization_problem()

    def define_optimization_problem(self):
        n_states = self.model.n_states
        n_controls = self.model.n_controls
        Q = self.Q
        R = self.R
        N = self.N
        T = self.T
        
        U = ca.SX.sym('U', n_controls, N)
        # P = ca.SX.sym('P', self.model.n_states + self.model.n_states)
        P = ca.SX.sym('P', n_states + N * (n_states + n_controls))
        X = ca.SX.sym('X', n_states, N + 1)
        obs_x = self.obstacle.x
        obs_y = self.obstacle.y
        obs_diam = self.obstacle.diam
        rob_diam = self.model.diam

        # Definiere die Kostenfunktion
        obj = 0
        g = []
        
        st = X[:, 0]
        g = ca.vertcat(g, st - P[:3])
        # for k in range(N):
        #     st = X[:, k]
        #     con = U[:, k]
        #     obj = obj + (st - P[3:]).T @ Q @ (st - P[3:]) + con.T @ R @ con
        #     st_next = X[:, k + 1]
        #     f_value = self.model.f(st, con)
        #     st_next_euler = st + T * f_value
        #     g = ca.vertcat(g, st_next - st_next_euler)
        
        
        # Loop over the prediction horizon
        for k in range(N):
            # Extract current state and control
            st = X[:, k]
            con = U[:, k]
            
            # Calculate the objective function
            state_ref = P[n_states + k * (n_states + n_controls): n_states + (k + 1) * (n_states + n_controls) - 2]
            control_ref = P[n_states + (k + 1) * (n_states + n_controls) - 2: n_states + (k + 1) * (n_states + n_controls)]
            
            state_error = st - state_ref
            control_error = con - control_ref
            
            obj += ca.mtimes([state_error.T, Q, state_error]) + ca.mtimes([control_error.T, R, control_error])
            
            # System dynamics constraints
            st_next = X[:, k + 1]
            f_value = self.model.f(st, con)
            st_next_euler = st + T * f_value
            g = ca.vertcat(g, st_next - st_next_euler)
                
        
            
        # Obstacle avoidance constraints
        for k in range(N + 1):
            obs_constraint = -ca.sqrt((X[0, k] - obs_x) ** 2 + (X[1, k] - obs_y) ** 2) + (rob_diam / 2 + obs_diam / 2)
            g = ca.vertcat(g, obs_constraint)

        # Make the decision variable one column vector
        self.OPT_variables = ca.vertcat(ca.reshape(X, n_states * ( N + 1), 1), ca.reshape(U, n_controls * N, 1))
        
        nlp_prob = {'f': obj, 'x': self.OPT_variables, 'g': g, 'p': P}
        
        opts = {'ipopt.max_iter': 2000,
                'ipopt.print_level': 0,
                'print_time': 0,
                'ipopt.acceptable_tol': 1e-8,
                'ipopt.acceptable_obj_change_tol': 1e-6}

        self.solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)

